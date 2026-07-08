"""Bulk-download recipe HTML with Scrapy, tuned for maximal throughput.

This is a self-contained, drop-in alternative to ``download_htmls.py``. It reads
the same source CSV and writes each page to
``data/raw/html_cache/<hash>.html``, where ``hash`` is the precomputed link
hash stored by ``prepare_dataset.py`` — exactly the cache layout that
``build_dataset.py`` reads back — so the two downloaders are interchangeable.

Run it directly:

    python scripts/scrapy_download.py

Tune concurrency without editing the file via env vars, e.g.:

    CONCURRENT_REQUESTS=400 CONCURRENT_PER_DOMAIN=24 python scripts/scrapy_download.py

Why Scrapy over the asyncio version: a single Twisted reactor drives all I/O,
DNS resolution runs on a dedicated thread pool, connection reuse / retry /
redirect handling are built in, and start_requests is consumed lazily so the
2M+ URL list never has to be materialised as pending tasks at once.

Every pending link is attempted. Failure handling is two-phase: Scrapy's
immediate RetryMiddleware is disabled; instead, first-round failures are
collected and re-tried once at the *end* of the crawl (via the spider_idle
signal), after every fresh URL has been attempted. A URL that fails both
rounds is deemed unreachable, written to FAILED_LOG, and its slot is refilled
with a fresh URL from the backlog beyond the FILE_LIMIT budget — so the crawl
keeps going until FILE_LIMIT files are on disk or the links run out.
"""

import asyncio
import logging
import os
import time

import polars as pl
import scrapy
from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import DontCloseSpider

from data_generation_config import (
    HTML_CACHE_DIR,
    CLEANED_DATASET_PATH,
    FAILED_LOG,
    FILE_LIMIT,
    USER_AGENT,
    CONCURRENT_PER_DOMAIN,
    CONCURRENT_REQUESTS,
    PROGRESS_EVERY,
    SEED_YIELD_EVERY,
    RETRY_BATCH_SIZE,
)

def load_failed_urls() -> set[str]:
    if not os.path.exists(FAILED_LOG):
        return set()
    with open(FAILED_LOG, "r") as f:
        return {line.strip() for line in f if line.strip()}


def load_links() -> list[tuple[str, str]]:
    """Read and shuffle every (link, hash) pair from the cleaned dataset.

    Links are already ``www.``-normalised by prepare_dataset.py, and ``hash``
    is the precomputed cache-filename stem from the same script — polars
    hashes are not stable across versions, so it must be read back, never
    recomputed here. The dataset is grouped by website, so in stored order
    almost all in-flight requests hit a single host, where
    CONCURRENT_REQUESTS_PER_DOMAIN throttles throughput to a trickle while the
    global budget sits idle. Shuffling spreads the concurrent slots across
    many domains at once — that's what actually unlocks the configured
    throughput (and is gentler per host).

    Reading + shuffling 2M+ rows takes a few seconds; we do it here, up front,
    rather than lazily inside the spider so it can't block the reactor thread mid
    crawl (which would stall all downloads).
    """
    return (
        pl.read_parquet(CLEANED_DATASET_PATH, columns=["link", "hash"])
        .sample(fraction=1.0, shuffle=True, seed=0)
        .rows()
    )


def pending_downloads(links: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Filter ``links`` down to (url, cache_path) pairs not yet on disk.

    Crucially this snapshots the cache directory **once** with a single
    ``os.listdir`` and tests set membership, instead of an ``os.path.exists``
    per link. At 2M+ links the per-link stat() approach costs ~135s of syscalls;
    doing it lazily inside the spider would run that on the reactor thread and
    freeze all downloads. The set-membership version is ~30x faster and runs
    here, before the reactor starts.
    """
    cached = set(os.listdir(HTML_CACHE_DIR)) if os.path.isdir(HTML_CACHE_DIR) else set()
    pending = []
    for link, link_hash in links:
        filename = link_hash + ".html"
        if filename not in cached:
            pending.append((f"http://{link}", os.path.join(HTML_CACHE_DIR, filename)))
    return pending


class RecipeSpider(scrapy.Spider):
    name = "recipes"

    def __init__(self, pending=None, backlog=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-built (url, cache_path) pairs still to fetch — already shuffled and
        # cache-filtered in main() before the reactor starts, so seeding does no
        # filesystem work and can't stall the reactor.
        self.pending = pending or []
        # Uncached pairs beyond the FILE_LIMIT budget. Each permanently failed
        # URL draws one replacement from here, so the number of *successful*
        # downloads converges on the budget instead of budget-minus-failures.
        self.backlog = iter(backlog or [])
        # Failed-download bookkeeping, still written to FAILED_LOG on close:
        # remember which URLs were known-failed coming in, recover any that now
        # succeed, and dedupe new failures. Scrapy runs callbacks serially on the
        # reactor thread, so these sets need no locking.
        self.previously_failed = load_failed_urls()
        self.recovered: set[str] = set()
        self.newly_failed: set[str] = set()
        self.saved = 0
        # First-round failures, retried once at the end of the crawl. A plain
        # list of (url, cache_path); callbacks run serially on the reactor
        # thread, so no locking needed.
        self.retry_pending: list[tuple[str, str]] = []
        self.retry_phase_started = False

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        # spider_idle fires when the scheduler is empty and nothing is in
        # flight — i.e. every fresh URL has been attempted. That's when we
        # inject the retry round.
        crawler.signals.connect(spider.on_idle, signal=signals.spider_idle)
        return spider

    # No link following: every request is a terminal page we save to disk.
    # Seeding does zero filesystem work — the cache filtering already happened in
    # pending_downloads() — so it never blocks the reactor.
    def _seed_requests(self):
        for url, path in self.pending:
            yield scrapy.Request(
                url,
                callback=self.save_page,
                errback=self.on_error,
                cb_kwargs={"path": path, "url": url},
                # We dedupe via the on-disk cache, so don't pay for the dupe filter.
                dont_filter=True,
            )

    # Scrapy >= 2.13 seeds the crawl from the async ``start()`` entry point;
    # the base class no longer consults ``start_requests`` (an overridden one is
    # simply ignored, which silently yields zero requests). ``start_requests``
    # is kept as a sync alias purely for Scrapy < 2.13 compatibility.
    #
    # With ~2M start requests, materialising them all before downloading would
    # block the reactor for ~a minute (building Request objects is ~30us each).
    # Awaiting sleep(0) every SEED_YIELD_EVERY requests hands control back to the
    # event loop so downloads run *while* we keep seeding — crawling starts in
    # the first second instead of after the whole list is built.
    async def start(self):
        for i, request in enumerate(self._seed_requests()):
            if i % SEED_YIELD_EVERY == 0:
                await asyncio.sleep(0)
            yield request

    def start_requests(self):
        return self._seed_requests()

    def save_page(self, response, path, url):
        # response.text handles charset detection; write atomically-ish per file.
        with open(path, "w", encoding="utf-8") as f:
            f.write(response.text)
        if url in self.previously_failed:
            self.recovered.add(url)  # drop it from the failed log on close

        # Heartbeat: per-page logging would flood at 200 concurrency, so emit a
        # progress line every PROGRESS_EVERY saves instead.
        self.saved += 1
        if self.saved % PROGRESS_EVERY == 0:
            self.logger.info("saved %d pages (latest: %s)", self.saved, url)

    def on_error(self, failure):
        # DNS errors, timeouts and HTTP error statuses (via HttpError) all land
        # here. First failure: queue the URL for one deferred retry at the end
        # of the crawl. Second failure: the URL is unreachable — record it for
        # FAILED_LOG (written on close) and draw a fresh replacement URL from
        # the backlog, so every permanently failed slot is refilled and the
        # success count still reaches the FILE_LIMIT budget. Replacements are
        # ordinary first-round requests: if one fails it re-enters the same
        # retry-then-replace cycle. Errback output is scheduled by Scrapy just
        # like callback output, so we can simply yield the new request here.
        # Nothing is logged to the console; failures are common here and would
        # drown out the progress output.
        request = failure.request
        if request.meta.get("is_retry"):
            self.newly_failed.add(request.cb_kwargs["url"])
            replacement = next(self.backlog, None)
            if replacement is not None:
                url, path = replacement
                yield scrapy.Request(
                    url,
                    callback=self.save_page,
                    errback=self.on_error,
                    cb_kwargs={"path": path, "url": url},
                    dont_filter=True,
                )
        else:
            self.retry_pending.append(
                (request.cb_kwargs["url"], request.cb_kwargs["path"])
            )

    def on_idle(self):
        # All fresh URLs are done; feed the retry queue in batches. Each batch
        # drains fully before the next idle event fires, so memory stays flat
        # even with hundreds of thousands of retries. Returning without raising
        # DontCloseSpider (queue empty) lets the spider close normally.
        if not self.retry_pending:
            return
        if not self.retry_phase_started:
            self.retry_phase_started = True
            self.logger.info(
                "retry phase: re-trying %d failed URLs", len(self.retry_pending)
            )
        batch = self.retry_pending[:RETRY_BATCH_SIZE]
        del self.retry_pending[:RETRY_BATCH_SIZE]
        for url, path in batch:
            self.crawler.engine.crawl(
                scrapy.Request(
                    url,
                    callback=self.save_page,
                    errback=self.on_error,
                    cb_kwargs={"path": path, "url": url},
                    meta={"is_retry": True},
                    dont_filter=True,
                )
            )
        raise DontCloseSpider

    def closed(self, reason):
        # Rewrite the failed log once: keep prior failures that didn't recover,
        # plus any new ones — deduplicated, matching the previous downloader.
        remaining = (self.previously_failed - self.recovered) | self.newly_failed
        if remaining:
            with open(FAILED_LOG, "w") as f:
                f.write("\n".join(sorted(remaining)) + "\n")
        elif os.path.exists(FAILED_LOG):
            open(FAILED_LOG, "w").close()
        # Console summary: only the success count (failures are in FAILED_LOG).
        self.logger.info("Done (%s): %d pages successfully downloaded.", reason, self.saved)


def scrape_htmls():
    os.makedirs(HTML_CACHE_DIR, exist_ok=True)

    # Do the heavy CSV read + shuffle here, before the reactor starts, so it
    # never blocks downloads. Printed so it's obvious the script is working
    # during the few-second load rather than looking hung.
    t0 = time.time()
    print(f"Reading and shuffling links from {CLEANED_DATASET_PATH} ...", flush=True)
    links = load_links()
    already_cached = len(os.listdir(HTML_CACHE_DIR))
    uncached = pending_downloads(links)

    # FILE_LIMIT targets the *total* cache size, counting files already on
    # disk. The first `budget` uncached links are seeded immediately; the rest
    # become the backlog that refills permanently failed slots one-for-one, so
    # the run ends with FILE_LIMIT files on disk (unless the backlog runs dry).
    if FILE_LIMIT is None:
        pending, backlog = uncached, []
    else:
        budget = max(FILE_LIMIT - already_cached, 0)
        pending, backlog = uncached[:budget], uncached[budget:]

    print(
        f"{len(links):,} links, {already_cached:,} already cached, "
        f"limit {'none' if FILE_LIMIT is None else format(FILE_LIMIT, ',')}, "
        f"{len(pending):,} to fetch (+{len(backlog):,} backlog). "
        f"Prepared in {time.time() - t0:.1f}s. Starting crawl.",
        flush=True,
    )

    process = CrawlerProcess(
        settings={
            "USER_AGENT": USER_AGENT,
            # --- Concurrency / throughput ---
            "CONCURRENT_REQUESTS": CONCURRENT_REQUESTS,
            "CONCURRENT_REQUESTS_PER_DOMAIN": CONCURRENT_PER_DOMAIN,
            "CONCURRENT_REQUESTS_PER_IP": 0,
            "DOWNLOAD_DELAY": 0,
            # Blocking DNS runs on the reactor thread pool; enlarge it so DNS
            # doesn't bottleneck connection setup at high concurrency.
            "REACTOR_THREADPOOL_MAXSIZE": 32,
            "DNSCACHE_ENABLED": True,
            "DNSCACHE_SIZE": 100_000,
            # Dead domains are a big chunk of this aged dataset; without this each
            # unresolvable host ties up a DNS thread for the full 60s default.
            "DNS_TIMEOUT": 10,
            # --- AutoThrottle: self-tunes per-server concurrency from observed
            # latency, so we don't hand-pick a number and we stop provoking the
            # 429/overload failures. CONCURRENT_REQUESTS* act as ceilings it works
            # under; TARGET_CONCURRENCY is how many parallel requests it aims for
            # per responsive host (raise for more speed, lower to be gentler).
            "AUTOTHROTTLE_ENABLED": True,
            "AUTOTHROTTLE_START_DELAY": 1.0,
            "AUTOTHROTTLE_MAX_DELAY": 30.0,
            "AUTOTHROTTLE_TARGET_CONCURRENCY": 8.0,
            # --- Per-request behaviour ---
            "DOWNLOAD_TIMEOUT": 30,
            # Immediate back-to-back retries are disabled; the spider defers
            # first-round failures to a single retry pass at the end of the
            # crawl (see on_error/on_idle), which gives flaky hosts time to
            # recover and keeps dead hosts from eating 3x their timeout.
            "RETRY_ENABLED": False,
            "REDIRECT_ENABLED": True,
            "COOKIES_ENABLED": False,  # cookie tracking is pure overhead for a one-shot fetch
            "ROBOTSTXT_OBEY": False,   # matches download_htmls.py behaviour
            "AJAXCRAWL_ENABLED": False,
            # Cap pathological pages so one giant response can't stall a slot / blow memory.
            "DOWNLOAD_MAXSIZE": 16 * 1024 * 1024,
            "DOWNLOAD_WARNSIZE": 8 * 1024 * 1024,
            # --- Reactor / logging ---
            "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
            "TELNETCONSOLE_ENABLED": False,
            "LOG_LEVEL": "INFO",
            # Default heartbeat is every 60s — too sparse to tell it's alive.
            "LOGSTATS_INTERVAL": 5,
        }
    )
    # Silence Scrapy's own per-failure console output (404/403 "Ignoring
    # response"). Failures still go to FAILED_LOG; the console keeps only
    # progress and the final success count.
    logging.getLogger("scrapy.spidermiddlewares.httperror").setLevel(logging.WARNING)

    process.crawl(RecipeSpider, pending=pending, backlog=backlog)
    process.start()


if __name__ == "__main__":
    scrape_htmls()
