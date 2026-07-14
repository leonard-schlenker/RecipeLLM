"""Bulk-download recipe HTML with Scrapy, tuned for maximal throughput.

This is a self-contained, drop-in alternative to ``download_htmls.py``. It
reads every unique link from the *full* source CSV (ORIG_DATASET_PATH) — not
the cleaned parquet, which drops links to equalise the per-website
distribution — and writes each page to ``HTML_CACHE_DIR/<hash>.html``. The
cache should hold every page we can get; re-sampling the dataset later can
then draw on all of them without re-scraping. ``hash`` is recomputed with the
same polars expression as ``prepare_dataset.py`` and validated against the
``hash`` column stored in the cleaned parquet (polars hashes are not stable
across versions); on mismatch the script aborts rather than write cache
filenames the rest of the pipeline can't find.

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
rounds is deemed unreachable, appended to FAILED_LOG immediately (so
interrupted runs keep their failures), and its slot is refilled with a fresh
URL from the backlog beyond the FILE_LIMIT budget — so the crawl keeps going
until FILE_LIMIT files are on disk or the links run out.

URLs recorded in FAILED_LOG are treated as permanently dead: subsequent runs
skip them up front instead of re-attempting them. Delete the file to give
them another chance (e.g. after a transient network problem on our side).
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
    ORIG_DATASET_PATH,
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


def validate_recomputed_hashes(recomputed: pl.DataFrame) -> None:
    """Abort unless the recomputed hashes reproduce the ``hash`` column stored
    in the cleaned parquet for every link in it.

    The stored column is the source of truth for cache filenames (polars
    hashes are not stable across polars versions). Recomputing from the full
    CSV is only safe while this polars version reproduces them bit-for-bit;
    if it doesn't, scraping would write filenames the rest of the pipeline
    can't find, so we refuse to run instead. (Same guard as
    migrate_html_cache.py.)
    """
    stored = pl.read_parquet(CLEANED_DATASET_PATH, columns=["link", "hash"]).unique(
        subset="link"
    )
    joined = stored.join(recomputed, on="link", how="left", suffix="_recomputed")
    bad = joined.filter(
        pl.col("hash_recomputed").is_null()
        | (pl.col("hash") != pl.col("hash_recomputed"))
    )
    if len(bad) > 0:
        row = bad.row(0, named=True)
        raise SystemExit(
            f"ABORT: recomputed hashes don't reproduce the stored `hash` column in "
            f"{CLEANED_DATASET_PATH} for {len(bad)} of {len(joined)} links "
            f"(e.g. {row['link']!r}: stored {row['hash']} != recomputed "
            f"{row['hash_recomputed']}). Scraping would write cache filenames the "
            "pipeline can't find. Did the polars version change, or are the "
            "datasets out of sync?"
        )


def load_links() -> list[tuple[str, str]]:
    """Read and shuffle every unique (link, hash) pair from the full source CSV.

    The scrape deliberately targets ORIG_DATASET_PATH, not the cleaned
    parquet: the cleaned one throws out links to equalise the per-website
    distribution, but the cache should hold every page we can get, so a later
    re-sampling can draw on all of them without re-scraping. Links are
    ``www.``-normalised and hashed exactly as prepare_dataset.py does, and the
    recomputed hashes are validated against the parquet's stored column before
    anything is fetched (see validate_recomputed_hashes).

    The dataset is grouped by website, so in stored order almost all in-flight
    requests hit a single host, where CONCURRENT_REQUESTS_PER_DOMAIN throttles
    throughput to a trickle while the global budget sits idle. Shuffling
    spreads the concurrent slots across many domains at once — that's what
    actually unlocks the configured throughput (and is gentler per host).

    Reading + hashing + shuffling 2M+ rows takes a little while; we do it here,
    up front, rather than lazily inside the spider so it can't block the
    reactor thread mid crawl (which would stall all downloads).
    """
    links = (
        pl.scan_csv(ORIG_DATASET_PATH)
        .select(
            pl.when(pl.col("link").str.starts_with("www."))
            .then(pl.col("link"))
            .otherwise(pl.concat_str(pl.lit("www."), pl.col("link")))
            .alias("link")
        )
        .unique(subset="link")
        .with_columns(pl.col("link").hash(seed=0).cast(pl.String).alias("hash"))
        .collect()
    )
    validate_recomputed_hashes(links)
    return links.sample(fraction=1.0, shuffle=True, seed=0).rows()


def pending_downloads(
    links: list[tuple[str, str]], failed: set[str]
) -> tuple[list[tuple[str, str]], int]:
    """Filter ``links`` down to (url, cache_path) pairs still worth fetching.

    Drops links whose cache file is already on disk, plus links in ``failed``
    — those already failed both attempts in a previous run and are considered
    permanently dead. Returns the pending pairs and the known-failed count
    (reported at startup so the cached/pending/failed numbers add up).

    Crucially this snapshots the cache directory **once** with a single
    ``os.listdir`` and tests set membership, instead of an ``os.path.exists``
    per link. At 2M+ links the per-link stat() approach costs ~135s of syscalls;
    doing it lazily inside the spider would run that on the reactor thread and
    freeze all downloads. The set-membership version is ~30x faster and runs
    here, before the reactor starts.
    """
    cached = set(os.listdir(HTML_CACHE_DIR)) if os.path.isdir(HTML_CACHE_DIR) else set()
    pending = []
    skipped_failed = 0
    for link, link_hash in links:
        filename = link_hash + ".html"
        if filename in cached:
            continue
        # FAILED_LOG stores the seeded form of the URL (scheme-prefixed), so
        # the membership test must use the same form — the bare ``link`` from
        # the parquet would never match.
        url = f"http://{link}"
        if url in failed:
            skipped_failed += 1
            continue
        pending.append((url, os.path.join(HTML_CACHE_DIR, filename)))
    return pending, skipped_failed


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
        # Requests still to be made ("left to visit" in the heartbeat).
        # A save finishes its URL for good (-1); a first-round failure
        # completes one request but queues a retry (net 0); a retry failure
        # either draws a replacement from the backlog (net 0) or dead-ends
        # once the backlog is dry (-1).
        self.remaining = len(self.pending)
        # URLs that fail both rounds are appended here the moment they become
        # permanent, so an interrupted run keeps them (waiting for a clean
        # close lost them: permanent failures only materialise in the
        # end-of-crawl retry phase, which interrupted runs never reach). No
        # dedupe needed: each URL is retried exactly once, and previously
        # logged URLs were filtered out of pending/backlog before the crawl.
        self.failed_log = open(FAILED_LOG, "a")
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

        # Heartbeat: per-page logging would flood at 200 concurrency, so emit a
        # progress line every PROGRESS_EVERY saves instead.
        self.saved += 1
        self.remaining -= 1
        if self.saved % PROGRESS_EVERY == 0:
            self.logger.info(
                "saved %d pages, %d left to visit (latest: %s)",
                self.saved,
                self.remaining,
                url,
            )

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
            self.failed_log.write(request.cb_kwargs["url"] + "\n")
            self.failed_log.flush()  # survive a hard kill mid retry-phase
            replacement = next(self.backlog, None)
            if replacement is None:
                # Backlog exhausted: the failed slot dead-ends instead of
                # respawning a replacement request.
                self.remaining -= 1
            else:
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
        # Failures were already appended to FAILED_LOG as they happened;
        # future runs skip everything in the log up front (delete the file to
        # re-attempt them).
        self.failed_log.close()
        # Console summary: only the success count (failures are in FAILED_LOG).
        self.logger.info("Done (%s): %d pages successfully downloaded.", reason, self.saved)


def scrape_htmls():
    os.makedirs(HTML_CACHE_DIR, exist_ok=True)

    # Do the heavy CSV read + shuffle here, before the reactor starts, so it
    # never blocks downloads. Printed so it's obvious the script is working
    # during the few-second load rather than looking hung.
    t0 = time.time()
    print(f"Reading and shuffling links from {ORIG_DATASET_PATH} ...", flush=True)
    links = load_links()
    already_cached = len(os.listdir(HTML_CACHE_DIR))
    uncached, skipped_failed = pending_downloads(links, load_failed_urls())

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
        f"{skipped_failed:,} known-failed skipped, "
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
