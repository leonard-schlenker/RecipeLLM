"""Bulk-download recipe HTML with Scrapy, tuned for maximal throughput.

This is a self-contained, drop-in alternative to ``download_htmls.py``. It reads
the same source CSV, applies the same ``www.``-prefix normalisation, and writes
each page to ``data/raw/html_cache/<md5(url)>.html`` — exactly the cache layout
that ``build_dataset.py`` reads back — so the two downloaders are interchangeable.

Run it directly:

    python scripts/scrapy_download.py

Tune concurrency without editing the file via env vars, e.g.:

    CONCURRENT_REQUESTS=400 CONCURRENT_PER_DOMAIN=24 python scripts/scrapy_download.py

Why Scrapy over the asyncio version: a single Twisted reactor drives all I/O,
DNS resolution runs on a dedicated thread pool, connection reuse / retry /
redirect handling are built in, and start_requests is consumed lazily so the
2M+ URL list never has to be materialised as pending tasks at once.
"""

import asyncio
import hashlib
import logging
import os
import time

import polars as pl
import scrapy
from scrapy.crawler import CrawlerProcess

# All data lives on the 128GB USB stick: the source CSV, the failed-downloads
# log, and the html_cache (214K+ files already copied there). Override RAW_DIR
# via the environment to point at a different mount (or "./data/raw" for the
# original local layout shared with download_htmls.py / build_dataset.py).
RAW_DIR = os.environ.get("RAW_DIR", "/Volumes/USB/raw")
SOURCE_CSV = os.path.join(RAW_DIR, "RecipeNLG_dataset.csv")
HTML_CACHE_DIR = os.path.join(RAW_DIR, "html_cache")
FAILED_LOG = os.path.join(RAW_DIR, "failed_downloads.log")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Throughput knobs — overridable from the environment for quick tuning.
CONCURRENT_REQUESTS = int(os.environ.get("CONCURRENT_REQUESTS", 200))
# Across many distinct hosts the global limit dominates; the per-domain cap is
# what keeps any single site from banning us. Raise it for more aggression.
CONCURRENT_PER_DOMAIN = int(os.environ.get("CONCURRENT_PER_DOMAIN", 16))

# Log a progress line every this many successfully-saved pages.
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", 100))

# --- TEMPORARY DISK CAP -----------------------------------------------------
# Hard limit on the TOTAL number of HTML files on disk (already-cached files +
# this run's downloads). At ~10K HTML files per GB, 1M files is ~100GB, which
# fits the current 128GB USB stick with headroom; the full ~2M-link dataset
# would overflow it. Remove this constant and the truncation block in main()
# once the dataset lives on bigger storage.
MAX_TOTAL_FILES = 1_000_000
# ---------------------------------------------------------------------------

# During seeding, hand control back to the event loop every this many requests
# so downloads run while the large start list is still being built.
SEED_YIELD_EVERY = 200


def load_failed_urls() -> set[str]:
    if not os.path.exists(FAILED_LOG):
        return set()
    with open(FAILED_LOG, "r") as f:
        return {line.strip() for line in f if line.strip()}


def load_links() -> list[str]:
    """Read, normalise and shuffle every recipe link from the source CSV.

    Normalisation matches download_htmls.py: ensure a ``www.`` prefix. The CSV is
    grouped by website, so in CSV order almost all in-flight requests hit a single
    host, where CONCURRENT_REQUESTS_PER_DOMAIN throttles throughput to a trickle
    while the global budget sits idle. Shuffling spreads the concurrent slots
    across many domains at once — that's what actually unlocks the configured
    throughput (and is gentler per host).

    Reading + shuffling 2M+ rows takes a few seconds; we do it here, up front,
    rather than lazily inside the spider so it can't block the reactor thread mid
    crawl (which would stall all downloads).
    """
    return (
        pl.read_csv(SOURCE_CSV, columns=["link"])
        .with_columns(
            pl.when(pl.col("link").str.starts_with("www."))
            .then(pl.col("link"))
            .otherwise(pl.lit("www.") + pl.col("link"))
        )["link"]
        .sample(fraction=1.0, shuffle=True, seed=0)
        .to_list()
    )


def pending_downloads(links: list[str]) -> list[tuple[str, str]]:
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
    for link in links:
        url = f"http://{link}"
        filename = hashlib.md5(url.encode()).hexdigest() + ".html"
        if filename not in cached:
            pending.append((url, os.path.join(HTML_CACHE_DIR, filename)))
    return pending


class RecipeSpider(scrapy.Spider):
    name = "recipes"

    def __init__(self, pending=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-built (url, cache_path) pairs still to fetch — already shuffled and
        # cache-filtered in main() before the reactor starts, so seeding does no
        # filesystem work and can't stall the reactor.
        self.pending = pending or []
        # Failed-download bookkeeping, still written to FAILED_LOG on close:
        # remember which URLs were known-failed coming in, recover any that now
        # succeed, and dedupe new failures. Scrapy runs callbacks serially on the
        # reactor thread, so these sets need no locking.
        self.previously_failed = load_failed_urls()
        self.recovered: set[str] = set()
        self.newly_failed: set[str] = set()
        self.saved = 0

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
        # Record the failure to FAILED_LOG (written on close) but don't log it to
        # the console — failures are the common case here and would drown out the
        # progress output.
        self.newly_failed.add(failure.request.url)

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


def main():
    os.makedirs(HTML_CACHE_DIR, exist_ok=True)

    # Do the heavy CSV read + shuffle here, before the reactor starts, so it
    # never blocks downloads. Printed so it's obvious the script is working
    # during the few-second load rather than looking hung.
    t0 = time.time()
    print(f"Reading and shuffling links from {SOURCE_CSV} ...", flush=True)
    links = load_links()
    pending = pending_downloads(links)
    already_cached = len(links) - len(pending)

    # --- TEMPORARY DISK CAP: keep total files on disk <= MAX_TOTAL_FILES so the
    # partial dataset fits the 128GB USB stick. Truncating the (already shuffled)
    # pending list guarantees we never write more than the cap. Delete this block
    # to fetch the full dataset once it lives on bigger storage.
    budget = max(0, MAX_TOTAL_FILES - already_cached)
    capped = len(pending) > budget
    if capped:
        pending = pending[:budget]
    # ---------------------------------------------------------------------------

    cap_note = (
        f" (capped at {MAX_TOTAL_FILES:,} total files for the 128GB stick)"
        if capped
        else ""
    )
    print(
        f"{len(links):,} links, {already_cached:,} already cached, "
        f"{len(pending):,} to fetch{cap_note}. "
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
            "RETRY_ENABLED": True,
            "RETRY_TIMES": 2,
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
    # Silence Scrapy's own per-failure console output (404/403 "Ignoring response"
    # and "Gave up retrying" for DNS/timeout). Failures still go to FAILED_LOG;
    # the console keeps only progress and the final success count.
    logging.getLogger("scrapy.spidermiddlewares.httperror").setLevel(logging.WARNING)
    logging.getLogger("scrapy.downloadermiddlewares.retry").setLevel(logging.CRITICAL)

    process.crawl(RecipeSpider, pending=pending)
    process.start()


if __name__ == "__main__":
    main()
