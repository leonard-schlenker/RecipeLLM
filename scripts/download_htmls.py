import asyncio
import hashlib
import os

import aiohttp
import polars as pl

HTML_CACHE_DIR = "./data/raw/html_cache"
FAILED_LOG = "./data/raw/failed_downloads.log"

# First pass: modest concurrency, give servers time to respond.
CONCURRENCY = 25
FAST_TIMEOUT = aiohttp.ClientTimeout(total=25, sock_connect=10, sock_read=20)

# Second pass: only the URLs that timed out / dropped, retried slowly.
SLOW_CONCURRENCY = 5
SLOW_TIMEOUT = aiohttp.ClientTimeout(total=60, sock_connect=20, sock_read=50)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def cache_path_for(url: str) -> str:
    filename = hashlib.md5(url.encode()).hexdigest() + ".html"
    return os.path.join(HTML_CACHE_DIR, filename)


def load_failed_urls() -> set[str]:
    if not os.path.exists(FAILED_LOG):
        return set()
    with open(FAILED_LOG, "r") as f:
        return {line.strip() for line in f if line.strip()}


def log_failed(url: str) -> None:
    with open(FAILED_LOG, "a") as f:
        f.write(url + "\n")


def remove_from_failed_log(url: str) -> None:
    if not os.path.exists(FAILED_LOG):
        return
    with open(FAILED_LOG, "r") as f:
        lines = [l.strip() for l in f if l.strip() != url]
    with open(FAILED_LOG, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


async def download(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    url: str,
    index: int,
    total: int,
    previously_failed: set[str],
    counters: dict,
    timeout: aiohttp.ClientTimeout,
    final_pass: bool,
) -> str | None:
    """Download one URL.

    Returns the URL if it failed with a transient error and should be retried
    in the slower pass; otherwise returns None. On a permanent failure (or when
    final_pass is set) the URL is logged to FAILED_LOG, matching the original
    one-URL-per-line format.
    """
    path = cache_path_for(url)

    if os.path.exists(path):
        counters["skipped"] += 1
        print(f"[{index}/{total}] cached   {url}")
        return None

    async with sem:
        try:
            async with session.get(url, timeout=timeout) as resp:
                resp.raise_for_status()
                html = await resp.text()
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            counters["downloaded"] += 1
            print(f"[{index}/{total}] fetched  {url}")
            if url in previously_failed:
                remove_from_failed_log(url)
            return None
        except aiohttp.ClientResponseError as e:
            # The server answered with an error status: permanent, don't retry.
            counters["failed"] += 1
            print(f"[{index}/{total}] FAILED   {url}  (HTTP {e.status} {e.message})")
            if url not in previously_failed:
                log_failed(url)
            return None
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            reason = "timeout" if isinstance(e, asyncio.TimeoutError) else f"{type(e).__name__}: {e}"
            if not final_pass:
                # Transient (timeout / connection drop): defer to the slow pass.
                print(f"[{index}/{total}] retry?   {url}  ({reason})")
                return url
            counters["failed"] += 1
            print(f"[{index}/{total}] FAILED   {url}  ({reason})")
            if url not in previously_failed:
                log_failed(url)
            return None
        except Exception as e:
            counters["failed"] += 1
            print(f"[{index}/{total}] FAILED   {url}  ({type(e).__name__}: {e})")
            if url not in previously_failed:
                log_failed(url)
            return None


async def run_pass(
    session: aiohttp.ClientSession,
    urls: list[str],
    concurrency: int,
    timeout: aiohttp.ClientTimeout,
    previously_failed: set[str],
    counters: dict,
    final_pass: bool,
) -> list[str]:
    """Run one download pass over urls, returning those to retry."""
    sem = asyncio.Semaphore(concurrency)
    total = len(urls)
    tasks = [
        download(session, sem, url, i, total, previously_failed, counters, timeout, final_pass)
        for i, url in enumerate(urls, start=1)
    ]
    results = await asyncio.gather(*tasks)
    return [u for u in results if u is not None]


async def main() -> None:
    df = pl.read_csv("./data/raw/RecipeNLG_dataset.csv")

    df = df.with_columns(
        pl.when(pl.col("link").str.starts_with("www.")).then(
            pl.col("link")
        ).otherwise(
            pl.lit("www.") + pl.col("link")
        )
    )

    df = df.with_columns(
        pl.col("link").str.extract(r"www\..*\.[a-z]{2,3}/", 0).alias("website")
    )

    os.makedirs(HTML_CACHE_DIR, exist_ok=True)

    links = df["link"].to_list()
    previously_failed = load_failed_urls()
    counters = {"downloaded": 0, "skipped": 0, "failed": 0}

    urls = [f"http://{link}" for link in links]

    # Many distinct hostnames -> use aiodns for concurrent DNS resolution, cache
    # lookups, and keep connections alive across requests via a single session.
    try:
        resolver = aiohttp.AsyncResolver()
    except Exception as e:
        print(f"aiodns unavailable ({e}); falling back to default resolver.")
        resolver = None

    connector = aiohttp.TCPConnector(
        limit=CONCURRENCY,
        limit_per_host=8,
        ttl_dns_cache=300,
        resolver=resolver,
    )

    async with aiohttp.ClientSession(connector=connector, headers=HEADERS) as session:
        # Fast pass: aggressive timeouts; transient failures are collected.
        print(f"Pass 1: {len(urls)} URLs  (concurrency={CONCURRENCY}, fast timeout)")
        to_retry = await run_pass(
            session, urls, CONCURRENCY, FAST_TIMEOUT, previously_failed, counters, final_pass=False
        )

        # Slow pass: retry only the timed-out / dropped URLs, gently.
        if to_retry:
            print(
                f"\nPass 2: retrying {len(to_retry)} timed-out URLs "
                f"(concurrency={SLOW_CONCURRENCY}, slow timeout)"
            )
            await run_pass(
                session, to_retry, SLOW_CONCURRENCY, SLOW_TIMEOUT, previously_failed, counters, final_pass=True
            )

    d, s, f = counters["downloaded"], counters["skipped"], counters["failed"]
    print(f"\nDone. fetched={d}  cached={s}  failed={f}")
    if f:
        print(f"Failed URLs logged to {FAILED_LOG} — re-run the script to retry them.")


if __name__ == "__main__":
    asyncio.run(main())
