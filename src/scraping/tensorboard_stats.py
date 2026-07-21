"""Scrapy extension that mirrors crawl statistics to TensorBoard.

TensorBoard is a file reader, not a metrics server: a ``SummaryWriter``
appends scalar events to a log directory and ``tensorboard --logdir
runs/scrape`` renders whatever it finds there, re-reading every ~30s while
the crawl is running. Each run writes to its own timestamped subdirectory,
so a restarted crawl shows up as a second overlaid curve instead of
corrupting the first.

Design: counting and sampling are decoupled. The spider (and Scrapy's own
downloader/middlewares) increment plain counters in the crawler's stats
collector; this extension wakes up once per TENSORBOARD_INTERVAL on a
Twisted ``LoopingCall`` and logs one point per tag. Per-event logging in
save_page would emit thousands of points a minute — TensorBoard downsamples
them anyway, and the event file bloats for nothing.

Counters vs. gauges: monotone counters (pages saved, responses, bytes) are
logged as the *delta* since the previous tick, which turns them into rates
where throttling, dead-host stalls and the retry phase are visible as dips;
their cumulative totals go to the ``progress/`` section. Gauges that can go
both ways (remaining, retry queue) are read off the spider and logged raw.

The tag prefix before the first ``/`` (``rate/``, ``http/``, ``progress/``)
is what TensorBoard groups plots by, so those prefixes *are* the dashboard
layout.

LoopingCall (rather than a timer thread) matters: the whole crawl runs on
the single reactor thread, so ``tick`` interleaves with spider callbacks and
can read spider state without locks — a few add_scalar calls are
microseconds, and the writer flushes from its own background thread.
"""

import os
import time

from scrapy import signals
from tensorboardX import SummaryWriter
from twisted.internet import task

from data_generation_config import TENSORBOARD_LOG_DIR, TENSORBOARD_INTERVAL

# Scrapy's downloader maintains one stats key per HTTP status seen, e.g.
# "downloader/response_status_count/200"; everything under this prefix is
# mirrored dynamically so new status codes need no code change.
STATUS_PREFIX = "downloader/response_status_count/"


class TensorboardStats:
    def __init__(self, stats):
        self.stats = stats
        # Last tick's snapshot of every counter we log as a rate, keyed by
        # stats-collector key; deltas against it turn totals into per-minute
        # rates.
        self.prev: dict[str, int] = {}
        self.step = 0

    @classmethod
    def from_crawler(cls, crawler):
        ext = cls(crawler.stats)
        crawler.signals.connect(ext.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(ext.spider_closed, signal=signals.spider_closed)
        return ext

    def spider_opened(self, spider):
        run_dir = os.path.join(
            TENSORBOARD_LOG_DIR, time.strftime("%Y-%m-%d_%H-%M-%S")
        )
        self.writer = SummaryWriter(run_dir, flush_secs=30)
        spider.logger.info("TensorBoard: logging to %s", run_dir)
        # now=False: skip the immediate t=0 firing so the first point is a
        # full interval's worth of downloads, not a zero at minute 0.
        self.loop = task.LoopingCall(self.tick, spider)
        self.loop.start(TENSORBOARD_INTERVAL, now=False)

    def _rate(self, tag, key, scale=1.0):
        value = self.stats.get_value(key, 0)
        self.writer.add_scalar(tag, (value - self.prev.get(key, 0)) * scale, self.step)
        self.prev[key] = value

    def tick(self, spider):
        # Step axis = ticks since start; with the default 60s interval that
        # is minutes of crawl time (TensorBoard's "Relative"/"Wall" x-axis
        # modes use the recorded walltime instead, if preferred).
        self.step += 1

        # --- rates (delta since last tick) ---
        self._rate("rate/htmls_per_min", "scrape/htmls_saved")
        self._rate("rate/responses_per_min", "response_received_count")
        self._rate("rate/requests_per_min", "downloader/request_count")
        self._rate("rate/failed_permanently_per_min", "scrape/failed_permanently")
        self._rate("rate/mb_per_min", "downloader/response_bytes", scale=1 / 1e6)
        # Per-status breakdown shows *why* throughput drops (429/403 bans
        # look very different from 404 dead pages or 5xx flakiness).
        for key in list(self.stats.get_stats()):
            if key.startswith(STATUS_PREFIX):
                self._rate(f"http/{key[len(STATUS_PREFIX):]}_per_min", key)

        # --- cumulative totals ---
        self.writer.add_scalar(
            "progress/htmls_saved_total",
            self.stats.get_value("scrape/htmls_saved", 0),
            self.step,
        )
        self.writer.add_scalar(
            "progress/failed_permanently_total",
            self.stats.get_value("scrape/failed_permanently", 0),
            self.step,
        )
        self.writer.add_scalar(
            "progress/backlog_drawn_total",
            self.stats.get_value("scrape/backlog_drawn", 0),
            self.step,
        )

        # --- gauges (read raw off the spider; needs RecipeSpider's
        # remaining/retry_pending attributes) ---
        self.writer.add_scalar("progress/remaining", spider.remaining, self.step)
        self.writer.add_scalar(
            "progress/retry_queue", len(spider.retry_pending), self.step
        )

    def spider_closed(self, spider, reason):
        # A final tick captures the partial last interval, then the writer
        # must be closed: it buffers, and without close() the tail of the
        # run can be lost. Stopping the loop also lets the reactor shut
        # down cleanly.
        if self.loop.running:
            self.loop.stop()
        self.tick(spider)
        self.writer.close()
