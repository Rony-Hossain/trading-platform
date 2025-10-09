"""
Data Collector Service - M2/M3 Implementation
Handles RLC job consumption, gap detection, and backfill orchestration.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from app.core.config import get_settings
from app.observability.metrics import (
    BACKFILL_COMPLETED,
    BACKFILL_ENQUEUED,
    BACKFILL_OLDEST_AGE,
    BACKFILL_QUEUE_DEPTH,
    GAPS_FOUND,
    JOBS_PROCESSED,
)
from app.services.database import DatabaseService

logger = logging.getLogger(__name__)


def _delta_from_interval(interval: str) -> timedelta:
    """Convert interval string to timedelta."""
    if interval == "1m":
        return timedelta(minutes=1)
    elif interval == "5m":
        return timedelta(minutes=5)
    elif interval == "15m":
        return timedelta(minutes=15)
    elif interval == "1h":
        return timedelta(hours=1)
    elif interval == "1d":
        return timedelta(days=1)
    else:
        return timedelta(minutes=1)  # default


class CollectorJobs:
    """Simple Redis-backed job queue for RLC integration and backfills."""

    def __init__(self, redis_url: str, jobs_key: str, backfill_keys: Dict[str, str]):
        try:
            import redis.asyncio as aioredis
            self.r = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        except ImportError:
            logger.warning("redis not installed; RLC mode disabled")
            self.r = None
        self.jobs_key = jobs_key
        self.backfill_keys = backfill_keys  # {"T0": "market:backfills:T0", ...}

    async def push_job(self, job: Dict) -> None:
        if not self.r:
            return
        await self.r.lpush(self.jobs_key, json.dumps(job))

    async def pop_job(self, timeout: int = 1) -> Optional[Dict]:
        if not self.r:
            return None
        try:
            v = await self.r.brpop(self.jobs_key, timeout=timeout)
            if not v:
                return None
            _, payload = v
            return json.loads(payload)
        except Exception as exc:
            logger.debug("pop_job error: %s", exc)
            return None

    async def push_backfill(self, tier: str, job: Dict) -> None:
        if not self.r:
            return
        key = self.backfill_keys.get(tier, self.backfill_keys["T2"])
        await self.r.lpush(key, json.dumps(job))

    async def pop_backfill(self, tiers: List[str], timeout: int = 1) -> Optional[Dict]:
        if not self.r:
            return None
        # Try tiers in priority order
        for tier in sorted(tiers, key=lambda t: {"T0": 0, "T1": 1, "T2": 2}.get(t, 9)):
            key = self.backfill_keys.get(tier)
            if not key:
                continue
            try:
                v = await self.r.brpop(key, timeout=0)  # non-blocking check
                if v:
                    _, payload = v
                    return json.loads(payload)
            except Exception:
                continue
        return None

    async def backfill_depth(self, tier: str) -> int:
        if not self.r:
            return 0
        key = self.backfill_keys.get(tier)
        if not key:
            return 0
        try:
            return await self.r.llen(key)
        except Exception:
            return 0

    async def backfill_oldest(self, tier: str) -> Optional[Dict]:
        if not self.r:
            return None
        key = self.backfill_keys.get(tier)
        if not key:
            return None
        try:
            oldest_raw = await self.r.lindex(key, -1)  # oldest at tail
            if oldest_raw:
                return json.loads(oldest_raw)
        except Exception:
            pass
        return None


class DataCollectorService:
    """
    Background service for collecting and storing market data.
    Implements M2 (RLC jobs + local sweep) and M3 (gap detection + backfill).
    """

    def __init__(self, db: DatabaseService, registry=None):
        self.db = db
        self.registry = registry  # provider registry from main
        self.running = False
        self.task: Optional[asyncio.Task] = None

        s = get_settings()
        self.jobs = CollectorJobs(
            s.RLC_REDIS_URL,
            s.RLC_REDIS_JOBS_KEY,
            s.RLC_REDIS_BACKFILL_KEYS,
        )

        # Backfill concurrency semaphores (per tier)
        self.sem = {
            "T0": asyncio.Semaphore(s.BACKFILL_MAX_CONCURRENCY_T0),
            "T1": asyncio.Semaphore(s.BACKFILL_MAX_CONCURRENCY_T1),
            "T2": asyncio.Semaphore(s.BACKFILL_MAX_CONCURRENCY_T2),
        }
        self.dispatch_tokens = s.BACKFILL_DISPATCH_RATE_PER_SEC
        self._last_dispatch = datetime.utcnow()

    # --------- PUBLIC TASKS ----------

    async def run(self) -> None:
        """Run job consumer + optional local sweeper + backfill worker."""
        if self.running:
            logger.warning("Data collector already running")
            return

        s = get_settings()
        self.running = True

        tasks = [asyncio.create_task(self.consume_jobs())]
        if s.LOCAL_SWEEP_ENABLED and not s.USE_RLC:
            tasks.append(asyncio.create_task(self.local_sweeper()))
        tasks.append(asyncio.create_task(self.backfill_worker()))
        tasks.append(asyncio.create_task(self.metrics_reporter()))

        self.task = asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Data collector started (USE_RLC=%s, LOCAL_SWEEP=%s)", s.USE_RLC, s.LOCAL_SWEEP_ENABLED)

    async def stop(self) -> None:
        """Stop the data collection background task."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Data collector stopped")

    # --------- JOB CONSUMER ----------

    async def consume_jobs(self) -> None:
        """Consume RLC jobs; fallback: no-op if queue is empty."""
        while self.running:
            job = await self.jobs.pop_job(timeout=2)
            if not job:
                await asyncio.sleep(0.1)
                continue
            try:
                t = job.get("type")
                if t == "bars_fetch":
                    await self._handle_bars_job(job)
                elif t == "quotes_fetch":
                    await self._handle_quotes_job(job)
                elif t == "backfill":
                    # push into backfill stream (throttled)
                    tier = job.get("priority", "T2")
                    await self.jobs.push_backfill(tier, job)
                else:
                    logger.debug("Unknown job type: %s", t)
            except Exception as exc:
                logger.error("Error handling job: %s", exc, exc_info=True)
                # TODO: emit metric + optional DLQ
                continue

    # --------- LOCAL SWEEPER (DEV) ----------

    async def local_sweeper(self) -> None:
        """Emit simple bars_fetch jobs by tier and cadence when RLC is off."""
        s = get_settings()
        offset = {"T0": 0, "T1": 0, "T2": 0}
        while self.running:
            now = datetime.utcnow()
            for tier, cadence in [("T0", s.CADENCE_T0), ("T1", s.CADENCE_T1), ("T2", s.CADENCE_T2)]:
                if "bars_1m" not in cadence and "eod" not in cadence:
                    continue
                try:
                    batch = await self.db.get_symbols_by_tier(tier, s.LOCAL_SWEEP_BATCH, offset[tier])
                except Exception as exc:
                    logger.warning("Error getting symbols for tier %s: %s", tier, exc)
                    offset[tier] = 0
                    continue

                if not batch:
                    offset[tier] = 0
                    continue
                offset[tier] += len(batch)

                end = now.replace(second=0, microsecond=0)
                start = end - timedelta(minutes=5) if tier != "T2" else end - timedelta(days=2)

                job = {
                    "type": "bars_fetch",
                    "symbols": batch,
                    "interval": "1m" if tier != "T2" else "1d",
                    "time_window": {"start": start.isoformat() + "Z", "end": end.isoformat() + "Z"},
                    "priority": tier,
                    "provider_hint": None,
                }
                await self.jobs.push_job(job)

            await asyncio.sleep(s.LOCAL_SWEEP_TICK_SEC)

    # --------- JOB HANDLERS ----------

    async def _handle_bars_job(self, job: Dict) -> None:
        """Handle bars_fetch job: rank providers, fetch, write, detect gaps."""
        if not self.registry:
            logger.warning("Registry not available; skipping bars job")
            return

        sym_list: List[str] = job.get("symbols", [])
        interval: str = job.get("interval", "1m")
        start = datetime.fromisoformat(job["time_window"]["start"].replace("Z", ""))
        end = datetime.fromisoformat(job["time_window"]["end"].replace("Z", ""))
        tier = job.get("priority", "T1")
        hint = job.get("provider_hint")

        # Rank providers once for this capability
        capability = "bars_1m" if interval == "1m" else "eod"
        providers = self.registry.rank(capability, provider_hint=hint)
        if not providers:
            logger.warning("No providers available for capability %s", capability)
            return

        for symbol in sym_list:
            provider_used = None
            bars_all: List[Dict] = []
            for pname in providers:
                adapter = self.registry.providers[pname].adapter
                try:
                    # Fetch bars from provider
                    data = await adapter.fetch_bars(symbol=symbol, start=start, end=end, interval=interval)
                    if not data:
                        continue
                    provider_used = pname
                    bars_all = data
                    JOBS_PROCESSED.labels(type="bars_fetch", provider=pname).inc()
                    break
                except Exception as exc:
                    logger.debug("Provider %s failed for %s: %s", pname, symbol, exc)
                    continue

            if not bars_all:
                # Could emit a small gap/backfill from start→end for this symbol
                await self._enqueue_gap(symbol, interval, start, end, tier)
                continue

            # Gap detection & upserts
            await self._write_with_gaps(symbol, interval, provider_used, bars_all, tier)

    async def _handle_quotes_job(self, job: Dict) -> None:
        """Handle quotes_fetch job: similar pattern to bars."""
        # Placeholder: same routing pattern, write to quotes table, etc.
        logger.debug("quotes_fetch job received but not implemented: %s", job)
        return

    # --------- GAP & BACKFILL CORE ----------

    async def _write_with_gaps(
        self,
        symbol: str,
        interval: str,
        provider_used: str,
        bars: List[Dict],
        tier: str,
    ) -> None:
        """Write bars and detect gaps."""
        # Sort bars and load cursor
        bars.sort(key=lambda b: b.get("ts", datetime.min))
        last_ts = await self.db.get_cursor(symbol, interval, provider_used)
        dt = _delta_from_interval(interval)

        # If no cursor, set it to first-interval-before first bar
        if last_ts is None and bars:
            last_ts = bars[0]["ts"] - dt

        expected = last_ts + dt if last_ts else (bars[0]["ts"] if bars else datetime.utcnow())

        # Detect gap before first bar
        if bars and bars[0]["ts"] > expected:
            await self._enqueue_gap(symbol, interval, expected, bars[0]["ts"] - dt, tier)

        # Bulk write all bars (idempotent)
        s = get_settings()
        await self.db.upsert_bars_bulk(bars, batch_size=s.LIVE_BATCH_SIZE, provider_used=provider_used)

        # Detect gaps between bars
        if len(bars) > 1:
            prev = bars[0]["ts"]
            for b in bars[1:]:
                want = prev + dt
                if b["ts"] > want:
                    await self._enqueue_gap(symbol, interval, want, b["ts"] - dt, tier)
                prev = b["ts"]

        # Advance cursor
        if bars:
            await self.db.update_cursor(symbol, interval, provider_used, bars[-1]["ts"], status="ok")

    async def _enqueue_gap(
        self,
        symbol: str,
        interval: str,
        start_ts: datetime,
        end_ts: datetime,
        tier: str,
    ) -> None:
        """Split large gaps into chunks and enqueue backfill jobs."""
        s = get_settings()
        chunk_duration = timedelta(minutes=s.BACKFILL_CHUNK_MINUTES) if interval == "1m" else timedelta(days=30)
        cur = start_ts
        count = 0
        while cur <= end_ts:
            nxt = min(end_ts, cur + chunk_duration)
            job = {
                "type": "backfill",
                "symbol": symbol,
                "interval": interval,
                "time_window": {"start": cur.isoformat() + "Z", "end": nxt.isoformat() + "Z"},
                "priority": tier,
            }
            await self.jobs.push_backfill(tier, job)
            GAPS_FOUND.labels(interval=interval).inc()
            BACKFILL_ENQUEUED.labels(tier=tier).inc()
            count += 1
            cur = nxt + _delta_from_interval(interval)

        if count > 0:
            logger.info("Enqueued %d backfill chunks for %s [%s] %s→%s", count, symbol, interval, start_ts, end_ts)

    # --------- BACKFILL WORKER ----------

    async def backfill_worker(self) -> None:
        """Backfill worker with rate-limiting and concurrency control."""
        s = get_settings()
        bucket = 0.0
        last = datetime.utcnow()
        while self.running:
            # refill tokens
            now = datetime.utcnow()
            bucket += (now - last).total_seconds() * s.BACKFILL_DISPATCH_RATE_PER_SEC
            bucket = min(bucket, 10.0)
            last = now

            if bucket < 1.0:
                await asyncio.sleep(0.1)
                continue

            job = await self.jobs.pop_backfill(["T0", "T1", "T2"], timeout=1)
            if not job:
                await asyncio.sleep(0.1)
                continue

            bucket -= 1.0
            tier = job.get("priority", "T2")
            async with self.sem.get(tier, self.sem["T2"]):
                await self._run_backfill(job)

    async def _run_backfill(self, job: Dict) -> None:
        """Execute a single backfill job."""
        if not self.registry:
            logger.warning("Registry not available; skipping backfill")
            BACKFILL_COMPLETED.labels(tier=job.get("priority", "T2"), status="failed").inc()
            return

        symbol = job["symbol"]
        interval = job.get("interval", "1m")
        start = datetime.fromisoformat(job["time_window"]["start"].replace("Z", ""))
        end = datetime.fromisoformat(job["time_window"]["end"].replace("Z", ""))
        tier = job.get("priority", "T2")

        providers = self.registry.rank("bars_1m" if interval == "1m" else "eod", provider_hint=None)
        s = get_settings()

        for pname in providers:
            adapter = self.registry.providers[pname].adapter
            try:
                data = await adapter.fetch_bars(symbol=symbol, start=start, end=end, interval=interval)
                if not data:
                    continue
                await self.db.upsert_bars_bulk(
                    data,
                    batch_size=s.BACKFILL_BATCH_SIZE,
                    provider_used=pname,
                )
                # advance cursor conservatively if needed
                if data:
                    max_ts = max(b["ts"] for b in data)
                    await self.db.update_cursor(symbol, interval, pname, max_ts, status="backfilled")
                BACKFILL_COMPLETED.labels(tier=tier, status="done").inc()
                JOBS_PROCESSED.labels(type="backfill", provider=pname).inc()
                logger.debug("Backfill completed for %s [%s] %s→%s via %s", symbol, interval, start, end, pname)
                return
            except Exception as exc:
                logger.debug("Backfill provider %s failed for %s: %s", pname, symbol, exc)
                continue

        # If we reach here, all providers failed
        BACKFILL_COMPLETED.labels(tier=tier, status="failed").inc()
        logger.warning("Backfill failed for %s [%s] %s→%s (all providers failed)", symbol, interval, start, end)

    # --------- METRICS REPORTER ----------

    async def metrics_reporter(self) -> None:
        """Periodically update backfill queue depth and oldest age metrics."""
        while self.running:
            try:
                for tier in ["T0", "T1", "T2"]:
                    depth = await self.jobs.backfill_depth(tier)
                    BACKFILL_QUEUE_DEPTH.set(depth)

                    if depth > 0:
                        oldest = await self.jobs.backfill_oldest(tier)
                        if oldest:
                            start_str = oldest.get("time_window", {}).get("start")
                            if start_str:
                                start_dt = datetime.fromisoformat(start_str.replace("Z", ""))
                                age = (datetime.utcnow() - start_dt).total_seconds()
                                BACKFILL_OLDEST_AGE.set(max(age, 0.0))
            except Exception as exc:
                logger.debug("Error updating backfill metrics: %s", exc)

            await asyncio.sleep(10)

    # --------- LEGACY API (for backwards compatibility) ----------

    async def start(self) -> None:
        """Legacy start method."""
        await self.run()

    async def collect_symbol_data(
        self,
        symbol: str,
        days_back: int = 365,
        force_update: bool = False,
    ) -> bool:
        """
        Legacy method for on-demand symbol collection.
        Now delegates to job queue.
        """
        logger.info("Legacy collect_symbol_data called for %s", symbol)
        now = datetime.utcnow()
        start = now - timedelta(days=days_back)
        job = {
            "type": "bars_fetch",
            "symbols": [symbol.upper()],
            "interval": "1d",
            "time_window": {"start": start.isoformat() + "Z", "end": now.isoformat() + "Z"},
            "priority": "T2",
            "provider_hint": None,
        }
        await self.jobs.push_job(job)
        return True

    async def get_collection_status(self) -> dict:
        """Get status information about the data collector."""
        s = get_settings()
        depths = {}
        for tier in ["T0", "T1", "T2"]:
            depths[tier] = await self.jobs.backfill_depth(tier)

        return {
            "running": self.running,
            "use_rlc": s.USE_RLC,
            "local_sweep_enabled": s.LOCAL_SWEEP_ENABLED,
            "backfill_queue_depths": depths,
            "policy_version": s.policy_version,
        }
