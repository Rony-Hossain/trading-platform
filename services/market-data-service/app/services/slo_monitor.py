"""
SLO Monitor Service.

Continuously samples symbol coverage to detect gap violations.
Updates the `slo_gap_violation_rate` metric used for SLO-based alerts.

Target SLOs:
- T0: <0.5% gap violation rate (p99.5 coverage)
- T1: <2% gap violation rate (p98 coverage)
- T2: <5% gap violation rate (p95 coverage)
"""
from __future__ import annotations
import asyncio
import random
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from app.observability.metrics import SLO_GAP_VIOLATION_RATE

if TYPE_CHECKING:
    from app.services.database import DatabaseService


class SLOMonitor:
    """
    Background service that samples symbols to measure gap SLO compliance.

    Runs every 30 seconds, samples 200 symbols per tier, and updates
    Prometheus metrics for alerting.
    """

    def __init__(self, db: DatabaseService):
        self.db = db
        self.running = False

    async def run(self):
        """Main monitoring loop."""
        self.running = True
        while self.running:
            try:
                # Sample T0 1m bars
                await self._sample("T0", "1m", sample_size=200)
                # Sample T1 1m bars
                await self._sample("T1", "1m", sample_size=200)
                # Sample T2 1d bars
                await self._sample("T2", "1d", sample_size=200)
            except Exception as e:
                # Log but don't crash the monitor
                print(f"[SLOMonitor] Error in sampling: {e}")

            await asyncio.sleep(30)

    async def stop(self):
        """Stop the monitoring loop."""
        self.running = False

    async def _sample(self, tier: str, interval: str, sample_size: int):
        """
        Sample symbols from a tier and measure gap violations.

        Args:
            tier: Symbol tier (T0, T1, T2)
            interval: Data interval (1m, 1d)
            sample_size: Number of symbols to sample (200 recommended)
        """
        # Get symbols in this tier (fetch 2x to ensure we can sample)
        symbols = await self._get_symbols_by_tier(tier, limit=sample_size * 2)

        # Sample randomly to avoid bias
        sampled = random.sample(symbols, k=min(sample_size, len(symbols)))

        if not sampled:
            # No symbols in tier, set violation rate to 0
            SLO_GAP_VIOLATION_RATE.labels(tier=tier, interval=interval).set(0.0)
            return

        now = datetime.now(timezone.utc)
        dt = timedelta(minutes=1) if interval == "1m" else timedelta(days=1)

        violations = 0
        for symbol in sampled:
            last_ts = await self._get_cursor(symbol, interval, source="auto")

            # Check if gap > 2x interval (SLO violation)
            if not last_ts or (now - last_ts) > (2 * dt):
                violations += 1

        # Calculate and update violation rate
        violation_rate = violations / len(sampled)
        SLO_GAP_VIOLATION_RATE.labels(tier=tier, interval=interval).set(violation_rate)

    async def _get_symbols_by_tier(self, tier: str, limit: int) -> list[str]:
        """
        Get symbols in a specific tier.

        Args:
            tier: Symbol tier (T0, T1, T2)
            limit: Maximum number of symbols to return

        Returns:
            List of symbol strings
        """
        sql = """
            SELECT symbol
            FROM symbol_universe
            WHERE tier = $1 AND active = true
            ORDER BY RANDOM()
            LIMIT $2
        """
        async with self.db.pool.acquire() as con:
            rows = await con.fetch(sql, tier, limit)
            return [r["symbol"] for r in rows]

    async def _get_cursor(
        self,
        symbol: str,
        interval: str,
        source: str = "auto"
    ) -> datetime | None:
        """
        Get the last ingestion cursor timestamp for a symbol.

        Args:
            symbol: Symbol to check
            interval: Data interval (1m, 1d)
            source: Provider source or "auto" for latest

        Returns:
            Last cursor timestamp or None if never ingested
        """
        if source == "auto":
            # Get latest cursor from any provider
            sql = """
                SELECT last_ts
                FROM ingestion_cursor
                WHERE symbol = $1 AND interval = $2
                ORDER BY last_ts DESC
                LIMIT 1
            """
            async with self.db.pool.acquire() as con:
                row = await con.fetchrow(sql, symbol, interval)
                return row["last_ts"] if row else None
        else:
            # Get cursor for specific provider
            sql = """
                SELECT last_ts
                FROM ingestion_cursor
                WHERE symbol = $1 AND interval = $2 AND provider = $3
                ORDER BY last_ts DESC
                LIMIT 1
            """
            async with self.db.pool.acquire() as con:
                row = await con.fetchrow(sql, symbol, interval, source)
                return row["last_ts"] if row else None
