"""
Tests for SLO monitoring service.

Validates that gap violation rate is correctly calculated.
"""
import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from app.observability.metrics import SLO_GAP_VIOLATION_RATE


class FakeDB:
    """Fake database for testing SLO monitor."""

    def __init__(self, now: datetime, good_symbols: list[str], bad_symbols: list[str]):
        """
        Args:
            now: Current timestamp
            good_symbols: Symbols with recent data (no gaps)
            bad_symbols: Symbols with stale data (gaps)
        """
        self.now = now
        self.good_symbols = good_symbols
        self.bad_symbols = bad_symbols
        self.pool = self  # Fake pool for async with

    def acquire(self):
        """Fake pool.acquire() context manager."""
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def fetch(self, sql: str, tier: str, limit: int):
        """Fake symbol fetching."""
        all_symbols = self.good_symbols + self.bad_symbols
        return [{"symbol": s} for s in all_symbols[:limit]]

    async def fetchrow(self, sql: str, symbol: str, interval: str):
        """Fake cursor fetching."""
        if symbol in self.good_symbols:
            # Recent data (1 minute ago)
            last_ts = self.now - timedelta(minutes=1)
        elif symbol in self.bad_symbols:
            # Stale data (5 minutes ago - gap violation for 1m interval)
            last_ts = self.now - timedelta(minutes=5)
        else:
            return None  # No data

        return {"last_ts": last_ts}


@pytest.mark.asyncio
async def test_slo_gap_violation_rate_40_percent():
    """Test SLO monitor with 40% violation rate (2 out of 5 symbols)."""
    now = datetime.now(timezone.utc)
    good_symbols = ["AAPL", "MSFT", "GOOGL"]
    bad_symbols = ["TSLA", "NVDA"]

    db = FakeDB(now, good_symbols, bad_symbols)

    from app.services.slo_monitor import SLOMonitor
    monitor = SLOMonitor(db)

    # Sample T0 tier
    await monitor._sample("T0", "1m", sample_size=5)

    # Check metric value
    metric = SLO_GAP_VIOLATION_RATE.labels(tier="T0", interval="1m")
    violation_rate = metric._value.get()

    # Should be 2/5 = 0.4
    assert 0.39 < violation_rate < 0.41, f"Expected ~0.4, got {violation_rate}"


@pytest.mark.asyncio
async def test_slo_gap_violation_rate_zero_percent():
    """Test SLO monitor with 0% violation rate (all symbols healthy)."""
    now = datetime.now(timezone.utc)
    good_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    bad_symbols = []

    db = FakeDB(now, good_symbols, bad_symbols)

    from app.services.slo_monitor import SLOMonitor
    monitor = SLOMonitor(db)

    await monitor._sample("T1", "1m", sample_size=5)

    metric = SLO_GAP_VIOLATION_RATE.labels(tier="T1", interval="1m")
    violation_rate = metric._value.get()

    assert violation_rate == 0.0, f"Expected 0.0, got {violation_rate}"


@pytest.mark.asyncio
async def test_slo_gap_violation_rate_100_percent():
    """Test SLO monitor with 100% violation rate (all symbols have gaps)."""
    now = datetime.now(timezone.utc)
    good_symbols = []
    bad_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

    db = FakeDB(now, good_symbols, bad_symbols)

    from app.services.slo_monitor import SLOMonitor
    monitor = SLOMonitor(db)

    await monitor._sample("T2", "1m", sample_size=5)

    metric = SLO_GAP_VIOLATION_RATE.labels(tier="T2", interval="1m")
    violation_rate = metric._value.get()

    assert violation_rate == 1.0, f"Expected 1.0, got {violation_rate}"


@pytest.mark.asyncio
async def test_slo_monitor_empty_tier():
    """Test SLO monitor when tier has no symbols."""
    now = datetime.now(timezone.utc)
    good_symbols = []
    bad_symbols = []

    db = FakeDB(now, good_symbols, bad_symbols)

    from app.services.slo_monitor import SLOMonitor
    monitor = SLOMonitor(db)

    await monitor._sample("T0", "1m", sample_size=5)

    metric = SLO_GAP_VIOLATION_RATE.labels(tier="T0", interval="1m")
    violation_rate = metric._value.get()

    # Should default to 0.0 when no symbols
    assert violation_rate == 0.0


def test_slo_monitor_event_loop_integration():
    """Test SLO monitor runs in event loop without errors."""
    now = datetime.now(timezone.utc)
    good_symbols = ["AAPL", "MSFT", "GOOGL"]
    bad_symbols = ["TSLA", "NVDA"]

    db = FakeDB(now, good_symbols, bad_symbols)

    async def run_sample():
        from app.services.slo_monitor import SLOMonitor
        monitor = SLOMonitor(db)
        await monitor._sample("T0", "1m", sample_size=5)

    # Run in existing event loop
    asyncio.get_event_loop().run_until_complete(run_sample())

    # Verify metric was updated
    metric = SLO_GAP_VIOLATION_RATE.labels(tier="T0", interval="1m")
    violation_rate = metric._value.get()

    assert 0.39 < violation_rate < 0.41


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
