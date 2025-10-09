"""
Type-safe job schemas with Pydantic models for validation.

Ensures all jobs follow a consistent schema and can be validated in CI fixtures.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


JobType = Literal["bars_fetch", "quotes_fetch", "backfill"]


class TimeWindow(BaseModel):
    """Time window for data fetching."""
    start: datetime
    end: datetime


class BaseJob(BaseModel):
    """Base job schema with common fields."""
    job_id: str = Field(..., description="Unique job identifier")
    type: JobType
    priority: Literal["T0", "T1", "T2"] = "T1"
    provider_hint: Optional[str] = Field(None, description="Preferred provider (optional)")


class BarsFetchJob(BaseJob):
    """Job for fetching OHLCV bars."""
    type: Literal["bars_fetch"] = "bars_fetch"
    symbols: List[str] = Field(..., min_items=1, max_items=1000)
    interval: Literal["1m", "5m", "15m", "1h", "1d"] = "1m"
    time_window: TimeWindow


class QuotesFetchJob(BaseJob):
    """Job for fetching L1 quotes."""
    type: Literal["quotes_fetch"] = "quotes_fetch"
    symbols: List[str] = Field(..., min_items=1, max_items=1000)


class BackfillJob(BaseJob):
    """Job for backfilling gaps."""
    type: Literal["backfill"] = "backfill"
    symbol: str
    interval: Literal["1m", "5m", "15m", "1h", "1d"] = "1m"
    time_window: TimeWindow


# CI Fixtures for validation testing
def fixture_bars_job() -> BarsFetchJob:
    """Fixture for CI testing of bars fetch job."""
    return BarsFetchJob(
        job_id="ci-bars-123",
        type="bars_fetch",
        symbols=["AAPL", "MSFT"],
        interval="1m",
        time_window=TimeWindow(
            start=datetime(2025, 10, 8, 13, 30, 0),
            end=datetime(2025, 10, 8, 13, 40, 0)
        ),
        priority="T0"
    )


def fixture_quotes_job() -> QuotesFetchJob:
    """Fixture for CI testing of quotes fetch job."""
    return QuotesFetchJob(
        job_id="ci-quotes-456",
        type="quotes_fetch",
        symbols=["TSLA", "NVDA"],
        priority="T1"
    )


def fixture_backfill_job() -> BackfillJob:
    """Fixture for CI testing of backfill job."""
    return BackfillJob(
        job_id="ci-backfill-789",
        type="backfill",
        symbol="GOOGL",
        interval="1m",
        time_window=TimeWindow(
            start=datetime(2025, 10, 8, 10, 0, 0),
            end=datetime(2025, 10, 8, 10, 15, 0)
        ),
        priority="T2"
    )
