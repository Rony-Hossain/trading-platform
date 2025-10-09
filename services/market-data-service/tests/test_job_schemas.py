"""
Tests for job schema validation with Pydantic.

Ensures all jobs follow consistent schema and can be validated in CI.
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from app.schemas.jobs import (
    BarsFetchJob,
    QuotesFetchJob,
    BackfillJob,
    TimeWindow,
    fixture_bars_job,
    fixture_quotes_job,
    fixture_backfill_job,
)


def test_bars_fetch_job_valid():
    """Test valid BarsFetchJob."""
    job = BarsFetchJob(
        job_id="test-123",
        type="bars_fetch",
        symbols=["AAPL", "MSFT"],
        interval="1m",
        time_window=TimeWindow(
            start=datetime(2025, 10, 8, 13, 30),
            end=datetime(2025, 10, 8, 13, 40)
        ),
        priority="T0"
    )

    assert job.job_id == "test-123"
    assert job.type == "bars_fetch"
    assert job.symbols == ["AAPL", "MSFT"]
    assert job.interval == "1m"
    assert job.priority == "T0"


def test_bars_fetch_job_invalid_interval():
    """Test BarsFetchJob rejects invalid interval."""
    with pytest.raises(ValidationError):
        BarsFetchJob(
            job_id="test-123",
            type="bars_fetch",
            symbols=["AAPL"],
            interval="3m",  # Invalid interval
            time_window=TimeWindow(
                start=datetime(2025, 10, 8, 13, 30),
                end=datetime(2025, 10, 8, 13, 40)
            ),
            priority="T0"
        )


def test_bars_fetch_job_empty_symbols():
    """Test BarsFetchJob rejects empty symbol list."""
    with pytest.raises(ValidationError):
        BarsFetchJob(
            job_id="test-123",
            type="bars_fetch",
            symbols=[],  # Empty list
            interval="1m",
            time_window=TimeWindow(
                start=datetime(2025, 10, 8, 13, 30),
                end=datetime(2025, 10, 8, 13, 40)
            ),
            priority="T0"
        )


def test_bars_fetch_job_too_many_symbols():
    """Test BarsFetchJob rejects >1000 symbols."""
    with pytest.raises(ValidationError):
        BarsFetchJob(
            job_id="test-123",
            type="bars_fetch",
            symbols=[f"SYM{i}" for i in range(1001)],  # 1001 symbols
            interval="1m",
            time_window=TimeWindow(
                start=datetime(2025, 10, 8, 13, 30),
                end=datetime(2025, 10, 8, 13, 40)
            ),
            priority="T0"
        )


def test_quotes_fetch_job_valid():
    """Test valid QuotesFetchJob."""
    job = QuotesFetchJob(
        job_id="test-456",
        type="quotes_fetch",
        symbols=["TSLA", "NVDA"],
        priority="T1"
    )

    assert job.job_id == "test-456"
    assert job.type == "quotes_fetch"
    assert job.symbols == ["TSLA", "NVDA"]
    assert job.priority == "T1"


def test_backfill_job_valid():
    """Test valid BackfillJob."""
    job = BackfillJob(
        job_id="test-789",
        type="backfill",
        symbol="GOOGL",
        interval="1m",
        time_window=TimeWindow(
            start=datetime(2025, 10, 8, 10, 0),
            end=datetime(2025, 10, 8, 10, 15)
        ),
        priority="T2"
    )

    assert job.job_id == "test-789"
    assert job.type == "backfill"
    assert job.symbol == "GOOGL"
    assert job.interval == "1m"
    assert job.priority == "T2"


def test_backfill_job_invalid_priority():
    """Test BackfillJob rejects invalid priority."""
    with pytest.raises(ValidationError):
        BackfillJob(
            job_id="test-789",
            type="backfill",
            symbol="GOOGL",
            interval="1m",
            time_window=TimeWindow(
                start=datetime(2025, 10, 8, 10, 0),
                end=datetime(2025, 10, 8, 10, 15)
            ),
            priority="T3"  # Invalid priority
        )


def test_fixture_bars_job():
    """Test bars fetch job fixture."""
    job = fixture_bars_job()

    assert isinstance(job, BarsFetchJob)
    assert job.job_id.startswith("ci-")
    assert job.type == "bars_fetch"
    assert len(job.symbols) > 0
    assert job.interval in ["1m", "5m", "15m", "1h", "1d"]
    assert job.priority in ["T0", "T1", "T2"]


def test_fixture_quotes_job():
    """Test quotes fetch job fixture."""
    job = fixture_quotes_job()

    assert isinstance(job, QuotesFetchJob)
    assert job.job_id.startswith("ci-")
    assert job.type == "quotes_fetch"
    assert len(job.symbols) > 0


def test_fixture_backfill_job():
    """Test backfill job fixture."""
    job = fixture_backfill_job()

    assert isinstance(job, BackfillJob)
    assert job.job_id.startswith("ci-")
    assert job.type == "backfill"
    assert isinstance(job.symbol, str)
    assert job.interval in ["1m", "5m", "15m", "1h", "1d"]


def test_time_window_valid():
    """Test TimeWindow validation."""
    tw = TimeWindow(
        start=datetime(2025, 10, 8, 10, 0),
        end=datetime(2025, 10, 8, 11, 0)
    )

    assert tw.start < tw.end


def test_provider_hint_optional():
    """Test that provider_hint is optional."""
    job = BarsFetchJob(
        job_id="test-123",
        type="bars_fetch",
        symbols=["AAPL"],
        interval="1m",
        time_window=TimeWindow(
            start=datetime(2025, 10, 8, 13, 30),
            end=datetime(2025, 10, 8, 13, 40)
        ),
        priority="T0"
        # No provider_hint specified
    )

    assert job.provider_hint is None


def test_provider_hint_specified():
    """Test that provider_hint can be specified."""
    job = BarsFetchJob(
        job_id="test-123",
        type="bars_fetch",
        symbols=["AAPL"],
        interval="1m",
        time_window=TimeWindow(
            start=datetime(2025, 10, 8, 13, 30),
            end=datetime(2025, 10, 8, 13, 40)
        ),
        priority="T0",
        provider_hint="polygon"
    )

    assert job.provider_hint == "polygon"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
