import asyncio
import random
import time
from datetime import datetime, timezone, timedelta
from typing import AsyncIterator
import structlog
from ..config import settings

log = structlog.get_logger(__name__)

class CircuitBreaker:
    def __init__(self, trip_rate: float, min_window: int, open_seconds: int):
        self.trip_rate = trip_rate
        self.min_window = min_window
        self.open_seconds = open_seconds
        self.window = []  # list[bool] (True = error)
        self.state = "CLOSED"
        self.next_probe = 0.0

    def record(self, error: bool):
        self.window.append(error)
        if len(self.window) > 200:
            self.window = self.window[-200:]
        if self.state == "CLOSED" and len(self.window) >= self.min_window:
            rate = sum(self.window) / len(self.window)
            if rate >= self.trip_rate:
                self.state = "OPEN"
                self.next_probe = time.time() + self.open_seconds
                log.warning("circuit_open", rate=rate)
        elif self.state == "HALF_OPEN":
            if not error:
                # success probe → close
                self.state = "CLOSED"
                self.window.clear()
            else:
                self.state = "OPEN"
                self.next_probe = time.time() + self.open_seconds

    def can_call(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN" and time.time() >= self.next_probe:
            self.state = "HALF_OPEN"
            return True
        return self.state == "HALF_OPEN"

async def retry_with_backoff(coro_func, max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 10.0):
    """
    ✅ Retry helper with exponential backoff and jitter.
    Retries transient errors before feeding circuit breaker.
    """
    for attempt in range(max_attempts):
        try:
            return await coro_func()
        except Exception as e:
            if attempt >= max_attempts - 1:
                # Last attempt failed, re-raise
                raise
            # Calculate exponential backoff with jitter
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
            log.warning("retry_attempt", attempt=attempt + 1, delay=delay, error=str(e))
            await asyncio.sleep(delay)

class AdapterBase:
    def __init__(self, name: str, out_queue, circuit: CircuitBreaker):
        self.name = name
        self.out_queue = out_queue
        self.circuit = circuit
        self.safety_lag = timedelta(seconds=60)
        self.since_ts = datetime.now(timezone.utc) - timedelta(minutes=5)

    async def fetch_window(self, start: datetime, end: datetime) -> AsyncIterator[dict]:
        """Override in child; yield provider items {source, external_id, url, title, body?, ...} """
        if False:
            yield {}

    async def run(self):
        while True:
            start = self.since_ts
            end = datetime.now(timezone.utc) - self.safety_lag
            if end <= start:
                await asyncio.sleep(0.5)
                continue
            try:
                if not self.circuit.can_call():
                    await asyncio.sleep(1.0)
                    continue

                # ✅ Wrap fetch_window in retry logic
                async def _fetch_with_retry():
                    items = []
                    async for item in self.fetch_window(start, end):
                        items.append(item)
                        await self.out_queue.put(item)
                    return items

                await retry_with_backoff(
                    _fetch_with_retry,
                    max_attempts=settings.ADAPTER_RETRY_ATTEMPTS,
                    base_delay=settings.ADAPTER_RETRY_BASE_DELAY,
                    max_delay=settings.ADAPTER_RETRY_MAX_DELAY
                )

                self.circuit.record(False)
                self.since_ts = end
            except Exception as e:
                log.error("adapter_error", source=self.name, err=str(e))
                self.circuit.record(True)
                await asyncio.sleep(1.0)
