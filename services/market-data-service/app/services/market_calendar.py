from __future__ import annotations

import functools
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import exchange_calendars as xcals


class CalendarManager:
    """Memoized helper that snaps timestamps to the XNYS schedule."""

    def __init__(self) -> None:
        self._calendar = xcals.get_calendar("XNYS")
        self._day_cache: Dict[str, Tuple[Optional[datetime], Optional[datetime]]] = {}
        self._preload(days_back=14, days_forward=7)

    def _preload(self, days_back: int, days_forward: int) -> None:
        today = datetime.now(timezone.utc).date()
        for delta in range(-days_back, days_forward + 1):
            day = today + timedelta(days=delta)
            key = day.isoformat()
            try:
                schedule = self._calendar.schedule.loc[key]
                open_dt = schedule["market_open"].to_pydatetime().astimezone(timezone.utc)
                close_dt = schedule["market_close"].to_pydatetime().astimezone(timezone.utc)
                self._day_cache[key] = (open_dt, close_dt)
            except KeyError:
                self._day_cache[key] = (None, None)

    @functools.lru_cache(maxsize=100_000)
    def align_minute(self, ts: datetime) -> datetime:
        """Snap timestamp to the exact exchange minute boundary in UTC."""
        ts = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
        key = ts.date().isoformat()
        if key not in self._day_cache:
            self._preload(1, 1)
        open_dt, close_dt = self._day_cache.get(key, (None, None))
        if open_dt is None or close_dt is None:
            return ts
        if ts < open_dt:
            return open_dt
        if ts > close_dt:
            return close_dt
        return ts


CAL = CalendarManager()
