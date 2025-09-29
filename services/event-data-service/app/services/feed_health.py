"""Shared feed health monitoring utilities."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class FeedStatus:
    """Current status snapshot for a single data feed."""

    name: str
    status: str = "unknown"
    consecutive_failures: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    last_event_count: Optional[int] = None
    message: Optional[str] = None
    alert_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.last_success:
            payload["last_success"] = self.last_success.isoformat()
        if self.last_failure:
            payload["last_failure"] = self.last_failure.isoformat()
        return payload


class AlertDispatcher:
    """Base dispatcher that can emit alerts when feeds degrade."""

    async def send_alert(self, payload: Dict[str, Any]) -> None:  # noqa: D401
        """Send an alert notification."""
        raise NotImplementedError


class LoggingAlertDispatcher(AlertDispatcher):
    async def send_alert(self, payload: Dict[str, Any]) -> None:
        logger.warning("Feed alert: %s", json.dumps(payload, sort_keys=True))


class WebhookAlertDispatcher(AlertDispatcher):
    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 5.0) -> None:
        self._url = url
        self._headers = headers or {}
        self._timeout = timeout

    async def send_alert(self, payload: Dict[str, Any]) -> None:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            await client.post(self._url, json=payload, headers=self._headers)


class FeedHealthMonitor:
    """Tracks feed health and triggers alerts on successive failures."""

    def __init__(
        self,
        service_name: str,
        alert_threshold: int = 3,
        dispatcher: Optional[AlertDispatcher] = None,
    ) -> None:
        self._service_name = service_name
        self._alert_threshold = max(alert_threshold, 1)
        self._dispatcher = dispatcher or LoggingAlertDispatcher()
        self._statuses: Dict[str, FeedStatus] = {}
        self._lock = asyncio.Lock()

    async def report_success(self, feed_name: str, event_count: Optional[int] = None, message: Optional[str] = None) -> None:
        async with self._lock:
            status = self._statuses.get(feed_name, FeedStatus(name=feed_name))
            previously_degraded = status.alert_active or status.consecutive_failures > 0
            status.status = "healthy"
            status.consecutive_failures = 0
            status.last_success = datetime.now(timezone.utc)
            status.last_event_count = event_count
            status.message = message
            status.alert_active = False
            self._statuses[feed_name] = status
        if previously_degraded:
            await self._dispatch(
                feed_name,
                severity="info",
                message=message or "Feed recovered",
                extra={"event_count": event_count},
            )

    async def report_failure(self, feed_name: str, error: str) -> None:
        async with self._lock:
            status = self._statuses.get(feed_name, FeedStatus(name=feed_name))
            status.consecutive_failures += 1
            status.last_failure = datetime.now(timezone.utc)
            status.status = "degraded" if status.consecutive_failures < self._alert_threshold else "down"
            status.message = error
            trigger_alert = status.consecutive_failures >= self._alert_threshold and not status.alert_active
            status.alert_active = status.consecutive_failures >= self._alert_threshold
            self._statuses[feed_name] = status
        if trigger_alert:
            await self._dispatch(
                feed_name,
                severity="critical",
                message=error,
                extra={"consecutive_failures": status.consecutive_failures},
            )

    async def report_skip(self, feed_name: str, reason: str) -> None:
        async with self._lock:
            status = self._statuses.get(feed_name, FeedStatus(name=feed_name))
            status.status = "paused"
            status.message = reason
            self._statuses[feed_name] = status

    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            return {name: status.to_dict() for name, status in self._statuses.items()}

    async def _dispatch(self, feed_name: str, severity: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "service": self._service_name,
            "feed": feed_name,
            "severity": severity,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            payload.update(extra)
        try:
            await self._dispatcher.send_alert(payload)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to send feed alert: %s", exc)


def build_default_monitor(service_name: str) -> FeedHealthMonitor:
    alert_url = os.getenv("EVENT_FEED_ALERT_WEBHOOK")
    alert_headers = None
    if alert_url:
        raw_headers = os.getenv("EVENT_FEED_ALERT_HEADERS")
        if raw_headers:
            try:
                alert_headers = json.loads(raw_headers)
            except json.JSONDecodeError:
                logger.warning("Invalid EVENT_FEED_ALERT_HEADERS value; ignoring")
    dispatcher: Optional[AlertDispatcher] = None
    if alert_url:
        dispatcher = WebhookAlertDispatcher(alert_url, headers=alert_headers)
    threshold = int(os.getenv("EVENT_FEED_ALERT_THRESHOLD", "3"))
    return FeedHealthMonitor(service_name=service_name, alert_threshold=threshold, dispatcher=dispatcher)
