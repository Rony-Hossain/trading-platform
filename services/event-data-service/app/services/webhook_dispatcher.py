"""Webhook dispatch utilities for event notifications."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class WebhookTarget:
    url: str
    headers: Optional[Dict[str, str]] = None
    timeout: float = 5.0
    name: Optional[str] = None


class EventWebhookDispatcher:
    """Dispatch event notifications to configured webhook targets."""

    def __init__(self, targets: Iterable[WebhookTarget]) -> None:
        self._targets: List[WebhookTarget] = list(targets)

    def has_targets(self) -> bool:
        return bool(self._targets)

    async def dispatch(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self._targets:
            return
        body = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": _sanitize(payload),
        }
        await asyncio.gather(*(self._send(target, body) for target in self._targets), return_exceptions=True)

    async def _send(self, target: WebhookTarget, body: Dict[str, Any]) -> None:
        headers = {"Content-Type": "application/json"}
        if target.headers:
            headers.update(target.headers)
        try:
            async with httpx.AsyncClient(timeout=target.timeout) as client:
                await client.post(target.url, json=body, headers=headers)
        except Exception as exc:  # noqa: BLE001
            label = target.name or target.url
            logger.warning("Webhook dispatch to %s failed: %s", label, exc)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def build_webhook_dispatcher(service_name: str) -> EventWebhookDispatcher:
    targets: List[WebhookTarget] = []
    # JSON configuration takes precedence
    json_config = os.getenv("EVENT_WEBHOOK_TARGETS")
    if json_config:
        try:
            data = json.loads(json_config)
            if isinstance(data, list):
                for entry in data:
                    url = entry.get("url") if isinstance(entry, dict) else None
                    if not url:
                        continue
                    headers = entry.get("headers") if isinstance(entry, dict) else None
                    timeout = float(entry.get("timeout", 5.0)) if isinstance(entry, dict) else 5.0
                    name = entry.get("name") if isinstance(entry, dict) else None
                    targets.append(WebhookTarget(url=url, headers=headers, timeout=timeout, name=name))
        except json.JSONDecodeError:
            logger.warning("Invalid EVENT_WEBHOOK_TARGETS JSON value; ignoring")
    if not targets:
        url = os.getenv("EVENT_WEBHOOK_URL")
        if url:
            headers_env = os.getenv("EVENT_WEBHOOK_HEADERS")
            headers: Optional[Dict[str, str]] = None
            if headers_env:
                try:
                    headers = json.loads(headers_env)
                except json.JSONDecodeError:
                    logger.warning("Invalid EVENT_WEBHOOK_HEADERS value; ignoring")
            timeout = float(os.getenv("EVENT_WEBHOOK_TIMEOUT", "5"))
            targets.append(WebhookTarget(url=url, headers=headers, timeout=timeout, name=service_name))
    return EventWebhookDispatcher(targets)
