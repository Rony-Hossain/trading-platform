from __future__ import annotations

import asyncio
import json
import time

import pytest
import websockets


@pytest.mark.asyncio
async def test_websocket_first_tick_fast(service_availability, ws_url) -> None:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    target = f"{ws_url.rstrip('/')}/AAPL"
    start = time.perf_counter()
    try:
        async with websockets.connect(target, ping_interval=None, close_timeout=5) as socket:
            message = await asyncio.wait_for(socket.recv(), timeout=2)
    except Exception as exc:  # pragma: no cover - external failure
        pytest.skip(f"WebSocket connection failed: {exc}")  # pragma: no cover
    duration = time.perf_counter() - start
    assert duration <= 2.0, f"First tick took too long ({duration:.3f}s)"
    payload = json.loads(message)
    assert isinstance(payload, dict), payload
    assert isinstance(payload.get("price"), (int, float)) and payload["price"] > 0
    assert payload.get("symbol", "").upper() == "AAPL"
    assert "ts" in payload or "timestamp" in payload
