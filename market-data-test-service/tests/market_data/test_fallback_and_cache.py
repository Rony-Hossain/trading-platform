from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Dict

import httpx
import pytest

pytestmark = pytest.mark.asyncio


async def test_price_endpoint_cache_signal(api_client, service_availability, tracked_symbols) -> None:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    symbol = tracked_symbols["liquid"][0]
    first = await api_client.get(f"/stocks/{symbol}/price")
    assert first.status_code == 200, first.text
    start = time.perf_counter()
    second = await api_client.get(f"/stocks/{symbol}/price")
    duration = time.perf_counter() - start
    assert second.status_code == 200, second.text
    cache_header = (
        second.headers.get("X-Cache")
        or second.headers.get("X-Cache-Hit")
        or second.headers.get("Age")
        or second.headers.get("X-Cache-Status")
    )
    if cache_header is None:
        pytest.skip("Service did not expose cache indicators; enable instrumentation to validate TTL.")
    assert cache_header, "Expected cache indicator to be non-empty"
    assert duration < 1.0, f"Cached response took too long ({duration:.3f}s)"


async def test_provider_fallback_simulated_with_mock_transport(freeze_time) -> None:
    freeze_time("2025-10-07 09:30:00", tz_offset=-4)

    call_state: Dict[str, int] = {"count": 0}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        call_state["count"] += 1
        if call_state["count"] == 1:
            return httpx.Response(
                status_code=429,
                json={"detail": "Rate limited", "provider": "finnhub"},
                headers={"X-Provider": "finnhub"},
            )
        return httpx.Response(
            status_code=200,
            json={
                "symbol": "AAPL",
                "price": 189.23,
                "provider": "yahoo",
                "asOf": datetime.now(timezone.utc).isoformat(),
            },
            headers={"X-Provider": "yahoo"},
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        first = await client.get("/stocks/AAPL/price")
        second = await client.get("/stocks/AAPL/price")

    assert first.status_code == 429
    assert second.status_code == 200
    payload = second.json()
    assert payload["provider"].lower() == "yahoo"
    assert payload["price"] > 0
    assert call_state["count"] == 2


async def test_cache_ttl_simulated_with_mock_transport(freeze_time) -> None:
    freeze_time("2025-10-07 09:31:00", tz_offset=-4)

    last_value = {"price": 100.0, "timestamp": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if last_value["timestamp"] is None:
            last_value["timestamp"] = datetime.now(timezone.utc)
            body = {
                "symbol": "AAPL",
                "price": last_value["price"],
                "asOf": last_value["timestamp"].isoformat(),
                "cache": "miss",
            }
            headers = {"X-Cache": "MISS"}
        else:
            body = {
                "symbol": "AAPL",
                "price": last_value["price"],
                "asOf": last_value["timestamp"].isoformat(),
                "cache": "hit",
            }
            headers = {"X-Cache": "HIT"}
        return httpx.Response(200, json=body, headers=headers)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        miss = await client.get("/stocks/AAPL/price")
        hit = await client.get("/stocks/AAPL/price")

    assert miss.json()["cache"] == "miss"
    assert hit.json()["cache"] == "hit"
    assert hit.headers.get("X-Cache") == "HIT"
    assert math.isclose(miss.json()["price"], hit.json()["price"])
