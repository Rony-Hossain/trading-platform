from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


async def test_admin_stats_exposes_counters(api_client, service_availability, optional_endpoint_guard) -> None:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    response = await api_client.get("/admin/stats")
    optional_endpoint_guard(response, "/admin/stats")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict), payload
    expected_keys = {"cache", "providers", "websocket", "requests"}
    assert expected_keys.intersection(payload.keys()), f"Missing expected counters: {payload}"
    cache_info = payload.get("cache", {})
    if isinstance(cache_info, dict):
        hits = cache_info.get("hits")
        misses = cache_info.get("misses")
        assert hits is None or hits >= 0
        assert misses is None or misses >= 0
