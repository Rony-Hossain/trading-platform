from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


async def test_coverage_flags_present(api_client, service_availability) -> None:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    response = await api_client.get("/stocks/AAPL/coverage")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict), payload
    expected = {"price", "history", "profile", "options"}
    missing = expected.difference(payload.keys())
    assert not missing, f"Coverage missing keys: {missing}"
    for key in expected:
        value = payload.get(key)
        assert isinstance(value, bool), f"{key} should be boolean (got {value})"
