from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


async def test_health_endpoint_reports_ok(api_client, service_availability) -> None:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    response = await api_client.get("/health")
    assert response.status_code == 200, response.text
    try:
        payload = response.json()
    except ValueError:
        pytest.fail(f"Health endpoint returned non-JSON payload: {response.text}")
    status = payload.get("status") or payload.get("state") or payload.get("service")
    assert status, f"Unexpected health payload: {payload}"
    assert isinstance(status, str)
