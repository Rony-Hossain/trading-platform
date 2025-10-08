from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


async def test_profile_contains_core_fields(
    api_client, service_availability, tracked_symbols
) -> None:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    response = await api_client.get(f"/stocks/{tracked_symbols['liquid'][0]}/profile")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict), payload
    required_fields = {"name", "exchange"}
    missing = {field for field in required_fields if not payload.get(field)}
    assert not missing, f"Profile missing fields: {missing} (payload: {payload})"
    sector = payload.get("sector") or payload.get("industry")
    if not sector:
        pytest.skip("Profile missing sector/industry metadata.")
    assert payload.get("symbol", "").upper() == tracked_symbols["liquid"][0]
    if "country" in payload:
        assert isinstance(payload["country"], str) and payload["country"]
