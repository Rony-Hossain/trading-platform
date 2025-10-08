from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.asyncio


async def test_options_chain_has_expiries_and_greeks(
    api_client, service_availability, tracked_symbols
) -> None:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    response = await api_client.get(f"/options/{tracked_symbols['liquid'][0]}/chain")
    assert response.status_code == 200, response.text
    payload = response.json()
    expiries = payload.get("expiries") or payload.get("expirationDates")
    assert isinstance(expiries, list) and expiries, f"Missing expiries: {payload}"
    chain = payload.get("chain") or payload.get("options")
    assert isinstance(chain, list) and chain, f"Missing options chain: {payload}"
    sample = chain[0]
    greeks = sample.get("greeks") or sample.get("metrics") or {}
    delta = greeks.get("delta")
    theta = greeks.get("theta")
    assert isinstance(delta, (int, float)) and not math.isnan(delta)
    assert isinstance(theta, (int, float)) and not math.isnan(theta)
