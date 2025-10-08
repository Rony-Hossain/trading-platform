from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("symbol", ["AAPL", "MSFT", "INTC"])
async def test_latest_price_positive_value(api_client, service_availability, symbol: str) -> None:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    response = await api_client.get(f"/stocks/{symbol}/price")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload.get("symbol", "").upper() == symbol
    price = payload.get("price") or payload.get("last") or payload.get("value")
    assert isinstance(price, (int, float)) and price > 0 and not math.isnan(price), payload
    assert "asOf" in payload or "timestamp" in payload


async def test_latest_price_invalid_symbol_returns_400(
    api_client, service_availability, tracked_symbols
) -> None:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    response = await api_client.get(f"/stocks/{tracked_symbols['bad_format']}/price")
    assert response.status_code == 400, response.text


async def test_latest_price_unknown_symbol_returns_404(
    api_client, service_availability, tracked_symbols
) -> None:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    response = await api_client.get(f"/stocks/{tracked_symbols['invalid']}/price")
    assert response.status_code == 404, response.text
    payload = response.json()
    assert any(key in payload for key in ("detail", "error", "message")), payload
