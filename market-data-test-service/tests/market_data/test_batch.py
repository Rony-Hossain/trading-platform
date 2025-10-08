from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.asyncio


async def test_batch_returns_data_per_symbol(api_client, service_availability, tracked_symbols) -> None:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    symbols = [tracked_symbols["liquid"][0], tracked_symbols["invalid"]]
    response = await api_client.post("/stocks/batch", json=symbols)
    assert response.status_code == 200, response.text
    payload = response.json()
    if isinstance(payload, list):
        by_symbol = {item.get("symbol"): item for item in payload if isinstance(item, dict)}
    elif isinstance(payload, dict):
        primary_key = next((key for key in ("results", "data", "quotes") if key in payload), None)
        if primary_key:
            container = payload[primary_key]
            by_symbol = {
                item.get("symbol"): item
                for item in container
                if isinstance(item, dict) and item.get("symbol")
            }
        else:
            by_symbol = payload
    else:
        pytest.fail(f"Unexpected payload type from batch endpoint: {payload!r}")

    for symbol in symbols:
        assert symbol in by_symbol, f"{symbol} missing in batch payload ({payload})"

    valid_entry = by_symbol[symbols[0]]
    price = (
        valid_entry.get("price")
        or valid_entry.get("last")
        or valid_entry.get("value")
        or valid_entry.get("data", {}).get("price")
    )
    assert isinstance(price, (int, float)) and not math.isnan(price) and price > 0, valid_entry
    invalid_entry = by_symbol[symbols[1]]
    assert any(
        key in invalid_entry for key in ("error", "detail", "status", "code")
    ), f"Expected error metadata for invalid symbol: {invalid_entry}"


async def test_batch_rejects_invalid_payload(api_client, service_availability) -> None:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    response = await api_client.post("/stocks/batch", json={"symbols": "AAPL"})
    assert response.status_code in {400, 422}, response.text
