from __future__ import annotations

import math
from datetime import datetime

import pytest

pytestmark = pytest.mark.asyncio


async def test_history_daily_range(
    api_client, service_availability, tracked_symbols, iso_parser
) -> None:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    params = {"interval": "1d", "range": "1y"}
    response = await api_client.get(f"/stocks/{tracked_symbols['liquid'][0]}/history", params=params)
    assert response.status_code == 200, response.text
    payload = response.json()
    if isinstance(payload, dict):
        candles = (
            payload.get("candles")
            or payload.get("data")
            or payload.get("results")
            or payload.get("history")
        )
    else:
        candles = payload
    assert isinstance(candles, list) and candles, f"Unexpected payload: {payload}"
    assert len(candles) >= 200, f"Expected >=200 candles, got {len(candles)}"
    timestamps = [iso_parser(candle.get("ts") or candle.get("timestamp")) for candle in candles]
    assert timestamps == sorted(timestamps), "Timestamps must be monotonically increasing"
    for candle in candles:
        for field in ("open", "high", "low", "close"):
            value = candle.get(field)
            assert isinstance(value, (int, float)), f"{field} missing or invalid: {candle}"
            assert value > 0 and not math.isnan(value), f"{field} invalid: {value}"
        volume = candle.get("volume")
        assert volume is None or (isinstance(volume, (int, float)) and volume >= 0)
        assert candle["low"] <= candle["open"] <= candle["high"]
        assert candle["low"] <= candle["close"] <= candle["high"]


async def test_history_invalid_interval_returns_422(api_client, service_availability) -> None:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    response = await api_client.get("/stocks/AAPL/history", params={"interval": "99h", "range": "5d"})
    assert response.status_code in {400, 422}, response.text


async def test_history_latest_candle_not_future(api_client, service_availability, iso_parser) -> None:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    response = await api_client.get("/stocks/MSFT/history", params={"interval": "1d", "range": "5d"})
    assert response.status_code == 200, response.text
    payload = response.json()
    candles = payload.get("candles") if isinstance(payload, dict) else payload
    assert candles, f"No candles returned: {payload}"
    last_ts = iso_parser(candles[-1].get("ts") or candles[-1].get("timestamp"))
    assert last_ts <= datetime.now(last_ts.tzinfo or None), f"Last candle is in the future: {last_ts}"
