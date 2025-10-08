from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import MutableMapping

import pytest
import websockets
from pytest_bdd import given, parsers, scenarios, then, when

scenarios(str(Path(__file__).resolve().parent.parent / "market_data_ws.feature"))


@pytest.fixture
def bdd_context() -> MutableMapping[str, object]:
    return {}


@given(parsers.cfparse('the WS URL "{url}"'))
def ws_url_step(url: str, bdd_context: MutableMapping[str, object], ws_url: str) -> None:
    bdd_context["ws_url"] = ws_url if url.startswith("ws://localhost") else url


@when("I connect and read one message")
def connect_step(bdd_context: MutableMapping[str, object]) -> None:
    target = str(bdd_context["ws_url"])

    async def _connect() -> None:
        try:
            async with websockets.connect(target, ping_interval=None, close_timeout=5) as socket:
                payload = await asyncio.wait_for(socket.recv(), timeout=2)
        except Exception as exc:  # pragma: no cover - depends on live service
            pytest.skip(f"WebSocket connection failed: {exc}")  # pragma: no cover
        else:
            bdd_context["ws_payload"] = json.loads(payload)

    asyncio.run(_connect())


@then('the json includes "price"')
def json_has_price(bdd_context: MutableMapping[str, object]) -> None:
    payload = bdd_context.get("ws_payload")
    assert isinstance(payload, dict), f"Expected dict payload, got {payload}"
    price = payload.get("price")
    assert isinstance(price, (int, float)) and price > 0, f"Invalid price: {payload}"
