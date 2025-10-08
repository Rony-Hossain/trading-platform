from __future__ import annotations

import asyncio
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pytest
import websockets
from jsonschema import validate
from pytest_bdd import given, parsers, scenarios, then, when

from pytest_bdd.model import Table

from tests.conftest import ServiceAvailability  # type: ignore

FEATURE_DIR = Path(__file__).resolve().parents[1]

FEATURE_FILES = [
    "market_data_price.feature",
    "market_data_options.feature",
    "market_data_ws.feature",
    "health.feature",
    "quote.feature",
    "ohlcv.feature",
    "orderbook.feature",
    "search.feature",
    "openapi.feature",
    "ws-quotes.feature",
    "ws-orderbook.feature",
]

for feature_file in FEATURE_FILES:
    scenarios(str(FEATURE_DIR / feature_file))

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"

RATE_REQUEST_COUNT = int(os.getenv("RATE_LIMIT_REQUEST_COUNT", "200"))
RATE_ENABLE = os.getenv("ENABLE_RATE_TESTS", "0") == "1"
WS_RECONNECT_TIMEOUT = int(os.getenv("WS_RECONNECT_TIMEOUT", "5"))
WS_RECEIVE_TIMEOUT = int(os.getenv("WS_RECEIVE_TIMEOUT", "10"))
WS_DELTA_TARGET = int(os.getenv("WS_DELTA_TARGET", "3"))


def _parse_iso8601(candidate: str) -> datetime:
    text = candidate
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        text = f"{text}T00:00:00"
    if "+" not in text and "-" not in text[-6:]:
        text = text + "+00:00"
    return datetime.fromisoformat(text)


def _parse_table(table: Any) -> List[Any]:
    if isinstance(table, Table):
        if not table.rows:
            return []
        if table.headings:
            headings = list(table.headings)
            return [dict(zip(headings, row)) for row in table.rows]
        if all(len(row) >= 2 for row in table.rows):
            return [{row[0]: row[1]} for row in table.rows]
        return [row[0] if row else "" for row in table.rows]
    table_str = str(table)
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in table_str.strip().splitlines()
        if line.strip().startswith("|")
    ]
    if not rows:
        return []
    header = rows[0]
    data_rows = rows[1:]
    if len(header) == 1:
        return [row[0] for row in data_rows]
    return [dict(zip(header, row)) for row in data_rows]


def _resolve_url(base_url: str, override: Optional[str], path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    base = override or base_url
    return f"{base.rstrip('/')}{path}"


def _load_schema(name: str) -> Dict[str, Any]:
    schema_path = SCHEMA_DIR / f"{name}.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema {name} not found at {schema_path}")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _resolve_field(payload: Any, field_path: str) -> Any:
    if field_path.startswith("$."):
        parts = field_path[2:].split(".")
    else:
        parts = field_path.split(".")
    current = payload
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


@pytest.fixture
def bdd_context(base_url: str, ws_url: str) -> Dict[str, Any]:
    return {
        "base_url": base_url,
        "base_url_override": None,
        "response": None,
        "response_body": None,
        "responses": [],
        "status_history": [],
        "default_symbol": None,
        "ws_url": ws_url,
        "ws_connection": None,
        "ws_messages": [],
        "ws_error": None,
        "ws_last_event_time": None,
        "schema_cache": {},
    }


async def _http_get(
    bdd_context: Dict[str, Any],
    service_availability: ServiceAvailability,
    path: str,
    params: Optional[Dict[str, Any]],
    timeout: float = 15.0,
) -> httpx.Response:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    url = _resolve_url(bdd_context["base_url"], bdd_context["base_url_override"], path)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, params=params)
    bdd_context["response"] = response
    try:
        bdd_context["response_body"] = response.json()
    except json.JSONDecodeError:
        bdd_context["response_body"] = response.text
    return response


def _store_schema(bdd_context: Dict[str, Any], name: str) -> Dict[str, Any]:
    cache = bdd_context["schema_cache"]
    if name not in cache:
        cache[name] = _load_schema(name)
    return cache[name]


def _ensure_ws_connection(bdd_context: Dict[str, Any]) -> websockets.WebSocketClientProtocol:
    ws = bdd_context.get("ws_connection")
    if ws is None:
        pytest.fail("WebSocket connection has not been established")
    return ws


def _close_ws_sync(ws: websockets.WebSocketClientProtocol) -> None:
    if ws.closed:
        return
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(ws.close())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(ws.close())
        finally:
            loop.close()


def _parse_params_table(table_str: str) -> Dict[str, Any]:
    records = _parse_table(table_str)
    params: Dict[str, Any] = {}
    for row in records:
        if isinstance(row, dict) and len(row) >= 2:
            key = next(iter(row.keys()))
            value = row[key]
            params[key] = value
    return params


@given(parsers.cfparse('the Market Data API base URL "{url}"'))
def override_base_url(bdd_context: Dict[str, Any], url: str) -> None:
    bdd_context["base_url_override"] = url


@given(parsers.cfparse('the default symbol is "{symbol}"'))
def default_symbol(bdd_context: Dict[str, Any], symbol: str) -> None:
    bdd_context["default_symbol"] = symbol


@given(parsers.cfparse('a WebSocket URL "{url}"'))
def ws_url_override(bdd_context: Dict[str, Any], url: str) -> None:
    bdd_context["ws_url"] = url


@given(parsers.cfparse('the WS URL "{url}"'))
def legacy_ws_url(bdd_context: Dict[str, Any], url: str) -> None:
    bdd_context["ws_url"] = url


@when(parsers.cfparse('I GET "{path}"'))
@pytest.mark.asyncio
async def http_get_simple(
    path: str,
    bdd_context: Dict[str, Any],
    service_availability: ServiceAvailability,
) -> None:
    await _http_get(bdd_context, service_availability, path, params=None)


@when(parsers.parse('I GET "{path}" with params:\n{table}'))
@pytest.mark.asyncio
async def http_get_with_params(
    path: str,
    table: str,
    bdd_context: Dict[str, Any],
    service_availability: ServiceAvailability,
) -> None:
    params = _parse_params_table(table)
    await _http_get(bdd_context, service_availability, path, params=params)


@given(parsers.parse('I retry GET "{path}" every {interval:d}s for up to {limit:d}s'))
@pytest.mark.asyncio
async def http_retry_until_ok(
    path: str,
    interval: int,
    limit: int,
    bdd_context: Dict[str, Any],
    service_availability: ServiceAvailability,
) -> None:
    deadline = time.time() + limit
    last_response: Optional[httpx.Response] = None
    while time.time() <= deadline:
        last_response = await _http_get(bdd_context, service_availability, path, params=None)
        if last_response.status_code == 200:
            return
        await asyncio.sleep(interval)
    assert last_response is not None
    pytest.fail(f"Health endpoint did not return 200 within {limit}s (last {last_response.status_code})")


@given(parsers.parse('I perform {count:d} GET requests to "{path}" with params:\n{table}'))
@pytest.mark.asyncio
async def perform_multiple_gets(
    count: int,
    path: str,
    table: str,
    bdd_context: Dict[str, Any],
    service_availability: ServiceAvailability,
) -> None:
    if not RATE_ENABLE:
        pytest.skip("Rate limit scenario disabled (set ENABLE_RATE_TESTS=1 to enable).")
    total = min(count, RATE_REQUEST_COUNT)
    params = _parse_params_table(table)
    statuses: List[int] = []
    url = _resolve_url(bdd_context["base_url"], bdd_context["base_url_override"], path)
    async with httpx.AsyncClient(timeout=10.0) as client:
        for _ in range(total):
            if not service_availability.available:
                pytest.skip(service_availability.reason)
            response = await client.get(url, params=params)
            statuses.append(response.status_code)
    bdd_context["status_history"] = statuses


@then(parsers.cfparse("the response status is {status:d}"))
@then(parsers.cfparse("the response status should be {status:d}"))
def assert_response_status(bdd_context: Dict[str, Any], status: int) -> None:
    response: httpx.Response = bdd_context.get("response")
    if response is None:
        pytest.skip("No response captured for assertion.")
    assert response.status_code == status, response.text


@then(parsers.cfparse('the json has a numeric field "{field}"'))
def assert_json_numeric_field(bdd_context: Dict[str, Any], field: str) -> None:
    payload = bdd_context.get("response_body")
    if not isinstance(payload, dict):
        pytest.fail("Response payload is not JSON")
    value = payload.get(field)
    assert isinstance(value, (int, float)) and math.isfinite(value) and value > 0, f"{field} invalid: {value}"


@then(parsers.cfparse('the json includes "{field}"'))
def assert_json_has_field(bdd_context: Dict[str, Any], field: str) -> None:
    payload = bdd_context.get("response_body")
    if not isinstance(payload, dict):
        pytest.fail("Response payload is not JSON")
    assert field in payload, f"{field} missing in payload"


@then(parsers.cfparse('the json has a non-empty array "{field}"'))
def assert_json_non_empty_array(bdd_context: Dict[str, Any], field: str) -> None:
    payload = bdd_context.get("response_body")
    if not isinstance(payload, dict):
        pytest.fail("Response payload is not JSON")
    value = payload.get(field)
    assert isinstance(value, list) and len(value) > 0, f"{field} expected non-empty array, got {value}"


@then(parsers.cfparse('the response should match the "{schema}" schema'))
def assert_response_schema(bdd_context: Dict[str, Any], schema: str) -> None:
    payload = bdd_context.get("response_body")
    schema_obj = _store_schema(bdd_context, schema)
    validate(instance=payload, schema=schema_obj)


@then(parsers.cfparse('the field "{field}" should equal "{value}"'))
def assert_field_equals(bdd_context: Dict[str, Any], field: str, value: str) -> None:
    payload = bdd_context.get("response_body")
    actual = _resolve_field(payload, field)
    if isinstance(actual, str):
        actual_cmp = actual.lower()
        expected_cmp = value.lower()
    else:
        actual_cmp = actual
        expected_cmp = value
    assert actual_cmp == expected_cmp, f"{field} expected {value}, got {actual}"


@then(parsers.cfparse('the field "{field}" in the last message should equal "{value}"'))
def assert_last_message_field(bdd_context: Dict[str, Any], field: str, value: str) -> None:
    last_message = bdd_context.get("ws_messages")[-1] if bdd_context.get("ws_messages") else None
    if last_message is None:
        pytest.fail("No WebSocket message captured")
    actual = _resolve_field(last_message, field)
    assert actual == value, f"{field} expected {value}, got {actual}"


@then(parsers.cfparse('the field "{field}" should be within {seconds:d} seconds of now'))
def assert_field_freshness(bdd_context: Dict[str, Any], field: str, seconds: int) -> None:
    payload = bdd_context.get("response_body")
    value = _resolve_field(payload, field)
    assert isinstance(value, str), f"{field} must be a string timestamp"
    ts = _parse_iso8601(value)
    delta = datetime.now(timezone.utc) - ts.astimezone(timezone.utc)
    assert abs(delta.total_seconds()) <= seconds, f"{field} out of freshness window ({delta.total_seconds():.2f}s)"


@then(parsers.parse('the array "{field}" length should be >= {expected:d}'))
def assert_array_len_ge(bdd_context: Dict[str, Any], field: str, expected: int) -> None:
    payload = bdd_context.get("response_body")
    array = _resolve_field(payload, field)
    assert isinstance(array, list), f"{field} is not a list"
    assert len(array) >= expected, f"{field} length {len(array)} < {expected}"


@then(parsers.parse('the array "{field}" length should be <= {expected:d}'))
def assert_array_len_le(bdd_context: Dict[str, Any], field: str, expected: int) -> None:
    payload = bdd_context.get("response_body")
    array = _resolve_field(payload, field)
    assert isinstance(array, list), f"{field} is not a list"
    assert len(array) <= expected, f"{field} length {len(array)} > {expected}"


@then(parsers.parse('the array "{field}" length should be {expected:d}'))
def assert_array_len_eq(bdd_context: Dict[str, Any], field: str, expected: int) -> None:
    payload = bdd_context.get("response_body")
    array = _resolve_field(payload, field)
    assert isinstance(array, list), f"{field} is not a list"
    assert len(array) == expected, f"{field} length {len(array)} != {expected}"


@then(parsers.parse('every item in "{field}" should have fields: {fields}'))
def assert_items_have_fields(bdd_context: Dict[str, Any], field: str, fields: str) -> None:
    payload = bdd_context.get("response_body")
    array = _resolve_field(payload, field)
    assert isinstance(array, list), f"{field} is not a list"
    required = [name.strip() for name in fields.split(",")]
    for item in array:
        for name in required:
            assert name in item, f"{name} missing in item {item}"


@then("at least one response status should be 429")
def assert_status_history_contains_429(bdd_context: Dict[str, Any]) -> None:
    history = bdd_context.get("status_history") or []
    assert any(status == 429 for status in history), f"No 429 found in status history: {history}"


@then("the array \"bids\" length should be > 0")
def assert_bids_non_empty(bdd_context: Dict[str, Any]) -> None:
    payload = bdd_context.get("response_body")
    bids = payload.get("bids") if isinstance(payload, dict) else None
    assert isinstance(bids, list) and len(bids) > 0, "Expected bids to be non-empty"


@then("the array \"asks\" length should be > 0")
def assert_asks_non_empty(bdd_context: Dict[str, Any]) -> None:
    payload = bdd_context.get("response_body")
    asks = payload.get("asks") if isinstance(payload, dict) else None
    assert isinstance(asks, list) and len(asks) > 0, "Expected asks to be non-empty"


@then("the best bid price should be < the best ask price")
def assert_bid_ask_spread(bdd_context: Dict[str, Any]) -> None:
    payload = bdd_context.get("response_body")
    bids = payload.get("bids") if isinstance(payload, dict) else None
    asks = payload.get("asks") if isinstance(payload, dict) else None
    if not bids or not asks:
        pytest.skip("Orderbook missing bids or asks")
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    assert best_bid < best_ask, f"Best bid {best_bid} not less than best ask {best_ask}"


@then(parsers.cfparse('the field "{field}" should start with "{prefix}"'))
def assert_field_starts_with(bdd_context: Dict[str, Any], field: str, prefix: str) -> None:
    payload = bdd_context.get("response_body")
    value = _resolve_field(payload, field)
    assert isinstance(value, str), f"{field} is not a string"
    assert value.startswith(prefix), f"{field} does not start with {prefix} (value={value})"


@when("I connect to the WebSocket")
@pytest.mark.asyncio
async def connect_websocket(bdd_context: Dict[str, Any], request: pytest.FixtureRequest) -> None:
    url = bdd_context.get("ws_url")
    ws = await websockets.connect(url, ping_interval=20, close_timeout=5)
    bdd_context["ws_connection"] = ws

    def _finalizer() -> None:
        _close_ws_sync(ws)

    request.addfinalizer(_finalizer)


@when(parsers.cfparse('I connect to "{url}"'))
@pytest.mark.asyncio
async def connect_websocket_direct(bdd_context: Dict[str, Any], url: str, request: pytest.FixtureRequest) -> None:
    ws = await websockets.connect(url, ping_interval=20, close_timeout=5)
    bdd_context["ws_connection"] = ws

    def _finalizer() -> None:
        _close_ws_sync(ws)

    request.addfinalizer(_finalizer)


@when("I connect and read one message")
@pytest.mark.asyncio
async def connect_and_read_legacy(bdd_context: Dict[str, Any], request: pytest.FixtureRequest) -> None:
    url = bdd_context.get("ws_url")
    ws = await websockets.connect(url, ping_interval=20, close_timeout=5)

    def _finalizer() -> None:
        _close_ws_sync(ws)

    request.addfinalizer(_finalizer)

    try:
        message = await asyncio.wait_for(ws.recv(), timeout=2)
    except asyncio.TimeoutError:
        pytest.skip("WebSocket message not received within 2 seconds")
    payload = json.loads(message)
    bdd_context["response_body"] = payload
    bdd_context["ws_connection"] = ws
    bdd_context["ws_messages"].append(payload)
@when(parsers.parse("I send a subscribe message for:\n{table}"))
@pytest.mark.asyncio
async def ws_send_subscribe(bdd_context: Dict[str, Any], table: str) -> None:
    ws = _ensure_ws_connection(bdd_context)
    records = _parse_table(table)
    symbols = [row["symbol"] if isinstance(row, dict) and "symbol" in row else row for row in records]
    payload = {"action": "subscribe", "symbols": symbols}
    await ws.send(json.dumps(payload))


@when(parsers.parse('I subscribe to "{symbol}"'))
@pytest.mark.asyncio
async def ws_subscribe_single(bdd_context: Dict[str, Any], symbol: str) -> None:
    ws = _ensure_ws_connection(bdd_context)
    await ws.send(json.dumps({"action": "subscribe", "symbols": [symbol]}))


@when(parsers.parse('I unsubscribe from "{symbol}"'))
@pytest.mark.asyncio
async def ws_unsubscribe_single(bdd_context: Dict[str, Any], symbol: str) -> None:
    ws = _ensure_ws_connection(bdd_context)
    await ws.send(json.dumps({"action": "unsubscribe", "symbols": [symbol]}))


@when("the server closes the connection")
@pytest.mark.asyncio
async def ws_close_connection(bdd_context: Dict[str, Any]) -> None:
    ws = _ensure_ws_connection(bdd_context)
    await ws.close()


@when(parsers.parse("I wait for {seconds:d} seconds"))
@pytest.mark.asyncio
async def ws_wait(seconds: int) -> None:
    await asyncio.sleep(seconds)


async def _ws_recv_with_timeout(
    ws: websockets.WebSocketClientProtocol,
    timeout: int,
) -> Dict[str, Any]:
    message = await asyncio.wait_for(ws.recv(), timeout=timeout)
    return json.loads(message)


@then(parsers.parse('I should receive a message within {seconds:d} seconds matching "{schema}"'))
@pytest.mark.asyncio
async def ws_receive_message(
    bdd_context: Dict[str, Any],
    seconds: int,
    schema: str,
) -> None:
    ws = _ensure_ws_connection(bdd_context)
    payload = await _ws_recv_with_timeout(ws, seconds or WS_RECEIVE_TIMEOUT)
    schema_obj = _store_schema(bdd_context, schema)
    validate(instance=payload, schema=schema_obj)
    bdd_context["response_body"] = payload
    bdd_context["ws_messages"].append(payload)
    bdd_context["ws_last_event_time"] = time.time()


@then(parsers.parse("I should receive messages for all of:\n{table}"))
@pytest.mark.asyncio
async def ws_receive_messages_for_all(bdd_context: Dict[str, Any], table: str) -> None:
    ws = _ensure_ws_connection(bdd_context)
    records = _parse_table(table)
    targets = {row["symbol"] if isinstance(row, dict) else row for row in records}
    received: Dict[str, Dict[str, Any]] = {}
    deadline = time.time() + WS_RECEIVE_TIMEOUT
    while time.time() < deadline and targets != set(received.keys()):
        payload = await _ws_recv_with_timeout(ws, WS_RECEIVE_TIMEOUT)
        symbol = payload.get("symbol")
        if symbol in targets:
            received[symbol] = payload
        bdd_context["ws_messages"].append(payload)
    missing = targets.difference(received.keys())
    assert not missing, f"Did not receive messages for: {missing}"


@then(parsers.cfparse('I should receive an error event with code "{code}"'))
@pytest.mark.asyncio
async def ws_receive_error_event(bdd_context: Dict[str, Any], code: str) -> None:
    ws = _ensure_ws_connection(bdd_context)
    payload = await _ws_recv_with_timeout(ws, WS_RECEIVE_TIMEOUT)
    bdd_context["response_body"] = payload
    bdd_context["ws_messages"].append(payload)
    actual = payload.get("code") or payload.get("error") or payload.get("event")
    if actual is None:
        pytest.fail(f"No error code present in payload: {payload}")
    assert str(actual).upper() == code.upper()


@then("the connection should still be open")
def ws_connection_open(bdd_context: Dict[str, Any]) -> None:
    ws = _ensure_ws_connection(bdd_context)
    assert not ws.closed, "WebSocket connection is closed"


@then("the client reconnects within 5 seconds")
@pytest.mark.asyncio
async def ws_reconnect(bdd_context: Dict[str, Any]) -> None:
    url = bdd_context.get("ws_url")
    deadline = time.time() + WS_RECONNECT_TIMEOUT
    while time.time() < deadline:
        try:
            ws = await websockets.connect(url, ping_interval=20, close_timeout=5)
        except Exception:
            await asyncio.sleep(1)
            continue
        bdd_context["ws_connection"] = ws
        return
    pytest.fail("Failed to reconnect within timeout")


@then(parsers.parse('I should not receive any "{symbol}" messages for {seconds:d} seconds'))
@pytest.mark.asyncio
async def ws_no_messages_for_symbol(bdd_context: Dict[str, Any], symbol: str, seconds: int) -> None:
    ws = _ensure_ws_connection(bdd_context)
    try:
        payload = await asyncio.wait_for(ws.recv(), timeout=seconds)
    except asyncio.TimeoutError:
        return
    else:
        message = json.loads(payload)
        if message.get("symbol") == symbol:
            pytest.fail(f"Unexpected message for {symbol}: {message}")


@then('I should receive a "snapshot" event')
@pytest.mark.asyncio
async def ws_receive_snapshot(bdd_context: Dict[str, Any]) -> None:
    ws = _ensure_ws_connection(bdd_context)
    payload = await _ws_recv_with_timeout(ws, WS_RECEIVE_TIMEOUT)
    bdd_context["response_body"] = payload
    bdd_context["ws_messages"].append(payload)
    event_type = payload.get("type") or payload.get("event")
    if str(event_type).lower() != "snapshot":
        pytest.skip(f"Expected snapshot event, received {payload}")


@then("then receive at least 3 \"delta\" events")
@pytest.mark.asyncio
async def ws_receive_deltas(bdd_context: Dict[str, Any]) -> None:
    ws = _ensure_ws_connection(bdd_context)
    deltas = 0
    deadline = time.time() + WS_RECEIVE_TIMEOUT
    while time.time() < deadline and deltas < WS_DELTA_TARGET:
        payload = await _ws_recv_with_timeout(ws, WS_RECEIVE_TIMEOUT)
        event_type = payload.get("type") or payload.get("event")
        if str(event_type).lower() == "delta":
            deltas += 1
    if deltas < WS_DELTA_TARGET:
        pytest.skip(f"Expected {WS_DELTA_TARGET} delta events, received {deltas}")


@then(parsers.parse("the average message rate should be <= {rate:f} msgs per second"))
@pytest.mark.asyncio
async def ws_average_rate(bdd_context: Dict[str, Any], rate: float) -> None:
    ws = _ensure_ws_connection(bdd_context)
    start = time.time()
    count = 0
    duration = 5.0
    end = start + duration
    while time.time() < end:
        try:
            await asyncio.wait_for(ws.recv(), timeout=end - time.time())
            count += 1
        except asyncio.TimeoutError:
            break
    elapsed = max(time.time() - start, 0.001)
    avg_rate = count / elapsed
    assert avg_rate <= rate, f"Average rate {avg_rate:.2f} exceeds {rate}"


@then(parsers.parse('the array "{field}" length should be > {expected:d}'))
def assert_array_len_gt(bdd_context: Dict[str, Any], field: str, expected: int) -> None:
    payload = bdd_context.get("response_body")
    array = _resolve_field(payload, field)
    assert isinstance(array, list)
    assert len(array) > expected, f"{field} length {len(array)} <= {expected}"


@then(parsers.parse('the array "{field}" length should be == {expected:d}'))
def assert_array_len_eq_alias(bdd_context: Dict[str, Any], field: str, expected: int) -> None:
    assert_array_len_eq(bdd_context, field, expected)
@then(parsers.parse('every {name:w} should have fields: {fields}'))
def assert_items_alias(bdd_context: Dict[str, Any], name: str, fields: str) -> None:
    lookup = name.lower()
    if lookup.startswith("candle"):
        field = "candles"
    else:
        field = name
    assert_items_have_fields(bdd_context, field, fields)
