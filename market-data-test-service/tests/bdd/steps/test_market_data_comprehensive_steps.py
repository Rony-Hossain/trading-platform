from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
import requests
import websockets
from jsonschema import validate as jsonschema_validate
from pytest_bdd import given, parsers, scenarios, then, when

# ---------- Feature wiring ----------
FEATURE_DIR = Path(__file__).resolve().parents[1]
scenarios(str(FEATURE_DIR / "quote.feature"))
scenarios(str(FEATURE_DIR / "ohlcv.feature"))
scenarios(str(FEATURE_DIR / "orderbook.feature"))
scenarios(str(FEATURE_DIR / "ws-quotes.feature"))


# ---------- Helpers ----------
def _base_url() -> str:
    return os.getenv("BASE_URL") or os.getenv("MARKET_DATA_URL") or "http://localhost:8002"


def _ws_url_default() -> str:
    base = _base_url().rstrip("/")
    return os.getenv("WS_URL") or base.replace("http", "ws") + "/ws/quotes"


def _load_schema(name: str) -> Dict[str, Any]:
    schema_path = FEATURE_DIR.parent / "schemas" / f"{name}.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _parse_iso(ts: str) -> float | None:
    try:
        from httpx._models import _parse_date

        return _parse_date(ts).timestamp()
    except Exception:
        from datetime import datetime

        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except Exception:
            return None


def _datatable_to_kv(datatable: List[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in datatable:
        if not row or len(row) < 2:
            continue
        key, value = row[0].strip(), row[1].strip()
        if key:
            out[key] = value
    return out


def _datatable_first_col(datatable: List[List[str]]) -> List[str]:
    values: List[str] = []
    for row in datatable:
        if not row:
            continue
        symbol = str(row[0]).strip()
        if symbol:
            values.append(symbol)
    return values


def _maybe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


def _build_url(bdd_context: Dict[str, Any], path: str) -> str:
    base = (bdd_context["base_url"] or "").rstrip("/")
    base_path = bdd_context.get("base_path") or ""
    if base_path:
        base += base_path
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def _run_async(bdd_context: Dict[str, Any], coro):
    loop: asyncio.AbstractEventLoop = bdd_context["loop"]
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        bdd_context["loop"] = loop
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ---------- Scenario context ----------
@pytest.fixture
def bdd_context():
    return {
        "base_url": _base_url(),
        "ws_url": _ws_url_default(),
        "base_path": "",
        "default_symbol": None,
        "resp": None,
        "json": None,
        "ws": None,
        "last_msg": None,
        "loop": asyncio.new_event_loop(),
    }


# ---------- GIVEN ----------
@given(parsers.parse('the API base path is "{base}"'))
def set_api_base_path(base: str, bdd_context):
    normalized = "/" + base.strip("/") if base.strip("/") else ""
    bdd_context["base_path"] = normalized


@given(parsers.parse('the default symbol is "{symbol}"'))
def set_default_symbol(symbol: str, bdd_context):
    bdd_context["default_symbol"] = symbol


@given(parsers.parse('a WebSocket URL "{ws}"'))
def set_ws_url(ws: str, bdd_context):
    bdd_context["ws_url"] = ws


# ---------- WHEN (HTTP) ----------
@when(parsers.parse('I GET "{path}"'))
def http_get_no_params(path: str, bdd_context):
    url = _build_url(bdd_context, path)
    try:
        resp = requests.get(url, timeout=10)
    except requests.RequestException as exc:
        pytest.fail(f"HTTP GET failed: {exc}\nurl={url}")
    bdd_context["resp"] = resp
    bdd_context["json"] = _maybe_json(resp)


@when(parsers.parse('I GET "{path}" with params:'))
def http_get_with_params(path: str, datatable: List[List[str]], bdd_context):
    params = _datatable_to_kv(datatable)
    url = _build_url(bdd_context, path)
    try:
        resp = requests.get(url, params=params, timeout=10)
    except requests.RequestException as exc:
        pytest.fail(f"HTTP GET failed: {exc}\nurl={url} params={params}")
    bdd_context["resp"] = resp
    bdd_context["json"] = _maybe_json(resp)


# ---------- WHEN (WebSocket) ----------
@when("I connect to the WebSocket")
def ws_connect(bdd_context):
    ws = _run_async(bdd_context, websockets.connect(bdd_context["ws_url"], ping_interval=15))
    bdd_context["ws"] = ws


@when(parsers.parse("I subscribe to {symbol}"))
def ws_subscribe_one(symbol: str, bdd_context):
    ws = bdd_context["ws"]
    assert ws is not None, "WS not connected"
    payload = json.dumps({"action": "subscribe", "symbols": [symbol.strip('"')]})
    _run_async(bdd_context, ws.send(payload))


@when(parsers.parse("I unsubscribe from {symbol}"))
def ws_unsubscribe_one(symbol: str, bdd_context):
    ws = bdd_context["ws"]
    assert ws is not None, "WS not connected"
    payload = json.dumps({"action": "unsubscribe", "symbols": [symbol.strip('"')]})
    _run_async(bdd_context, ws.send(payload))


@when("I send a subscribe message for:")
def ws_subscribe_table(datatable: List[List[str]], bdd_context):
    ws = bdd_context["ws"]
    assert ws is not None, "WS not connected"
    symbols = _datatable_first_col(datatable)
    payload = json.dumps({"action": "subscribe", "symbols": symbols})
    _run_async(bdd_context, ws.send(payload))


@when(parsers.parse("I wait for {seconds:d} seconds"))
def wait_seconds(seconds: int):
    time.sleep(seconds)


@when("the server closes the connection")
def server_closes_connection(bdd_context):
    ws = bdd_context["ws"]
    assert ws is not None, "WS not connected"
    _run_async(bdd_context, ws.close())
    bdd_context["ws"] = None


# ---------- THEN (HTTP) ----------
@then(parsers.parse("the response status should be {code:d}"))
def assert_status(code: int, bdd_context):
    resp = bdd_context["resp"]
    assert resp is not None, "No HTTP response captured"
    assert (
        resp.status_code == code
    ), f"Expected {code}, got {resp.status_code}. Body: {resp.text[:500]}"


@then(parsers.parse('the response should match the "{schema_name}" schema'))
def assert_schema(schema_name: str, bdd_context):
    payload = bdd_context["json"]
    assert isinstance(payload, (dict, list)), f"Response is not JSON object/array: {payload!r}"
    jsonschema_validate(payload, _load_schema(schema_name))


@then(parsers.parse('the field "{path}" should equal "{expect}"'))
def assert_field_eq(path: str, expect: str, bdd_context):
    payload = bdd_context["json"]
    assert isinstance(payload, dict), "JSON object expected"
    cur: Any = payload
    for part in path.lstrip("$.").split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    assert str(cur) == expect, f'Expected {path} == "{expect}", got {cur!r}'


@then(parsers.parse('the field "{path}" should be within {secs:d} seconds of now'))
def assert_fresh(path: str, secs: int, bdd_context):
    payload = bdd_context["json"]
    assert isinstance(payload, dict), "JSON object expected"
    cur: Any = payload
    for part in path.lstrip("$.").split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    ts = _parse_iso(str(cur))
    assert ts is not None, f"Unparseable timestamp at {path}: {cur!r}"
    age = time.time() - ts
    assert age < secs, f"Timestamp stale: {age:.1f}s > {secs}s"


@then(parsers.re(r'^every candle should have fields:\s*(?P<csv>.+)$'))
def assert_candle_fields(csv: str, bdd_context):
    payload = bdd_context["json"] or {}
    candles = payload.get("candles") or []
    fields = [f.strip() for f in csv.split(",")]
    assert isinstance(candles, list), "candles is not a list"
    for idx, candle in enumerate(candles):
        for field in fields:
            assert field in candle, f"candle[{idx}] missing '{field}'"


@then(parsers.re(r'^the array "(?P<name>[^"]+)" length should be (?P<op>>=|<=|==|>|<) (?P<n>\d+)$'))
def assert_array_len(name: str, op: str, n: str, bdd_context):
    payload = bdd_context["json"] or {}
    array = payload.get(name)
    assert isinstance(array, list), f"{name} is not a list"
    length = len(array)
    target = int(n)
    checks = {"<": length < target, ">": length > target, "==": length == target, "<=": length <= target, ">=": length >= target}
    assert checks[op], f"len({name})={length} does not satisfy {op} {target}"


@then("the best bid price should be < the best ask price")
def assert_spread_positive(bdd_context):
    payload = bdd_context["json"] or {}
    bids = payload.get("bids") or []
    asks = payload.get("asks") or []
    assert bids and asks, "Empty bids/asks"
    assert bids[0][0] < asks[0][0], f"best_bid={bids[0][0]} !< best_ask={asks[0][0]}"


# ---------- THEN (WebSocket) ----------
@then(parsers.parse('I should receive a message within {secs:d} seconds matching "{kind}"'))
def ws_receive_one(secs: int, kind: str, bdd_context):
    ws = bdd_context["ws"]
    assert ws is not None, "WS not connected"
    raw = _run_async(bdd_context, asyncio.wait_for(ws.recv(), timeout=secs))
    data = json.loads(raw)
    bdd_context["last_msg"] = data
    if kind.lower() == "quote":
        assert "symbol" in data and "price" in data, f"Not a quote: {data}"


@then("I should receive messages for all of:")
def ws_receive_all(datatable: List[List[str]], bdd_context):
    ws = bdd_context["ws"]
    assert ws is not None, "WS not connected"
    want = set(_datatable_first_col(datatable))
    seen, deadline = set(), time.time() + 15
    while time.time() < deadline and seen != want:
        raw = _run_async(bdd_context, asyncio.wait_for(ws.recv(), timeout=10))
        data = json.loads(raw)
        symbol = data.get("symbol")
        if symbol in want:
            seen.add(symbol)
    missing = want - seen
    assert not missing, f"Missing symbols: {missing}"


@then(parsers.parse('I should receive an error event with code "{code}"'))
def ws_expect_error(code: str, bdd_context):
    ws = bdd_context["ws"]
    assert ws is not None, "WS not connected"
    deadline = time.time() + 10
    while time.time() < deadline:
        raw = _run_async(bdd_context, asyncio.wait_for(ws.recv(), timeout=10))
        data = json.loads(raw)
        if str(data.get("code")) == code or str((data.get("error") or {}).get("code")) == code:
            return
    pytest.fail(f"No error event with code {code!r}")


@then("the connection should still be open")
def ws_open(bdd_context):
    ws = bdd_context["ws"]
    assert ws is not None, "WS not connected"
    assert not ws.closed


@then(parsers.parse('the field "{path}" in the last message should equal "{expect}"'))
def last_message_field_equals(path: str, expect: str, bdd_context):
    msg = bdd_context.get("last_msg")
    assert isinstance(msg, dict), "No last WS message to assert on"
    cur: Any = msg
    for part in path.lstrip("$.").split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    assert str(cur) == expect, f'Expected last_msg.{path} == "{expect}", got {cur!r}'


@then(parsers.parse("the client reconnects within {seconds:d} seconds"))
def client_reconnects_within(seconds: int, bdd_context):
    pytest.skip("Reconnect behavior not implemented in tests yet.")


# ---------- Cleanup ----------
@pytest.fixture(autouse=True)
def _cleanup_ws(bdd_context):
    try:
        yield
    finally:
        ws = bdd_context.get("ws")
        if ws and not ws.closed:
            try:
                _run_async(bdd_context, ws.close())
            except Exception:
                pass
        loop: asyncio.AbstractEventLoop = bdd_context.get("loop")
        if loop and not loop.is_closed():
            loop.close()
