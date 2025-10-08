from datetime import datetime, timedelta, timezone
from typing import List, Literal
import random

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect

app = FastAPI(title="Test Market Data Stub")

# ---------- REST: /api/v1/quote ----------
@app.get("/api/v1/quote")
def quote(symbol: str = Query(..., min_length=1)):
    bad = {"ZZZZ", "UNKNOWN", "??"}
    if symbol.upper() in bad:
        raise HTTPException(status_code=422, detail="Unknown symbol")
    px = round(random.uniform(100, 200), 2)
    return {
        "symbol": symbol.upper(),
        "price": px,
        "bid": round(px - 0.05, 2),
        "ask": round(px + 0.05, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ---------- REST: /api/v1/ohlcv ----------
@app.get("/api/v1/ohlcv")
def ohlcv(
    symbol: str,
    interval: Literal["1m", "5m", "15m", "1h", "1d"],
    start: datetime,
    end: datetime,
):
    if start >= end:
        raise HTTPException(status_code=422, detail="start must be < end")

    allowed = {"1m", "5m", "15m", "1h", "1d"}
    if interval not in allowed:
        raise HTTPException(status_code=400, detail="invalid interval")

    candles = []
    cur = start
    step = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "1d": timedelta(days=1),
    }[interval]

    last_close = round(random.uniform(100, 200), 2)
    while cur < end:
        # simple rule: skip weekends for intraday intervals, allow all days for 1d
        if interval == "1d" or cur.weekday() < 5:
            o = last_close
            h = round(o + random.uniform(0.0, 1.0), 2)
            l = round(o - random.uniform(0.0, 1.0), 2)
            c = round(random.uniform(l, h), 2)
            v = random.randint(1000, 100000)
            candles.append(
                {"t": cur.replace(tzinfo=timezone.utc).isoformat(), "o": o, "h": h, "l": l, "c": c, "v": v}
            )
            last_close = c
        cur += step

    return {"symbol": symbol.upper(), "interval": interval, "candles": candles}

# ---------- REST: /api/v1/orderbook ----------
@app.get("/api/v1/orderbook")
def orderbook(symbol: str, depth: int = Query(5, ge=1, le=50)):
    illiquid = {"HALT", "ILLQ"}
    if symbol.upper() in illiquid:
        return {"symbol": symbol.upper(), "bids": [], "asks": [], "timestamp": datetime.now(timezone.utc).isoformat()}

    if depth <= 0:
        raise HTTPException(status_code=422, detail="depth must be >= 1")
    mid = round(random.uniform(100, 200), 2)
    tick = 0.01
    bids = [[round(mid - (i + 1) * tick, 2), random.randint(10, 1000)] for i in range(depth)]
    asks = [[round(mid + (i + 1) * tick, 2), random.randint(10, 1000)] for i in range(depth)]
    return {"symbol": symbol.upper(), "bids": bids, "asks": asks, "timestamp": datetime.now(timezone.utc).isoformat()}

# ---------- WS: /ws/quotes ----------
class Hub:
    def __init__(self):
        self.clients: dict[WebSocket, set[str]] = {}

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients[ws] = set()

    def disconnect(self, ws: WebSocket):
        self.clients.pop(ws, None)

    async def subscribe(self, ws: WebSocket, symbols: List[str]):
        self.clients[ws] |= {s.upper() for s in symbols}

    async def send_one(self, ws: WebSocket, symbol: str):
        px = round(random.uniform(100, 200), 2)
        msg = {"type": "quote", "symbol": symbol, "price": px, "timestamp": datetime.now(timezone.utc).isoformat()}
        await ws.send_json(msg)

hub = Hub()

@app.websocket("/ws/quotes")
async def ws_quotes(ws: WebSocket):
    await hub.connect(ws)
    try:
        while True:
            msg = await ws.receive_json()
            action = msg.get("action")
            if action == "subscribe":
                symbols = msg.get("symbols", [])
                bad = {"UNKNOWN", "ZZZZ", "??"}
                # emit error events for bad ones, normal ticks for good ones
                for s in symbols:
                    sU = s.upper()
                    if sU in bad:
                        await ws.send_json({"type": "error", "code": "unknown_symbol", "symbol": sU})
                    else:
                        await hub.subscribe(ws, [sU])
                        await hub.send_one(ws, sU)
            elif action == "ping":
                await ws.send_json({"type": "pong"})
            elif action == "close":
                await ws.close(code=1001)
    except WebSocketDisconnect:
        hub.disconnect(ws)
