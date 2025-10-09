import asyncio
import contextlib
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from dotenv import load_dotenv

from .core.config import Settings, get_settings, hot_reload, validate_settings
from .services import MarketDataService
from .services.options_service import OptionContract, OptionsChain, options_service

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instance
market_service = MarketDataService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await market_service.start_background_tasks()
    logger.info("Market Data Service started")
    yield
    # Shutdown
    market_service.background_tasks_running = False
    await market_service.data_collector.stop()
    logger.info("Market Data Service stopped")

app = FastAPI(
    title="Market Data Service",
    description="Reliable real-time and historical stock data",
    version="1.0.0",
    lifespan=lifespan
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    stats = await market_service.get_stats()
    settings = get_settings()
    return {
        "service": "Market Data Service",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "env": settings.env,
        "policy_version": settings.policy_version,
        "stats": stats,
        "endpoints": {
            "stock_price": "/stocks/{symbol}/price",
            "historical": "/stocks/{symbol}/history",
            "company_profile": "/stocks/{symbol}/profile",
            "news_sentiment": "/stocks/{symbol}/sentiment",
            "options_metrics": "/options/{symbol}/metrics",
            "options_metrics_history": "/options/{symbol}/metrics/history",
            "unusual_options_activity": "/options/{symbol}/unusual",
            "options_flow_analysis": "/options/{symbol}/flow",
            "macro_snapshot": "/factors/macro",
            "macro_history": "/factors/macro/{factor_key}/history",
            "macro_refresh": "/admin/macro/refresh",
            "batch_quotes": "/stocks/batch",
            "health": "/health",
            "websocket": "/ws/{symbol}"
        }
    }

@app.get("/health")
async def health_check():
    stats = await market_service.get_stats()
    settings = get_settings()
    return {
        "status": "healthy",
        "service": "market-data-service",
        "timestamp": datetime.now().isoformat(),
        "env": settings.env,
        "policy_version": settings.policy_version,
        "providers_status": {
            provider["name"]: provider["available"]
            for provider in stats["providers"]
        },
        "macro": stats.get("macro", {}),
        "options_metrics": stats.get("options_metrics", {})
    }


@app.post("/ops/validate")
def ops_validate():
    candidate = Settings()
    ok, reason = validate_settings(candidate)
    return {"ok": ok, "reason": reason, "policy_version": candidate.policy_version}


@app.post("/ops/reload")
async def ops_reload():
    result = hot_reload()
    if result.get("ok"):
        market_service.reload_configuration()
        result["policy_version"] = get_settings().policy_version
    return result


@app.get("/stats/providers")
async def stats_providers():
    registry = market_service.registry
    breakers = market_service.breakers
    settings = get_settings()

    providers = []
    for name, entry in registry.providers.items():
        providers.append(
            {
                "provider": name,
                "state": breakers.get_state(name),
                "capabilities": sorted(entry.capabilities),
                "latency_p95_ms": entry.stats.latency_p95_ms,
                "error_ewma": entry.stats.error_ewma,
                "completeness_deficit": entry.stats.completeness_deficit,
                "health_score": registry.health_score(name),
                "history": [{"t": t, "h": h} for (t, h) in entry.h_history],
            }
        )

    return {
        "policy_version": settings.policy_version,
        "policy": {
            "bars_1m": settings.POLICY_BARS_1M,
            "eod": settings.POLICY_EOD,
            "quotes_l1": settings.POLICY_QUOTES_L1,
            "options_chain": settings.POLICY_OPTIONS_CHAIN,
        },
        "registered_capabilities": registry.capabilities_map(),
        "providers": providers,
    }


@app.get("/stats/cadence")
async def stats_cadence():
    """Return universe tiering and cadence policy."""
    settings = get_settings()
    return {
        "tiers": {"max": settings.TIER_MAXS},
        "cadence": {
            "T0": settings.CADENCE_T0,
            "T1": settings.CADENCE_T1,
            "T2": settings.CADENCE_T2,
        },
        "use_rlc": settings.USE_RLC,
        "local_sweep": settings.LOCAL_SWEEP_ENABLED,
        "policy_version": settings.policy_version,
    }


@app.get("/ops/cursor/{symbol}/{interval}/{source}")
async def read_cursor(symbol: str, interval: str, source: str):
    """Read ingestion cursor for a symbol/interval/source."""
    from .services.database import db_service
    ts = await db_service.get_cursor(symbol, interval, source)
    return {
        "symbol": symbol,
        "interval": interval,
        "source": source,
        "last_ts": ts.isoformat() + "Z" if ts else None,
    }


@app.post("/ops/backfill")
async def trigger_backfill(payload: dict):
    """
    Manually trigger a backfill.
    Payload: {symbol, interval, start, end, priority}
    """
    from .services.database import db_service
    req = {**payload}
    try:
        start = datetime.fromisoformat(req["start"].replace("Z", ""))
        end = datetime.fromisoformat(req["end"].replace("Z", ""))
    except Exception:
        raise HTTPException(400, "Invalid start/end ISO8601")

    await db_service.enqueue_backfill(
        req["symbol"], req.get("interval", "1m"), start, end, req.get("priority", "T2")
    )
    return {"enqueued": True}

@app.get("/factors/macro")
async def get_macro_factors(factors: Optional[List[str]] = Query(None)):
    """Return snapshot of configured macro factors."""
    return await market_service.get_macro_snapshot(factors)


@app.get("/factors/macro/{factor_key}/history")
async def get_macro_history_endpoint(factor_key: str, lookback_days: int = 90):
    """Return macro factor history for the specified key."""
    try:
        return await market_service.get_macro_history(factor_key, lookback_days)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown macro factor {factor_key.upper()}")


@app.post("/admin/macro/refresh")
async def refresh_macro_endpoint(factor: Optional[str] = Query(None)):
    """Trigger a manual macro factor refresh."""
    return await market_service.refresh_macro_factors(factor)


@app.get("/stocks/{symbol}/price")
async def get_stock_price(symbol: str):
    """Get current stock price"""
    return await market_service.get_stock_price(symbol.upper())

@app.get("/stocks/{symbol}/history")
async def get_historical_data(symbol: str, period: str = "1mo"):
    """Get historical stock data"""
    return await market_service.get_historical_data(symbol.upper(), period)


@app.get("/stocks/{symbol}/intraday")
async def get_intraday_data(symbol: str, interval: str = "1m"):
    """Get intraday stock data (1-minute bars)."""
    return await market_service.get_intraday_data(symbol.upper(), interval)


@app.post("/stocks/batch")
async def get_multiple_stocks(symbols: List[str]):
    """Get prices for multiple stocks"""
    results = []
    for symbol in symbols:
        try:
            data = await market_service.get_stock_price(symbol.upper())
            results.append({"status": "success", **data})
        except Exception as e:
            results.append({
                "status": "error",
                "symbol": symbol.upper(),
                "error": str(e)
            })
    return {"results": results}

@app.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time price updates"""
    await market_service.connection_manager.connect(websocket, symbol.upper())
    try:
        # Send initial data
        try:
            initial_data = await market_service.get_stock_price(symbol.upper())
            await websocket.send_text(json.dumps(initial_data))
        except:
            pass
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client message (ping/pong or subscription changes)
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                # Echo back for now (can add subscription management later)
                await websocket.send_text(json.dumps({"status": "received", "message": message}))
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_text(json.dumps({"type": "ping", "timestamp": datetime.now().isoformat()}))
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from {symbol}")
    finally:
        market_service.connection_manager.disconnect(websocket)


def _find_contract(contracts: List[OptionContract], target_symbol: Optional[str]) -> Optional[OptionContract]:
    if not target_symbol:
        return None
    for contract in contracts:
        if contract.symbol == target_symbol:
            return contract
    return None


async def _options_chain_payload(symbol: str) -> Tuple[Dict[str, Any], OptionsChain]:
    """Fetch the latest options chain and return a summary payload."""
    chain = await options_service.fetch_options_chain(symbol)
    total_calls = len(chain.calls)
    total_puts = len(chain.puts)
    expiries = [exp.isoformat() for exp in chain.expiries]

    return {
        "type": "options_chain",
        "symbol": symbol,
        "underlying_price": chain.underlying_price,
        "timestamp": datetime.now().isoformat(),
        "calls": total_calls,
        "puts": total_puts,
        "expiries": expiries,
    }, chain


def _options_update_payload(symbol: str, chain: OptionsChain, metrics) -> Dict[str, Any]:
    """Build a payload for incremental options metrics updates."""
    call_volume = metrics.call_volume or 0
    put_volume = metrics.put_volume or 0
    total_volume = call_volume + put_volume
    volume_ratio = (call_volume / put_volume) if put_volume else None
    iv_rank = metrics.metadata.get("iv_rank")

    atm_call_symbol = metrics.metadata.get("atm_call")
    atm_put_symbol = metrics.metadata.get("atm_put")
    atm_call = _find_contract(chain.calls, atm_call_symbol)
    atm_put = _find_contract(chain.puts, atm_put_symbol)

    if iv_rank is None and metrics.atm_iv is not None:
        iv_rank = max(0.0, min(100.0, metrics.atm_iv * 100))

    return {
        "type": "options_update",
        "symbol": symbol,
        "underlying_price": chain.underlying_price,
        "timestamp": datetime.now().isoformat(),
        "iv_rank": iv_rank,
        "atm_call_iv": (atm_call.implied_volatility if atm_call else metrics.atm_iv),
        "atm_put_iv": (atm_put.implied_volatility if atm_put else metrics.atm_iv),
        "volume_ratio": volume_ratio,
        "total_volume": total_volume,
    }


async def _options_consumer(websocket: WebSocket):
    """Listen for client messages to keep the WebSocket alive."""
    try:
        while True:
            message = await websocket.receive_text()
            if message.strip().lower() == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}))
    except WebSocketDisconnect:
        raise
    except Exception as exc:
        logger.error("Options WebSocket consumer error: %s", exc)


async def _options_producer(websocket: WebSocket, symbol: str, interval: int):
    """Continuously push options updates to the client."""
    try:
        chain_payload, chain = await _options_chain_payload(symbol)
    except Exception as exc:
        logger.error("Error fetching initial options chain for %s: %s", symbol, exc)
        await websocket.send_text(
            json.dumps(
                {
                    "type": "options_error",
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "error": "Failed to load options chain",
                }
            )
        )
        raise

    await websocket.send_text(json.dumps(chain_payload))

    async def send_update(current_chain: OptionsChain):
        try:
            metrics = options_service.calculate_chain_metrics(current_chain)
            update = _options_update_payload(symbol, current_chain, metrics)
            await websocket.send_text(json.dumps(update))
        except Exception as exc:
            logger.error("Error generating options update for %s: %s", symbol, exc)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "options_error",
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "error": "Failed to generate options update",
                    }
                )
            )

    # Send initial metrics right away
    await send_update(chain)

    refresh_interval = max(1, interval)
    while True:
        await asyncio.sleep(refresh_interval)
        try:
            chain = await options_service.fetch_options_chain(symbol)
        except Exception as exc:
            logger.error("Error refreshing options chain for %s: %s", symbol, exc)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "options_error",
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "error": "Failed to refresh options chain",
                    }
                )
            )
            continue

        await send_update(chain)


async def _handle_options_websocket(websocket: WebSocket, symbol: Optional[str]):
    """Shared handler for options WebSocket connections with symbol from path or query."""
    if not symbol:
        symbol = websocket.query_params.get("symbol")
    if not symbol:
        await websocket.accept()
        await websocket.send_text(json.dumps({"type": "error", "error": "symbol query parameter is required"}))
        await websocket.close(code=4400)
        return

    symbol = symbol.upper()
    await websocket.accept()

    interval = getattr(market_service, "update_interval", 5)
    consumer_task = asyncio.create_task(_options_consumer(websocket))
    producer_task = asyncio.create_task(_options_producer(websocket, symbol, interval))

    try:
        done, pending = await asyncio.wait(
            [consumer_task, producer_task],
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for task in done:
            exc = task.exception()
            if exc and not isinstance(exc, WebSocketDisconnect):
                logger.error("Options WebSocket task error for %s: %s", symbol, exc)
    except WebSocketDisconnect:
        logger.info("Options WebSocket client disconnected for %s", symbol)
    finally:
        for task in (consumer_task, producer_task):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        with contextlib.suppress(Exception):
            await websocket.close()


@app.websocket("/ws/options")
async def options_websocket_query(websocket: WebSocket):
    """Options WebSocket endpoint using symbol query parameter."""
    await _handle_options_websocket(websocket, None)


@app.websocket("/ws/options/{symbol}")
async def options_websocket_path(websocket: WebSocket, symbol: str):
    """Options WebSocket endpoint with symbol in the path."""
    await _handle_options_websocket(websocket, symbol)


@app.get("/options/{symbol}/chain")
async def get_options_chain(symbol: str):
    """Return the full options chain for a symbol."""
    try:
        chain = await options_service.fetch_options_chain(symbol.upper())
    except Exception as exc:
        logger.error("Error fetching options chain for %s: %s", symbol, exc)
        raise HTTPException(status_code=502, detail=f"Unable to fetch options chain for {symbol.upper()}") from exc

    return jsonable_encoder(chain)


def _validate_sentiment(sentiment: str, allowed: List[str]) -> str:
    normalized = (sentiment or "all").lower()
    if normalized not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sentiment '{sentiment}'. Expected one of {', '.join(allowed)}.",
        )
    return normalized


def _validate_complexity(complexity: str, allowed: List[str]) -> str:
    normalized = (complexity or "all").lower()
    if normalized not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid complexity '{complexity}'. Expected one of {', '.join(allowed)}.",
        )
    return normalized


@app.get("/options/{symbol}/strategies")
async def get_options_strategies(
    symbol: str,
    sentiment: str = Query("all", description="Filter by sentiment: bullish, bearish, neutral, or all"),
    complexity: str = Query("all", description="Filter by complexity: beginner, intermediate, advanced, or all"),
):
    """Return advanced options strategies derived from the options chain."""
    sentiment_normalized = _validate_sentiment(sentiment, ["bullish", "bearish", "neutral", "all"])
    complexity_normalized = _validate_complexity(complexity, ["beginner", "intermediate", "advanced", "all"])

    try:
        strategies = await options_service.get_advanced_strategies(
            symbol.upper(), sentiment=sentiment_normalized, complexity=complexity_normalized
        )
    except Exception as exc:
        logger.error("Error generating strategies for %s: %s", symbol, exc)
        raise HTTPException(status_code=502, detail=f"Unable to generate strategies for {symbol.upper()}") from exc

    payload = {"symbol": symbol.upper(), "count": len(strategies), "strategies": strategies}
    return jsonable_encoder(payload)


@app.get("/options/{symbol}/suggestions")
async def get_options_suggestions(
    symbol: str,
    sentiment: str = Query("bullish", description="Trading bias for suggestions"),
    target_delta: float = Query(0.3, ge=0, le=1, description="Approximate option delta to target"),
    max_dte: int = Query(7, ge=1, le=60, description="Maximum days to expiration"),
    min_liquidity: float = Query(50, ge=0, le=100, description="Minimum liquidity score threshold"),
):
    """Return tailored day-trading suggestions built from the options chain."""
    sentiment_normalized = _validate_sentiment(sentiment, ["bullish", "bearish", "neutral"])

    try:
        chain = await options_service.fetch_options_chain(symbol.upper())
        suggestions = options_service.suggest_day_trade(
            symbol.upper(),
            sentiment_normalized,
            chain.underlying_price,
            target_delta=target_delta,
            max_dte=max_dte,
            min_liquidity=min_liquidity,
        )
    except Exception as exc:
        logger.error("Error generating suggestions for %s: %s", symbol, exc)
        raise HTTPException(status_code=502, detail=f"Unable to generate suggestions for {symbol.upper()}") from exc

    if not suggestions:
        raise HTTPException(
            status_code=404,
            detail=f"No trade suggestions available for {symbol.upper()} with current filters",
        )

    payload = {
        "symbol": symbol.upper(),
        "underlying_price": chain.underlying_price,
        "suggestions": suggestions,
    }
    return jsonable_encoder(payload)


@app.get("/options/{symbol}/metrics")
async def get_options_metrics_endpoint(symbol: str):
    """Return latest ATM IV, skew, and implied move metrics."""
    return await market_service.get_options_metrics(symbol.upper())


@app.get("/options/{symbol}/metrics/history")
async def get_options_metrics_history_endpoint(symbol: str, limit: int = 50):
    """Return historical options metrics records."""
    history = await market_service.get_options_metrics_history(symbol.upper(), limit)
    return {"symbol": symbol.upper(), "metrics": history}

@app.get("/options/{symbol}/unusual")
async def get_unusual_options_activity_endpoint(
    symbol: str, 
    lookback_days: int = Query(20, ge=1, le=90, description="Days to look back for unusual activity")
):
    """Detect and return unusual options activity for a symbol"""
    return await market_service.get_unusual_options_activity(symbol.upper(), lookback_days)

@app.get("/options/{symbol}/flow")
async def get_options_flow_analysis_endpoint(symbol: str):
    """Get comprehensive options flow analysis including unusual activity and smart money signals"""
    return await market_service.get_options_flow_analysis(symbol.upper())

@app.get("/stocks/{symbol}/profile")
async def get_company_profile(symbol: str):
    """Get company profile data"""
    return await market_service.get_company_profile(symbol.upper())

@app.get("/stocks/{symbol}/sentiment")
async def get_news_sentiment(symbol: str):
    """Get news sentiment data"""
    return await market_service.get_news_sentiment(symbol.upper())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=True
    )

