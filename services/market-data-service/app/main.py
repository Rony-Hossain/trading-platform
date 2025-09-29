from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from .services import MarketDataService

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
    logger.info("Market Data Service stopped")

app = FastAPI(
    title="Market Data Service",
    description="Reliable real-time and historical stock data",
    version="1.0.0",
    lifespan=lifespan
)

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
    return {
        "service": "Market Data Service",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
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
    return {
        "status": "healthy",
        "service": "market-data-service",
        "timestamp": datetime.now().isoformat(),
        "providers_status": {
            provider["name"]: provider["available"]
            for provider in stats["providers"]
        },
        "macro": stats.get("macro", {}),
        "options_metrics": stats.get("options_metrics", {})
    }

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

