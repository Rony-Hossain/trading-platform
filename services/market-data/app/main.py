from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import time
from typing import List, Optional
import asyncio
from contextlib import asynccontextmanager

from .config import settings
from .database import get_db_session, init_db
from .cache import get_redis, init_redis
from .models import StockPrice, HistoricalData, SearchResult, CompanyProfile
from .services.market_data_service import MarketDataService
from .services.cache_service import CacheService
from .middleware import setup_middleware

logger = structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Market Data API")
    await init_redis()
    await init_db()
    logger.info("Market Data API started successfully")
    yield
    # Shutdown
    logger.info("Shutting down Market Data API")

app = FastAPI(
    title="Market Data API",
    description="Real-time stock data, historical prices, and company information",
    version="1.0.0",
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    try:
        # Check database connection
        async with get_db_session() as db:
            await db.execute("SELECT 1")
        
        # Check Redis connection
        redis = await get_redis()
        await redis.ping()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "market-data-api"
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )

@app.get("/stocks/{symbol}", response_model=StockPrice)
async def get_stock_price(
    symbol: str,
    db=Depends(get_db_session),
    redis=Depends(get_redis)
):
    """Get real-time stock price"""
    try:
        symbol = symbol.upper()
        logger.info("Fetching stock price", symbol=symbol)
        
        cache_service = CacheService(redis)
        market_service = MarketDataService(db, cache_service)
        
        price_data = await market_service.get_current_price(symbol)
        if not price_data:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        
        return price_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching stock price", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/stocks/{symbol}/history")
async def get_historical_data(
    symbol: str,
    period: str = Query(default="1y", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)"),
    limit: Optional[int] = Query(default=None, description="Maximum number of data points"),
    db=Depends(get_db_session),
    redis=Depends(get_redis)
):
    """Get historical OHLCV data"""
    try:
        symbol = symbol.upper()
        logger.info("Fetching historical data", symbol=symbol, period=period)
        
        cache_service = CacheService(redis)
        market_service = MarketDataService(db, cache_service)
        
        historical_data = await market_service.get_historical_data(symbol, period, limit)
        
        return {
            "symbol": symbol,
            "period": period,
            "data": historical_data
        }
    except Exception as e:
        logger.error("Error fetching historical data", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/stocks/search", response_model=List[SearchResult])
async def search_symbols(
    q: str = Query(description="Search query"),
    limit: int = Query(default=10, le=50, description="Maximum number of results"),
    db=Depends(get_db_session),
    redis=Depends(get_redis)
):
    """Search for stock symbols"""
    try:
        logger.info("Searching symbols", query=q, limit=limit)
        
        cache_service = CacheService(redis)
        market_service = MarketDataService(db, cache_service)
        
        results = await market_service.search_symbols(q, limit)
        return results
    except Exception as e:
        logger.error("Error searching symbols", query=q, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/stocks/{symbol}/profile", response_model=CompanyProfile)
async def get_company_profile(
    symbol: str,
    db=Depends(get_db_session),
    redis=Depends(get_redis)
):
    """Get company profile information"""
    try:
        symbol = symbol.upper()
        logger.info("Fetching company profile", symbol=symbol)
        
        cache_service = CacheService(redis)
        market_service = MarketDataService(db, cache_service)
        
        profile = await market_service.get_company_profile(symbol)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile for {symbol} not found")
        
        return profile
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching company profile", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/stocks/batch")
async def get_batch_prices(
    symbols: List[str],
    db=Depends(get_db_session),
    redis=Depends(get_redis)
):
    """Get prices for multiple symbols"""
    try:
        if len(symbols) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 symbols allowed")
        
        symbols = [s.upper() for s in symbols]
        logger.info("Fetching batch prices", symbols=symbols)
        
        cache_service = CacheService(redis)
        market_service = MarketDataService(db, cache_service)
        
        prices = await market_service.get_batch_prices(symbols)
        return {"prices": prices}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching batch prices", symbols=symbols, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# WebSocket endpoint for real-time data (placeholder)
@app.websocket("/realtime")
async def websocket_endpoint(websocket):
    """WebSocket for real-time stock updates"""
    await websocket.accept()
    try:
        # Implementation for real-time data streaming
        # This would connect to market data feeds and push updates
        while True:
            await asyncio.sleep(1)
            # Placeholder - would send real market data
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": time.time()
            })
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )