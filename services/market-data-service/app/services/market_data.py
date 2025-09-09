import asyncio
import logging
import os
from fastapi import HTTPException
from typing import Dict, List
from ..providers import YahooFinanceProvider, FinnhubProvider
from .cache import DataCache
from .websocket import ConnectionManager
from ..core.config import settings

logger = logging.getLogger(__name__)

class MarketDataService:
    def __init__(self):
        # Get Finnhub API key from environment (via settings/env)
        finnhub_api_key = os.getenv("FINNHUB_API_KEY") or settings.finnhub_api_key
        
        # Initialize providers - only add Finnhub if we have a valid API key
        self.providers = []
        
        if finnhub_api_key and finnhub_api_key != "your_finnhub_api_key_here":
            try:
                finnhub_provider = FinnhubProvider(api_key=finnhub_api_key)
                self.providers.append(finnhub_provider)
                logger.info("Finnhub provider initialized with API key")
            except Exception as e:
                logger.warning(f"Failed to initialize Finnhub provider: {e}")
        else:
            logger.warning("No valid Finnhub API key found. Using Yahoo Finance only.")
        
        # Always add Yahoo Finance as fallback
        self.providers.append(YahooFinanceProvider())
        
        # Cache TTL configurable via env/settings
        self.cache = DataCache(ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", settings.cache_ttl_seconds)))
        # Real-time broadcast interval (seconds)
        self.update_interval = int(os.getenv("WEBSOCKET_UPDATE_INTERVAL", "5"))
        self.connection_manager = ConnectionManager()
        self.background_tasks_running = False
    
    async def get_stock_price(self, symbol: str) -> Dict:
        """Get stock price with caching and fallback"""
        cache_key = f"price:{symbol}"
        
        # Try cache first
        cached_data = self.cache.get(cache_key)
        if cached_data:
            logger.info(f"Cache hit for {symbol}")
            return cached_data
        
        # Try providers
        for provider in self.providers:
            if not provider.is_available:
                continue
                
            data = await provider.get_price(symbol)
            if data:
                self.cache.set(cache_key, data)
                logger.info(f"Fetched {symbol} from {provider.name}")
                return data
        
        # No provider worked
        raise HTTPException(
            status_code=503,
            detail=f"Unable to fetch data for {symbol} from any provider"
        )
    
    async def get_historical_data(self, symbol: str, period: str = "1mo") -> Dict:
        """Get historical data"""
        for provider in self.providers:
            data = await provider.get_historical(symbol, period)
            if data:
                return data
        
        raise HTTPException(
            status_code=503,
            detail=f"Unable to fetch historical data for {symbol}"
        )
    
    async def start_background_tasks(self):
        """Start background tasks for cache cleanup and real-time updates"""
        if self.background_tasks_running:
            return
        
        self.background_tasks_running = True
        
        # Start cache cleanup task
        asyncio.create_task(self._cache_cleanup_task())
        
        # Start real-time data broadcasting
        asyncio.create_task(self._real_time_broadcast_task())
        
        logger.info("Background tasks started")
    
    async def _cache_cleanup_task(self):
        """Clean up expired cache entries"""
        while self.background_tasks_running:
            self.cache.clear_expired()
            await asyncio.sleep(60)  # Clean up every minute
    
    async def _real_time_broadcast_task(self):
        """Broadcast real-time data to WebSocket clients"""
        while self.background_tasks_running:
            if self.connection_manager.symbol_subscribers:
                for symbol in list(self.connection_manager.symbol_subscribers.keys()):
                    try:
                        data = await self.get_stock_price(symbol)
                        await self.connection_manager.broadcast_to_symbol(symbol, data)
                    except Exception as e:
                        logger.error(f"Error broadcasting {symbol}: {e}")
            
            await asyncio.sleep(self.update_interval)  # Update interval configurable
    
    async def get_company_profile(self, symbol: str) -> Dict:
        """Get company profile data (Finnhub only)"""
        for provider in self.providers:
            if hasattr(provider, 'get_company_profile'):
                data = await provider.get_company_profile(symbol)
                if data:
                    return data
        
        raise HTTPException(
            status_code=503,
            detail=f"Unable to fetch company profile for {symbol}"
        )
    
    async def get_news_sentiment(self, symbol: str) -> Dict:
        """Get news sentiment data (Finnhub only)"""
        for provider in self.providers:
            if hasattr(provider, 'get_news_sentiment'):
                data = await provider.get_news_sentiment(symbol)
                if data:
                    return data
        
        raise HTTPException(
            status_code=503,
            detail=f"Unable to fetch news sentiment for {symbol}"
        )
    
    def get_stats(self) -> Dict:
        return {
            "providers": [
                {
                    "name": provider.name,
                    "available": provider.is_available,
                    "last_error": provider.last_error
                }
                for provider in self.providers
            ],
            "cache": self.cache.stats(),
            "websocket": self.connection_manager.stats()
        }
