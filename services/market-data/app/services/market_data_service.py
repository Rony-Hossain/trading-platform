import asyncio
import yfinance as yf
import structlog
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update
from sqlalchemy.orm import selectinload

from ..models import StockPrice, HistoricalData, SearchResult, CompanyProfile
from ..models import Candle, CompanyProfileDB, SymbolDirectory
from ..cache import CacheService, CacheKeys
from ..config import settings
from .external_data_providers import ExternalDataProvider

logger = structlog.get_logger(__name__)

class MarketDataService:
    def __init__(self, db: AsyncSession, cache: CacheService):
        self.db = db
        self.cache = cache
        self.external_provider = ExternalDataProvider()
    
    async def get_current_price(self, symbol: str) -> Optional[StockPrice]:
        """Get current stock price with caching"""
        cache_key = CacheKeys.stock_price(symbol)
        
        # Try cache first
        cached_price = await self.cache.get(cache_key)
        if cached_price:
            logger.debug("Price cache hit", symbol=symbol)
            return StockPrice(**cached_price)
        
        # Cache miss - fetch from external API
        logger.debug("Price cache miss, fetching from external API", symbol=symbol)
        price_data = await self.external_provider.get_current_price(symbol)
        
        if price_data:
            # Cache the result
            await self.cache.set(cache_key, price_data.dict(), settings.PRICE_CACHE_TTL)
            
            # Store in database for historical tracking
            await self._store_price_in_db(price_data)
            
            return price_data
        
        return None
    
    async def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1y", 
        limit: Optional[int] = None
    ) -> List[HistoricalData]:
        """Get historical OHLCV data"""
        cache_key = CacheKeys.historical_data(symbol, period)
        
        # Try cache first
        cached_data = await self.cache.get(cache_key)
        if cached_data:
            logger.debug("Historical data cache hit", symbol=symbol, period=period)
            data = [HistoricalData(**item) for item in cached_data]
            return data[-limit:] if limit else data
        
        # Try database first for recent data
        db_data = await self._get_historical_from_db(symbol, period)
        
        if db_data and self._is_data_fresh(db_data, period):
            logger.debug("Using fresh database data", symbol=symbol, period=period)
            await self.cache.set(cache_key, [item.dict() for item in db_data], 3600)
            return db_data[-limit:] if limit else db_data
        
        # Fetch from external API
        logger.debug("Fetching historical data from external API", symbol=symbol, period=period)
        external_data = await self.external_provider.get_historical_data(symbol, period)
        
        if external_data:
            # Store in database
            await self._store_historical_in_db(symbol, external_data)
            
            # Cache the result
            await self.cache.set(cache_key, [item.dict() for item in external_data], 3600)
            
            return external_data[-limit:] if limit else external_data
        
        # Fallback to database data even if not fresh
        return db_data[-limit:] if (db_data and limit) else (db_data or [])
    
    async def search_symbols(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search for stock symbols"""
        cache_key = CacheKeys.search_results(query)
        
        # Try cache first
        cached_results = await self.cache.get(cache_key)
        if cached_results:
            logger.debug("Search cache hit", query=query)
            return [SearchResult(**item) for item in cached_results]
        
        # Search in database first
        db_results = await self._search_symbols_in_db(query, limit)
        
        if db_results:
            # Cache database results
            await self.cache.set(
                cache_key, 
                [item.dict() for item in db_results], 
                settings.SEARCH_CACHE_TTL
            )
            return db_results
        
        # Fallback to external API search
        external_results = await self.external_provider.search_symbols(query, limit)
        
        if external_results:
            # Store new symbols in database
            await self._store_symbols_in_db(external_results)
            
            # Cache results
            await self.cache.set(
                cache_key,
                [item.dict() for item in external_results],
                settings.SEARCH_CACHE_TTL
            )
            
            return external_results
        
        return []
    
    async def get_company_profile(self, symbol: str) -> Optional[CompanyProfile]:
        """Get company profile information"""
        cache_key = CacheKeys.company_profile(symbol)
        
        # Try cache first
        cached_profile = await self.cache.get(cache_key)
        if cached_profile:
            logger.debug("Profile cache hit", symbol=symbol)
            return CompanyProfile(**cached_profile)
        
        # Try database
        db_profile = await self._get_profile_from_db(symbol)
        if db_profile and self._is_profile_fresh(db_profile):
            profile = CompanyProfile(**db_profile)
            await self.cache.set(cache_key, profile.dict(), settings.PROFILE_CACHE_TTL)
            return profile
        
        # Fetch from external API
        external_profile = await self.external_provider.get_company_profile(symbol)
        
        if external_profile:
            # Store in database
            await self._store_profile_in_db(external_profile)
            
            # Cache the result
            await self.cache.set(cache_key, external_profile.dict(), settings.PROFILE_CACHE_TTL)
            
            return external_profile
        
        return None
    
    async def get_batch_prices(self, symbols: List[str]) -> List[StockPrice]:
        """Get prices for multiple symbols efficiently"""
        cache_key = CacheKeys.batch_prices(symbols)
        
        # Try cache first
        cached_batch = await self.cache.get(cache_key)
        if cached_batch:
            logger.debug("Batch prices cache hit", symbols=symbols)
            return [StockPrice(**item) for item in cached_batch]
        
        # Get individual cached prices
        cache_keys = [CacheKeys.stock_price(symbol) for symbol in symbols]
        cached_prices = await self.cache.get_many(cache_keys)
        
        uncached_symbols = []
        results = []
        
        for symbol in symbols:
            cache_key = CacheKeys.stock_price(symbol)
            if cache_key in cached_prices:
                results.append(StockPrice(**cached_prices[cache_key]))
            else:
                uncached_symbols.append(symbol)
        
        # Fetch uncached symbols
        if uncached_symbols:
            external_prices = await self.external_provider.get_batch_prices(uncached_symbols)
            
            if external_prices:
                # Cache individual prices
                cache_data = {}
                for price in external_prices:
                    cache_data[CacheKeys.stock_price(price.symbol)] = price.dict()
                    results.append(price)
                    
                    # Store in database
                    await self._store_price_in_db(price)
                
                await self.cache.set_many(cache_data, settings.PRICE_CACHE_TTL)
        
        # Cache the batch result
        await self.cache.set(cache_key, [item.dict() for item in results], 300)  # 5 min cache
        
        return results
    
    # Database helper methods
    async def _store_price_in_db(self, price: StockPrice):
        """Store price data in database"""
        try:
            # Convert to database candle format
            candle = Candle(
                symbol=price.symbol,
                ts=price.timestamp,
                open=price.price,  # For real-time, open = current price
                high=price.day_high or price.price,
                low=price.day_low or price.price,
                close=price.price,
                volume=price.volume
            )
            
            # Upsert the candle
            stmt = insert(Candle).values(candle.__dict__)
            stmt = stmt.on_duplicate_key_update(
                close=stmt.inserted.close,
                high=stmt.inserted.high,
                low=stmt.inserted.low,
                volume=stmt.inserted.volume
            )
            
            await self.db.execute(stmt)
            await self.db.commit()
            
        except Exception as e:
            logger.error("Failed to store price in database", symbol=price.symbol, error=str(e))
            await self.db.rollback()
    
    async def _get_historical_from_db(self, symbol: str, period: str) -> List[HistoricalData]:
        """Get historical data from database"""
        try:
            # Calculate date range based on period
            end_date = datetime.now()
            start_date = self._calculate_start_date(period)
            
            stmt = select(Candle).where(
                Candle.symbol == symbol,
                Candle.ts >= start_date,
                Candle.ts <= end_date
            ).order_by(Candle.ts)
            
            result = await self.db.execute(stmt)
            candles = result.scalars().all()
            
            return [
                HistoricalData(
                    date=candle.ts.strftime("%Y-%m-%d"),
                    open=float(candle.open),
                    high=float(candle.high),
                    low=float(candle.low),
                    close=float(candle.close),
                    volume=candle.volume
                )
                for candle in candles
            ]
            
        except Exception as e:
            logger.error("Failed to get historical data from database", symbol=symbol, error=str(e))
            return []
    
    async def _store_historical_in_db(self, symbol: str, data: List[HistoricalData]):
        """Store historical data in database"""
        try:
            candles = []
            for item in data:
                candle = Candle(
                    symbol=symbol,
                    ts=datetime.fromisoformat(item.date),
                    open=item.open,
                    high=item.high,
                    low=item.low,
                    close=item.close,
                    volume=item.volume
                )
                candles.append(candle.__dict__)
            
            if candles:
                stmt = insert(Candle).values(candles)
                stmt = stmt.on_duplicate_key_update(
                    open=stmt.inserted.open,
                    high=stmt.inserted.high,
                    low=stmt.inserted.low,
                    close=stmt.inserted.close,
                    volume=stmt.inserted.volume
                )
                
                await self.db.execute(stmt)
                await self.db.commit()
                
        except Exception as e:
            logger.error("Failed to store historical data", symbol=symbol, error=str(e))
            await self.db.rollback()
    
    def _calculate_start_date(self, period: str) -> datetime:
        """Calculate start date based on period string"""
        end_date = datetime.now()
        
        period_map = {
            "1d": timedelta(days=1),
            "5d": timedelta(days=5),
            "1mo": timedelta(days=30),
            "3mo": timedelta(days=90),
            "6mo": timedelta(days=180),
            "1y": timedelta(days=365),
            "2y": timedelta(days=730),
            "5y": timedelta(days=1825),
        }
        
        delta = period_map.get(period, timedelta(days=365))
        return end_date - delta
    
    def _is_data_fresh(self, data: List[HistoricalData], period: str) -> bool:
        """Check if historical data is fresh enough"""
        if not data:
            return False
        
        latest_date = datetime.fromisoformat(data[-1].date)
        now = datetime.now()
        
        # For intraday periods, data should be from today
        if period in ["1d", "5d"]:
            return latest_date.date() == now.date()
        
        # For longer periods, data should be within last few days
        return (now - latest_date).days <= 3
    
    async def _search_symbols_in_db(self, query: str, limit: int) -> List[SearchResult]:
        """Search symbols in database"""
        try:
            stmt = select(SymbolDirectory).where(
                SymbolDirectory.symbol.ilike(f"%{query.upper()}%") |
                SymbolDirectory.name.ilike(f"%{query}%")
            ).limit(limit)
            
            result = await self.db.execute(stmt)
            symbols = result.scalars().all()
            
            return [
                SearchResult(
                    symbol=symbol.symbol,
                    name=symbol.name,
                    exchange=symbol.exchange,
                    type=symbol.type
                )
                for symbol in symbols
            ]
            
        except Exception as e:
            logger.error("Failed to search symbols in database", query=query, error=str(e))
            return []
    
    async def _store_symbols_in_db(self, symbols: List[SearchResult]):
        """Store symbols in database"""
        try:
            symbol_data = []
            for symbol in symbols:
                symbol_data.append({
                    'symbol': symbol.symbol,
                    'name': symbol.name,
                    'exchange': symbol.exchange,
                    'type': symbol.type
                })
            
            if symbol_data:
                stmt = insert(SymbolDirectory).values(symbol_data)
                stmt = stmt.on_duplicate_key_update(
                    name=stmt.inserted.name,
                    exchange=stmt.inserted.exchange,
                    type=stmt.inserted.type
                )
                
                await self.db.execute(stmt)
                await self.db.commit()
                
        except Exception as e:
            logger.error("Failed to store symbols in database", error=str(e))
            await self.db.rollback()
    
    async def _get_profile_from_db(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company profile from database"""
        try:
            stmt = select(CompanyProfileDB).where(CompanyProfileDB.symbol == symbol)
            result = await self.db.execute(stmt)
            profile = result.scalar_one_or_none()
            
            if profile:
                return {
                    'symbol': profile.symbol,
                    'name': profile.name,
                    'sector': profile.sector,
                    'industry': profile.industry,
                    'description': profile.description,
                    'website': profile.website,
                    'market_cap': profile.market_cap,
                    'beta': float(profile.beta) if profile.beta else None,
                    'pe_ratio': float(profile.pe_ratio) if profile.pe_ratio else None,
                    'dividend_yield': float(profile.dividend_yield) if profile.dividend_yield else None,
                    'employees': profile.employees,
                    'headquarters': profile.headquarters,
                    'updated_at': profile.updated_at
                }
            
            return None
            
        except Exception as e:
            logger.error("Failed to get profile from database", symbol=symbol, error=str(e))
            return None
    
    async def _store_profile_in_db(self, profile: CompanyProfile):
        """Store company profile in database"""
        try:
            profile_data = {
                'symbol': profile.symbol,
                'name': profile.name,
                'sector': profile.sector,
                'industry': profile.industry,
                'description': profile.description,
                'website': profile.website,
                'market_cap': profile.market_cap,
                'beta': profile.beta,
                'pe_ratio': profile.pe_ratio,
                'dividend_yield': profile.dividend_yield,
                'employees': profile.employees,
                'headquarters': profile.headquarters
            }
            
            stmt = insert(CompanyProfileDB).values(profile_data)
            stmt = stmt.on_duplicate_key_update(**{
                k: stmt.inserted[k] for k in profile_data.keys() if k != 'symbol'
            })
            
            await self.db.execute(stmt)
            await self.db.commit()
            
        except Exception as e:
            logger.error("Failed to store profile in database", symbol=profile.symbol, error=str(e))
            await self.db.rollback()
    
    def _is_profile_fresh(self, profile_data: Dict[str, Any]) -> bool:
        """Check if profile data is fresh (less than 24 hours old)"""
        if 'updated_at' not in profile_data:
            return False
        
        updated_at = profile_data['updated_at']
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        return (datetime.now() - updated_at).total_seconds() < 86400  # 24 hours