import asyncio
import yfinance as yf
import httpx
import structlog
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import time
from tenacity import retry, stop_after_attempt, wait_exponential

from ..models import StockPrice, HistoricalData, SearchResult, CompanyProfile
from ..config import settings

logger = structlog.get_logger(__name__)

class ExternalDataProvider:
    """Aggregates multiple external data providers with fallback logic"""
    
    def __init__(self):
        self.providers = [
            YFinanceProvider(),
            AlphaVantageProvider(),
            PolygonProvider(),
            FinnhubProvider()
        ]
        self.rate_limiter = RateLimiter(settings.EXTERNAL_API_RATE_LIMIT)
    
    async def get_current_price(self, symbol: str) -> Optional[StockPrice]:
        """Get current price with provider fallback"""
        for provider in self.providers:
            if provider.is_available():
                try:
                    await self.rate_limiter.acquire()
                    result = await provider.get_current_price(symbol)
                    if result:
                        logger.debug("Price fetched successfully", 
                                   symbol=symbol, provider=provider.name)
                        return result
                except Exception as e:
                    logger.warning("Provider failed for current price", 
                                 symbol=symbol, provider=provider.name, error=str(e))
                    continue
        
        logger.error("All providers failed for current price", symbol=symbol)
        return None
    
    async def get_historical_data(self, symbol: str, period: str) -> Optional[List[HistoricalData]]:
        """Get historical data with provider fallback"""
        for provider in self.providers:
            if provider.is_available():
                try:
                    await self.rate_limiter.acquire()
                    result = await provider.get_historical_data(symbol, period)
                    if result:
                        logger.debug("Historical data fetched successfully", 
                                   symbol=symbol, period=period, provider=provider.name)
                        return result
                except Exception as e:
                    logger.warning("Provider failed for historical data", 
                                 symbol=symbol, provider=provider.name, error=str(e))
                    continue
        
        logger.error("All providers failed for historical data", symbol=symbol, period=period)
        return None
    
    async def search_symbols(self, query: str, limit: int = 10) -> Optional[List[SearchResult]]:
        """Search symbols with provider fallback"""
        for provider in self.providers:
            if provider.is_available() and provider.supports_search():
                try:
                    await self.rate_limiter.acquire()
                    result = await provider.search_symbols(query, limit)
                    if result:
                        logger.debug("Symbol search successful", 
                                   query=query, provider=provider.name)
                        return result
                except Exception as e:
                    logger.warning("Provider failed for symbol search", 
                                 query=query, provider=provider.name, error=str(e))
                    continue
        
        logger.error("All providers failed for symbol search", query=query)
        return None
    
    async def get_company_profile(self, symbol: str) -> Optional[CompanyProfile]:
        """Get company profile with provider fallback"""
        for provider in self.providers:
            if provider.is_available() and provider.supports_profiles():
                try:
                    await self.rate_limiter.acquire()
                    result = await provider.get_company_profile(symbol)
                    if result:
                        logger.debug("Company profile fetched successfully", 
                                   symbol=symbol, provider=provider.name)
                        return result
                except Exception as e:
                    logger.warning("Provider failed for company profile", 
                                 symbol=symbol, provider=provider.name, error=str(e))
                    continue
        
        logger.error("All providers failed for company profile", symbol=symbol)
        return None
    
    async def get_batch_prices(self, symbols: List[str]) -> Optional[List[StockPrice]]:
        """Get batch prices with provider fallback"""
        for provider in self.providers:
            if provider.is_available() and provider.supports_batch():
                try:
                    await self.rate_limiter.acquire()
                    result = await provider.get_batch_prices(symbols)
                    if result:
                        logger.debug("Batch prices fetched successfully", 
                                   symbols=symbols, provider=provider.name)
                        return result
                except Exception as e:
                    logger.warning("Provider failed for batch prices", 
                                 symbols=symbols, provider=provider.name, error=str(e))
                    continue
        
        # Fallback to individual requests
        logger.info("Batch failed, falling back to individual requests", symbols=symbols)
        results = []
        for symbol in symbols:
            price = await self.get_current_price(symbol)
            if price:
                results.append(price)
        
        return results if results else None


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_second: float):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
    
    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        now = time.time()
        time_since_last = now - self.last_call
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_call = time.time()


class BaseProvider:
    """Base class for external data providers"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.client = None
    
    def is_available(self) -> bool:
        """Check if provider is available (has API key, etc.)"""
        return True
    
    def supports_search(self) -> bool:
        """Check if provider supports symbol search"""
        return False
    
    def supports_profiles(self) -> bool:
        """Check if provider supports company profiles"""
        return False
    
    def supports_batch(self) -> bool:
        """Check if provider supports batch requests"""
        return False
    
    async def get_current_price(self, symbol: str) -> Optional[StockPrice]:
        raise NotImplementedError
    
    async def get_historical_data(self, symbol: str, period: str) -> Optional[List[HistoricalData]]:
        raise NotImplementedError
    
    async def search_symbols(self, query: str, limit: int = 10) -> Optional[List[SearchResult]]:
        raise NotImplementedError
    
    async def get_company_profile(self, symbol: str) -> Optional[CompanyProfile]:
        raise NotImplementedError
    
    async def get_batch_prices(self, symbols: List[str]) -> Optional[List[StockPrice]]:
        raise NotImplementedError


class YFinanceProvider(BaseProvider):
    """Yahoo Finance provider using yfinance library"""
    
    def __init__(self):
        super().__init__()
    
    def supports_search(self) -> bool:
        return True
    
    def supports_profiles(self) -> bool:
        return True
    
    def supports_batch(self) -> bool:
        return True
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_current_price(self, symbol: str) -> Optional[StockPrice]:
        """Get current price from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'regularMarketPrice' not in info:
                return None
            
            return StockPrice(
                symbol=symbol,
                price=info.get('regularMarketPrice', 0),
                change=info.get('regularMarketChange', 0),
                changePercent=info.get('regularMarketChangePercent', 0),
                volume=info.get('regularMarketVolume', 0),
                timestamp=datetime.now(),
                bid=info.get('bid'),
                ask=info.get('ask'),
                dayHigh=info.get('regularMarketDayHigh'),
                dayLow=info.get('regularMarketDayLow'),
                previousClose=info.get('regularMarketPreviousClose')
            )
        except Exception as e:
            logger.error("YFinance current price error", symbol=symbol, error=str(e))
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_historical_data(self, symbol: str, period: str) -> Optional[List[HistoricalData]]:
        """Get historical data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return None
            
            data = []
            for date, row in hist.iterrows():
                data.append(HistoricalData(
                    date=date.strftime("%Y-%m-%d"),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume'])
                ))
            
            return data
        except Exception as e:
            logger.error("YFinance historical data error", symbol=symbol, error=str(e))
            return None
    
    async def search_symbols(self, query: str, limit: int = 10) -> Optional[List[SearchResult]]:
        """Search symbols using Yahoo Finance"""
        # Note: yfinance doesn't have a direct search API
        # This is a simplified implementation
        try:
            # Try common suffixes for the query
            possible_symbols = [
                query.upper(),
                f"{query.upper()}.L",  # London
                f"{query.upper()}.TO",  # Toronto
            ]
            
            results = []
            for symbol in possible_symbols[:limit]:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    if info and 'shortName' in info:
                        results.append(SearchResult(
                            symbol=symbol,
                            name=info.get('shortName', symbol),
                            exchange=info.get('exchange', 'Unknown'),
                            type='stock'
                        ))
                except:
                    continue
            
            return results if results else None
        except Exception as e:
            logger.error("YFinance search error", query=query, error=str(e))
            return None
    
    async def get_company_profile(self, symbol: str) -> Optional[CompanyProfile]:
        """Get company profile from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return None
            
            return CompanyProfile(
                symbol=symbol,
                name=info.get('shortName', info.get('longName', symbol)),
                sector=info.get('sector'),
                industry=info.get('industry'),
                description=info.get('longBusinessSummary'),
                website=info.get('website'),
                marketCap=info.get('marketCap'),
                beta=info.get('beta'),
                peRatio=info.get('trailingPE'),
                dividendYield=info.get('dividendYield'),
                employees=info.get('fullTimeEmployees'),
                headquarters=f"{info.get('city', '')}, {info.get('country', '')}" if info.get('city') else None
            )
        except Exception as e:
            logger.error("YFinance profile error", symbol=symbol, error=str(e))
            return None
    
    async def get_batch_prices(self, symbols: List[str]) -> Optional[List[StockPrice]]:
        """Get batch prices from Yahoo Finance"""
        try:
            symbols_str = " ".join(symbols)
            tickers = yf.Tickers(symbols_str)
            
            results = []
            for symbol in symbols:
                try:
                    ticker = getattr(tickers.tickers, symbol, None)
                    if ticker:
                        price = await self.get_current_price(symbol)
                        if price:
                            results.append(price)
                except:
                    continue
            
            return results if results else None
        except Exception as e:
            logger.error("YFinance batch prices error", symbols=symbols, error=str(e))
            return None


class AlphaVantageProvider(BaseProvider):
    """Alpha Vantage provider"""
    
    def __init__(self):
        super().__init__()
        self.api_key = settings.ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def supports_search(self) -> bool:
        return True
    
    async def get_current_price(self, symbol: str) -> Optional[StockPrice]:
        """Get current price from Alpha Vantage"""
        if not self.is_available():
            return None
        
        # Implementation would go here
        # This is a placeholder
        return None
    
    async def get_historical_data(self, symbol: str, period: str) -> Optional[List[HistoricalData]]:
        """Get historical data from Alpha Vantage"""
        if not self.is_available():
            return None
        
        # Implementation would go here
        return None


class PolygonProvider(BaseProvider):
    """Polygon.io provider"""
    
    def __init__(self):
        super().__init__()
        self.api_key = settings.POLYGON_API_KEY
        self.base_url = "https://api.polygon.io"
    
    def is_available(self) -> bool:
        return bool(self.api_key)


class FinnhubProvider(BaseProvider):
    """Finnhub provider"""
    
    def __init__(self):
        super().__init__()
        self.api_key = settings.FINNHUB_API_KEY
        self.base_url = "https://finnhub.io/api/v1"
    
    def is_available(self) -> bool:
        return bool(self.api_key)