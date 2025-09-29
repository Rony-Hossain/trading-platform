import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Set, Optional
import json

from .database import db_service
from ..providers.finnhub_provider import FinnhubProvider
from ..providers.yahoo_finance import YahooFinanceProvider
from ..core.config import settings

logger = logging.getLogger(__name__)

class DataCollectorService:
    """Background service for collecting and storing historical stock data"""
    
    def __init__(self):
        self.running = False
        self.providers = {
            'yahoo': YahooFinanceProvider(),
            'finnhub': FinnhubProvider()
        }
        self.popular_symbols = {
            # Tech Giants
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            # Major Market ETFs
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VEA', 'VWO',
            # Sector ETFs  
            'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLB', 'XLP', 'XLY', 'XLU', 'XLRE',
            # Financial Services
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK.B',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT',
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB',
            # Consumer
            'WMT', 'PG', 'KO', 'PEP', 'MCD', 'NKE',
            # Fintech/Trading
            'HOOD', 'COIN', 'SQ', 'PYPL', 'V', 'MA',
            # Canadian Major Stocks
            'SHOP', 'RY.TO', 'TD.TO', 'BMO.TO', 'BNS.TO', 'CNQ.TO', 'SU.TO',
            # Misc
            'UBER', 'SPOT', 'WBD'
        }
        self.collection_interval = 3600  # 1 hour
        self.task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the data collection background task"""
        if self.running:
            logger.warning("Data collector already running")
            return
        
        self.running = True
        self.task = asyncio.create_task(self._collection_loop())
        logger.info("Data collector started")
    
    async def stop(self):
        """Stop the data collection background task"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Data collector stopped")
    
    async def _collection_loop(self):
        """Main collection loop that runs in background"""
        while self.running:
            try:
                await self._collect_popular_symbols()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _collect_popular_symbols(self):
        """Collect historical data for popular symbols"""
        logger.info("Starting data collection for popular symbols")
        
        for symbol in self.popular_symbols:
            try:
                await self.collect_symbol_data(symbol, days_back=30)
                await asyncio.sleep(2)  # Avoid rate limiting
            except Exception as e:
                logger.error(f"Failed to collect data for {symbol}: {e}")
        
        logger.info("Completed data collection cycle")
    
    async def collect_symbol_data(
        self, 
        symbol: str, 
        days_back: int = 365,
        force_update: bool = False
    ) -> bool:
        """
        Collect historical data for a specific symbol
        
        Args:
            symbol: Stock symbol to collect
            days_back: Number of days of history to collect
            force_update: If True, collect all data regardless of existing data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            symbol = symbol.upper()
            
            # Check existing data coverage unless forcing update
            if not force_update:
                coverage = await db_service.get_data_coverage(symbol)
                if coverage:
                    earliest, latest = coverage
                    days_since_latest = (datetime.now() - latest.replace(tzinfo=None)).days
                    
                    # If we have recent data (within 2 days), skip
                    if days_since_latest < 2:
                        logger.debug(f"Skipping {symbol} - recent data available (latest: {latest})")
                        return True
                    
                    # Adjust collection period to fill gaps
                    days_back = min(days_back, days_since_latest + 5)
            
            # Try Yahoo Finance first (more reliable for historical data)
            historical_data = None
            provider_used = None
            
            try:
                response = await self.providers['yahoo'].get_historical(
                    symbol, 
                    period=f"{days_back}d"
                )
                if response and 'data' in response:
                    # Convert provider response to database format
                    historical_data = []
                    for item in response['data']:
                        historical_data.append({
                            'timestamp': item['date'],
                            'open': item['open'],
                            'high': item['high'],
                            'low': item['low'],
                            'close': item['close'],
                            'volume': item['volume']
                        })
                    provider_used = 'yahoo'
                    logger.debug(f"Retrieved {len(historical_data)} candles for {symbol} from Yahoo Finance")
            except Exception as e:
                logger.warning(f"Yahoo Finance failed for {symbol}: {e}")
            
            # Fallback to Finnhub if Yahoo fails
            if not historical_data:
                try:
                    # Map days to appropriate period string for Finnhub
                    if days_back <= 5:
                        period = "5d"
                    elif days_back <= 30:
                        period = "1mo"
                    elif days_back <= 90:
                        period = "3mo"
                    elif days_back <= 180:
                        period = "6mo"
                    else:
                        period = "1y"
                    
                    response = await self.providers['finnhub'].get_historical(symbol, period)
                    if response and 'data' in response:
                        # Convert provider response to database format
                        historical_data = []
                        for item in response['data']:
                            historical_data.append({
                                'timestamp': item['date'],
                                'open': item['open'],
                                'high': item['high'],
                                'low': item['low'],
                                'close': item['close'],
                                'volume': item['volume']
                            })
                        provider_used = 'finnhub'
                        logger.debug(f"Retrieved {len(historical_data)} candles for {symbol} from Finnhub")
                except Exception as e:
                    logger.warning(f"Finnhub also failed for {symbol}: {e}")
            
            if not historical_data:
                logger.error(f"No provider could fetch data for {symbol}")
                return False
            
            # Store in database
            if settings.store_historical_data:
                stored_count = await db_service.store_candle_data(symbol, historical_data)
                logger.info(f"Stored {stored_count} candles for {symbol} (source: {provider_used})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {e}")
            return False
    
    async def add_symbol_to_collection(self, symbol: str):
        """Add a new symbol to the collection set"""
        self.popular_symbols.add(symbol.upper())
        logger.info(f"Added {symbol} to collection set")
    
    async def remove_symbol_from_collection(self, symbol: str):
        """Remove a symbol from the collection set"""
        self.popular_symbols.discard(symbol.upper())
        logger.info(f"Removed {symbol} from collection set")
    
    async def force_collect_symbol(self, symbol: str) -> bool:
        """Force immediate collection of a symbol's data"""
        logger.info(f"Force collecting data for {symbol}")
        return await self.collect_symbol_data(symbol, days_back=365, force_update=True)
    
    async def get_collection_status(self) -> dict:
        """Get status information about the data collector"""
        coverage_info = {}
        
        for symbol in list(self.popular_symbols)[:5]:  # Sample a few symbols
            try:
                coverage = await db_service.get_data_coverage(symbol)
                if coverage:
                    earliest, latest = coverage
                    coverage_info[symbol] = {
                        'earliest': earliest.isoformat(),
                        'latest': latest.isoformat(),
                        'days_coverage': (latest - earliest).days
                    }
            except Exception:
                pass
        
        return {
            'running': self.running,
            'symbols_tracked': len(self.popular_symbols),
            'collection_interval_seconds': self.collection_interval,
            'sample_coverage': coverage_info
        }

# Global data collector instance
data_collector = DataCollectorService()