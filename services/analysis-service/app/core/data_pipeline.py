"""
Data Pipeline Module
Handles data collection, processing, and feature engineering for analysis
"""

import pandas as pd
import numpy as np
import asyncio
import aiohttp
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union, Any, Set
from dataclasses import dataclass
from ..models.options_ramp import OptionsRampAnalyzer
from ..services.data_leakage_audit import DataLeakageAuditor, AuditResult
import json

logger = logging.getLogger(__name__)

@dataclass
class CachedResponse:
    data: Any
    fetched_at: datetime


class FactorClient:
    """Fetches macro, options, sentiment, and fundamentals factors with simple caching."""

    def __init__(
        self,
        market_data_url: str = "http://localhost:8002",
        sentiment_service_url: str = "http://localhost:8004",
        fundamentals_service_url: str = "http://localhost:8005",
        cache_ttl_seconds: int = 300,
        request_timeout: float = 10.0,
    ) -> None:
        self.market_data_url = market_data_url.rstrip('/')
        self.sentiment_url = sentiment_service_url.rstrip('/')
        self.fundamentals_url = fundamentals_service_url.rstrip('/')
        self.cache_ttl = cache_ttl_seconds
        self.timeout = aiohttp.ClientTimeout(total=request_timeout)
        self._cache: Dict[str, CachedResponse] = {}

    def _cache_key(self, prefix: str, identifier: str) -> str:
        return f"{prefix}:{identifier}".lower()

    def _get_cached(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if not entry:
            return None
        if (datetime.utcnow() - entry.fetched_at).total_seconds() > self.cache_ttl:
            self._cache.pop(key, None)
            return None
        return entry.data

    def _set_cache(self, key: str, value: Any) -> None:
        self._cache[key] = CachedResponse(data=value, fetched_at=datetime.utcnow())

    async def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        if response.content_type == 'application/json':
                            return await response.json()
                        text = await response.text()
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            logger.warning("Non-JSON response from %s", url)
                            return None
                    logger.warning("Request to %s failed with status %s", url, response.status)
        except Exception as exc:
            logger.error("Request to %s failed: %s", url, exc)
        return None

    async def get_options_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        key = self._cache_key('options_metrics', symbol)
        cached = self._get_cached(key)
        if cached is not None:
            return cached
        url = f"{self.market_data_url}/options/{symbol}/metrics"
        data = await self._get(url)
        if data:
            self._set_cache(key, data)
        return data

    async def get_options_history(self, symbol: str, limit: int = 60) -> Optional[List[Dict[str, Any]]]:
        key = self._cache_key('options_history', f"{symbol}:{limit}")
        cached = self._get_cached(key)
        if cached is not None:
            return cached
        url = f"{self.market_data_url}/options/{symbol}/metrics/history"
        data = await self._get(url, params={'limit': limit})
        history: Optional[List[Dict[str, Any]]] = None
        if isinstance(data, list):
            history = data
        elif isinstance(data, dict):
            history = data.get('metrics') or data.get('history')
        if history:
            self._set_cache(key, history)
        return history

    async def get_macro_snapshot(self, factors: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        key = self._cache_key('macro_snapshot', ','.join(factors) if factors else 'all')
        cached = self._get_cached(key)
        if cached is not None:
            return cached
        url = f"{self.market_data_url}/factors/macro"
        params = {'factors': ','.join(factors)} if factors else None
        data = await self._get(url, params=params)
        if data:
            self._set_cache(key, data)
        return data

    async def get_macro_history(self, factor_key: str, lookback_days: int = 120) -> Optional[Dict[str, Any]]:
        """Fetch historical macro factor data for a given key."""
        cache_identifier = f"{factor_key}:{lookback_days}"
        key = self._cache_key('macro_history', cache_identifier)
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        url = f"{self.market_data_url}/factors/macro/{factor_key}/history"
        params = {'lookback_days': lookback_days}
        data = await self._get(url, params=params)
        if data:
            self._set_cache(key, data)
        return data

    async def get_sentiment_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        key = self._cache_key('sentiment_summary', symbol)
        cached = self._get_cached(key)
        if cached is not None:
            return cached
        url = f"{self.sentiment_url}/summary/{symbol}"
        data = await self._get(url)
        if data:
            self._set_cache(key, data)
        return data

    async def get_surprise_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        key = self._cache_key('surprise', symbol)
        cached = self._get_cached(key)
        if cached is not None:
            return cached
        url = f"{self.fundamentals_url}/surprise/{symbol}"
        data = await self._get(url)
        if data:
            self._set_cache(key, data)
        return data

    async def get_ownership_flow(self, symbol: str) -> Optional[Dict[str, Any]]:
        key = self._cache_key('ownership_flow', symbol)
        cached = self._get_cached(key)
        if cached is not None:
            return cached
        url = f"{self.fundamentals_url}/ownership/flow/{symbol}"
        data = await self._get(url)
        if data:
            self._set_cache(key, data)
        return data

    async def get_analyst_momentum(self, symbol: str) -> Optional[Dict[str, Any]]:
        key = self._cache_key('analyst_momentum', symbol)
        cached = self._get_cached(key)
        if cached is not None:
            return cached
        url = f"{self.fundamentals_url}/analysts/momentum/{symbol}"
        data = await self._get(url)
        if data:
            self._set_cache(key, data)
        return data


@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str = "unknown"

class DataCollector:
    """Collects market data from various sources"""
    
    def __init__(self, market_data_service_url: str = "http://localhost:8002"):
        self.market_data_url = market_data_service_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_historical_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Get historical data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = f"{self.market_data_url}/stocks/{symbol}/history"
            params = {"period": period}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_historical_data(data, symbol)
                else:
                    logger.error(f"Failed to get historical data for {symbol}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    async def get_current_price(self, symbol: str) -> Optional[MarketData]:
        """Get current price data for a symbol"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = f"{self.market_data_url}/stocks/{symbol}/price"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_current_data(data)
                else:
                    logger.error(f"Failed to get current price for {symbol}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None
    
    async def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, Optional[MarketData]]:
        """Get current data for multiple symbols"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = f"{self.market_data_url}/stocks/batch"
            
            async with self.session.post(
                url,
                json=symbols,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_batch_data(data)
                else:
                    logger.error(f"Failed to get batch data: {response.status}")
                    return {symbol: None for symbol in symbols}
                    
        except Exception as e:
            logger.error(f"Error fetching batch data: {e}")
            return {symbol: None for symbol in symbols}
    
    def _parse_historical_data(self, data: Dict, symbol: str) -> pd.DataFrame:
        """Parse historical data response into DataFrame"""
        try:
            if 'data' not in data or not data['data']:
                return pd.DataFrame()
            
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df['symbol'] = symbol
            
            # Ensure numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Error parsing historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _parse_current_data(self, data: Dict) -> Optional[MarketData]:
        """Parse current price data into MarketData object"""
        try:
            return MarketData(
                symbol=data['symbol'],
                timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                open=data.get('open', data['price']),
                high=data.get('high', data['price']),
                low=data.get('low', data['price']),
                close=data['price'],
                volume=data.get('volume', 0),
                source=data.get('source', 'unknown')
            )
        except Exception as e:
            logger.error(f"Error parsing current data: {e}")
            return None
    
    def _parse_batch_data(self, data: Dict) -> Dict[str, Optional[MarketData]]:
        """Parse batch response into MarketData objects"""
        result = {}
        try:
            for item in data.get('results', []):
                if item['status'] == 'success':
                    result[item['symbol']] = self._parse_current_data(item)
                else:
                    result[item['symbol']] = None
                    logger.warning(f"Failed to get data for {item['symbol']}: {item.get('error', 'Unknown error')}")
            
            return result
        except Exception as e:
            logger.error(f"Error parsing batch data: {e}")
            return {}

class FeatureEngineer:
    """Feature engineering for market data"""
    
    @staticmethod
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Time features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['dayofyear'] = df.index.dayofyear
        df['week'] = df.index.isocalendar().week
        df['quarter'] = df.index.quarter
        
        # Market session features
        df['is_monday'] = (df['dayofweek'] == 0).astype(int)
        df['is_friday'] = (df['dayofweek'] == 4).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        return df
    
    @staticmethod
    def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        df = df.copy()
        
        # Price features
        df['price_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['returns_1d'] = df['returns']
        df['returns_2d'] = df['close'].pct_change(2)
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_10d'] = df['close'].pct_change(10)
        df['returns_20d'] = df['close'].pct_change(20)
        
        # Log returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility (rolling standard deviation of returns)
        df['volatility_5d'] = df['returns'].rolling(5).std()
        df['volatility_10d'] = df['returns'].rolling(10).std()
        df['volatility_20d'] = df['returns'].rolling(20).std()
        
        # Price position features
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['open_position'] = (df['open'] - df['low']) / (df['high'] - df['low'])
        
        # Gap features
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_percent'] = df['gap'] / df['close'].shift(1)
        
        return df
    
    @staticmethod
    def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        df = df.copy()
        
        # Volume features
        df['volume_sma_5'] = df['volume'].rolling(5).mean()
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        
        df['volume_ratio_5'] = df['volume'] / df['volume_sma_5']
        df['volume_ratio_10'] = df['volume'] / df['volume_sma_10']
        df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']
        
        # Volume price trend
        df['vpt'] = (df['volume'] * df['returns']).cumsum()
        
        # Money flow features
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['money_flow'] = df['typical_price'] * df['volume']
        df['money_flow_ratio'] = df['money_flow'] / df['money_flow'].rolling(20).mean()
        
        return df
    
    @staticmethod
    def add_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """Add lagged features for specified columns"""
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    @staticmethod
    def add_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
        """Add rolling statistics features"""
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_mean_{window}'] = df[col].rolling(window).mean()
                    df[f'{col}_std_{window}'] = df[col].rolling(window).std()
                    df[f'{col}_min_{window}'] = df[col].rolling(window).min()
                    df[f'{col}_max_{window}'] = df[col].rolling(window).max()
                    df[f'{col}_median_{window}'] = df[col].rolling(window).median()
        
        return df

class DataPipeline:
    """Complete data processing pipeline"""
    
    def __init__(self, market_data_url: str = "http://localhost:8002"):
        self.collector = DataCollector(market_data_url)
        sentiment_url = os.getenv("SENTIMENT_SERVICE_URL", "http://localhost:8004")
        fundamentals_url = os.getenv("FUNDAMENTALS_SERVICE_URL", "http://localhost:8005")
        self.factor_client = FactorClient(
            market_data_url=market_data_url,
            sentiment_service_url=sentiment_url,
            fundamentals_service_url=fundamentals_url,
            cache_ttl_seconds=int(os.getenv("FACTOR_CACHE_TTL_SECONDS", "300")),
            request_timeout=float(os.getenv("FACTOR_REQUEST_TIMEOUT", "10")),
        )
        self.feature_engineer = FeatureEngineer()
        self.leakage_auditor = DataLeakageAuditor()
    
    async def prepare_data_for_analysis(self, symbol: str, period: str = "1y", 
                                      add_features: bool = True) -> Optional[pd.DataFrame]:
        """
        Complete data preparation pipeline
        
        Args:
            symbol: Stock symbol
            period: Time period for historical data
            add_features: Whether to add engineered features
            
        Returns:
            Prepared DataFrame ready for analysis
        """
        async with self.collector as collector:
            # Get historical data
            df = await collector.get_historical_data(symbol, period)
            
            if df is None or df.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Add engineered features if requested
            if add_features:
                df = self._add_all_features(df)

            # Clean data
            df = self._clean_data(df)

            df = await self._augment_with_external_features(df, symbol)

            logger.info(f"Prepared {len(df)} data points for {symbol}")
            return df
    
    def _add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all engineered features to the dataset"""
        df = self.feature_engineer.add_time_features(df)
        df = self.feature_engineer.add_price_features(df)
        df = self.feature_engineer.add_volume_features(df)
        
        # Add lag features for key metrics
        lag_columns = ['close', 'volume', 'returns']
        lags = [1, 2, 3, 5]
        df = self.feature_engineer.add_lag_features(df, lag_columns, lags)
        
        # Add rolling features
        rolling_columns = ['close', 'volume', 'returns']
        windows = [5, 10, 20]
        df = self.feature_engineer.add_rolling_features(df, rolling_columns, windows)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the data by handling missing values and outliers"""
        # Forward fill missing values for price data
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].fillna(method='ffill')
        
        # Fill volume with 0 if missing
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # For other columns, use forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove rows where all price data is still missing
        df = df.dropna(subset=price_cols, how='all')
        
        return df
    

    async def _augment_with_external_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Enrich the OHLCV frame with macro, options, sentiment, and fundamentals features.
        """
        if df is None or df.empty:
            return df

        df_aug = df.copy()
        df_aug.index = pd.to_datetime(df_aug.index)
        df_aug.sort_index(inplace=True)
        df_aug['__calendar_date'] = df_aug.index.normalize()

        symbol_normalized = symbol.upper()
        date_span = df_aug['__calendar_date'].max() - df_aug['__calendar_date'].min()
        date_span_days = int(date_span.days) if pd.notnull(date_span) else 30
        lookback_days = max(30, min(365, date_span_days + 5))

        macro_keys = ['VIX', 'US10Y', 'US02Y', 'EURUSD', 'WTI']

        fetch_tasks = {
            'options_history': asyncio.create_task(
                self.factor_client.get_options_history(symbol_normalized, limit=min(len(df_aug) + 30, 180))
            ),
            'macro_snapshot': asyncio.create_task(self.factor_client.get_macro_snapshot(macro_keys)),
            'sentiment_summary': asyncio.create_task(self.factor_client.get_sentiment_summary(symbol_normalized)),
            'surprise': asyncio.create_task(self.factor_client.get_surprise_analysis(symbol_normalized)),
            'ownership': asyncio.create_task(self.factor_client.get_ownership_flow(symbol_normalized)),
            'analyst': asyncio.create_task(self.factor_client.get_analyst_momentum(symbol_normalized)),
        }
        macro_history_tasks = {
            key: asyncio.create_task(self.factor_client.get_macro_history(key, lookback_days))
            for key in macro_keys
        }

        fetch_results: Dict[str, Any] = {}
        for name, task in fetch_tasks.items():
            try:
                fetch_results[name] = await task
            except Exception as exc:
                logger.error("Failed to fetch %s for %s: %s", name, symbol_normalized, exc)
                fetch_results[name] = None

        macro_histories: Dict[str, Any] = {}
        for key, task in macro_history_tasks.items():
            try:
                macro_histories[key] = await task
            except Exception as exc:
                logger.error("Failed to fetch macro history %s for %s: %s", key, symbol_normalized, exc)
                macro_histories[key] = None

        columns_to_ffill: Set[str] = set()
        metadata_datetime_columns: List[str] = []

        options_frame = self._build_options_feature_frame(fetch_results.get('options_history'))
        if not options_frame.empty:
            df_aug = df_aug.join(options_frame, on='__calendar_date')
            columns_to_ffill.update([col for col in options_frame.columns if col != 'options_last_as_of'])
            if 'options_last_as_of' in options_frame.columns:
                metadata_datetime_columns.append('options_last_as_of')

        macro_frame = self._build_macro_feature_frame(macro_histories)
        if not macro_frame.empty:
            df_aug = df_aug.join(macro_frame, on='__calendar_date')
            columns_to_ffill.update(macro_frame.columns.tolist())

        sentiment_frame = self._build_sentiment_feature_frame(fetch_results.get('sentiment_summary'))
        if not sentiment_frame.empty:
            df_aug = df_aug.join(sentiment_frame, on='__calendar_date')
            for column in sentiment_frame.columns:
                if column == 'sentiment_last_updated':
                    metadata_datetime_columns.append(column)
                else:
                    columns_to_ffill.add(column)

        surprise_frame = self._build_surprise_feature_frame(fetch_results.get('surprise'))
        if not surprise_frame.empty:
            df_aug = df_aug.join(surprise_frame, on='__calendar_date')
            columns_to_ffill.update([col for col in surprise_frame.columns if col != 'fund_surprise_event_date'])
            if 'fund_surprise_event_date' in surprise_frame.columns:
                metadata_datetime_columns.append('fund_surprise_event_date')

        analyst_frame = self._build_analyst_feature_frame(fetch_results.get('analyst'))
        if not analyst_frame.empty:
            df_aug = df_aug.join(analyst_frame, on='__calendar_date')
            columns_to_ffill.update([col for col in analyst_frame.columns if col != 'fund_analyst_analysis_end'])
            if 'fund_analyst_analysis_end' in analyst_frame.columns:
                metadata_datetime_columns.append('fund_analyst_analysis_end')

        ownership_frame = self._build_ownership_feature_frame(fetch_results.get('ownership'))
        if not ownership_frame.empty:
            df_aug = df_aug.join(ownership_frame, on='__calendar_date')
            columns_to_ffill.update([col for col in ownership_frame.columns if col != 'fund_ownership_analysis_date'])
            if 'fund_ownership_analysis_date' in ownership_frame.columns:
                metadata_datetime_columns.append('fund_ownership_analysis_date')

        if 'options_last_as_of' in df_aug.columns:
            df_aug['options_last_as_of'] = pd.to_datetime(df_aug['options_last_as_of'], errors='coerce')
            df_aug['options_data_age_days'] = (df_aug['__calendar_date'] - df_aug['options_last_as_of'].dt.normalize()).dt.days
            df_aug['options_data_age_days'] = df_aug['options_data_age_days'].where(df_aug['options_data_age_days'] >= 0)
            df_aug['options_data_is_stale'] = (df_aug['options_data_age_days'].fillna(0) > 5).astype(int)

        if 'sentiment_last_updated' in df_aug.columns:
            df_aug['sentiment_last_updated'] = pd.to_datetime(df_aug['sentiment_last_updated'], errors='coerce')
            df_aug['sentiment_data_age_days'] = (df_aug['__calendar_date'] - df_aug['sentiment_last_updated'].dt.normalize()).dt.days
            df_aug['sentiment_data_age_days'] = df_aug['sentiment_data_age_days'].where(df_aug['sentiment_data_age_days'] >= 0)
            df_aug['sentiment_data_is_stale'] = (df_aug['sentiment_data_age_days'].fillna(0) > 2).astype(int)

        if 'fund_surprise_event_date' in df_aug.columns:
            df_aug['fund_surprise_event_date'] = pd.to_datetime(df_aug['fund_surprise_event_date'], errors='coerce')
            df_aug['fund_surprise_days_since_event'] = (df_aug['__calendar_date'] - df_aug['fund_surprise_event_date'].dt.normalize()).dt.days
            df_aug['fund_surprise_days_since_event'] = df_aug['fund_surprise_days_since_event'].where(df_aug['fund_surprise_days_since_event'] >= 0)
            df_aug['fund_surprise_is_recent'] = (df_aug['fund_surprise_days_since_event'].fillna(np.inf) <= 30).astype(int)
            columns_to_ffill.add('fund_surprise_is_recent')

        if 'fund_analyst_analysis_end' in df_aug.columns:
            df_aug['fund_analyst_analysis_end'] = pd.to_datetime(df_aug['fund_analyst_analysis_end'], errors='coerce')
            df_aug['fund_analyst_days_since_analysis'] = (df_aug['__calendar_date'] - df_aug['fund_analyst_analysis_end'].dt.normalize()).dt.days
            df_aug['fund_analyst_days_since_analysis'] = df_aug['fund_analyst_days_since_analysis'].where(df_aug['fund_analyst_days_since_analysis'] >= 0)
            df_aug['fund_analyst_is_recent'] = (df_aug['fund_analyst_days_since_analysis'].fillna(np.inf) <= 30).astype(int)
            columns_to_ffill.add('fund_analyst_is_recent')

        if 'fund_ownership_analysis_date' in df_aug.columns:
            df_aug['fund_ownership_analysis_date'] = pd.to_datetime(df_aug['fund_ownership_analysis_date'], errors='coerce')
            df_aug['fund_ownership_days_since_analysis'] = (df_aug['__calendar_date'] - df_aug['fund_ownership_analysis_date'].dt.normalize()).dt.days
            df_aug['fund_ownership_days_since_analysis'] = df_aug['fund_ownership_days_since_analysis'].where(df_aug['fund_ownership_days_since_analysis'] >= 0)
            df_aug['fund_ownership_is_recent'] = (df_aug['fund_ownership_days_since_analysis'].fillna(np.inf) <= 90).astype(int)
            columns_to_ffill.add('fund_ownership_is_recent')

        drop_candidates = [col for col in metadata_datetime_columns if col in df_aug.columns]
        if drop_candidates:
            df_aug.drop(columns=drop_candidates, inplace=True)

        ffill_columns = [col for col in columns_to_ffill if col in df_aug.columns]
        if ffill_columns:
            df_aug[ffill_columns] = df_aug[ffill_columns].ffill()

        df_aug = self._add_interaction_features(df_aug)
        df_aug.drop(columns=['__calendar_date'], inplace=True)
        return df_aug


    @staticmethod
    def _build_options_feature_frame(history: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]) -> pd.DataFrame:
        if not history:
            return pd.DataFrame()
        if isinstance(history, dict):
            history_list = history.get('metrics') or history.get('history')
        else:
            history_list = history
        if not history_list:
            return pd.DataFrame()

        frame = pd.DataFrame(history_list)
        if frame.empty:
            return pd.DataFrame()

        as_of_col = next((candidate for candidate in ('as_of', 'timestamp', 'ts', 'created_at') if candidate in frame.columns), None)
        if as_of_col is None:
            return pd.DataFrame()

        frame['options_last_as_of'] = pd.to_datetime(frame[as_of_col], errors='coerce')
        frame = frame.dropna(subset=['options_last_as_of'])
        frame['__date'] = frame['options_last_as_of'].dt.normalize()

        if 'metadata' in frame.columns:
            frame = frame.drop(columns=['metadata'])

        rename_map = {
            'atm_iv': 'options_atm_iv',
            'implied_move_pct': 'options_implied_move_pct',
            'straddle_price': 'options_straddle_price',
            'put_call_volume_ratio': 'options_put_call_volume_ratio',
            'put_call_oi_ratio': 'options_put_call_oi_ratio',
            'iv_25d_call': 'options_iv_25d_call',
            'iv_25d_put': 'options_iv_25d_put',
            'iv_skew_25d': 'options_iv_skew_25d',
            'iv_skew_25d_pct': 'options_iv_skew_25d_pct',
            'call_volume': 'options_call_volume',
            'put_volume': 'options_put_volume',
            'call_open_interest': 'options_call_oi',
            'put_open_interest': 'options_put_oi',
            'underlying_price': 'options_underlying_price',
        }
        frame = frame.rename(columns=rename_map)

        keep_columns = ['__date', 'options_last_as_of'] + list(rename_map.values())
        keep_columns = [col for col in keep_columns if col in frame.columns]
        frame = frame[keep_columns]

        numeric_columns = [col for col in frame.columns if col not in ('__date', 'options_last_as_of')]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')

        frame = frame.sort_values('options_last_as_of')
        frame = frame.drop_duplicates('__date', keep='last')
        frame.set_index('__date', inplace=True)

        if 'options_call_volume' in frame.columns or 'options_put_volume' in frame.columns:
            call_volume = frame['options_call_volume'] if 'options_call_volume' in frame.columns else pd.Series(0, index=frame.index)
            put_volume = frame['options_put_volume'] if 'options_put_volume' in frame.columns else pd.Series(0, index=frame.index)
            frame['options_total_volume'] = call_volume.fillna(0) + put_volume.fillna(0)

        if 'options_call_oi' in frame.columns or 'options_put_oi' in frame.columns:
            call_oi = frame['options_call_oi'] if 'options_call_oi' in frame.columns else pd.Series(0, index=frame.index)
            put_oi = frame['options_put_oi'] if 'options_put_oi' in frame.columns else pd.Series(0, index=frame.index)
            frame['options_total_oi'] = call_oi.fillna(0) + put_oi.fillna(0)

        if 'options_atm_iv' in frame.columns:
            frame['options_atm_iv_change_5d'] = frame['options_atm_iv'].diff(5)
            frame['options_atm_iv_change_20d'] = frame['options_atm_iv'].diff(20)

        if 'options_implied_move_pct' in frame.columns:
            frame['options_implied_move_change_5d'] = frame['options_implied_move_pct'].diff(5)
            frame['options_implied_move_change_20d'] = frame['options_implied_move_pct'].diff(20)

        if 'options_total_volume' in frame.columns:
            rolling_mean = frame['options_total_volume'].rolling(20).mean()
            rolling_std = frame['options_total_volume'].rolling(20).std()
            frame['options_volume_zscore_20d'] = (frame['options_total_volume'] - rolling_mean) / (rolling_std + 1e-9)
            frame['options_volume_zscore_20d'] = frame['options_volume_zscore_20d'].replace([np.inf, -np.inf], np.nan)

        ramp_analyzer = OptionsRampAnalyzer()
        ramp_features = []
        if 'options_total_volume' in frame.columns:
            ramp_features.append(ramp_analyzer.compute(frame['options_total_volume'], 'options_total_volume'))
        if 'options_total_oi' in frame.columns:
            ramp_features.append(ramp_analyzer.compute(frame['options_total_oi'], 'options_total_oi'))

        for feature_frame in ramp_features:
            if feature_frame is not None and not feature_frame.empty:
                frame = frame.join(feature_frame, how='left')

        if {'options_total_volume_signal', 'options_total_oi_signal'}.issubset(frame.columns):
            frame['options_positioning_signal'] = (
                frame['options_total_volume_signal'].fillna(0) * 0.5 +
                frame['options_total_oi_signal'].fillna(0) * 0.5
            )
        elif 'options_total_volume_signal' in frame.columns:
            frame['options_positioning_signal'] = frame['options_total_volume_signal']
        elif 'options_total_oi_signal' in frame.columns:
            frame['options_positioning_signal'] = frame['options_total_oi_signal']

        all_numeric = [col for col in frame.columns if col != 'options_last_as_of']
        for column in all_numeric:
            frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)

        return frame


    @staticmethod
    def _build_macro_feature_frame(macro_histories: Dict[str, Any]) -> pd.DataFrame:
        if not macro_histories:
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        for key, payload in macro_histories.items():
            if not payload:
                continue
            history = payload.get('data') if isinstance(payload, dict) else None
            if not history:
                continue
            frame = pd.DataFrame(history)
            if frame.empty:
                continue
            timestamp_col = next((candidate for candidate in ('timestamp', 'ts', 'as_of') if candidate in frame.columns), None)
            if timestamp_col is None:
                continue
            frame['timestamp'] = pd.to_datetime(frame[timestamp_col], errors='coerce')
            frame = frame.dropna(subset=['timestamp'])
            frame['__date'] = frame['timestamp'].dt.normalize()
            column_name = f"macro_{key.lower()}"
            frame[column_name] = pd.to_numeric(frame.get('value'), errors='coerce')
            subset = frame[['__date', column_name]].dropna(subset=[column_name])
            if subset.empty:
                continue
            subset = subset.sort_values('__date').drop_duplicates('__date', keep='last').set_index('__date')
            frames.append(subset)

        if not frames:
            return pd.DataFrame()

        macro_df = pd.concat(frames, axis=1)
        macro_df.sort_index(inplace=True)

        if {'macro_us10y', 'macro_us02y'}.issubset(macro_df.columns):
            macro_df['macro_yield_curve'] = macro_df['macro_us10y'] - macro_df['macro_us02y']
            macro_df['macro_rate_slope_20d'] = macro_df['macro_yield_curve'].diff(20)

        if 'macro_vix' in macro_df.columns:
            macro_df['macro_vix_change_5d'] = macro_df['macro_vix'].diff(5)
            macro_df['macro_vix_zscore_20d'] = (macro_df['macro_vix'] - macro_df['macro_vix'].rolling(20).mean()) / (macro_df['macro_vix'].rolling(20).std() + 1e-9)
            macro_df['macro_vix_zscore_20d'] = macro_df['macro_vix_zscore_20d'].replace([np.inf, -np.inf], np.nan)

        if 'macro_wti' in macro_df.columns:
            macro_df['macro_wti_change_5d'] = macro_df['macro_wti'].diff(5)

        if 'macro_eurusd' in macro_df.columns:
            macro_df['macro_eurusd_change_5d'] = macro_df['macro_eurusd'].diff(5)

        macro_df.index.name = None
        return macro_df


    @staticmethod
    def _build_sentiment_feature_frame(summary: Optional[Dict[str, Any]]) -> pd.DataFrame:
        if not summary:
            return pd.DataFrame()
        timestamp = summary.get('summary_timestamp') or summary.get('analysis_timestamp') or summary.get('generated_at')
        if not timestamp:
            return pd.DataFrame()
        ts = pd.to_datetime(timestamp, errors='coerce')
        if pd.isna(ts):
            return pd.DataFrame()
        date_index = ts.normalize()
        score = DataPipeline._safe_to_float(summary.get('sentiment_score', summary.get('average_sentiment')))
        confidence = DataPipeline._safe_to_float(summary.get('confidence'))
        mentions = DataPipeline._safe_to_float(summary.get('recent_mentions', summary.get('total_mentions')))
        trend = DataPipeline._direction_to_score(summary.get('trending_direction', summary.get('sentiment_trend')))
        themes = summary.get('key_themes') or summary.get('top_keywords') or []
        theme_count = len(themes) if isinstance(themes, list) else 0
        data = {
            'sentiment_score': score,
            'sentiment_confidence': confidence,
            'sentiment_recent_mentions': mentions,
            'sentiment_trend_score': trend,
            'sentiment_theme_count': theme_count,
            'sentiment_last_updated': ts,
        }
        return pd.DataFrame([data], index=[date_index])


    @staticmethod
    def _build_surprise_feature_frame(surprise: Optional[Dict[str, Any]]) -> pd.DataFrame:
        if not surprise:
            return pd.DataFrame()
        event_date = surprise.get('event_date')
        if not event_date:
            return pd.DataFrame()
        event_ts = pd.to_datetime(event_date, errors='coerce')
        if pd.isna(event_ts):
            return pd.DataFrame()
        date_index = event_ts.normalize()
        metrics = surprise.get('surprise_metrics', {})
        consensus = surprise.get('consensus_quality', {})
        data = {
            'fund_surprise_percent': DataPipeline._safe_to_float(metrics.get('surprise_percent')),
            'fund_surprise_standardized': DataPipeline._safe_to_float(metrics.get('surprise_standardized')),
            'fund_surprise_score': DataPipeline._safe_to_float(metrics.get('surprise_score')),
            'fund_surprise_direction': DataPipeline._direction_to_score(metrics.get('surprise_direction')),
            'fund_consensus_confidence': DataPipeline._safe_to_float(consensus.get('consensus_confidence')),
            'fund_consensus_analyst_count': DataPipeline._safe_to_float(consensus.get('analyst_count')),
            'fund_surprise_event_date': event_ts,
        }
        return pd.DataFrame([data], index=[date_index])


    @staticmethod
    def _build_analyst_feature_frame(momentum: Optional[Dict[str, Any]]) -> pd.DataFrame:
        if not momentum:
            return pd.DataFrame()
        analysis_period = momentum.get('analysis_period', {})
        end_date = analysis_period.get('end_date') or momentum.get('analysis_date')
        if not end_date:
            return pd.DataFrame()
        end_ts = pd.to_datetime(end_date, errors='coerce')
        if pd.isna(end_ts):
            return pd.DataFrame()
        date_index = end_ts.normalize()
        revision_activity = momentum.get('revision_activity', {})
        momentum_scores = momentum.get('momentum_scores', {})
        price_targets = momentum.get('price_target_activity', {})
        event_flags = momentum.get('event_indicators', {})
        data = {
            'fund_analyst_total_revisions': DataPipeline._safe_to_float(revision_activity.get('total_revisions')),
            'fund_analyst_upgrades': DataPipeline._safe_to_float(revision_activity.get('upgrades')),
            'fund_analyst_downgrades': DataPipeline._safe_to_float(revision_activity.get('downgrades')),
            'fund_analyst_net_rating_changes': DataPipeline._safe_to_float(revision_activity.get('net_rating_changes')),
            'fund_analyst_revision_intensity': DataPipeline._safe_to_float(revision_activity.get('revision_intensity')),
            'fund_analyst_rating_momentum': DataPipeline._safe_to_float(momentum_scores.get('rating_momentum')),
            'fund_analyst_price_target_momentum': DataPipeline._safe_to_float(momentum_scores.get('price_target_momentum')),
            'fund_analyst_momentum_acceleration': DataPipeline._safe_to_float(momentum_scores.get('momentum_acceleration')),
            'fund_analyst_conviction': DataPipeline._safe_to_float(momentum_scores.get('conviction_score')),
            'fund_analyst_avg_price_target_change_pct': DataPipeline._safe_to_float(price_targets.get('average_change_pct')),
            'fund_analyst_pre_earnings_flag': int(bool(event_flags.get('pre_earnings_momentum'))),
            'fund_analyst_smart_money_following': int(bool(event_flags.get('smart_money_following'))),
            'fund_analyst_analysis_end': end_ts,
        }
        return pd.DataFrame([data], index=[date_index])


    @staticmethod
    def _build_ownership_feature_frame(flow: Optional[Dict[str, Any]]) -> pd.DataFrame:
        if not flow:
            return pd.DataFrame()
        analysis_date = flow.get('analysis_date')
        if not analysis_date:
            return pd.DataFrame()
        analysis_ts = pd.to_datetime(analysis_date, errors='coerce')
        if pd.isna(analysis_ts):
            return pd.DataFrame()
        date_index = analysis_ts.normalize()
        insider = flow.get('insider_flow', {})
        institutional = flow.get('institutional_flow', {})
        smart_money = flow.get('smart_money_signals', {})
        data = {
            'fund_insider_net_value': DataPipeline._safe_to_float(insider.get('net_value')),
            'fund_insider_net_shares': DataPipeline._safe_to_float(insider.get('net_shares')),
            'fund_institutional_net_value': DataPipeline._safe_to_float(institutional.get('net_value')),
            'fund_institutional_net_shares': DataPipeline._safe_to_float(institutional.get('net_shares')),
            'fund_cluster_buying_flag': int(bool(smart_money.get('cluster_buying'))),
            'fund_cluster_selling_flag': int(bool(smart_money.get('cluster_selling'))),
            'fund_smart_money_score': DataPipeline._safe_to_float(smart_money.get('smart_money_score')),
            'fund_smart_money_confidence': DataPipeline._safe_to_float(smart_money.get('confidence_level')),
            'fund_ownership_analysis_date': analysis_ts,
        }
        return pd.DataFrame([data], index=[date_index])


    @staticmethod
    def _direction_to_score(direction: Optional[Any]) -> Optional[float]:
        if direction is None:
            return None
        if isinstance(direction, bool):
            return 1.0 if direction else -1.0
        direction_str = str(direction).upper()
        mapping = {
            'UP': 1.0,
            'IMPROVING': 1.0,
            'POSITIVE': 1.0,
            'BULLISH': 1.0,
            'DOWN': -1.0,
            'DECLINING': -1.0,
            'NEGATIVE': -1.0,
            'BEARISH': -1.0,
            'STABLE': 0.0,
            'NEUTRAL': 0.0,
        }
        return mapping.get(direction_str, 0.0)


    @staticmethod
    def _safe_to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        returns_5d = df['returns_5d'] if 'returns_5d' in df.columns else pd.Series(0, index=df.index)
        macro_rate = df['macro_rate_slope_20d'] if 'macro_rate_slope_20d' in df.columns else pd.Series(0, index=df.index)
        df['interaction_momentum_rate'] = returns_5d.fillna(0) * macro_rate.fillna(0)

        iv_change = df['options_atm_iv_change_5d'] if 'options_atm_iv_change_5d' in df.columns else pd.Series(0, index=df.index)
        sentiment_score = df['sentiment_score'] if 'sentiment_score' in df.columns else pd.Series(0, index=df.index)
        df['interaction_iv_sentiment'] = iv_change.fillna(0) * sentiment_score.fillna(0)

        surprise_pct = df['fund_surprise_percent'] if 'fund_surprise_percent' in df.columns else pd.Series(0, index=df.index)
        df['interaction_surprise_sentiment'] = surprise_pct.fillna(0) * sentiment_score.fillna(0)

        if 'options_total_volume' in df.columns and 'volume' in df.columns:
            volume_ratio = df['options_total_volume'].fillna(0) / df['volume'].replace(0, np.nan)
            df['interaction_options_to_equity_volume'] = volume_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

        if 'sentiment_trend_score' in df.columns:
            df['interaction_sentiment_trend_momentum'] = df['sentiment_trend_score'].fillna(0) * returns_5d.fillna(0)

        interaction_columns = [col for col in [
            'interaction_momentum_rate',
            'interaction_iv_sentiment',
            'interaction_surprise_sentiment',
            'interaction_options_to_equity_volume' if 'interaction_options_to_equity_volume' in df.columns else None,
            'interaction_sentiment_trend_momentum' if 'interaction_sentiment_trend_momentum' in df.columns else None,
        ] if col]

        for column in interaction_columns:
            df[column] = df[column].replace([np.inf, -np.inf], np.nan).fillna(0)

        return df

    async def get_realtime_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time data with basic features"""
        async with self.collector as collector:
            current_data = await collector.get_current_price(symbol)
            
            if current_data is None:
                return None
            
            # Get some recent historical data for context
            df = await collector.get_historical_data(symbol, "5d")
            
            features = {
                'symbol': current_data.symbol,
                'current_price': current_data.close,
                'timestamp': current_data.timestamp.isoformat(),
                'source': current_data.source
            }
            
            if df is not None and not df.empty:
                # Add contextual features
                recent_close = df['close'].iloc[-5:] if len(df) >= 5 else df['close']
                features.update({
                    'price_change_5d': current_data.close - recent_close.iloc[0] if len(recent_close) > 0 else 0,
                    'volatility_5d': recent_close.pct_change().std() if len(recent_close) > 1 else 0,
                    'avg_volume_5d': df['volume'].iloc[-5:].mean() if 'volume' in df.columns else 0,
                })
            
            return features

    async def audit_data_leakage(self, 
                               symbol: str, 
                               period: str = "2y",
                               target_col: str = 'close') -> AuditResult:
        """
        Perform comprehensive data leakage audit on the data pipeline.
        
        Args:
            symbol: Stock symbol to audit
            period: Data period for audit
            target_col: Target column name
            
        Returns:
            Comprehensive audit results
        """
        logger.info(f"Starting data leakage audit for {symbol}")
        
        try:
            # Get processed data
            data = await self.prepare_data_for_analysis(symbol, period, add_features=True)
            
            if data is None or len(data) < 100:
                raise ValueError(f"Insufficient data for audit: {len(data) if data is not None else 0} samples")
            
            # Add target if not present
            if target_col not in data.columns and 'close' in data.columns:
                data[target_col] = data['close'].pct_change().shift(-1)  # Next day return
            
            # Identify feature categories for targeted auditing
            options_features = self._identify_options_features(data.columns)
            event_features = self._identify_event_features(data.columns)
            
            # Run comprehensive audit
            audit_result = await self.leakage_auditor.comprehensive_audit(
                data=data,
                target_col=target_col,
                options_features=options_features,
                event_features=event_features,
                preprocessing_info={
                    'symbol': symbol,
                    'period': period,
                    'pipeline_version': '1.0'
                }
            )
            
            logger.info(f"Audit completed for {symbol}: {len(audit_result.violations)} violations found")
            return audit_result
            
        except Exception as e:
            logger.error(f"Error during leakage audit for {symbol}: {e}")
            raise
    
    def _identify_options_features(self, columns: List[str]) -> List[str]:
        """Identify options-related features from column names."""
        options_features = []
        options_patterns = [
            r'.*iv.*', r'.*implied.*', r'.*option.*', r'.*volatility.*',
            r'.*gamma.*', r'.*delta.*', r'.*theta.*', r'.*vega.*', r'.*rho.*',
            r'.*skew.*', r'.*smile.*', r'.*surface.*'
        ]
        
        for col in columns:
            col_lower = col.lower()
            for pattern in options_patterns:
                if re.match(pattern, col_lower):
                    options_features.append(col)
                    break
        
        return options_features
    
    def _identify_event_features(self, columns: List[str]) -> List[str]:
        """Identify event-related features from column names."""
        event_features = []
        event_patterns = [
            r'.*earnings.*', r'.*event.*', r'.*announcement.*', 
            r'.*news.*', r'.*catalyst.*', r'.*surprise.*',
            r'.*guidance.*', r'.*split.*', r'.*dividend.*'
        ]
        
        for col in columns:
            col_lower = col.lower()
            for pattern in event_patterns:
                if re.match(pattern, col_lower):
                    event_features.append(col)
                    break
        
        return event_features
    
    async def automated_leakage_check(self, 
                                    symbol: str, 
                                    period: str = "1y",
                                    compliance_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Automated leakage check with pass/fail results.
        
        Args:
            symbol: Stock symbol to check
            period: Data period for check
            compliance_threshold: Minimum compliance score to pass
            
        Returns:
            Pass/fail results with summary
        """
        logger.info(f"Running automated leakage check for {symbol}")
        
        try:
            audit_result = await self.audit_data_leakage(symbol, period)
            
            # Determine pass/fail
            passed = audit_result.compliance_score >= compliance_threshold
            
            # Count critical/high severity violations
            critical_violations = [v for v in audit_result.violations if v.severity == 'critical']
            high_violations = [v for v in audit_result.violations if v.severity == 'high']
            
            # Auto-fail on critical violations
            if critical_violations:
                passed = False
            
            return {
                'symbol': symbol,
                'check_timestamp': datetime.now().isoformat(),
                'passed': passed,
                'compliance_score': audit_result.compliance_score,
                'compliance_threshold': compliance_threshold,
                'total_violations': len(audit_result.violations),
                'critical_violations': len(critical_violations),
                'high_violations': len(high_violations),
                'violations_by_type': audit_result.audit_summary['violations_by_type'],
                'key_issues': [v.description for v in critical_violations + high_violations[:3]],
                'recommendations': audit_result.recommendations[:5],
                'status': 'PASS' if passed else 'FAIL',
                'message': self._generate_check_message(passed, audit_result)
            }
            
        except Exception as e:
            logger.error(f"Automated leakage check failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'check_timestamp': datetime.now().isoformat(),
                'passed': False,
                'status': 'ERROR',
                'message': f'Check failed: {str(e)}',
                'error': str(e)
            }
    
    def _generate_check_message(self, passed: bool, audit_result: AuditResult) -> str:
        """Generate human-readable check message."""
        if passed:
            return f"✅ Data pipeline passed leakage check with {audit_result.compliance_score:.1%} compliance"
        else:
            critical_count = len([v for v in audit_result.violations if v.severity == 'critical'])
            if critical_count > 0:
                return f"❌ CRITICAL: {critical_count} critical leakage violations detected"
            else:
                return f"⚠️ Data pipeline failed with {audit_result.compliance_score:.1%} compliance (threshold: 80%)"





