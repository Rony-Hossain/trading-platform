"""
Data Pipeline Module
Handles data collection, processing, and feature engineering for analysis
"""

import pandas as pd
import numpy as np
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

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
        self.feature_engineer = FeatureEngineer()
    
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