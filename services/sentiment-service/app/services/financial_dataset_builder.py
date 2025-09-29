"""
Financial dataset preparation pipeline for multi-target transformer training.
Creates labeled datasets with multiple financial targets from sentiment and market data.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from sqlalchemy.orm import Session
import yfinance as yf
from textblob import TextBlob
import re

from .financial_transformer import FinancialDataPoint, FinancialTarget
from ..core.database import SentimentPost

logger = logging.getLogger(__name__)

class LabelingStrategy(str, Enum):
    FORWARD_LOOKING = "forward_looking"  # Use future price movement
    CONCURRENT = "concurrent"  # Use same-day price movement
    LAGGED = "lagged"  # Use past price movement

@dataclass
class DatasetConfig:
    lookback_days: int = 365  # Days of historical data
    min_sentiment_per_symbol: int = 50  # Minimum sentiment posts per symbol
    price_data_buffer_days: int = 10  # Extra days for price calculations
    volatility_window: int = 20  # Days for volatility calculation
    
    # Labeling thresholds
    price_direction_threshold: float = 0.02  # 2% threshold for up/down
    volatility_low_percentile: float = 33.33  # Bottom 33% = low volatility
    volatility_high_percentile: float = 66.67  # Top 33% = high volatility
    
    # Quality filters
    min_text_length: int = 10
    max_text_length: int = 512
    exclude_retweets: bool = True
    exclude_duplicates: bool = True

class FinancialDatasetBuilder:
    """Build training datasets with multiple financial targets"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.price_cache = {}  # Cache for price data
        
    async def build_dataset(self, db: Session, symbols: List[str], 
                          end_date: Optional[datetime] = None) -> List[FinancialDataPoint]:
        """
        Build a comprehensive dataset with multiple financial targets.
        
        Args:
            db: Database session
            symbols: List of stock symbols to include
            end_date: End date for data collection (defaults to now)
            
        Returns:
            List of FinancialDataPoint objects ready for training
        """
        try:
            if end_date is None:
                end_date = datetime.now()
            
            start_date = end_date - timedelta(days=self.config.lookback_days)
            
            logger.info(f"Building dataset for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
            
            all_data_points = []
            
            for symbol in symbols:
                logger.info(f"Processing symbol: {symbol}")
                
                # Get sentiment data
                sentiment_data = await self._get_sentiment_data(db, symbol, start_date, end_date)
                if len(sentiment_data) < self.config.min_sentiment_per_symbol:
                    logger.warning(f"Insufficient sentiment data for {symbol}: {len(sentiment_data)} posts")
                    continue
                
                # Get price data  
                price_data = await self._get_price_data(symbol, start_date, end_date)
                if price_data is None or len(price_data) < 20:
                    logger.warning(f"Insufficient price data for {symbol}")
                    continue
                
                # Create data points
                symbol_data_points = await self._create_data_points(sentiment_data, price_data, symbol)
                all_data_points.extend(symbol_data_points)
                
                logger.info(f"Created {len(symbol_data_points)} data points for {symbol}")
            
            # Apply quality filters
            filtered_data_points = self._apply_quality_filters(all_data_points)
            
            logger.info(f"Dataset creation complete: {len(filtered_data_points)} total data points")
            return filtered_data_points
            
        except Exception as e:
            logger.error(f"Dataset building failed: {e}")
            raise
    
    async def _get_sentiment_data(self, db: Session, symbol: str, 
                                start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get sentiment data from database"""
        try:
            query = db.query(SentimentPost).filter(
                SentimentPost.symbol == symbol,
                SentimentPost.post_timestamp >= start_date,
                SentimentPost.post_timestamp <= end_date
            ).order_by(SentimentPost.post_timestamp)
            
            posts = query.all()
            
            if not posts:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for post in posts:
                data.append({
                    'id': post.id,
                    'symbol': post.symbol,
                    'content': post.content,
                    'author': post.author,
                    'platform': post.platform,
                    'sentiment_score': post.sentiment_score,
                    'sentiment_label': post.sentiment_label,
                    'confidence': post.confidence,
                    'timestamp': post.post_timestamp,
                    'url': post.url,
                    'engagement': post.engagement
                })
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching sentiment data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _get_price_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get price data from Yahoo Finance with caching"""
        try:
            cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"
            if cache_key in self.price_cache:
                return self.price_cache[cache_key]
            
            # Add buffer for calculations
            buffer_start = start_date - timedelta(days=self.config.price_data_buffer_days)
            buffer_end = end_date + timedelta(days=self.config.price_data_buffer_days)
            
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=buffer_start, end=buffer_end)
            
            if df.empty:
                logger.warning(f"No price data available for {symbol}")
                return None
            
            # Calculate additional metrics
            df = self._calculate_price_metrics(df)
            
            # Cache the data
            self.price_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return None
    
    def _calculate_price_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional price metrics"""
        try:
            # Daily returns
            df['daily_return'] = df['Close'].pct_change()
            
            # Forward returns (for labeling)
            df['forward_1d_return'] = df['daily_return'].shift(-1)
            df['forward_2d_return'] = df['Close'].pct_change(periods=-2)
            df['forward_5d_return'] = df['Close'].pct_change(periods=-5)
            
            # Volatility (rolling standard deviation)
            df['volatility_20d'] = df['daily_return'].rolling(window=self.config.volatility_window).std()
            df['volatility_5d'] = df['daily_return'].rolling(window=5).std()
            
            # Price direction flags
            df['price_up_1d'] = (df['forward_1d_return'] > self.config.price_direction_threshold).astype(int)
            df['price_down_1d'] = (df['forward_1d_return'] < -self.config.price_direction_threshold).astype(int)
            df['price_flat_1d'] = ((abs(df['forward_1d_return']) <= self.config.price_direction_threshold)).astype(int)
            
            # Volume metrics
            df['volume_20d_avg'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_20d_avg']
            
            # High-low spread
            df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating price metrics: {e}")
            return df
    
    async def _create_data_points(self, sentiment_df: pd.DataFrame, 
                                price_df: pd.DataFrame, symbol: str) -> List[FinancialDataPoint]:
        """Create labeled data points from sentiment and price data"""
        data_points = []
        
        try:
            for _, sentiment_row in sentiment_df.iterrows():
                # Get corresponding price data
                sentiment_date = sentiment_row['timestamp'].date()
                
                # Find closest trading day
                price_row = self._find_closest_price_data(price_df, sentiment_date)
                if price_row is None:
                    continue
                
                # Create targets
                targets = self._create_financial_targets(price_row, price_df, sentiment_row)
                if not targets:
                    continue
                
                # Clean and preprocess text
                cleaned_text = self._preprocess_text(sentiment_row['content'])
                if not cleaned_text:
                    continue
                
                # Create data point
                data_point = FinancialDataPoint(
                    text=cleaned_text,
                    symbol=symbol,
                    timestamp=sentiment_row['timestamp'],
                    targets=targets,
                    metadata={
                        'sentiment_id': sentiment_row['id'],
                        'author': sentiment_row['author'],
                        'platform': sentiment_row['platform'],
                        'original_sentiment_score': sentiment_row['sentiment_score'],
                        'original_sentiment_label': sentiment_row['sentiment_label'],
                        'confidence': sentiment_row['confidence'],
                        'engagement': sentiment_row['engagement'],
                        'price_date': price_row.name.date().isoformat(),
                        'close_price': float(price_row['Close']),
                        'volume': int(price_row['Volume']) if not np.isnan(price_row['Volume']) else 0
                    }
                )
                
                data_points.append(data_point)
                
        except Exception as e:
            logger.error(f"Error creating data points for {symbol}: {e}")
        
        return data_points
    
    def _find_closest_price_data(self, price_df: pd.DataFrame, target_date) -> Optional[pd.Series]:
        """Find the closest trading day to the sentiment date"""
        try:
            price_df.index = pd.to_datetime(price_df.index)
            target_datetime = pd.to_datetime(target_date)
            
            # Find the closest date
            closest_idx = price_df.index.get_indexer([target_datetime], method='nearest')[0]
            
            if closest_idx == -1:
                return None
            
            return price_df.iloc[closest_idx]
            
        except Exception as e:
            logger.error(f"Error finding closest price data: {e}")
            return None
    
    def _create_financial_targets(self, price_row: pd.Series, price_df: pd.DataFrame, 
                                sentiment_row: pd.Series) -> Dict[FinancialTarget, int]:
        """Create target labels for all financial objectives"""
        targets = {}
        
        try:
            # 1. Sentiment target (traditional sentiment classification)
            sentiment_label = sentiment_row['sentiment_label']
            if sentiment_label == 'BULLISH':
                targets[FinancialTarget.SENTIMENT] = 2  # positive
            elif sentiment_label == 'BEARISH':
                targets[FinancialTarget.SENTIMENT] = 0  # negative  
            else:
                targets[FinancialTarget.SENTIMENT] = 1  # neutral
            
            # 2. Price direction target (next-day price movement)
            forward_return = price_row.get('forward_1d_return', np.nan)
            if not np.isnan(forward_return):
                if forward_return > self.config.price_direction_threshold:
                    targets[FinancialTarget.PRICE_DIRECTION] = 2  # up
                elif forward_return < -self.config.price_direction_threshold:
                    targets[FinancialTarget.PRICE_DIRECTION] = 0  # down
                else:
                    targets[FinancialTarget.PRICE_DIRECTION] = 1  # flat
            
            # 3. Volatility target (expected volatility level)
            volatility = price_row.get('volatility_20d', np.nan)
            if not np.isnan(volatility):
                # Calculate percentiles from the entire price series
                vol_series = price_df['volatility_20d'].dropna()
                if len(vol_series) > 0:
                    low_threshold = np.percentile(vol_series, self.config.volatility_low_percentile)
                    high_threshold = np.percentile(vol_series, self.config.volatility_high_percentile)
                    
                    if volatility <= low_threshold:
                        targets[FinancialTarget.VOLATILITY] = 0  # low
                    elif volatility >= high_threshold:
                        targets[FinancialTarget.VOLATILITY] = 2  # high
                    else:
                        targets[FinancialTarget.VOLATILITY] = 1  # medium
            
            # 4. Price magnitude target (regression - actual return magnitude)
            if not np.isnan(forward_return):
                targets[FinancialTarget.PRICE_MAGNITUDE] = float(abs(forward_return))
            
        except Exception as e:
            logger.error(f"Error creating financial targets: {e}")
        
        return targets
    
    def _preprocess_text(self, text: str) -> Optional[str]:
        """Clean and preprocess text content"""
        try:
            if not text or not isinstance(text, str):
                return None
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove mentions and hashtags for cleaner text (but keep the content)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag content
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Length filtering
            if len(text) < self.config.min_text_length or len(text) > self.config.max_text_length:
                return None
            
            # Remove common spam patterns
            spam_patterns = [
                r'\$\w+\s+to\s+the\s+moon',
                r'buy\s+now\s*!+',
                r'guaranteed\s+profit',
                r'pump\s+and\s+dump'
            ]
            
            for pattern in spam_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return None
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return None
    
    def _apply_quality_filters(self, data_points: List[FinancialDataPoint]) -> List[FinancialDataPoint]:
        """Apply quality filters to the dataset"""
        try:
            filtered_points = []
            
            # Track duplicates
            seen_texts = set()
            
            for point in data_points:
                # Skip if we've seen this text before (duplicate detection)
                if self.config.exclude_duplicates:
                    text_lower = point.text.lower()
                    if text_lower in seen_texts:
                        continue
                    seen_texts.add(text_lower)
                
                # Check if all targets are present
                required_targets = [
                    FinancialTarget.SENTIMENT,
                    FinancialTarget.PRICE_DIRECTION,
                    FinancialTarget.VOLATILITY,
                    FinancialTarget.PRICE_MAGNITUDE
                ]
                
                if all(target in point.targets for target in required_targets):
                    filtered_points.append(point)
            
            logger.info(f"Quality filtering: {len(data_points)} -> {len(filtered_points)} data points")
            return filtered_points
            
        except Exception as e:
            logger.error(f"Error applying quality filters: {e}")
            return data_points
    
    def create_train_val_test_split(self, data_points: List[FinancialDataPoint], 
                                  train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[List, List, List]:
        """Split dataset into train/validation/test sets by time"""
        try:
            # Sort by timestamp
            sorted_points = sorted(data_points, key=lambda x: x.timestamp)
            
            n_total = len(sorted_points)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_data = sorted_points[:n_train]
            val_data = sorted_points[n_train:n_train + n_val]
            test_data = sorted_points[n_train + n_val:]
            
            logger.info(f"Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
            
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error(f"Error splitting dataset: {e}")
            return data_points, [], []
    
    def get_dataset_statistics(self, data_points: List[FinancialDataPoint]) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        try:
            stats = {
                'total_samples': len(data_points),
                'symbols': list(set(point.symbol for point in data_points)),
                'date_range': {
                    'start': min(point.timestamp for point in data_points).isoformat(),
                    'end': max(point.timestamp for point in data_points).isoformat()
                },
                'target_distributions': {}
            }
            
            # Calculate target distributions
            for target in FinancialTarget:
                target_values = [point.targets.get(target) for point in data_points if target in point.targets]
                
                if target == FinancialTarget.PRICE_MAGNITUDE:  # Regression target
                    stats['target_distributions'][target.value] = {
                        'type': 'regression',
                        'count': len(target_values),
                        'mean': float(np.mean(target_values)),
                        'std': float(np.std(target_values)),
                        'min': float(np.min(target_values)),
                        'max': float(np.max(target_values))
                    }
                else:  # Classification targets
                    from collections import Counter
                    value_counts = Counter(target_values)
                    stats['target_distributions'][target.value] = {
                        'type': 'classification',
                        'class_counts': dict(value_counts),
                        'total': len(target_values)
                    }
            
            # Text length statistics
            text_lengths = [len(point.text) for point in data_points]
            stats['text_statistics'] = {
                'mean_length': float(np.mean(text_lengths)),
                'std_length': float(np.std(text_lengths)),
                'min_length': int(np.min(text_lengths)),
                'max_length': int(np.max(text_lengths))
            }
            
            # Platform distribution
            platforms = [point.metadata.get('platform', 'unknown') for point in data_points]
            platform_counts = {}
            for platform in platforms:
                platform_counts[platform] = platform_counts.get(platform, 0) + 1
            stats['platform_distribution'] = platform_counts
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating dataset statistics: {e}")
            return {'error': str(e)}
    
    def save_dataset(self, data_points: List[FinancialDataPoint], filepath: str):
        """Save dataset to file"""
        try:
            dataset_data = []
            for point in data_points:
                dataset_data.append({
                    'text': point.text,
                    'symbol': point.symbol,
                    'timestamp': point.timestamp.isoformat(),
                    'targets': {target.value: value for target, value in point.targets.items()},
                    'metadata': point.metadata
                })
            
            df = pd.DataFrame(dataset_data)
            df.to_pickle(filepath)
            logger.info(f"Dataset saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise
    
    def load_dataset(self, filepath: str) -> List[FinancialDataPoint]:
        """Load dataset from file"""
        try:
            df = pd.read_pickle(filepath)
            
            data_points = []
            for _, row in df.iterrows():
                targets = {FinancialTarget(k): v for k, v in row['targets'].items()}
                
                point = FinancialDataPoint(
                    text=row['text'],
                    symbol=row['symbol'],
                    timestamp=pd.to_datetime(row['timestamp']),
                    targets=targets,
                    metadata=row['metadata']
                )
                data_points.append(point)
            
            logger.info(f"Loaded {len(data_points)} data points from {filepath}")
            return data_points
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise