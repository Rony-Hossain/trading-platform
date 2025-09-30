"""
Signals Persistence Layer for TimescaleDB

This module provides comprehensive data persistence for fundamental signals
including Form 4 and Form 13F derived signals with TimescaleDB optimization.

Key Features:
1. TimescaleDB Integration: Optimized time-series storage for signals
2. Signal Versioning: Track signal changes and updates over time
3. Performance Optimization: Efficient queries with proper indexing
4. Data Retention: Automated cleanup of old signals
5. API Integration: FastAPI endpoints for signal access

Applications:
- Real-time signal storage and retrieval
- Historical signal analysis
- Performance tracking and backtesting
- Signal distribution to trading systems
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import uuid

import pandas as pd
import asyncpg
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB
import redis

from app.core.form4_clustering import InsiderSignal, Form4Clusterer
from app.core.form13f_aggregation import AggregatedSignal, Form13FAggregator

logger = logging.getLogger(__name__)


class SignalsPersistence:
    """
    Persistence layer for fundamental signals with TimescaleDB optimization.
    """
    
    def __init__(self, 
                 database_url: str,
                 redis_url: Optional[str] = None,
                 signal_retention_days: int = 365):
        """
        Initialize signals persistence.
        
        Parameters:
        - database_url: PostgreSQL/TimescaleDB connection URL
        - redis_url: Optional Redis URL for caching
        - signal_retention_days: Days to retain signals
        """
        self.database_url = database_url
        self.redis_url = redis_url
        self.signal_retention_days = signal_retention_days
        
        # Database connections
        self.engine = create_async_engine(database_url)
        self.async_session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        
        # Redis for caching
        self.redis_client = None
        if redis_url:
            self.redis_client = redis.from_url(redis_url)
        
        # Table definitions
        self._define_tables()
    
    def _define_tables(self):
        """Define database table schemas."""
        self.metadata = MetaData()
        
        # Form 4 insider signals table
        self.form4_signals = Table(
            'form4_signals',
            self.metadata,
            Column('signal_id', String(50), primary_key=True),
            Column('ticker', String(10), index=True),
            Column('signal_type', String(20), index=True),
            Column('signal_strength', Float),
            Column('confidence', Float),
            Column('generated_date', DateTime, index=True),
            Column('expiry_date', DateTime, index=True),
            Column('contributing_filings', JSONB),
            Column('cluster_analysis', JSONB),
            Column('metadata', JSONB),
            Column('created_at', DateTime, default=datetime.utcnow),
            Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
            Column('is_active', Boolean, default=True, index=True)
        )
        
        # Form 13F aggregated signals table
        self.form13f_signals = Table(
            'form13f_signals',
            self.metadata,
            Column('signal_id', String(50), primary_key=True),
            Column('cusip', String(9), index=True),
            Column('ticker', String(10), index=True),
            Column('security_name', String(200)),
            Column('signal_direction', String(20), index=True),
            Column('signal_strength', Float),
            Column('confidence_level', Float),
            Column('signal_date', DateTime, index=True),
            Column('total_institutions', Integer),
            Column('buying_institutions', Integer),
            Column('selling_institutions', Integer),
            Column('net_flow_value', Float),
            Column('net_flow_shares', Integer),
            Column('smart_money_score', Float),
            Column('consensus_score', Float),
            Column('contributing_changes', JSONB),
            Column('created_at', DateTime, default=datetime.utcnow),
            Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
            Column('is_active', Boolean, default=True, index=True)
        )
        
        # Signal performance tracking
        self.signal_performance = Table(
            'signal_performance',
            self.metadata,
            Column('performance_id', String(50), primary_key=True),
            Column('signal_id', String(50), index=True),
            Column('signal_type', String(20), index=True),
            Column('ticker', String(10), index=True),
            Column('entry_date', DateTime, index=True),
            Column('exit_date', DateTime),
            Column('entry_price', Float),
            Column('exit_price', Float),
            Column('return_1d', Float),
            Column('return_5d', Float),
            Column('return_21d', Float),
            Column('return_63d', Float),
            Column('max_return', Float),
            Column('min_return', Float),
            Column('volatility', Float),
            Column('benchmark_return_1d', Float),
            Column('benchmark_return_5d', Float),
            Column('benchmark_return_21d', Float),
            Column('benchmark_return_63d', Float),
            Column('alpha_1d', Float),
            Column('alpha_5d', Float),
            Column('alpha_21d', Float),
            Column('alpha_63d', Float),
            Column('created_at', DateTime, default=datetime.utcnow)
        )
        
        # Signal subscriptions for API access
        self.signal_subscriptions = Table(
            'signal_subscriptions',
            self.metadata,
            Column('subscription_id', String(50), primary_key=True),
            Column('user_id', String(50), index=True),
            Column('signal_types', JSONB),  # List of signal types
            Column('tickers', JSONB),       # List of tickers
            Column('min_strength', Float),
            Column('min_confidence', Float),
            Column('webhook_url', String(500)),
            Column('is_active', Boolean, default=True, index=True),
            Column('created_at', DateTime, default=datetime.utcnow),
            Column('last_notified', DateTime)
        )
    
    async def initialize_database(self):
        """Initialize database tables and TimescaleDB hypertables."""
        try:
            async with self.engine.begin() as conn:
                # Create tables
                await conn.run_sync(self.metadata.create_all)
                
                # Create TimescaleDB hypertables for time-series optimization
                hypertable_queries = [
                    """
                    SELECT create_hypertable('form4_signals', 'generated_date', 
                                           chunk_time_interval => INTERVAL '1 month',
                                           if_not_exists => TRUE);
                    """,
                    """
                    SELECT create_hypertable('form13f_signals', 'signal_date', 
                                           chunk_time_interval => INTERVAL '1 month',
                                           if_not_exists => TRUE);
                    """,
                    """
                    SELECT create_hypertable('signal_performance', 'entry_date', 
                                           chunk_time_interval => INTERVAL '1 month',
                                           if_not_exists => TRUE);
                    """
                ]
                
                for query in hypertable_queries:
                    try:
                        await conn.execute(text(query))
                        logger.info("Created TimescaleDB hypertable")
                    except Exception as e:
                        logger.warning(f"Hypertable creation failed (may already exist): {e}")
                
                # Create indexes for better performance
                index_queries = [
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_form4_signals_ticker_date ON form4_signals (ticker, generated_date DESC);",
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_form4_signals_strength ON form4_signals (signal_strength DESC) WHERE is_active = true;",
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_form13f_signals_ticker_date ON form13f_signals (ticker, signal_date DESC);",
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_form13f_signals_strength ON form13f_signals (signal_strength DESC) WHERE is_active = true;",
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signal_performance_ticker_date ON signal_performance (ticker, entry_date DESC);",
                ]
                
                for query in index_queries:
                    try:
                        await conn.execute(text(query))
                    except Exception as e:
                        logger.warning(f"Index creation failed: {e}")
                
                logger.info("Database initialization completed")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def store_form4_signals(self, signals: List[InsiderSignal]) -> bool:
        """
        Store Form 4 insider signals.
        
        Parameters:
        - signals: List of insider signals to store
        
        Returns:
        - Success status
        """
        if not signals:
            return True
        
        try:
            async with self.async_session() as session:
                # Prepare data for bulk insert
                signal_data = []
                
                for signal in signals:
                    signal_dict = signal.to_dict()
                    
                    # Convert datetime strings back to datetime objects
                    signal_dict['generated_date'] = datetime.fromisoformat(signal_dict['generated_date'])
                    signal_dict['expiry_date'] = datetime.fromisoformat(signal_dict['expiry_date'])
                    
                    signal_data.append(signal_dict)
                
                # Bulk insert
                query = self.form4_signals.insert()
                await session.execute(query, signal_data)
                await session.commit()
                
                # Update cache
                if self.redis_client:
                    await self._update_signal_cache('form4', signals)
                
                logger.info(f"Stored {len(signals)} Form 4 signals")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store Form 4 signals: {e}")
            return False
    
    async def store_form13f_signals(self, signals: List[AggregatedSignal]) -> bool:
        """
        Store Form 13F aggregated signals.
        
        Parameters:
        - signals: List of aggregated signals to store
        
        Returns:
        - Success status
        """
        if not signals:
            return True
        
        try:
            async with self.async_session() as session:
                # Prepare data for bulk insert
                signal_data = []
                
                for signal in signals:
                    signal_dict = signal.to_dict()
                    
                    # Convert datetime strings back to datetime objects
                    signal_dict['signal_date'] = datetime.fromisoformat(signal_dict['signal_date'])
                    
                    # Generate unique signal ID
                    signal_dict['signal_id'] = f"13f_{signal.cusip}_{signal.signal_date.strftime('%Y%m%d')}"
                    
                    signal_data.append(signal_dict)
                
                # Bulk insert
                query = self.form13f_signals.insert()
                await session.execute(query, signal_data)
                await session.commit()
                
                # Update cache
                if self.redis_client:
                    await self._update_signal_cache('form13f', signals)
                
                logger.info(f"Stored {len(signals)} Form 13F signals")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store Form 13F signals: {e}")
            return False
    
    async def get_active_signals(self, 
                                signal_type: str,
                                ticker: Optional[str] = None,
                                min_strength: float = 0.0,
                                min_confidence: float = 0.0,
                                limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve active signals with filtering.
        
        Parameters:
        - signal_type: 'form4' or 'form13f'
        - ticker: Optional ticker filter
        - min_strength: Minimum signal strength
        - min_confidence: Minimum confidence level
        - limit: Maximum number of results
        
        Returns:
        - List of signal dictionaries
        """
        try:
            # Try cache first
            cache_key = f"signals:{signal_type}:{ticker}:{min_strength}:{min_confidence}:{limit}"
            
            if self.redis_client:
                cached_signals = self.redis_client.get(cache_key)
                if cached_signals:
                    return json.loads(cached_signals)
            
            # Query database
            if signal_type == 'form4':
                table = self.form4_signals
                strength_col = 'signal_strength'
                confidence_col = 'confidence'
                date_col = 'generated_date'
            elif signal_type == 'form13f':
                table = self.form13f_signals
                strength_col = 'signal_strength'
                confidence_col = 'confidence_level'
                date_col = 'signal_date'
            else:
                raise ValueError(f"Unknown signal type: {signal_type}")
            
            query = table.select().where(
                table.c.is_active == True
            ).where(
                table.c[strength_col] >= min_strength
            ).where(
                table.c[confidence_col] >= min_confidence
            )
            
            if ticker:
                query = query.where(table.c.ticker == ticker)
            
            query = query.order_by(table.c[date_col].desc()).limit(limit)
            
            async with self.async_session() as session:
                result = await session.execute(query)
                signals = [dict(row) for row in result.fetchall()]
            
            # Cache results
            if self.redis_client:
                self.redis_client.setex(cache_key, 300, json.dumps(signals, default=str))  # 5 min cache
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to retrieve {signal_type} signals: {e}")
            return []
    
    async def update_signal_performance(self, 
                                      signal_id: str,
                                      ticker: str,
                                      signal_type: str,
                                      entry_date: datetime,
                                      stock_prices: pd.Series,
                                      benchmark_prices: pd.Series) -> bool:
        """
        Update signal performance metrics.
        
        Parameters:
        - signal_id: Signal identifier
        - ticker: Stock ticker
        - signal_type: Type of signal
        - entry_date: Signal entry date
        - stock_prices: Stock price series
        - benchmark_prices: Benchmark price series
        
        Returns:
        - Success status
        """
        try:
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                entry_date, stock_prices, benchmark_prices
            )
            
            if not performance_metrics:
                return False
            
            # Prepare performance record
            performance_data = {
                'performance_id': str(uuid.uuid4()),
                'signal_id': signal_id,
                'signal_type': signal_type,
                'ticker': ticker,
                'entry_date': entry_date,
                **performance_metrics
            }
            
            async with self.async_session() as session:
                query = self.signal_performance.insert()
                await session.execute(query, [performance_data])
                await session.commit()
            
            logger.debug(f"Updated performance for signal {signal_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update signal performance: {e}")
            return False
    
    def _calculate_performance_metrics(self, 
                                     entry_date: datetime,
                                     stock_prices: pd.Series,
                                     benchmark_prices: pd.Series) -> Optional[Dict[str, float]]:
        """Calculate performance metrics for signal."""
        try:
            # Find entry price
            entry_idx = stock_prices.index.searchsorted(entry_date)
            if entry_idx >= len(stock_prices):
                return None
            
            entry_price = stock_prices.iloc[entry_idx]
            
            # Calculate returns for different periods
            periods = [1, 5, 21, 63]  # 1 day, 1 week, 1 month, 3 months
            metrics = {'entry_price': entry_price}
            
            for days in periods:
                end_idx = min(entry_idx + days, len(stock_prices) - 1)
                
                if end_idx > entry_idx:
                    exit_price = stock_prices.iloc[end_idx]
                    stock_return = (exit_price - entry_price) / entry_price
                    
                    # Benchmark return
                    bench_entry = benchmark_prices.iloc[entry_idx]
                    bench_exit = benchmark_prices.iloc[end_idx]
                    bench_return = (bench_exit - bench_entry) / bench_entry
                    
                    # Alpha (excess return)
                    alpha = stock_return - bench_return
                    
                    metrics[f'return_{days}d'] = stock_return
                    metrics[f'benchmark_return_{days}d'] = bench_return
                    metrics[f'alpha_{days}d'] = alpha
            
            # Calculate max/min returns and volatility
            future_prices = stock_prices.iloc[entry_idx:min(entry_idx + 63, len(stock_prices))]
            future_returns = (future_prices - entry_price) / entry_price
            
            metrics['max_return'] = future_returns.max()
            metrics['min_return'] = future_returns.min()
            metrics['volatility'] = future_returns.std()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance calculation failed: {e}")
            return None
    
    async def get_signal_performance_stats(self, 
                                         signal_type: Optional[str] = None,
                                         ticker: Optional[str] = None,
                                         days_back: int = 90) -> Dict[str, Any]:
        """
        Get aggregated signal performance statistics.
        
        Parameters:
        - signal_type: Optional signal type filter
        - ticker: Optional ticker filter
        - days_back: Days to look back for statistics
        
        Returns:
        - Performance statistics
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            query = self.signal_performance.select().where(
                self.signal_performance.c.entry_date >= cutoff_date
            )
            
            if signal_type:
                query = query.where(self.signal_performance.c.signal_type == signal_type)
            
            if ticker:
                query = query.where(self.signal_performance.c.ticker == ticker)
            
            async with self.async_session() as session:
                result = await session.execute(query)
                performance_data = [dict(row) for row in result.fetchall()]
            
            if not performance_data:
                return {}
            
            df = pd.DataFrame(performance_data)
            
            # Calculate statistics
            stats = {
                'total_signals': len(df),
                'avg_return_1d': df['return_1d'].mean(),
                'avg_return_5d': df['return_5d'].mean(),
                'avg_return_21d': df['return_21d'].mean(),
                'avg_return_63d': df['return_63d'].mean(),
                'avg_alpha_1d': df['alpha_1d'].mean(),
                'avg_alpha_5d': df['alpha_5d'].mean(),
                'avg_alpha_21d': df['alpha_21d'].mean(),
                'avg_alpha_63d': df['alpha_63d'].mean(),
                'win_rate_1d': (df['return_1d'] > 0).mean(),
                'win_rate_5d': (df['return_5d'] > 0).mean(),
                'win_rate_21d': (df['return_21d'] > 0).mean(),
                'win_rate_63d': (df['return_63d'] > 0).mean(),
                'avg_volatility': df['volatility'].mean(),
                'max_return': df['max_return'].max(),
                'min_return': df['min_return'].min()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {}
    
    async def cleanup_expired_signals(self) -> int:
        """
        Clean up expired and old signals.
        
        Returns:
        - Number of signals cleaned up
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.signal_retention_days)
            
            async with self.async_session() as session:
                # Mark Form 4 signals as inactive
                form4_query = self.form4_signals.update().where(
                    self.form4_signals.c.expiry_date < datetime.utcnow()
                ).values(is_active=False)
                
                form4_result = await session.execute(form4_query)
                
                # Mark old Form 13F signals as inactive
                form13f_query = self.form13f_signals.update().where(
                    self.form13f_signals.c.signal_date < cutoff_date
                ).values(is_active=False)
                
                form13f_result = await session.execute(form13f_query)
                
                await session.commit()
                
                total_cleaned = form4_result.rowcount + form13f_result.rowcount
                
                logger.info(f"Cleaned up {total_cleaned} expired signals")
                return total_cleaned
                
        except Exception as e:
            logger.error(f"Signal cleanup failed: {e}")
            return 0
    
    async def _update_signal_cache(self, signal_type: str, signals: List) -> bool:
        """Update Redis cache with new signals."""
        if not self.redis_client:
            return False
        
        try:
            # Cache latest signals by ticker
            ticker_signals = {}
            
            for signal in signals:
                ticker = signal.ticker if hasattr(signal, 'ticker') else signal.get('ticker')
                if ticker:
                    if ticker not in ticker_signals:
                        ticker_signals[ticker] = []
                    ticker_signals[ticker].append(signal.to_dict() if hasattr(signal, 'to_dict') else signal)
            
            # Update cache for each ticker
            for ticker, ticker_signal_list in ticker_signals.items():
                cache_key = f"latest_signals:{signal_type}:{ticker}"
                self.redis_client.setex(cache_key, 1800, json.dumps(ticker_signal_list, default=str))  # 30 min cache
            
            # Update global latest signals cache
            global_cache_key = f"latest_signals:{signal_type}:all"
            all_signals = [s.to_dict() if hasattr(s, 'to_dict') else s for s in signals[-50:]]  # Last 50 signals
            self.redis_client.setex(global_cache_key, 1800, json.dumps(all_signals, default=str))
            
            return True
            
        except Exception as e:
            logger.error(f"Cache update failed: {e}")
            return False
    
    async def get_signal_subscriptions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get signal subscriptions for a user."""
        try:
            query = self.signal_subscriptions.select().where(
                self.signal_subscriptions.c.user_id == user_id
            ).where(
                self.signal_subscriptions.c.is_active == True
            )
            
            async with self.async_session() as session:
                result = await session.execute(query)
                return [dict(row) for row in result.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get subscriptions: {e}")
            return []
    
    async def create_signal_subscription(self, 
                                       user_id: str,
                                       signal_types: List[str],
                                       tickers: List[str],
                                       min_strength: float = 50.0,
                                       min_confidence: float = 0.5,
                                       webhook_url: Optional[str] = None) -> str:
        """Create a new signal subscription."""
        try:
            subscription_id = str(uuid.uuid4())
            
            subscription_data = {
                'subscription_id': subscription_id,
                'user_id': user_id,
                'signal_types': signal_types,
                'tickers': tickers,
                'min_strength': min_strength,
                'min_confidence': min_confidence,
                'webhook_url': webhook_url
            }
            
            async with self.async_session() as session:
                query = self.signal_subscriptions.insert()
                await session.execute(query, [subscription_data])
                await session.commit()
            
            logger.info(f"Created subscription {subscription_id} for user {user_id}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Failed to create subscription: {e}")
            raise