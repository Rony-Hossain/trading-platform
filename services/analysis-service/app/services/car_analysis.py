"""
Cumulative Abnormal Return (CAR) Analysis Service for Event-Driven Strategy Layer

This module implements comprehensive CAR studies to identify optimal holding periods
and empirically-driven regime parameters for event-driven trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import asyncpg
from ..core.database import get_database_url

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Event types for CAR analysis"""
    EARNINGS = "earnings"
    FDA_APPROVAL = "fda_approval"
    MERGER_ACQUISITION = "merger_acquisition"
    DIVIDEND = "dividend"
    STOCK_SPLIT = "stock_split"
    ANALYST_UPGRADE = "analyst_upgrade"
    ANALYST_DOWNGRADE = "analyst_downgrade"
    NEWS_POSITIVE = "news_positive"
    NEWS_NEGATIVE = "news_negative"
    INSIDER_TRADING = "insider_trading"
    INSTITUTIONAL_FLOW = "institutional_flow"
    OPTIONS_FLOW = "options_flow"

class Sector(Enum):
    """Industry sectors for sector-specific analysis"""
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    ENERGY = "energy"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    INDUSTRIALS = "industrials"
    MATERIALS = "materials"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    TELECOMMUNICATIONS = "telecommunications"

@dataclass
class EventData:
    """Event data structure for CAR analysis"""
    symbol: str
    event_type: EventType
    event_date: datetime
    sector: Optional[Sector] = None
    event_magnitude: Optional[float] = None  # Event strength/significance
    pre_event_volume: Optional[float] = None
    event_description: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class CARResults:
    """CAR analysis results"""
    car_values: np.ndarray  # Cumulative abnormal returns over time
    optimal_holding_period: int  # Days for optimal return
    expected_return: float
    return_volatility: float
    skewness: float
    kurtosis: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float  # Percentage of positive returns
    profit_distribution: Dict[str, float]  # Percentiles of returns
    statistical_significance: Dict[str, float]  # T-stats, p-values
    regime_parameters: Dict[str, float]

class CARAnalyzer:
    """Cumulative Abnormal Return Analysis Engine"""
    
    def __init__(self):
        self.estimation_window = 252  # Trading days for market model estimation
        self.event_windows = [-10, -5, -1, 0, 1, 2, 5, 10, 20, 30, 60]  # Days around event
        self.min_observations = 50  # Minimum events needed for statistical significance
        
    async def calculate_car(
        self,
        events: List[EventData],
        price_data: pd.DataFrame,
        market_data: pd.DataFrame,
        event_type: EventType,
        sector: Optional[Sector] = None
    ) -> CARResults:
        """
        Calculate Cumulative Abnormal Returns for event type/sector combination
        
        Args:
            events: List of event data
            price_data: Stock price data (symbol, date, return)
            market_data: Market index data (date, return)
            event_type: Type of event to analyze
            sector: Optional sector filter
            
        Returns:
            CARResults with comprehensive analysis
        """
        logger.info(f"Computing CAR analysis for {event_type.value}, sector: {sector}")
        
        # Filter events
        filtered_events = [
            e for e in events 
            if e.event_type == event_type and (sector is None or e.sector == sector)
        ]
        
        if len(filtered_events) < self.min_observations:
            raise ValueError(f"Insufficient observations: {len(filtered_events)} < {self.min_observations}")
        
        abnormal_returns_matrix = []
        valid_events = []
        
        for event in filtered_events:
            try:
                # Calculate abnormal returns for this event
                abnormal_returns = await self._calculate_event_abnormal_returns(
                    event, price_data, market_data
                )
                if abnormal_returns is not None:
                    abnormal_returns_matrix.append(abnormal_returns)
                    valid_events.append(event)
            except Exception as e:
                logger.warning(f"Failed to process event {event.symbol} on {event.event_date}: {e}")
                continue
        
        if len(abnormal_returns_matrix) < self.min_observations:
            raise ValueError(f"Insufficient valid events after processing: {len(abnormal_returns_matrix)}")
        
        # Convert to numpy array for analysis
        ar_matrix = np.array(abnormal_returns_matrix)  # Shape: (n_events, n_days)
        
        # Calculate cumulative abnormal returns
        car_matrix = np.cumsum(ar_matrix, axis=1)
        car_mean = np.mean(car_matrix, axis=0)
        car_std = np.std(car_matrix, axis=0)
        
        # Find optimal holding period
        optimal_period_idx = np.argmax(car_mean)
        optimal_holding_period = self.event_windows[optimal_period_idx]
        
        # Calculate comprehensive statistics
        final_cars = car_matrix[:, optimal_period_idx]
        
        results = CARResults(
            car_values=car_mean,
            optimal_holding_period=optimal_holding_period,
            expected_return=np.mean(final_cars),
            return_volatility=np.std(final_cars),
            skewness=stats.skew(final_cars),
            kurtosis=stats.kurtosis(final_cars),
            sharpe_ratio=np.mean(final_cars) / np.std(final_cars) if np.std(final_cars) > 0 else 0,
            max_drawdown=self._calculate_max_drawdown(car_mean),
            hit_rate=np.sum(final_cars > 0) / len(final_cars),
            profit_distribution=self._calculate_profit_distribution(final_cars),
            statistical_significance=self._calculate_statistical_significance(final_cars),
            regime_parameters=self._derive_regime_parameters(car_matrix, valid_events)
        )
        
        logger.info(f"CAR analysis complete: Expected return={results.expected_return:.4f}, "
                   f"Optimal period={results.optimal_holding_period} days")
        
        return results
    
    async def _calculate_event_abnormal_returns(
        self,
        event: EventData,
        price_data: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Calculate abnormal returns for a single event"""
        
        # Get price data for the symbol
        symbol_data = price_data[price_data['symbol'] == event.symbol].copy()
        symbol_data = symbol_data.sort_values('date')
        
        # Find event date index
        event_date_data = symbol_data[symbol_data['date'] >= event.event_date]
        if event_date_data.empty:
            return None
            
        event_idx = event_date_data.index[0]
        
        # Define estimation and event windows
        estimation_start = event_idx - self.estimation_window - 10
        estimation_end = event_idx - 11
        
        if estimation_start < 0:
            return None
            
        # Get estimation period data
        estimation_data = symbol_data.iloc[estimation_start:estimation_end]
        if len(estimation_data) < self.estimation_window * 0.8:  # Need at least 80% of data
            return None
        
        # Merge with market data
        estimation_merged = estimation_data.merge(market_data, on='date', suffixes=('_stock', '_market'))
        if len(estimation_merged) < len(estimation_data) * 0.8:
            return None
        
        # Estimate market model (CAPM): R_stock = alpha + beta * R_market + epsilon
        X = estimation_merged['return_market'].values
        y = estimation_merged['return_stock'].values
        
        # Handle missing values
        valid_idx = ~(np.isnan(X) | np.isnan(y))
        if np.sum(valid_idx) < len(X) * 0.8:
            return None
            
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        # OLS regression
        beta = np.cov(X_clean, y_clean)[0, 1] / np.var(X_clean)
        alpha = np.mean(y_clean) - beta * np.mean(X_clean)
        
        # Calculate abnormal returns for event window
        abnormal_returns = []
        
        for day_offset in self.event_windows:
            event_day_idx = event_idx + day_offset
            
            if event_day_idx < 0 or event_day_idx >= len(symbol_data):
                abnormal_returns.append(0.0)
                continue
                
            # Get actual return
            actual_return = symbol_data.iloc[event_day_idx]['return']
            
            # Get market return for the same date
            event_date = symbol_data.iloc[event_day_idx]['date']
            market_return_data = market_data[market_data['date'] == event_date]
            
            if market_return_data.empty:
                abnormal_returns.append(0.0)
                continue
                
            market_return = market_return_data.iloc[0]['return']
            
            # Calculate expected return using market model
            expected_return = alpha + beta * market_return
            
            # Abnormal return = Actual - Expected
            abnormal_return = actual_return - expected_return
            abnormal_returns.append(abnormal_return)
        
        return np.array(abnormal_returns)
    
    def _calculate_max_drawdown(self, car_series: np.ndarray) -> float:
        """Calculate maximum drawdown from CAR series"""
        cumulative = np.maximum.accumulate(car_series)
        drawdown = cumulative - car_series
        return np.max(drawdown)
    
    def _calculate_profit_distribution(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate profit distribution percentiles"""
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        return {
            f"p{p}": np.percentile(returns, p) for p in percentiles
        }
    
    def _calculate_statistical_significance(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate statistical significance metrics"""
        n = len(returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # T-test against zero
        t_stat = mean_return / (std_return / np.sqrt(n)) if std_return > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "confidence_95_lower": mean_return - 1.96 * (std_return / np.sqrt(n)),
            "confidence_95_upper": mean_return + 1.96 * (std_return / np.sqrt(n)),
            "is_significant": p_value < 0.05
        }
    
    def _derive_regime_parameters(
        self,
        car_matrix: np.ndarray,
        events: List[EventData]
    ) -> Dict[str, float]:
        """Derive regime-specific trading parameters from CAR analysis"""
        
        final_returns = car_matrix[:, -1]  # Use final CAR values
        
        # Volatility-based position sizing
        volatility = np.std(final_returns)
        
        # Kelly criterion for position sizing
        win_rate = np.sum(final_returns > 0) / len(final_returns)
        avg_win = np.mean(final_returns[final_returns > 0]) if np.any(final_returns > 0) else 0
        avg_loss = np.mean(final_returns[final_returns < 0]) if np.any(final_returns < 0) else 0
        
        kelly_fraction = 0
        if avg_loss != 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * abs(avg_loss)) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Dynamic stop-loss based on distribution
        stop_loss_threshold = np.percentile(final_returns, 10)  # 10th percentile
        
        # Profit target based on distribution
        profit_target = np.percentile(final_returns, 75)  # 75th percentile
        
        # Entry timing confidence threshold
        confidence_threshold = 0.6 if win_rate > 0.55 else 0.7
        
        # Market regime sensitivity
        regime_sensitivity = min(volatility * 2, 1.0)  # Higher vol = more regime sensitive
        
        return {
            "position_size_kelly": kelly_fraction,
            "stop_loss_threshold": stop_loss_threshold,
            "profit_target": profit_target,
            "confidence_threshold": confidence_threshold,
            "regime_sensitivity": regime_sensitivity,
            "min_hold_period": max(1, self.event_windows[np.argmax(car_matrix.mean(axis=0))] // 2),
            "max_hold_period": min(30, self.event_windows[np.argmax(car_matrix.mean(axis=0))] * 2),
            "volatility_factor": volatility,
            "expected_alpha": np.mean(final_returns)
        }

class EventRegimeIdentifier:
    """Identifies market regimes for event-driven strategies"""
    
    def __init__(self, car_analyzer: CARAnalyzer):
        self.car_analyzer = car_analyzer
        self.regime_cache = {}
        
    async def identify_regimes_by_event_sector(
        self,
        historical_events: List[EventData],
        price_data: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Dict[Tuple[EventType, Optional[Sector]], CARResults]:
        """
        Identify optimal regimes for each event-type/sector combination
        
        Returns:
            Dictionary mapping (event_type, sector) to CAR analysis results
        """
        logger.info("Starting regime identification across event types and sectors")
        
        regimes = {}
        
        # Get unique event type/sector combinations
        combinations = set()
        for event in historical_events:
            combinations.add((event.event_type, event.sector))
        
        logger.info(f"Analyzing {len(combinations)} event-type/sector combinations")
        
        # Analyze each combination
        for event_type, sector in combinations:
            try:
                car_results = await self.car_analyzer.calculate_car(
                    historical_events, price_data, market_data, event_type, sector
                )
                regimes[(event_type, sector)] = car_results
                
                logger.info(f"Regime analysis complete for {event_type.value}/{sector}: "
                           f"Expected return = {car_results.expected_return:.4f}, "
                           f"Optimal holding = {car_results.optimal_holding_period} days")
                
            except Exception as e:
                logger.warning(f"Failed to analyze {event_type.value}/{sector}: {e}")
                continue
        
        logger.info(f"Regime identification complete: {len(regimes)} regimes identified")
        return regimes
    
    def get_regime_parameters(
        self,
        event_type: EventType,
        sector: Optional[Sector] = None
    ) -> Optional[Dict[str, float]]:
        """Get trading parameters for specific event type/sector"""
        key = (event_type, sector)
        if key in self.regime_cache:
            return self.regime_cache[key].regime_parameters
        return None
    
    async def update_regime_cache(self, regimes: Dict[Tuple[EventType, Optional[Sector]], CARResults]):
        """Update the regime cache with new analysis results"""
        self.regime_cache.update(regimes)
        
        # Store in database for persistence
        await self._persist_regimes(regimes)
    
    async def _persist_regimes(self, regimes: Dict[Tuple[EventType, Optional[Sector]], CARResults]):
        """Persist regime analysis to database"""
        try:
            conn = await asyncpg.connect(get_database_url())
            
            # Create table if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS event_regime_analysis (
                    id SERIAL PRIMARY KEY,
                    event_type VARCHAR(50) NOT NULL,
                    sector VARCHAR(50),
                    optimal_holding_period INTEGER,
                    expected_return FLOAT,
                    return_volatility FLOAT,
                    skewness FLOAT,
                    kurtosis FLOAT,
                    sharpe_ratio FLOAT,
                    hit_rate FLOAT,
                    regime_parameters JSONB,
                    statistical_significance JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(event_type, sector)
                )
            """)
            
            # Insert/update regime data
            for (event_type, sector), results in regimes.items():
                await conn.execute("""
                    INSERT INTO event_regime_analysis 
                    (event_type, sector, optimal_holding_period, expected_return, 
                     return_volatility, skewness, kurtosis, sharpe_ratio, hit_rate,
                     regime_parameters, statistical_significance)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (event_type, sector) 
                    DO UPDATE SET 
                        optimal_holding_period = EXCLUDED.optimal_holding_period,
                        expected_return = EXCLUDED.expected_return,
                        return_volatility = EXCLUDED.return_volatility,
                        skewness = EXCLUDED.skewness,
                        kurtosis = EXCLUDED.kurtosis,
                        sharpe_ratio = EXCLUDED.sharpe_ratio,
                        hit_rate = EXCLUDED.hit_rate,
                        regime_parameters = EXCLUDED.regime_parameters,
                        statistical_significance = EXCLUDED.statistical_significance,
                        created_at = CURRENT_TIMESTAMP
                """, 
                    event_type.value,
                    sector.value if sector else None,
                    results.optimal_holding_period,
                    results.expected_return,
                    results.return_volatility,
                    results.skewness,
                    results.kurtosis,
                    results.sharpe_ratio,
                    results.hit_rate,
                    results.regime_parameters,
                    results.statistical_significance
                )
            
            await conn.close()
            logger.info(f"Persisted {len(regimes)} regime analyses to database")
            
        except Exception as e:
            logger.error(f"Failed to persist regime analysis: {e}")

# Factory function for easy access
async def create_car_analyzer() -> CARAnalyzer:
    """Create and initialize CAR analyzer"""
    return CARAnalyzer()

async def create_regime_identifier() -> EventRegimeIdentifier:
    """Create and initialize regime identifier"""
    car_analyzer = await create_car_analyzer()
    return EventRegimeIdentifier(car_analyzer)