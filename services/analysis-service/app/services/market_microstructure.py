"""
Market Microstructure Analysis for Event-Driven Trading

This module provides sophisticated market microstructure analysis including:
- Order flow analysis and imbalance detection
- Bid-ask spread dynamics and liquidity metrics
- Volume profile analysis and VWAP calculations
- Price impact modeling and execution cost estimation
- Market maker behavior detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
from scipy import stats
from scipy.optimize import minimize
import asyncpg
from ..core.database import get_database_url
from .car_analysis import EventType, Sector

logger = logging.getLogger(__name__)

@dataclass
class OrderFlowData:
    """Order flow and market microstructure data"""
    timestamp: datetime
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    trade_direction: Optional[int] = None  # 1=buy, -1=sell, 0=unknown
    market_maker_presence: Optional[float] = None
    spread_bps: Optional[float] = None

@dataclass
class LiquidityMetrics:
    """Comprehensive liquidity analysis metrics"""
    bid_ask_spread: float
    spread_bps: float
    quoted_spread: float
    effective_spread: float
    realized_spread: float
    price_impact: float
    market_depth: float
    order_imbalance: float
    kyle_lambda: float  # Price impact coefficient
    amihud_illiquidity: float
    roll_measure: float  # Roll's bid-ask bounce measure
    volume_weighted_spread: float
    relative_tick_size: float

@dataclass
class VolumeProfile:
    """Volume profile analysis results"""
    price_levels: np.ndarray
    volume_by_price: np.ndarray
    poc: float  # Point of Control (highest volume price)
    value_area_high: float
    value_area_low: float
    value_area_volume_pct: float
    developing_poc: float
    volume_imbalance_areas: List[Tuple[float, float]]  # (price_low, price_high) areas

@dataclass
class MicrostructureRegime:
    """Market microstructure regime characteristics"""
    regime_id: str
    liquidity_level: str  # "high", "medium", "low"
    spread_regime: str  # "tight", "normal", "wide"
    volatility_regime: str  # "low", "normal", "high"
    order_flow_regime: str  # "balanced", "buy_pressure", "sell_pressure"
    typical_spread_bps: float
    typical_market_depth: float
    typical_price_impact: float
    optimal_execution_strategy: Dict[str, Any]

class MicrostructureAnalyzer:
    """Advanced market microstructure analysis engine"""
    
    def __init__(self):
        self.tick_size = 0.01  # Standard tick size
        self.value_area_percentage = 0.70  # 70% for value area calculation
        
    async def analyze_order_flow(
        self,
        order_flow_data: List[OrderFlowData],
        window_minutes: int = 5
    ) -> Dict[str, float]:
        """
        Analyze order flow patterns and imbalances
        
        Args:
            order_flow_data: List of order flow snapshots
            window_minutes: Analysis window in minutes
            
        Returns:
            Dictionary with order flow metrics
        """
        if not order_flow_data:
            return {}
            
        df = pd.DataFrame([
            {
                'timestamp': d.timestamp,
                'bid_price': d.bid_price,
                'ask_price': d.ask_price,
                'bid_size': d.bid_size,
                'ask_size': d.ask_size,
                'last_price': d.last_price,
                'volume': d.volume,
                'trade_direction': d.trade_direction or 0
            }
            for d in order_flow_data
        ])
        
        if df.empty:
            return {}
        
        df = df.sort_values('timestamp')
        df['spread'] = df['ask_price'] - df['bid_price']
        df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
        df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])
        df['dollar_volume'] = df['volume'] * df['last_price']
        
        # Calculate rolling metrics
        window_size = max(1, len(df) // 10)  # Adaptive window size
        
        metrics = {
            'avg_spread_bps': (df['spread'] / df['mid_price'] * 10000).mean(),
            'avg_order_imbalance': df['order_imbalance'].mean(),
            'order_imbalance_volatility': df['order_imbalance'].std(),
            'buy_volume_ratio': len(df[df['trade_direction'] == 1]) / len(df) if len(df) > 0 else 0,
            'sell_volume_ratio': len(df[df['trade_direction'] == -1]) / len(df) if len(df) > 0 else 0,
            'total_dollar_volume': df['dollar_volume'].sum(),
            'avg_trade_size': df['volume'].mean(),
            'large_trade_ratio': len(df[df['volume'] > df['volume'].quantile(0.9)]) / len(df),
            'spread_volatility': df['spread'].std(),
            'quote_intensity': len(df) / ((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60),
        }
        
        # Add momentum and flow direction metrics
        if len(df) > window_size:
            df['price_momentum'] = df['last_price'].rolling(window_size).apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
            )
            df['volume_momentum'] = df['volume'].rolling(window_size).mean()
            
            metrics.update({
                'price_momentum': df['price_momentum'].iloc[-1] if not pd.isna(df['price_momentum'].iloc[-1]) else 0,
                'volume_trend': df['volume_momentum'].iloc[-1] / df['volume'].mean() - 1 if df['volume'].mean() > 0 else 0,
            })
        
        return metrics
    
    async def calculate_liquidity_metrics(
        self,
        order_flow_data: List[OrderFlowData],
        trade_data: pd.DataFrame
    ) -> LiquidityMetrics:
        """
        Calculate comprehensive liquidity metrics
        
        Args:
            order_flow_data: Order book snapshots
            trade_data: Historical trade data with columns: timestamp, price, volume, direction
            
        Returns:
            LiquidityMetrics object with all calculated metrics
        """
        if not order_flow_data:
            raise ValueError("No order flow data provided")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                'timestamp': d.timestamp,
                'bid_price': d.bid_price,
                'ask_price': d.ask_price,
                'bid_size': d.bid_size,
                'ask_size': d.ask_size,
                'last_price': d.last_price,
                'volume': d.volume
            }
            for d in order_flow_data
        ])
        
        df = df.sort_values('timestamp')
        df['spread'] = df['ask_price'] - df['bid_price']
        df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
        df['quoted_spread'] = df['spread']
        df['spread_bps'] = (df['spread'] / df['mid_price'] * 10000)
        df['market_depth'] = df['bid_size'] + df['ask_size']
        df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])
        
        # Basic liquidity metrics
        bid_ask_spread = df['spread'].mean()
        spread_bps = df['spread_bps'].mean()
        quoted_spread = df['quoted_spread'].mean()
        market_depth = df['market_depth'].mean()
        order_imbalance = df['order_imbalance'].mean()
        
        # Calculate effective spread (requires trade data)
        effective_spread = 0.0
        realized_spread = 0.0
        price_impact = 0.0
        kyle_lambda = 0.0
        amihud_illiquidity = 0.0
        roll_measure = 0.0
        
        if not trade_data.empty:
            # Effective spread calculation
            trade_data_sorted = trade_data.sort_values('timestamp')
            
            # Merge trades with quotes (approximate)
            merged_data = []
            for _, trade in trade_data_sorted.iterrows():
                # Find closest quote
                time_diffs = np.abs([(q - trade['timestamp']).total_seconds() for q in df['timestamp']])
                closest_idx = np.argmin(time_diffs)
                
                if time_diffs[closest_idx] < 60:  # Within 1 minute
                    quote = df.iloc[closest_idx]
                    merged_data.append({
                        'trade_price': trade['price'],
                        'trade_volume': trade['volume'],
                        'trade_direction': trade.get('direction', 0),
                        'mid_price': quote['mid_price'],
                        'spread': quote['spread']
                    })
            
            if merged_data:
                merged_df = pd.DataFrame(merged_data)
                
                # Effective spread: 2 * |trade_price - mid_price|
                merged_df['effective_spread'] = 2 * np.abs(merged_df['trade_price'] - merged_df['mid_price'])
                effective_spread = merged_df['effective_spread'].mean()
                
                # Price impact (Kyle's lambda): regression of price change on signed volume
                if len(merged_df) > 10:
                    merged_df['signed_volume'] = merged_df['trade_volume'] * merged_df['trade_direction']
                    merged_df['price_change'] = merged_df['trade_price'].diff()
                    
                    valid_data = merged_df.dropna()
                    if len(valid_data) > 5:
                        # Linear regression: price_change = lambda * signed_volume + error
                        X = valid_data['signed_volume'].values.reshape(-1, 1)
                        y = valid_data['price_change'].values
                        
                        if np.var(X) > 0:
                            kyle_lambda = np.cov(X.flatten(), y)[0, 1] / np.var(X)
                            price_impact = kyle_lambda * valid_data['signed_volume'].abs().mean()
                
                # Amihud illiquidity measure: |return| / dollar_volume
                merged_df['return'] = merged_df['trade_price'].pct_change().abs()
                merged_df['dollar_volume'] = merged_df['trade_price'] * merged_df['trade_volume']
                valid_amihud = merged_df.dropna()
                
                if len(valid_amihud) > 0 and valid_amihud['dollar_volume'].mean() > 0:
                    amihud_illiquidity = (valid_amihud['return'] / valid_amihud['dollar_volume']).mean()
                
                # Roll's measure (from price changes)
                price_changes = merged_df['trade_price'].diff().dropna()
                if len(price_changes) > 1:
                    # Roll measure = sqrt(-covariance of successive price changes)
                    cov_changes = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
                    if cov_changes < 0:
                        roll_measure = np.sqrt(-cov_changes)
        
        # Volume-weighted spread
        volume_weighted_spread = np.average(df['spread'], weights=df['market_depth'])
        
        # Relative tick size
        relative_tick_size = self.tick_size / df['mid_price'].mean()
        
        return LiquidityMetrics(
            bid_ask_spread=bid_ask_spread,
            spread_bps=spread_bps,
            quoted_spread=quoted_spread,
            effective_spread=effective_spread,
            realized_spread=realized_spread,
            price_impact=price_impact,
            market_depth=market_depth,
            order_imbalance=order_imbalance,
            kyle_lambda=kyle_lambda,
            amihud_illiquidity=amihud_illiquidity,
            roll_measure=roll_measure,
            volume_weighted_spread=volume_weighted_spread,
            relative_tick_size=relative_tick_size
        )
    
    async def analyze_volume_profile(
        self,
        trade_data: pd.DataFrame,
        price_bins: int = 100
    ) -> VolumeProfile:
        """
        Analyze volume profile and identify key price levels
        
        Args:
            trade_data: DataFrame with columns: price, volume
            price_bins: Number of price bins for profile
            
        Returns:
            VolumeProfile with key levels and statistics
        """
        if trade_data.empty:
            raise ValueError("No trade data provided")
        
        # Calculate volume at each price level
        min_price = trade_data['price'].min()
        max_price = trade_data['price'].max()
        
        price_levels = np.linspace(min_price, max_price, price_bins)
        volume_by_price = np.zeros(price_bins)
        
        # Bin the volume by price
        for i, price in enumerate(price_levels[:-1]):
            price_mask = (trade_data['price'] >= price) & (trade_data['price'] < price_levels[i + 1])
            volume_by_price[i] = trade_data[price_mask]['volume'].sum()
        
        # Handle the last bin (inclusive)
        last_bin_mask = (trade_data['price'] >= price_levels[-2])
        volume_by_price[-1] = trade_data[last_bin_mask]['volume'].sum()
        
        # Find Point of Control (POC) - price with highest volume
        poc_idx = np.argmax(volume_by_price)
        poc = price_levels[poc_idx]
        
        # Calculate Value Area (70% of volume around POC)
        total_volume = volume_by_price.sum()
        target_volume = total_volume * self.value_area_percentage
        
        # Expand from POC until we capture target volume
        value_area_volume = volume_by_price[poc_idx]
        lower_idx = poc_idx
        upper_idx = poc_idx
        
        while value_area_volume < target_volume and (lower_idx > 0 or upper_idx < len(price_levels) - 1):
            # Decide whether to expand up or down based on which has more volume
            expand_down = (lower_idx > 0) and (
                upper_idx >= len(price_levels) - 1 or 
                volume_by_price[lower_idx - 1] >= volume_by_price[upper_idx + 1]
            )
            
            if expand_down:
                lower_idx -= 1
                value_area_volume += volume_by_price[lower_idx]
            else:
                upper_idx += 1
                value_area_volume += volume_by_price[upper_idx]
        
        value_area_high = price_levels[upper_idx]
        value_area_low = price_levels[lower_idx]
        value_area_volume_pct = value_area_volume / total_volume
        
        # Developing POC (weighted average price)
        developing_poc = np.average(price_levels, weights=volume_by_price)
        
        # Identify volume imbalance areas (areas with unusually low volume)
        volume_threshold = np.percentile(volume_by_price[volume_by_price > 0], 25)
        imbalance_areas = []
        
        in_imbalance = False
        imbalance_start = None
        
        for i, volume in enumerate(volume_by_price):
            if volume < volume_threshold and not in_imbalance:
                in_imbalance = True
                imbalance_start = price_levels[i]
            elif volume >= volume_threshold and in_imbalance:
                in_imbalance = False
                imbalance_areas.append((imbalance_start, price_levels[i]))
        
        # Handle case where imbalance extends to the end
        if in_imbalance:
            imbalance_areas.append((imbalance_start, price_levels[-1]))
        
        return VolumeProfile(
            price_levels=price_levels,
            volume_by_price=volume_by_price,
            poc=poc,
            value_area_high=value_area_high,
            value_area_low=value_area_low,
            value_area_volume_pct=value_area_volume_pct,
            developing_poc=developing_poc,
            volume_imbalance_areas=imbalance_areas
        )
    
    def identify_microstructure_regime(
        self,
        liquidity_metrics: LiquidityMetrics,
        volume_profile: VolumeProfile,
        market_conditions: Dict[str, float]
    ) -> MicrostructureRegime:
        """
        Identify current market microstructure regime
        
        Args:
            liquidity_metrics: Current liquidity conditions
            volume_profile: Current volume distribution
            market_conditions: Additional market context (volatility, etc.)
            
        Returns:
            MicrostructureRegime classification with optimal execution strategy
        """
        # Classify liquidity level
        if liquidity_metrics.spread_bps < 5 and liquidity_metrics.market_depth > 10000:
            liquidity_level = "high"
        elif liquidity_metrics.spread_bps < 15 and liquidity_metrics.market_depth > 5000:
            liquidity_level = "medium"
        else:
            liquidity_level = "low"
        
        # Classify spread regime
        if liquidity_metrics.spread_bps < 5:
            spread_regime = "tight"
        elif liquidity_metrics.spread_bps < 15:
            spread_regime = "normal"
        else:
            spread_regime = "wide"
        
        # Classify volatility regime
        volatility = market_conditions.get('volatility', 0.2)
        if volatility < 0.15:
            volatility_regime = "low"
        elif volatility < 0.3:
            volatility_regime = "normal"
        else:
            volatility_regime = "high"
        
        # Classify order flow regime
        if abs(liquidity_metrics.order_imbalance) < 0.1:
            order_flow_regime = "balanced"
        elif liquidity_metrics.order_imbalance > 0.1:
            order_flow_regime = "buy_pressure"
        else:
            order_flow_regime = "sell_pressure"
        
        # Determine optimal execution strategy
        optimal_execution_strategy = self._determine_execution_strategy(
            liquidity_level, spread_regime, volatility_regime, order_flow_regime,
            liquidity_metrics, volume_profile
        )
        
        regime_id = f"{liquidity_level}_{spread_regime}_{volatility_regime}_{order_flow_regime}"
        
        return MicrostructureRegime(
            regime_id=regime_id,
            liquidity_level=liquidity_level,
            spread_regime=spread_regime,
            volatility_regime=volatility_regime,
            order_flow_regime=order_flow_regime,
            typical_spread_bps=liquidity_metrics.spread_bps,
            typical_market_depth=liquidity_metrics.market_depth,
            typical_price_impact=liquidity_metrics.price_impact,
            optimal_execution_strategy=optimal_execution_strategy
        )
    
    def _determine_execution_strategy(
        self,
        liquidity_level: str,
        spread_regime: str,
        volatility_regime: str,
        order_flow_regime: str,
        liquidity_metrics: LiquidityMetrics,
        volume_profile: VolumeProfile
    ) -> Dict[str, Any]:
        """Determine optimal execution strategy based on microstructure regime"""
        
        strategy = {
            "primary_strategy": "twap",  # Default
            "urgency_factor": 0.5,
            "max_participation_rate": 0.2,
            "use_hidden_orders": False,
            "prefer_dark_pools": False,
            "slice_size_factor": 0.1,
            "time_horizon_minutes": 30,
            "price_improvement_threshold": 0.5,  # bps
        }
        
        # High liquidity regimes
        if liquidity_level == "high":
            strategy.update({
                "primary_strategy": "market_making" if order_flow_regime == "balanced" else "twap",
                "max_participation_rate": 0.3,
                "slice_size_factor": 0.15,
                "time_horizon_minutes": 15,
            })
        
        # Low liquidity regimes
        elif liquidity_level == "low":
            strategy.update({
                "primary_strategy": "iceberg",
                "use_hidden_orders": True,
                "prefer_dark_pools": True,
                "max_participation_rate": 0.1,
                "slice_size_factor": 0.05,
                "time_horizon_minutes": 60,
            })
        
        # Wide spread regimes
        if spread_regime == "wide":
            strategy.update({
                "primary_strategy": "limit_order_book",
                "price_improvement_threshold": liquidity_metrics.spread_bps * 0.3,
                "urgency_factor": 0.3,
            })
        
        # High volatility adjustments
        if volatility_regime == "high":
            strategy.update({
                "urgency_factor": min(strategy["urgency_factor"] + 0.2, 1.0),
                "time_horizon_minutes": max(strategy["time_horizon_minutes"] - 10, 5),
            })
        
        # Order flow imbalance adjustments
        if order_flow_regime in ["buy_pressure", "sell_pressure"]:
            strategy.update({
                "urgency_factor": min(strategy["urgency_factor"] + 0.15, 1.0),
                "use_hidden_orders": True,
            })
        
        # Volume profile considerations
        if len(volume_profile.volume_imbalance_areas) > 2:
            strategy.update({
                "primary_strategy": "vwap_aware",
                "avoid_imbalance_areas": True,
            })
        
        return strategy

# Integration with event-driven CAR analysis
class EventMicrostructureIntegrator:
    """Integrates microstructure analysis with event-driven CAR studies"""
    
    def __init__(self, microstructure_analyzer: MicrostructureAnalyzer):
        self.microstructure_analyzer = microstructure_analyzer
        
    async def analyze_event_microstructure_impact(
        self,
        event_type: EventType,
        sector: Optional[Sector],
        pre_event_data: List[OrderFlowData],
        post_event_data: List[OrderFlowData],
        event_trade_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze how events impact market microstructure
        
        Returns:
            Comprehensive analysis of microstructure changes around events
        """
        # Analyze pre-event conditions
        pre_event_metrics = await self.microstructure_analyzer.calculate_liquidity_metrics(
            pre_event_data, event_trade_data
        )
        
        # Analyze post-event conditions
        post_event_metrics = await self.microstructure_analyzer.calculate_liquidity_metrics(
            post_event_data, event_trade_data
        )
        
        # Calculate changes
        spread_impact = post_event_metrics.spread_bps - pre_event_metrics.spread_bps
        depth_impact = (post_event_metrics.market_depth - pre_event_metrics.market_depth) / pre_event_metrics.market_depth
        imbalance_change = abs(post_event_metrics.order_imbalance) - abs(pre_event_metrics.order_imbalance)
        
        # Volume profile analysis
        volume_profile = await self.microstructure_analyzer.analyze_volume_profile(event_trade_data)
        
        return {
            "event_type": event_type.value,
            "sector": sector.value if sector else None,
            "spread_impact_bps": spread_impact,
            "depth_impact_pct": depth_impact * 100,
            "imbalance_change": imbalance_change,
            "liquidity_deterioration": spread_impact > 5 and depth_impact < -0.2,
            "execution_difficulty": self._assess_execution_difficulty(
                spread_impact, depth_impact, imbalance_change
            ),
            "optimal_execution_timing": self._determine_execution_timing(
                pre_event_metrics, post_event_metrics, volume_profile
            ),
            "pre_event_regime": self.microstructure_analyzer.identify_microstructure_regime(
                pre_event_metrics, volume_profile, {"volatility": 0.2}
            ),
            "post_event_regime": self.microstructure_analyzer.identify_microstructure_regime(
                post_event_metrics, volume_profile, {"volatility": 0.3}
            )
        }
    
    def _assess_execution_difficulty(
        self, 
        spread_impact: float, 
        depth_impact: float, 
        imbalance_change: float
    ) -> str:
        """Assess execution difficulty based on microstructure changes"""
        
        difficulty_score = 0
        
        # Spread widening increases difficulty
        if spread_impact > 10:
            difficulty_score += 3
        elif spread_impact > 5:
            difficulty_score += 2
        elif spread_impact > 2:
            difficulty_score += 1
        
        # Depth reduction increases difficulty
        if depth_impact < -0.5:
            difficulty_score += 3
        elif depth_impact < -0.2:
            difficulty_score += 2
        elif depth_impact < -0.1:
            difficulty_score += 1
        
        # Order flow imbalance increases difficulty
        if abs(imbalance_change) > 0.3:
            difficulty_score += 2
        elif abs(imbalance_change) > 0.1:
            difficulty_score += 1
        
        if difficulty_score >= 6:
            return "very_high"
        elif difficulty_score >= 4:
            return "high"
        elif difficulty_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _determine_execution_timing(
        self,
        pre_event_metrics: LiquidityMetrics,
        post_event_metrics: LiquidityMetrics,
        volume_profile: VolumeProfile
    ) -> Dict[str, str]:
        """Determine optimal execution timing relative to events"""
        
        timing_strategy = {}
        
        # If post-event liquidity is much worse, execute before
        if (post_event_metrics.spread_bps > pre_event_metrics.spread_bps * 1.5 or
            post_event_metrics.market_depth < pre_event_metrics.market_depth * 0.7):
            timing_strategy["primary"] = "pre_event"
            timing_strategy["reasoning"] = "liquidity_deterioration"
        
        # If post-event has better liquidity (unusual), execute after
        elif (post_event_metrics.spread_bps < pre_event_metrics.spread_bps * 0.8 and
              post_event_metrics.market_depth > pre_event_metrics.market_depth * 1.2):
            timing_strategy["primary"] = "post_event"
            timing_strategy["reasoning"] = "liquidity_improvement"
        
        # Default to gradual execution around event
        else:
            timing_strategy["primary"] = "gradual"
            timing_strategy["reasoning"] = "mixed_conditions"
        
        return timing_strategy

# Factory functions
async def create_microstructure_analyzer() -> MicrostructureAnalyzer:
    """Create microstructure analyzer"""
    return MicrostructureAnalyzer()

async def create_event_microstructure_integrator() -> EventMicrostructureIntegrator:
    """Create event-microstructure integrator"""
    analyzer = await create_microstructure_analyzer()
    return EventMicrostructureIntegrator(analyzer)