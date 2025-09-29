"""
Adaptive Market Impact and Slippage Model

Implements sophisticated market microstructure models for realistic execution cost estimation:
- Almgren-Chriss optimal execution model
- Square root market impact model  
- Order size vs Average Daily Volume (ADV) analysis
- Bid-ask spread volatility modeling
- Dynamic slippage based on market conditions

Key Features:
- Adaptive impact based on order size relative to liquidity
- Temporary vs permanent market impact decomposition
- Volatility-adjusted execution costs
- Real-time liquidity assessment
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import warnings
from scipy import optimize
import asyncio
import math

logger = logging.getLogger(__name__)

class ImpactModel(Enum):
    """Market impact model types"""
    ALMGREN_CHRISS = "almgren_chriss"
    SQUARE_ROOT = "square_root"
    LINEAR = "linear"
    POWER_LAW = "power_law"
    HYBRID = "hybrid"

class ExecutionStyle(Enum):
    """Execution style/urgency"""
    AGGRESSIVE = "aggressive"      # Market orders, immediate execution
    MODERATE = "moderate"          # Mix of market and limit orders
    PASSIVE = "passive"           # Limit orders, patient execution
    TWAP = "twap"                 # Time-weighted average price
    VWAP = "vwap"                 # Volume-weighted average price

@dataclass
class MarketMicrostructure:
    """Market microstructure data for impact modeling"""
    symbol: str
    current_price: float
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    average_daily_volume: float  # ADV in shares
    average_daily_value: float   # ADV in dollars
    volatility_annualized: float
    market_cap: Optional[float] = None
    tick_size: float = 0.01
    
    # Spread metrics
    bid_ask_spread_bps: float = field(init=False)
    spread_volatility: float = field(init=False)
    relative_spread: float = field(init=False)
    
    # Liquidity metrics
    depth_imbalance: float = field(init=False)
    liquidity_score: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived metrics"""
        mid_price = (self.bid_price + self.ask_price) / 2
        spread = self.ask_price - self.bid_price
        
        self.bid_ask_spread_bps = (spread / mid_price) * 10000
        self.relative_spread = spread / mid_price
        
        # Simple depth imbalance
        total_depth = self.bid_size + self.ask_size
        self.depth_imbalance = (self.bid_size - self.ask_size) / total_depth if total_depth > 0 else 0
        
        # Basic liquidity score (0-1, higher is more liquid)
        self.liquidity_score = min(1.0, (self.average_daily_value / 1000000) / 100)  # $100M ADV = 1.0
        
        # Default spread volatility (would be calculated from historical data)
        self.spread_volatility = self.bid_ask_spread_bps * 0.5

@dataclass
class OrderProfile:
    """Order execution profile"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    target_price: Optional[float] = None
    execution_style: ExecutionStyle = ExecutionStyle.MODERATE
    time_horizon_minutes: float = 60.0  # Execution time horizon
    participation_rate: float = 0.2  # Max participation in volume (20%)
    urgency_factor: float = 0.5  # 0 = very patient, 1 = very urgent
    
    # Advanced parameters
    risk_aversion: float = 1.0  # Almgren-Chriss risk aversion parameter
    max_adv_participation: float = 0.25  # Max 25% of ADV per day

@dataclass
class ImpactEstimate:
    """Market impact and execution cost estimate"""
    symbol: str
    order_quantity: float
    order_value: float
    
    # Core impact components
    temporary_impact_bps: float
    permanent_impact_bps: float
    total_impact_bps: float
    
    # Execution costs
    bid_ask_cost_bps: float
    timing_cost_bps: float
    opportunity_cost_bps: float
    total_execution_cost_bps: float
    
    # Dollar amounts
    total_cost_dollars: float
    impact_cost_dollars: float
    
    # Market context
    adv_participation: float  # Order size as % of ADV
    liquidity_score: float
    execution_time_estimate_minutes: float
    price_improvement_probability: float
    
    # Model metadata
    impact_model: ImpactModel
    calculation_timestamp: datetime
    market_conditions: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'order_quantity': self.order_quantity,
            'order_value': self.order_value,
            'temporary_impact_bps': self.temporary_impact_bps,
            'permanent_impact_bps': self.permanent_impact_bps,
            'total_impact_bps': self.total_impact_bps,
            'bid_ask_cost_bps': self.bid_ask_cost_bps,
            'timing_cost_bps': self.timing_cost_bps,
            'opportunity_cost_bps': self.opportunity_cost_bps,
            'total_execution_cost_bps': self.total_execution_cost_bps,
            'total_cost_dollars': self.total_cost_dollars,
            'impact_cost_dollars': self.impact_cost_dollars,
            'adv_participation': self.adv_participation,
            'liquidity_score': self.liquidity_score,
            'execution_time_estimate_minutes': self.execution_time_estimate_minutes,
            'price_improvement_probability': self.price_improvement_probability,
            'impact_model': self.impact_model.value,
            'calculation_timestamp': self.calculation_timestamp.isoformat(),
            'market_conditions': self.market_conditions
        }

class MarketImpactModel:
    """Comprehensive adaptive market impact model"""
    
    def __init__(self):
        # Model parameters (calibrated to empirical studies)
        self.almgren_chriss_params = {
            'eta': 2.5e-6,     # Temporary impact coefficient
            'gamma': 2.5e-7,   # Permanent impact coefficient  
            'sigma_factor': 1.0 # Volatility scaling
        }
        
        self.square_root_params = {
            'alpha': 0.314,    # Square root coefficient (Barra)
            'beta': 0.5,       # Power law exponent
            'min_impact': 0.1  # Minimum impact (bps)
        }
        
        # Market regime adjustments
        self.volatility_adjustments = {
            'low_vol': 0.8,     # < 15% annualized vol
            'normal_vol': 1.0,  # 15-30% annualized vol
            'high_vol': 1.5     # > 30% annualized vol
        }
        
    def estimate_market_impact(self, market_data: MarketMicrostructure, 
                              order: OrderProfile,
                              model: ImpactModel = ImpactModel.HYBRID) -> ImpactEstimate:
        """
        Estimate comprehensive market impact for an order.
        
        Args:
            market_data: Current market microstructure data
            order: Order profile and execution parameters
            model: Impact model to use
            
        Returns:
            Comprehensive impact estimate
        """
        
        try:
            # Basic calculations
            order_value = order.quantity * market_data.current_price
            adv_participation = order.quantity / market_data.average_daily_volume
            
            # Choose impact model
            if model == ImpactModel.ALMGREN_CHRISS:
                temp_impact, perm_impact = self._almgren_chriss_impact(market_data, order)
            elif model == ImpactModel.SQUARE_ROOT:
                temp_impact, perm_impact = self._square_root_impact(market_data, order)
            elif model == ImpactModel.LINEAR:
                temp_impact, perm_impact = self._linear_impact(market_data, order)
            elif model == ImpactModel.POWER_LAW:
                temp_impact, perm_impact = self._power_law_impact(market_data, order)
            elif model == ImpactModel.HYBRID:
                temp_impact, perm_impact = self._hybrid_impact(market_data, order)
            else:
                raise ValueError(f"Unsupported impact model: {model}")
            
            # Apply market regime adjustments
            vol_regime = self._get_volatility_regime(market_data.volatility_annualized)
            vol_adjustment = self.volatility_adjustments[vol_regime]
            
            temp_impact *= vol_adjustment
            perm_impact *= vol_adjustment
            
            total_impact = temp_impact + perm_impact
            
            # Calculate additional execution costs
            bid_ask_cost = self._calculate_bid_ask_cost(market_data, order)
            timing_cost = self._calculate_timing_cost(market_data, order)
            opportunity_cost = self._calculate_opportunity_cost(market_data, order)
            
            total_execution_cost = total_impact + bid_ask_cost + timing_cost + opportunity_cost
            
            # Estimate execution characteristics
            execution_time = self._estimate_execution_time(market_data, order)
            price_improvement_prob = self._estimate_price_improvement(market_data, order)
            
            # Convert to dollar amounts
            total_cost_dollars = (total_execution_cost / 10000) * order_value
            impact_cost_dollars = (total_impact / 10000) * order_value
            
            # Market conditions summary
            market_conditions = {
                'volatility_regime': vol_regime,
                'liquidity_score': market_data.liquidity_score,
                'spread_bps': market_data.bid_ask_spread_bps,
                'depth_imbalance': market_data.depth_imbalance,
                'vol_adjustment': vol_adjustment
            }
            
            return ImpactEstimate(
                symbol=order.symbol,
                order_quantity=order.quantity,
                order_value=order_value,
                temporary_impact_bps=temp_impact,
                permanent_impact_bps=perm_impact,
                total_impact_bps=total_impact,
                bid_ask_cost_bps=bid_ask_cost,
                timing_cost_bps=timing_cost,
                opportunity_cost_bps=opportunity_cost,
                total_execution_cost_bps=total_execution_cost,
                total_cost_dollars=total_cost_dollars,
                impact_cost_dollars=impact_cost_dollars,
                adv_participation=adv_participation,
                liquidity_score=market_data.liquidity_score,
                execution_time_estimate_minutes=execution_time,
                price_improvement_probability=price_improvement_prob,
                impact_model=model,
                calculation_timestamp=datetime.now(),
                market_conditions=market_conditions
            )
            
        except Exception as e:
            logger.error(f"Error estimating market impact: {e}")
            # Return conservative fallback estimate
            return self._fallback_impact_estimate(market_data, order, model)
    
    def _almgren_chriss_impact(self, market_data: MarketMicrostructure, 
                              order: OrderProfile) -> Tuple[float, float]:
        """Almgren-Chriss optimal execution model"""
        
        params = self.almgren_chriss_params
        
        # Market impact parameters
        eta = params['eta'] * (market_data.volatility_annualized ** 2)
        gamma = params['gamma'] * market_data.volatility_annualized
        
        # Order characteristics
        X = order.quantity  # Total order size
        T = order.time_horizon_minutes / (60 * 24)  # Convert to days
        sigma = market_data.volatility_annualized / np.sqrt(252)  # Daily volatility
        
        # Almgren-Chriss formulation
        # Temporary impact: η * v_i (impact of trading rate)
        trading_rate = X / (T * 252 * 6.5 * 60)  # Shares per minute during market hours
        temporary_impact = eta * trading_rate * 10000  # Convert to bps
        
        # Permanent impact: γ * X (impact of total order)
        permanent_impact = gamma * X * 10000  # Convert to bps
        
        # Apply liquidity adjustments
        liquidity_factor = 1 / (market_data.liquidity_score + 0.1)  # Higher for illiquid stocks
        temporary_impact *= liquidity_factor
        permanent_impact *= liquidity_factor
        
        return temporary_impact, permanent_impact
    
    def _square_root_impact(self, market_data: MarketMicrostructure, 
                           order: OrderProfile) -> Tuple[float, float]:
        """Square root market impact model (Barra/MSCI)"""
        
        params = self.square_root_params
        
        # Participation rate relative to ADV
        participation = order.quantity / market_data.average_daily_volume
        
        # Square root impact: α * σ * (X/V)^β
        alpha = params['alpha']
        beta = params['beta']
        
        volatility_daily = market_data.volatility_annualized / np.sqrt(252)
        
        # Temporary impact (scales with urgency)
        urgency_multiplier = 1 + order.urgency_factor
        temporary_impact = alpha * volatility_daily * (participation ** beta) * urgency_multiplier * 10000
        
        # Permanent impact (smaller component, ~1/3 of temporary)
        permanent_impact = temporary_impact * 0.33
        
        # Minimum impact floor
        temporary_impact = max(temporary_impact, params['min_impact'])
        permanent_impact = max(permanent_impact, params['min_impact'] * 0.33)
        
        return temporary_impact, permanent_impact
    
    def _linear_impact(self, market_data: MarketMicrostructure, 
                      order: OrderProfile) -> Tuple[float, float]:
        """Simple linear impact model"""
        
        participation = order.quantity / market_data.average_daily_volume
        base_impact = participation * 50  # 50 bps per 1% of ADV
        
        # Adjust for volatility
        vol_adjustment = market_data.volatility_annualized / 0.25  # Normalize to 25% vol
        
        temporary_impact = base_impact * vol_adjustment * 0.7
        permanent_impact = base_impact * vol_adjustment * 0.3
        
        return temporary_impact, permanent_impact
    
    def _power_law_impact(self, market_data: MarketMicrostructure, 
                         order: OrderProfile) -> Tuple[float, float]:
        """Power law impact model"""
        
        participation = order.quantity / market_data.average_daily_volume
        
        # Power law: impact ∝ (order_size)^α
        alpha = 0.6  # Typical empirical value
        base_impact = 100 * (participation ** alpha)  # 100 bps coefficient
        
        # Volatility scaling
        vol_factor = market_data.volatility_annualized / 0.25
        
        temporary_impact = base_impact * vol_factor * 0.75
        permanent_impact = base_impact * vol_factor * 0.25
        
        return temporary_impact, permanent_impact
    
    def _hybrid_impact(self, market_data: MarketMicrostructure, 
                      order: OrderProfile) -> Tuple[float, float]:
        """Hybrid model combining Almgren-Chriss and square root"""
        
        # Get estimates from both models
        ac_temp, ac_perm = self._almgren_chriss_impact(market_data, order)
        sr_temp, sr_perm = self._square_root_impact(market_data, order)
        
        # Weight based on order size and liquidity
        participation = order.quantity / market_data.average_daily_volume
        
        if participation < 0.01:  # Small orders: favor square root
            weight_ac = 0.3
        elif participation > 0.1:  # Large orders: favor Almgren-Chriss
            weight_ac = 0.8
        else:  # Medium orders: balanced
            weight_ac = 0.5
        
        weight_sr = 1 - weight_ac
        
        temporary_impact = weight_ac * ac_temp + weight_sr * sr_temp
        permanent_impact = weight_ac * ac_perm + weight_sr * sr_perm
        
        return temporary_impact, permanent_impact
    
    def _calculate_bid_ask_cost(self, market_data: MarketMicrostructure, 
                               order: OrderProfile) -> float:
        """Calculate bid-ask spread crossing cost"""
        
        # Base cost is half the spread (assuming mid-point reference)
        base_cost = market_data.bid_ask_spread_bps / 2
        
        # Adjust based on execution style
        style_multipliers = {
            ExecutionStyle.AGGRESSIVE: 1.0,   # Pay full spread
            ExecutionStyle.MODERATE: 0.7,    # Some spread savings
            ExecutionStyle.PASSIVE: 0.3,     # Significant spread savings
            ExecutionStyle.TWAP: 0.5,        # Mixed execution
            ExecutionStyle.VWAP: 0.5         # Mixed execution
        }
        
        multiplier = style_multipliers.get(order.execution_style, 0.7)
        
        # Adjust for spread volatility (higher volatility = harder to get good fills)
        spread_vol_adjustment = 1 + (market_data.spread_volatility / 100)
        
        return base_cost * multiplier * spread_vol_adjustment
    
    def _calculate_timing_cost(self, market_data: MarketMicrostructure, 
                              order: OrderProfile) -> float:
        """Calculate timing cost due to price drift during execution"""
        
        # Timing cost based on volatility and execution time
        execution_time_days = order.time_horizon_minutes / (60 * 24)
        daily_volatility = market_data.volatility_annualized / np.sqrt(252)
        
        # Timing risk scales with sqrt(time) and volatility
        timing_volatility = daily_volatility * np.sqrt(execution_time_days)
        
        # Convert to cost (assuming ~1 std dev adverse movement)
        timing_cost = timing_volatility * 10000 * 0.4  # 40% of 1-std move
        
        # Adjust for urgency (more urgent = higher timing cost)
        urgency_adjustment = 0.5 + order.urgency_factor
        
        return timing_cost * urgency_adjustment
    
    def _calculate_opportunity_cost(self, market_data: MarketMicrostructure, 
                                   order: OrderProfile) -> float:
        """Calculate opportunity cost of delayed execution"""
        
        # Opportunity cost increases with urgency and decreases with patience
        base_opportunity_cost = market_data.volatility_annualized * 2  # 2% of annual vol
        
        # Scale by urgency and execution horizon
        urgency_factor = order.urgency_factor
        horizon_factor = np.sqrt(order.time_horizon_minutes / 60)  # Square root of hours
        
        opportunity_cost = base_opportunity_cost * urgency_factor * horizon_factor
        
        return opportunity_cost
    
    def _estimate_execution_time(self, market_data: MarketMicrostructure, 
                                order: OrderProfile) -> float:
        """Estimate realistic execution time in minutes"""
        
        # Base on participation rate and market liquidity
        participation = order.quantity / market_data.average_daily_volume
        
        # Estimate days to complete at given participation rate
        days_to_complete = participation / order.max_adv_participation
        
        # Convert to minutes (assume 6.5 hour trading day)
        minutes_to_complete = days_to_complete * 6.5 * 60
        
        # Apply execution style adjustments
        style_factors = {
            ExecutionStyle.AGGRESSIVE: 0.2,   # Very fast
            ExecutionStyle.MODERATE: 1.0,    # Normal pace
            ExecutionStyle.PASSIVE: 3.0,     # Patient execution
            ExecutionStyle.TWAP: 1.5,        # Spread over time
            ExecutionStyle.VWAP: 1.2         # Volume-driven pace
        }
        
        style_factor = style_factors.get(order.execution_style, 1.0)
        
        # Consider market liquidity
        liquidity_factor = 2.0 - market_data.liquidity_score  # Less liquid = longer time
        
        return max(1.0, minutes_to_complete * style_factor * liquidity_factor)
    
    def _estimate_price_improvement(self, market_data: MarketMicrostructure, 
                                   order: OrderProfile) -> float:
        """Estimate probability of price improvement"""
        
        # Base probability depends on execution style
        base_probabilities = {
            ExecutionStyle.AGGRESSIVE: 0.1,   # Low chance with market orders
            ExecutionStyle.MODERATE: 0.4,    # Moderate chance
            ExecutionStyle.PASSIVE: 0.7,     # High chance with limit orders
            ExecutionStyle.TWAP: 0.5,        # Mixed
            ExecutionStyle.VWAP: 0.5         # Mixed
        }
        
        base_prob = base_probabilities.get(order.execution_style, 0.4)
        
        # Adjust for market conditions
        spread_factor = max(0.5, 1 - (market_data.bid_ask_spread_bps / 100))  # Wider spreads = less improvement
        liquidity_factor = market_data.liquidity_score  # More liquid = more improvement
        
        return min(0.9, base_prob * spread_factor * liquidity_factor)
    
    def _get_volatility_regime(self, volatility: float) -> str:
        """Classify volatility regime"""
        if volatility < 0.15:
            return 'low_vol'
        elif volatility > 0.30:
            return 'high_vol'
        else:
            return 'normal_vol'
    
    def _fallback_impact_estimate(self, market_data: MarketMicrostructure, 
                                 order: OrderProfile, model: ImpactModel) -> ImpactEstimate:
        """Conservative fallback estimate when calculations fail"""
        
        participation = order.quantity / market_data.average_daily_volume
        order_value = order.quantity * market_data.current_price
        
        # Conservative estimates
        temporary_impact = max(5.0, participation * 100)  # At least 5 bps
        permanent_impact = temporary_impact * 0.3
        total_impact = temporary_impact + permanent_impact
        
        bid_ask_cost = market_data.bid_ask_spread_bps / 2
        timing_cost = 2.0
        opportunity_cost = 1.0
        
        total_execution_cost = total_impact + bid_ask_cost + timing_cost + opportunity_cost
        
        return ImpactEstimate(
            symbol=order.symbol,
            order_quantity=order.quantity,
            order_value=order_value,
            temporary_impact_bps=temporary_impact,
            permanent_impact_bps=permanent_impact,
            total_impact_bps=total_impact,
            bid_ask_cost_bps=bid_ask_cost,
            timing_cost_bps=timing_cost,
            opportunity_cost_bps=opportunity_cost,
            total_execution_cost_bps=total_execution_cost,
            total_cost_dollars=(total_execution_cost / 10000) * order_value,
            impact_cost_dollars=(total_impact / 10000) * order_value,
            adv_participation=participation,
            liquidity_score=market_data.liquidity_score,
            execution_time_estimate_minutes=60.0,
            price_improvement_probability=0.3,
            impact_model=model,
            calculation_timestamp=datetime.now(),
            market_conditions={'fallback': True}
        )

    def optimize_execution_strategy(self, market_data: MarketMicrostructure, 
                                   order: OrderProfile) -> Dict[str, Any]:
        """
        Optimize execution strategy to minimize total cost.
        
        Returns:
            Optimal execution parameters and cost breakdown
        """
        
        try:
            # Test different execution styles
            strategies = {}
            
            for style in ExecutionStyle:
                test_order = OrderProfile(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    target_price=order.target_price,
                    execution_style=style,
                    time_horizon_minutes=order.time_horizon_minutes,
                    participation_rate=order.participation_rate,
                    urgency_factor=order.urgency_factor
                )
                
                impact_estimate = self.estimate_market_impact(
                    market_data, test_order, ImpactModel.HYBRID
                )
                
                strategies[style.value] = {
                    'total_cost_bps': impact_estimate.total_execution_cost_bps,
                    'total_cost_dollars': impact_estimate.total_cost_dollars,
                    'execution_time_minutes': impact_estimate.execution_time_estimate_minutes,
                    'price_improvement_prob': impact_estimate.price_improvement_probability,
                    'impact_estimate': impact_estimate.to_dict()
                }
            
            # Find optimal strategy
            optimal_style = min(strategies.keys(), 
                              key=lambda s: strategies[s]['total_cost_bps'])
            
            return {
                'optimal_strategy': optimal_style,
                'cost_savings_bps': max(strategies.values(), 
                                      key=lambda s: s['total_cost_bps'])['total_cost_bps'] - 
                                    strategies[optimal_style]['total_cost_bps'],
                'all_strategies': strategies,
                'recommendation': f"Use {optimal_style} execution for optimal cost",
                'market_conditions': {
                    'liquidity_score': market_data.liquidity_score,
                    'spread_bps': market_data.bid_ask_spread_bps,
                    'adv_participation': order.quantity / market_data.average_daily_volume
                }
            }
            
        except Exception as e:
            logger.error(f"Error optimizing execution strategy: {e}")
            return {
                'error': f'Strategy optimization failed: {str(e)}',
                'fallback_recommendation': 'Use moderate execution style'
            }