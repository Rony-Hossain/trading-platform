"""
Market Regime Filter for Event Trade Execution

This module implements sophisticated market regime analysis and filtering
to ensure event trades are only executed in favorable market conditions.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
from decimal import Decimal

class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_MARKET = "bull_market"                   # Strong uptrend
    BEAR_MARKET = "bear_market"                   # Strong downtrend
    SIDEWAYS_MARKET = "sideways_market"           # Range-bound
    HIGH_VOLATILITY = "high_volatility"           # Elevated volatility
    LOW_VOLATILITY = "low_volatility"             # Compressed volatility
    CRISIS_MODE = "crisis_mode"                   # Market stress/panic
    RECOVERY_MODE = "recovery_mode"               # Post-crisis recovery
    TRENDING_UP = "trending_up"                   # Moderate uptrend
    TRENDING_DOWN = "trending_down"               # Moderate downtrend
    DISTRIBUTION = "distribution"                 # Institutional selling
    ACCUMULATION = "accumulation"                 # Institutional buying
    BREAKOUT_MODE = "breakout_mode"               # Breaking key levels

class RegimeFavorability(Enum):
    """Regime favorability for event trades"""
    HIGHLY_FAVORABLE = "highly_favorable"         # Ideal conditions
    FAVORABLE = "favorable"                       # Good conditions
    NEUTRAL = "neutral"                           # Mixed conditions
    UNFAVORABLE = "unfavorable"                   # Poor conditions
    HIGHLY_UNFAVORABLE = "highly_unfavorable"     # Avoid trading

class EventType(Enum):
    """Event types for regime compatibility"""
    EARNINGS = "earnings"
    FDA_APPROVAL = "fda_approval"
    MERGER_ACQUISITION = "merger_acquisition"
    PRODUCT_LAUNCH = "product_launch"
    REGULATORY = "regulatory"
    ANALYST_UPGRADE = "analyst_upgrade"
    ANALYST_DOWNGRADE = "analyst_downgrade"
    GUIDANCE = "guidance"
    SPINOFF = "spinoff"
    DIVIDEND = "dividend"

@dataclass
class RegimeIndicators:
    """Market regime technical indicators"""
    vix_level: float                              # VIX volatility index
    vix_term_structure: float                     # VIX9D/VIX ratio
    market_trend: float                           # SPY 20/50/200 MA trend
    sector_rotation: float                        # Sector momentum dispersion
    credit_spreads: float                         # Investment grade spreads
    risk_appetite: float                          # Risk-on/risk-off sentiment
    liquidity_conditions: float                   # Market liquidity score
    correlation_regime: float                     # Cross-asset correlations
    volatility_clustering: float                  # Vol clustering measure
    breadth_indicators: Dict[str, float]          # Market breadth metrics

@dataclass
class RegimeAnalysis:
    """Comprehensive market regime analysis"""
    timestamp: datetime
    primary_regime: MarketRegime
    secondary_regimes: List[MarketRegime]
    regime_strength: float                        # Confidence in regime classification
    regime_duration: timedelta                    # How long in current regime
    regime_transition_probability: float          # Likelihood of regime change
    
    # Regime characteristics
    volatility_percentile: float                  # Historical volatility percentile
    trend_strength: float                         # Trend momentum strength
    mean_reversion_tendency: float                # Mean reversion likelihood
    breakout_potential: float                     # Potential for large moves
    
    # Risk metrics
    tail_risk: float                              # Extreme move probability
    correlation_risk: float                       # Cross-asset correlation risk
    liquidity_risk: float                         # Market liquidity risk
    
    # Favorability assessment
    overall_favorability: RegimeFavorability
    event_specific_favorability: Dict[EventType, RegimeFavorability]

@dataclass
class TradeExecutionDecision:
    """Event trade execution decision with regime context"""
    symbol: str
    event_type: EventType
    event_details: Dict
    regime_analysis: RegimeAnalysis
    
    # Decision outcome
    execution_approved: bool
    favorability_score: float
    risk_adjustment_factor: float
    position_size_modifier: float
    
    # Reasoning
    approval_reasons: List[str]
    rejection_reasons: List[str]
    risk_mitigation_required: List[str]
    
    # Execution parameters
    recommended_entry_timing: Optional[str]
    regime_stop_loss: Optional[float]
    regime_profit_target_adjustment: Optional[float]

class MarketRegimeFilter:
    """Advanced market regime filtering for event trade execution"""
    
    def __init__(self):
        # Regime favorability matrix for different event types
        self.event_regime_favorability = {
            EventType.EARNINGS: {
                MarketRegime.BULL_MARKET: RegimeFavorability.HIGHLY_FAVORABLE,
                MarketRegime.TRENDING_UP: RegimeFavorability.FAVORABLE,
                MarketRegime.SIDEWAYS_MARKET: RegimeFavorability.NEUTRAL,
                MarketRegime.LOW_VOLATILITY: RegimeFavorability.FAVORABLE,
                MarketRegime.HIGH_VOLATILITY: RegimeFavorability.UNFAVORABLE,
                MarketRegime.BEAR_MARKET: RegimeFavorability.UNFAVORABLE,
                MarketRegime.CRISIS_MODE: RegimeFavorability.HIGHLY_UNFAVORABLE,
                MarketRegime.ACCUMULATION: RegimeFavorability.FAVORABLE
            },
            EventType.FDA_APPROVAL: {
                MarketRegime.BULL_MARKET: RegimeFavorability.HIGHLY_FAVORABLE,
                MarketRegime.TRENDING_UP: RegimeFavorability.HIGHLY_FAVORABLE,
                MarketRegime.LOW_VOLATILITY: RegimeFavorability.FAVORABLE,
                MarketRegime.HIGH_VOLATILITY: RegimeFavorability.NEUTRAL,  # Biotech can handle vol
                MarketRegime.BEAR_MARKET: RegimeFavorability.UNFAVORABLE,
                MarketRegime.CRISIS_MODE: RegimeFavorability.HIGHLY_UNFAVORABLE,
                MarketRegime.RECOVERY_MODE: RegimeFavorability.FAVORABLE
            },
            EventType.MERGER_ACQUISITION: {
                MarketRegime.BULL_MARKET: RegimeFavorability.HIGHLY_FAVORABLE,
                MarketRegime.TRENDING_UP: RegimeFavorability.FAVORABLE,
                MarketRegime.SIDEWAYS_MARKET: RegimeFavorability.FAVORABLE,
                MarketRegime.LOW_VOLATILITY: RegimeFavorability.HIGHLY_FAVORABLE,
                MarketRegime.HIGH_VOLATILITY: RegimeFavorability.UNFAVORABLE,
                MarketRegime.BEAR_MARKET: RegimeFavorability.UNFAVORABLE,
                MarketRegime.CRISIS_MODE: RegimeFavorability.HIGHLY_UNFAVORABLE
            }
        }
        
        # Minimum favorability thresholds for execution
        self.execution_thresholds = {
            'conservative': RegimeFavorability.FAVORABLE,
            'moderate': RegimeFavorability.NEUTRAL,
            'aggressive': RegimeFavorability.UNFAVORABLE
        }
        
        # Regime indicator thresholds
        self.regime_thresholds = {
            'vix_crisis': 30.0,           # VIX above 30 = crisis
            'vix_low_vol': 15.0,          # VIX below 15 = low vol
            'trend_strength_bull': 0.7,   # Strong bull trend
            'trend_strength_bear': -0.7,  # Strong bear trend
            'correlation_risk': 0.8,      # High correlation = risk
            'liquidity_stress': 0.3       # Low liquidity = stress
        }
        
    async def analyze_market_regime(
        self,
        market_data: Dict,
        lookback_periods: Optional[Dict[str, int]] = None
    ) -> RegimeAnalysis:
        """
        Comprehensive market regime analysis using multiple indicators
        """
        if not lookback_periods:
            lookback_periods = {
                'short_term': 5,    # 5 days
                'medium_term': 20,  # 20 days
                'long_term': 60     # 60 days
            }
        
        # Calculate regime indicators
        indicators = self._calculate_regime_indicators(market_data, lookback_periods)
        
        # Determine primary regime
        primary_regime = self._classify_primary_regime(indicators)
        
        # Identify secondary regimes
        secondary_regimes = self._identify_secondary_regimes(indicators, primary_regime)
        
        # Calculate regime characteristics
        regime_strength = self._calculate_regime_strength(indicators, primary_regime)
        regime_duration = self._estimate_regime_duration(market_data, primary_regime)
        transition_prob = self._calculate_transition_probability(indicators)
        
        # Regime-specific metrics
        vol_percentile = self._calculate_volatility_percentile(indicators)
        trend_strength = self._calculate_trend_strength(indicators)
        mean_reversion = self._calculate_mean_reversion_tendency(indicators)
        breakout_potential = self._calculate_breakout_potential(indicators)
        
        # Risk assessments
        tail_risk = self._assess_tail_risk(indicators)
        correlation_risk = self._assess_correlation_risk(indicators)
        liquidity_risk = self._assess_liquidity_risk(indicators)
        
        # Overall favorability
        overall_favorability = self._assess_overall_favorability(
            primary_regime, indicators, regime_strength
        )
        
        # Event-specific favorability
        event_favorability = self._assess_event_specific_favorability(
            primary_regime, secondary_regimes, indicators
        )
        
        return RegimeAnalysis(
            timestamp=datetime.utcnow(),
            primary_regime=primary_regime,
            secondary_regimes=secondary_regimes,
            regime_strength=regime_strength,
            regime_duration=regime_duration,
            regime_transition_probability=transition_prob,
            volatility_percentile=vol_percentile,
            trend_strength=trend_strength,
            mean_reversion_tendency=mean_reversion,
            breakout_potential=breakout_potential,
            tail_risk=tail_risk,
            correlation_risk=correlation_risk,
            liquidity_risk=liquidity_risk,
            overall_favorability=overall_favorability,
            event_specific_favorability=event_favorability
        )
    
    async def evaluate_trade_execution(
        self,
        symbol: str,
        event_type: EventType,
        event_details: Dict,
        regime_analysis: RegimeAnalysis,
        risk_tolerance: str = 'moderate'
    ) -> TradeExecutionDecision:
        """
        Evaluate whether to execute an event trade based on market regime
        """
        # Get event-specific favorability
        event_favorability = regime_analysis.event_specific_favorability.get(
            event_type, RegimeFavorability.NEUTRAL
        )
        
        # Check execution threshold
        threshold = self.execution_thresholds[risk_tolerance]
        execution_approved = self._meets_execution_threshold(event_favorability, threshold)
        
        # Calculate favorability score
        favorability_score = self._calculate_favorability_score(
            regime_analysis, event_type, event_favorability
        )
        
        # Calculate risk adjustment factor
        risk_adjustment = self._calculate_risk_adjustment_factor(regime_analysis)
        
        # Calculate position size modifier
        position_modifier = self._calculate_position_size_modifier(
            regime_analysis, event_favorability, risk_adjustment
        )
        
        # Generate reasoning
        approval_reasons, rejection_reasons = self._generate_decision_reasoning(
            regime_analysis, event_type, event_favorability, execution_approved
        )
        
        # Risk mitigation requirements
        risk_mitigation = self._identify_risk_mitigation_requirements(
            regime_analysis, event_type
        )
        
        # Execution timing and parameters
        entry_timing = self._recommend_entry_timing(regime_analysis, event_type)
        regime_stop = self._calculate_regime_stop_loss(regime_analysis, risk_adjustment)
        target_adjustment = self._calculate_profit_target_adjustment(
            regime_analysis, event_favorability
        )
        
        return TradeExecutionDecision(
            symbol=symbol,
            event_type=event_type,
            event_details=event_details,
            regime_analysis=regime_analysis,
            execution_approved=execution_approved,
            favorability_score=favorability_score,
            risk_adjustment_factor=risk_adjustment,
            position_size_modifier=position_modifier,
            approval_reasons=approval_reasons,
            rejection_reasons=rejection_reasons,
            risk_mitigation_required=risk_mitigation,
            recommended_entry_timing=entry_timing,
            regime_stop_loss=regime_stop,
            regime_profit_target_adjustment=target_adjustment
        )
    
    def _calculate_regime_indicators(
        self, market_data: Dict, lookback_periods: Dict[str, int]
    ) -> RegimeIndicators:
        """Calculate comprehensive regime indicators"""
        
        # VIX analysis
        vix_level = market_data.get('vix', 20.0)
        vix_term_structure = market_data.get('vix9d_vix_ratio', 1.0)
        
        # Market trend analysis
        market_trend = self._calculate_market_trend(market_data, lookback_periods)
        
        # Sector rotation
        sector_rotation = market_data.get('sector_rotation_score', 0.5)
        
        # Credit and risk metrics
        credit_spreads = market_data.get('credit_spreads', 100.0)  # basis points
        risk_appetite = self._calculate_risk_appetite(market_data)
        
        # Liquidity conditions
        liquidity_conditions = market_data.get('liquidity_score', 0.5)
        
        # Correlation regime
        correlation_regime = market_data.get('correlation_level', 0.5)
        
        # Volatility clustering
        volatility_clustering = self._calculate_volatility_clustering(market_data)
        
        # Market breadth
        breadth_indicators = {
            'advance_decline': market_data.get('advance_decline_ratio', 1.0),
            'new_highs_lows': market_data.get('new_highs_lows_ratio', 1.0),
            'volume_breadth': market_data.get('volume_breadth', 0.5),
            'sector_breadth': market_data.get('sector_breadth', 0.5)
        }
        
        return RegimeIndicators(
            vix_level=vix_level,
            vix_term_structure=vix_term_structure,
            market_trend=market_trend,
            sector_rotation=sector_rotation,
            credit_spreads=credit_spreads,
            risk_appetite=risk_appetite,
            liquidity_conditions=liquidity_conditions,
            correlation_regime=correlation_regime,
            volatility_clustering=volatility_clustering,
            breadth_indicators=breadth_indicators
        )
    
    def _classify_primary_regime(self, indicators: RegimeIndicators) -> MarketRegime:
        """Classify the primary market regime"""
        
        # Crisis mode detection (highest priority)
        if (indicators.vix_level > self.regime_thresholds['vix_crisis'] or
            indicators.liquidity_conditions < self.regime_thresholds['liquidity_stress'] or
            indicators.correlation_regime > self.regime_thresholds['correlation_risk']):
            return MarketRegime.CRISIS_MODE
        
        # Bull/Bear market detection
        if indicators.market_trend > self.regime_thresholds['trend_strength_bull']:
            if indicators.vix_level < self.regime_thresholds['vix_low_vol']:
                return MarketRegime.BULL_MARKET
            else:
                return MarketRegime.TRENDING_UP
        elif indicators.market_trend < self.regime_thresholds['trend_strength_bear']:
            return MarketRegime.BEAR_MARKET
        
        # Volatility-based regimes
        if indicators.vix_level < self.regime_thresholds['vix_low_vol']:
            return MarketRegime.LOW_VOLATILITY
        elif indicators.vix_level > 25.0:  # High but not crisis
            return MarketRegime.HIGH_VOLATILITY
        
        # Default to sideways market
        return MarketRegime.SIDEWAYS_MARKET
    
    def _identify_secondary_regimes(
        self, indicators: RegimeIndicators, primary_regime: MarketRegime
    ) -> List[MarketRegime]:
        """Identify secondary regime characteristics"""
        secondary = []
        
        # Accumulation/Distribution patterns
        if indicators.breadth_indicators['volume_breadth'] > 0.7:
            secondary.append(MarketRegime.ACCUMULATION)
        elif indicators.breadth_indicators['volume_breadth'] < 0.3:
            secondary.append(MarketRegime.DISTRIBUTION)
        
        # Recovery mode
        if (primary_regime == MarketRegime.CRISIS_MODE and 
            indicators.risk_appetite > 0.6):
            secondary.append(MarketRegime.RECOVERY_MODE)
        
        # Breakout potential
        if indicators.volatility_clustering > 0.7:
            secondary.append(MarketRegime.BREAKOUT_MODE)
        
        return secondary
    
    def _calculate_regime_strength(
        self, indicators: RegimeIndicators, primary_regime: MarketRegime
    ) -> float:
        """Calculate confidence in regime classification"""
        
        strength_factors = []
        
        # Trend consistency
        if primary_regime in [MarketRegime.BULL_MARKET, MarketRegime.BEAR_MARKET]:
            strength_factors.append(abs(indicators.market_trend))
        
        # Volatility consistency
        if primary_regime == MarketRegime.LOW_VOLATILITY:
            strength_factors.append(1.0 - min(1.0, indicators.vix_level / 20.0))
        elif primary_regime == MarketRegime.HIGH_VOLATILITY:
            strength_factors.append(min(1.0, (indicators.vix_level - 20.0) / 20.0))
        
        # Breadth confirmation
        breadth_strength = np.mean(list(indicators.breadth_indicators.values()))
        strength_factors.append(breadth_strength)
        
        return np.mean(strength_factors) if strength_factors else 0.5
    
    def _assess_overall_favorability(
        self, primary_regime: MarketRegime, indicators: RegimeIndicators, strength: float
    ) -> RegimeFavorability:
        """Assess overall market favorability for event trading"""
        
        # Crisis mode = highly unfavorable
        if primary_regime == MarketRegime.CRISIS_MODE:
            return RegimeFavorability.HIGHLY_UNFAVORABLE
        
        # Bear market = unfavorable
        if primary_regime == MarketRegime.BEAR_MARKET:
            return RegimeFavorability.UNFAVORABLE
        
        # Bull market with strong conviction = highly favorable
        if primary_regime == MarketRegime.BULL_MARKET and strength > 0.7:
            return RegimeFavorability.HIGHLY_FAVORABLE
        
        # Low volatility = favorable for most events
        if primary_regime == MarketRegime.LOW_VOLATILITY:
            return RegimeFavorability.FAVORABLE
        
        # High volatility = unfavorable
        if primary_regime == MarketRegime.HIGH_VOLATILITY:
            return RegimeFavorability.UNFAVORABLE
        
        # Default to neutral
        return RegimeFavorability.NEUTRAL
    
    def _assess_event_specific_favorability(
        self, primary_regime: MarketRegime, secondary_regimes: List[MarketRegime],
        indicators: RegimeIndicators
    ) -> Dict[EventType, RegimeFavorability]:
        """Assess favorability for specific event types"""
        
        favorability = {}
        
        for event_type in EventType:
            if event_type in self.event_regime_favorability:
                base_favorability = self.event_regime_favorability[event_type].get(
                    primary_regime, RegimeFavorability.NEUTRAL
                )
                
                # Adjust based on secondary regimes and indicators
                adjusted_favorability = self._adjust_favorability_for_context(
                    base_favorability, secondary_regimes, indicators, event_type
                )
                
                favorability[event_type] = adjusted_favorability
            else:
                favorability[event_type] = RegimeFavorability.NEUTRAL
        
        return favorability
    
    def _meets_execution_threshold(
        self, favorability: RegimeFavorability, threshold: RegimeFavorability
    ) -> bool:
        """Check if favorability meets execution threshold"""
        
        favorability_order = [
            RegimeFavorability.HIGHLY_UNFAVORABLE,
            RegimeFavorability.UNFAVORABLE,
            RegimeFavorability.NEUTRAL,
            RegimeFavorability.FAVORABLE,
            RegimeFavorability.HIGHLY_FAVORABLE
        ]
        
        return favorability_order.index(favorability) >= favorability_order.index(threshold)
    
    def _calculate_favorability_score(
        self, regime_analysis: RegimeAnalysis, event_type: EventType,
        event_favorability: RegimeFavorability
    ) -> float:
        """Calculate numerical favorability score (0-1)"""
        
        favorability_scores = {
            RegimeFavorability.HIGHLY_UNFAVORABLE: 0.1,
            RegimeFavorability.UNFAVORABLE: 0.3,
            RegimeFavorability.NEUTRAL: 0.5,
            RegimeFavorability.FAVORABLE: 0.7,
            RegimeFavorability.HIGHLY_FAVORABLE: 0.9
        }
        
        base_score = favorability_scores[event_favorability]
        
        # Adjust for regime strength
        strength_adjustment = (regime_analysis.regime_strength - 0.5) * 0.2
        
        # Adjust for risk metrics
        risk_adjustment = -(regime_analysis.tail_risk + regime_analysis.correlation_risk) * 0.1
        
        final_score = base_score + strength_adjustment + risk_adjustment
        return max(0.0, min(1.0, final_score))
    
    def _calculate_risk_adjustment_factor(self, regime_analysis: RegimeAnalysis) -> float:
        """Calculate risk adjustment factor for position sizing"""
        
        base_adjustment = 1.0
        
        # Adjust for volatility
        vol_adjustment = 1.0 - (regime_analysis.volatility_percentile - 0.5) * 0.3
        
        # Adjust for tail risk
        tail_adjustment = 1.0 - regime_analysis.tail_risk * 0.4
        
        # Adjust for correlation risk
        corr_adjustment = 1.0 - regime_analysis.correlation_risk * 0.2
        
        # Adjust for liquidity risk
        liquidity_adjustment = 1.0 - regime_analysis.liquidity_risk * 0.3
        
        return max(0.2, min(1.5, base_adjustment * vol_adjustment * tail_adjustment * 
                           corr_adjustment * liquidity_adjustment))
    
    def _calculate_position_size_modifier(
        self, regime_analysis: RegimeAnalysis, event_favorability: RegimeFavorability,
        risk_adjustment: float
    ) -> float:
        """Calculate position size modifier based on regime"""
        
        favorability_modifiers = {
            RegimeFavorability.HIGHLY_UNFAVORABLE: 0.0,  # No position
            RegimeFavorability.UNFAVORABLE: 0.3,
            RegimeFavorability.NEUTRAL: 0.6,
            RegimeFavorability.FAVORABLE: 1.0,
            RegimeFavorability.HIGHLY_FAVORABLE: 1.2
        }
        
        base_modifier = favorability_modifiers[event_favorability]
        
        # Apply risk adjustment
        final_modifier = base_modifier * risk_adjustment
        
        # Consider regime strength
        strength_bonus = regime_analysis.regime_strength * 0.2
        
        return max(0.0, min(1.5, final_modifier + strength_bonus))
    
    def _generate_decision_reasoning(
        self, regime_analysis: RegimeAnalysis, event_type: EventType,
        event_favorability: RegimeFavorability, execution_approved: bool
    ) -> Tuple[List[str], List[str]]:
        """Generate human-readable reasoning for the decision"""
        
        approval_reasons = []
        rejection_reasons = []
        
        if execution_approved:
            approval_reasons.append(f"Event favorability: {event_favorability.value}")
            approval_reasons.append(f"Primary regime: {regime_analysis.primary_regime.value}")
            
            if regime_analysis.regime_strength > 0.7:
                approval_reasons.append(f"Strong regime conviction ({regime_analysis.regime_strength:.1%})")
            
            if regime_analysis.tail_risk < 0.3:
                approval_reasons.append("Low tail risk environment")
                
            if regime_analysis.liquidity_risk < 0.4:
                approval_reasons.append("Adequate market liquidity")
        else:
            rejection_reasons.append(f"Event favorability too low: {event_favorability.value}")
            rejection_reasons.append(f"Unfavorable regime: {regime_analysis.primary_regime.value}")
            
            if regime_analysis.tail_risk > 0.6:
                rejection_reasons.append("High tail risk environment")
                
            if regime_analysis.correlation_risk > 0.7:
                rejection_reasons.append("High correlation risk")
                
            if regime_analysis.liquidity_risk > 0.6:
                rejection_reasons.append("Poor liquidity conditions")
        
        return approval_reasons, rejection_reasons
    
    def _identify_risk_mitigation_requirements(
        self, regime_analysis: RegimeAnalysis, event_type: EventType
    ) -> List[str]:
        """Identify required risk mitigation measures"""
        
        mitigation = []
        
        if regime_analysis.volatility_percentile > 0.8:
            mitigation.append("Reduce position size due to high volatility")
            mitigation.append("Implement tighter stop losses")
        
        if regime_analysis.tail_risk > 0.5:
            mitigation.append("Consider protective puts or hedging")
            mitigation.append("Reduce holding period")
        
        if regime_analysis.correlation_risk > 0.6:
            mitigation.append("Monitor correlation breakdown")
            mitigation.append("Avoid concentration in correlated assets")
        
        if regime_analysis.liquidity_risk > 0.5:
            mitigation.append("Use limit orders only")
            mitigation.append("Avoid after-hours trading")
        
        if regime_analysis.regime_transition_probability > 0.7:
            mitigation.append("Monitor for regime change signals")
            mitigation.append("Prepare for rapid position adjustment")
        
        return mitigation
    
    # Helper methods for calculations
    def _calculate_market_trend(self, market_data: Dict, lookback_periods: Dict) -> float:
        """Calculate overall market trend strength"""
        # In a real implementation, this would analyze moving averages, momentum, etc.
        return market_data.get('trend_score', 0.0)  # -1 to 1
    
    def _calculate_risk_appetite(self, market_data: Dict) -> float:
        """Calculate risk appetite indicators"""
        # In a real implementation, this would use risk-on/risk-off metrics
        return market_data.get('risk_appetite', 0.5)  # 0 to 1
    
    def _calculate_volatility_clustering(self, market_data: Dict) -> float:
        """Calculate volatility clustering measure"""
        # In a real implementation, this would use GARCH-type models
        return market_data.get('vol_clustering', 0.5)  # 0 to 1
    
    def _estimate_regime_duration(self, market_data: Dict, regime: MarketRegime) -> timedelta:
        """Estimate how long the current regime has been in place"""
        # In a real implementation, this would track regime changes
        return timedelta(days=market_data.get('regime_duration_days', 10))
    
    def _calculate_transition_probability(self, indicators: RegimeIndicators) -> float:
        """Calculate probability of regime transition"""
        # In a real implementation, this would use regime switching models
        return 0.2  # 20% chance of regime change
    
    def _calculate_volatility_percentile(self, indicators: RegimeIndicators) -> float:
        """Calculate current volatility percentile"""
        # In a real implementation, this would use historical VIX data
        return min(1.0, indicators.vix_level / 40.0)
    
    def _calculate_trend_strength(self, indicators: RegimeIndicators) -> float:
        """Calculate trend strength"""
        return abs(indicators.market_trend)
    
    def _calculate_mean_reversion_tendency(self, indicators: RegimeIndicators) -> float:
        """Calculate mean reversion tendency"""
        # Higher correlation typically means more mean reversion
        return indicators.correlation_regime
    
    def _calculate_breakout_potential(self, indicators: RegimeIndicators) -> float:
        """Calculate potential for large breakout moves"""
        # Low volatility + high clustering = breakout potential
        if indicators.vix_level < 15 and indicators.volatility_clustering > 0.7:
            return 0.8
        return 0.3
    
    def _assess_tail_risk(self, indicators: RegimeIndicators) -> float:
        """Assess tail risk probability"""
        # VIX and correlation are key tail risk indicators
        vix_risk = min(1.0, indicators.vix_level / 50.0)
        correlation_risk = indicators.correlation_regime
        return (vix_risk + correlation_risk) / 2
    
    def _assess_correlation_risk(self, indicators: RegimeIndicators) -> float:
        """Assess correlation risk"""
        return indicators.correlation_regime
    
    def _assess_liquidity_risk(self, indicators: RegimeIndicators) -> float:
        """Assess liquidity risk"""
        return 1.0 - indicators.liquidity_conditions
    
    def _adjust_favorability_for_context(
        self, base_favorability: RegimeFavorability, secondary_regimes: List[MarketRegime],
        indicators: RegimeIndicators, event_type: EventType
    ) -> RegimeFavorability:
        """Adjust favorability based on additional context"""
        
        # For now, return base favorability
        # In a real implementation, this would apply sophisticated adjustments
        return base_favorability
    
    def _recommend_entry_timing(
        self, regime_analysis: RegimeAnalysis, event_type: EventType
    ) -> Optional[str]:
        """Recommend optimal entry timing"""
        
        if regime_analysis.primary_regime == MarketRegime.HIGH_VOLATILITY:
            return "Wait for volatility compression"
        elif regime_analysis.breakout_potential > 0.7:
            return "Enter on initial breakout confirmation"
        elif regime_analysis.mean_reversion_tendency > 0.7:
            return "Enter on pullback to support"
        else:
            return "Enter at market open"
    
    def _calculate_regime_stop_loss(
        self, regime_analysis: RegimeAnalysis, risk_adjustment: float
    ) -> Optional[float]:
        """Calculate regime-based stop loss adjustment"""
        
        if regime_analysis.volatility_percentile > 0.8:
            return 0.85  # Tighter stops in high vol
        elif regime_analysis.primary_regime == MarketRegime.CRISIS_MODE:
            return 0.80  # Very tight stops in crisis
        else:
            return None  # Use standard stops
    
    def _calculate_profit_target_adjustment(
        self, regime_analysis: RegimeAnalysis, event_favorability: RegimeFavorability
    ) -> Optional[float]:
        """Calculate profit target adjustment based on regime"""
        
        if event_favorability == RegimeFavorability.HIGHLY_FAVORABLE:
            return 1.25  # 25% higher targets
        elif regime_analysis.breakout_potential > 0.7:
            return 1.20  # 20% higher targets
        elif regime_analysis.primary_regime == MarketRegime.HIGH_VOLATILITY:
            return 0.85  # 15% lower targets
        else:
            return None  # Standard targets