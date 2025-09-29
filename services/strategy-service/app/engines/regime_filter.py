"""Market Regime Filter

Advanced regime detection and filtering system that analyzes market conditions
to determine when to execute trades based on favorable regime tags.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

import httpx
import numpy as np

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Market regime classifications."""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"
    SIDEWAYS_CHOPPY = "sideways_choppy"
    CRISIS_MODE = "crisis_mode"
    RECOVERY_MODE = "recovery_mode"
    DISTRIBUTION = "distribution"
    ACCUMULATION = "accumulation"


class StrategyType(Enum):
    """Strategy types that require regime filtering."""
    EVENT_DRIVEN = "event_driven"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    BREAKOUT = "breakout"
    GAP_TRADING = "gap_trading"


@dataclass
class RegimeMetrics:
    """Market regime quantitative metrics."""
    symbol: str
    price_trend_20d: float          # 20-day price trend slope
    price_trend_50d: float          # 50-day price trend slope
    volatility_regime: float        # Volatility percentile (0-100)
    volume_regime: float            # Volume percentile (0-100)
    correlation_spy: float          # Correlation with SPY
    vix_level: float               # VIX level
    vix_term_structure: float      # VIX term structure slope
    breadth_ratio: float           # Market breadth ratio
    sector_rotation: float         # Sector rotation intensity
    momentum_factor: float         # Cross-sectional momentum
    value_factor: float            # Value factor performance
    quality_factor: float          # Quality factor performance
    timestamp: datetime


@dataclass
class RegimeState:
    """Current market regime state."""
    primary_regime: RegimeType
    secondary_regime: Optional[RegimeType]
    confidence: float              # 0.0 to 1.0
    regime_duration_days: int
    regime_strength: float         # -1.0 to 1.0
    transition_probability: float   # Probability of regime change
    favorable_strategies: List[StrategyType]
    unfavorable_strategies: List[StrategyType]
    metrics: RegimeMetrics
    generated_at: datetime


@dataclass
class RegimeFilter:
    """Filter criteria for strategy execution."""
    strategy_type: StrategyType
    required_regimes: List[RegimeType]
    prohibited_regimes: List[RegimeType]
    min_confidence: float
    max_transition_probability: float
    min_regime_duration_days: int
    custom_conditions: Optional[Dict[str, Any]] = None


@dataclass
class FilterResult:
    """Result of regime filtering."""
    symbol: str
    strategy_type: StrategyType
    approved: bool
    confidence: float
    regime_state: RegimeState
    filter_criteria: RegimeFilter
    rejection_reasons: List[str]
    score_breakdown: Dict[str, float]
    recommendation: str
    timestamp: datetime


class RegimeAnalyzer:
    """Analyzes market regimes and provides filtering capabilities."""
    
    def __init__(self, 
                 market_data_url: str = "http://localhost:8002",
                 analysis_service_url: str = "http://localhost:8003"):
        self.market_data_url = market_data_url
        self.analysis_service_url = analysis_service_url
        self.client = None
        
        # Regime detection thresholds
        self.regime_thresholds = {
            "trend_slope_threshold": 0.001,    # Daily trend slope
            "volatility_high": 75,             # High volatility percentile
            "volatility_low": 25,              # Low volatility percentile
            "volume_high": 80,                 # High volume percentile
            "correlation_threshold": 0.7,       # Market correlation threshold
            "vix_crisis": 30,                  # VIX crisis level
            "vix_complacency": 15,             # VIX complacency level
            "breadth_strong": 0.6,             # Strong market breadth
            "breadth_weak": 0.4                # Weak market breadth
        }
        
        # Strategy regime preferences
        self.strategy_preferences = {
            StrategyType.EVENT_DRIVEN: {
                "favorable": [RegimeType.LOW_VOLATILITY, RegimeType.BULL_TRENDING, RegimeType.RECOVERY_MODE],
                "unfavorable": [RegimeType.CRISIS_MODE, RegimeType.HIGH_VOLATILITY],
                "min_confidence": 0.6,
                "max_transition_prob": 0.3
            },
            StrategyType.MOMENTUM: {
                "favorable": [RegimeType.BULL_TRENDING, RegimeType.BEAR_TRENDING],
                "unfavorable": [RegimeType.SIDEWAYS_CHOPPY, RegimeType.DISTRIBUTION],
                "min_confidence": 0.7,
                "max_transition_prob": 0.2
            },
            StrategyType.MEAN_REVERSION: {
                "favorable": [RegimeType.SIDEWAYS_CHOPPY, RegimeType.LOW_VOLATILITY, RegimeType.ACCUMULATION],
                "unfavorable": [RegimeType.BULL_TRENDING, RegimeType.BEAR_TRENDING, RegimeType.CRISIS_MODE],
                "min_confidence": 0.65,
                "max_transition_prob": 0.25
            },
            StrategyType.TREND_FOLLOWING: {
                "favorable": [RegimeType.BULL_TRENDING, RegimeType.BEAR_TRENDING, RegimeType.RECOVERY_MODE],
                "unfavorable": [RegimeType.SIDEWAYS_CHOPPY, RegimeType.DISTRIBUTION],
                "min_confidence": 0.75,
                "max_transition_prob": 0.15
            },
            StrategyType.BREAKOUT: {
                "favorable": [RegimeType.LOW_VOLATILITY, RegimeType.ACCUMULATION],
                "unfavorable": [RegimeType.HIGH_VOLATILITY, RegimeType.CRISIS_MODE],
                "min_confidence": 0.8,
                "max_transition_prob": 0.1
            },
            StrategyType.GAP_TRADING: {
                "favorable": [RegimeType.HIGH_VOLATILITY, RegimeType.CRISIS_MODE, RegimeType.RECOVERY_MODE],
                "unfavorable": [RegimeType.LOW_VOLATILITY, RegimeType.SIDEWAYS_CHOPPY],
                "min_confidence": 0.6,
                "max_transition_prob": 0.4
            }
        }
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def analyze_regime(self, symbol: str = "SPY") -> Optional[RegimeState]:
        """Analyze current market regime for the symbol."""
        try:
            # Gather regime metrics
            metrics = await self._gather_regime_metrics(symbol)
            if not metrics:
                return None
            
            # Classify primary regime
            primary_regime = self._classify_primary_regime(metrics)
            
            # Classify secondary regime
            secondary_regime = self._classify_secondary_regime(metrics, primary_regime)
            
            # Calculate confidence
            confidence = self._calculate_regime_confidence(metrics, primary_regime)
            
            # Calculate regime duration
            duration_days = await self._estimate_regime_duration(symbol, primary_regime)
            
            # Calculate regime strength
            strength = self._calculate_regime_strength(metrics, primary_regime)
            
            # Calculate transition probability
            transition_prob = self._calculate_transition_probability(metrics)
            
            # Determine favorable/unfavorable strategies
            favorable, unfavorable = self._determine_strategy_alignment(primary_regime)
            
            return RegimeState(
                primary_regime=primary_regime,
                secondary_regime=secondary_regime,
                confidence=confidence,
                regime_duration_days=duration_days,
                regime_strength=strength,
                transition_probability=transition_prob,
                favorable_strategies=favorable,
                unfavorable_strategies=unfavorable,
                metrics=metrics,
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze regime for {symbol}: {e}")
            return None
    
    async def filter_strategy(self, symbol: str, strategy_type: StrategyType, 
                            custom_filter: Optional[RegimeFilter] = None) -> FilterResult:
        """Filter strategy execution based on current regime."""
        try:
            # Get current regime state
            regime_state = await self.analyze_regime(symbol)
            if not regime_state:
                return FilterResult(
                    symbol=symbol,
                    strategy_type=strategy_type,
                    approved=False,
                    confidence=0.0,
                    regime_state=None,
                    filter_criteria=custom_filter,
                    rejection_reasons=["Failed to analyze regime"],
                    score_breakdown={},
                    recommendation="Cannot determine regime - avoid trading",
                    timestamp=datetime.utcnow()
                )
            
            # Use custom filter or default
            filter_criteria = custom_filter or self._build_default_filter(strategy_type)
            
            # Apply filter
            approved, reasons, scores = self._apply_regime_filter(regime_state, filter_criteria)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(regime_state, strategy_type, approved)
            
            return FilterResult(
                symbol=symbol,
                strategy_type=strategy_type,
                approved=approved,
                confidence=regime_state.confidence,
                regime_state=regime_state,
                filter_criteria=filter_criteria,
                rejection_reasons=reasons,
                score_breakdown=scores,
                recommendation=recommendation,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to filter strategy {strategy_type} for {symbol}: {e}")
            return FilterResult(
                symbol=symbol,
                strategy_type=strategy_type,
                approved=False,
                confidence=0.0,
                regime_state=None,
                filter_criteria=custom_filter,
                rejection_reasons=[f"Filter error: {str(e)}"],
                score_breakdown={},
                recommendation="Error in regime analysis - avoid trading",
                timestamp=datetime.utcnow()
            )
    
    async def _gather_regime_metrics(self, symbol: str) -> Optional[RegimeMetrics]:
        """Gather all metrics needed for regime analysis."""
        try:
            # Get price history for trend analysis
            price_data = await self._get_price_history(symbol, days=100)
            if not price_data:
                return None
            
            # Calculate price trends
            prices = np.array([p["close"] for p in price_data])
            dates = np.arange(len(prices))
            
            # 20-day trend
            trend_20d = np.polyfit(dates[-20:], prices[-20:], 1)[0] if len(prices) >= 20 else 0
            
            # 50-day trend
            trend_50d = np.polyfit(dates[-50:], prices[-50:], 1)[0] if len(prices) >= 50 else 0
            
            # Volatility regime
            returns = np.diff(prices) / prices[:-1]
            current_vol = np.std(returns[-20:]) if len(returns) >= 20 else 0
            historical_vols = [np.std(returns[i:i+20]) for i in range(len(returns)-20)]
            vol_percentile = (np.sum(np.array(historical_vols) < current_vol) / len(historical_vols)) * 100
            
            # Volume regime
            volumes = np.array([p["volume"] for p in price_data])
            current_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
            vol_percentile = (np.sum(volumes < current_volume) / len(volumes)) * 100
            
            # Market data
            market_data = await self._get_market_indicators()
            
            return RegimeMetrics(
                symbol=symbol,
                price_trend_20d=trend_20d,
                price_trend_50d=trend_50d,
                volatility_regime=vol_percentile,
                volume_regime=vol_percentile,
                correlation_spy=market_data.get("correlation_spy", 0.5),
                vix_level=market_data.get("vix", 20),
                vix_term_structure=market_data.get("vix_term_structure", 0),
                breadth_ratio=market_data.get("breadth_ratio", 0.5),
                sector_rotation=market_data.get("sector_rotation", 0.5),
                momentum_factor=market_data.get("momentum_factor", 0),
                value_factor=market_data.get("value_factor", 0),
                quality_factor=market_data.get("quality_factor", 0),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to gather regime metrics: {e}")
            return None
    
    async def _get_price_history(self, symbol: str, days: int) -> Optional[List[Dict[str, Any]]]:
        """Get historical price data."""
        try:
            response = await self.client.get(
                f"{self.market_data_url}/history/{symbol}",
                params={"period": f"{days}d", "interval": "1d"}
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return None
    
    async def _get_market_indicators(self) -> Dict[str, Any]:
        """Get broad market indicators for regime analysis."""
        try:
            response = await self.client.get(
                f"{self.analysis_service_url}/market/indicators"
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            # Return default values if service unavailable
            return {
                "vix": 20.0,
                "vix_term_structure": 0.0,
                "breadth_ratio": 0.5,
                "sector_rotation": 0.5,
                "correlation_spy": 0.5,
                "momentum_factor": 0.0,
                "value_factor": 0.0,
                "quality_factor": 0.0
            }
    
    def _classify_primary_regime(self, metrics: RegimeMetrics) -> RegimeType:
        """Classify the primary market regime."""
        # Crisis mode detection
        if metrics.vix_level > self.regime_thresholds["vix_crisis"]:
            return RegimeType.CRISIS_MODE
        
        # Trend detection
        if (metrics.price_trend_20d > self.regime_thresholds["trend_slope_threshold"] and 
            metrics.price_trend_50d > self.regime_thresholds["trend_slope_threshold"]):
            return RegimeType.BULL_TRENDING
        elif (metrics.price_trend_20d < -self.regime_thresholds["trend_slope_threshold"] and 
              metrics.price_trend_50d < -self.regime_thresholds["trend_slope_threshold"]):
            return RegimeType.BEAR_TRENDING
        
        # Volatility regime
        if metrics.volatility_regime > self.regime_thresholds["volatility_high"]:
            return RegimeType.HIGH_VOLATILITY
        elif metrics.volatility_regime < self.regime_thresholds["volatility_low"]:
            return RegimeType.LOW_VOLATILITY
        
        # Market phase detection
        if metrics.breadth_ratio > self.regime_thresholds["breadth_strong"] and metrics.volume_regime > 60:
            return RegimeType.ACCUMULATION
        elif metrics.breadth_ratio < self.regime_thresholds["breadth_weak"] and metrics.volume_regime > 60:
            return RegimeType.DISTRIBUTION
        
        # Recovery detection
        if (metrics.vix_level < 25 and metrics.price_trend_20d > 0 and 
            metrics.breadth_ratio > 0.5):
            return RegimeType.RECOVERY_MODE
        
        # Default to sideways/choppy
        return RegimeType.SIDEWAYS_CHOPPY
    
    def _classify_secondary_regime(self, metrics: RegimeMetrics, primary: RegimeType) -> Optional[RegimeType]:
        """Classify secondary regime characteristics."""
        secondary_candidates = []
        
        # Check for volatility overlay
        if primary != RegimeType.HIGH_VOLATILITY and metrics.volatility_regime > 70:
            secondary_candidates.append(RegimeType.HIGH_VOLATILITY)
        elif primary != RegimeType.LOW_VOLATILITY and metrics.volatility_regime < 30:
            secondary_candidates.append(RegimeType.LOW_VOLATILITY)
        
        # Check for accumulation/distribution overlay
        if primary not in [RegimeType.ACCUMULATION, RegimeType.DISTRIBUTION]:
            if metrics.breadth_ratio > 0.7 and metrics.volume_regime > 70:
                secondary_candidates.append(RegimeType.ACCUMULATION)
            elif metrics.breadth_ratio < 0.3 and metrics.volume_regime > 70:
                secondary_candidates.append(RegimeType.DISTRIBUTION)
        
        return secondary_candidates[0] if secondary_candidates else None
    
    def _calculate_regime_confidence(self, metrics: RegimeMetrics, regime: RegimeType) -> float:
        """Calculate confidence in regime classification."""
        base_confidence = 0.5
        
        # VIX confirmation
        if regime == RegimeType.CRISIS_MODE and metrics.vix_level > 35:
            base_confidence += 0.3
        elif regime == RegimeType.LOW_VOLATILITY and metrics.vix_level < 15:
            base_confidence += 0.2
        
        # Trend consistency
        if regime in [RegimeType.BULL_TRENDING, RegimeType.BEAR_TRENDING]:
            trend_consistency = abs(metrics.price_trend_20d - metrics.price_trend_50d)
            if trend_consistency < 0.0005:  # Very consistent trends
                base_confidence += 0.2
        
        # Volume confirmation
        if metrics.volume_regime > 70 or metrics.volume_regime < 30:
            base_confidence += 0.1
        
        # Breadth confirmation
        if abs(metrics.breadth_ratio - 0.5) > 0.2:
            base_confidence += 0.1
        
        return max(0.1, min(0.95, base_confidence))
    
    async def _estimate_regime_duration(self, symbol: str, regime: RegimeType) -> int:
        """Estimate how long the current regime has been in place."""
        # Simplified - in production, this would analyze historical regime changes
        try:
            # For now, return default durations based on regime type
            default_durations = {
                RegimeType.CRISIS_MODE: 30,
                RegimeType.BULL_TRENDING: 60,
                RegimeType.BEAR_TRENDING: 45,
                RegimeType.HIGH_VOLATILITY: 15,
                RegimeType.LOW_VOLATILITY: 45,
                RegimeType.SIDEWAYS_CHOPPY: 30,
                RegimeType.RECOVERY_MODE: 20,
                RegimeType.ACCUMULATION: 40,
                RegimeType.DISTRIBUTION: 35
            }
            return default_durations.get(regime, 30)
        except Exception:
            return 30
    
    def _calculate_regime_strength(self, metrics: RegimeMetrics, regime: RegimeType) -> float:
        """Calculate the strength of the current regime."""
        if regime == RegimeType.BULL_TRENDING:
            return min(1.0, metrics.price_trend_20d * 1000)  # Scale trend slope
        elif regime == RegimeType.BEAR_TRENDING:
            return max(-1.0, metrics.price_trend_20d * 1000)
        elif regime == RegimeType.HIGH_VOLATILITY:
            return min(1.0, (metrics.volatility_regime - 50) / 50)
        elif regime == RegimeType.LOW_VOLATILITY:
            return min(1.0, (50 - metrics.volatility_regime) / 50)
        elif regime == RegimeType.CRISIS_MODE:
            return min(1.0, (metrics.vix_level - 20) / 30)
        else:
            return 0.0
    
    def _calculate_transition_probability(self, metrics: RegimeMetrics) -> float:
        """Calculate probability of regime transition."""
        # Factors that increase transition probability
        transition_score = 0.0
        
        # VIX term structure inversion
        if metrics.vix_term_structure < -0.1:
            transition_score += 0.3
        
        # Extreme volatility levels
        if metrics.volatility_regime > 90 or metrics.volatility_regime < 10:
            transition_score += 0.2
        
        # Diverging trends
        trend_divergence = abs(metrics.price_trend_20d - metrics.price_trend_50d)
        if trend_divergence > 0.001:
            transition_score += 0.2
        
        # Factor rotation
        if abs(metrics.momentum_factor) > 0.02 or abs(metrics.value_factor) > 0.02:
            transition_score += 0.1
        
        return min(0.8, transition_score)
    
    def _determine_strategy_alignment(self, regime: RegimeType) -> Tuple[List[StrategyType], List[StrategyType]]:
        """Determine which strategies are favored/disfavored in this regime."""
        favorable = []
        unfavorable = []
        
        for strategy, prefs in self.strategy_preferences.items():
            if regime in prefs["favorable"]:
                favorable.append(strategy)
            elif regime in prefs["unfavorable"]:
                unfavorable.append(strategy)
        
        return favorable, unfavorable
    
    def _build_default_filter(self, strategy_type: StrategyType) -> RegimeFilter:
        """Build default filter criteria for a strategy type."""
        prefs = self.strategy_preferences.get(strategy_type, {})
        
        return RegimeFilter(
            strategy_type=strategy_type,
            required_regimes=prefs.get("favorable", []),
            prohibited_regimes=prefs.get("unfavorable", []),
            min_confidence=prefs.get("min_confidence", 0.6),
            max_transition_probability=prefs.get("max_transition_prob", 0.3),
            min_regime_duration_days=5
        )
    
    def _apply_regime_filter(self, regime_state: RegimeState, 
                           filter_criteria: RegimeFilter) -> Tuple[bool, List[str], Dict[str, float]]:
        """Apply regime filter and return approval status."""
        reasons = []
        scores = {}
        
        # Check required regimes
        if filter_criteria.required_regimes:
            if regime_state.primary_regime not in filter_criteria.required_regimes:
                if regime_state.secondary_regime not in filter_criteria.required_regimes:
                    reasons.append(f"Regime {regime_state.primary_regime.value} not in required regimes")
        
        # Check prohibited regimes
        if regime_state.primary_regime in filter_criteria.prohibited_regimes:
            reasons.append(f"Regime {regime_state.primary_regime.value} is prohibited")
        
        if (regime_state.secondary_regime and 
            regime_state.secondary_regime in filter_criteria.prohibited_regimes):
            reasons.append(f"Secondary regime {regime_state.secondary_regime.value} is prohibited")
        
        # Check confidence
        scores["confidence"] = regime_state.confidence
        if regime_state.confidence < filter_criteria.min_confidence:
            reasons.append(f"Regime confidence {regime_state.confidence:.2f} below threshold {filter_criteria.min_confidence:.2f}")
        
        # Check transition probability
        scores["transition_probability"] = regime_state.transition_probability
        if regime_state.transition_probability > filter_criteria.max_transition_probability:
            reasons.append(f"Transition probability {regime_state.transition_probability:.2f} above threshold {filter_criteria.max_transition_probability:.2f}")
        
        # Check regime duration
        scores["regime_duration"] = regime_state.regime_duration_days
        if regime_state.regime_duration_days < filter_criteria.min_regime_duration_days:
            reasons.append(f"Regime duration {regime_state.regime_duration_days}d below minimum {filter_criteria.min_regime_duration_days}d")
        
        # Calculate composite score
        scores["composite"] = (
            scores["confidence"] * 0.5 +
            (1 - scores["transition_probability"]) * 0.3 +
            min(1.0, scores["regime_duration"] / 30) * 0.2
        )
        
        approved = len(reasons) == 0
        return approved, reasons, scores
    
    def _generate_recommendation(self, regime_state: RegimeState, 
                                strategy_type: StrategyType, approved: bool) -> str:
        """Generate trading recommendation based on regime analysis."""
        if approved:
            strength_desc = "strong" if regime_state.regime_strength > 0.5 else "moderate"
            return f"APPROVED: {strength_desc} {regime_state.primary_regime.value} regime favors {strategy_type.value} strategies"
        else:
            return f"REJECTED: Current {regime_state.primary_regime.value} regime not suitable for {strategy_type.value} strategies"


async def test_regime_filter():
    """Test the regime filtering functionality."""
    async with RegimeAnalyzer() as analyzer:
        # Test regime analysis
        regime_state = await analyzer.analyze_regime("SPY")
        
        if regime_state:
            print(f"Current regime: {regime_state.primary_regime.value}")
            print(f"Confidence: {regime_state.confidence:.2%}")
            print(f"Duration: {regime_state.regime_duration_days} days")
            print(f"Transition probability: {regime_state.transition_probability:.2%}")
            
            # Test strategy filtering
            for strategy in StrategyType:
                result = await analyzer.filter_strategy("SPY", strategy)
                status = "✅ APPROVED" if result.approved else "❌ REJECTED"
                print(f"{status}: {strategy.value} - {result.recommendation}")
        else:
            print("Failed to analyze regime")


if __name__ == "__main__":
    asyncio.run(test_regime_filter())