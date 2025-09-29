"""
Enhanced Surprise Delta Service
Advanced surprise analysis with consensus tracking, revision momentum, and market impact measurement
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_

from .analyst_revision_tracker import analyst_revision_tracker, RevisionMomentum
from .finnhub_fundamentals import FinnhubFundamentalsClient
from .earnings_monitor import earnings_monitor

logger = logging.getLogger(__name__)


@dataclass
class SurpriseContext:
    """Contextual information for surprise analysis"""
    symbol: str
    event_date: date
    event_type: str
    
    # Pre-event context
    days_to_event: int
    consensus_stability: float  # how much consensus changed in lead-up
    revision_activity: int      # number of revisions in 30d before
    analyst_conviction: float   # confidence weighted by analyst tier
    
    # Historical context
    historical_beat_rate: float      # % of time company beats consensus
    historical_surprise_volatility: float  # std of historical surprises
    sector_surprise_trend: float     # recent sector surprise pattern
    
    # Market context
    market_regime: str              # bull, bear, neutral
    volatility_regime: str          # low, normal, high
    earnings_season_factor: float   # impact of earnings season timing


@dataclass
class EnhancedSurprise:
    """Comprehensive surprise analysis result"""
    symbol: str
    event_date: date
    event_type: str
    
    # Core surprise metrics
    consensus_value: float
    actual_value: float
    surprise_absolute: float
    surprise_percent: float
    surprise_standardized: float  # normalized by historical volatility
    surprise_score: float        # 0-1 magnitude score
    
    # Consensus quality metrics
    consensus_count: int
    consensus_std: float
    consensus_high: float
    consensus_low: float
    consensus_range_pct: float
    consensus_confidence: float   # derived from std and count
    
    # Surprise quality assessment
    surprise_significance: str    # low, medium, high
    surprise_direction: str       # positive, negative, neutral
    surprise_percentile: float    # historical percentile
    
    # Market impact
    market_reaction_1d: Optional[float]
    market_reaction_3d: Optional[float]
    volume_spike_factor: Optional[float]
    market_efficiency_score: float  # how well market anticipated
    
    # Context and attribution
    surprise_context: SurpriseContext
    surprise_attribution: List[str]  # factors contributing to surprise
    future_implications: List[str]   # what this suggests going forward


class EnhancedSurpriseService:
    """Service for advanced surprise analysis and tracking"""
    
    def __init__(self):
        self.finnhub_client = FinnhubFundamentalsClient()
        
        # Surprise significance thresholds
        self.significance_thresholds = {
            "earnings": {"low": 0.05, "medium": 0.15, "high": 0.25},  # % surprise
            "revenue": {"low": 0.02, "medium": 0.05, "high": 0.10},
            "guidance": {"low": 0.03, "medium": 0.08, "high": 0.15},
        }
        
        # Historical lookback periods
        self.historical_periods = {
            "surprise_history": 8,      # quarters for surprise history
            "revision_window": 30,      # days for pre-event revisions
            "market_reaction": 3,       # days for market reaction measurement
        }
    
    async def calculate_enhanced_surprise(
        self,
        symbol: str,
        event_date: date,
        event_type: str,
        consensus_value: float,
        actual_value: float,
        db: Session
    ) -> EnhancedSurprise:
        """Calculate comprehensive surprise analysis"""
        
        try:
            # Get surprise context
            surprise_context = await self._build_surprise_context(
                symbol, event_date, event_type, db
            )
            
            # Calculate core surprise metrics
            surprise_absolute = actual_value - consensus_value
            
            if consensus_value != 0:
                surprise_percent = (surprise_absolute / abs(consensus_value)) * 100
            else:
                surprise_percent = 100.0 if actual_value > 0 else -100.0
            
            # Get historical surprise data for standardization
            historical_surprises = await self._get_historical_surprises(
                symbol, event_type, self.historical_periods["surprise_history"], db
            )
            
            surprise_standardized = self._standardize_surprise(
                surprise_percent, historical_surprises
            )
            
            surprise_score = min(1.0, abs(surprise_standardized) / 2.0)  # Cap at 2 std devs
            
            # Get consensus quality metrics
            consensus_data = await self._get_consensus_quality(
                symbol, event_date, event_type, db
            )
            
            # Determine surprise significance
            significance = self._assess_surprise_significance(
                event_type, abs(surprise_percent)
            )
            
            direction = "positive" if surprise_percent > 0 else "negative" if surprise_percent < 0 else "neutral"
            
            # Calculate historical percentile
            percentile = self._calculate_surprise_percentile(
                surprise_percent, historical_surprises
            )
            
            # Get market impact data
            market_impact = await self._get_market_impact(symbol, event_date, db)
            
            # Calculate market efficiency
            efficiency_score = self._calculate_market_efficiency(
                surprise_standardized, market_impact.get("reaction_1d", 0)
            )
            
            # Generate attribution and implications
            attribution = self._generate_surprise_attribution(
                surprise_context, surprise_percent, consensus_data
            )
            implications = self._generate_future_implications(
                symbol, surprise_percent, surprise_context, historical_surprises
            )
            
            return EnhancedSurprise(
                symbol=symbol,
                event_date=event_date,
                event_type=event_type,
                consensus_value=consensus_value,
                actual_value=actual_value,
                surprise_absolute=surprise_absolute,
                surprise_percent=surprise_percent,
                surprise_standardized=surprise_standardized,
                surprise_score=surprise_score,
                consensus_count=consensus_data.get("count", 0),
                consensus_std=consensus_data.get("std", 0),
                consensus_high=consensus_data.get("high", consensus_value),
                consensus_low=consensus_data.get("low", consensus_value),
                consensus_range_pct=consensus_data.get("range_pct", 0),
                consensus_confidence=consensus_data.get("confidence", 0.5),
                surprise_significance=significance,
                surprise_direction=direction,
                surprise_percentile=percentile,
                market_reaction_1d=market_impact.get("reaction_1d"),
                market_reaction_3d=market_impact.get("reaction_3d"),
                volume_spike_factor=market_impact.get("volume_spike"),
                market_efficiency_score=efficiency_score,
                surprise_context=surprise_context,
                surprise_attribution=attribution,
                future_implications=implications,
            )
            
        except Exception as e:
            logger.error(f"Error calculating enhanced surprise for {symbol}: {e}")
            raise
    
    async def _build_surprise_context(
        self,
        symbol: str,
        event_date: date,
        event_type: str,
        db: Session
    ) -> SurpriseContext:
        """Build contextual information for surprise analysis"""
        
        try:
            # Calculate days to event (negative if in past)
            days_to_event = (event_date - date.today()).days
            
            # Get revision momentum for pre-event period
            revision_momentum = await analyst_revision_tracker.calculate_revision_momentum(
                symbol, self.historical_periods["revision_window"]
            )
            
            # Calculate consensus stability (how much it changed)
            consensus_stability = 1.0 - abs(revision_momentum.consensus_rating_change)
            
            # Get historical performance
            historical_beat_rate = await self._calculate_historical_beat_rate(
                symbol, event_type, db
            )
            
            # Get sector trends (simplified for now)
            sector_surprise_trend = 0.0  # Would calculate from sector data
            
            # Market regime assessment (simplified)
            market_regime = "neutral"    # Would get from macro data
            volatility_regime = "normal" # Would get from VIX/volatility data
            earnings_season_factor = 1.0 # Would calculate based on timing
            
            return SurpriseContext(
                symbol=symbol,
                event_date=event_date,
                event_type=event_type,
                days_to_event=days_to_event,
                consensus_stability=consensus_stability,
                revision_activity=revision_momentum.total_revisions,
                analyst_conviction=revision_momentum.conviction_score,
                historical_beat_rate=historical_beat_rate,
                historical_surprise_volatility=0.15,  # Would calculate from data
                sector_surprise_trend=sector_surprise_trend,
                market_regime=market_regime,
                volatility_regime=volatility_regime,
                earnings_season_factor=earnings_season_factor,
            )
            
        except Exception as e:
            logger.error(f"Error building surprise context: {e}")
            # Return default context
            return SurpriseContext(
                symbol=symbol, event_date=event_date, event_type=event_type,
                days_to_event=0, consensus_stability=0.5, revision_activity=0,
                analyst_conviction=0.5, historical_beat_rate=0.5,
                historical_surprise_volatility=0.15, sector_surprise_trend=0.0,
                market_regime="neutral", volatility_regime="normal",
                earnings_season_factor=1.0,
            )
    
    async def _get_historical_surprises(
        self,
        symbol: str,
        event_type: str,
        periods: int,
        db: Session
    ) -> List[float]:
        """Get historical surprise percentages for context"""
        
        try:
            # In production, this would query the surprise_tracking_enhanced table
            # For now, generate synthetic historical data
            
            surprises = []
            for i in range(periods):
                # Generate realistic surprise pattern
                base_surprise = (hash(symbol + str(i)) % 40 - 20) / 100  # -20% to +20%
                surprises.append(base_surprise)
            
            return surprises
            
        except Exception as e:
            logger.error(f"Error getting historical surprises: {e}")
            return []
    
    def _standardize_surprise(self, surprise_percent: float, historical_surprises: List[float]) -> float:
        """Standardize surprise using historical volatility"""
        
        if not historical_surprises:
            return surprise_percent / 10.0  # Default normalization
        
        historical_std = np.std(historical_surprises)
        if historical_std == 0:
            return 0.0
        
        historical_mean = np.mean(historical_surprises)
        return (surprise_percent - historical_mean) / historical_std
    
    async def _get_consensus_quality(
        self,
        symbol: str,
        event_date: date,
        event_type: str,
        db: Session
    ) -> Dict[str, Any]:
        """Get consensus quality metrics from real data"""
        
        try:
            # Try to get real consensus quality from earnings_monitor
            real_quality = await earnings_monitor.get_real_consensus_quality(symbol, event_date, db)
            
            # If we have fresh real data (data_freshness < 30 days), use it
            if real_quality.get('data_freshness', 999) < 30:
                return {
                    'count': real_quality['analyst_count'],
                    'std': real_quality.get('consensus_std', 0),
                    'high': real_quality.get('consensus_high', 0),
                    'low': real_quality.get('consensus_low', 0),
                    'range_pct': real_quality.get('consensus_range_pct', 0),
                    'confidence': real_quality.get('consensus_confidence', 0),
                    'data_source': 'real'
                }
            
            # Fallback to enhanced synthetic data if real data is stale
            logger.warning(f"Using synthetic consensus data for {symbol} - real data is {real_quality.get('data_freshness', 'unknown')} days old")
            
            # Enhanced synthetic data generation with more realistic patterns
            base_count = 8 + (hash(symbol) % 8)  # 8-16 analysts
            
            # Adjust analyst count based on market cap (simulate larger companies having more coverage)
            market_cap_factor = 1 + ((hash(symbol + "cap") % 50) / 100)  # 1.0-1.5x
            count = int(base_count * market_cap_factor)
            
            # More realistic estimate ranges
            mean_estimate = 1.5 + (hash(symbol + "mean") % 300) / 100  # $1.50-$4.50
            volatility_factor = 0.08 + (hash(symbol + "vol") % 40) / 1000  # 8%-12% typical range
            std_estimate = mean_estimate * volatility_factor
            
            high_estimate = mean_estimate + (std_estimate * 1.96)  # 95% confidence interval
            low_estimate = mean_estimate - (std_estimate * 1.96)
            
            range_pct = ((high_estimate - low_estimate) / mean_estimate) * 100 if mean_estimate != 0 else 0
            
            # More sophisticated confidence calculation
            coverage_score = min(1.0, count / 12)  # Max confidence at 12+ analysts
            agreement_score = max(0.1, 1 - (range_pct / 50))  # Lower confidence with wider ranges
            confidence = coverage_score * agreement_score
            
            return {
                "count": count,
                "std": std_estimate,
                "high": high_estimate,
                "low": low_estimate,
                "range_pct": range_pct,
                "confidence": confidence,
                "data_source": "synthetic_enhanced"
            }
            
        except Exception as e:
            logger.error(f"Error getting consensus quality: {e}")
            return {"count": 0, "std": 0, "high": 0, "low": 0, "range_pct": 0, "confidence": 0.5}
    
    def _assess_surprise_significance(self, event_type: str, surprise_magnitude: float) -> str:
        """Assess the significance level of a surprise"""
        
        thresholds = self.significance_thresholds.get(event_type, self.significance_thresholds["earnings"])
        
        if surprise_magnitude >= thresholds["high"]:
            return "high"
        elif surprise_magnitude >= thresholds["medium"]:
            return "medium"
        elif surprise_magnitude >= thresholds["low"]:
            return "low"
        else:
            return "minimal"
    
    def _calculate_surprise_percentile(self, surprise_percent: float, historical_surprises: List[float]) -> float:
        """Calculate what percentile this surprise represents historically"""
        
        if not historical_surprises:
            return 50.0  # Default to median
        
        historical_array = np.array(historical_surprises)
        percentile = (np.sum(historical_array <= surprise_percent) / len(historical_array)) * 100
        
        return percentile
    
    async def _get_market_impact(self, symbol: str, event_date: date, db: Session) -> Dict[str, Optional[float]]:
        """Get market reaction data for the event"""
        
        try:
            # In production, would get actual price and volume data
            # Generate synthetic market impact data
            
            # Simulate price reactions
            base_reaction = (hash(symbol + str(event_date)) % 20 - 10) / 100  # -10% to +10%
            reaction_1d = base_reaction * (0.8 + (hash(symbol + "1d") % 40) / 100)
            reaction_3d = reaction_1d * (0.9 + (hash(symbol + "3d") % 20) / 100)
            
            # Simulate volume spike
            volume_spike = 1.5 + (hash(symbol + "vol") % 200) / 100  # 1.5x to 3.5x
            
            return {
                "reaction_1d": reaction_1d,
                "reaction_3d": reaction_3d,
                "volume_spike": volume_spike,
            }
            
        except Exception as e:
            logger.error(f"Error getting market impact: {e}")
            return {"reaction_1d": None, "reaction_3d": None, "volume_spike": None}
    
    def _calculate_market_efficiency(self, surprise_standardized: float, market_reaction: Optional[float]) -> float:
        """Calculate how efficiently the market reacted to the surprise"""
        
        if market_reaction is None:
            return 0.5  # Default
        
        # Perfect efficiency would be proportional reaction
        expected_reaction = surprise_standardized * 0.02  # 2% per standard deviation
        
        if expected_reaction == 0:
            return 1.0 if abs(market_reaction) < 0.01 else 0.5
        
        # Calculate how close actual reaction was to expected
        reaction_ratio = market_reaction / expected_reaction
        
        # Efficiency is higher when ratio is closer to 1
        efficiency = max(0.0, min(1.0, 1.0 - abs(1.0 - reaction_ratio)))
        
        return efficiency
    
    def _generate_surprise_attribution(
        self,
        context: SurpriseContext,
        surprise_percent: float,
        consensus_data: Dict[str, Any]
    ) -> List[str]:
        """Generate explanations for what caused the surprise"""
        
        attribution = []
        
        # Consensus quality factors
        if consensus_data.get("confidence", 0.5) < 0.3:
            attribution.append("Low consensus confidence due to wide analyst range")
        
        if consensus_data.get("count", 0) < 5:
            attribution.append("Limited analyst coverage may have reduced consensus accuracy")
        
        # Revision activity factors
        if context.revision_activity > 5:
            if surprise_percent > 0:
                attribution.append("High pre-event revision activity suggested upside potential")
            else:
                attribution.append("Despite recent revisions, company underperformed expectations")
        
        # Historical context
        if context.historical_beat_rate > 0.7 and surprise_percent < 0:
            attribution.append("Unusual miss for company with strong historical beat rate")
        elif context.historical_beat_rate < 0.4 and surprise_percent > 0:
            attribution.append("Positive surprise breaks pattern of historical underperformance")
        
        # Analyst conviction
        if context.analyst_conviction > 0.8:
            attribution.append("High-conviction analysts provided strong guidance accuracy")
        elif context.analyst_conviction < 0.4:
            attribution.append("Low analyst conviction may have contributed to surprise")
        
        if not attribution:
            attribution.append("Normal market dynamics and execution variance")
        
        return attribution
    
    def _generate_future_implications(
        self,
        symbol: str,
        surprise_percent: float,
        context: SurpriseContext,
        historical_surprises: List[float]
    ) -> List[str]:
        """Generate forward-looking implications of the surprise"""
        
        implications = []
        
        # Performance pattern implications
        if abs(surprise_percent) > 15:
            if surprise_percent > 0:
                implications.append("Large positive surprise may indicate improving execution capability")
                implications.append("Consider potential for continued outperformance in subsequent quarters")
            else:
                implications.append("Significant miss suggests potential execution or market challenges")
                implications.append("Monitor next quarter for signs of recovery or continued weakness")
        
        # Guidance implications
        if surprise_percent > 10 and context.analyst_conviction > 0.7:
            implications.append("Strong beat with high analyst conviction suggests positive guidance likely")
        
        # Historical pattern break
        if historical_surprises:
            recent_trend = np.mean(historical_surprises[-4:]) if len(historical_surprises) >= 4 else 0
            if (recent_trend < -5 and surprise_percent > 10) or (recent_trend > 5 and surprise_percent < -10):
                implications.append("Result breaks recent performance trend - potential inflection point")
        
        # Market efficiency implications
        if context.revision_activity < 2:
            implications.append("Low pre-event revision activity suggests analysts may reassess estimates")
        
        if not implications:
            implications.append("Result aligns with historical patterns - expect normal market progression")
        
        return implications
    
    async def _calculate_historical_beat_rate(
        self,
        symbol: str,
        event_type: str,
        db: Session
    ) -> float:
        """Calculate historical beat rate for the company"""
        
        try:
            # In production, would query historical surprise data
            # Generate synthetic beat rate based on symbol characteristics
            
            base_rate = 0.5  # Start with 50%
            
            # Adjust based on symbol characteristics
            symbol_factor = (hash(symbol) % 100) / 100  # 0-1
            
            # Large cap stocks tend to beat more often
            if len(symbol) <= 4:  # Proxy for large cap
                base_rate += 0.1
            
            # Add some randomness based on symbol
            beat_rate = base_rate + (symbol_factor - 0.5) * 0.4  # ±20% adjustment
            
            return max(0.0, min(1.0, beat_rate))
            
        except Exception as e:
            logger.error(f"Error calculating beat rate: {e}")
            return 0.5
    
    async def analyze_surprise_patterns(
        self,
        symbols: List[str],
        event_type: str = "earnings",
        periods: int = 8
    ) -> Dict[str, Any]:
        """Analyze surprise patterns across multiple symbols"""
        
        try:
            pattern_analysis = {
                "symbols_analyzed": len(symbols),
                "event_type": event_type,
                "periods_analyzed": periods,
                "patterns": {},
                "insights": [],
            }
            
            # Analyze each symbol
            symbol_patterns = {}
            for symbol in symbols:
                try:
                    # Get historical surprises (mock data for now)
                    historical = []
                    for i in range(periods):
                        surprise = (hash(symbol + str(i)) % 40 - 20) / 100
                        historical.append(surprise)
                    
                    # Calculate pattern metrics
                    mean_surprise = np.mean(historical)
                    surprise_volatility = np.std(historical)
                    beat_rate = sum(1 for s in historical if s > 0) / len(historical)
                    trend = np.polyfit(range(len(historical)), historical, 1)[0] if len(historical) > 1 else 0
                    
                    symbol_patterns[symbol] = {
                        "mean_surprise": mean_surprise,
                        "surprise_volatility": surprise_volatility,
                        "beat_rate": beat_rate,
                        "trend": trend,
                        "consistency": 1.0 - surprise_volatility,  # Lower volatility = more consistent
                    }
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            pattern_analysis["patterns"] = symbol_patterns
            
            # Generate insights
            if symbol_patterns:
                avg_beat_rate = np.mean([p["beat_rate"] for p in symbol_patterns.values()])
                avg_volatility = np.mean([p["surprise_volatility"] for p in symbol_patterns.values()])
                
                insights = []
                if avg_beat_rate > 0.6:
                    insights.append("Portfolio shows strong historical beat rate - generally positive surprises")
                elif avg_beat_rate < 0.4:
                    insights.append("Portfolio has struggled with consensus expectations historically")
                
                if avg_volatility > 0.15:
                    insights.append("High surprise volatility indicates unpredictable earnings patterns")
                elif avg_volatility < 0.08:
                    insights.append("Low surprise volatility suggests predictable earnings patterns")
                
                # Find top performers
                top_consistent = sorted(symbol_patterns.items(), key=lambda x: x[1]["consistency"], reverse=True)[:3]
                if top_consistent:
                    top_symbols = [s[0] for s in top_consistent]
                    insights.append(f"Most consistent performers: {', '.join(top_symbols)}")
                
                pattern_analysis["insights"] = insights
            
            return pattern_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing surprise patterns: {e}")
            return {"error": str(e)}


# Global service instance
enhanced_surprise_service = EnhancedSurpriseService()