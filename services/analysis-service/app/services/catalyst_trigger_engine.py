"""
Comprehensive Catalyst Trigger Engine

This module implements a unified catalyst detection system that combines:
- Event occurrence detection and validation
- Volatility-normalized surprise threshold analysis
- Sentiment spike filtering and momentum detection
- Multi-factor signal combination and scoring
- Temporal alignment and causal inference
- Risk-adjusted catalyst impact assessment

Key Features:
- Multi-source signal fusion (events, sentiment, technicals, fundamentals)
- Adaptive threshold calibration based on asset characteristics
- Sentiment momentum and spike detection algorithms
- Cross-validation between data sources for signal confirmation
- Causal impact attribution with statistical significance testing
- Real-time catalyst scoring and confidence assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import asyncio
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import related services
from .volatility_threshold_calibration import (
    VolatilityThresholdCalibrator, EventType, SectorType,
    create_volatility_threshold_calibrator
)

logger = logging.getLogger(__name__)

class CatalystType(Enum):
    """Types of catalyst triggers"""
    FUNDAMENTAL_SURPRISE = "fundamental_surprise"
    SENTIMENT_SPIKE = "sentiment_spike"
    TECHNICAL_BREAKOUT = "technical_breakout"
    NEWS_EVENT = "news_event"
    INSIDER_ACTIVITY = "insider_activity"
    ANALYST_ACTION = "analyst_action"
    REGULATORY_CHANGE = "regulatory_change"
    EARNINGS_SURPRISE = "earnings_surprise"
    GUIDANCE_REVISION = "guidance_revision"
    PRODUCT_CATALYST = "product_catalyst"

class CatalystStrength(Enum):
    """Catalyst signal strength classification"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"
    EXTREME = "extreme"

class SentimentDirection(Enum):
    """Sentiment change direction"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"

@dataclass
class EventSignal:
    """Event occurrence signal with metadata"""
    symbol: str
    event_type: EventType
    event_time: datetime
    surprise_value: float
    surprise_magnitude: float
    exceeds_threshold: bool
    confidence: float
    event_description: str
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SentimentSignal:
    """Sentiment spike signal with characteristics"""
    symbol: str
    sentiment_time: datetime
    sentiment_score: float
    sentiment_change: float
    volume_spike: float
    direction: SentimentDirection
    momentum: float
    persistence: float  # How long sentiment has been elevated
    source_diversity: float  # Number of different sources
    credibility_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TechnicalSignal:
    """Technical analysis signal"""
    symbol: str
    signal_time: datetime
    signal_type: str
    strength: float
    price_move: float
    volume_confirmation: bool
    resistance_break: bool
    momentum_confirmation: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CatalystTrigger:
    """Unified catalyst trigger combining multiple signals"""
    trigger_id: str
    symbol: str
    trigger_time: datetime
    catalyst_type: CatalystType
    strength: CatalystStrength
    confidence: float
    
    # Component signals
    event_signal: Optional[EventSignal] = None
    sentiment_signal: Optional[SentimentSignal] = None
    technical_signal: Optional[TechnicalSignal] = None
    
    # Derived metrics
    combined_score: float = 0.0
    risk_adjusted_score: float = 0.0
    expected_impact: float = 0.0
    time_horizon: str = "1d"
    
    # Validation metrics
    signal_alignment: float = 0.0  # How well signals align temporally
    cross_validation: float = 0.0  # Cross-validation between sources
    statistical_significance: float = 0.0
    
    # Meta information
    sector: Optional[SectorType] = None
    market_cap: Optional[str] = None
    trading_session: str = "regular"
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class SentimentSpikeDetector:
    """Advanced sentiment spike detection and filtering"""
    
    def __init__(self):
        self.min_spike_threshold = 2.0  # Minimum 2-sigma spike
        self.min_volume_increase = 1.5  # 50% volume increase minimum
        self.persistence_window = timedelta(hours=4)  # Window for persistence
        self.credibility_weights = {
            "institutional": 0.4,
            "analyst": 0.3,
            "retail": 0.2,
            "social": 0.1
        }
    
    def detect_sentiment_spike(
        self,
        symbol: str,
        sentiment_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> Optional[SentimentSignal]:
        """Detect sentiment spikes with validation"""
        
        if len(sentiment_data) < 10:
            return None
        
        # Ensure timestamp column exists
        if 'timestamp' not in sentiment_data.columns:
            logger.warning(f"No timestamp column in sentiment data for {symbol}")
            return None
        
        # Sort by timestamp
        sentiment_data = sentiment_data.sort_values('timestamp')
        
        # Calculate rolling statistics
        sentiment_data['rolling_mean'] = sentiment_data['sentiment_score'].rolling(10).mean()
        sentiment_data['rolling_std'] = sentiment_data['sentiment_score'].rolling(10).std()
        
        # Detect spikes (z-score approach)
        sentiment_data['z_score'] = (
            (sentiment_data['sentiment_score'] - sentiment_data['rolling_mean']) / 
            sentiment_data['rolling_std'].replace(0, 1)
        )
        
        # Find recent spikes
        recent_window = datetime.now() - timedelta(hours=24)
        recent_data = sentiment_data[sentiment_data['timestamp'] >= recent_window]
        
        if recent_data.empty:
            return None
        
        # Get the most recent significant spike
        spike_candidates = recent_data[abs(recent_data['z_score']) >= self.min_spike_threshold]
        
        if spike_candidates.empty:
            return None
        
        # Select strongest recent spike
        latest_spike = spike_candidates.loc[spike_candidates['z_score'].abs().idxmax()]
        
        # Calculate sentiment direction
        if latest_spike['sentiment_score'] > 0.1:
            direction = SentimentDirection.BULLISH
        elif latest_spike['sentiment_score'] < -0.1:
            direction = SentimentDirection.BEARISH
        else:
            direction = SentimentDirection.NEUTRAL
        
        # Calculate sentiment change
        if len(recent_data) >= 2:
            baseline = recent_data['sentiment_score'].iloc[0]
            current = latest_spike['sentiment_score']
            sentiment_change = current - baseline
        else:
            sentiment_change = latest_spike['z_score'] * latest_spike['rolling_std']
        
        # Calculate volume spike if data available
        volume_spike = 1.0
        if volume_data is not None and not volume_data.empty:
            volume_spike = self._calculate_volume_spike(symbol, volume_data)
        
        # Calculate momentum and persistence
        momentum = self._calculate_sentiment_momentum(recent_data)
        persistence = self._calculate_sentiment_persistence(recent_data, latest_spike['timestamp'])
        
        # Calculate source diversity and credibility
        source_diversity = self._calculate_source_diversity(sentiment_data)
        credibility_score = self._calculate_credibility_score(sentiment_data)
        
        return SentimentSignal(
            symbol=symbol,
            sentiment_time=latest_spike['timestamp'],
            sentiment_score=latest_spike['sentiment_score'],
            sentiment_change=sentiment_change,
            volume_spike=volume_spike,
            direction=direction,
            momentum=momentum,
            persistence=persistence,
            source_diversity=source_diversity,
            credibility_score=credibility_score,
            metadata={
                "z_score": latest_spike['z_score'],
                "rolling_mean": latest_spike['rolling_mean'],
                "rolling_std": latest_spike['rolling_std'],
                "detection_method": "z_score_spike"
            }
        )
    
    def _calculate_volume_spike(self, symbol: str, volume_data: pd.DataFrame) -> float:
        """Calculate volume spike magnitude"""
        
        if len(volume_data) < 5:
            return 1.0
        
        # Calculate average volume over last 5 periods
        recent_vol = volume_data['volume'].tail(5).mean()
        baseline_vol = volume_data['volume'].iloc[:-5].tail(20).mean()
        
        if baseline_vol == 0:
            return 1.0
        
        return recent_vol / baseline_vol
    
    def _calculate_sentiment_momentum(self, sentiment_data: pd.DataFrame) -> float:
        """Calculate sentiment momentum over recent period"""
        
        if len(sentiment_data) < 3:
            return 0.0
        
        # Simple momentum calculation using linear regression slope
        x = np.arange(len(sentiment_data))
        y = sentiment_data['sentiment_score'].values
        
        if len(x) >= 2:
            slope, _, r_value, _, _ = stats.linregress(x, y)
            return slope * (r_value ** 2)  # Weight by R-squared
        
        return 0.0
    
    def _calculate_sentiment_persistence(self, sentiment_data: pd.DataFrame, spike_time: datetime) -> float:
        """Calculate how long sentiment has been elevated"""
        
        # Look for sustained elevated sentiment before spike
        persistent_data = sentiment_data[sentiment_data['timestamp'] <= spike_time]
        
        if len(persistent_data) < 2:
            return 0.0
        
        # Count consecutive periods of elevated sentiment
        threshold = persistent_data['sentiment_score'].median()
        elevated_mask = persistent_data['sentiment_score'] > threshold
        
        # Find longest consecutive streak
        streaks = []
        current_streak = 0
        
        for is_elevated in elevated_mask:
            if is_elevated:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        if not streaks:
            return 0.0
        
        max_streak = max(streaks)
        return min(max_streak / 10.0, 1.0)  # Normalize to 0-1
    
    def _calculate_source_diversity(self, sentiment_data: pd.DataFrame) -> float:
        """Calculate diversity of sentiment sources"""
        
        if 'source' not in sentiment_data.columns:
            return 0.5  # Default moderate diversity
        
        unique_sources = sentiment_data['source'].nunique()
        total_sources = len(sentiment_data)
        
        if total_sources == 0:
            return 0.0
        
        # Normalize diversity score
        diversity = unique_sources / min(total_sources, 10)
        return min(diversity, 1.0)
    
    def _calculate_credibility_score(self, sentiment_data: pd.DataFrame) -> float:
        """Calculate weighted credibility score based on sources"""
        
        if 'source' not in sentiment_data.columns:
            return 0.5  # Default moderate credibility
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for source in sentiment_data['source'].unique():
            source_weight = self.credibility_weights.get(source, 0.1)
            source_count = (sentiment_data['source'] == source).sum()
            
            total_weight += source_weight * source_count
            weighted_score += source_weight * source_count
        
        if total_weight == 0:
            return 0.5
        
        return min(weighted_score / total_weight, 1.0)

class CatalystTriggerEngine:
    """Main catalyst trigger detection and scoring engine"""
    
    def __init__(self):
        self.volatility_calibrator: Optional[VolatilityThresholdCalibrator] = None
        self.sentiment_detector = SentimentSpikeDetector()
        
        # Catalyst scoring weights
        self.signal_weights = {
            "event": 0.4,
            "sentiment": 0.35,
            "technical": 0.25
        }
        
        # Time alignment tolerance
        self.time_alignment_window = timedelta(hours=6)
        
        # Minimum confidence thresholds
        self.min_confidence_thresholds = {
            CatalystStrength.WEAK: 0.3,
            CatalystStrength.MODERATE: 0.5,
            CatalystStrength.STRONG: 0.7,
            CatalystStrength.VERY_STRONG: 0.85,
            CatalystStrength.EXTREME: 0.95
        }
    
    async def initialize(self):
        """Initialize the catalyst trigger engine"""
        self.volatility_calibrator = await create_volatility_threshold_calibrator()
    
    async def detect_catalyst_trigger(
        self,
        symbol: str,
        event_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[pd.DataFrame] = None,
        technical_data: Optional[Dict[str, Any]] = None,
        price_data: Optional[pd.DataFrame] = None,
        sector: Optional[SectorType] = None,
        current_time: Optional[datetime] = None
    ) -> Optional[CatalystTrigger]:
        """Detect and score catalyst triggers using multi-factor analysis"""
        
        if current_time is None:
            current_time = datetime.now()
        
        if not self.volatility_calibrator:
            await self.initialize()
        
        # Generate individual signals
        event_signal = await self._process_event_signal(
            symbol, event_data, price_data, sector, current_time
        )
        
        sentiment_signal = self._process_sentiment_signal(
            symbol, sentiment_data, current_time
        )
        
        technical_signal = self._process_technical_signal(
            symbol, technical_data, current_time
        )
        
        # Require at least one significant signal
        signals_present = sum([
            event_signal is not None,
            sentiment_signal is not None,
            technical_signal is not None
        ])
        
        if signals_present == 0:
            return None
        
        # Determine primary catalyst type
        catalyst_type = self._determine_catalyst_type(
            event_signal, sentiment_signal, technical_signal
        )
        
        # Calculate signal alignment and cross-validation
        signal_alignment = self._calculate_signal_alignment(
            event_signal, sentiment_signal, technical_signal, current_time
        )
        
        cross_validation = self._calculate_cross_validation(
            event_signal, sentiment_signal, technical_signal
        )
        
        # Calculate combined score
        combined_score = self._calculate_combined_score(
            event_signal, sentiment_signal, technical_signal
        )
        
        # Calculate risk-adjusted score
        risk_adjusted_score = self._calculate_risk_adjusted_score(
            combined_score, symbol, sector, current_time
        )
        
        # Determine catalyst strength
        strength = self._determine_catalyst_strength(combined_score, cross_validation)
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(
            event_signal, sentiment_signal, technical_signal,
            signal_alignment, cross_validation
        )
        
        # Calculate expected impact and time horizon
        expected_impact = self._estimate_expected_impact(
            symbol, catalyst_type, combined_score, sector
        )
        
        time_horizon = self._estimate_time_horizon(catalyst_type, strength)
        
        # Calculate statistical significance
        statistical_significance = self._calculate_statistical_significance(
            event_signal, sentiment_signal, technical_signal
        )
        
        # Create catalyst trigger
        trigger = CatalystTrigger(
            trigger_id=f"{symbol}_{catalyst_type.value}_{int(current_time.timestamp())}",
            symbol=symbol,
            trigger_time=current_time,
            catalyst_type=catalyst_type,
            strength=strength,
            confidence=confidence,
            event_signal=event_signal,
            sentiment_signal=sentiment_signal,
            technical_signal=technical_signal,
            combined_score=combined_score,
            risk_adjusted_score=risk_adjusted_score,
            expected_impact=expected_impact,
            time_horizon=time_horizon,
            signal_alignment=signal_alignment,
            cross_validation=cross_validation,
            statistical_significance=statistical_significance,
            sector=sector,
            trading_session=self._determine_trading_session(current_time),
            metadata={
                "signals_present": signals_present,
                "detection_timestamp": current_time,
                "engine_version": "1.0"
            }
        )
        
        return trigger
    
    async def _process_event_signal(
        self,
        symbol: str,
        event_data: Optional[Dict[str, Any]],
        price_data: Optional[pd.DataFrame],
        sector: Optional[SectorType],
        current_time: datetime
    ) -> Optional[EventSignal]:
        """Process event occurrence signal"""
        
        if not event_data or not self.volatility_calibrator:
            return None
        
        try:
            # Extract event information
            event_type_str = event_data.get("event_type", "earnings")
            surprise_value = event_data.get("surprise_value", 0.0)
            event_time_str = event_data.get("event_time")
            
            # Parse event type
            try:
                event_type = EventType(event_type_str.lower())
            except ValueError:
                event_type = EventType.EARNINGS
            
            # Parse event time
            if isinstance(event_time_str, str):
                event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
            elif isinstance(event_time_str, datetime):
                event_time = event_time_str
            else:
                event_time = current_time
            
            # Calculate adaptive threshold using volatility calibration
            threshold_result = await self.volatility_calibrator.get_adaptive_threshold(
                symbol=symbol,
                event_type=event_type,
                surprise_value=surprise_value,
                price_data=price_data,
                sector=sector
            )
            
            return EventSignal(
                symbol=symbol,
                event_type=event_type,
                event_time=event_time,
                surprise_value=surprise_value,
                surprise_magnitude=abs(surprise_value),
                exceeds_threshold=threshold_result["exceeds_threshold"],
                confidence=threshold_result["signal_confidence"],
                event_description=event_data.get("description", f"{event_type.value} event"),
                source=event_data.get("source", "unknown"),
                metadata={
                    "threshold_info": threshold_result,
                    "original_event_data": event_data
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing event signal for {symbol}: {e}")
            return None
    
    def _process_sentiment_signal(
        self,
        symbol: str,
        sentiment_data: Optional[pd.DataFrame],
        current_time: datetime
    ) -> Optional[SentimentSignal]:
        """Process sentiment spike signal"""
        
        if sentiment_data is None or sentiment_data.empty:
            return None
        
        try:
            return self.sentiment_detector.detect_sentiment_spike(symbol, sentiment_data)
        except Exception as e:
            logger.error(f"Error processing sentiment signal for {symbol}: {e}")
            return None
    
    def _process_technical_signal(
        self,
        symbol: str,
        technical_data: Optional[Dict[str, Any]],
        current_time: datetime
    ) -> Optional[TechnicalSignal]:
        """Process technical analysis signal"""
        
        if not technical_data:
            return None
        
        try:
            return TechnicalSignal(
                symbol=symbol,
                signal_time=current_time,
                signal_type=technical_data.get("signal_type", "breakout"),
                strength=technical_data.get("strength", 0.5),
                price_move=technical_data.get("price_move", 0.0),
                volume_confirmation=technical_data.get("volume_confirmation", False),
                resistance_break=technical_data.get("resistance_break", False),
                momentum_confirmation=technical_data.get("momentum_confirmation", False),
                metadata=technical_data
            )
        except Exception as e:
            logger.error(f"Error processing technical signal for {symbol}: {e}")
            return None
    
    def _determine_catalyst_type(
        self,
        event_signal: Optional[EventSignal],
        sentiment_signal: Optional[SentimentSignal],
        technical_signal: Optional[TechnicalSignal]
    ) -> CatalystType:
        """Determine primary catalyst type based on signals"""
        
        # Event-driven catalysts take priority
        if event_signal and event_signal.exceeds_threshold:
            if event_signal.event_type == EventType.EARNINGS:
                return CatalystType.EARNINGS_SURPRISE
            elif event_signal.event_type == EventType.GUIDANCE:
                return CatalystType.GUIDANCE_REVISION
            elif event_signal.event_type == EventType.FDA_APPROVAL:
                return CatalystType.PRODUCT_CATALYST
            elif event_signal.event_type == EventType.REGULATORY:
                return CatalystType.REGULATORY_CHANGE
            elif event_signal.event_type in [EventType.ANALYST_UPGRADE, EventType.ANALYST_DOWNGRADE]:
                return CatalystType.ANALYST_ACTION
            else:
                return CatalystType.FUNDAMENTAL_SURPRISE
        
        # Sentiment-driven catalysts
        if sentiment_signal and sentiment_signal.volume_spike > 1.5:
            return CatalystType.SENTIMENT_SPIKE
        
        # Technical catalysts
        if technical_signal and technical_signal.strength > 0.7:
            return CatalystType.TECHNICAL_BREAKOUT
        
        # Default to news event
        return CatalystType.NEWS_EVENT
    
    def _calculate_signal_alignment(
        self,
        event_signal: Optional[EventSignal],
        sentiment_signal: Optional[SentimentSignal],
        technical_signal: Optional[TechnicalSignal],
        current_time: datetime
    ) -> float:
        """Calculate temporal alignment between signals"""
        
        signal_times = []
        
        if event_signal:
            signal_times.append(event_signal.event_time)
        if sentiment_signal:
            signal_times.append(sentiment_signal.sentiment_time)
        if technical_signal:
            signal_times.append(technical_signal.signal_time)
        
        if len(signal_times) < 2:
            return 1.0  # Perfect alignment for single signal
        
        # Calculate time spreads
        max_time = max(signal_times)
        min_time = min(signal_times)
        time_spread = (max_time - min_time).total_seconds() / 3600  # Hours
        
        # Calculate alignment score (better alignment = smaller time spread)
        max_acceptable_spread = self.time_alignment_window.total_seconds() / 3600
        alignment_score = max(0.0, 1.0 - (time_spread / max_acceptable_spread))
        
        return alignment_score
    
    def _calculate_cross_validation(
        self,
        event_signal: Optional[EventSignal],
        sentiment_signal: Optional[SentimentSignal],
        technical_signal: Optional[TechnicalSignal]
    ) -> float:
        """Calculate cross-validation between different signal sources"""
        
        validations = []
        
        # Event-Sentiment validation
        if event_signal and sentiment_signal:
            # Check if sentiment direction aligns with event surprise
            event_positive = event_signal.surprise_value > 0
            sentiment_positive = sentiment_signal.direction in [SentimentDirection.BULLISH]
            
            alignment = 1.0 if event_positive == sentiment_positive else 0.3
            validations.append(alignment)
        
        # Event-Technical validation
        if event_signal and technical_signal:
            # Check if technical momentum aligns with event
            event_strength = event_signal.confidence
            tech_strength = technical_signal.strength
            
            alignment = min(event_strength, tech_strength)
            validations.append(alignment)
        
        # Sentiment-Technical validation
        if sentiment_signal and technical_signal:
            # Check if technical breakout aligns with sentiment
            sentiment_strength = sentiment_signal.momentum
            tech_strength = technical_signal.strength
            
            alignment = (sentiment_strength + tech_strength) / 2.0
            validations.append(alignment)
        
        if not validations:
            return 0.5  # No cross-validation possible
        
        return np.mean(validations)
    
    def _calculate_combined_score(
        self,
        event_signal: Optional[EventSignal],
        sentiment_signal: Optional[SentimentSignal],
        technical_signal: Optional[TechnicalSignal]
    ) -> float:
        """Calculate weighted combined score from all signals"""
        
        total_weight = 0.0
        weighted_score = 0.0
        
        # Event signal contribution
        if event_signal:
            event_score = event_signal.confidence if event_signal.exceeds_threshold else 0.0
            weighted_score += self.signal_weights["event"] * event_score
            total_weight += self.signal_weights["event"]
        
        # Sentiment signal contribution
        if sentiment_signal:
            sentiment_score = min(
                sentiment_signal.momentum + sentiment_signal.persistence,
                1.0
            )
            weighted_score += self.signal_weights["sentiment"] * sentiment_score
            total_weight += self.signal_weights["sentiment"]
        
        # Technical signal contribution
        if technical_signal:
            tech_score = technical_signal.strength
            weighted_score += self.signal_weights["technical"] * tech_score
            total_weight += self.signal_weights["technical"]
        
        if total_weight == 0:
            return 0.0
        
        return weighted_score / total_weight
    
    def _calculate_risk_adjusted_score(
        self,
        combined_score: float,
        symbol: str,
        sector: Optional[SectorType],
        current_time: datetime
    ) -> float:
        """Calculate risk-adjusted catalyst score"""
        
        # Base risk adjustment
        risk_adjustment = 1.0
        
        # Sector risk adjustment
        if sector:
            sector_risk_adjustments = {
                SectorType.BIOTECH: 0.7,      # Higher risk
                SectorType.TECHNOLOGY: 0.85,  # Moderate risk
                SectorType.UTILITIES: 1.1,    # Lower risk
                SectorType.ENERGY: 0.8,       # Higher volatility risk
                SectorType.FINANCIALS: 0.9    # Regulatory risk
            }
            risk_adjustment *= sector_risk_adjustments.get(sector, 1.0)
        
        # Time-of-day risk adjustment
        market_hour = current_time.hour
        if market_hour < 9 or market_hour > 16:  # After/before market hours
            risk_adjustment *= 0.9  # Slightly higher risk
        
        return combined_score * risk_adjustment
    
    def _determine_catalyst_strength(
        self,
        combined_score: float,
        cross_validation: float
    ) -> CatalystStrength:
        """Determine catalyst strength classification"""
        
        # Adjust score by cross-validation quality
        adjusted_score = combined_score * (0.7 + 0.3 * cross_validation)
        
        if adjusted_score >= 0.9:
            return CatalystStrength.EXTREME
        elif adjusted_score >= 0.75:
            return CatalystStrength.VERY_STRONG
        elif adjusted_score >= 0.6:
            return CatalystStrength.STRONG
        elif adjusted_score >= 0.4:
            return CatalystStrength.MODERATE
        else:
            return CatalystStrength.WEAK
    
    def _calculate_overall_confidence(
        self,
        event_signal: Optional[EventSignal],
        sentiment_signal: Optional[SentimentSignal],
        technical_signal: Optional[TechnicalSignal],
        signal_alignment: float,
        cross_validation: float
    ) -> float:
        """Calculate overall confidence in catalyst trigger"""
        
        # Base confidence from individual signals
        signal_confidences = []
        
        if event_signal and event_signal.exceeds_threshold:
            signal_confidences.append(event_signal.confidence)
        
        if sentiment_signal:
            sentiment_conf = min(
                sentiment_signal.credibility_score + sentiment_signal.source_diversity,
                1.0
            )
            signal_confidences.append(sentiment_conf)
        
        if technical_signal:
            tech_conf = technical_signal.strength
            signal_confidences.append(tech_conf)
        
        if not signal_confidences:
            return 0.0
        
        base_confidence = np.mean(signal_confidences)
        
        # Boost confidence with good alignment and cross-validation
        alignment_boost = signal_alignment * 0.2
        validation_boost = cross_validation * 0.2
        
        total_confidence = base_confidence + alignment_boost + validation_boost
        
        return min(total_confidence, 1.0)
    
    def _estimate_expected_impact(
        self,
        symbol: str,
        catalyst_type: CatalystType,
        combined_score: float,
        sector: Optional[SectorType]
    ) -> float:
        """Estimate expected price impact from catalyst"""
        
        # Base impact expectations by catalyst type
        base_impacts = {
            CatalystType.EARNINGS_SURPRISE: 0.05,
            CatalystType.FDA_APPROVAL: 0.15,
            CatalystType.GUIDANCE_REVISION: 0.08,
            CatalystType.SENTIMENT_SPIKE: 0.03,
            CatalystType.TECHNICAL_BREAKOUT: 0.04,
            CatalystType.ANALYST_ACTION: 0.02,
            CatalystType.REGULATORY_CHANGE: 0.10
        }
        
        base_impact = base_impacts.get(catalyst_type, 0.05)
        
        # Sector impact multipliers
        if sector:
            sector_multipliers = {
                SectorType.BIOTECH: 2.0,      # High impact events
                SectorType.TECHNOLOGY: 1.2,   # Moderate impact
                SectorType.UTILITIES: 0.7,    # Lower impact
                SectorType.ENERGY: 1.3,       # Commodity sensitivity
                SectorType.FINANCIALS: 1.1    # Regulatory sensitivity
            }
            base_impact *= sector_multipliers.get(sector, 1.0)
        
        # Scale by signal strength
        expected_impact = base_impact * combined_score
        
        return expected_impact
    
    def _estimate_time_horizon(
        self,
        catalyst_type: CatalystType,
        strength: CatalystStrength
    ) -> str:
        """Estimate impact time horizon"""
        
        # Base time horizons by catalyst type
        time_horizons = {
            CatalystType.SENTIMENT_SPIKE: "intraday",
            CatalystType.TECHNICAL_BREAKOUT: "1-3d",
            CatalystType.EARNINGS_SURPRISE: "1-5d",
            CatalystType.GUIDANCE_REVISION: "1-2w",
            CatalystType.FDA_APPROVAL: "1-4w",
            CatalystType.REGULATORY_CHANGE: "2-8w"
        }
        
        base_horizon = time_horizons.get(catalyst_type, "1-3d")
        
        # Adjust for strength
        if strength in [CatalystStrength.VERY_STRONG, CatalystStrength.EXTREME]:
            return base_horizon  # Strong catalysts maintain their timeline
        elif strength == CatalystStrength.WEAK:
            # Weak catalysts may take longer to play out
            horizon_map = {
                "intraday": "1d",
                "1-3d": "1w",
                "1-5d": "2w",
                "1-2w": "1m",
                "1-4w": "2m",
                "2-8w": "3m"
            }
            return horizon_map.get(base_horizon, base_horizon)
        
        return base_horizon
    
    def _calculate_statistical_significance(
        self,
        event_signal: Optional[EventSignal],
        sentiment_signal: Optional[SentimentSignal],
        technical_signal: Optional[TechnicalSignal]
    ) -> float:
        """Calculate statistical significance of catalyst trigger"""
        
        significance_scores = []
        
        # Event significance (based on surprise threshold)
        if event_signal and event_signal.exceeds_threshold:
            # Higher surprise magnitude = higher significance
            sig_score = min(event_signal.surprise_magnitude * 2, 1.0)
            significance_scores.append(sig_score)
        
        # Sentiment significance (based on z-score)
        if sentiment_signal and sentiment_signal.metadata.get("z_score"):
            z_score = abs(sentiment_signal.metadata["z_score"])
            # Convert z-score to significance (2+ sigma is significant)
            sig_score = min(z_score / 3.0, 1.0)
            significance_scores.append(sig_score)
        
        # Technical significance (based on strength and volume)
        if technical_signal:
            tech_sig = technical_signal.strength
            if technical_signal.volume_confirmation:
                tech_sig *= 1.2
            significance_scores.append(min(tech_sig, 1.0))
        
        if not significance_scores:
            return 0.5
        
        return np.mean(significance_scores)
    
    def _determine_trading_session(self, current_time: datetime) -> str:
        """Determine current trading session"""
        
        hour = current_time.hour
        
        if 4 <= hour < 9:
            return "pre_market"
        elif 9 <= hour < 16:
            return "regular"
        elif 16 <= hour < 20:
            return "after_hours"
        else:
            return "overnight"

# Factory function
async def create_catalyst_trigger_engine() -> CatalystTriggerEngine:
    """Create and initialize catalyst trigger engine"""
    engine = CatalystTriggerEngine()
    await engine.initialize()
    return engine