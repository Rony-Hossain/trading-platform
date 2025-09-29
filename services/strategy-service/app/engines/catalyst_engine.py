"""Event Catalyst Engine

Advanced event-driven trading logic that combines event occurrence, surprise thresholds,
sentiment spike filters, and regime tags to generate high-quality trading signals.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


class EventTriggerType(Enum):
    """Types of event triggers."""
    EARNINGS_SURPRISE = "earnings_surprise"
    FDA_APPROVAL = "fda_approval"
    ANALYST_UPGRADE = "analyst_upgrade"
    ANALYST_DOWNGRADE = "analyst_downgrade"
    MERGER_ANNOUNCEMENT = "merger_announcement"
    PRODUCT_LAUNCH = "product_launch"
    GUIDANCE_REVISION = "guidance_revision"
    REGULATORY_DECISION = "regulatory_decision"


class SentimentSignal(Enum):
    """Sentiment signal strength."""
    STRONG_POSITIVE = "strong_positive"
    MODERATE_POSITIVE = "moderate_positive"
    NEUTRAL = "neutral"
    MODERATE_NEGATIVE = "moderate_negative"
    STRONG_NEGATIVE = "strong_negative"


class RegimeType(Enum):
    """Market regime classification."""
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


@dataclass
class EventData:
    """Event information for catalyst analysis."""
    event_id: str
    symbol: str
    event_type: EventTriggerType
    scheduled_at: datetime
    actual_value: Optional[float] = None
    expected_value: Optional[float] = None
    surprise_magnitude: Optional[float] = None
    impact_score: int = 5
    source: str = "unknown"


@dataclass
class SentimentData:
    """Sentiment analysis data."""
    symbol: str
    sentiment_score: float  # -1.0 to 1.0
    sentiment_signal: SentimentSignal
    volume_score: float  # Relative sentiment volume
    momentum_score: float  # Sentiment momentum/acceleration
    credibility_score: float  # Source credibility weight
    timestamp: datetime


@dataclass
class RegimeData:
    """Market regime information."""
    regime_type: RegimeType
    confidence: float  # 0.0 to 1.0
    volatility_percentile: float
    trend_strength: float
    regime_duration_days: int
    timestamp: datetime


@dataclass
class CatalystSignal:
    """Generated catalyst trading signal."""
    symbol: str
    signal_strength: float  # -1.0 to 1.0 (negative = short, positive = long)
    confidence: float  # 0.0 to 1.0
    direction: str  # "long", "short", "neutral"
    trigger_type: EventTriggerType
    surprise_score: float
    sentiment_score: float
    regime_score: float
    composite_score: float
    holding_period_hours: int
    stop_loss_pct: float
    take_profit_pct: float
    event_data: EventData
    sentiment_data: SentimentData
    regime_data: RegimeData
    generated_at: datetime
    expires_at: datetime


class CatalystEngine:
    """Engine for generating event-driven catalyst signals."""
    
    def __init__(self, 
                 event_service_url: str = "http://localhost:8010",
                 sentiment_service_url: str = "http://localhost:8007",
                 analysis_service_url: str = "http://localhost:8003"):
        self.event_service_url = event_service_url
        self.sentiment_service_url = sentiment_service_url
        self.analysis_service_url = analysis_service_url
        self.client = None
        
        # Configurable thresholds
        self.surprise_thresholds = {
            EventTriggerType.EARNINGS_SURPRISE: 0.15,  # 15% surprise threshold
            EventTriggerType.FDA_APPROVAL: 0.50,       # 50% probability threshold
            EventTriggerType.ANALYST_UPGRADE: 2.0,     # 2 standard deviations
            EventTriggerType.ANALYST_DOWNGRADE: 2.0,
            EventTriggerType.MERGER_ANNOUNCEMENT: 0.20,
            EventTriggerType.PRODUCT_LAUNCH: 0.30,
            EventTriggerType.GUIDANCE_REVISION: 0.25,
            EventTriggerType.REGULATORY_DECISION: 0.40
        }
        
        # Sentiment spike thresholds
        self.sentiment_thresholds = {
            "volume_spike": 2.5,      # 2.5x normal sentiment volume
            "momentum_threshold": 0.3,  # 30% sentiment acceleration
            "credibility_min": 0.6,    # Minimum credibility score
            "sentiment_magnitude": 0.4  # Minimum absolute sentiment
        }
        
        # Regime favorability scores
        self.regime_favorability = {
            RegimeType.LOW_VOLATILITY: {
                "long_bias": 0.2, "short_bias": -0.1, "neutral_bias": 0.1
            },
            RegimeType.HIGH_VOLATILITY: {
                "long_bias": -0.1, "short_bias": 0.3, "neutral_bias": -0.2
            },
            RegimeType.TRENDING_UP: {
                "long_bias": 0.4, "short_bias": -0.3, "neutral_bias": 0.0
            },
            RegimeType.TRENDING_DOWN: {
                "long_bias": -0.3, "short_bias": 0.4, "neutral_bias": 0.0
            },
            RegimeType.SIDEWAYS: {
                "long_bias": 0.0, "short_bias": 0.0, "neutral_bias": 0.2
            },
            RegimeType.CRISIS: {
                "long_bias": -0.5, "short_bias": 0.1, "neutral_bias": -0.3
            }
        }
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def analyze_event_catalyst(self, symbol: str, lookback_hours: int = 24) -> Optional[CatalystSignal]:
        """Analyze potential catalyst signals for a symbol."""
        try:
            # Fetch recent events
            events = await self._fetch_recent_events(symbol, lookback_hours)
            if not events:
                logger.info(f"No recent events found for {symbol}")
                return None
            
            # Get latest sentiment data
            sentiment = await self._fetch_sentiment_data(symbol)
            if not sentiment:
                logger.warning(f"No sentiment data available for {symbol}")
                return None
            
            # Get current regime data
            regime = await self._fetch_regime_data(symbol)
            if not regime:
                logger.warning(f"No regime data available for {symbol}")
                return None
            
            # Find the most significant event
            primary_event = self._select_primary_event(events)
            if not primary_event:
                return None
            
            # Generate catalyst signal
            signal = await self._generate_catalyst_signal(
                primary_event, sentiment, regime
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to analyze catalyst for {symbol}: {e}")
            return None
    
    async def _fetch_recent_events(self, symbol: str, lookback_hours: int) -> List[EventData]:
        """Fetch recent events for the symbol."""
        try:
            since = datetime.utcnow() - timedelta(hours=lookback_hours)
            params = {
                "symbol": symbol,
                "start_date": since.isoformat(),
                "limit": 50
            }
            
            response = await self.client.get(
                f"{self.event_service_url}/events",
                params=params
            )
            response.raise_for_status()
            
            events_data = response.json()
            events = []
            
            for event_json in events_data:
                try:
                    event_type = EventTriggerType(event_json.get("category", "unknown"))
                except ValueError:
                    continue  # Skip unknown event types
                
                # Calculate surprise if available
                surprise_magnitude = None
                actual = event_json.get("metadata", {}).get("actual_value")
                expected = event_json.get("metadata", {}).get("expected_value")
                
                if actual is not None and expected is not None and expected != 0:
                    surprise_magnitude = abs((actual - expected) / expected)
                
                event = EventData(
                    event_id=event_json["id"],
                    symbol=event_json["symbol"],
                    event_type=event_type,
                    scheduled_at=datetime.fromisoformat(event_json["scheduled_at"].replace("Z", "+00:00")),
                    actual_value=actual,
                    expected_value=expected,
                    surprise_magnitude=surprise_magnitude,
                    impact_score=event_json.get("impact_score", 5),
                    source=event_json.get("source", "unknown")
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to fetch events for {symbol}: {e}")
            return []
    
    async def _fetch_sentiment_data(self, symbol: str) -> Optional[SentimentData]:
        """Fetch latest sentiment data for the symbol."""
        try:
            response = await self.client.get(
                f"{self.sentiment_service_url}/sentiment/{symbol}/latest"
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Map sentiment score to signal enum
            sentiment_score = data.get("sentiment_score", 0.0)
            if sentiment_score >= 0.4:
                signal = SentimentSignal.STRONG_POSITIVE
            elif sentiment_score >= 0.2:
                signal = SentimentSignal.MODERATE_POSITIVE
            elif sentiment_score <= -0.4:
                signal = SentimentSignal.STRONG_NEGATIVE
            elif sentiment_score <= -0.2:
                signal = SentimentSignal.MODERATE_NEGATIVE
            else:
                signal = SentimentSignal.NEUTRAL
            
            return SentimentData(
                symbol=symbol,
                sentiment_score=sentiment_score,
                sentiment_signal=signal,
                volume_score=data.get("volume_score", 1.0),
                momentum_score=data.get("momentum_score", 0.0),
                credibility_score=data.get("credibility_score", 0.5),
                timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat()))
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch sentiment for {symbol}: {e}")
            return None
    
    async def _fetch_regime_data(self, symbol: str) -> Optional[RegimeData]:
        """Fetch current market regime data."""
        try:
            response = await self.client.get(
                f"{self.analysis_service_url}/regime/{symbol}"
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Map regime classification
            regime_str = data.get("regime_type", "sideways").lower()
            regime_type = RegimeType.SIDEWAYS  # Default
            
            if "low_vol" in regime_str or "calm" in regime_str:
                regime_type = RegimeType.LOW_VOLATILITY
            elif "high_vol" in regime_str or "volatile" in regime_str:
                regime_type = RegimeType.HIGH_VOLATILITY
            elif "trend_up" in regime_str or "bullish" in regime_str:
                regime_type = RegimeType.TRENDING_UP
            elif "trend_down" in regime_str or "bearish" in regime_str:
                regime_type = RegimeType.TRENDING_DOWN
            elif "crisis" in regime_str or "stress" in regime_str:
                regime_type = RegimeType.CRISIS
            
            return RegimeData(
                regime_type=regime_type,
                confidence=data.get("confidence", 0.5),
                volatility_percentile=data.get("volatility_percentile", 50.0),
                trend_strength=data.get("trend_strength", 0.0),
                regime_duration_days=data.get("regime_duration_days", 1),
                timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat()))
            )
            
        except Exception as e:
            logger.warning(f"Failed to fetch regime data for {symbol}: {e}")
            # Return default regime data
            return RegimeData(
                regime_type=RegimeType.SIDEWAYS,
                confidence=0.3,
                volatility_percentile=50.0,
                trend_strength=0.0,
                regime_duration_days=1,
                timestamp=datetime.utcnow()
            )
    
    def _select_primary_event(self, events: List[EventData]) -> Optional[EventData]:
        """Select the most significant event for analysis."""
        if not events:
            return None
        
        # Score events by recency, impact, and surprise
        scored_events = []
        now = datetime.utcnow()
        
        for event in events:
            # Recency score (more recent = higher score)
            hours_ago = (now - event.scheduled_at).total_seconds() / 3600
            recency_score = max(0, 1 - (hours_ago / 24))  # Decay over 24 hours
            
            # Impact score (normalized)
            impact_score = event.impact_score / 10.0
            
            # Surprise score
            surprise_score = 0.0
            if event.surprise_magnitude is not None:
                threshold = self.surprise_thresholds.get(event.event_type, 0.2)
                surprise_score = min(1.0, event.surprise_magnitude / threshold)
            
            # Composite score
            composite = (recency_score * 0.4 + impact_score * 0.3 + surprise_score * 0.3)
            scored_events.append((composite, event))
        
        # Return highest scoring event
        scored_events.sort(key=lambda x: x[0], reverse=True)
        return scored_events[0][1] if scored_events else None
    
    async def _generate_catalyst_signal(self, 
                                      event: EventData, 
                                      sentiment: SentimentData, 
                                      regime: RegimeData) -> Optional[CatalystSignal]:
        """Generate trading signal from catalyst analysis."""
        
        # 1. Calculate surprise score
        surprise_score = self._calculate_surprise_score(event)
        
        # 2. Calculate sentiment score  
        sentiment_score = self._calculate_sentiment_score(sentiment)
        
        # 3. Calculate regime score
        regime_score = self._calculate_regime_score(regime, sentiment.sentiment_score)
        
        # 4. Check minimum thresholds
        if not self._meets_trigger_thresholds(surprise_score, sentiment_score, regime_score):
            logger.info(f"Signal for {event.symbol} does not meet minimum thresholds")
            return None
        
        # 5. Calculate composite signal
        composite_score = self._calculate_composite_score(
            surprise_score, sentiment_score, regime_score, event.event_type
        )
        
        # 6. Determine direction and strength
        direction, signal_strength = self._determine_signal_direction(
            composite_score, sentiment.sentiment_score, event
        )
        
        if direction == "neutral":
            return None
        
        # 7. Calculate confidence
        confidence = self._calculate_confidence(
            surprise_score, sentiment_score, regime_score, sentiment.credibility_score
        )
        
        # 8. Determine position parameters
        holding_period, stop_loss, take_profit = self._calculate_position_parameters(
            event.event_type, regime.regime_type, abs(signal_strength)
        )
        
        # 9. Generate signal
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=holding_period)
        
        return CatalystSignal(
            symbol=event.symbol,
            signal_strength=signal_strength,
            confidence=confidence,
            direction=direction,
            trigger_type=event.event_type,
            surprise_score=surprise_score,
            sentiment_score=sentiment_score,
            regime_score=regime_score,
            composite_score=composite_score,
            holding_period_hours=holding_period,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            event_data=event,
            sentiment_data=sentiment,
            regime_data=regime,
            generated_at=now,
            expires_at=expires_at
        )
    
    def _calculate_surprise_score(self, event: EventData) -> float:
        """Calculate normalized surprise score."""
        if event.surprise_magnitude is None:
            return 0.3  # Default moderate score for events without surprise data
        
        threshold = self.surprise_thresholds.get(event.event_type, 0.2)
        return min(1.0, event.surprise_magnitude / threshold)
    
    def _calculate_sentiment_score(self, sentiment: SentimentData) -> float:
        """Calculate sentiment contribution score."""
        # Base sentiment score
        base_score = abs(sentiment.sentiment_score)
        
        # Volume spike multiplier
        volume_multiplier = min(2.0, sentiment.volume_score / self.sentiment_thresholds["volume_spike"])
        
        # Momentum bonus
        momentum_bonus = min(0.3, abs(sentiment.momentum_score) / self.sentiment_thresholds["momentum_threshold"])
        
        # Credibility factor
        credibility_factor = max(0.5, sentiment.credibility_score)
        
        return min(1.0, (base_score + momentum_bonus) * volume_multiplier * credibility_factor)
    
    def _calculate_regime_score(self, regime: RegimeData, sentiment_direction: float) -> float:
        """Calculate regime favorability score."""
        regime_scores = self.regime_favorability.get(regime.regime_type, {
            "long_bias": 0.0, "short_bias": 0.0, "neutral_bias": 0.0
        })
        
        if sentiment_direction > 0.1:
            base_score = regime_scores["long_bias"]
        elif sentiment_direction < -0.1:
            base_score = regime_scores["short_bias"]
        else:
            base_score = regime_scores["neutral_bias"]
        
        # Adjust for regime confidence
        adjusted_score = base_score * regime.confidence
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, (adjusted_score + 1.0) / 2.0))
    
    def _meets_trigger_thresholds(self, surprise_score: float, sentiment_score: float, regime_score: float) -> bool:
        """Check if signal meets minimum trigger thresholds."""
        return (
            surprise_score >= 0.3 or  # Moderate surprise required
            sentiment_score >= 0.4 or  # Strong sentiment spike required  
            regime_score >= 0.7        # Very favorable regime required
        )
    
    def _calculate_composite_score(self, surprise_score: float, sentiment_score: float, 
                                 regime_score: float, event_type: EventTriggerType) -> float:
        """Calculate weighted composite signal score."""
        # Event-type specific weights
        if event_type in [EventTriggerType.EARNINGS_SURPRISE, EventTriggerType.GUIDANCE_REVISION]:
            weights = {"surprise": 0.5, "sentiment": 0.3, "regime": 0.2}
        elif event_type in [EventTriggerType.FDA_APPROVAL, EventTriggerType.REGULATORY_DECISION]:
            weights = {"surprise": 0.4, "sentiment": 0.4, "regime": 0.2}
        else:
            weights = {"surprise": 0.35, "sentiment": 0.35, "regime": 0.3}
        
        return (
            surprise_score * weights["surprise"] +
            sentiment_score * weights["sentiment"] +
            regime_score * weights["regime"]
        )
    
    def _determine_signal_direction(self, composite_score: float, sentiment_direction: float, 
                                  event: EventData) -> Tuple[str, float]:
        """Determine signal direction and strength."""
        if composite_score < 0.4:
            return "neutral", 0.0
        
        # Determine direction based on sentiment and event type
        if sentiment_direction > 0.1:
            direction = "long"
            strength = composite_score
        elif sentiment_direction < -0.1:
            direction = "short"
            strength = -composite_score
        else:
            # Use event impact for direction
            if event.impact_score >= 7:
                direction = "long"
                strength = composite_score * 0.7  # Reduced strength for uncertain direction
            else:
                return "neutral", 0.0
        
        return direction, strength
    
    def _calculate_confidence(self, surprise_score: float, sentiment_score: float, 
                            regime_score: float, credibility_score: float) -> float:
        """Calculate overall signal confidence."""
        # Average of component scores
        base_confidence = (surprise_score + sentiment_score + regime_score) / 3.0
        
        # Adjust for credibility
        adjusted_confidence = base_confidence * (0.5 + 0.5 * credibility_score)
        
        return min(1.0, adjusted_confidence)
    
    def _calculate_position_parameters(self, event_type: EventTriggerType, 
                                     regime_type: RegimeType, signal_strength: float) -> Tuple[int, float, float]:
        """Calculate holding period, stop loss, and take profit parameters."""
        # Base parameters by event type
        base_params = {
            EventTriggerType.EARNINGS_SURPRISE: {"hours": 48, "stop": 0.08, "profit": 0.15},
            EventTriggerType.FDA_APPROVAL: {"hours": 72, "stop": 0.12, "profit": 0.25},
            EventTriggerType.ANALYST_UPGRADE: {"hours": 24, "stop": 0.05, "profit": 0.10},
            EventTriggerType.ANALYST_DOWNGRADE: {"hours": 24, "stop": 0.05, "profit": 0.10},
            EventTriggerType.MERGER_ANNOUNCEMENT: {"hours": 120, "stop": 0.06, "profit": 0.20},
            EventTriggerType.PRODUCT_LAUNCH: {"hours": 96, "stop": 0.10, "profit": 0.18},
            EventTriggerType.GUIDANCE_REVISION: {"hours": 36, "stop": 0.07, "profit": 0.12},
            EventTriggerType.REGULATORY_DECISION: {"hours": 48, "stop": 0.09, "profit": 0.16}
        }
        
        params = base_params.get(event_type, {"hours": 48, "stop": 0.08, "profit": 0.15})
        
        # Adjust for regime
        if regime_type == RegimeType.HIGH_VOLATILITY:
            params["stop"] *= 1.5
            params["profit"] *= 1.3
            params["hours"] = int(params["hours"] * 0.8)
        elif regime_type == RegimeType.LOW_VOLATILITY:
            params["stop"] *= 0.7
            params["profit"] *= 0.8
            params["hours"] = int(params["hours"] * 1.2)
        elif regime_type == RegimeType.CRISIS:
            params["stop"] *= 2.0
            params["profit"] *= 0.6
            params["hours"] = int(params["hours"] * 0.5)
        
        # Adjust for signal strength
        strength_multiplier = 0.5 + signal_strength
        params["profit"] *= strength_multiplier
        
        return params["hours"], params["stop"], params["profit"]


async def test_catalyst_engine():
    """Test the catalyst engine functionality."""
    async with CatalystEngine() as engine:
        # Test catalyst analysis
        signal = await engine.analyze_event_catalyst("AAPL", lookback_hours=24)
        
        if signal:
            print(f"Generated catalyst signal for {signal.symbol}")
            print(f"Direction: {signal.direction}")
            print(f"Strength: {signal.signal_strength:.3f}")
            print(f"Confidence: {signal.confidence:.3f}")
            print(f"Trigger: {signal.trigger_type.value}")
            print(f"Holding period: {signal.holding_period_hours} hours")
            print(f"Stop loss: {signal.stop_loss_pct:.2%}")
            print(f"Take profit: {signal.take_profit_pct:.2%}")
        else:
            print("No catalyst signal generated")


if __name__ == "__main__":
    asyncio.run(test_catalyst_engine())