"""
Event-window specific sentiment momentum analysis for pre-event detection.
Implements EMA/acceleration metrics tied to scheduled events like earnings announcements.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from sqlalchemy.orm import Session
import yfinance as yf
from scipy import stats
from sklearn.metrics import mean_squared_error, accuracy_score

logger = logging.getLogger(__name__)

class EventType(str, Enum):
    EARNINGS = "earnings"
    FDA_APPROVAL = "fda_approval"
    MERGER_ANNOUNCEMENT = "merger_announcement"
    PRODUCT_LAUNCH = "product_launch"
    GUIDANCE_UPDATE = "guidance_update"
    ANALYST_DAY = "analyst_day"
    REGULATORY_DECISION = "regulatory_decision"

class MomentumDirection(str, Enum):
    BULLISH_BUILDUP = "bullish_buildup"
    BEARISH_BUILDUP = "bearish_buildup"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"

@dataclass
class EventWindow:
    event_type: EventType
    event_date: datetime
    symbol: str
    pre_event_hours: int = 72  # 72 hours before event
    post_event_hours: int = 24  # 24 hours after event
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MomentumMetrics:
    timestamp: datetime
    sentiment_ema_4h: float  # 4-hour EMA
    sentiment_ema_12h: float  # 12-hour EMA
    sentiment_ema_24h: float  # 24-hour EMA
    
    # Acceleration metrics
    ema_acceleration_4h: float  # Rate of change in 4h EMA
    ema_acceleration_12h: float  # Rate of change in 12h EMA
    
    # Volume-weighted sentiment
    volume_weighted_sentiment: float
    sentiment_volume_ratio: float
    
    # Positioning metrics
    bullish_momentum_score: float  # 0-1 score for bullish positioning
    bearish_momentum_score: float  # 0-1 score for bearish positioning
    momentum_direction: MomentumDirection
    
    # Confidence metrics
    momentum_strength: float  # 0-1 strength of momentum signal
    signal_confidence: float  # 0-1 confidence in direction

@dataclass
class PreEventAnalysis:
    event_window: EventWindow
    momentum_timeline: List[MomentumMetrics]
    
    # Summary metrics
    peak_bullish_momentum: float
    peak_bearish_momentum: float
    momentum_buildup_score: float  # Overall positioning buildup score
    direction_consistency: float  # How consistent the direction is
    
    # Prediction signals
    predicted_direction: MomentumDirection
    signal_strength: float
    confidence_score: float
    
    # Validation (if post-event data available)
    actual_price_move: Optional[float] = None
    prediction_accuracy: Optional[bool] = None

@dataclass
class EventOutcome:
    event_window: EventWindow
    pre_event_analysis: PreEventAnalysis
    
    # Actual market reaction
    price_move_24h: float  # Price movement 24h after event
    price_move_1w: float   # Price movement 1 week after event
    volume_spike: float    # Volume increase during event
    volatility_spike: float # Volatility increase during event
    
    # Performance metrics
    momentum_prediction_accuracy: bool
    direction_prediction_accuracy: bool
    signal_strength_correlation: float

class SentimentMomentumAnalyzer:
    """Analyzes sentiment momentum in event-specific windows"""
    
    def __init__(self):
        self.ema_alpha_4h = 2 / (4 + 1)   # 4-hour EMA smoothing factor
        self.ema_alpha_12h = 2 / (12 + 1) # 12-hour EMA smoothing factor
        self.ema_alpha_24h = 2 / (24 + 1) # 24-hour EMA smoothing factor
        
        # Known events cache
        self.events_cache = {}
        
    async def analyze_pre_event_momentum(self, event_window: EventWindow, 
                                       db: Session) -> PreEventAnalysis:
        """
        Analyze sentiment momentum in the pre-event window.
        
        Args:
            event_window: Event information and time windows
            db: Database session
            
        Returns:
            Comprehensive pre-event momentum analysis
        """
        try:
            logger.info(f"Analyzing pre-event momentum for {event_window.symbol} {event_window.event_type.value}")
            
            # Get sentiment data for the pre-event window
            start_time = event_window.event_date - timedelta(hours=event_window.pre_event_hours)
            end_time = event_window.event_date
            
            sentiment_data = await self._get_sentiment_data(
                db, event_window.symbol, start_time, end_time
            )
            
            if len(sentiment_data) < 10:
                logger.warning(f"Insufficient sentiment data for {event_window.symbol}: {len(sentiment_data)} posts")
                return self._create_empty_analysis(event_window)
            
            # Calculate momentum metrics timeline
            momentum_timeline = self._calculate_momentum_timeline(sentiment_data, event_window)
            
            # Analyze momentum patterns
            analysis = self._analyze_momentum_patterns(event_window, momentum_timeline)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Pre-event momentum analysis failed: {e}")
            return self._create_empty_analysis(event_window)
    
    def _calculate_momentum_timeline(self, sentiment_data: pd.DataFrame, 
                                   event_window: EventWindow) -> List[MomentumMetrics]:
        """Calculate momentum metrics over time"""
        try:
            # Resample sentiment data to hourly buckets
            sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'])
            sentiment_data.set_index('timestamp', inplace=True)
            
            # Aggregate by hour
            hourly_agg = sentiment_data.groupby(pd.Grouper(freq='H')).agg({
                'sentiment_score': ['mean', 'std', 'count'],
                'confidence': 'mean',
                'engagement': lambda x: x.apply(lambda y: y.get('likes', 0) if isinstance(y, dict) else 0).sum()
            }).fillna(0)
            
            # Flatten column names
            hourly_agg.columns = ['sentiment_mean', 'sentiment_std', 'post_count', 'confidence_mean', 'total_engagement']
            
            # Calculate EMAs and momentum metrics
            momentum_timeline = []
            ema_4h = None
            ema_12h = None
            ema_24h = None
            prev_ema_4h = None
            prev_ema_12h = None
            
            for timestamp, row in hourly_agg.iterrows():
                if row['post_count'] == 0:
                    continue
                
                sentiment = row['sentiment_mean']
                post_count = row['post_count']
                
                # Initialize EMAs
                if ema_4h is None:
                    ema_4h = sentiment
                    ema_12h = sentiment
                    ema_24h = sentiment
                else:
                    # Update EMAs
                    prev_ema_4h = ema_4h
                    prev_ema_12h = ema_12h
                    
                    ema_4h = (sentiment * self.ema_alpha_4h) + (ema_4h * (1 - self.ema_alpha_4h))
                    ema_12h = (sentiment * self.ema_alpha_12h) + (ema_12h * (1 - self.ema_alpha_12h))
                    ema_24h = (sentiment * self.ema_alpha_24h) + (ema_24h * (1 - self.ema_alpha_24h))
                
                # Calculate acceleration (rate of change in EMA)
                ema_accel_4h = (ema_4h - prev_ema_4h) if prev_ema_4h is not None else 0
                ema_accel_12h = (ema_12h - prev_ema_12h) if prev_ema_12h is not None else 0
                
                # Volume-weighted sentiment (using post count as proxy for volume)
                volume_weighted_sentiment = sentiment * np.log(1 + post_count)
                sentiment_volume_ratio = sentiment * post_count / max(hourly_agg['post_count'].mean(), 1)
                
                # Calculate positioning scores
                bullish_score, bearish_score = self._calculate_positioning_scores(
                    sentiment, ema_4h, ema_12h, ema_accel_4h, ema_accel_12h, post_count
                )
                
                # Determine momentum direction
                momentum_direction = self._determine_momentum_direction(bullish_score, bearish_score)
                
                # Calculate momentum strength and confidence
                momentum_strength = max(bullish_score, bearish_score)
                signal_confidence = self._calculate_signal_confidence(
                    ema_4h, ema_12h, ema_24h, row['sentiment_std'], row['confidence_mean']
                )
                
                metrics = MomentumMetrics(
                    timestamp=timestamp,
                    sentiment_ema_4h=ema_4h,
                    sentiment_ema_12h=ema_12h,
                    sentiment_ema_24h=ema_24h,
                    ema_acceleration_4h=ema_accel_4h,
                    ema_acceleration_12h=ema_accel_12h,
                    volume_weighted_sentiment=volume_weighted_sentiment,
                    sentiment_volume_ratio=sentiment_volume_ratio,
                    bullish_momentum_score=bullish_score,
                    bearish_momentum_score=bearish_score,
                    momentum_direction=momentum_direction,
                    momentum_strength=momentum_strength,
                    signal_confidence=signal_confidence
                )
                
                momentum_timeline.append(metrics)
            
            return momentum_timeline
            
        except Exception as e:
            logger.error(f"Error calculating momentum timeline: {e}")
            return []
    
    def _calculate_positioning_scores(self, sentiment: float, ema_4h: float, ema_12h: float,
                                    ema_accel_4h: float, ema_accel_12h: float, post_count: int) -> Tuple[float, float]:
        """Calculate bullish and bearish positioning scores"""
        try:
            # Bullish indicators
            bullish_factors = []
            
            # Sentiment above neutral
            bullish_factors.append(max(0, sentiment))
            
            # Short-term EMA above long-term EMA (golden cross signal)
            if ema_4h > ema_12h:
                bullish_factors.append(0.3)
            
            # Positive acceleration in EMAs
            bullish_factors.append(max(0, ema_accel_4h) * 2)
            bullish_factors.append(max(0, ema_accel_12h) * 1.5)
            
            # Volume factor (more posts = stronger signal)
            volume_factor = min(1.0, np.log(1 + post_count) / 5)
            bullish_factors.append(volume_factor * max(0, sentiment))
            
            # Bearish indicators
            bearish_factors = []
            
            # Sentiment below neutral
            bearish_factors.append(max(0, -sentiment))
            
            # Short-term EMA below long-term EMA (death cross signal)
            if ema_4h < ema_12h:
                bearish_factors.append(0.3)
            
            # Negative acceleration in EMAs
            bearish_factors.append(max(0, -ema_accel_4h) * 2)
            bearish_factors.append(max(0, -ema_accel_12h) * 1.5)
            
            # Volume factor for bearish sentiment
            bearish_factors.append(volume_factor * max(0, -sentiment))
            
            # Normalize scores to 0-1 range
            bullish_score = min(1.0, sum(bullish_factors) / 3)
            bearish_score = min(1.0, sum(bearish_factors) / 3)
            
            return bullish_score, bearish_score
            
        except Exception as e:
            logger.error(f"Error calculating positioning scores: {e}")
            return 0.0, 0.0
    
    def _determine_momentum_direction(self, bullish_score: float, bearish_score: float) -> MomentumDirection:
        """Determine overall momentum direction"""
        diff = abs(bullish_score - bearish_score)
        
        if diff < 0.1:  # Very close scores
            return MomentumDirection.NEUTRAL
        elif bullish_score > bearish_score + 0.2:
            return MomentumDirection.BULLISH_BUILDUP
        elif bearish_score > bullish_score + 0.2:
            return MomentumDirection.BEARISH_BUILDUP
        else:
            return MomentumDirection.VOLATILE
    
    def _calculate_signal_confidence(self, ema_4h: float, ema_12h: float, ema_24h: float,
                                   sentiment_std: float, confidence_mean: float) -> float:
        """Calculate confidence in the momentum signal"""
        try:
            confidence_factors = []
            
            # EMA alignment (all pointing same direction)
            if (ema_4h > ema_12h > ema_24h) or (ema_4h < ema_12h < ema_24h):
                confidence_factors.append(0.4)
            elif (ema_4h > ema_12h) or (ema_12h > ema_24h):
                confidence_factors.append(0.2)
            
            # Low volatility in sentiment (more consistent signal)
            volatility_factor = max(0, 1 - sentiment_std)
            confidence_factors.append(volatility_factor * 0.3)
            
            # High confidence in individual sentiment scores
            confidence_factors.append(confidence_mean * 0.3)
            
            return min(1.0, sum(confidence_factors))
            
        except Exception as e:
            logger.error(f"Error calculating signal confidence: {e}")
            return 0.0
    
    def _analyze_momentum_patterns(self, event_window: EventWindow, 
                                 momentum_timeline: List[MomentumMetrics]) -> PreEventAnalysis:
        """Analyze patterns in the momentum timeline"""
        try:
            if not momentum_timeline:
                return self._create_empty_analysis(event_window)
            
            # Extract key metrics
            bullish_scores = [m.bullish_momentum_score for m in momentum_timeline]
            bearish_scores = [m.bearish_momentum_score for m in momentum_timeline]
            
            peak_bullish_momentum = max(bullish_scores)
            peak_bearish_momentum = max(bearish_scores)
            
            # Calculate momentum buildup score (increasing trend toward event)
            buildup_score = self._calculate_buildup_score(momentum_timeline)
            
            # Calculate direction consistency
            directions = [m.momentum_direction for m in momentum_timeline]
            direction_consistency = self._calculate_direction_consistency(directions)
            
            # Predict overall direction based on recent momentum
            recent_metrics = momentum_timeline[-6:]  # Last 6 hours
            predicted_direction = self._predict_event_direction(recent_metrics)
            
            # Calculate signal strength (average of recent momentum strength)
            recent_strengths = [m.momentum_strength for m in recent_metrics]
            signal_strength = np.mean(recent_strengths) if recent_strengths else 0
            
            # Calculate overall confidence
            recent_confidences = [m.signal_confidence for m in recent_metrics]
            confidence_score = np.mean(recent_confidences) if recent_confidences else 0
            
            # Boost confidence for consistent patterns
            confidence_score *= direction_consistency
            
            return PreEventAnalysis(
                event_window=event_window,
                momentum_timeline=momentum_timeline,
                peak_bullish_momentum=peak_bullish_momentum,
                peak_bearish_momentum=peak_bearish_momentum,
                momentum_buildup_score=buildup_score,
                direction_consistency=direction_consistency,
                predicted_direction=predicted_direction,
                signal_strength=signal_strength,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error analyzing momentum patterns: {e}")
            return self._create_empty_analysis(event_window)
    
    def _calculate_buildup_score(self, momentum_timeline: List[MomentumMetrics]) -> float:
        """Calculate momentum buildup score (trend toward event)"""
        try:
            if len(momentum_timeline) < 3:
                return 0.0
            
            # Divide timeline into early, middle, late periods
            n = len(momentum_timeline)
            early_period = momentum_timeline[:n//3]
            late_period = momentum_timeline[2*n//3:]
            
            # Calculate average momentum strength for each period
            early_strength = np.mean([m.momentum_strength for m in early_period])
            late_strength = np.mean([m.momentum_strength for m in late_period])
            
            # Buildup score is the increase from early to late period
            buildup_score = max(0, late_strength - early_strength)
            
            return min(1.0, buildup_score)
            
        except Exception as e:
            logger.error(f"Error calculating buildup score: {e}")
            return 0.0
    
    def _calculate_direction_consistency(self, directions: List[MomentumDirection]) -> float:
        """Calculate consistency of momentum direction"""
        try:
            if not directions:
                return 0.0
            
            # Count occurrences of each direction
            direction_counts = {}
            for direction in directions:
                direction_counts[direction] = direction_counts.get(direction, 0) + 1
            
            # Calculate consistency as the proportion of the most common direction
            max_count = max(direction_counts.values())
            consistency = max_count / len(directions)
            
            return consistency
            
        except Exception as e:
            logger.error(f"Error calculating direction consistency: {e}")
            return 0.0
    
    def _predict_event_direction(self, recent_metrics: List[MomentumMetrics]) -> MomentumDirection:
        """Predict direction based on recent momentum"""
        try:
            if not recent_metrics:
                return MomentumDirection.NEUTRAL
            
            # Weight recent metrics more heavily
            total_bullish = 0
            total_bearish = 0
            total_weight = 0
            
            for i, metric in enumerate(recent_metrics):
                weight = i + 1  # More recent = higher weight
                total_bullish += metric.bullish_momentum_score * weight
                total_bearish += metric.bearish_momentum_score * weight
                total_weight += weight
            
            avg_bullish = total_bullish / total_weight
            avg_bearish = total_bearish / total_weight
            
            if avg_bullish > avg_bearish + 0.1:
                return MomentumDirection.BULLISH_BUILDUP
            elif avg_bearish > avg_bullish + 0.1:
                return MomentumDirection.BEARISH_BUILDUP
            else:
                return MomentumDirection.NEUTRAL
                
        except Exception as e:
            logger.error(f"Error predicting event direction: {e}")
            return MomentumDirection.NEUTRAL
    
    async def validate_momentum_signals(self, analysis: PreEventAnalysis, 
                                      db: Session) -> EventOutcome:
        """Validate momentum signals against actual price movements"""
        try:
            logger.info(f"Validating momentum signals for {analysis.event_window.symbol}")
            
            # Get price data for validation
            price_data = await self._get_price_data_for_validation(analysis.event_window)
            
            if price_data is None:
                logger.warning("No price data available for validation")
                return self._create_empty_outcome(analysis)
            
            # Calculate actual price movements
            event_date = analysis.event_window.event_date
            price_move_24h = self._calculate_price_move(price_data, event_date, hours=24)
            price_move_1w = self._calculate_price_move(price_data, event_date, hours=24*7)
            
            # Calculate volume and volatility spikes
            volume_spike = self._calculate_volume_spike(price_data, event_date)
            volatility_spike = self._calculate_volatility_spike(price_data, event_date)
            
            # Evaluate prediction accuracy
            momentum_accuracy = self._evaluate_momentum_prediction(
                analysis.predicted_direction, price_move_24h
            )
            
            direction_accuracy = self._evaluate_direction_prediction(
                analysis.predicted_direction, price_move_24h
            )
            
            # Calculate correlation between signal strength and price move magnitude
            signal_correlation = abs(analysis.signal_strength * price_move_24h)
            
            outcome = EventOutcome(
                event_window=analysis.event_window,
                pre_event_analysis=analysis,
                price_move_24h=price_move_24h,
                price_move_1w=price_move_1w,
                volume_spike=volume_spike,
                volatility_spike=volatility_spike,
                momentum_prediction_accuracy=momentum_accuracy,
                direction_prediction_accuracy=direction_accuracy,
                signal_strength_correlation=signal_correlation
            )
            
            return outcome
            
        except Exception as e:
            logger.error(f"Error validating momentum signals: {e}")
            return self._create_empty_outcome(analysis)
    
    async def _get_sentiment_data(self, db: Session, symbol: str, 
                                start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get sentiment data from database"""
        try:
            from ..core.database import SentimentPost
            
            query = db.query(SentimentPost).filter(
                SentimentPost.symbol == symbol,
                SentimentPost.post_timestamp >= start_time,
                SentimentPost.post_timestamp <= end_time
            ).order_by(SentimentPost.post_timestamp)
            
            posts = query.all()
            
            if not posts:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for post in posts:
                data.append({
                    'timestamp': post.post_timestamp,
                    'sentiment_score': post.sentiment_score,
                    'confidence': post.confidence,
                    'engagement': post.engagement or {},
                    'content': post.content
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error fetching sentiment data: {e}")
            return pd.DataFrame()
    
    async def _get_price_data_for_validation(self, event_window: EventWindow) -> Optional[pd.DataFrame]:
        """Get price data for validation"""
        try:
            start_date = event_window.event_date - timedelta(days=7)
            end_date = event_window.event_date + timedelta(days=7)
            
            ticker = yf.Ticker(event_window.symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1h')
            
            if df.empty:
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            return None
    
    def _calculate_price_move(self, price_data: pd.DataFrame, event_date: datetime, hours: int) -> float:
        """Calculate price movement after event"""
        try:
            event_price = price_data.loc[price_data.index <= event_date, 'Close'].iloc[-1]
            future_date = event_date + timedelta(hours=hours)
            future_price = price_data.loc[price_data.index <= future_date, 'Close'].iloc[-1]
            
            return (future_price - event_price) / event_price
            
        except Exception as e:
            logger.error(f"Error calculating price move: {e}")
            return 0.0
    
    def _calculate_volume_spike(self, price_data: pd.DataFrame, event_date: datetime) -> float:
        """Calculate volume spike during event"""
        try:
            pre_event_volume = price_data.loc[price_data.index < event_date, 'Volume'].tail(24).mean()
            event_volume = price_data.loc[price_data.index >= event_date, 'Volume'].head(6).mean()
            
            return event_volume / pre_event_volume if pre_event_volume > 0 else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating volume spike: {e}")
            return 1.0
    
    def _calculate_volatility_spike(self, price_data: pd.DataFrame, event_date: datetime) -> float:
        """Calculate volatility spike during event"""
        try:
            pre_event_returns = price_data.loc[price_data.index < event_date, 'Close'].pct_change().tail(24)
            event_returns = price_data.loc[price_data.index >= event_date, 'Close'].pct_change().head(6)
            
            pre_event_vol = pre_event_returns.std()
            event_vol = event_returns.std()
            
            return event_vol / pre_event_vol if pre_event_vol > 0 else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating volatility spike: {e}")
            return 1.0
    
    def _evaluate_momentum_prediction(self, predicted_direction: MomentumDirection, actual_move: float) -> bool:
        """Evaluate if momentum prediction was correct"""
        if predicted_direction == MomentumDirection.BULLISH_BUILDUP:
            return actual_move > 0.01  # 1% threshold
        elif predicted_direction == MomentumDirection.BEARISH_BUILDUP:
            return actual_move < -0.01  # -1% threshold
        else:
            return abs(actual_move) <= 0.01  # Neutral/volatile prediction
    
    def _evaluate_direction_prediction(self, predicted_direction: MomentumDirection, actual_move: float) -> bool:
        """Evaluate if direction prediction was correct"""
        if predicted_direction == MomentumDirection.BULLISH_BUILDUP:
            return actual_move > 0
        elif predicted_direction == MomentumDirection.BEARISH_BUILDUP:
            return actual_move < 0
        else:
            return True  # Neutral predictions are always "correct"
    
    def _create_empty_analysis(self, event_window: EventWindow) -> PreEventAnalysis:
        """Create empty analysis for insufficient data"""
        return PreEventAnalysis(
            event_window=event_window,
            momentum_timeline=[],
            peak_bullish_momentum=0.0,
            peak_bearish_momentum=0.0,
            momentum_buildup_score=0.0,
            direction_consistency=0.0,
            predicted_direction=MomentumDirection.NEUTRAL,
            signal_strength=0.0,
            confidence_score=0.0
        )
    
    def _create_empty_outcome(self, analysis: PreEventAnalysis) -> EventOutcome:
        """Create empty outcome for validation failure"""
        return EventOutcome(
            event_window=analysis.event_window,
            pre_event_analysis=analysis,
            price_move_24h=0.0,
            price_move_1w=0.0,
            volume_spike=1.0,
            volatility_spike=1.0,
            momentum_prediction_accuracy=False,
            direction_prediction_accuracy=False,
            signal_strength_correlation=0.0
        )