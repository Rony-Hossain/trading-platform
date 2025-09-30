"""
Gap Trading Engine

Comprehensive gap analysis system for identifying and analyzing gap continuation vs fade patterns
with sophisticated pre/post-market price handling and gap classification algorithms.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, time
from enum import Enum
from typing import Dict, List, Optional, Tuple, NamedTuple
import pandas as pd
import numpy as np
from decimal import Decimal

class GapType(Enum):
    """Gap classification types"""
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    
class GapSize(Enum):
    """Gap size classifications"""
    MICRO = "micro"          # < 1%
    SMALL = "small"          # 1-3%
    MEDIUM = "medium"        # 3-7%
    LARGE = "large"          # 7-15%
    MASSIVE = "massive"      # > 15%

class GapDirection(Enum):
    """Gap continuation analysis"""
    CONTINUATION = "continuation"  # Gap continues in same direction
    FADE = "fade"                 # Gap fills/reverses
    PARTIAL_FILL = "partial_fill" # Gap partially fills
    NEUTRAL = "neutral"           # No clear direction

class MarketSession(Enum):
    """Market session types"""
    PRE_MARKET = "pre_market"     # 4:00 AM - 9:30 AM ET
    REGULAR = "regular"           # 9:30 AM - 4:00 PM ET
    AFTER_HOURS = "after_hours"   # 4:00 PM - 8:00 PM ET
    OVERNIGHT = "overnight"       # 8:00 PM - 4:00 AM ET

@dataclass
class PrePostMarketData:
    """Pre/post-market trading data"""
    session: MarketSession
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    vwap: float
    session_start: datetime
    session_end: datetime
    trade_count: int
    spread_avg: float
    liquidity_score: float

@dataclass
class GapAnalysis:
    """Comprehensive gap analysis result"""
    symbol: str
    detection_time: datetime
    gap_type: GapType
    gap_size: GapSize
    gap_percentage: float
    gap_points: float
    
    # Price levels
    previous_close: float
    current_open: float
    gap_high: float
    gap_low: float
    
    # Pre/post market context
    pre_market_data: Optional[PrePostMarketData]
    overnight_data: Optional[PrePostMarketData]
    
    # Gap characteristics
    volume_surge: bool
    volume_ratio: float
    news_catalyst: bool
    earnings_gap: bool
    
    # Technical analysis
    support_resistance_levels: List[float]
    fibonacci_levels: Dict[str, float]
    gap_fill_probability: float
    continuation_probability: float
    
    # Risk metrics
    volatility_context: float
    average_gap_size: float
    historical_gap_behavior: Dict[str, float]
    
    # Trading metrics
    optimal_entry_price: float
    stop_loss_level: float
    profit_targets: List[float]
    risk_reward_ratio: float

@dataclass
class GapMonitoringResult:
    """Real-time gap monitoring result"""
    gap_analysis: GapAnalysis
    current_direction: GapDirection
    fill_percentage: float
    time_since_open: timedelta
    volume_profile: Dict[str, float]
    price_action_strength: float
    momentum_indicators: Dict[str, float]
    liquidity_analysis: Dict[str, float]

class GapTradingEngine:
    """Advanced gap trading analysis engine"""
    
    def __init__(self):
        self.market_hours = {
            'pre_market_start': time(4, 0),    # 4:00 AM ET
            'market_open': time(9, 30),        # 9:30 AM ET
            'market_close': time(16, 0),       # 4:00 PM ET
            'after_hours_end': time(20, 0)     # 8:00 PM ET
        }
        
        # Gap classification thresholds
        self.gap_size_thresholds = {
            GapSize.MICRO: 0.01,     # 1%
            GapSize.SMALL: 0.03,     # 3%
            GapSize.MEDIUM: 0.07,    # 7%
            GapSize.LARGE: 0.15      # 15%
        }
        
        # Historical gap statistics for probability calculations
        self.gap_statistics = {
            'continuation_rate_by_size': {
                GapSize.MICRO: 0.45,
                GapSize.SMALL: 0.55,
                GapSize.MEDIUM: 0.68,
                GapSize.LARGE: 0.72,
                GapSize.MASSIVE: 0.78
            },
            'fill_rate_by_size': {
                GapSize.MICRO: 0.85,
                GapSize.SMALL: 0.72,
                GapSize.MEDIUM: 0.58,
                GapSize.LARGE: 0.42,
                GapSize.MASSIVE: 0.28
            }
        }
        
    async def analyze_gap(
        self,
        symbol: str,
        current_price_data: Dict,
        previous_close: float,
        pre_market_data: Optional[Dict] = None,
        overnight_data: Optional[Dict] = None,
        volume_data: Optional[Dict] = None,
        news_events: Optional[List] = None
    ) -> GapAnalysis:
        """
        Comprehensive gap analysis with pre/post-market context
        """
        current_open = current_price_data.get('open', current_price_data.get('price'))
        
        # Calculate gap metrics
        gap_points = current_open - previous_close
        gap_percentage = gap_points / previous_close
        
        # Classify gap
        gap_type = GapType.GAP_UP if gap_points > 0 else GapType.GAP_DOWN
        gap_size = self._classify_gap_size(abs(gap_percentage))
        
        # Process pre/post market data
        pre_market_analysis = None
        if pre_market_data:
            pre_market_analysis = self._analyze_pre_market_session(pre_market_data)
            
        overnight_analysis = None
        if overnight_data:
            overnight_analysis = self._analyze_overnight_session(overnight_data)
        
        # Volume analysis
        volume_surge = False
        volume_ratio = 1.0
        if volume_data:
            avg_volume = volume_data.get('average_volume', 1)
            current_volume = volume_data.get('current_volume', avg_volume)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            volume_surge = volume_ratio > 2.0
        
        # News catalyst detection
        news_catalyst = bool(news_events)
        earnings_gap = self._detect_earnings_gap(news_events) if news_events else False
        
        # Technical analysis
        support_resistance = self._calculate_support_resistance(
            symbol, current_open, previous_close, gap_percentage
        )
        fibonacci_levels = self._calculate_fibonacci_levels(
            previous_close, current_open, gap_type
        )
        
        # Probability calculations
        gap_fill_prob = self._calculate_gap_fill_probability(
            gap_size, volume_surge, news_catalyst, earnings_gap
        )
        continuation_prob = self._calculate_continuation_probability(
            gap_size, volume_surge, pre_market_analysis, news_catalyst
        )
        
        # Risk metrics
        volatility_context = self._calculate_volatility_context(symbol, gap_percentage)
        avg_gap_size = self._get_historical_average_gap_size(symbol)
        historical_behavior = self._analyze_historical_gap_behavior(symbol, gap_size)
        
        # Trading setup
        entry_price, stop_loss, targets, risk_reward = self._calculate_trading_setup(
            current_open, previous_close, gap_type, gap_size, support_resistance
        )
        
        return GapAnalysis(
            symbol=symbol,
            detection_time=datetime.utcnow(),
            gap_type=gap_type,
            gap_size=gap_size,
            gap_percentage=gap_percentage,
            gap_points=gap_points,
            previous_close=previous_close,
            current_open=current_open,
            gap_high=max(current_open, previous_close),
            gap_low=min(current_open, previous_close),
            pre_market_data=pre_market_analysis,
            overnight_data=overnight_analysis,
            volume_surge=volume_surge,
            volume_ratio=volume_ratio,
            news_catalyst=news_catalyst,
            earnings_gap=earnings_gap,
            support_resistance_levels=support_resistance,
            fibonacci_levels=fibonacci_levels,
            gap_fill_probability=gap_fill_prob,
            continuation_probability=continuation_prob,
            volatility_context=volatility_context,
            average_gap_size=avg_gap_size,
            historical_gap_behavior=historical_behavior,
            optimal_entry_price=entry_price,
            stop_loss_level=stop_loss,
            profit_targets=targets,
            risk_reward_ratio=risk_reward
        )
    
    async def monitor_gap_behavior(
        self,
        gap_analysis: GapAnalysis,
        current_price_data: Dict,
        volume_profile: Optional[Dict] = None
    ) -> GapMonitoringResult:
        """
        Real-time monitoring of gap behavior after market open
        """
        current_price = current_price_data.get('price', gap_analysis.current_open)
        current_time = datetime.utcnow()
        
        # Calculate gap fill percentage
        fill_percentage = self._calculate_gap_fill_percentage(
            gap_analysis, current_price
        )
        
        # Determine current direction
        direction = self._determine_gap_direction(gap_analysis, current_price, fill_percentage)
        
        # Time analysis
        time_since_open = current_time - gap_analysis.detection_time
        
        # Volume profile analysis
        volume_analysis = self._analyze_volume_profile(volume_profile) if volume_profile else {}
        
        # Price action strength
        price_strength = self._calculate_price_action_strength(
            gap_analysis, current_price, time_since_open
        )
        
        # Momentum indicators
        momentum = self._calculate_momentum_indicators(
            gap_analysis, current_price_data, time_since_open
        )
        
        # Liquidity analysis
        liquidity = self._analyze_liquidity_conditions(current_price_data, volume_profile)
        
        return GapMonitoringResult(
            gap_analysis=gap_analysis,
            current_direction=direction,
            fill_percentage=fill_percentage,
            time_since_open=time_since_open,
            volume_profile=volume_analysis,
            price_action_strength=price_strength,
            momentum_indicators=momentum,
            liquidity_analysis=liquidity
        )
    
    def _classify_gap_size(self, gap_percentage: float) -> GapSize:
        """Classify gap size based on percentage"""
        if gap_percentage >= self.gap_size_thresholds[GapSize.LARGE]:
            return GapSize.MASSIVE
        elif gap_percentage >= self.gap_size_thresholds[GapSize.MEDIUM]:
            return GapSize.LARGE
        elif gap_percentage >= self.gap_size_thresholds[GapSize.SMALL]:
            return GapSize.MEDIUM
        elif gap_percentage >= self.gap_size_thresholds[GapSize.MICRO]:
            return GapSize.SMALL
        else:
            return GapSize.MICRO
    
    def _analyze_pre_market_session(self, pre_market_data: Dict) -> PrePostMarketData:
        """Analyze pre-market trading session"""
        return PrePostMarketData(
            session=MarketSession.PRE_MARKET,
            open_price=pre_market_data.get('open', 0),
            high_price=pre_market_data.get('high', 0),
            low_price=pre_market_data.get('low', 0),
            close_price=pre_market_data.get('close', 0),
            volume=pre_market_data.get('volume', 0),
            vwap=pre_market_data.get('vwap', 0),
            session_start=pre_market_data.get('session_start'),
            session_end=pre_market_data.get('session_end'),
            trade_count=pre_market_data.get('trade_count', 0),
            spread_avg=pre_market_data.get('spread_avg', 0),
            liquidity_score=self._calculate_liquidity_score(pre_market_data)
        )
    
    def _analyze_overnight_session(self, overnight_data: Dict) -> PrePostMarketData:
        """Analyze overnight trading session"""
        return PrePostMarketData(
            session=MarketSession.OVERNIGHT,
            open_price=overnight_data.get('open', 0),
            high_price=overnight_data.get('high', 0),
            low_price=overnight_data.get('low', 0),
            close_price=overnight_data.get('close', 0),
            volume=overnight_data.get('volume', 0),
            vwap=overnight_data.get('vwap', 0),
            session_start=overnight_data.get('session_start'),
            session_end=overnight_data.get('session_end'),
            trade_count=overnight_data.get('trade_count', 0),
            spread_avg=overnight_data.get('spread_avg', 0),
            liquidity_score=self._calculate_liquidity_score(overnight_data)
        )
    
    def _calculate_liquidity_score(self, session_data: Dict) -> float:
        """Calculate liquidity score for trading session"""
        volume = session_data.get('volume', 0)
        trade_count = session_data.get('trade_count', 1)
        spread = session_data.get('spread_avg', 0.01)
        
        # Higher volume, more trades, tighter spreads = higher liquidity
        volume_score = min(1.0, volume / 100000)  # Normalize volume
        trade_score = min(1.0, trade_count / 1000)  # Normalize trade count
        spread_score = max(0.0, 1.0 - spread * 100)  # Lower spread = higher score
        
        return (volume_score + trade_score + spread_score) / 3
    
    def _detect_earnings_gap(self, news_events: List) -> bool:
        """Detect if gap is earnings-related"""
        if not news_events:
            return False
            
        earnings_keywords = ['earnings', 'eps', 'revenue', 'quarterly', 'guidance']
        for event in news_events:
            event_text = str(event).lower()
            if any(keyword in event_text for keyword in earnings_keywords):
                return True
        return False
    
    def _calculate_support_resistance(
        self, symbol: str, current_open: float, previous_close: float, gap_percentage: float
    ) -> List[float]:
        """Calculate key support/resistance levels"""
        gap_midpoint = (current_open + previous_close) / 2
        
        # Basic support/resistance levels
        levels = [
            previous_close,  # Gap fill level
            gap_midpoint,    # 50% gap fill
            current_open     # Gap origin
        ]
        
        # Add percentage-based levels
        gap_range = abs(current_open - previous_close)
        levels.extend([
            current_open + gap_range * 0.618,  # 61.8% extension
            current_open + gap_range * 1.0,    # 100% extension
            current_open - gap_range * 0.382,  # 38.2% retracement
        ])
        
        return sorted(set(levels))
    
    def _calculate_fibonacci_levels(
        self, previous_close: float, current_open: float, gap_type: GapType
    ) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels for gap"""
        gap_range = current_open - previous_close
        
        if gap_type == GapType.GAP_UP:
            return {
                '0%': current_open,
                '23.6%': current_open - gap_range * 0.236,
                '38.2%': current_open - gap_range * 0.382,
                '50%': current_open - gap_range * 0.5,
                '61.8%': current_open - gap_range * 0.618,
                '100%': previous_close
            }
        else:
            return {
                '0%': current_open,
                '23.6%': current_open + gap_range * 0.236,
                '38.2%': current_open + gap_range * 0.382,
                '50%': current_open + gap_range * 0.5,
                '61.8%': current_open + gap_range * 0.618,
                '100%': previous_close
            }
    
    def _calculate_gap_fill_probability(
        self, gap_size: GapSize, volume_surge: bool, news_catalyst: bool, earnings_gap: bool
    ) -> float:
        """Calculate probability of gap filling"""
        base_probability = self.gap_statistics['fill_rate_by_size'][gap_size]
        
        # Adjust based on factors
        if volume_surge:
            base_probability *= 0.85  # High volume reduces fill probability
        if news_catalyst:
            base_probability *= 0.75  # News catalyst reduces fill probability
        if earnings_gap:
            base_probability *= 0.65  # Earnings gaps less likely to fill quickly
            
        return max(0.1, min(0.95, base_probability))
    
    def _calculate_continuation_probability(
        self, gap_size: GapSize, volume_surge: bool, 
        pre_market_data: Optional[PrePostMarketData], news_catalyst: bool
    ) -> float:
        """Calculate probability of gap continuation"""
        base_probability = self.gap_statistics['continuation_rate_by_size'][gap_size]
        
        # Adjust based on factors
        if volume_surge:
            base_probability *= 1.15  # High volume increases continuation probability
        if news_catalyst:
            base_probability *= 1.25  # News catalyst increases continuation probability
        if pre_market_data and pre_market_data.liquidity_score > 0.7:
            base_probability *= 1.1   # Good pre-market liquidity helps continuation
            
        return max(0.1, min(0.95, base_probability))
    
    def _calculate_volatility_context(self, symbol: str, gap_percentage: float) -> float:
        """Calculate gap size relative to historical volatility"""
        # In a real implementation, this would use historical volatility data
        # For now, return a normalized value
        return min(2.0, abs(gap_percentage) / 0.02)  # Normalize to 2% baseline
    
    def _get_historical_average_gap_size(self, symbol: str) -> float:
        """Get historical average gap size for the symbol"""
        # In a real implementation, this would query historical data
        return 0.025  # 2.5% average gap
    
    def _analyze_historical_gap_behavior(self, symbol: str, gap_size: GapSize) -> Dict[str, float]:
        """Analyze historical gap behavior patterns"""
        # In a real implementation, this would analyze historical gap data
        return {
            'average_time_to_fill': 45.0,  # minutes
            'continuation_success_rate': self.gap_statistics['continuation_rate_by_size'][gap_size],
            'fade_success_rate': self.gap_statistics['fill_rate_by_size'][gap_size],
            'partial_fill_rate': 0.35
        }
    
    def _calculate_trading_setup(
        self, current_open: float, previous_close: float, gap_type: GapType,
        gap_size: GapSize, support_resistance: List[float]
    ) -> Tuple[float, float, List[float], float]:
        """Calculate optimal trading setup"""
        gap_range = abs(current_open - previous_close)
        
        if gap_type == GapType.GAP_UP:
            # For gap up - look for continuation
            entry_price = current_open + gap_range * 0.1  # Enter on slight pullback
            stop_loss = current_open - gap_range * 0.3    # Stop below gap
            targets = [
                current_open + gap_range * 0.5,  # First target
                current_open + gap_range * 1.0,  # Second target
                current_open + gap_range * 1.618 # Extended target
            ]
        else:
            # For gap down - look for continuation
            entry_price = current_open - gap_range * 0.1  # Enter on slight bounce
            stop_loss = current_open + gap_range * 0.3    # Stop above gap
            targets = [
                current_open - gap_range * 0.5,  # First target
                current_open - gap_range * 1.0,  # Second target
                current_open - gap_range * 1.618 # Extended target
            ]
        
        # Calculate risk/reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(targets[0] - entry_price)
        risk_reward = reward / risk if risk > 0 else 0
        
        return entry_price, stop_loss, targets, risk_reward
    
    def _calculate_gap_fill_percentage(self, gap_analysis: GapAnalysis, current_price: float) -> float:
        """Calculate what percentage of the gap has been filled"""
        gap_range = abs(gap_analysis.current_open - gap_analysis.previous_close)
        
        if gap_analysis.gap_type == GapType.GAP_UP:
            # For gap up, filling means price moving down toward previous close
            price_move = gap_analysis.current_open - current_price
            fill_percentage = price_move / gap_range if gap_range > 0 else 0
        else:
            # For gap down, filling means price moving up toward previous close
            price_move = current_price - gap_analysis.current_open
            fill_percentage = price_move / gap_range if gap_range > 0 else 0
            
        return max(0, min(1, fill_percentage))
    
    def _determine_gap_direction(
        self, gap_analysis: GapAnalysis, current_price: float, fill_percentage: float
    ) -> GapDirection:
        """Determine current gap direction behavior"""
        if fill_percentage >= 0.9:
            return GapDirection.FADE  # Almost fully filled
        elif fill_percentage >= 0.3:
            return GapDirection.PARTIAL_FILL
        elif fill_percentage <= 0.1:
            # Check if continuing in gap direction
            if gap_analysis.gap_type == GapType.GAP_UP and current_price > gap_analysis.current_open:
                return GapDirection.CONTINUATION
            elif gap_analysis.gap_type == GapType.GAP_DOWN and current_price < gap_analysis.current_open:
                return GapDirection.CONTINUATION
            else:
                return GapDirection.NEUTRAL
        else:
            return GapDirection.NEUTRAL
    
    def _analyze_volume_profile(self, volume_profile: Dict) -> Dict[str, float]:
        """Analyze volume profile characteristics"""
        if not volume_profile:
            return {}
            
        return {
            'volume_weighted_price': volume_profile.get('vwap', 0),
            'high_volume_price': volume_profile.get('poc', 0),  # Point of Control
            'volume_distribution': volume_profile.get('distribution', 0.5),
            'buying_pressure': volume_profile.get('buying_pressure', 0.5)
        }
    
    def _calculate_price_action_strength(
        self, gap_analysis: GapAnalysis, current_price: float, time_elapsed: timedelta
    ) -> float:
        """Calculate price action strength"""
        gap_range = abs(gap_analysis.current_open - gap_analysis.previous_close)
        price_change = abs(current_price - gap_analysis.current_open)
        
        # Normalize by time (stronger moves in shorter time get higher scores)
        time_factor = max(0.1, 60 / max(1, time_elapsed.total_seconds() / 60))  # minutes
        strength = (price_change / gap_range) * time_factor if gap_range > 0 else 0
        
        return min(2.0, strength)  # Cap at 2.0
    
    def _calculate_momentum_indicators(
        self, gap_analysis: GapAnalysis, current_price_data: Dict, time_elapsed: timedelta
    ) -> Dict[str, float]:
        """Calculate momentum indicators"""
        current_price = current_price_data.get('price', gap_analysis.current_open)
        
        # Simple momentum calculations
        price_momentum = (current_price - gap_analysis.current_open) / gap_analysis.current_open
        
        return {
            'price_momentum': price_momentum,
            'volume_momentum': current_price_data.get('volume_ratio', 1.0),
            'time_decay': max(0.1, 1.0 - time_elapsed.total_seconds() / 3600),  # Decay over 1 hour
            'gap_momentum': abs(gap_analysis.gap_percentage)
        }
    
    def _analyze_liquidity_conditions(
        self, current_price_data: Dict, volume_profile: Optional[Dict]
    ) -> Dict[str, float]:
        """Analyze current liquidity conditions"""
        return {
            'bid_ask_spread': current_price_data.get('spread', 0.01),
            'market_depth': current_price_data.get('depth_score', 0.5),
            'order_flow': current_price_data.get('order_flow', 0.5),
            'liquidity_score': volume_profile.get('liquidity_score', 0.5) if volume_profile else 0.5
        }