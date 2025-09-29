"""Gap Trading Engine

Advanced gap trading system that identifies pre/post-market gaps and determines
whether to trade for continuation or mean reversion (fade) based on multiple factors.
"""

import asyncio
import logging
from datetime import datetime, timedelta, time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

import httpx

logger = logging.getLogger(__name__)


class GapType(Enum):
    """Types of price gaps."""
    OPENING_GAP = "opening_gap"
    INTRADAY_GAP = "intraday_gap"
    OVERNIGHT_GAP = "overnight_gap"
    WEEKEND_GAP = "weekend_gap"


class GapDirection(Enum):
    """Direction of the gap."""
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"


class GapStrategy(Enum):
    """Gap trading strategy."""
    CONTINUATION = "continuation"  # Trade in direction of gap
    FADE = "fade"                 # Trade against the gap (mean reversion)
    HOLD = "hold"                 # No trade - wait and see


class MarketSession(Enum):
    """Market session timing."""
    PRE_MARKET = "pre_market"     # 4:00 AM - 9:30 AM ET
    REGULAR_HOURS = "regular_hours"  # 9:30 AM - 4:00 PM ET
    AFTER_HOURS = "after_hours"   # 4:00 PM - 8:00 PM ET
    OVERNIGHT = "overnight"       # 8:00 PM - 4:00 AM ET


@dataclass
class GapData:
    """Price gap information."""
    symbol: str
    gap_type: GapType
    gap_direction: GapDirection
    gap_size_dollars: float
    gap_size_percent: float
    previous_close: float
    current_price: float
    gap_open_price: float
    volume_ratio: float  # Current volume vs average
    timestamp: datetime
    session: MarketSession


@dataclass
class GapContext:
    """Additional context for gap analysis."""
    symbol: str
    average_true_range: float
    recent_volatility: float
    support_level: Optional[float]
    resistance_level: Optional[float]
    sector_performance: float
    market_performance: float
    earnings_proximity_days: Optional[int]
    has_recent_news: bool
    institutional_flow: float  # Net institutional buying/selling
    options_activity: Dict[str, float]


@dataclass
class GapSignal:
    """Gap trading signal."""
    symbol: str
    strategy: GapStrategy
    entry_price: float
    target_price: float
    stop_loss_price: float
    position_size_pct: float
    confidence: float
    holding_period_minutes: int
    gap_data: GapData
    context: GapContext
    reasoning: str
    generated_at: datetime
    expires_at: datetime


class GapTradingEngine:
    """Engine for gap trading strategy generation."""
    
    def __init__(self, 
                 market_data_url: str = "http://localhost:8002",
                 analysis_service_url: str = "http://localhost:8003"):
        self.market_data_url = market_data_url
        self.analysis_service_url = analysis_service_url
        self.client = None
        
        # Gap trading thresholds
        self.gap_thresholds = {
            "min_gap_percent": 0.02,      # 2% minimum gap
            "large_gap_percent": 0.05,    # 5% large gap threshold
            "huge_gap_percent": 0.10,     # 10% huge gap threshold
            "max_tradeable_gap": 0.20,    # 20% maximum gap to trade
        }
        
        # Volume thresholds
        self.volume_thresholds = {
            "min_volume_ratio": 1.5,      # 1.5x average volume
            "high_volume_ratio": 3.0,     # 3x average volume
            "extreme_volume_ratio": 5.0,  # 5x average volume
        }
        
        # Time thresholds
        self.time_windows = {
            "pre_market_start": time(4, 0),   # 4:00 AM ET
            "market_open": time(9, 30),       # 9:30 AM ET
            "market_close": time(16, 0),      # 4:00 PM ET
            "after_hours_end": time(20, 0),   # 8:00 PM ET
        }
        
        # Strategy parameters
        self.strategy_params = {
            "continuation": {
                "target_ratio": 0.618,     # 61.8% of gap fill as target
                "stop_ratio": 0.382,       # 38.2% gap retracement as stop
                "hold_minutes": 60,        # Hold for 1 hour max
                "min_confidence": 0.6      # Minimum confidence threshold
            },
            "fade": {
                "target_ratio": 0.85,      # 85% gap fill as target
                "stop_ratio": 0.15,        # 15% extension beyond gap as stop
                "hold_minutes": 120,       # Hold for 2 hours max
                "min_confidence": 0.65     # Minimum confidence threshold
            }
        }
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def analyze_gap_opportunity(self, symbol: str) -> Optional[GapSignal]:
        """Analyze gap trading opportunity for a symbol."""
        try:
            # 1. Detect gap
            gap_data = await self._detect_gap(symbol)
            if not gap_data:
                return None
            
            # 2. Check if gap is tradeable
            if not self._is_tradeable_gap(gap_data):
                logger.info(f"Gap for {symbol} is not tradeable: {gap_data.gap_size_percent:.2%}")
                return None
            
            # 3. Gather context
            context = await self._gather_gap_context(symbol, gap_data)
            if not context:
                return None
            
            # 4. Determine strategy
            strategy = self._determine_gap_strategy(gap_data, context)
            if strategy == GapStrategy.HOLD:
                logger.info(f"No clear gap strategy for {symbol}")
                return None
            
            # 5. Calculate signal parameters
            signal = self._calculate_gap_signal(gap_data, context, strategy)
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to analyze gap opportunity for {symbol}: {e}")
            return None
    
    async def _detect_gap(self, symbol: str) -> Optional[GapData]:
        """Detect price gaps for the symbol."""
        try:
            # Get current price and session info
            current_data = await self._get_current_price_data(symbol)
            if not current_data:
                return None
            
            current_price = current_data["price"]
            current_session = self._determine_market_session()
            
            # Get previous close
            previous_close = await self._get_previous_close(symbol)
            if not previous_close:
                return None
            
            # Calculate gap
            gap_size_dollars = current_price - previous_close
            gap_size_percent = gap_size_dollars / previous_close
            
            # Determine gap type and direction
            gap_direction = GapDirection.GAP_UP if gap_size_dollars > 0 else GapDirection.GAP_DOWN
            
            # Determine gap type based on timing
            if current_session == MarketSession.PRE_MARKET:
                gap_type = GapType.OVERNIGHT_GAP
            elif current_session == MarketSession.REGULAR_HOURS:
                gap_type = GapType.OPENING_GAP
            else:
                gap_type = GapType.INTRADAY_GAP
            
            # Get volume data
            volume_ratio = current_data.get("volume_ratio", 1.0)
            
            return GapData(
                symbol=symbol,
                gap_type=gap_type,
                gap_direction=gap_direction,
                gap_size_dollars=abs(gap_size_dollars),
                gap_size_percent=abs(gap_size_percent),
                previous_close=previous_close,
                current_price=current_price,
                gap_open_price=current_price,  # Simplification
                volume_ratio=volume_ratio,
                timestamp=datetime.utcnow(),
                session=current_session
            )
            
        except Exception as e:
            logger.error(f"Failed to detect gap for {symbol}: {e}")
            return None
    
    async def _get_current_price_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current price and volume data."""
        try:
            response = await self.client.get(
                f"{self.market_data_url}/quote/{symbol}"
            )
            response.raise_for_status()
            
            data = response.json()
            return {
                "price": data.get("price", data.get("last", 0)),
                "volume": data.get("volume", 0),
                "volume_ratio": data.get("volume_ratio", 1.0),
                "timestamp": data.get("timestamp")
            }
            
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return None
    
    async def _get_previous_close(self, symbol: str) -> Optional[float]:
        """Get previous trading day's closing price."""
        try:
            response = await self.client.get(
                f"{self.market_data_url}/history/{symbol}",
                params={"period": "2d", "interval": "1d"}
            )
            response.raise_for_status()
            
            data = response.json()
            if data and len(data) >= 2:
                return data[-2].get("close")  # Previous day's close
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get previous close for {symbol}: {e}")
            return None
    
    def _determine_market_session(self) -> MarketSession:
        """Determine current market session."""
        now = datetime.utcnow()
        current_time = now.time()
        
        # Convert to ET (simplified - assumes current time is ET)
        if self.time_windows["pre_market_start"] <= current_time < self.time_windows["market_open"]:
            return MarketSession.PRE_MARKET
        elif self.time_windows["market_open"] <= current_time < self.time_windows["market_close"]:
            return MarketSession.REGULAR_HOURS
        elif self.time_windows["market_close"] <= current_time < self.time_windows["after_hours_end"]:
            return MarketSession.AFTER_HOURS
        else:
            return MarketSession.OVERNIGHT
    
    def _is_tradeable_gap(self, gap_data: GapData) -> bool:
        """Check if gap meets trading criteria."""
        return (
            gap_data.gap_size_percent >= self.gap_thresholds["min_gap_percent"] and
            gap_data.gap_size_percent <= self.gap_thresholds["max_tradeable_gap"] and
            gap_data.volume_ratio >= self.volume_thresholds["min_volume_ratio"]
        )
    
    async def _gather_gap_context(self, symbol: str, gap_data: GapData) -> Optional[GapContext]:
        """Gather additional context for gap analysis."""
        try:
            # Get technical indicators
            technical_data = await self._get_technical_indicators(symbol)
            
            # Get market context
            market_context = await self._get_market_context(symbol)
            
            # Get fundamental context
            fundamental_context = await self._get_fundamental_context(symbol)
            
            return GapContext(
                symbol=symbol,
                average_true_range=technical_data.get("atr", gap_data.gap_size_dollars),
                recent_volatility=technical_data.get("volatility", 0.02),
                support_level=technical_data.get("support"),
                resistance_level=technical_data.get("resistance"),
                sector_performance=market_context.get("sector_performance", 0.0),
                market_performance=market_context.get("market_performance", 0.0),
                earnings_proximity_days=fundamental_context.get("earnings_days"),
                has_recent_news=fundamental_context.get("has_news", False),
                institutional_flow=market_context.get("institutional_flow", 0.0),
                options_activity=market_context.get("options", {})
            )
            
        except Exception as e:
            logger.error(f"Failed to gather context for {symbol}: {e}")
            return None
    
    async def _get_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Get technical analysis indicators."""
        try:
            response = await self.client.get(
                f"{self.analysis_service_url}/technical/{symbol}"
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return {}
    
    async def _get_market_context(self, symbol: str) -> Dict[str, Any]:
        """Get market and sector context."""
        try:
            response = await self.client.get(
                f"{self.analysis_service_url}/market-context/{symbol}"
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return {}
    
    async def _get_fundamental_context(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental context like earnings proximity."""
        try:
            response = await self.client.get(
                f"http://localhost:8010/events",
                params={"symbol": symbol, "limit": 5}
            )
            response.raise_for_status()
            
            events = response.json()
            context = {"has_news": len(events) > 0}
            
            # Check for upcoming earnings
            now = datetime.utcnow()
            for event in events:
                event_date = datetime.fromisoformat(event["scheduled_at"].replace("Z", "+00:00"))
                days_to_event = (event_date - now).days
                
                if event.get("category") == "earnings" and 0 <= days_to_event <= 30:
                    context["earnings_days"] = days_to_event
                    break
            
            return context
            
        except Exception:
            return {}
    
    def _determine_gap_strategy(self, gap_data: GapData, context: GapContext) -> GapStrategy:
        """Determine whether to play continuation or fade."""
        continuation_score = 0.0
        fade_score = 0.0
        
        # 1. Gap size analysis
        if gap_data.gap_size_percent < self.gap_thresholds["large_gap_percent"]:
            # Small gaps often continue
            continuation_score += 0.3
        elif gap_data.gap_size_percent > self.gap_thresholds["huge_gap_percent"]:
            # Huge gaps often fade
            fade_score += 0.4
        else:
            # Medium gaps - check other factors
            continuation_score += 0.1
            fade_score += 0.1
        
        # 2. Volume analysis
        if gap_data.volume_ratio > self.volume_thresholds["extreme_volume_ratio"]:
            # Extreme volume suggests more movement to come
            continuation_score += 0.3
        elif gap_data.volume_ratio > self.volume_thresholds["high_volume_ratio"]:
            continuation_score += 0.2
        else:
            fade_score += 0.1
        
        # 3. Market context
        if context.sector_performance > 0.02:  # Strong sector
            continuation_score += 0.2
        elif context.sector_performance < -0.02:  # Weak sector
            if gap_data.gap_direction == GapDirection.GAP_DOWN:
                continuation_score += 0.2
            else:
                fade_score += 0.2
        
        # 4. Technical levels
        if gap_data.gap_direction == GapDirection.GAP_UP:
            if context.resistance_level and gap_data.current_price > context.resistance_level:
                fade_score += 0.3  # Broke resistance, might retrace
            else:
                continuation_score += 0.2
        else:  # GAP_DOWN
            if context.support_level and gap_data.current_price < context.support_level:
                fade_score += 0.3  # Broke support, might bounce
            else:
                continuation_score += 0.2
        
        # 5. Volatility context
        volatility_ratio = gap_data.gap_size_percent / context.recent_volatility
        if volatility_ratio > 3.0:  # Gap much larger than normal volatility
            fade_score += 0.3
        elif volatility_ratio < 1.5:  # Gap smaller than normal volatility
            continuation_score += 0.2
        
        # 6. Earnings proximity
        if context.earnings_proximity_days is not None and context.earnings_proximity_days <= 3:
            continuation_score += 0.2  # Earnings gaps often continue
        
        # 7. Session timing
        if gap_data.session == MarketSession.PRE_MARKET:
            fade_score += 0.1  # Pre-market moves often fade
        elif gap_data.session == MarketSession.REGULAR_HOURS:
            continuation_score += 0.2  # Regular hours gaps more reliable
        
        # Determine strategy
        if continuation_score > fade_score + 0.2:
            return GapStrategy.CONTINUATION
        elif fade_score > continuation_score + 0.2:
            return GapStrategy.FADE
        else:
            return GapStrategy.HOLD
    
    def _calculate_gap_signal(self, gap_data: GapData, context: GapContext, strategy: GapStrategy) -> GapSignal:
        """Calculate specific signal parameters."""
        params = self.strategy_params[strategy.value]
        
        # Calculate entry price (current price with small buffer)
        entry_price = gap_data.current_price
        
        # Calculate target and stop based on strategy
        if strategy == GapStrategy.CONTINUATION:
            # Target: Further in gap direction
            if gap_data.gap_direction == GapDirection.GAP_UP:
                target_price = entry_price + (gap_data.gap_size_dollars * params["target_ratio"])
                stop_loss_price = gap_data.previous_close + (gap_data.gap_size_dollars * params["stop_ratio"])
            else:
                target_price = entry_price - (gap_data.gap_size_dollars * params["target_ratio"])
                stop_loss_price = gap_data.previous_close - (gap_data.gap_size_dollars * params["stop_ratio"])
        
        else:  # FADE
            # Target: Back toward previous close
            if gap_data.gap_direction == GapDirection.GAP_UP:
                target_price = gap_data.previous_close + (gap_data.gap_size_dollars * (1 - params["target_ratio"]))
                stop_loss_price = entry_price + (gap_data.gap_size_dollars * params["stop_ratio"])
            else:
                target_price = gap_data.previous_close - (gap_data.gap_size_dollars * (1 - params["target_ratio"]))
                stop_loss_price = entry_price - (gap_data.gap_size_dollars * params["stop_ratio"])
        
        # Calculate confidence
        confidence = self._calculate_gap_confidence(gap_data, context, strategy)
        
        # Calculate position size based on confidence and volatility
        position_size_pct = min(0.05, confidence * 0.08)  # Max 5% position
        
        # Generate reasoning
        reasoning = self._generate_gap_reasoning(gap_data, context, strategy)
        
        now = datetime.utcnow()
        expires_at = now + timedelta(minutes=params["hold_minutes"])
        
        return GapSignal(
            symbol=gap_data.symbol,
            strategy=strategy,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss_price=stop_loss_price,
            position_size_pct=position_size_pct,
            confidence=confidence,
            holding_period_minutes=params["hold_minutes"],
            gap_data=gap_data,
            context=context,
            reasoning=reasoning,
            generated_at=now,
            expires_at=expires_at
        )
    
    def _calculate_gap_confidence(self, gap_data: GapData, context: GapContext, strategy: GapStrategy) -> float:
        """Calculate confidence score for the gap signal."""
        base_confidence = 0.5
        
        # Volume factor
        if gap_data.volume_ratio > self.volume_thresholds["high_volume_ratio"]:
            base_confidence += 0.2
        elif gap_data.volume_ratio < self.volume_thresholds["min_volume_ratio"]:
            base_confidence -= 0.1
        
        # Gap size factor
        gap_vs_volatility = gap_data.gap_size_percent / context.recent_volatility
        if 1.5 <= gap_vs_volatility <= 3.0:  # Optimal range
            base_confidence += 0.15
        elif gap_vs_volatility > 5.0:  # Too large
            base_confidence -= 0.2
        
        # Market alignment
        if abs(context.sector_performance) > 0.02:
            base_confidence += 0.1
        
        # Technical level alignment
        if strategy == GapStrategy.FADE:
            if gap_data.gap_direction == GapDirection.GAP_UP and context.resistance_level:
                if gap_data.current_price > context.resistance_level:
                    base_confidence += 0.15
            elif gap_data.gap_direction == GapDirection.GAP_DOWN and context.support_level:
                if gap_data.current_price < context.support_level:
                    base_confidence += 0.15
        
        return max(0.0, min(1.0, base_confidence))
    
    def _generate_gap_reasoning(self, gap_data: GapData, context: GapContext, strategy: GapStrategy) -> str:
        """Generate human-readable reasoning for the gap trade."""
        reasoning_parts = []
        
        # Gap description
        reasoning_parts.append(
            f"{gap_data.gap_size_percent:.1%} {gap_data.gap_direction.value} "
            f"on {gap_data.volume_ratio:.1f}x volume"
        )
        
        # Strategy rationale
        if strategy == GapStrategy.CONTINUATION:
            reasoning_parts.append("Playing for continuation based on")
        else:
            reasoning_parts.append("Playing for fade/reversion based on")
        
        # Key factors
        factors = []
        
        if gap_data.volume_ratio > self.volume_thresholds["high_volume_ratio"]:
            factors.append("high volume")
        
        if context.earnings_proximity_days and context.earnings_proximity_days <= 3:
            factors.append("earnings proximity")
        
        if abs(context.sector_performance) > 0.02:
            factors.append("sector alignment")
        
        if gap_data.gap_size_percent > self.gap_thresholds["large_gap_percent"]:
            factors.append("large gap size")
        
        if factors:
            reasoning_parts.append(", ".join(factors))
        
        return ". ".join(reasoning_parts) + "."


async def test_gap_trading_engine():
    """Test the gap trading engine functionality."""
    async with GapTradingEngine() as engine:
        # Test gap analysis
        signal = await engine.analyze_gap_opportunity("AAPL")
        
        if signal:
            print(f"Generated gap signal for {signal.symbol}")
            print(f"Strategy: {signal.strategy.value}")
            print(f"Entry: ${signal.entry_price:.2f}")
            print(f"Target: ${signal.target_price:.2f}")
            print(f"Stop: ${signal.stop_loss_price:.2f}")
            print(f"Confidence: {signal.confidence:.2%}")
            print(f"Position size: {signal.position_size_pct:.2%}")
            print(f"Reasoning: {signal.reasoning}")
        else:
            print("No gap trading signal generated")


if __name__ == "__main__":
    asyncio.run(test_gap_trading_engine())