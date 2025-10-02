"""
Event-Aware Stop Loss System

Implements dynamic stop losses that tighten around known events
(earnings, Fed announcements, etc.) and revert to normal levels afterward.

Acceptance: Lower tail losses around events with no IR degradation
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of market events requiring tighter stops"""
    EARNINGS = "earnings"
    FED_ANNOUNCEMENT = "fed_announcement"
    FOMC_MEETING = "fomc_meeting"
    ECONOMIC_DATA = "economic_data"
    DIVIDEND_EXDATE = "dividend_exdate"
    SPLIT = "split"
    MERGER_ACQUISITION = "merger_acquisition"
    REGULATORY = "regulatory"


class StopRegime(Enum):
    """Stop loss regime states"""
    NORMAL = "normal"
    EVENT_TIGHTENED = "event_tightened"
    HALTED = "halted"


@dataclass
class EventWindow:
    """Event window configuration"""
    event_type: EventType
    event_time: datetime
    pre_event_hours: float = 24.0  # Hours before event
    post_event_hours: float = 4.0  # Hours after event

    def is_inside_window(self, current_time: datetime) -> bool:
        """Check if current time is inside event window"""
        pre_start = self.event_time - timedelta(hours=self.pre_event_hours)
        post_end = self.event_time + timedelta(hours=self.post_event_hours)
        return pre_start <= current_time <= post_end

    def get_distance_to_event(self, current_time: datetime) -> float:
        """Get hours until/since event (negative if before, positive if after)"""
        delta = current_time - self.event_time
        return delta.total_seconds() / 3600.0


@dataclass
class StopLossConfig:
    """Stop loss configuration"""
    # Normal regime stops
    normal_stop_loss_pct: float = 0.02  # 2% stop loss
    normal_take_profit_pct: float = 0.05  # 5% take profit

    # Event regime stops (tighter)
    event_stop_loss_pct: float = 0.01  # 1% stop loss (tighter)
    event_take_profit_pct: float = 0.03  # 3% take profit (lower)

    # Trailing stop configuration
    enable_trailing_stop: bool = True
    trailing_stop_activation_pct: float = 0.02  # Activate after 2% gain
    trailing_stop_distance_pct: float = 0.01  # Trail by 1%

    # Event window defaults
    default_pre_event_hours: float = 24.0
    default_post_event_hours: float = 4.0

    # Event-specific adjustments
    event_type_multipliers: Dict[EventType, float] = None

    def __post_init__(self):
        """Initialize event type multipliers"""
        if self.event_type_multipliers is None:
            self.event_type_multipliers = {
                EventType.EARNINGS: 0.5,  # Very tight (50% of normal)
                EventType.FED_ANNOUNCEMENT: 0.6,  # Tight (60%)
                EventType.FOMC_MEETING: 0.6,
                EventType.ECONOMIC_DATA: 0.7,
                EventType.DIVIDEND_EXDATE: 0.9,  # Minimal tightening
                EventType.SPLIT: 0.8,
                EventType.MERGER_ACQUISITION: 0.5,
                EventType.REGULATORY: 0.6
            }

    def get_event_stop_loss(self, event_type: EventType) -> float:
        """Get stop loss percentage for specific event type"""
        multiplier = self.event_type_multipliers.get(event_type, 0.7)
        return self.normal_stop_loss_pct * multiplier


@dataclass
class StopLossState:
    """Current stop loss state for a position"""
    symbol: str
    entry_price: float
    entry_time: datetime
    current_price: float
    current_time: datetime

    # Stop levels
    stop_loss_price: float
    take_profit_price: float
    trailing_stop_price: Optional[float] = None

    # Regime
    regime: StopRegime = StopRegime.NORMAL
    active_event: Optional[EventWindow] = None

    # High water mark for trailing stops
    highest_price: Optional[float] = None

    # Exit tracking
    is_stopped_out: bool = False
    exit_reason: Optional[str] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None


class EventAwareStopLoss:
    """
    Event-Aware Stop Loss Manager

    Dynamically adjusts stop losses based on upcoming/ongoing events.
    Tightens stops during event windows to reduce tail risk.
    """

    def __init__(self, config: Optional[StopLossConfig] = None):
        """
        Initialize event-aware stop loss system.

        Args:
            config: Stop loss configuration (uses defaults if None)
        """
        self.config = config or StopLossConfig()
        self.event_calendar: Dict[str, List[EventWindow]] = {}  # symbol -> events
        self.position_states: Dict[str, StopLossState] = {}  # symbol -> state

    def register_event(self,
                      symbol: str,
                      event_type: EventType,
                      event_time: datetime,
                      pre_event_hours: Optional[float] = None,
                      post_event_hours: Optional[float] = None) -> EventWindow:
        """
        Register a known event for a symbol.

        Args:
            symbol: Stock symbol
            event_type: Type of event
            event_time: Event datetime
            pre_event_hours: Hours before event to tighten (default from config)
            post_event_hours: Hours after event to tighten (default from config)

        Returns:
            EventWindow object
        """
        event = EventWindow(
            event_type=event_type,
            event_time=event_time,
            pre_event_hours=pre_event_hours or self.config.default_pre_event_hours,
            post_event_hours=post_event_hours or self.config.default_post_event_hours
        )

        if symbol not in self.event_calendar:
            self.event_calendar[symbol] = []

        self.event_calendar[symbol].append(event)
        logger.info(
            f"Registered {event_type.value} event for {symbol} at {event_time}"
        )

        return event

    def get_active_event(self, symbol: str, current_time: datetime) -> Optional[EventWindow]:
        """
        Get active event window for symbol at current time.

        Args:
            symbol: Stock symbol
            current_time: Current datetime

        Returns:
            EventWindow if inside event window, None otherwise
        """
        if symbol not in self.event_calendar:
            return None

        for event in self.event_calendar[symbol]:
            if event.is_inside_window(current_time):
                return event

        return None

    def initialize_position(self,
                          symbol: str,
                          entry_price: float,
                          entry_time: datetime,
                          position_side: str = "long") -> StopLossState:
        """
        Initialize stop loss state for a new position.

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            entry_time: Entry datetime
            position_side: "long" or "short"

        Returns:
            StopLossState object
        """
        # Check if we're in an event window
        active_event = self.get_active_event(symbol, entry_time)

        if active_event:
            # Use tightened stops for event window
            regime = StopRegime.EVENT_TIGHTENED
            stop_loss_pct = self.config.get_event_stop_loss(active_event.event_type)
            take_profit_pct = self.config.event_take_profit_pct

            logger.info(
                f"Position entered during {active_event.event_type.value} event window. "
                f"Using tightened stops: {stop_loss_pct*100:.1f}%"
            )
        else:
            # Use normal stops
            regime = StopRegime.NORMAL
            stop_loss_pct = self.config.normal_stop_loss_pct
            take_profit_pct = self.config.normal_take_profit_pct

        # Calculate stop levels based on position side
        if position_side == "long":
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct)
        else:  # short
            stop_loss_price = entry_price * (1 + stop_loss_pct)
            take_profit_price = entry_price * (1 - take_profit_pct)

        state = StopLossState(
            symbol=symbol,
            entry_price=entry_price,
            entry_time=entry_time,
            current_price=entry_price,
            current_time=entry_time,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            regime=regime,
            active_event=active_event,
            highest_price=entry_price
        )

        self.position_states[symbol] = state

        return state

    def update_position(self,
                       symbol: str,
                       current_price: float,
                       current_time: datetime,
                       position_side: str = "long") -> Tuple[StopLossState, bool]:
        """
        Update position state and check for stop triggers.

        Args:
            symbol: Stock symbol
            current_price: Current market price
            current_time: Current datetime
            position_side: "long" or "short"

        Returns:
            Tuple of (updated_state, should_exit)
        """
        if symbol not in self.position_states:
            raise ValueError(f"No position state found for {symbol}")

        state = self.position_states[symbol]
        state.current_price = current_price
        state.current_time = current_time

        # Update high water mark
        if position_side == "long":
            if state.highest_price is None or current_price > state.highest_price:
                state.highest_price = current_price
        else:  # short
            if state.highest_price is None or current_price < state.highest_price:
                state.highest_price = current_price

        # Check for regime changes (event windows)
        self._update_regime(state, current_time, position_side)

        # Update trailing stop if enabled
        if self.config.enable_trailing_stop:
            self._update_trailing_stop(state, position_side)

        # Check for stop triggers
        should_exit, exit_reason = self._check_stop_triggers(state, position_side)

        if should_exit:
            state.is_stopped_out = True
            state.exit_reason = exit_reason
            state.exit_price = current_price
            state.exit_time = current_time

            logger.info(
                f"Stop triggered for {symbol}: {exit_reason} at {current_price:.2f}"
            )

        return state, should_exit

    def _update_regime(self,
                      state: StopLossState,
                      current_time: datetime,
                      position_side: str):
        """Update stop loss regime based on event windows"""
        active_event = self.get_active_event(state.symbol, current_time)

        # Check if regime changed
        old_regime = state.regime

        if active_event and state.regime == StopRegime.NORMAL:
            # Entering event window - tighten stops
            state.regime = StopRegime.EVENT_TIGHTENED
            state.active_event = active_event

            # Recalculate stop levels
            stop_loss_pct = self.config.get_event_stop_loss(active_event.event_type)

            if position_side == "long":
                new_stop_loss = state.entry_price * (1 - stop_loss_pct)
                # Only tighten, never loosen
                state.stop_loss_price = max(state.stop_loss_price, new_stop_loss)
            else:  # short
                new_stop_loss = state.entry_price * (1 + stop_loss_pct)
                state.stop_loss_price = min(state.stop_loss_price, new_stop_loss)

            logger.info(
                f"Tightening stops for {state.symbol} due to {active_event.event_type.value}. "
                f"New stop: {state.stop_loss_price:.2f}"
            )

        elif not active_event and state.regime == StopRegime.EVENT_TIGHTENED:
            # Exiting event window - revert to normal
            state.regime = StopRegime.NORMAL
            state.active_event = None

            # Revert to normal stop levels (but don't loosen existing stop)
            normal_stop_pct = self.config.normal_stop_loss_pct

            if position_side == "long":
                normal_stop = state.entry_price * (1 - normal_stop_pct)
                state.stop_loss_price = max(state.stop_loss_price, normal_stop)
            else:  # short
                normal_stop = state.entry_price * (1 + normal_stop_pct)
                state.stop_loss_price = min(state.stop_loss_price, normal_stop)

            logger.info(
                f"Reverting to normal stops for {state.symbol}. "
                f"Stop: {state.stop_loss_price:.2f}"
            )

    def _update_trailing_stop(self, state: StopLossState, position_side: str):
        """Update trailing stop based on price movement"""
        if not self.config.enable_trailing_stop:
            return

        # Check if trailing stop should be activated
        activation_threshold = state.entry_price * (
            1 + self.config.trailing_stop_activation_pct
        )

        if position_side == "long":
            # Activate trailing stop if price above threshold
            if state.current_price >= activation_threshold:
                trailing_stop = state.highest_price * (
                    1 - self.config.trailing_stop_distance_pct
                )

                # Trail the stop up, never down
                if state.trailing_stop_price is None:
                    state.trailing_stop_price = trailing_stop
                else:
                    state.trailing_stop_price = max(state.trailing_stop_price, trailing_stop)

        else:  # short
            activation_threshold = state.entry_price * (
                1 - self.config.trailing_stop_activation_pct
            )

            if state.current_price <= activation_threshold:
                trailing_stop = state.highest_price * (
                    1 + self.config.trailing_stop_distance_pct
                )

                if state.trailing_stop_price is None:
                    state.trailing_stop_price = trailing_stop
                else:
                    state.trailing_stop_price = min(state.trailing_stop_price, trailing_stop)

    def _check_stop_triggers(self,
                            state: StopLossState,
                            position_side: str) -> Tuple[bool, Optional[str]]:
        """Check if any stop has been triggered"""

        if position_side == "long":
            # Check stop loss
            if state.current_price <= state.stop_loss_price:
                return True, f"stop_loss (regime: {state.regime.value})"

            # Check trailing stop
            if state.trailing_stop_price and state.current_price <= state.trailing_stop_price:
                return True, "trailing_stop"

            # Check take profit
            if state.current_price >= state.take_profit_price:
                return True, "take_profit"

        else:  # short
            # Check stop loss
            if state.current_price >= state.stop_loss_price:
                return True, f"stop_loss (regime: {state.regime.value})"

            # Check trailing stop
            if state.trailing_stop_price and state.current_price >= state.trailing_stop_price:
                return True, "trailing_stop"

            # Check take profit
            if state.current_price <= state.take_profit_price:
                return True, "take_profit"

        return False, None

    def close_position(self, symbol: str):
        """Close position and remove state"""
        if symbol in self.position_states:
            del self.position_states[symbol]

    def get_position_state(self, symbol: str) -> Optional[StopLossState]:
        """Get current position state"""
        return self.position_states.get(symbol)

    def get_stop_regime_stats(self) -> Dict:
        """Get statistics on stop regimes"""
        total_positions = len(self.position_states)

        if total_positions == 0:
            return {
                "total_positions": 0,
                "normal_regime": 0,
                "event_tightened": 0,
                "halted": 0
            }

        stats = {
            "total_positions": total_positions,
            "normal_regime": sum(1 for s in self.position_states.values() if s.regime == StopRegime.NORMAL),
            "event_tightened": sum(1 for s in self.position_states.values() if s.regime == StopRegime.EVENT_TIGHTENED),
            "halted": sum(1 for s in self.position_states.values() if s.regime == StopRegime.HALTED),
            "trailing_stops_active": sum(1 for s in self.position_states.values() if s.trailing_stop_price is not None)
        }

        return stats


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize stop loss system
    config = StopLossConfig(
        normal_stop_loss_pct=0.02,
        event_stop_loss_pct=0.01,
        enable_trailing_stop=True
    )

    stop_manager = EventAwareStopLoss(config)

    # Register earnings event
    earnings_time = datetime(2025, 10, 15, 16, 0, 0)  # After market close
    stop_manager.register_event(
        symbol="AAPL",
        event_type=EventType.EARNINGS,
        event_time=earnings_time,
        pre_event_hours=24,
        post_event_hours=4
    )

    # Enter position before earnings
    entry_time = datetime(2025, 10, 14, 14, 0, 0)
    entry_price = 150.0

    state = stop_manager.initialize_position(
        symbol="AAPL",
        entry_price=entry_price,
        entry_time=entry_time,
        position_side="long"
    )

    print(f"Position initialized:")
    print(f"  Entry: ${entry_price:.2f}")
    print(f"  Stop loss: ${state.stop_loss_price:.2f}")
    print(f"  Take profit: ${state.take_profit_price:.2f}")
    print(f"  Regime: {state.regime.value}")

    # Update position (price moves up)
    update_time = datetime(2025, 10, 14, 15, 0, 0)
    current_price = 152.0

    state, should_exit = stop_manager.update_position(
        symbol="AAPL",
        current_price=current_price,
        current_time=update_time,
        position_side="long"
    )

    print(f"\nPosition updated:")
    print(f"  Current price: ${current_price:.2f}")
    print(f"  Should exit: {should_exit}")
    print(f"  Trailing stop: ${state.trailing_stop_price:.2f if state.trailing_stop_price else 'None'}")

    # Get regime stats
    stats = stop_manager.get_stop_regime_stats()
    print(f"\nRegime stats: {stats}")
