"""
Event-Aware Backtesting Engine

Advanced backtesting system specifically designed for event-driven trading strategies
with sophisticated event detection, stop-loss management, and performance attribution.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import pandas as pd
import numpy as np
from decimal import Decimal
import asyncio

class EventType(Enum):
    """Event types for backtesting"""
    EARNINGS = "earnings"
    FDA_APPROVAL = "fda_approval"
    MERGER_ACQUISITION = "merger_acquisition"
    PRODUCT_LAUNCH = "product_launch"
    REGULATORY = "regulatory"
    ANALYST_ACTION = "analyst_action"
    GUIDANCE = "guidance"
    SPINOFF = "spinoff"
    DIVIDEND = "dividend"
    CATALYST = "catalyst"

class TradeDirection(Enum):
    """Trade direction"""
    LONG = "long"
    SHORT = "short"

class ExitReason(Enum):
    """Reasons for trade exit"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TIME_DECAY = "time_decay"
    EVENT_RESOLVED = "event_resolved"
    REGIME_CHANGE = "regime_change"
    MANUAL_EXIT = "manual_exit"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY_STOP = "volatility_stop"

class StopLossType(Enum):
    """Stop loss types"""
    FIXED_PERCENTAGE = "fixed_percentage"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    TRAILING_STOP = "trailing_stop"
    TIME_BASED = "time_based"
    ATR_BASED = "atr_based"
    EVENT_SPECIFIC = "event_specific"
    REGIME_ADJUSTED = "regime_adjusted"

@dataclass
class BacktestEvent:
    """Event data for backtesting"""
    timestamp: datetime
    symbol: str
    event_type: EventType
    event_description: str
    surprise_magnitude: float           # Actual vs expected surprise
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    expected_move: Optional[float] = None  # Options-implied expected move
    actual_move_1d: Optional[float] = None
    actual_move_3d: Optional[float] = None
    actual_move_5d: Optional[float] = None
    volume_surge: Optional[float] = None
    analyst_rating: Optional[str] = None
    confidence_score: Optional[float] = None

@dataclass
class StopLossConfig:
    """Stop loss configuration for event trades"""
    stop_type: StopLossType
    base_percentage: float              # Base stop loss percentage
    volatility_multiplier: float        # ATR or volatility adjustment
    max_stop_loss: float                # Maximum stop loss allowed
    min_stop_loss: float                # Minimum stop loss allowed
    trailing_threshold: Optional[float] = None  # When to activate trailing
    time_decay_factor: Optional[float] = None   # Time-based stop tightening
    event_specific_adjustments: Dict[EventType, float] = field(default_factory=dict)

@dataclass
class EventTrade:
    """Individual event trade record"""
    trade_id: str
    symbol: str
    event: BacktestEvent
    entry_timestamp: datetime
    entry_price: float
    direction: TradeDirection
    position_size: float
    stop_loss_price: float
    take_profit_price: Optional[float]
    
    # Exit data
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    
    # Performance metrics
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    holding_period_hours: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    
    # Risk metrics
    realized_volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    
    # Event-specific metrics
    time_to_event_resolution: Optional[float] = None
    event_surprise_accuracy: Optional[float] = None
    regime_at_entry: Optional[str] = None
    regime_at_exit: Optional[str] = None

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    
    # Event-specific metrics
    event_type_performance: Dict[EventType, Dict[str, float]]
    surprise_magnitude_performance: Dict[str, float]
    market_cap_performance: Dict[str, float]
    sector_performance: Dict[str, float]
    
    # Risk metrics
    var_95: float                       # Value at Risk 95%
    expected_shortfall: float           # Conditional VaR
    maximum_consecutive_losses: int
    average_holding_period: float
    
    # Stop loss analysis
    stop_loss_effectiveness: Dict[StopLossType, Dict[str, float]]
    optimal_stop_loss_levels: Dict[EventType, float]
    
    # Regime analysis
    regime_performance: Dict[str, Dict[str, float]]
    
    # All trades
    trades: List[EventTrade]

class EventAwareBacktestEngine:
    """Advanced event-aware backtesting engine"""
    
    def __init__(self):
        # Default stop loss configurations for different event types
        self.default_stop_configs = {
            EventType.EARNINGS: StopLossConfig(
                stop_type=StopLossType.VOLATILITY_ADJUSTED,
                base_percentage=0.08,           # 8% base stop
                volatility_multiplier=1.5,     # 1.5x ATR
                max_stop_loss=0.15,            # Max 15% stop
                min_stop_loss=0.03,            # Min 3% stop
                time_decay_factor=0.02         # 2% tighter per day
            ),
            EventType.FDA_APPROVAL: StopLossConfig(
                stop_type=StopLossType.EVENT_SPECIFIC,
                base_percentage=0.12,           # 12% base stop
                volatility_multiplier=2.0,     # 2x volatility
                max_stop_loss=0.25,            # Max 25% stop
                min_stop_loss=0.05,            # Min 5% stop
                trailing_threshold=0.15        # Trail after 15% gain
            ),
            EventType.MERGER_ACQUISITION: StopLossConfig(
                stop_type=StopLossType.FIXED_PERCENTAGE,
                base_percentage=0.05,           # 5% tight stop
                volatility_multiplier=1.0,
                max_stop_loss=0.08,            # Max 8% stop
                min_stop_loss=0.02,            # Min 2% stop
                time_decay_factor=0.01         # 1% tighter per day
            )
        }
        
        # Default configuration
        self.default_config = StopLossConfig(
            stop_type=StopLossType.VOLATILITY_ADJUSTED,
            base_percentage=0.10,
            volatility_multiplier=1.5,
            max_stop_loss=0.20,
            min_stop_loss=0.04,
            time_decay_factor=0.015
        )
        
    async def run_backtest(
        self,
        events: List[BacktestEvent],
        price_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        position_sizing: str = 'equal_weight',
        max_positions: int = 10,
        stop_loss_config: Optional[Dict[EventType, StopLossConfig]] = None,
        regime_data: Optional[pd.DataFrame] = None,
        include_costs: bool = True,
        commission_rate: float = 0.001,
        slippage_bps: float = 5.0
    ) -> BacktestResults:
        """
        Run comprehensive event-aware backtest
        """
        # Use provided or default stop loss configurations
        stop_configs = stop_loss_config or self.default_stop_configs
        
        # Initialize portfolio tracking
        portfolio = {
            'cash': initial_capital,
            'positions': {},
            'trades': [],
            'equity_curve': [],
            'drawdown_curve': []
        }
        
        # Filter events within date range
        filtered_events = [
            event for event in events
            if start_date <= event.timestamp <= end_date
        ]
        
        print(f"Running backtest on {len(filtered_events)} events from {start_date.date()} to {end_date.date()}")
        
        # Sort events by timestamp
        filtered_events.sort(key=lambda x: x.timestamp)
        
        # Process each event
        for event in filtered_events:
            await self._process_event(
                event=event,
                portfolio=portfolio,
                price_data=price_data,
                stop_configs=stop_configs,
                position_sizing=position_sizing,
                max_positions=max_positions,
                regime_data=regime_data,
                include_costs=include_costs,
                commission_rate=commission_rate,
                slippage_bps=slippage_bps
            )
            
            # Update portfolio value
            self._update_portfolio_metrics(portfolio, price_data, event.timestamp)
        
        # Close any remaining positions at end date
        await self._close_all_positions(portfolio, price_data, end_date)
        
        # Calculate comprehensive results
        results = self._calculate_results(
            portfolio=portfolio,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        return results
    
    async def _process_event(
        self,
        event: BacktestEvent,
        portfolio: Dict,
        price_data: Dict[str, pd.DataFrame],
        stop_configs: Dict[EventType, StopLossConfig],
        position_sizing: str,
        max_positions: int,
        regime_data: Optional[pd.DataFrame],
        include_costs: bool,
        commission_rate: float,
        slippage_bps: float
    ):
        """Process individual event for potential trading"""
        
        # Check if we have price data for this symbol
        if event.symbol not in price_data:
            return
        
        symbol_data = price_data[event.symbol]
        
        # Find the closest price data to event timestamp
        event_date = event.timestamp.date()
        available_dates = symbol_data.index.date
        
        # Find next trading day after event
        future_dates = [d for d in available_dates if d > event_date]
        if not future_dates:
            return
        
        entry_date = min(future_dates)
        entry_row = symbol_data[symbol_data.index.date == entry_date].iloc[0]
        
        # Determine if we should enter the trade
        should_enter = self._should_enter_trade(event, portfolio, max_positions)
        if not should_enter:
            return
        
        # Calculate position size
        position_size = self._calculate_position_size(
            event=event,
            portfolio=portfolio,
            entry_price=entry_row['open'],
            position_sizing=position_sizing
        )
        
        if position_size <= 0:
            return
        
        # Calculate stop loss
        stop_config = stop_configs.get(event.event_type, self.default_config)
        stop_loss_price = self._calculate_stop_loss(
            event=event,
            entry_price=entry_row['open'],
            direction=TradeDirection.LONG,  # Assume long for now
            config=stop_config,
            symbol_data=symbol_data,
            entry_date=entry_date
        )
        
        # Calculate take profit (optional)
        take_profit_price = self._calculate_take_profit(
            event=event,
            entry_price=entry_row['open'],
            direction=TradeDirection.LONG
        )
        
        # Create trade
        trade = EventTrade(
            trade_id=f"{event.symbol}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}",
            symbol=event.symbol,
            event=event,
            entry_timestamp=datetime.combine(entry_date, datetime.min.time()),
            entry_price=entry_row['open'],
            direction=TradeDirection.LONG,
            position_size=position_size,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            regime_at_entry=self._get_regime_at_date(regime_data, entry_date) if regime_data is not None else None
        )
        
        # Apply costs
        if include_costs:
            entry_cost = trade.entry_price * position_size * commission_rate
            slippage_cost = trade.entry_price * position_size * (slippage_bps / 10000)
            total_entry_cost = entry_cost + slippage_cost
            trade.entry_price += total_entry_cost / position_size
        
        # Execute trade
        portfolio['cash'] -= trade.entry_price * position_size
        portfolio['positions'][trade.trade_id] = trade
        
        # Monitor trade for exit
        await self._monitor_trade_exit(
            trade=trade,
            symbol_data=symbol_data,
            portfolio=portfolio,
            include_costs=include_costs,
            commission_rate=commission_rate,
            slippage_bps=slippage_bps,
            regime_data=regime_data
        )
    
    async def _monitor_trade_exit(
        self,
        trade: EventTrade,
        symbol_data: pd.DataFrame,
        portfolio: Dict,
        include_costs: bool,
        commission_rate: float,
        slippage_bps: float,
        regime_data: Optional[pd.DataFrame]
    ):
        """Monitor trade for exit conditions"""
        
        entry_date = trade.entry_timestamp.date()
        
        # Get future price data
        future_data = symbol_data[symbol_data.index.date > entry_date].copy()
        if future_data.empty:
            return
        
        # Track maximum excursions
        max_favorable = 0.0
        max_adverse = 0.0
        
        # Monitor each day after entry
        for date, row in future_data.iterrows():
            current_price = row['close']
            
            # Calculate unrealized PnL
            unrealized_pnl = (current_price - trade.entry_price) * trade.position_size
            unrealized_pnl_pct = (current_price - trade.entry_price) / trade.entry_price
            
            # Update excursions
            if unrealized_pnl > max_favorable:
                max_favorable = unrealized_pnl
            if unrealized_pnl < max_adverse:
                max_adverse = unrealized_pnl
            
            # Check exit conditions
            exit_reason = None
            exit_price = current_price
            
            # Stop loss check
            if current_price <= trade.stop_loss_price:
                exit_reason = ExitReason.STOP_LOSS
                exit_price = trade.stop_loss_price
            
            # Take profit check
            elif trade.take_profit_price and current_price >= trade.take_profit_price:
                exit_reason = ExitReason.TAKE_PROFIT
                exit_price = trade.take_profit_price
            
            # Time decay check (exit after 5 days if no clear direction)
            elif (date.date() - entry_date).days >= 5 and abs(unrealized_pnl_pct) < 0.02:
                exit_reason = ExitReason.TIME_DECAY
            
            # Volatility stop (exit if daily move > 15%)
            elif abs(row['high'] - row['low']) / row['open'] > 0.15:
                exit_reason = ExitReason.VOLATILITY_STOP
            
            # Maximum holding period (15 days for most events)
            elif (date.date() - entry_date).days >= 15:
                exit_reason = ExitReason.TIME_DECAY
            
            # Exit the trade if any condition is met
            if exit_reason:
                await self._exit_trade(
                    trade=trade,
                    exit_date=date.date(),
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    portfolio=portfolio,
                    include_costs=include_costs,
                    commission_rate=commission_rate,
                    slippage_bps=slippage_bps,
                    max_favorable=max_favorable,
                    max_adverse=max_adverse,
                    regime_data=regime_data
                )
                break
    
    async def _exit_trade(
        self,
        trade: EventTrade,
        exit_date,
        exit_price: float,
        exit_reason: ExitReason,
        portfolio: Dict,
        include_costs: bool,
        commission_rate: float,
        slippage_bps: float,
        max_favorable: float,
        max_adverse: float,
        regime_data: Optional[pd.DataFrame]
    ):
        """Execute trade exit"""
        
        # Apply exit costs
        if include_costs:
            exit_cost = exit_price * trade.position_size * commission_rate
            slippage_cost = exit_price * trade.position_size * (slippage_bps / 10000)
            total_exit_cost = exit_cost + slippage_cost
            exit_price -= total_exit_cost / trade.position_size
        
        # Update trade record
        trade.exit_timestamp = datetime.combine(exit_date, datetime.min.time())
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.pnl = (exit_price - trade.entry_price) * trade.position_size
        trade.pnl_percentage = (exit_price - trade.entry_price) / trade.entry_price
        trade.holding_period_hours = (trade.exit_timestamp - trade.entry_timestamp).total_seconds() / 3600
        trade.max_favorable_excursion = max_favorable
        trade.max_adverse_excursion = max_adverse
        trade.regime_at_exit = self._get_regime_at_date(regime_data, exit_date) if regime_data is not None else None
        
        # Update portfolio
        portfolio['cash'] += exit_price * trade.position_size
        portfolio['trades'].append(trade)
        del portfolio['positions'][trade.trade_id]
    
    def _should_enter_trade(
        self, event: BacktestEvent, portfolio: Dict, max_positions: int
    ) -> bool:
        """Determine if we should enter a trade for this event"""
        
        # Check position limits
        if len(portfolio['positions']) >= max_positions:
            return False
        
        # Check if we have sufficient capital
        if portfolio['cash'] < 1000:  # Minimum $1000 for trade
            return False
        
        # Event-specific filters
        if event.confidence_score and event.confidence_score < 0.6:
            return False
        
        # Surprise magnitude filter
        if abs(event.surprise_magnitude) < 0.05:  # Less than 5% surprise
            return False
        
        return True
    
    def _calculate_position_size(
        self,
        event: BacktestEvent,
        portfolio: Dict,
        entry_price: float,
        position_sizing: str
    ) -> float:
        """Calculate position size for the trade"""
        
        available_capital = portfolio['cash']
        
        if position_sizing == 'equal_weight':
            # Equal weight among all positions
            target_allocation = 0.10  # 10% per position
            dollar_amount = available_capital * target_allocation
            return int(dollar_amount / entry_price)
        
        elif position_sizing == 'kelly':
            # Kelly criterion based on confidence and expected return
            win_prob = event.confidence_score or 0.6
            expected_return = abs(event.surprise_magnitude) * 2  # Assume 2x leverage on surprise
            kelly_fraction = (win_prob * (1 + expected_return) - 1) / expected_return
            kelly_fraction = max(0.02, min(0.15, kelly_fraction))  # Cap between 2% and 15%
            
            dollar_amount = available_capital * kelly_fraction
            return int(dollar_amount / entry_price)
        
        else:  # fixed_amount
            return int(10000 / entry_price)  # Fixed $10,000 per trade
    
    def _calculate_stop_loss(
        self,
        event: BacktestEvent,
        entry_price: float,
        direction: TradeDirection,
        config: StopLossConfig,
        symbol_data: pd.DataFrame,
        entry_date
    ) -> float:
        """Calculate stop loss price based on configuration"""
        
        if config.stop_type == StopLossType.FIXED_PERCENTAGE:
            stop_distance = entry_price * config.base_percentage
            return entry_price - stop_distance if direction == TradeDirection.LONG else entry_price + stop_distance
        
        elif config.stop_type == StopLossType.VOLATILITY_ADJUSTED:
            # Calculate ATR-based stop
            atr = self._calculate_atr(symbol_data, entry_date, period=14)
            stop_distance = atr * config.volatility_multiplier
            
            # Apply base percentage as minimum
            min_stop_distance = entry_price * config.base_percentage
            stop_distance = max(stop_distance, min_stop_distance)
            
            # Apply maximum stop loss limit
            max_stop_distance = entry_price * config.max_stop_loss
            stop_distance = min(stop_distance, max_stop_distance)
            
            return entry_price - stop_distance if direction == TradeDirection.LONG else entry_price + stop_distance
        
        elif config.stop_type == StopLossType.EVENT_SPECIFIC:
            # Event-specific stop loss adjustment
            base_stop = config.base_percentage
            
            # Adjust based on event type
            if event.event_type == EventType.FDA_APPROVAL:
                base_stop *= 1.5  # Wider stops for binary events
            elif event.event_type == EventType.MERGER_ACQUISITION:
                base_stop *= 0.7  # Tighter stops for deal risk
            elif event.event_type == EventType.EARNINGS:
                base_stop *= 1.0  # Standard stops
            
            # Adjust based on surprise magnitude
            surprise_adjustment = min(1.5, 1.0 + abs(event.surprise_magnitude))
            base_stop *= surprise_adjustment
            
            # Apply limits
            base_stop = max(config.min_stop_loss, min(config.max_stop_loss, base_stop))
            
            stop_distance = entry_price * base_stop
            return entry_price - stop_distance if direction == TradeDirection.LONG else entry_price + stop_distance
        
        else:  # Default to fixed percentage
            stop_distance = entry_price * config.base_percentage
            return entry_price - stop_distance if direction == TradeDirection.LONG else entry_price + stop_distance
    
    def _calculate_take_profit(
        self, event: BacktestEvent, entry_price: float, direction: TradeDirection
    ) -> Optional[float]:
        """Calculate take profit level"""
        
        # Event-specific take profit levels
        if event.event_type == EventType.MERGER_ACQUISITION:
            # Conservative take profit for M&A (deal spread capture)
            target_return = 0.05  # 5% target
        elif event.event_type == EventType.FDA_APPROVAL:
            # Larger target for binary events
            target_return = 0.30  # 30% target
        elif event.event_type == EventType.EARNINGS:
            # Moderate target for earnings
            target_return = 0.15  # 15% target
        else:
            # Default target
            target_return = 0.12  # 12% target
        
        # Adjust based on surprise magnitude
        adjusted_target = target_return * (1 + abs(event.surprise_magnitude))
        adjusted_target = min(0.50, adjusted_target)  # Cap at 50%
        
        if direction == TradeDirection.LONG:
            return entry_price * (1 + adjusted_target)
        else:
            return entry_price * (1 - adjusted_target)
    
    def _calculate_atr(self, symbol_data: pd.DataFrame, entry_date, period: int = 14) -> float:
        """Calculate Average True Range"""
        
        # Get data before entry date
        historical_data = symbol_data[symbol_data.index.date < entry_date].tail(period + 1)
        
        if len(historical_data) < 2:
            # Fallback to simple range if insufficient data
            recent_data = symbol_data[symbol_data.index.date <= entry_date].tail(5)
            if not recent_data.empty:
                return recent_data['high'].mean() - recent_data['low'].mean()
            return 0.02  # 2% default
        
        # Calculate True Range
        historical_data = historical_data.copy()
        historical_data['prev_close'] = historical_data['close'].shift(1)
        
        historical_data['tr1'] = historical_data['high'] - historical_data['low']
        historical_data['tr2'] = abs(historical_data['high'] - historical_data['prev_close'])
        historical_data['tr3'] = abs(historical_data['low'] - historical_data['prev_close'])
        
        historical_data['true_range'] = historical_data[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Return ATR
        return historical_data['true_range'].tail(period).mean()
    
    def _get_regime_at_date(self, regime_data: pd.DataFrame, date) -> str:
        """Get market regime at specific date"""
        if regime_data is None:
            return "unknown"
        
        # Find closest date in regime data
        regime_dates = regime_data.index.date
        valid_dates = [d for d in regime_dates if d <= date]
        
        if not valid_dates:
            return "unknown"
        
        closest_date = max(valid_dates)
        regime_row = regime_data[regime_data.index.date == closest_date]
        
        if not regime_row.empty:
            return regime_row.iloc[0].get('regime', 'unknown')
        
        return "unknown"
    
    def _update_portfolio_metrics(self, portfolio: Dict, price_data: Dict, timestamp: datetime):
        """Update portfolio equity curve and drawdown"""
        
        # Calculate current portfolio value
        current_value = portfolio['cash']
        
        # Add value of open positions
        for trade in portfolio['positions'].values():
            if trade.symbol in price_data:
                symbol_data = price_data[trade.symbol]
                current_date = timestamp.date()
                
                # Find price data for current date
                date_data = symbol_data[symbol_data.index.date <= current_date]
                if not date_data.empty:
                    current_price = date_data.iloc[-1]['close']
                    position_value = current_price * trade.position_size
                    current_value += position_value
        
        # Record equity point
        portfolio['equity_curve'].append({
            'timestamp': timestamp,
            'equity': current_value
        })
        
        # Calculate drawdown
        if len(portfolio['equity_curve']) > 1:
            peak_equity = max(point['equity'] for point in portfolio['equity_curve'])
            current_drawdown = (current_value - peak_equity) / peak_equity
            
            portfolio['drawdown_curve'].append({
                'timestamp': timestamp,
                'drawdown': current_drawdown
            })
    
    async def _close_all_positions(self, portfolio: Dict, price_data: Dict, end_date: datetime):
        """Close all remaining positions at backtest end"""
        
        remaining_trades = list(portfolio['positions'].values())
        
        for trade in remaining_trades:
            if trade.symbol in price_data:
                symbol_data = price_data[trade.symbol]
                end_data = symbol_data[symbol_data.index.date <= end_date.date()]
                
                if not end_data.empty:
                    exit_price = end_data.iloc[-1]['close']
                    
                    await self._exit_trade(
                        trade=trade,
                        exit_date=end_date.date(),
                        exit_price=exit_price,
                        exit_reason=ExitReason.TIME_DECAY,
                        portfolio=portfolio,
                        include_costs=True,
                        commission_rate=0.001,
                        slippage_bps=5.0,
                        max_favorable=0.0,
                        max_adverse=0.0,
                        regime_data=None
                    )
    
    def _calculate_results(
        self,
        portfolio: Dict,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float
    ) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        
        trades = portfolio['trades']
        
        if not trades:
            # Return empty results if no trades
            return BacktestResults(
                total_return=0.0, annual_return=0.0, sharpe_ratio=0.0,
                sortino_ratio=0.0, max_drawdown=0.0, win_rate=0.0,
                profit_factor=0.0, total_trades=0, winning_trades=0,
                losing_trades=0, average_win=0.0, average_loss=0.0,
                largest_win=0.0, largest_loss=0.0,
                event_type_performance={}, surprise_magnitude_performance={},
                market_cap_performance={}, sector_performance={},
                var_95=0.0, expected_shortfall=0.0,
                maximum_consecutive_losses=0, average_holding_period=0.0,
                stop_loss_effectiveness={}, optimal_stop_loss_levels={},
                regime_performance={}, trades=[]
            )
        
        # Basic performance metrics
        total_pnl = sum(trade.pnl for trade in trades if trade.pnl)
        final_equity = portfolio['equity_curve'][-1]['equity'] if portfolio['equity_curve'] else initial_capital
        total_return = (final_equity - initial_capital) / initial_capital
        
        # Annualized return
        days = (end_date - start_date).days
        annual_return = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0
        
        # Trade statistics
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        average_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        average_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Risk metrics
        returns = [t.pnl_percentage for t in trades if t.pnl_percentage]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
        
        negative_returns = [r for r in returns if r < 0]
        sortino_ratio = np.mean(returns) / np.std(negative_returns) * np.sqrt(252) if negative_returns and np.std(negative_returns) > 0 else 0
        
        # Maximum drawdown
        if portfolio['drawdown_curve']:
            max_drawdown = min(point['drawdown'] for point in portfolio['drawdown_curve'])
        else:
            max_drawdown = 0.0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Event type performance
        event_type_performance = {}
        for event_type in EventType:
            event_trades = [t for t in trades if t.event.event_type == event_type]
            if event_trades:
                event_pnl = sum(t.pnl for t in event_trades if t.pnl)
                event_win_rate = len([t for t in event_trades if t.pnl and t.pnl > 0]) / len(event_trades)
                event_type_performance[event_type] = {
                    'total_pnl': event_pnl,
                    'win_rate': event_win_rate,
                    'trade_count': len(event_trades),
                    'average_return': np.mean([t.pnl_percentage for t in event_trades if t.pnl_percentage])
                }
        
        # Additional analysis placeholders
        surprise_magnitude_performance = {}
        market_cap_performance = {}
        sector_performance = {}
        stop_loss_effectiveness = {}
        optimal_stop_loss_levels = {}
        regime_performance = {}
        
        return BacktestResults(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            average_win=average_win,
            average_loss=average_loss,
            largest_win=max([t.pnl for t in trades if t.pnl], default=0),
            largest_loss=min([t.pnl for t in trades if t.pnl], default=0),
            event_type_performance=event_type_performance,
            surprise_magnitude_performance=surprise_magnitude_performance,
            market_cap_performance=market_cap_performance,
            sector_performance=sector_performance,
            var_95=np.percentile(returns, 5) if returns else 0,
            expected_shortfall=np.mean([r for r in returns if r <= np.percentile(returns, 5)]) if returns else 0,
            maximum_consecutive_losses=self._calculate_max_consecutive_losses(trades),
            average_holding_period=np.mean([t.holding_period_hours for t in trades if t.holding_period_hours]) if trades else 0,
            stop_loss_effectiveness=stop_loss_effectiveness,
            optimal_stop_loss_levels=optimal_stop_loss_levels,
            regime_performance=regime_performance,
            trades=trades
        )
    
    def _calculate_max_consecutive_losses(self, trades: List[EventTrade]) -> int:
        """Calculate maximum consecutive losing trades"""
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade.pnl and trade.pnl <= 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive