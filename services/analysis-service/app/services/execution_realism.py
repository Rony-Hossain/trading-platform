"""
Execution Realism Framework for Realistic Trading Simulation.

Bridges the gap between idealized backtesting and live trading reality by modeling:
1. Market microstructure effects and latency
2. Queue position and fill probability
3. Trading halts and market closures
4. Slippage and market impact modeling
5. Order book dynamics and liquidity
6. Transaction cost modeling
7. Execution timing and delays
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timedelta, time
import logging
from enum import Enum
import random
from scipy import stats
from scipy.interpolate import interp1d
import bisect

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for execution modeling."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side specification."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class MarketState(Enum):
    """Market state for halt handling."""
    OPEN = "open"
    CLOSED = "closed"
    HALTED = "halted"
    PRE_MARKET = "pre_market"
    POST_MARKET = "post_market"


@dataclass
class Order:
    """Order representation for execution simulation."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    queue_position: Optional[int] = None
    estimated_fill_time: Optional[datetime] = None


@dataclass
class Fill:
    """Order fill representation."""
    order_id: str
    fill_id: str
    timestamp: datetime
    price: float
    quantity: float
    fees: float
    is_partial: bool


@dataclass
class MarketData:
    """Real-time market data for execution modeling."""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    vwap: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None


@dataclass
class LatencyProfile:
    """Network and processing latency configuration."""
    market_data_latency_ms: float = 2.0  # Market data feed latency
    order_latency_ms: float = 5.0        # Order submission latency
    fill_latency_ms: float = 3.0         # Fill confirmation latency
    cancel_latency_ms: float = 4.0       # Order cancellation latency
    jitter_ms: float = 1.0               # Random latency variation


@dataclass
class LiquidityProfile:
    """Market liquidity characteristics."""
    average_bid_ask_spread_bps: float = 5.0    # Average spread in basis points
    average_depth_usd: float = 100000.0        # Average depth per level
    price_levels: int = 5                      # Number of price levels
    depth_decay_factor: float = 0.7            # Depth decay by level
    liquidity_regeneration_rate: float = 0.1   # How fast liquidity regenerates


class LatencySimulator:
    """Simulates realistic network and processing latencies."""
    
    def __init__(self, profile: LatencyProfile):
        self.profile = profile
        
    def get_market_data_latency(self) -> float:
        """Get market data feed latency in milliseconds."""
        base_latency = self.profile.market_data_latency_ms
        jitter = np.random.normal(0, self.profile.jitter_ms)
        return max(0.1, base_latency + jitter)
    
    def get_order_latency(self) -> float:
        """Get order submission latency in milliseconds."""
        base_latency = self.profile.order_latency_ms
        jitter = np.random.normal(0, self.profile.jitter_ms)
        return max(0.1, base_latency + jitter)
    
    def get_fill_latency(self) -> float:
        """Get fill confirmation latency in milliseconds."""
        base_latency = self.profile.fill_latency_ms
        jitter = np.random.normal(0, self.profile.jitter_ms)
        return max(0.1, base_latency + jitter)
    
    def get_cancel_latency(self) -> float:
        """Get order cancellation latency in milliseconds."""
        base_latency = self.profile.cancel_latency_ms
        jitter = np.random.normal(0, self.profile.jitter_ms)
        return max(0.1, base_latency + jitter)


class OrderBook:
    """Simplified order book for queue position and fill modeling."""
    
    def __init__(self, symbol: str, liquidity_profile: LiquidityProfile):
        self.symbol = symbol
        self.liquidity_profile = liquidity_profile
        self.bids = {}  # price -> total_quantity
        self.asks = {}  # price -> total_quantity
        self.bid_queue = {}  # price -> [order_queue]
        self.ask_queue = {}  # price -> [order_queue]
        self.last_update = None
        
    def update_market_data(self, market_data: MarketData):
        """Update order book based on market data."""
        self.last_update = market_data.timestamp
        
        # Generate synthetic order book levels
        self._generate_synthetic_levels(market_data)
        
    def _generate_synthetic_levels(self, market_data: MarketData):
        """Generate synthetic order book levels from market data."""
        spread = market_data.ask - market_data.bid
        tick_size = self._estimate_tick_size(market_data.last_price)
        
        # Clear existing levels
        self.bids.clear()
        self.asks.clear()
        
        # Generate bid levels
        for i in range(self.liquidity_profile.price_levels):
            price = market_data.bid - (i * tick_size)
            depth = self.liquidity_profile.average_depth_usd * (
                self.liquidity_profile.depth_decay_factor ** i
            )
            quantity = depth / price
            self.bids[price] = quantity
            
        # Generate ask levels
        for i in range(self.liquidity_profile.price_levels):
            price = market_data.ask + (i * tick_size)
            depth = self.liquidity_profile.average_depth_usd * (
                self.liquidity_profile.depth_decay_factor ** i
            )
            quantity = depth / price
            self.asks[price] = quantity
    
    def _estimate_tick_size(self, price: float) -> float:
        """Estimate tick size based on price level."""
        if price < 1.0:
            return 0.0001
        elif price < 10.0:
            return 0.01
        elif price < 100.0:
            return 0.01
        else:
            return 0.01
    
    def add_order_to_queue(self, order: Order) -> int:
        """Add order to queue and return queue position."""
        if order.order_type != OrderType.LIMIT:
            return 0  # Market orders don't queue
            
        if order.side == OrderSide.BUY:
            queue = self.bid_queue.setdefault(order.price, [])
        else:
            queue = self.ask_queue.setdefault(order.price, [])
            
        queue.append(order.order_id)
        return len(queue)
    
    def get_queue_position(self, order: Order) -> int:
        """Get current queue position for an order."""
        if order.order_type != OrderType.LIMIT:
            return 0
            
        if order.side == OrderSide.BUY:
            queue = self.bid_queue.get(order.price, [])
        else:
            queue = self.ask_queue.get(order.price, [])
            
        try:
            return queue.index(order.order_id) + 1
        except ValueError:
            return 0
    
    def estimate_fill_probability(self, order: Order, time_horizon_sec: float) -> float:
        """Estimate probability of fill within time horizon."""
        if order.order_type == OrderType.MARKET:
            return 1.0  # Market orders fill immediately (if liquidity exists)
            
        # For limit orders, consider queue position and price level
        queue_pos = self.get_queue_position(order)
        
        if order.side == OrderSide.BUY:
            best_bid = max(self.bids.keys()) if self.bids else 0
            if order.price >= best_bid:
                # Aggressive limit order
                fill_prob = min(0.9, 1.0 / (1 + queue_pos * 0.1))
            else:
                # Passive limit order
                fill_prob = min(0.5, 0.1 / (1 + queue_pos * 0.05))
        else:
            best_ask = min(self.asks.keys()) if self.asks else float('inf')
            if order.price <= best_ask:
                # Aggressive limit order
                fill_prob = min(0.9, 1.0 / (1 + queue_pos * 0.1))
            else:
                # Passive limit order
                fill_prob = min(0.5, 0.1 / (1 + queue_pos * 0.05))
        
        # Adjust for time horizon
        time_factor = min(1.0, time_horizon_sec / 60.0)  # More likely over longer time
        return fill_prob * time_factor


class MarketHaltSimulator:
    """Simulates market halts and trading session management."""
    
    def __init__(self):
        self.halt_probability_per_day = 0.01  # 1% chance of halt per day
        self.halt_duration_mean_minutes = 15
        self.halt_duration_std_minutes = 10
        self.current_halts = {}  # symbol -> halt_end_time
        
        # Trading session times (US market hours)
        self.market_open_time = time(9, 30)
        self.market_close_time = time(16, 0)
        self.pre_market_open = time(4, 0)
        self.post_market_close = time(20, 0)
    
    def get_market_state(self, timestamp: datetime, symbol: str) -> MarketState:
        """Get current market state for symbol."""
        
        # Check for specific halts
        if symbol in self.current_halts:
            if timestamp < self.current_halts[symbol]:
                return MarketState.HALTED
            else:
                del self.current_halts[symbol]
        
        # Check trading session
        current_time = timestamp.time()
        
        if current_time < self.pre_market_open:
            return MarketState.CLOSED
        elif current_time < self.market_open_time:
            return MarketState.PRE_MARKET
        elif current_time < self.market_close_time:
            return MarketState.OPEN
        elif current_time < self.post_market_close:
            return MarketState.POST_MARKET
        else:
            return MarketState.CLOSED
    
    def simulate_halt(self, timestamp: datetime, symbol: str) -> bool:
        """Simulate potential trading halt."""
        
        # Random halt simulation
        if np.random.random() < self.halt_probability_per_day / (24 * 60):  # Per minute probability
            halt_duration = max(1, np.random.normal(
                self.halt_duration_mean_minutes,
                self.halt_duration_std_minutes
            ))
            halt_end = timestamp + timedelta(minutes=halt_duration)
            self.current_halts[symbol] = halt_end
            logger.info(f"Trading halt simulated for {symbol} until {halt_end}")
            return True
        
        return False


class SlippageModel:
    """Models market impact and slippage effects."""
    
    def __init__(self):
        self.linear_impact_bps = 0.5      # Linear impact coefficient
        self.sqrt_impact_bps = 2.0        # Square root impact coefficient
        self.temporary_impact_decay = 0.5  # Temporary impact decay rate
        
    def calculate_slippage(
        self,
        order: Order,
        market_data: MarketData,
        adv_usd: float = 1000000  # Average daily volume in USD
    ) -> float:
        """
        Calculate expected slippage for an order.
        
        Args:
            order: Order to execute
            market_data: Current market data
            adv_usd: Average daily volume in USD
            
        Returns:
            Expected slippage in price units
        """
        
        order_value = order.quantity * market_data.last_price
        participation_rate = order_value / adv_usd
        
        # Linear impact component
        linear_impact = self.linear_impact_bps * participation_rate / 10000
        
        # Square root impact component
        sqrt_impact = self.sqrt_impact_bps * np.sqrt(participation_rate) / 10000
        
        # Total impact
        total_impact_bps = linear_impact + sqrt_impact
        slippage = market_data.last_price * total_impact_bps
        
        # Apply direction
        if order.side == OrderSide.BUY:
            return slippage  # Positive slippage (worse price)
        else:
            return -slippage  # Negative slippage (worse price)
    
    def calculate_bid_ask_impact(
        self,
        order: Order,
        market_data: MarketData
    ) -> float:
        """Calculate bid-ask spread impact."""
        
        spread = market_data.ask - market_data.bid
        
        if order.order_type == OrderType.MARKET:
            # Market orders pay the spread
            if order.side == OrderSide.BUY:
                return (market_data.ask - market_data.last_price)
            else:
                return (market_data.bid - market_data.last_price)
        else:
            # Limit orders may avoid spread
            return 0.0


class TransactionCostModel:
    """Models comprehensive transaction costs."""
    
    def __init__(self):
        self.commission_per_share = 0.001   # $0.001 per share
        self.commission_min = 1.0           # Minimum $1 commission
        self.commission_max = 50.0          # Maximum $50 commission
        self.sec_fee_rate = 0.0000051       # SEC fee rate
        self.taf_fee_rate = 0.000119        # TAF fee rate (per share)
        self.borrowing_rate_bps = 25        # Stock borrowing cost (for shorts)
        
    def calculate_commission(self, order: Order) -> float:
        """Calculate commission costs."""
        commission = order.quantity * self.commission_per_share
        return max(self.commission_min, min(self.commission_max, commission))
    
    def calculate_fees(self, order: Order, fill_price: float) -> float:
        """Calculate regulatory fees."""
        trade_value = order.quantity * fill_price
        
        sec_fee = trade_value * self.sec_fee_rate if order.side == OrderSide.SELL else 0
        taf_fee = order.quantity * self.taf_fee_rate
        
        return sec_fee + taf_fee
    
    def calculate_borrowing_cost(
        self,
        order: Order,
        holding_period_days: float
    ) -> float:
        """Calculate stock borrowing costs for short positions."""
        
        if order.side == OrderSide.SELL:  # Short sale
            daily_rate = self.borrowing_rate_bps / 10000 / 365
            return order.quantity * order.price * daily_rate * holding_period_days
        return 0.0


class ExecutionSimulator:
    """Comprehensive execution simulator combining all realism factors."""
    
    def __init__(
        self,
        latency_profile: Optional[LatencyProfile] = None,
        liquidity_profile: Optional[LiquidityProfile] = None
    ):
        self.latency_simulator = LatencySimulator(latency_profile or LatencyProfile())
        self.liquidity_profile = liquidity_profile or LiquidityProfile()
        self.halt_simulator = MarketHaltSimulator()
        self.slippage_model = SlippageModel()
        self.cost_model = TransactionCostModel()
        
        self.order_books = {}  # symbol -> OrderBook
        self.pending_orders = {}  # order_id -> Order
        self.fills = []  # List of Fill objects
        self.current_time = None
        
    def submit_order(
        self,
        order: Order,
        market_data: MarketData,
        current_time: datetime
    ) -> Dict[str, any]:
        """
        Submit order for execution simulation.
        
        Returns:
            Dictionary with execution details and timing
        """
        
        self.current_time = current_time
        order.timestamp = current_time
        
        # Check market state
        market_state = self.halt_simulator.get_market_state(current_time, order.symbol)
        
        if market_state in [MarketState.CLOSED, MarketState.HALTED]:
            order.status = OrderStatus.REJECTED
            return {
                'order_id': order.order_id,
                'status': 'rejected',
                'reason': f'Market {market_state.value}',
                'timestamp': current_time
            }
        
        # Add order latency
        order_latency_ms = self.latency_simulator.get_order_latency()
        order_receipt_time = current_time + timedelta(milliseconds=order_latency_ms)
        
        # Initialize order book if needed
        if order.symbol not in self.order_books:
            self.order_books[order.symbol] = OrderBook(order.symbol, self.liquidity_profile)
        
        order_book = self.order_books[order.symbol]
        order_book.update_market_data(market_data)
        
        # Process order based on type
        if order.order_type == OrderType.MARKET:
            execution_result = self._execute_market_order(order, market_data, order_receipt_time)
        else:
            execution_result = self._process_limit_order(order, market_data, order_receipt_time)
        
        self.pending_orders[order.order_id] = order
        
        return execution_result
    
    def _execute_market_order(
        self,
        order: Order,
        market_data: MarketData,
        execution_time: datetime
    ) -> Dict[str, any]:
        """Execute market order immediately."""
        
        # Calculate execution price with slippage
        slippage = self.slippage_model.calculate_slippage(order, market_data)
        spread_impact = self.slippage_model.calculate_bid_ask_impact(order, market_data)
        
        if order.side == OrderSide.BUY:
            execution_price = market_data.ask + slippage
        else:
            execution_price = market_data.bid + slippage
        
        execution_price += spread_impact
        
        # Calculate costs
        commission = self.cost_model.calculate_commission(order)
        fees = self.cost_model.calculate_fees(order, execution_price)
        total_fees = commission + fees
        
        # Create fill
        fill = Fill(
            order_id=order.order_id,
            fill_id=f"{order.order_id}_fill_1",
            timestamp=execution_time + timedelta(milliseconds=self.latency_simulator.get_fill_latency()),
            price=execution_price,
            quantity=order.quantity,
            fees=total_fees,
            is_partial=False
        )
        
        self.fills.append(fill)
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = execution_price
        
        return {
            'order_id': order.order_id,
            'status': 'filled',
            'fill_price': execution_price,
            'fill_quantity': order.quantity,
            'fees': total_fees,
            'slippage': slippage,
            'execution_time': execution_time,
            'fill_time': fill.timestamp
        }
    
    def _process_limit_order(
        self,
        order: Order,
        market_data: MarketData,
        submission_time: datetime
    ) -> Dict[str, any]:
        """Process limit order with queue modeling."""
        
        order_book = self.order_books[order.symbol]
        
        # Add to queue
        queue_position = order_book.add_order_to_queue(order)
        order.queue_position = queue_position
        
        # Estimate fill probability and time
        fill_probability = order_book.estimate_fill_probability(order, 300)  # 5 minutes
        
        if fill_probability > 0.7:  # High probability orders
            estimated_fill_delay = np.random.exponential(30)  # Average 30 seconds
            order.estimated_fill_time = submission_time + timedelta(seconds=estimated_fill_delay)
        else:
            order.estimated_fill_time = None
        
        return {
            'order_id': order.order_id,
            'status': 'pending',
            'queue_position': queue_position,
            'fill_probability': fill_probability,
            'estimated_fill_time': order.estimated_fill_time,
            'submission_time': submission_time
        }
    
    def update_market_data(self, market_data: MarketData):
        """Update market data and process pending limit orders."""
        
        self.current_time = market_data.timestamp
        
        # Update order book
        if market_data.symbol in self.order_books:
            order_book = self.order_books[market_data.symbol]
            order_book.update_market_data(market_data)
            
            # Check pending limit orders for fills
            self._process_pending_limit_orders(market_data)
    
    def _process_pending_limit_orders(self, market_data: MarketData):
        """Process pending limit orders for potential fills."""
        
        pending_symbol_orders = [
            order for order in self.pending_orders.values()
            if order.symbol == market_data.symbol and order.status == OrderStatus.PENDING
        ]
        
        for order in pending_symbol_orders:
            if self._should_limit_order_fill(order, market_data):
                self._fill_limit_order(order, market_data)
    
    def _should_limit_order_fill(self, order: Order, market_data: MarketData) -> bool:
        """Determine if limit order should fill based on market data."""
        
        if order.side == OrderSide.BUY:
            # Buy order fills if market trades at or below limit price
            return market_data.last_price <= order.price
        else:
            # Sell order fills if market trades at or above limit price
            return market_data.last_price >= order.price
    
    def _fill_limit_order(self, order: Order, market_data: MarketData):
        """Fill a limit order."""
        
        # Use limit price for fill
        fill_price = order.price
        
        # Calculate costs
        commission = self.cost_model.calculate_commission(order)
        fees = self.cost_model.calculate_fees(order, fill_price)
        total_fees = commission + fees
        
        # Create fill
        fill = Fill(
            order_id=order.order_id,
            fill_id=f"{order.order_id}_fill_limit",
            timestamp=self.current_time + timedelta(milliseconds=self.latency_simulator.get_fill_latency()),
            price=fill_price,
            quantity=order.quantity,
            fees=total_fees,
            is_partial=False
        )
        
        self.fills.append(fill)
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
    
    def get_execution_summary(self) -> Dict[str, any]:
        """Get comprehensive execution summary."""
        
        if not self.fills:
            return {}
        
        total_fees = sum(fill.fees for fill in self.fills)
        total_volume = sum(fill.quantity * fill.price for fill in self.fills)
        avg_execution_delay = np.mean([
            (fill.timestamp - order.timestamp).total_seconds()
            for order in self.pending_orders.values()
            for fill in self.fills
            if fill.order_id == order.order_id and order.timestamp
        ])
        
        fill_rate = len([o for o in self.pending_orders.values() if o.status == OrderStatus.FILLED]) / len(self.pending_orders)
        
        return {
            'total_fills': len(self.fills),
            'total_fees': total_fees,
            'total_volume': total_volume,
            'fee_rate_bps': (total_fees / total_volume) * 10000 if total_volume > 0 else 0,
            'avg_execution_delay_sec': avg_execution_delay,
            'fill_rate': fill_rate,
            'orders_processed': len(self.pending_orders)
        }


class RealisticBacktester:
    """Backtesting framework with execution realism."""
    
    def __init__(
        self,
        execution_simulator: ExecutionSimulator,
        universe: List[str],
        start_date: datetime,
        end_date: datetime
    ):
        self.execution_simulator = execution_simulator
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.backtest_results = []
        
    def run_strategy_backtest(
        self,
        strategy_signals: pd.DataFrame,  # columns: timestamp, symbol, signal, quantity
        market_data: pd.DataFrame,       # columns: timestamp, symbol, bid, ask, last, volume
        initial_capital: float = 1000000.0
    ) -> Dict[str, any]:
        """
        Run realistic backtest with execution simulation.
        
        Args:
            strategy_signals: DataFrame with strategy signals
            market_data: DataFrame with market data
            initial_capital: Starting capital
            
        Returns:
            Comprehensive backtest results
        """
        
        portfolio_value = initial_capital
        positions = {symbol: 0.0 for symbol in self.universe}
        cash = initial_capital
        
        # Merge signals and market data by timestamp
        combined_data = self._prepare_backtest_data(strategy_signals, market_data)
        
        portfolio_history = []
        execution_stats = []
        
        for timestamp, data_group in combined_data.groupby('timestamp'):
            
            # Process market data updates
            for _, market_row in data_group[data_group['data_type'] == 'market'].iterrows():
                market_data_point = MarketData(
                    timestamp=timestamp,
                    symbol=market_row['symbol'],
                    bid=market_row['bid'],
                    ask=market_row['ask'],
                    last_price=market_row['last'],
                    volume=market_row['volume'],
                    bid_size=market_row.get('bid_size', 100),
                    ask_size=market_row.get('ask_size', 100)
                )
                self.execution_simulator.update_market_data(market_data_point)
            
            # Process strategy signals
            for _, signal_row in data_group[data_group['data_type'] == 'signal'].iterrows():
                
                if signal_row['signal'] != 0:  # Non-zero signal
                    
                    # Create order
                    order = Order(
                        order_id=f"order_{timestamp}_{signal_row['symbol']}",
                        symbol=signal_row['symbol'],
                        side=OrderSide.BUY if signal_row['signal'] > 0 else OrderSide.SELL,
                        order_type=OrderType.MARKET,  # Simplified to market orders
                        quantity=abs(signal_row['quantity'])
                    )
                    
                    # Get current market data for this symbol
                    symbol_market_data = data_group[
                        (data_group['symbol'] == signal_row['symbol']) & 
                        (data_group['data_type'] == 'market')
                    ]
                    
                    if not symbol_market_data.empty:
                        market_row = symbol_market_data.iloc[0]
                        market_data_point = MarketData(
                            timestamp=timestamp,
                            symbol=market_row['symbol'],
                            bid=market_row['bid'],
                            ask=market_row['ask'],
                            last_price=market_row['last'],
                            volume=market_row['volume'],
                            bid_size=market_row.get('bid_size', 100),
                            ask_size=market_row.get('ask_size', 100)
                        )
                        
                        # Submit order
                        execution_result = self.execution_simulator.submit_order(
                            order, market_data_point, timestamp
                        )
                        
                        # Update portfolio if order filled
                        if execution_result['status'] == 'filled':
                            
                            trade_value = execution_result['fill_quantity'] * execution_result['fill_price']
                            fees = execution_result['fees']
                            
                            if order.side == OrderSide.BUY:
                                positions[order.symbol] += execution_result['fill_quantity']
                                cash -= (trade_value + fees)
                            else:
                                positions[order.symbol] -= execution_result['fill_quantity']
                                cash += (trade_value - fees)
                            
                            execution_stats.append({
                                'timestamp': timestamp,
                                'symbol': order.symbol,
                                'side': order.side.value,
                                'quantity': execution_result['fill_quantity'],
                                'price': execution_result['fill_price'],
                                'fees': fees,
                                'slippage': execution_result.get('slippage', 0)
                            })
            
            # Calculate portfolio value
            current_portfolio_value = cash
            for symbol in self.universe:
                if positions[symbol] != 0:
                    # Get current market price
                    symbol_data = data_group[
                        (data_group['symbol'] == symbol) & 
                        (data_group['data_type'] == 'market')
                    ]
                    if not symbol_data.empty:
                        last_price = symbol_data.iloc[0]['last']
                        current_portfolio_value += positions[symbol] * last_price
            
            portfolio_history.append({
                'timestamp': timestamp,
                'portfolio_value': current_portfolio_value,
                'cash': cash,
                'positions': positions.copy()
            })
        
        # Calculate performance metrics
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.set_index('timestamp', inplace=True)
        
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / initial_capital) - 1
        execution_summary = self.execution_simulator.get_execution_summary()
        
        return {
            'portfolio_history': portfolio_df,
            'execution_stats': pd.DataFrame(execution_stats),
            'total_return': total_return,
            'execution_summary': execution_summary,
            'final_positions': positions,
            'final_cash': cash
        }
    
    def _prepare_backtest_data(
        self,
        signals: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare combined data for backtesting."""
        
        # Add data type indicators
        signals_copy = signals.copy()
        signals_copy['data_type'] = 'signal'
        
        market_copy = market_data.copy()
        market_copy['data_type'] = 'market'
        
        # Combine and sort by timestamp
        combined = pd.concat([signals_copy, market_copy], ignore_index=True)
        combined.sort_values('timestamp', inplace=True)
        
        return combined