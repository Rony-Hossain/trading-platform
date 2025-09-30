"""
Execution Realism Framework

This module provides realistic execution modeling for backtesting and live trading,
accounting for real-world trading constraints and market microstructure effects:

1. Latency Modeling: Network, processing, and exchange latencies
2. Queue Position: Order book position modeling and fill probability
3. Market Impact: Price impact from order size and market conditions
4. Trading Halts: Circuit breaker and halt handling
5. Slippage Models: Various slippage estimation methods
6. Partial Fills: Realistic fill modeling with time decay

Key Features:
- Multi-venue execution with venue-specific characteristics
- Realistic latency distributions based on market data
- Queue position modeling using limit order book dynamics
- Market impact models (linear, square-root, power-law)
- Circuit breaker and trading halt simulation
- Execution cost analysis and attribution

References:
- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions.
- Kissell, R. (2013). The Science of Algorithmic Trading and Portfolio Management.
- Hasbrouck, J. (2007). Empirical Market Microstructure.
"""

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from collections import deque
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import numba

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for execution modeling."""
    MARKET = "market"
    LIMIT = "limit"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class VenueType(Enum):
    """Trading venue types."""
    PRIMARY_EXCHANGE = "primary_exchange"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    MARKET_MAKER = "market_maker"
    CROSSING_NETWORK = "crossing_network"


class HaltReason(Enum):
    """Reasons for trading halts."""
    CIRCUIT_BREAKER = "circuit_breaker"
    NEWS_PENDING = "news_pending"
    VOLATILITY = "volatility"
    REGULATORY = "regulatory"
    TECHNICAL = "technical"


@dataclass
class LatencyProfile:
    """Latency characteristics for execution modeling."""
    network_latency_ms: float = 1.0      # Network round-trip latency
    processing_latency_ms: float = 0.5    # Internal processing latency
    exchange_latency_ms: float = 0.3      # Exchange processing latency
    venue_latency_ms: float = 0.2         # Venue-specific latency
    jitter_std_ms: float = 0.1            # Latency standard deviation
    
    def total_latency_ms(self) -> float:
        """Calculate total expected latency."""
        return (self.network_latency_ms + self.processing_latency_ms + 
                self.exchange_latency_ms + self.venue_latency_ms)
    
    def sample_latency(self) -> float:
        """Sample latency from distribution."""
        base_latency = self.total_latency_ms()
        jitter = np.random.normal(0, self.jitter_std_ms)
        return max(0.1, base_latency + jitter)  # Minimum 0.1ms


@dataclass
class VenueCharacteristics:
    """Trading venue characteristics."""
    venue_id: str
    venue_type: VenueType
    latency_profile: LatencyProfile
    market_share: float = 0.1             # Venue market share
    fill_probability: float = 0.8         # Base fill probability
    fee_rate: float = 0.0003             # Trading fee rate
    min_size: int = 1                    # Minimum order size
    max_size: int = 1000000              # Maximum order size
    supports_hidden: bool = False         # Supports hidden orders
    tick_size: float = 0.01              # Minimum price increment
    
    def __post_init__(self):
        if not (0 <= self.market_share <= 1):
            raise ValueError("market_share must be between 0 and 1")
        if not (0 <= self.fill_probability <= 1):
            raise ValueError("fill_probability must be between 0 and 1")


@dataclass
class MarketImpactModel:
    """Market impact model parameters."""
    linear_coeff: float = 0.1            # Linear impact coefficient
    sqrt_coeff: float = 0.05             # Square-root impact coefficient
    power_coeff: float = 0.02            # Power-law impact coefficient
    power_exponent: float = 0.6          # Power-law exponent
    temporary_decay: float = 0.5         # Temporary impact decay rate
    adv_scaling: float = 1.0             # Average daily volume scaling
    volatility_scaling: float = 1.0      # Volatility scaling factor
    
    def calculate_impact(self, 
                        order_size: float,
                        adv: float,
                        volatility: float,
                        participation_rate: float = 0.1) -> Tuple[float, float]:
        """
        Calculate permanent and temporary market impact.
        
        Parameters:
        - order_size: Order size in shares
        - adv: Average daily volume
        - volatility: Price volatility
        - participation_rate: Participation rate in volume
        
        Returns:
        - (permanent_impact, temporary_impact) in basis points
        """
        # Normalize order size by ADV
        size_ratio = order_size / (adv * self.adv_scaling)
        
        # Linear impact
        linear_impact = self.linear_coeff * size_ratio
        
        # Square-root impact (Almgren-Chriss model)
        sqrt_impact = self.sqrt_coeff * np.sqrt(size_ratio)
        
        # Power-law impact
        power_impact = self.power_coeff * (size_ratio ** self.power_exponent)
        
        # Combine impacts
        permanent_impact = (linear_impact + sqrt_impact) * volatility * self.volatility_scaling
        temporary_impact = (permanent_impact + power_impact) * (1 + participation_rate)
        
        # Apply temporary decay
        temporary_impact *= (1 - self.temporary_decay)
        
        return permanent_impact, temporary_impact


@dataclass
class OrderBookState:
    """Simplified order book state for queue modeling."""
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    bid_depth: List[Tuple[float, float]] = field(default_factory=list)  # (price, size)
    ask_depth: List[Tuple[float, float]] = field(default_factory=list)  # (price, size)
    last_update: float = field(default_factory=time.time)
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask_price - self.bid_price
    
    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        return self.spread / self.mid_price * 10000


@dataclass
class ExecutionOrder:
    """Order for execution simulation."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None         # For limit orders
    venue_id: Optional[str] = None        # Preferred venue
    timestamp: float = field(default_factory=time.time)
    parent_order_id: Optional[str] = None # For order slicing
    time_in_force: str = "DAY"           # DAY, IOC, FOK, GTC
    hidden: bool = False                 # Hidden/iceberg order
    min_quantity: Optional[float] = None  # Minimum fill size
    
    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("price required for limit orders")


@dataclass
class ExecutionResult:
    """Result of order execution attempt."""
    order_id: str
    status: OrderStatus
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    total_fees: float = 0.0
    venue_id: Optional[str] = None
    execution_time: float = field(default_factory=time.time)
    queue_position: Optional[int] = None
    market_impact_bps: float = 0.0
    slippage_bps: float = 0.0
    latency_ms: float = 0.0
    rejection_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'total_fees': self.total_fees,
            'venue_id': self.venue_id,
            'execution_time': self.execution_time,
            'queue_position': self.queue_position,
            'market_impact_bps': self.market_impact_bps,
            'slippage_bps': self.slippage_bps,
            'latency_ms': self.latency_ms,
            'rejection_reason': self.rejection_reason
        }


class ExecutionSimulator:
    """
    Realistic execution simulator with microstructure modeling.
    
    Simulates order execution with realistic latencies, queue positions,
    market impact, and other microstructure effects.
    """
    
    def __init__(self):
        """Initialize execution simulator."""
        # Venue configuration
        self.venues: Dict[str, VenueCharacteristics] = {}
        self.market_impact_model = MarketImpactModel()
        
        # Market state
        self.order_books: Dict[str, OrderBookState] = {}
        self.trading_halts: Dict[str, Tuple[float, HaltReason]] = {}
        self.circuit_breaker_levels = {
            'level_1': 0.07,  # 7% decline triggers 15-minute halt
            'level_2': 0.13,  # 13% decline triggers 15-minute halt
            'level_3': 0.20   # 20% decline triggers trading halt for the day
        }
        
        # Execution tracking
        self.pending_orders: Dict[str, ExecutionOrder] = {}
        self.execution_history: List[ExecutionResult] = []
        self.order_queues: Dict[str, deque] = {}  # Venue order queues
        
        # Performance metrics
        self.adv_estimates: Dict[str, float] = {}  # Average daily volume
        self.volatility_estimates: Dict[str, float] = {}  # Volatility estimates
        
        # Initialize default venues
        self._initialize_default_venues()
    
    def _initialize_default_venues(self):
        """Initialize default venue configurations."""
        # Primary exchange (e.g., NYSE, NASDAQ)
        self.venues['PRIMARY'] = VenueCharacteristics(
            venue_id='PRIMARY',
            venue_type=VenueType.PRIMARY_EXCHANGE,
            latency_profile=LatencyProfile(
                network_latency_ms=1.2,
                processing_latency_ms=0.3,
                exchange_latency_ms=0.5,
                venue_latency_ms=0.1
            ),
            market_share=0.3,
            fill_probability=0.9,
            fee_rate=0.0003,
            supports_hidden=False
        )
        
        # Dark pool
        self.venues['DARK'] = VenueCharacteristics(
            venue_id='DARK',
            venue_type=VenueType.DARK_POOL,
            latency_profile=LatencyProfile(
                network_latency_ms=2.0,
                processing_latency_ms=1.0,
                exchange_latency_ms=0.8,
                venue_latency_ms=0.3
            ),
            market_share=0.15,
            fill_probability=0.6,
            fee_rate=0.0001,
            supports_hidden=True
        )
        
        # ECN
        self.venues['ECN'] = VenueCharacteristics(
            venue_id='ECN',
            venue_type=VenueType.ECN,
            latency_profile=LatencyProfile(
                network_latency_ms=0.8,
                processing_latency_ms=0.2,
                exchange_latency_ms=0.3,
                venue_latency_ms=0.1
            ),
            market_share=0.2,
            fill_probability=0.8,
            fee_rate=0.0002,
            supports_hidden=True
        )
    
    def update_market_data(self, 
                          symbol: str,
                          order_book: OrderBookState,
                          adv: Optional[float] = None,
                          volatility: Optional[float] = None):
        """
        Update market data for execution simulation.
        
        Parameters:
        - symbol: Security symbol
        - order_book: Current order book state
        - adv: Average daily volume
        - volatility: Price volatility
        """
        self.order_books[symbol] = order_book
        
        if adv is not None:
            self.adv_estimates[symbol] = adv
        
        if volatility is not None:
            self.volatility_estimates[symbol] = volatility
    
    def submit_order(self, order: ExecutionOrder) -> str:
        """
        Submit order for execution.
        
        Parameters:
        - order: Order to execute
        
        Returns:
        - Order ID
        """
        logger.debug(f"Submitting order {order.order_id} for {order.symbol}")
        
        # Check for trading halts
        if self._is_trading_halted(order.symbol):
            result = ExecutionResult(
                order_id=order.order_id,
                status=OrderStatus.REJECTED,
                rejection_reason="Trading halted"
            )
            self.execution_history.append(result)
            return order.order_id
        
        # Store pending order
        self.pending_orders[order.order_id] = order
        
        # Route to venue
        venue_id = self._route_order(order)
        
        # Simulate execution
        result = self._execute_order(order, venue_id)
        
        # Update order status
        if result.status == OrderStatus.FILLED:
            self.pending_orders.pop(order.order_id, None)
        
        self.execution_history.append(result)
        
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel pending order.
        
        Parameters:
        - order_id: Order ID to cancel
        
        Returns:
        - True if successfully cancelled
        """
        if order_id in self.pending_orders:
            order = self.pending_orders.pop(order_id)
            
            result = ExecutionResult(
                order_id=order_id,
                status=OrderStatus.CANCELLED
            )
            self.execution_history.append(result)
            
            logger.debug(f"Cancelled order {order_id}")
            return True
        
        return False
    
    def get_execution_results(self, order_id: Optional[str] = None) -> List[ExecutionResult]:
        """
        Get execution results.
        
        Parameters:
        - order_id: Optional order ID filter
        
        Returns:
        - List of execution results
        """
        if order_id is not None:
            return [r for r in self.execution_history if r.order_id == order_id]
        
        return self.execution_history.copy()
    
    def _route_order(self, order: ExecutionOrder) -> str:
        """
        Route order to appropriate venue.
        
        Parameters:
        - order: Order to route
        
        Returns:
        - Selected venue ID
        """
        # Use specified venue if provided
        if order.venue_id and order.venue_id in self.venues:
            return order.venue_id
        
        # Smart order routing based on order characteristics
        if order.order_type == OrderType.MARKET:
            # Route market orders to venue with best fill probability
            best_venue = max(self.venues.keys(), 
                           key=lambda v: self.venues[v].fill_probability)
            return best_venue
        
        elif order.hidden and any(v.supports_hidden for v in self.venues.values()):
            # Route hidden orders to dark pools
            dark_venues = [v for v in self.venues.keys() 
                          if self.venues[v].supports_hidden]
            return max(dark_venues, 
                      key=lambda v: self.venues[v].market_share)
        
        else:
            # Default routing to primary exchange
            return 'PRIMARY'
    
    def _execute_order(self, order: ExecutionOrder, venue_id: str) -> ExecutionResult:
        """
        Execute order at specified venue.
        
        Parameters:
        - order: Order to execute
        - venue_id: Venue for execution
        
        Returns:
        - Execution result
        """
        venue = self.venues[venue_id]
        
        # Simulate latency
        latency_ms = venue.latency_profile.sample_latency()
        
        # Get current order book
        if order.symbol not in self.order_books:
            return ExecutionResult(
                order_id=order.order_id,
                status=OrderStatus.REJECTED,
                rejection_reason=f"No market data for {order.symbol}",
                latency_ms=latency_ms
            )
        
        order_book = self.order_books[order.symbol]
        
        # Simulate execution based on order type
        if order.order_type == OrderType.MARKET:
            return self._execute_market_order(order, venue, order_book, latency_ms)
        
        elif order.order_type == OrderType.LIMIT:
            return self._execute_limit_order(order, venue, order_book, latency_ms)
        
        elif order.order_type == OrderType.IOC:
            return self._execute_ioc_order(order, venue, order_book, latency_ms)
        
        else:
            return ExecutionResult(
                order_id=order.order_id,
                status=OrderStatus.REJECTED,
                rejection_reason=f"Unsupported order type: {order.order_type}",
                latency_ms=latency_ms
            )
    
    def _execute_market_order(self, 
                            order: ExecutionOrder,
                            venue: VenueCharacteristics,
                            order_book: OrderBookState,
                            latency_ms: float) -> ExecutionResult:
        """Execute market order."""
        # Market orders fill at best available price
        if order.side == OrderSide.BUY:
            fill_price = order_book.ask_price
            available_size = order_book.ask_size
        else:
            fill_price = order_book.bid_price
            available_size = order_book.bid_size
        
        # Calculate fill probability based on venue and market conditions
        base_fill_prob = venue.fill_probability
        
        # Adjust for order size relative to available liquidity
        size_ratio = order.quantity / max(available_size, 1)
        size_penalty = min(0.3, size_ratio * 0.2)  # Max 30% penalty
        fill_prob = max(0.1, base_fill_prob - size_penalty)
        
        # Simulate fill decision
        if np.random.random() > fill_prob:
            return ExecutionResult(
                order_id=order.order_id,
                status=OrderStatus.REJECTED,
                rejection_reason="Insufficient liquidity",
                latency_ms=latency_ms
            )
        
        # Calculate filled quantity (may be partial)
        if order.quantity <= available_size:
            filled_qty = order.quantity
        else:
            # Partial fill based on available liquidity
            fill_ratio = np.random.uniform(0.5, 1.0)  # 50-100% of available
            filled_qty = min(order.quantity, available_size * fill_ratio)
        
        # Calculate market impact
        market_impact_bps = self._calculate_market_impact(
            order.symbol, order.quantity, order_book
        )
        
        # Apply market impact to fill price
        if order.side == OrderSide.BUY:
            final_price = fill_price * (1 + market_impact_bps / 10000)
        else:
            final_price = fill_price * (1 - market_impact_bps / 10000)
        
        # Calculate fees
        fees = filled_qty * final_price * venue.fee_rate
        
        # Calculate slippage
        benchmark_price = order_book.mid_price
        if order.side == OrderSide.BUY:
            slippage_bps = (final_price - benchmark_price) / benchmark_price * 10000
        else:
            slippage_bps = (benchmark_price - final_price) / benchmark_price * 10000
        
        # Determine status
        if filled_qty >= order.quantity:
            status = OrderStatus.FILLED
        else:
            status = OrderStatus.PARTIALLY_FILLED
        
        return ExecutionResult(
            order_id=order.order_id,
            status=status,
            filled_quantity=filled_qty,
            avg_fill_price=final_price,
            total_fees=fees,
            venue_id=venue.venue_id,
            market_impact_bps=market_impact_bps,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms
        )
    
    def _execute_limit_order(self,
                           order: ExecutionOrder,
                           venue: VenueCharacteristics,
                           order_book: OrderBookState,
                           latency_ms: float) -> ExecutionResult:
        """Execute limit order with queue position modeling."""
        # Check if limit price is executable immediately
        if order.side == OrderSide.BUY and order.price >= order_book.ask_price:
            # Limit buy at or above ask - execute as market order
            return self._execute_market_order(order, venue, order_book, latency_ms)
        
        elif order.side == OrderSide.SELL and order.price <= order_book.bid_price:
            # Limit sell at or below bid - execute as market order
            return self._execute_market_order(order, venue, order_book, latency_ms)
        
        # Order goes to order book - simulate queue position
        queue_position = self._estimate_queue_position(order, venue, order_book)
        
        # Simulate fill probability based on queue position
        fill_prob = self._calculate_limit_fill_probability(
            order, venue, order_book, queue_position
        )
        
        if np.random.random() > fill_prob:
            return ExecutionResult(
                order_id=order.order_id,
                status=OrderStatus.PENDING,
                queue_position=queue_position,
                latency_ms=latency_ms
            )
        
        # Simulate partial fill
        fill_ratio = np.random.uniform(0.3, 1.0)
        filled_qty = order.quantity * fill_ratio
        
        # Calculate fees
        fees = filled_qty * order.price * venue.fee_rate
        
        # No market impact for limit orders at limit price
        market_impact_bps = 0.0
        
        # Calculate slippage relative to mid-price
        benchmark_price = order_book.mid_price
        if order.side == OrderSide.BUY:
            slippage_bps = (order.price - benchmark_price) / benchmark_price * 10000
        else:
            slippage_bps = (benchmark_price - order.price) / benchmark_price * 10000
        
        status = OrderStatus.FILLED if filled_qty >= order.quantity else OrderStatus.PARTIALLY_FILLED
        
        return ExecutionResult(
            order_id=order.order_id,
            status=status,
            filled_quantity=filled_qty,
            avg_fill_price=order.price,
            total_fees=fees,
            venue_id=venue.venue_id,
            queue_position=queue_position,
            market_impact_bps=market_impact_bps,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms
        )
    
    def _execute_ioc_order(self,
                         order: ExecutionOrder,
                         venue: VenueCharacteristics,
                         order_book: OrderBookState,
                         latency_ms: float) -> ExecutionResult:
        """Execute Immediate-or-Cancel order."""
        # IOC orders execute immediately at best available price or cancel
        if order.side == OrderSide.BUY:
            if order.price and order.price < order_book.ask_price:
                # Price not acceptable - cancel
                return ExecutionResult(
                    order_id=order.order_id,
                    status=OrderStatus.CANCELLED,
                    rejection_reason="Price not acceptable for IOC order",
                    latency_ms=latency_ms
                )
            fill_price = order_book.ask_price
            available_size = order_book.ask_size
        else:
            if order.price and order.price > order_book.bid_price:
                # Price not acceptable - cancel
                return ExecutionResult(
                    order_id=order.order_id,
                    status=OrderStatus.CANCELLED,
                    rejection_reason="Price not acceptable for IOC order",
                    latency_ms=latency_ms
                )
            fill_price = order_book.bid_price
            available_size = order_book.bid_size
        
        # Execute as market order with immediate execution
        filled_qty = min(order.quantity, available_size)
        
        if filled_qty == 0:
            return ExecutionResult(
                order_id=order.order_id,
                status=OrderStatus.CANCELLED,
                rejection_reason="No liquidity available",
                latency_ms=latency_ms
            )
        
        # Calculate market impact and fees
        market_impact_bps = self._calculate_market_impact(
            order.symbol, filled_qty, order_book
        )
        
        if order.side == OrderSide.BUY:
            final_price = fill_price * (1 + market_impact_bps / 10000)
        else:
            final_price = fill_price * (1 - market_impact_bps / 10000)
        
        fees = filled_qty * final_price * venue.fee_rate
        
        # Calculate slippage
        benchmark_price = order_book.mid_price
        if order.side == OrderSide.BUY:
            slippage_bps = (final_price - benchmark_price) / benchmark_price * 10000
        else:
            slippage_bps = (benchmark_price - final_price) / benchmark_price * 10000
        
        status = OrderStatus.FILLED if filled_qty >= order.quantity else OrderStatus.PARTIALLY_FILLED
        
        return ExecutionResult(
            order_id=order.order_id,
            status=status,
            filled_quantity=filled_qty,
            avg_fill_price=final_price,
            total_fees=fees,
            venue_id=venue.venue_id,
            market_impact_bps=market_impact_bps,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms
        )
    
    def _calculate_market_impact(self,
                               symbol: str,
                               order_size: float,
                               order_book: OrderBookState) -> float:
        """Calculate market impact in basis points."""
        # Get market parameters
        adv = self.adv_estimates.get(symbol, 1000000)  # Default 1M shares ADV
        volatility = self.volatility_estimates.get(symbol, 0.02)  # Default 2% volatility
        
        # Calculate participation rate
        participation_rate = order_size / adv
        
        # Use market impact model
        permanent_impact, temporary_impact = self.market_impact_model.calculate_impact(
            order_size, adv, volatility, participation_rate
        )
        
        return temporary_impact * 10000  # Convert to basis points
    
    def _estimate_queue_position(self,
                               order: ExecutionOrder,
                               venue: VenueCharacteristics,
                               order_book: OrderBookState) -> int:
        """Estimate queue position for limit order."""
        # Simplified queue position modeling
        # In practice, this would use order book depth and arrival rates
        
        if order.side == OrderSide.BUY:
            if order.price == order_book.bid_price:
                # At best bid - estimate based on venue market share
                base_queue = order_book.bid_size * venue.market_share
                position = int(np.random.uniform(1, base_queue + 1))
            else:
                # Away from best bid - likely at front of queue
                position = int(np.random.uniform(1, 10))
        else:
            if order.price == order_book.ask_price:
                # At best ask
                base_queue = order_book.ask_size * venue.market_share
                position = int(np.random.uniform(1, base_queue + 1))
            else:
                # Away from best ask
                position = int(np.random.uniform(1, 10))
        
        return max(1, position)
    
    def _calculate_limit_fill_probability(self,
                                        order: ExecutionOrder,
                                        venue: VenueCharacteristics,
                                        order_book: OrderBookState,
                                        queue_position: int) -> float:
        """Calculate fill probability for limit order."""
        # Base fill probability from venue
        base_prob = venue.fill_probability
        
        # Adjust for queue position (higher position = lower probability)
        queue_penalty = min(0.8, queue_position / 100.0)  # Max 80% penalty
        fill_prob = base_prob * (1 - queue_penalty)
        
        # Adjust for spread (tighter spread = higher fill probability)
        spread_bps = order_book.spread_bps
        spread_bonus = max(0, (50 - spread_bps) / 100)  # Bonus for tight spreads
        fill_prob = min(1.0, fill_prob + spread_bonus)
        
        return max(0.01, fill_prob)  # Minimum 1% fill probability
    
    def _is_trading_halted(self, symbol: str) -> bool:
        """Check if trading is halted for symbol."""
        if symbol in self.trading_halts:
            halt_start, reason = self.trading_halts[symbol]
            current_time = time.time()
            
            # Check if halt is still active (assuming 15-minute halts)
            if current_time - halt_start < 900:  # 15 minutes
                return True
            else:
                # Remove expired halt
                del self.trading_halts[symbol]
        
        return False
    
    def trigger_circuit_breaker(self, 
                               symbol: str, 
                               price_decline: float,
                               reason: HaltReason = HaltReason.CIRCUIT_BREAKER):
        """Trigger circuit breaker halt."""
        logger.warning(f"Circuit breaker triggered for {symbol}: {price_decline:.1%} decline")
        
        self.trading_halts[symbol] = (time.time(), reason)
        
        # Cancel all pending orders for halted symbol
        orders_to_cancel = [
            order_id for order_id, order in self.pending_orders.items()
            if order.symbol == symbol
        ]
        
        for order_id in orders_to_cancel:
            self.cancel_order(order_id)
    
    def get_execution_analytics(self) -> Dict[str, Any]:
        """Get execution analytics and performance metrics."""
        if not self.execution_history:
            return {}
        
        results_df = pd.DataFrame([r.to_dict() for r in self.execution_history])
        
        # Fill rate
        total_orders = len(results_df)
        filled_orders = len(results_df[results_df['status'] == 'filled'])
        fill_rate = filled_orders / total_orders if total_orders > 0 else 0
        
        # Average metrics for filled orders
        filled_df = results_df[results_df['status'] == 'filled']
        
        if len(filled_df) > 0:
            avg_slippage = filled_df['slippage_bps'].mean()
            avg_market_impact = filled_df['market_impact_bps'].mean()
            avg_latency = filled_df['latency_ms'].mean()
            total_fees = filled_df['total_fees'].sum()
        else:
            avg_slippage = avg_market_impact = avg_latency = total_fees = 0
        
        # Venue analytics
        venue_stats = {}
        if len(filled_df) > 0:
            venue_stats = filled_df.groupby('venue_id').agg({
                'slippage_bps': 'mean',
                'market_impact_bps': 'mean',
                'latency_ms': 'mean',
                'order_id': 'count'
            }).to_dict('index')
        
        return {
            'total_orders': total_orders,
            'fill_rate': fill_rate,
            'avg_slippage_bps': avg_slippage,
            'avg_market_impact_bps': avg_market_impact,
            'avg_latency_ms': avg_latency,
            'total_fees': total_fees,
            'venue_analytics': venue_stats
        }