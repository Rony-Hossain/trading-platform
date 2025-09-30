"""
Execution Modeling Engine

Advanced execution simulation system that models realistic trading latency,
slippage, and supports aggressive order types like limit orders and IOC orders.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import asyncio
import numpy as np
import pandas as pd
from decimal import Decimal
import uuid

class OrderType(Enum):
    """Order type classifications"""
    MARKET = "market"                   # Execute immediately at market price
    LIMIT = "limit"                     # Execute at specified price or better
    STOP = "stop"                       # Trigger market order when price hit
    STOP_LIMIT = "stop_limit"           # Trigger limit order when price hit
    IOC = "immediate_or_cancel"         # Fill immediately or cancel
    FOK = "fill_or_kill"               # Fill completely or cancel
    MOC = "market_on_close"            # Execute at closing price
    LOC = "limit_on_close"             # Limit order for closing auction

class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"                 # Order submitted, awaiting execution
    PARTIALLY_FILLED = "partially_filled"  # Partial execution
    FILLED = "filled"                   # Complete execution
    CANCELLED = "cancelled"             # Order cancelled
    REJECTED = "rejected"               # Order rejected by exchange
    EXPIRED = "expired"                 # Order expired (IOC, FOK)

class VenueType(Enum):
    """Trading venue types"""
    EXCHANGE = "exchange"               # Primary exchange (NYSE, NASDAQ)
    DARK_POOL = "dark_pool"            # Dark pool ATS
    ECN = "ecn"                        # Electronic Communication Network
    MARKET_MAKER = "market_maker"       # Market maker venue
    RETAIL = "retail"                   # Retail broker internalization

@dataclass
class LatencyProfile:
    """Latency characteristics for different venues"""
    venue_type: VenueType
    base_latency_ms: float              # Base one-way latency
    latency_std_ms: float               # Latency standard deviation
    processing_time_ms: float           # Order processing time
    market_data_latency_ms: float       # Market data feed latency
    network_jitter_ms: float            # Network jitter component
    
class MarketMicrostructure:
    """Market microstructure parameters"""
    
    def __init__(self):
        # Bid-ask spread modeling
        self.min_spread_bps = 1.0          # Minimum spread in basis points
        self.typical_spread_bps = 5.0      # Typical spread for liquid stocks
        self.spread_volatility_factor = 2.0 # Spread widens with volatility
        
        # Order book depth
        self.typical_book_depth = 10000    # Shares at best price
        self.depth_decay_factor = 0.7      # Depth decay across price levels
        self.large_order_threshold = 5000  # Large order impact threshold
        
        # Liquidity parameters
        self.liquidity_recovery_time = 30  # Seconds for liquidity recovery
        self.price_impact_factor = 0.5     # Price impact coefficient
        self.temporary_impact_decay = 0.8  # Temporary impact decay rate

@dataclass
class OrderRequest:
    """Order execution request"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"          # DAY, GTC, IOC, FOK
    venue_preference: Optional[VenueType] = None
    urgency: float = 0.5                # 0=patient, 1=urgent
    max_participation_rate: float = 0.20 # Max % of volume
    
    # Timestamps
    submission_time: Optional[datetime] = None
    client_timestamp: Optional[datetime] = None

@dataclass
class Fill:
    """Individual fill record"""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    venue: VenueType
    timestamp: datetime
    commission: float
    fees: float
    liquidity_flag: str                 # "REMOVE" or "ADD"

@dataclass
class OrderExecution:
    """Complete order execution result"""
    order_id: str
    original_request: OrderRequest
    status: OrderStatus
    fills: List[Fill]
    
    # Execution metrics
    total_quantity_filled: int
    average_fill_price: float
    total_commission: float
    total_fees: float
    
    # Timing metrics
    submission_timestamp: datetime
    first_fill_timestamp: Optional[datetime]
    completion_timestamp: Optional[datetime]
    total_execution_time_ms: float
    
    # Market impact metrics
    benchmark_price: float              # Price when order submitted
    implementation_shortfall: float     # Total cost vs benchmark
    market_impact_bps: float           # Price impact in basis points
    timing_cost_bps: float             # Cost due to timing/latency
    
    # Venue breakdown
    venue_fills: Dict[VenueType, int]
    venue_avg_prices: Dict[VenueType, float]

class ExecutionModelingEngine:
    """Advanced execution modeling with latency simulation"""
    
    def __init__(self):
        # Venue latency profiles
        self.venue_profiles = {
            VenueType.EXCHANGE: LatencyProfile(
                venue_type=VenueType.EXCHANGE,
                base_latency_ms=150.0,         # Primary exchange latency
                latency_std_ms=25.0,
                processing_time_ms=50.0,
                market_data_latency_ms=100.0,
                network_jitter_ms=15.0
            ),
            VenueType.ECN: LatencyProfile(
                venue_type=VenueType.ECN,
                base_latency_ms=80.0,          # Faster ECN
                latency_std_ms=15.0,
                processing_time_ms=30.0,
                market_data_latency_ms=60.0,
                network_jitter_ms=10.0
            ),
            VenueType.DARK_POOL: LatencyProfile(
                venue_type=VenueType.DARK_POOL,
                base_latency_ms=200.0,         # Slower dark pool
                latency_std_ms=40.0,
                processing_time_ms=80.0,
                market_data_latency_ms=150.0,
                network_jitter_ms=20.0
            ),
            VenueType.MARKET_MAKER: LatencyProfile(
                venue_type=VenueType.MARKET_MAKER,
                base_latency_ms=120.0,
                latency_std_ms=20.0,
                processing_time_ms=40.0,
                market_data_latency_ms=90.0,
                network_jitter_ms=12.0
            ),
            VenueType.RETAIL: LatencyProfile(
                venue_type=VenueType.RETAIL,
                base_latency_ms=300.0,         # Retail internalization
                latency_std_ms=50.0,
                processing_time_ms=100.0,
                market_data_latency_ms=200.0,
                network_jitter_ms=25.0
            )
        }
        
        self.microstructure = MarketMicrostructure()
        
        # Order book simulation
        self.order_books = {}               # Symbol -> OrderBook
        self.market_data_feeds = {}         # Symbol -> MarketData
        
        # Execution queue for latency simulation
        self.execution_queue = asyncio.Queue()
        self.pending_orders = {}            # order_id -> OrderRequest
        
    async def submit_order(
        self,
        order_request: OrderRequest,
        current_market_data: Dict[str, Any]
    ) -> str:
        """
        Submit order for execution with latency simulation
        """
        # Generate order ID and timestamp
        order_request.order_id = str(uuid.uuid4())
        order_request.submission_time = datetime.utcnow()
        order_request.client_timestamp = datetime.utcnow()
        
        # Store pending order
        self.pending_orders[order_request.order_id] = order_request
        
        # Select venue based on order characteristics
        selected_venue = self._select_optimal_venue(order_request, current_market_data)
        
        # Calculate execution latency
        latency_ms = self._calculate_execution_latency(selected_venue, order_request)
        
        # Schedule execution after latency delay
        execution_time = datetime.utcnow() + timedelta(milliseconds=latency_ms)
        
        # Add to execution queue
        await self.execution_queue.put({
            'order_id': order_request.order_id,
            'venue': selected_venue,
            'scheduled_time': execution_time,
            'market_data_snapshot': current_market_data.copy()
        })
        
        return order_request.order_id
    
    async def process_execution_queue(self) -> List[OrderExecution]:
        """
        Process pending executions (should be called continuously)
        """
        executions = []
        current_time = datetime.utcnow()
        
        # Process all ready executions
        ready_executions = []
        while not self.execution_queue.empty():
            try:
                queued_item = self.execution_queue.get_nowait()
                if queued_item['scheduled_time'] <= current_time:
                    ready_executions.append(queued_item)
                else:
                    # Put back if not ready
                    await self.execution_queue.put(queued_item)
                    break
            except asyncio.QueueEmpty:
                break
        
        # Execute ready orders
        for queued_item in ready_executions:
            execution = await self._execute_order(
                order_id=queued_item['order_id'],
                venue=queued_item['venue'],
                market_data=queued_item['market_data_snapshot']
            )
            executions.append(execution)
        
        return executions
    
    async def _execute_order(
        self,
        order_id: str,
        venue: VenueType,
        market_data: Dict[str, Any]
    ) -> OrderExecution:
        """
        Execute individual order with venue-specific logic
        """
        order_request = self.pending_orders[order_id]
        execution_start = datetime.utcnow()
        
        # Get current market conditions
        symbol_data = market_data.get(order_request.symbol, {})
        bid_price = symbol_data.get('bid', 100.0)
        ask_price = symbol_data.get('ask', 100.1)
        last_price = symbol_data.get('last', 100.05)
        volume = symbol_data.get('volume', 1000000)
        
        # Simulate order book depth and spread
        spread_bps = self._calculate_current_spread(symbol_data, venue)
        book_depth = self._calculate_book_depth(symbol_data, venue)
        
        # Execute based on order type
        fills = []
        if order_request.order_type == OrderType.MARKET:
            fills = await self._execute_market_order(
                order_request, bid_price, ask_price, book_depth, venue
            )
        elif order_request.order_type == OrderType.LIMIT:
            fills = await self._execute_limit_order(
                order_request, bid_price, ask_price, book_depth, venue
            )
        elif order_request.order_type == OrderType.IOC:
            fills = await self._execute_ioc_order(
                order_request, bid_price, ask_price, book_depth, venue
            )
        elif order_request.order_type == OrderType.FOK:
            fills = await self._execute_fok_order(
                order_request, bid_price, ask_price, book_depth, venue
            )
        else:
            # Default to market order
            fills = await self._execute_market_order(
                order_request, bid_price, ask_price, book_depth, venue
            )
        
        # Calculate execution metrics
        execution = self._calculate_execution_metrics(
            order_request, fills, last_price, execution_start
        )
        
        # Remove from pending orders
        del self.pending_orders[order_id]
        
        return execution
    
    async def _execute_market_order(
        self,
        order: OrderRequest,
        bid_price: float,
        ask_price: float,
        book_depth: int,
        venue: VenueType
    ) -> List[Fill]:
        """Execute market order with realistic slippage"""
        
        fills = []
        remaining_qty = order.quantity
        
        # Determine aggressive price based on side
        if order.side == OrderSide.BUY:
            base_price = ask_price
        else:
            base_price = bid_price
        
        # Calculate market impact based on order size
        order_impact = self._calculate_market_impact(
            order.quantity, book_depth, venue
        )
        
        # Execute in chunks if large order
        chunk_size = min(remaining_qty, book_depth)
        chunk_count = 0
        
        while remaining_qty > 0 and chunk_count < 10:  # Max 10 chunks
            fill_qty = min(remaining_qty, chunk_size)
            
            # Apply slippage that increases with order progression
            progression_factor = chunk_count * 0.1  # 10 bps per chunk
            slippage_bps = order_impact + progression_factor
            
            if order.side == OrderSide.BUY:
                fill_price = base_price * (1 + slippage_bps / 10000)
            else:
                fill_price = base_price * (1 - slippage_bps / 10000)
            
            # Create fill
            fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=fill_qty,
                price=fill_price,
                venue=venue,
                timestamp=datetime.utcnow(),
                commission=self._calculate_commission(fill_qty, fill_price, venue),
                fees=self._calculate_fees(fill_qty, fill_price, venue),
                liquidity_flag="REMOVE"  # Market orders remove liquidity
            )
            
            fills.append(fill)
            remaining_qty -= fill_qty
            chunk_count += 1
            
            # Reduce available depth for next chunk
            book_depth = max(100, int(book_depth * 0.7))
            
            # Add microsecond delay for realism
            await asyncio.sleep(0.001)
        
        return fills
    
    async def _execute_limit_order(
        self,
        order: OrderRequest,
        bid_price: float,
        ask_price: float,
        book_depth: int,
        venue: VenueType
    ) -> List[Fill]:
        """Execute limit order with price improvement logic"""
        
        fills = []
        
        # Check if limit order can execute immediately
        can_execute = False
        if order.side == OrderSide.BUY and order.limit_price >= ask_price:
            can_execute = True
            execution_price = min(order.limit_price, ask_price)
        elif order.side == OrderSide.SELL and order.limit_price <= bid_price:
            can_execute = True
            execution_price = max(order.limit_price, bid_price)
        
        if can_execute:
            # Immediate execution with possible price improvement
            available_qty = min(order.quantity, book_depth)
            
            # Simulate partial fill for large orders
            if order.quantity > book_depth:
                fill_qty = book_depth
                # In reality, remainder would wait in order book
            else:
                fill_qty = order.quantity
            
            fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=fill_qty,
                price=execution_price,
                venue=venue,
                timestamp=datetime.utcnow(),
                commission=self._calculate_commission(fill_qty, execution_price, venue),
                fees=self._calculate_fees(fill_qty, execution_price, venue),
                liquidity_flag="REMOVE" if can_execute else "ADD"
            )
            
            fills.append(fill)
        
        # If order doesn't execute immediately, it would rest in book
        # For simulation purposes, we assume partial execution
        
        return fills
    
    async def _execute_ioc_order(
        self,
        order: OrderRequest,
        bid_price: float,
        ask_price: float,
        book_depth: int,
        venue: VenueType
    ) -> List[Fill]:
        """Execute Immediate or Cancel order"""
        
        fills = []
        
        # IOC orders must execute immediately or be cancelled
        if order.side == OrderSide.BUY:
            if order.limit_price and order.limit_price >= ask_price:
                # Can execute at ask or better
                available_qty = min(order.quantity, book_depth)
                execution_price = min(order.limit_price, ask_price)
                
                fill = Fill(
                    fill_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=available_qty,
                    price=execution_price,
                    venue=venue,
                    timestamp=datetime.utcnow(),
                    commission=self._calculate_commission(available_qty, execution_price, venue),
                    fees=self._calculate_fees(available_qty, execution_price, venue),
                    liquidity_flag="REMOVE"
                )
                
                fills.append(fill)
            
        else:  # SELL
            if order.limit_price and order.limit_price <= bid_price:
                # Can execute at bid or better
                available_qty = min(order.quantity, book_depth)
                execution_price = max(order.limit_price, bid_price)
                
                fill = Fill(
                    fill_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=available_qty,
                    price=execution_price,
                    venue=venue,
                    timestamp=datetime.utcnow(),
                    commission=self._calculate_commission(available_qty, execution_price, venue),
                    fees=self._calculate_fees(available_qty, execution_price, venue),
                    liquidity_flag="REMOVE"
                )
                
                fills.append(fill)
        
        # Remainder is cancelled (IOC behavior)
        return fills
    
    async def _execute_fok_order(
        self,
        order: OrderRequest,
        bid_price: float,
        ask_price: float,
        book_depth: int,
        venue: VenueType
    ) -> List[Fill]:
        """Execute Fill or Kill order"""
        
        fills = []
        
        # FOK orders must fill completely or not at all
        can_fill_completely = False
        execution_price = 0.0
        
        if order.side == OrderSide.BUY:
            if order.limit_price and order.limit_price >= ask_price and order.quantity <= book_depth:
                can_fill_completely = True
                execution_price = min(order.limit_price, ask_price)
        else:  # SELL
            if order.limit_price and order.limit_price <= bid_price and order.quantity <= book_depth:
                can_fill_completely = True
                execution_price = max(order.limit_price, bid_price)
        
        if can_fill_completely:
            fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                venue=venue,
                timestamp=datetime.utcnow(),
                commission=self._calculate_commission(order.quantity, execution_price, venue),
                fees=self._calculate_fees(order.quantity, execution_price, venue),
                liquidity_flag="REMOVE"
            )
            
            fills.append(fill)
        
        # If can't fill completely, entire order is cancelled (FOK behavior)
        return fills
    
    def _select_optimal_venue(
        self, order: OrderRequest, market_data: Dict[str, Any]
    ) -> VenueType:
        """Select optimal venue based on order characteristics"""
        
        # Use venue preference if specified
        if order.venue_preference:
            return order.venue_preference
        
        # Smart order routing logic
        if order.urgency > 0.8:
            # High urgency - use fastest venue
            return VenueType.ECN
        elif order.quantity > 10000:
            # Large order - consider dark pool
            return VenueType.DARK_POOL
        elif order.order_type in [OrderType.IOC, OrderType.FOK]:
            # Aggressive orders - use exchange
            return VenueType.EXCHANGE
        else:
            # Default to exchange
            return VenueType.EXCHANGE
    
    def _calculate_execution_latency(
        self, venue: VenueType, order: OrderRequest
    ) -> float:
        """Calculate realistic execution latency"""
        
        profile = self.venue_profiles[venue]
        
        # Base latency with random variation
        base_latency = np.random.normal(
            profile.base_latency_ms,
            profile.latency_std_ms
        )
        
        # Add processing time
        processing_latency = profile.processing_time_ms
        
        # Add network jitter
        jitter = np.random.normal(0, profile.network_jitter_ms)
        
        # Additional latency for complex orders
        complexity_factor = 1.0
        if order.order_type in [OrderType.IOC, OrderType.FOK]:
            complexity_factor = 1.2
        elif order.order_type == OrderType.STOP_LIMIT:
            complexity_factor = 1.5
        
        total_latency = (base_latency + processing_latency + jitter) * complexity_factor
        
        return max(50.0, total_latency)  # Minimum 50ms latency
    
    def _calculate_current_spread(
        self, symbol_data: Dict[str, Any], venue: VenueType
    ) -> float:
        """Calculate current bid-ask spread in basis points"""
        
        # Get base spread
        volatility = symbol_data.get('volatility', 0.02)
        
        # Base spread increases with volatility
        spread_bps = self.microstructure.min_spread_bps + (
            volatility * self.microstructure.spread_volatility_factor * 10000
        )
        
        # Venue-specific spread adjustments
        venue_multipliers = {
            VenueType.EXCHANGE: 1.0,
            VenueType.ECN: 0.8,          # Tighter spreads
            VenueType.DARK_POOL: 1.2,     # Wider spreads
            VenueType.MARKET_MAKER: 1.1,
            VenueType.RETAIL: 1.5        # Widest spreads
        }
        
        return spread_bps * venue_multipliers.get(venue, 1.0)
    
    def _calculate_book_depth(
        self, symbol_data: Dict[str, Any], venue: VenueType
    ) -> int:
        """Calculate available order book depth"""
        
        base_depth = self.microstructure.typical_book_depth
        
        # Adjust based on venue
        venue_depth_factors = {
            VenueType.EXCHANGE: 1.0,
            VenueType.ECN: 0.6,
            VenueType.DARK_POOL: 0.8,
            VenueType.MARKET_MAKER: 0.7,
            VenueType.RETAIL: 0.3
        }
        
        depth = int(base_depth * venue_depth_factors.get(venue, 1.0))
        
        # Add randomness
        depth = int(np.random.normal(depth, depth * 0.2))
        
        return max(100, depth)
    
    def _calculate_market_impact(
        self, quantity: int, book_depth: int, venue: VenueType
    ) -> float:
        """Calculate market impact in basis points"""
        
        # Impact increases non-linearly with order size
        size_ratio = quantity / book_depth
        
        # Square root impact model
        base_impact = self.microstructure.price_impact_factor * np.sqrt(size_ratio) * 10
        
        # Venue adjustments
        venue_impact_factors = {
            VenueType.EXCHANGE: 1.0,
            VenueType.ECN: 0.8,
            VenueType.DARK_POOL: 0.6,    # Lower impact in dark pools
            VenueType.MARKET_MAKER: 0.9,
            VenueType.RETAIL: 1.2
        }
        
        impact_bps = base_impact * venue_impact_factors.get(venue, 1.0)
        
        return max(0.1, impact_bps)
    
    def _calculate_commission(self, quantity: int, price: float, venue: VenueType) -> float:
        """Calculate commission fees"""
        
        # Commission rates per share (in dollars)
        commission_rates = {
            VenueType.EXCHANGE: 0.0005,    # $0.0005 per share
            VenueType.ECN: 0.0003,         # Lower fees
            VenueType.DARK_POOL: 0.0002,   # Lowest fees
            VenueType.MARKET_MAKER: 0.0004,
            VenueType.RETAIL: 0.001        # Highest fees
        }
        
        rate = commission_rates.get(venue, 0.0005)
        return quantity * rate
    
    def _calculate_fees(self, quantity: int, price: float, venue: VenueType) -> float:
        """Calculate exchange and regulatory fees"""
        
        notional = quantity * price
        
        # SEC fee (sells only) + exchange fees
        sec_fee = 0.0000231 * notional  # SEC fee rate
        exchange_fee = notional * 0.0000119  # Exchange fee
        
        return sec_fee + exchange_fee
    
    def _calculate_execution_metrics(
        self,
        order_request: OrderRequest,
        fills: List[Fill],
        benchmark_price: float,
        execution_start: datetime
    ) -> OrderExecution:
        """Calculate comprehensive execution metrics"""
        
        if not fills:
            # No fills - order cancelled/expired
            return OrderExecution(
                order_id=order_request.order_id,
                original_request=order_request,
                status=OrderStatus.CANCELLED,
                fills=[],
                total_quantity_filled=0,
                average_fill_price=0.0,
                total_commission=0.0,
                total_fees=0.0,
                submission_timestamp=order_request.submission_time,
                first_fill_timestamp=None,
                completion_timestamp=datetime.utcnow(),
                total_execution_time_ms=(datetime.utcnow() - order_request.submission_time).total_seconds() * 1000,
                benchmark_price=benchmark_price,
                implementation_shortfall=0.0,
                market_impact_bps=0.0,
                timing_cost_bps=0.0,
                venue_fills={},
                venue_avg_prices={}
            )
        
        # Calculate basic metrics
        total_qty = sum(fill.quantity for fill in fills)
        total_notional = sum(fill.quantity * fill.price for fill in fills)
        avg_price = total_notional / total_qty if total_qty > 0 else 0
        
        total_commission = sum(fill.commission for fill in fills)
        total_fees = sum(fill.fees for fill in fills)
        
        # Timing metrics
        first_fill_time = min(fill.timestamp for fill in fills)
        last_fill_time = max(fill.timestamp for fill in fills)
        total_exec_time = (last_fill_time - order_request.submission_time).total_seconds() * 1000
        
        # Market impact analysis
        if order_request.side == OrderSide.BUY:
            price_diff = avg_price - benchmark_price
        else:
            price_diff = benchmark_price - avg_price
        
        market_impact_bps = (price_diff / benchmark_price) * 10000
        
        # Implementation shortfall
        ideal_notional = order_request.quantity * benchmark_price
        actual_notional = total_notional if order_request.side == OrderSide.BUY else -total_notional
        implementation_shortfall = actual_notional - ideal_notional + total_commission + total_fees
        
        # Venue breakdown
        venue_fills = {}
        venue_notionals = {}
        for fill in fills:
            venue_fills[fill.venue] = venue_fills.get(fill.venue, 0) + fill.quantity
            venue_notionals[fill.venue] = venue_notionals.get(fill.venue, 0) + (fill.quantity * fill.price)
        
        venue_avg_prices = {}
        for venue, notional in venue_notionals.items():
            venue_avg_prices[venue] = notional / venue_fills[venue]
        
        # Determine final status
        if total_qty == order_request.quantity:
            status = OrderStatus.FILLED
        elif total_qty > 0:
            status = OrderStatus.PARTIALLY_FILLED
        else:
            status = OrderStatus.CANCELLED
        
        return OrderExecution(
            order_id=order_request.order_id,
            original_request=order_request,
            status=status,
            fills=fills,
            total_quantity_filled=total_qty,
            average_fill_price=avg_price,
            total_commission=total_commission,
            total_fees=total_fees,
            submission_timestamp=order_request.submission_time,
            first_fill_timestamp=first_fill_time,
            completion_timestamp=last_fill_time,
            total_execution_time_ms=total_exec_time,
            benchmark_price=benchmark_price,
            implementation_shortfall=implementation_shortfall,
            market_impact_bps=market_impact_bps,
            timing_cost_bps=0.0,  # Would calculate from timing analysis
            venue_fills=venue_fills,
            venue_avg_prices=venue_avg_prices
        )
    
    async def get_execution_analytics(
        self, executions: List[OrderExecution]
    ) -> Dict[str, Any]:
        """Generate comprehensive execution analytics"""
        
        if not executions:
            return {"message": "No executions to analyze"}
        
        # Basic statistics
        total_orders = len(executions)
        filled_orders = len([e for e in executions if e.status == OrderStatus.FILLED])
        fill_rate = filled_orders / total_orders if total_orders > 0 else 0
        
        # Latency analysis
        execution_times = [e.total_execution_time_ms for e in executions if e.total_execution_time_ms]
        avg_execution_time = np.mean(execution_times) if execution_times else 0
        
        # Market impact analysis
        market_impacts = [e.market_impact_bps for e in executions if e.market_impact_bps]
        avg_market_impact = np.mean(market_impacts) if market_impacts else 0
        
        # Venue analysis
        venue_stats = {}
        for execution in executions:
            for venue, qty in execution.venue_fills.items():
                if venue not in venue_stats:
                    venue_stats[venue] = {'quantity': 0, 'orders': 0}
                venue_stats[venue]['quantity'] += qty
                venue_stats[venue]['orders'] += 1
        
        # Cost analysis
        total_commissions = sum(e.total_commission for e in executions)
        total_fees = sum(e.total_fees for e in executions)
        total_shortfall = sum(e.implementation_shortfall for e in executions)
        
        return {
            'execution_summary': {
                'total_orders': total_orders,
                'filled_orders': filled_orders,
                'fill_rate': fill_rate,
                'average_execution_time_ms': avg_execution_time,
                'average_market_impact_bps': avg_market_impact
            },
            'cost_analysis': {
                'total_commissions': total_commissions,
                'total_fees': total_fees,
                'total_implementation_shortfall': total_shortfall,
                'average_commission_per_order': total_commissions / total_orders if total_orders > 0 else 0
            },
            'venue_breakdown': {
                venue.value: stats for venue, stats in venue_stats.items()
            },
            'latency_distribution': {
                'min_ms': min(execution_times) if execution_times else 0,
                'max_ms': max(execution_times) if execution_times else 0,
                'median_ms': np.median(execution_times) if execution_times else 0,
                'p95_ms': np.percentile(execution_times, 95) if execution_times else 0
            },
            'market_impact_distribution': {
                'min_bps': min(market_impacts) if market_impacts else 0,
                'max_bps': max(market_impacts) if market_impacts else 0,
                'median_bps': np.median(market_impacts) if market_impacts else 0,
                'p95_bps': np.percentile(market_impacts, 95) if market_impacts else 0
            }
        }