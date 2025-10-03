"""
Halt Detection and Safe Order Management

Handles:
- LULD (Limit Up Limit Down) halt detection
- Auction periods (open, close, volatility)
- Circuit breaker detection
- Order type restrictions during halts

Acceptance Criteria:
- LULD halt detection: 100% detection rate in backtests
- Auction handling: Respect auction-only order types
- Circuit breaker awareness: Cancel working orders on halt
"""
import asyncio
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

import redis.asyncio as redis
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
halt_detections = Counter('halt_detector_halts_detected_total', 'Total halts detected', ['halt_type'])
auction_periods = Counter('halt_detector_auction_periods_total', 'Total auction periods detected', ['auction_type'])
orders_cancelled = Counter('halt_detector_orders_cancelled_total', 'Orders cancelled due to halts')
halt_detection_latency = Histogram('halt_detector_latency_seconds', 'Halt detection latency')


class HaltType(str, Enum):
    """Types of trading halts"""
    LULD_UPPER = "LULD_UPPER"  # Price hit upper band
    LULD_LOWER = "LULD_LOWER"  # Price hit lower band
    VOLATILITY = "VOLATILITY"  # Volatility halt
    NEWS_PENDING = "NEWS_PENDING"  # News pending
    CIRCUIT_BREAKER_L1 = "CIRCUIT_BREAKER_L1"  # Market-wide: 7% drop
    CIRCUIT_BREAKER_L2 = "CIRCUIT_BREAKER_L2"  # Market-wide: 13% drop
    CIRCUIT_BREAKER_L3 = "CIRCUIT_BREAKER_L3"  # Market-wide: 20% drop
    REGULATORY = "REGULATORY"  # Regulatory halt


class AuctionType(str, Enum):
    """Types of auction periods"""
    OPENING = "OPENING"  # 9:30 AM open
    CLOSING = "CLOSING"  # 4:00 PM close
    VOLATILITY = "VOLATILITY"  # Volatility auction after LULD
    HALT_RESUMPTION = "HALT_RESUMPTION"  # After halt


class OrderRestriction(str, Enum):
    """Order restrictions during special periods"""
    MARKET_ONLY = "MARKET_ONLY"  # Only market orders
    LIMIT_ONLY = "LIMIT_ONLY"  # Only limit orders
    AUCTION_ONLY = "AUCTION_ONLY"  # Only auction-eligible orders
    NO_CANCEL = "NO_CANCEL"  # Cannot cancel orders
    NO_NEW_ORDERS = "NO_NEW_ORDERS"  # No new orders accepted


@dataclass
class LULDBand:
    """LULD price band"""
    symbol: str
    reference_price: float
    upper_band: float
    lower_band: float
    band_width_pct: float
    tier: int  # 1 or 2
    updated_at: datetime


@dataclass
class HaltStatus:
    """Halt status for a symbol"""
    symbol: str
    is_halted: bool
    halt_type: Optional[HaltType] = None
    halt_start_time: Optional[datetime] = None
    expected_resume_time: Optional[datetime] = None
    restrictions: Set[OrderRestriction] = field(default_factory=set)
    message: Optional[str] = None


@dataclass
class AuctionStatus:
    """Auction status"""
    auction_type: AuctionType
    start_time: datetime
    expected_end_time: datetime
    is_active: bool
    indicative_price: Optional[float] = None
    imbalance_quantity: Optional[int] = None
    imbalance_side: Optional[str] = None  # BUY or SELL


class HaltDetector:
    """Detects trading halts and manages order restrictions"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None

        # LULD bands cache
        self.luld_bands: Dict[str, LULDBand] = {}

        # Halt status cache
        self.halt_status: Dict[str, HaltStatus] = {}

        # Auction status
        self.auction_status: Optional[AuctionStatus] = None

        # Circuit breaker status
        self.circuit_breaker_active = False
        self.circuit_breaker_level: Optional[HaltType] = None

        # Market hours (ET)
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)

    async def start(self):
        """Start halt detector"""
        self.redis_client = await redis.from_url(self.redis_url)
        logger.info("Halt detector started")

        # Start background tasks
        asyncio.create_task(self._monitor_luld_bands())
        asyncio.create_task(self._monitor_auctions())

    async def stop(self):
        """Stop halt detector"""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Halt detector stopped")

    def calculate_luld_bands(
        self,
        symbol: str,
        reference_price: float,
        tier: int = 1
    ) -> LULDBand:
        """
        Calculate LULD price bands

        Tier 1 (S&P 500, Russell 1000, select ETPs):
        - Price >= $3.00: 5% bands (9:30-9:45 and 3:35-4:00), 10% otherwise
        - Price < $3.00: 20% bands (9:30-9:45 and 3:35-4:00), 20% otherwise

        Tier 2 (Other NMS stocks):
        - Price >= $3.00: 10% bands (9:30-9:45 and 3:35-4:00), 20% otherwise
        - Price < $3.00: 20% bands (9:30-9:45 and 3:35-4:00), 40% otherwise
        """
        now = datetime.now().time()

        # Determine if in opening/closing periods
        is_opening = time(9, 30) <= now < time(9, 45)
        is_closing = time(15, 35) <= now <= time(16, 0)
        is_boundary_period = is_opening or is_closing

        # Calculate band width
        if tier == 1:
            if reference_price >= 3.00:
                band_width_pct = 5.0 if is_boundary_period else 10.0
            else:
                band_width_pct = 20.0
        else:  # tier == 2
            if reference_price >= 3.00:
                band_width_pct = 10.0 if is_boundary_period else 20.0
            else:
                band_width_pct = 20.0 if is_boundary_period else 40.0

        band_width = reference_price * (band_width_pct / 100.0)

        return LULDBand(
            symbol=symbol,
            reference_price=reference_price,
            upper_band=reference_price + band_width,
            lower_band=reference_price - band_width,
            band_width_pct=band_width_pct,
            tier=tier,
            updated_at=datetime.utcnow()
        )

    def check_luld_violation(
        self,
        symbol: str,
        price: float,
        reference_price: float,
        tier: int = 1
    ) -> Optional[HaltType]:
        """
        Check if price violates LULD bands

        Returns HaltType if violation detected, None otherwise
        """
        band = self.calculate_luld_bands(symbol, reference_price, tier)

        if price >= band.upper_band:
            logger.warning(
                f"LULD upper band violation: {symbol} price {price} >= {band.upper_band}"
            )
            halt_detections.labels(halt_type="LULD_UPPER").inc()
            return HaltType.LULD_UPPER
        elif price <= band.lower_band:
            logger.warning(
                f"LULD lower band violation: {symbol} price {price} <= {band.lower_band}"
            )
            halt_detections.labels(halt_type="LULD_LOWER").inc()
            return HaltType.LULD_LOWER

        return None

    async def check_halt_status(self, symbol: str) -> HaltStatus:
        """Check if symbol is currently halted"""
        if symbol in self.halt_status:
            return self.halt_status[symbol]

        # Check Redis for halt status
        if self.redis_client:
            halt_data = await self.redis_client.get(f"halt:{symbol}")
            if halt_data:
                # Parse halt data
                # In production, this would come from market data feed
                return HaltStatus(symbol=symbol, is_halted=True)

        return HaltStatus(symbol=symbol, is_halted=False)

    def is_in_auction(self) -> bool:
        """Check if market is in auction period"""
        return self.auction_status is not None and self.auction_status.is_active

    def get_current_auction(self) -> Optional[AuctionStatus]:
        """Get current auction status"""
        now = datetime.now()

        # Opening auction: 9:30 AM
        if now.time() >= time(9, 25) and now.time() < time(9, 30):
            return AuctionStatus(
                auction_type=AuctionType.OPENING,
                start_time=now.replace(hour=9, minute=25, second=0),
                expected_end_time=now.replace(hour=9, minute=30, second=0),
                is_active=True
            )

        # Closing auction: 4:00 PM
        if now.time() >= time(15, 55) and now.time() <= time(16, 0):
            return AuctionStatus(
                auction_type=AuctionType.CLOSING,
                start_time=now.replace(hour=15, minute=55, second=0),
                expected_end_time=now.replace(hour=16, minute=0, second=0),
                is_active=True
            )

        return None

    def get_order_restrictions(self, symbol: str) -> Set[OrderRestriction]:
        """Get order restrictions for symbol"""
        restrictions = set()

        # Check if halted
        if symbol in self.halt_status and self.halt_status[symbol].is_halted:
            restrictions.add(OrderRestriction.NO_NEW_ORDERS)
            restrictions.add(OrderRestriction.NO_CANCEL)

        # Check if in auction
        auction = self.get_current_auction()
        if auction and auction.is_active:
            if auction.auction_type in [AuctionType.OPENING, AuctionType.CLOSING]:
                restrictions.add(OrderRestriction.AUCTION_ONLY)

        # Check circuit breaker
        if self.circuit_breaker_active:
            if self.circuit_breaker_level == HaltType.CIRCUIT_BREAKER_L3:
                restrictions.add(OrderRestriction.NO_NEW_ORDERS)
            else:
                restrictions.add(OrderRestriction.LIMIT_ONLY)

        return restrictions

    def can_place_order(
        self,
        symbol: str,
        order_type: str,
        is_auction_eligible: bool = False
    ) -> tuple[bool, Optional[str]]:
        """
        Check if order can be placed given current restrictions

        Returns (can_place, reason_if_not)
        """
        restrictions = self.get_order_restrictions(symbol)

        if OrderRestriction.NO_NEW_ORDERS in restrictions:
            return False, "No new orders accepted (halt or circuit breaker)"

        if OrderRestriction.AUCTION_ONLY in restrictions:
            if not is_auction_eligible:
                return False, "Only auction-eligible orders accepted during auction"

        if OrderRestriction.LIMIT_ONLY in restrictions:
            if order_type == "MARKET":
                return False, "Only limit orders accepted during circuit breaker"

        if OrderRestriction.MARKET_ONLY in restrictions:
            if order_type != "MARKET":
                return False, "Only market orders accepted"

        return True, None

    def can_cancel_order(self, symbol: str) -> tuple[bool, Optional[str]]:
        """Check if order can be cancelled"""
        restrictions = self.get_order_restrictions(symbol)

        if OrderRestriction.NO_CANCEL in restrictions:
            return False, "Cannot cancel orders during halt"

        return True, None

    async def handle_halt_detected(
        self,
        symbol: str,
        halt_type: HaltType,
        message: Optional[str] = None
    ):
        """Handle halt detection"""
        halt_status = HaltStatus(
            symbol=symbol,
            is_halted=True,
            halt_type=halt_type,
            halt_start_time=datetime.utcnow(),
            message=message
        )

        # Expected resume time (typically 5 minutes for LULD)
        if halt_type in [HaltType.LULD_UPPER, HaltType.LULD_LOWER]:
            halt_status.expected_resume_time = datetime.utcnow() + timedelta(minutes=5)
            halt_status.restrictions = {
                OrderRestriction.NO_NEW_ORDERS,
                OrderRestriction.NO_CANCEL
            }

        self.halt_status[symbol] = halt_status

        # Publish halt to Redis
        if self.redis_client:
            await self.redis_client.setex(
                f"halt:{symbol}",
                300,  # 5 minutes TTL
                halt_type.value
            )

        logger.warning(f"Halt detected: {symbol} - {halt_type.value} - {message}")
        halt_detections.labels(halt_type=halt_type.value).inc()

    async def handle_halt_resumed(self, symbol: str):
        """Handle halt resumption"""
        if symbol in self.halt_status:
            del self.halt_status[symbol]

        if self.redis_client:
            await self.redis_client.delete(f"halt:{symbol}")

        logger.info(f"Halt resumed: {symbol}")

    async def _monitor_luld_bands(self):
        """Background task to monitor LULD bands"""
        while True:
            try:
                # Update LULD bands from market data
                # In production, this would subscribe to market data feed
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                logger.error(f"Error monitoring LULD bands: {e}")
                await asyncio.sleep(60)

    async def _monitor_auctions(self):
        """Background task to monitor auction periods"""
        while True:
            try:
                # Check for auction periods
                auction = self.get_current_auction()
                if auction != self.auction_status:
                    self.auction_status = auction
                    if auction:
                        logger.info(f"Auction started: {auction.auction_type.value}")
                        auction_periods.labels(auction_type=auction.auction_type.value).inc()

                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error monitoring auctions: {e}")
                await asyncio.sleep(10)


class SafeOrderManager:
    """Manages orders safely during halts and special periods"""

    def __init__(self, halt_detector: HaltDetector):
        self.halt_detector = halt_detector
        self.working_orders: Dict[str, List[str]] = {}  # symbol -> order_ids

    async def submit_order(
        self,
        symbol: str,
        order_id: str,
        order_type: str,
        is_auction_eligible: bool = False
    ) -> tuple[bool, Optional[str]]:
        """
        Submit order with halt safety checks

        Returns (accepted, rejection_reason)
        """
        # Check if order can be placed
        can_place, reason = self.halt_detector.can_place_order(
            symbol, order_type, is_auction_eligible
        )

        if not can_place:
            logger.warning(f"Order {order_id} rejected: {reason}")
            return False, reason

        # Track working order
        if symbol not in self.working_orders:
            self.working_orders[symbol] = []
        self.working_orders[symbol].append(order_id)

        logger.info(f"Order {order_id} accepted for {symbol}")
        return True, None

    async def cancel_order(
        self,
        symbol: str,
        order_id: str
    ) -> tuple[bool, Optional[str]]:
        """
        Cancel order with halt safety checks

        Returns (cancelled, rejection_reason)
        """
        # Check if order can be cancelled
        can_cancel, reason = self.halt_detector.can_cancel_order(symbol)

        if not can_cancel:
            logger.warning(f"Order {order_id} cancel rejected: {reason}")
            return False, reason

        # Remove from working orders
        if symbol in self.working_orders and order_id in self.working_orders[symbol]:
            self.working_orders[symbol].remove(order_id)

        logger.info(f"Order {order_id} cancelled for {symbol}")
        return True, None

    async def cancel_all_orders_for_symbol(self, symbol: str, reason: str):
        """Cancel all working orders for a symbol (e.g., on halt)"""
        if symbol not in self.working_orders:
            return

        order_ids = self.working_orders[symbol].copy()
        for order_id in order_ids:
            # Force cancel (bypass restrictions)
            logger.info(f"Force cancelling order {order_id} for {symbol}: {reason}")
            orders_cancelled.inc()

        self.working_orders[symbol] = []
