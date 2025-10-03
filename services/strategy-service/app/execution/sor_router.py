"""
Smart Order Router (SOR)
Routes orders to optimal venues based on spread, latency, fees, and fill probability

Acceptance Criteria:
- Slippage reduction ≥10% vs baseline (30-day average)
- SOR decision latency p99 < 10ms
- Routing accuracy: ≥95% select optimal venue in hindsight analysis
"""
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from pathlib import Path
import yaml
from enum import Enum

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    IOC = "IOC"
    POST_ONLY = "POST_ONLY"
    FOK = "FOK"


@dataclass
class Order:
    """Order representation"""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    strategy_id: Optional[str] = None
    urgency: str = "normal"  # low, normal, high


@dataclass
class VenueProfile:
    """Venue characteristics"""
    name: str
    venue_id: str
    latency_p50_ms: float
    latency_p99_ms: float
    maker_fee_bps: float
    taker_fee_bps: float
    min_tick_size: float
    min_order_size: int
    max_order_size: int
    supports_ioc: bool
    supports_post_only: bool
    supports_fok: bool
    historical_fill_rate: float
    avg_spread_bps: float
    liquidity_score: float
    priority: int
    enabled: bool = True


@dataclass
class MarketData:
    """Market data snapshot"""
    symbol: str
    bid: float
    ask: float
    last: float
    bid_size: int
    ask_size: int
    spread_bps: float
    timestamp: datetime


@dataclass
class RoutingDecision:
    """SOR routing decision"""
    venue: VenueProfile
    score: float
    decision_time_ms: float
    reasoning: Dict[str, float]
    alternatives: List[Tuple[str, float]]
    timestamp: datetime


class SmartOrderRouter:
    """
    Smart Order Router
    Selects optimal venue for order execution
    """

    def __init__(self, config_path: Optional[Path] = None):
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "venue_profiles.yaml"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Parse venue profiles
        self.venues = self._parse_venues(config['venues'])
        self.routing_config = config['routing_config']

        # Routing weights
        self.weights = self.routing_config['weights']

        # Symbol overrides
        self.symbol_overrides = self._parse_symbol_overrides(
            self.routing_config.get('symbol_overrides', {})
        )

        # Time-based routing
        self.time_based_routing = self.routing_config.get('time_based_routing', {})

        # Performance tracking
        self.routing_history = []
        self.performance_stats = {
            'total_routes': 0,
            'avg_decision_time_ms': 0.0,
            'venue_selections': {}
        }

        logger.info(f"SOR initialized with {len(self.venues)} venues")

    def _parse_venues(self, venues_config: List[Dict]) -> Dict[str, VenueProfile]:
        """Parse venue configurations"""
        venues = {}

        for venue_cfg in venues_config:
            venue = VenueProfile(
                name=venue_cfg['name'],
                venue_id=venue_cfg['venue_id'],
                latency_p50_ms=venue_cfg['latency_p50_ms'],
                latency_p99_ms=venue_cfg['latency_p99_ms'],
                maker_fee_bps=venue_cfg['maker_fee_bps'],
                taker_fee_bps=venue_cfg['taker_fee_bps'],
                min_tick_size=venue_cfg['min_tick_size'],
                min_order_size=venue_cfg['min_order_size'],
                max_order_size=venue_cfg['max_order_size'],
                supports_ioc=venue_cfg['supports_ioc'],
                supports_post_only=venue_cfg['supports_post_only'],
                supports_fok=venue_cfg['supports_fok'],
                historical_fill_rate=venue_cfg['historical_fill_rate'],
                avg_spread_bps=venue_cfg['avg_spread_bps'],
                liquidity_score=venue_cfg['liquidity_score'],
                priority=venue_cfg['priority']
            )
            venues[venue.name] = venue

        return venues

    def _parse_symbol_overrides(self, overrides_config: Dict) -> Dict[str, List[str]]:
        """Parse symbol-specific venue preferences"""
        symbol_map = {}

        for category, config in overrides_config.items():
            symbols = config.get('symbols', [])
            preferred_venues = config.get('preferred_venues', [])

            for symbol in symbols:
                symbol_map[symbol] = preferred_venues

        return symbol_map

    def route_order(
        self,
        order: Order,
        market_data: MarketData
    ) -> RoutingDecision:
        """
        Route order to optimal venue

        Args:
            order: Order to route
            market_data: Current market data

        Returns:
            RoutingDecision with selected venue
        """
        start_time = time.time()

        # Filter eligible venues
        eligible_venues = self._filter_eligible_venues(order, market_data)

        if not eligible_venues:
            raise ValueError(f"No eligible venues for order: {order}")

        # Get current time-based weights
        current_weights = self._get_time_based_weights()

        # Score each venue
        scores = []
        for venue in eligible_venues:
            score = self._calculate_venue_score(
                venue,
                order,
                market_data,
                current_weights
            )
            scores.append((venue, score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Select best venue
        best_venue, best_score = scores[0]

        # Calculate decision time
        decision_time_ms = (time.time() - start_time) * 1000

        # Build reasoning
        reasoning = self._build_reasoning(best_venue, order, market_data, current_weights)

        # Create decision
        decision = RoutingDecision(
            venue=best_venue,
            score=best_score,
            decision_time_ms=decision_time_ms,
            reasoning=reasoning,
            alternatives=[(v.name, s) for v, s in scores[1:4]],  # Top 3 alternatives
            timestamp=datetime.utcnow()
        )

        # Update stats
        self._update_stats(decision)

        # Assert latency budget
        if decision_time_ms > self.routing_config['latency_budget_ms']:
            logger.warning(
                f"SOR decision exceeded latency budget: {decision_time_ms:.2f}ms "
                f"> {self.routing_config['latency_budget_ms']}ms"
            )

        logger.info(
            f"Routed {order.symbol} ({order.quantity} shares) to {best_venue.name} "
            f"(score: {best_score:.3f}, decision time: {decision_time_ms:.2f}ms)"
        )

        return decision

    def _filter_eligible_venues(
        self,
        order: Order,
        market_data: MarketData
    ) -> List[VenueProfile]:
        """Filter venues that can handle this order"""
        eligible = []

        for venue in self.venues.values():
            # Check if venue is enabled
            if not venue.enabled:
                continue

            # Check order size limits
            if order.quantity < venue.min_order_size:
                continue

            if order.quantity > venue.max_order_size:
                continue

            # Check order type support
            if order.order_type == OrderType.IOC and not venue.supports_ioc:
                continue

            if order.order_type == OrderType.POST_ONLY and not venue.supports_post_only:
                continue

            if order.order_type == OrderType.FOK and not venue.supports_fok:
                continue

            eligible.append(venue)

        return eligible

    def _get_time_based_weights(self) -> Dict[str, float]:
        """Get weights based on current time"""
        now = datetime.now().time()

        for period_name, period_config in self.time_based_routing.items():
            if period_name in ['market_open', 'market_close', 'normal']:
                start = dt_time.fromisoformat(period_config['start'])
                end = dt_time.fromisoformat(period_config['end'])

                if start <= now <= end:
                    # Return custom weights for this period
                    custom_weights = {}

                    for key in ['spread', 'latency', 'fees', 'fill_probability']:
                        weight_key = f'{key}_weight'
                        if weight_key in period_config:
                            custom_weights[key] = period_config[weight_key]
                        else:
                            custom_weights[key] = self.weights[key]

                    # Normalize weights
                    total = sum(custom_weights.values())
                    return {k: v/total for k, v in custom_weights.items()}

        # Return default weights
        return self.weights

    def _calculate_venue_score(
        self,
        venue: VenueProfile,
        order: Order,
        market_data: MarketData,
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate venue score

        Args:
            venue: Venue to score
            order: Order being routed
            market_data: Market data
            weights: Scoring weights

        Returns:
            Score (0-1, higher is better)
        """
        # 1. Spread score (lower spread is better)
        spread_score = 1.0 - min(venue.avg_spread_bps / 10.0, 1.0)

        # 2. Latency score (lower latency is better)
        latency_score = 1.0 - min(venue.latency_p50_ms / 100.0, 1.0)

        # 3. Fee score (lower fees are better, rebates are best)
        fee = venue.maker_fee_bps if order.order_type == OrderType.POST_ONLY else venue.taker_fee_bps
        fee_score = 1.0 - min((fee + 1.0) / 3.0, 1.0)  # Normalize -0.9 to 2.0 range

        # 4. Fill probability
        fill_prob_score = venue.historical_fill_rate

        # Weighted total
        total_score = (
            weights['spread'] * spread_score +
            weights['latency'] * latency_score +
            weights['fees'] * fee_score +
            weights['fill_probability'] * fill_prob_score
        )

        # Symbol-specific boost
        if order.symbol in self.symbol_overrides:
            preferred_venues = self.symbol_overrides[order.symbol]
            if venue.name in preferred_venues:
                total_score *= 1.1  # 10% boost for preferred venues

        # Priority boost
        priority_boost = (4 - venue.priority) * 0.05  # Higher priority = boost
        total_score += priority_boost

        return min(total_score, 1.0)

    def _build_reasoning(
        self,
        venue: VenueProfile,
        order: Order,
        market_data: MarketData,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Build reasoning for venue selection"""
        return {
            "spread_bps": venue.avg_spread_bps,
            "latency_p50_ms": venue.latency_p50_ms,
            "fee_bps": venue.maker_fee_bps if order.order_type == OrderType.POST_ONLY else venue.taker_fee_bps,
            "fill_rate": venue.historical_fill_rate,
            "liquidity_score": venue.liquidity_score,
            "weights_used": weights
        }

    def _update_stats(self, decision: RoutingDecision):
        """Update performance statistics"""
        self.performance_stats['total_routes'] += 1

        # Update average decision time
        n = self.performance_stats['total_routes']
        old_avg = self.performance_stats['avg_decision_time_ms']
        new_avg = (old_avg * (n-1) + decision.decision_time_ms) / n
        self.performance_stats['avg_decision_time_ms'] = new_avg

        # Track venue selections
        venue_name = decision.venue.name
        if venue_name not in self.performance_stats['venue_selections']:
            self.performance_stats['venue_selections'][venue_name] = 0
        self.performance_stats['venue_selections'][venue_name] += 1

        # Store in history (keep last 1000)
        self.routing_history.append(decision)
        if len(self.routing_history) > 1000:
            self.routing_history.pop(0)

    def get_performance_stats(self) -> Dict:
        """Get routing performance statistics"""
        return {
            **self.performance_stats,
            "p99_decision_time_ms": self._calculate_p99_latency(),
            "venue_distribution": self._get_venue_distribution()
        }

    def _calculate_p99_latency(self) -> float:
        """Calculate p99 decision latency"""
        if not self.routing_history:
            return 0.0

        latencies = [d.decision_time_ms for d in self.routing_history]
        latencies.sort()

        p99_idx = int(len(latencies) * 0.99)
        return latencies[p99_idx] if p99_idx < len(latencies) else latencies[-1]

    def _get_venue_distribution(self) -> Dict[str, float]:
        """Get distribution of venue selections"""
        total = self.performance_stats['total_routes']

        if total == 0:
            return {}

        return {
            venue: count / total
            for venue, count in self.performance_stats['venue_selections'].items()
        }


if __name__ == "__main__":
    """Example usage"""
    logging.basicConfig(level=logging.INFO)

    # Create router
    router = SmartOrderRouter()

    # Create sample order
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=500,
        order_type=OrderType.LIMIT,
        limit_price=185.50
    )

    # Create sample market data
    market_data = MarketData(
        symbol="AAPL",
        bid=185.48,
        ask=185.52,
        last=185.50,
        bid_size=1000,
        ask_size=800,
        spread_bps=2.16,
        timestamp=datetime.utcnow()
    )

    # Route order
    decision = router.route_order(order, market_data)

    print(f"\n{'='*60}")
    print("ROUTING DECISION")
    print(f"{'='*60}")
    print(f"Selected Venue: {decision.venue.name}")
    print(f"Score: {decision.score:.3f}")
    print(f"Decision Time: {decision.decision_time_ms:.2f}ms")
    print(f"\nReasoning:")
    for key, value in decision.reasoning.items():
        print(f"  {key}: {value}")
    print(f"\nAlternatives:")
    for venue_name, score in decision.alternatives:
        print(f"  {venue_name}: {score:.3f}")

    # Get performance stats
    stats = router.get_performance_stats()
    print(f"\n{'='*60}")
    print("PERFORMANCE STATS")
    print(f"{'='*60}")
    print(f"Total Routes: {stats['total_routes']}")
    print(f"Avg Decision Time: {stats['avg_decision_time_ms']:.2f}ms")
    print(f"P99 Decision Time: {stats['p99_decision_time_ms']:.2f}ms")
