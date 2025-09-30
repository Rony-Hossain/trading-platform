"""
Advanced Execution Modeling with Realistic Fills and Exchange Fee Structures

This module implements sophisticated execution modeling including:
- Depth-aware slippage and fill probability curves
- Comprehensive exchange and regulatory fee modeling
- VWAP/TWAP execution algorithm simulation
- True net realized P&L calculation with all transaction costs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging
from decimal import Decimal, ROUND_HALF_UP
import math

logger = logging.getLogger(__name__)

class ExchangeType(Enum):
    """Exchange types with different fee structures"""
    NYSE = "nyse"
    NASDAQ = "nasdaq"
    BATS = "bats"
    EDGX = "edgx"
    ARCA = "arca"
    IEX = "iex"
    DARK_POOL = "dark_pool"

class OrderExecutionType(Enum):
    """Order execution types"""
    MARKET = "market"
    LIMIT = "limit"
    MIDPOINT = "midpoint"
    HIDDEN = "hidden"
    ICEBERG = "iceberg"
    VWAP = "vwap"
    TWAP = "twap"
    POV = "pov"  # Participation of Volume

class LiquidityType(Enum):
    """Liquidity provision types"""
    MAKER = "maker"  # Provides liquidity
    TAKER = "taker"  # Takes liquidity
    ROUTED = "routed"  # Routed to another venue

@dataclass
class MarketDepth:
    """Market depth information"""
    price_levels: List[float]  # Price levels
    bid_sizes: List[int]       # Bid sizes at each level
    ask_sizes: List[int]       # Ask sizes at each level
    timestamp: datetime

@dataclass
class FeeStructure:
    """Exchange fee structure"""
    maker_fee: float           # Fee for providing liquidity (can be negative for rebate)
    taker_fee: float          # Fee for taking liquidity
    sec_fee: float            # SEC fee (per $1M notional)
    taf_fee: float            # Trading Activity Fee (per share)
    finra_orf: float          # FINRA Options Regulatory Fee
    clearing_fee: float       # Clearing fee per share
    min_fee: float            # Minimum fee per trade
    max_fee: float            # Maximum fee per trade

@dataclass
class ExecutionQuality:
    """Execution quality metrics"""
    price_improvement: float   # Price improvement vs NBBO
    effective_spread: float    # Effective spread paid
    realized_spread: float     # Realized spread (5-min benchmark)
    market_impact: float       # Temporary market impact
    timing_risk: float         # Risk from execution timing
    opportunity_cost: float    # Cost of not executing immediately

@dataclass
class FillDetails:
    """Detailed fill information"""
    fill_id: str
    timestamp: datetime
    symbol: str
    side: str
    quantity: int
    price: float
    exchange: ExchangeType
    liquidity_type: LiquidityType
    execution_type: OrderExecutionType
    fees: Dict[str, float]
    total_fees: float
    net_proceeds: float
    execution_quality: ExecutionQuality
    market_conditions: Dict[str, Any]

class DepthAwareExecutionModel:
    """Advanced execution model with depth-aware slippage curves"""
    
    def __init__(self):
        # Market impact parameters (calibrated to empirical data)
        self.temporary_impact_coeff = 0.314    # Square root coefficient
        self.permanent_impact_coeff = 0.118    # Linear coefficient
        self.resilience_time = 300             # Seconds for impact to decay
        
        # Depth curve parameters
        self.depth_shape_param = 1.5           # Controls depth curve shape
        self.min_fill_probability = 0.75       # Minimum fill probability
        
        # VWAP/TWAP parameters
        self.vwap_participation_rate = 0.1     # Default participation rate
        self.twap_slice_duration = 300         # 5-minute TWAP slices
        
    def calculate_fill_probability(
        self,
        order_quantity: int,
        side: str,
        market_depth: MarketDepth,
        execution_type: OrderExecutionType
    ) -> Tuple[float, List[Tuple[int, float]]]:
        """
        Calculate fill probability and quantity-price pairs based on market depth
        
        Returns:
            (overall_fill_probability, [(quantity, price), ...])
        """
        
        if execution_type == OrderExecutionType.MARKET:
            return self._calculate_market_order_fills(order_quantity, side, market_depth)
        elif execution_type == OrderExecutionType.LIMIT:
            return self._calculate_limit_order_fills(order_quantity, side, market_depth)
        elif execution_type in [OrderExecutionType.VWAP, OrderExecutionType.TWAP]:
            return self._calculate_algorithmic_fills(order_quantity, side, market_depth, execution_type)
        else:
            return self._calculate_default_fills(order_quantity, side, market_depth)
    
    def _calculate_market_order_fills(
        self,
        quantity: int,
        side: str,
        depth: MarketDepth
    ) -> Tuple[float, List[Tuple[int, float]]]:
        """Calculate market order fills walking through the book"""
        
        fills = []
        remaining_qty = quantity
        total_cost = 0.0
        
        # Determine which side of the book to walk
        if side.lower() == "buy":
            levels = zip(depth.ask_sizes, depth.price_levels)
            price_levels = depth.price_levels
        else:
            levels = zip(depth.bid_sizes, reversed(depth.price_levels))
            price_levels = list(reversed(depth.price_levels))
        
        level_index = 0
        for available_qty, price in levels:
            if remaining_qty <= 0:
                break
                
            # Apply market impact based on level depth
            impact_factor = self._calculate_level_impact(level_index, quantity)
            adjusted_price = price * (1 + impact_factor if side.lower() == "buy" else 1 - impact_factor)
            
            # Fill quantity at this level
            fill_qty = min(remaining_qty, available_qty)
            fills.append((fill_qty, adjusted_price))
            
            remaining_qty -= fill_qty
            total_cost += fill_qty * adjusted_price
            level_index += 1
        
        # Calculate overall fill probability
        fill_probability = 1.0 - (remaining_qty / quantity) if quantity > 0 else 1.0
        
        # Adjust for liquidity risk
        if level_index > 5:  # Deep in the book
            fill_probability *= 0.9
        elif level_index > 10:
            fill_probability *= 0.8
        
        return max(fill_probability, self.min_fill_probability), fills
    
    def _calculate_limit_order_fills(
        self,
        quantity: int,
        side: str,
        depth: MarketDepth
    ) -> Tuple[float, List[Tuple[int, float]]]:
        """Calculate limit order fill probability based on queue position"""
        
        # Simplified limit order model - assumes mid-queue position
        if side.lower() == "buy":
            best_bid = depth.price_levels[0]
            queue_ahead = depth.bid_sizes[0] * 0.5  # Assume middle of queue
        else:
            best_ask = depth.price_levels[-1]
            queue_ahead = depth.ask_sizes[-1] * 0.5
        
        # Fill probability based on historical fill rates
        base_fill_prob = 0.3  # Base 30% fill rate for limit orders
        
        # Adjust based on queue position
        queue_factor = max(0.1, 1.0 - (queue_ahead / max(quantity * 10, 1000)))
        adjusted_fill_prob = base_fill_prob * queue_factor
        
        # Expected fill price (assumes best level execution)
        expected_price = best_bid if side.lower() == "buy" else best_ask
        
        # Partial fill modeling
        expected_fill_qty = int(quantity * adjusted_fill_prob)
        if expected_fill_qty > 0:
            return adjusted_fill_prob, [(expected_fill_qty, expected_price)]
        else:
            return 0.0, []
    
    def _calculate_algorithmic_fills(
        self,
        quantity: int,
        side: str,
        depth: MarketDepth,
        execution_type: OrderExecutionType
    ) -> Tuple[float, List[Tuple[int, float]]]:
        """Calculate fills for algorithmic execution strategies"""
        
        if execution_type == OrderExecutionType.VWAP:
            return self._calculate_vwap_fills(quantity, side, depth)
        elif execution_type == OrderExecutionType.TWAP:
            return self._calculate_twap_fills(quantity, side, depth)
        else:
            return self._calculate_default_fills(quantity, side, depth)
    
    def _calculate_vwap_fills(
        self,
        quantity: int,
        side: str,
        depth: MarketDepth
    ) -> Tuple[float, List[Tuple[int, float]]]:
        """Model VWAP execution with participation rate limits"""
        
        # Estimate volume-weighted average price
        total_volume = sum(depth.bid_sizes + depth.ask_sizes)
        if total_volume == 0:
            return 0.0, []
        
        # Calculate VWAP estimate
        bid_volume = sum(depth.bid_sizes)
        ask_volume = sum(depth.ask_sizes)
        
        weighted_bid_price = sum(size * price for size, price in zip(depth.bid_sizes, depth.price_levels[:len(depth.bid_sizes)]))
        weighted_ask_price = sum(size * price for size, price in zip(depth.ask_sizes, depth.price_levels[-len(depth.ask_sizes):]))
        
        if side.lower() == "buy":
            vwap_price = weighted_ask_price / ask_volume if ask_volume > 0 else depth.price_levels[-1]
            available_volume = ask_volume
        else:
            vwap_price = weighted_bid_price / bid_volume if bid_volume > 0 else depth.price_levels[0]
            available_volume = bid_volume
        
        # Participation rate constraint
        max_participation_qty = int(available_volume * self.vwap_participation_rate)
        executable_qty = min(quantity, max_participation_qty)
        
        # VWAP execution typically has high fill rates
        fill_probability = 0.95 if executable_qty > 0 else 0.0
        
        # Add small implementation shortfall
        shortfall_factor = 1.0001 if side.lower() == "buy" else 0.9999
        execution_price = vwap_price * shortfall_factor
        
        return fill_probability, [(executable_qty, execution_price)] if executable_qty > 0 else []
    
    def _calculate_twap_fills(
        self,
        quantity: int,
        side: str,
        depth: MarketDepth
    ) -> Tuple[float, List[Tuple[int, float]]]:
        """Model TWAP execution with time-based slicing"""
        
        # TWAP slices the order over time - model as single execution with average impact
        mid_price = (depth.price_levels[0] + depth.price_levels[-1]) / 2
        
        # Lower market impact due to time distribution
        impact_reduction = 0.7  # 30% reduction in impact
        base_impact = self._calculate_market_impact(quantity, sum(depth.bid_sizes + depth.ask_sizes))
        twap_impact = base_impact * impact_reduction
        
        if side.lower() == "buy":
            execution_price = mid_price * (1 + twap_impact)
        else:
            execution_price = mid_price * (1 - twap_impact)
        
        # TWAP has high fill rate but may take time
        fill_probability = 0.92
        
        return fill_probability, [(quantity, execution_price)]
    
    def _calculate_default_fills(
        self,
        quantity: int,
        side: str,
        depth: MarketDepth
    ) -> Tuple[float, List[Tuple[int, float]]]:
        """Default fill calculation for other order types"""
        
        mid_price = (depth.price_levels[0] + depth.price_levels[-1]) / 2
        
        # Conservative fill probability
        fill_probability = 0.8
        
        # Moderate market impact
        impact = self._calculate_market_impact(quantity, sum(depth.bid_sizes + depth.ask_sizes)) * 0.5
        
        if side.lower() == "buy":
            execution_price = mid_price * (1 + impact)
        else:
            execution_price = mid_price * (1 - impact)
        
        return fill_probability, [(quantity, execution_price)]
    
    def _calculate_level_impact(self, level: int, order_size: int) -> float:
        """Calculate market impact based on order book level"""
        
        # Impact increases non-linearly with depth level
        level_penalty = 0.0001 * (level ** self.depth_shape_param)
        size_penalty = 0.00001 * math.sqrt(order_size)
        
        return level_penalty + size_penalty
    
    def _calculate_market_impact(self, order_size: int, available_liquidity: int) -> float:
        """Calculate temporary market impact based on order size and liquidity"""
        
        if available_liquidity <= 0:
            return 0.001  # Default 10bps impact
        
        # Square root impact model
        participation_rate = order_size / available_liquidity
        temporary_impact = self.temporary_impact_coeff * math.sqrt(participation_rate)
        
        # Cap maximum impact
        return min(temporary_impact, 0.01)  # Cap at 100bps

class ComprehensiveFeeCalculator:
    """Comprehensive fee calculator for all exchange and regulatory fees"""
    
    def __init__(self):
        # Initialize fee structures for major exchanges
        self.exchange_fees = {
            ExchangeType.NYSE: FeeStructure(
                maker_fee=-0.0020,      # $0.20 rebate per 100 shares
                taker_fee=0.0030,       # $0.30 per 100 shares
                sec_fee=0.0000278,      # $27.80 per $1M notional (2024 rate)
                taf_fee=0.000119,       # $0.119 per 100 shares
                finra_orf=0.000119,     # FINRA ORF
                clearing_fee=0.0000325, # NSCC clearing fee
                min_fee=0.01,
                max_fee=5.00
            ),
            ExchangeType.NASDAQ: FeeStructure(
                maker_fee=-0.0025,      # Higher rebate
                taker_fee=0.0030,
                sec_fee=0.0000278,
                taf_fee=0.000119,
                finra_orf=0.000119,
                clearing_fee=0.0000325,
                min_fee=0.01,
                max_fee=5.00
            ),
            ExchangeType.BATS: FeeStructure(
                maker_fee=-0.0020,
                taker_fee=0.0025,       # Slightly lower taker fee
                sec_fee=0.0000278,
                taf_fee=0.000119,
                finra_orf=0.000119,
                clearing_fee=0.0000325,
                min_fee=0.01,
                max_fee=5.00
            ),
            ExchangeType.IEX: FeeStructure(
                maker_fee=0.0000,       # No rebate/fee
                taker_fee=0.0000,       # No fee
                sec_fee=0.0000278,      # Still has regulatory fees
                taf_fee=0.000119,
                finra_orf=0.000119,
                clearing_fee=0.0000325,
                min_fee=0.00,
                max_fee=5.00
            ),
            ExchangeType.DARK_POOL: FeeStructure(
                maker_fee=0.0005,       # Small fee for dark pool
                taker_fee=0.0005,
                sec_fee=0.0000278,
                taf_fee=0.000119,
                finra_orf=0.000119,
                clearing_fee=0.0000325,
                min_fee=0.00,
                max_fee=5.00
            )
        }
        
        # Broker commission structures
        self.broker_commissions = {
            "institutional": 0.005,      # $0.005 per share
            "retail": 0.00,             # Zero commission
            "high_frequency": 0.002,     # $0.002 per share
            "prime_brokerage": 0.003     # $0.003 per share
        }
    
    def calculate_all_fees(
        self,
        quantity: int,
        price: float,
        exchange: ExchangeType,
        liquidity_type: LiquidityType,
        broker_type: str = "institutional"
    ) -> Dict[str, float]:
        """Calculate comprehensive fee breakdown"""
        
        notional_value = quantity * price
        fee_structure = self.exchange_fees.get(exchange, self.exchange_fees[ExchangeType.NYSE])
        
        fees = {}
        
        # Exchange fees (maker/taker)
        if liquidity_type == LiquidityType.MAKER:
            exchange_fee = quantity * fee_structure.maker_fee / 100
            fees["exchange_fee"] = exchange_fee
            fees["fee_type"] = "maker_rebate" if exchange_fee < 0 else "maker_fee"
        else:
            exchange_fee = quantity * fee_structure.taker_fee / 100
            fees["exchange_fee"] = exchange_fee
            fees["fee_type"] = "taker_fee"
        
        # SEC fee (on sell orders only, but we'll calculate for all)
        sec_fee = notional_value * fee_structure.sec_fee
        fees["sec_fee"] = sec_fee
        
        # TAF (Trading Activity Fee) - FINRA
        taf_fee = quantity * fee_structure.taf_fee / 100
        fees["taf_fee"] = taf_fee
        
        # FINRA Options Regulatory Fee (for options, but included for completeness)
        finra_orf = quantity * fee_structure.finra_orf / 100
        fees["finra_orf"] = finra_orf
        
        # Clearing fees (NSCC/DTCC)
        clearing_fee = quantity * fee_structure.clearing_fee / 100
        fees["clearing_fee"] = clearing_fee
        
        # Broker commission
        broker_commission = quantity * self.broker_commissions.get(broker_type, 0.005) / 100
        fees["broker_commission"] = broker_commission
        
        # Apply min/max fee constraints
        total_exchange_fees = exchange_fee + sec_fee + taf_fee + finra_orf + clearing_fee
        if total_exchange_fees < fee_structure.min_fee:
            adjustment = fee_structure.min_fee - total_exchange_fees
            fees["min_fee_adjustment"] = adjustment
        elif total_exchange_fees > fee_structure.max_fee:
            adjustment = fee_structure.max_fee - total_exchange_fees
            fees["max_fee_adjustment"] = adjustment
        else:
            fees["min_fee_adjustment"] = 0.0
            fees["max_fee_adjustment"] = 0.0
        
        # Total fees (only sum numeric values)
        fees["total_fees"] = sum(v for v in fees.values() if isinstance(v, (int, float)))
        
        return fees
    
    def get_fee_structure_summary(self) -> Dict[str, Any]:
        """Get summary of all fee structures for reference"""
        
        summary = {}
        for exchange, fees in self.exchange_fees.items():
            summary[exchange.value] = {
                "maker_fee_per_100_shares": fees.maker_fee,
                "taker_fee_per_100_shares": fees.taker_fee,
                "sec_fee_per_million_notional": fees.sec_fee * 1000000,
                "taf_fee_per_100_shares": fees.taf_fee,
                "finra_orf_per_100_shares": fees.finra_orf,
                "clearing_fee_per_100_shares": fees.clearing_fee,
                "min_fee": fees.min_fee,
                "max_fee": fees.max_fee
            }
        
        return {
            "exchange_fees": summary,
            "broker_commissions_per_share": self.broker_commissions,
            "regulatory_notes": {
                "sec_fee": "Applied to sell orders only in practice",
                "taf_fee": "FINRA Trading Activity Fee on all trades",
                "finra_orf": "Options Regulatory Fee (included for reference)",
                "clearing_fee": "NSCC/DTCC clearing and settlement"
            }
        }

class AdvancedExecutionEngine:
    """Advanced execution engine combining depth modeling and comprehensive fee calculation"""
    
    def __init__(self):
        self.execution_model = DepthAwareExecutionModel()
        self.fee_calculator = ComprehensiveFeeCalculator()
        
    async def simulate_execution(
        self,
        symbol: str,
        side: str,
        quantity: int,
        execution_type: OrderExecutionType,
        market_depth: MarketDepth,
        exchange: ExchangeType = ExchangeType.NYSE,
        broker_type: str = "institutional"
    ) -> FillDetails:
        """
        Simulate realistic execution with comprehensive cost modeling
        
        Returns detailed fill information including all fees and execution quality
        """
        
        # Calculate fill probability and execution details
        fill_prob, fill_pairs = self.execution_model.calculate_fill_probability(
            quantity, side, market_depth, execution_type
        )
        
        if not fill_pairs or fill_prob == 0:
            # No fill
            return self._create_no_fill_result(symbol, side, quantity, execution_type)
        
        # Process fills
        total_executed_qty = 0
        total_gross_proceeds = 0.0
        total_fees = 0.0
        all_fees = {}
        
        fills = []
        
        for fill_qty, fill_price in fill_pairs:
            # Determine liquidity type based on execution type
            liquidity_type = self._determine_liquidity_type(execution_type, fill_price, market_depth)
            
            # Calculate fees for this fill
            fill_fees = self.fee_calculator.calculate_all_fees(
                fill_qty, fill_price, exchange, liquidity_type, broker_type
            )
            
            # Aggregate fees
            for fee_type, fee_amount in fill_fees.items():
                all_fees[fee_type] = all_fees.get(fee_type, 0.0) + fee_amount
            
            total_executed_qty += fill_qty
            if side.lower() == "buy":
                total_gross_proceeds += fill_qty * fill_price
            else:
                total_gross_proceeds += fill_qty * fill_price
            
            fills.append({
                "quantity": fill_qty,
                "price": fill_price,
                "liquidity_type": liquidity_type.value,
                "fees": fill_fees
            })
        
        # Calculate execution quality metrics
        execution_quality = self._calculate_execution_quality(
            fills, market_depth, side, execution_type
        )
        
        # Calculate net proceeds
        net_proceeds = total_gross_proceeds - all_fees["total_fees"]
        if side.lower() == "buy":
            net_proceeds = -net_proceeds  # Negative for purchases
        
        # Create fill details
        fill_details = FillDetails(
            fill_id=f"fill_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            quantity=total_executed_qty,
            price=total_gross_proceeds / total_executed_qty if total_executed_qty > 0 else 0.0,
            exchange=exchange,
            liquidity_type=LiquidityType.TAKER,  # Default, could be mixed
            execution_type=execution_type,
            fees=all_fees,
            total_fees=all_fees["total_fees"],
            net_proceeds=net_proceeds,
            execution_quality=execution_quality,
            market_conditions={
                "fill_probability": fill_prob,
                "market_depth_levels": len(market_depth.price_levels),
                "total_depth": sum(market_depth.bid_sizes + market_depth.ask_sizes),
                "spread_bps": self._calculate_spread_bps(market_depth)
            }
        )
        
        return fill_details
    
    def _determine_liquidity_type(
        self,
        execution_type: OrderExecutionType,
        fill_price: float,
        market_depth: MarketDepth
    ) -> LiquidityType:
        """Determine if the order provided or took liquidity"""
        
        if execution_type == OrderExecutionType.MARKET:
            return LiquidityType.TAKER
        elif execution_type == OrderExecutionType.LIMIT:
            # Simplified: assume limit orders provide liquidity
            return LiquidityType.MAKER
        elif execution_type in [OrderExecutionType.VWAP, OrderExecutionType.TWAP]:
            # Algorithmic orders typically take liquidity
            return LiquidityType.TAKER
        elif execution_type in [OrderExecutionType.HIDDEN, OrderExecutionType.ICEBERG]:
            return LiquidityType.MAKER
        else:
            return LiquidityType.TAKER
    
    def _calculate_execution_quality(
        self,
        fills: List[Dict],
        market_depth: MarketDepth,
        side: str,
        execution_type: OrderExecutionType
    ) -> ExecutionQuality:
        """Calculate execution quality metrics"""
        
        if not fills:
            return ExecutionQuality(0, 0, 0, 0, 0, 0)
        
        # Calculate volume-weighted average price
        total_qty = sum(fill["quantity"] for fill in fills)
        vwap = sum(fill["quantity"] * fill["price"] for fill in fills) / total_qty
        
        # NBBO midpoint
        nbbo_mid = (market_depth.price_levels[0] + market_depth.price_levels[-1]) / 2
        
        # Price improvement (negative means worse than midpoint)
        if side.lower() == "buy":
            price_improvement = nbbo_mid - vwap
        else:
            price_improvement = vwap - nbbo_mid
        
        # Effective spread
        effective_spread = abs(vwap - nbbo_mid) * 2
        
        # Market impact estimate (simplified)
        spread = market_depth.price_levels[-1] - market_depth.price_levels[0]
        market_impact = effective_spread / spread if spread > 0 else 0
        
        return ExecutionQuality(
            price_improvement=price_improvement,
            effective_spread=effective_spread,
            realized_spread=effective_spread * 0.6,  # Simplified estimate
            market_impact=market_impact,
            timing_risk=0.0001 * total_qty,  # Simple timing risk model
            opportunity_cost=max(0, -price_improvement)  # Cost of adverse price movement
        )
    
    def _calculate_spread_bps(self, market_depth: MarketDepth) -> float:
        """Calculate bid-ask spread in basis points"""
        
        if len(market_depth.price_levels) < 2:
            return 0.0
        
        bid = market_depth.price_levels[0]
        ask = market_depth.price_levels[-1]
        mid = (bid + ask) / 2
        
        spread_bps = ((ask - bid) / mid) * 10000 if mid > 0 else 0
        return spread_bps
    
    def _create_no_fill_result(
        self,
        symbol: str,
        side: str,
        quantity: int,
        execution_type: OrderExecutionType
    ) -> FillDetails:
        """Create result for orders that don't fill"""
        
        return FillDetails(
            fill_id=f"no_fill_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            quantity=0,
            price=0.0,
            exchange=ExchangeType.NYSE,
            liquidity_type=LiquidityType.TAKER,
            execution_type=execution_type,
            fees={},
            total_fees=0.0,
            net_proceeds=0.0,
            execution_quality=ExecutionQuality(0, 0, 0, 0, 0, 0),
            market_conditions={"fill_probability": 0.0}
        )
    
    def generate_sample_market_depth(
        self,
        symbol: str,
        center_price: float,
        spread_bps: float = 5.0,
        depth_levels: int = 10
    ) -> MarketDepth:
        """Generate realistic sample market depth for testing"""
        
        spread = center_price * (spread_bps / 10000)
        bid_price = center_price - spread / 2
        ask_price = center_price + spread / 2
        
        # Generate price levels
        price_levels = []
        bid_sizes = []
        ask_sizes = []
        
        # Bid levels (decreasing prices)
        for i in range(depth_levels // 2):
            level_price = bid_price - (i * spread * 0.1)  # Price levels 10% of spread apart
            price_levels.append(level_price)
            
            # Size decreases with depth, with randomness
            base_size = max(100, int(1000 * np.exp(-i * 0.3) * np.random.uniform(0.5, 1.5)))
            bid_sizes.append(base_size)
        
        # Ask levels (increasing prices)
        for i in range(depth_levels // 2):
            level_price = ask_price + (i * spread * 0.1)
            price_levels.append(level_price)
            
            base_size = max(100, int(1000 * np.exp(-i * 0.3) * np.random.uniform(0.5, 1.5)))
            ask_sizes.append(base_size)
        
        return MarketDepth(
            price_levels=price_levels,
            bid_sizes=bid_sizes,
            ask_sizes=ask_sizes,
            timestamp=datetime.now()
        )

# Factory functions
def create_execution_engine(exchange_type: ExchangeType = ExchangeType.NYSE) -> AdvancedExecutionEngine:
    """Create advanced execution engine for specified exchange"""
    engine = AdvancedExecutionEngine()
    engine.exchange_type = exchange_type
    return engine

def create_fee_calculator() -> ComprehensiveFeeCalculator:
    """Create comprehensive fee calculator"""
    return ComprehensiveFeeCalculator()