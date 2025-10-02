"""
Backtest Engine Extensions

Adds event-aware stops and latency modeling to the backtest engine.
Week 10 enhancements for realistic execution simulation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

from ..risk.event_stops import EventAwareStopLoss, EventType, StopLossConfig

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for execution simulation"""
    MARKET = "market"
    LIMIT = "limit"
    IOC = "ioc"  # Immediate or Cancel
    MID_PEG = "mid_peg"  # Pegged to mid-price
    TWAP = "twap"
    VWAP = "vwap"


@dataclass
class LatencyConfig:
    """Latency modeling configuration"""
    # Decision to fill delay
    exec_latency_ms: int = 300  # 200-500ms typical
    latency_std_ms: int = 50  # Variability

    # Enable advanced order types
    enable_ioc: bool = True
    enable_mid_peg: bool = True

    # State-dependent impact
    enable_state_impact: bool = True
    high_vol_multiplier: float = 1.5  # Higher impact in high volatility
    low_liquidity_multiplier: float = 2.0  # Higher impact in low liquidity


@dataclass
class ExecutionResult:
    """Result of order execution with latency"""
    symbol: str
    order_type: OrderType
    intended_price: float  # Price when decision made
    fill_price: float  # Actual fill price after latency
    quantity: int
    side: str  # 'buy' or 'sell'

    decision_time: datetime
    fill_time: datetime
    latency_ms: float

    slippage_bps: float  # Slippage in basis points
    market_impact_bps: float  # Market impact in basis points

    filled: bool = True
    partial_fill_quantity: int = 0

    # State-dependent factors
    volatility_regime: str = "normal"  # normal, high, extreme
    liquidity_regime: str = "normal"  # normal, low, illiquid


class BacktestWithLatency:
    """
    Backtest engine extension with latency modeling and order types.

    Simulates 200-500ms decision→fill delay.
    Supports IOC and mid-peg orders.
    State-dependent market impact.
    """

    def __init__(self, config: Optional[LatencyConfig] = None):
        """
        Initialize latency modeling.

        Args:
            config: Latency configuration (uses defaults if None)
        """
        self.config = config or LatencyConfig()
        self.execution_history: List[ExecutionResult] = []

    def simulate_execution(self,
                          symbol: str,
                          order_type: OrderType,
                          quantity: int,
                          side: str,
                          decision_time: datetime,
                          decision_price: float,
                          market_data: pd.DataFrame,
                          current_index: int) -> ExecutionResult:
        """
        Simulate order execution with latency and market impact.

        Args:
            symbol: Stock symbol
            order_type: Type of order
            quantity: Order quantity
            side: 'buy' or 'sell'
            decision_time: When decision was made
            decision_price: Price when decision was made
            market_data: Full market data DataFrame
            current_index: Current position in data

        Returns:
            ExecutionResult with fill details
        """
        # Simulate latency delay
        latency_ms = self._simulate_latency()
        latency_seconds = latency_ms / 1000.0

        # Determine fill time and price
        fill_time = decision_time + timedelta(seconds=latency_seconds)

        # Get market state after latency
        fill_price, market_state = self._get_fill_price(
            order_type=order_type,
            decision_price=decision_price,
            quantity=quantity,
            side=side,
            market_data=market_data,
            current_index=current_index,
            latency_seconds=latency_seconds
        )

        # Calculate slippage (price movement during latency)
        if side == "buy":
            slippage_bps = ((fill_price - decision_price) / decision_price) * 10000
        else:  # sell
            slippage_bps = ((decision_price - fill_price) / decision_price) * 10000

        # Calculate market impact
        market_impact_bps = self._calculate_market_impact(
            quantity=quantity,
            decision_price=decision_price,
            volatility_regime=market_state['volatility_regime'],
            liquidity_regime=market_state['liquidity_regime']
        )

        # Apply market impact to fill price
        if side == "buy":
            fill_price *= (1 + market_impact_bps / 10000)
        else:  # sell
            fill_price *= (1 - market_impact_bps / 10000)

        # Check if order filled (some order types may not fill)
        filled = self._check_fill_probability(order_type, market_state)

        result = ExecutionResult(
            symbol=symbol,
            order_type=order_type,
            intended_price=decision_price,
            fill_price=fill_price,
            quantity=quantity if filled else 0,
            side=side,
            decision_time=decision_time,
            fill_time=fill_time,
            latency_ms=latency_ms,
            slippage_bps=slippage_bps,
            market_impact_bps=market_impact_bps,
            filled=filled,
            partial_fill_quantity=quantity if filled else 0,
            volatility_regime=market_state['volatility_regime'],
            liquidity_regime=market_state['liquidity_regime']
        )

        self.execution_history.append(result)

        return result

    def _simulate_latency(self) -> float:
        """Simulate execution latency with normal distribution"""
        latency = np.random.normal(
            self.config.exec_latency_ms,
            self.config.latency_std_ms
        )

        # Clamp to reasonable range (50ms - 2000ms)
        latency = max(50, min(2000, latency))

        return latency

    def _get_fill_price(self,
                       order_type: OrderType,
                       decision_price: float,
                       quantity: int,
                       side: str,
                       market_data: pd.DataFrame,
                       current_index: int,
                       latency_seconds: float) -> Tuple[float, Dict]:
        """
        Get fill price based on order type and market conditions.

        Returns:
            Tuple of (fill_price, market_state)
        """
        # Estimate price after latency (simple linear interpolation)
        if current_index + 1 < len(market_data):
            next_price = market_data.iloc[current_index + 1]['close']
            price_after_latency = decision_price + (next_price - decision_price) * 0.5
        else:
            price_after_latency = decision_price

        # Determine market state
        market_state = self._assess_market_state(market_data, current_index)

        # Determine fill price based on order type
        if order_type == OrderType.MARKET:
            # Market order: fill at price after latency
            fill_price = price_after_latency

        elif order_type == OrderType.IOC:
            # Immediate or Cancel: try to fill immediately at best available
            if self.config.enable_ioc:
                # Simulate crossing spread
                spread_bps = market_state.get('spread_bps', 10)
                if side == "buy":
                    fill_price = decision_price * (1 + spread_bps / 20000)  # Half spread
                else:
                    fill_price = decision_price * (1 - spread_bps / 20000)
            else:
                fill_price = price_after_latency

        elif order_type == OrderType.MID_PEG:
            # Mid-peg: fill at mid-price
            if self.config.enable_mid_peg:
                fill_price = decision_price  # Assume mid-price execution
            else:
                fill_price = price_after_latency

        elif order_type == OrderType.LIMIT:
            # Limit order: fill only if price favorable
            # Simplified: assume fills at limit price if market moved favorably
            fill_price = decision_price

        else:
            # Default to market
            fill_price = price_after_latency

        return fill_price, market_state

    def _assess_market_state(self,
                            market_data: pd.DataFrame,
                            current_index: int) -> Dict:
        """
        Assess current market state (volatility, liquidity).

        Returns:
            Dictionary with market state metrics
        """
        # Calculate recent volatility
        if current_index >= 20:
            recent_returns = market_data.iloc[current_index-20:current_index]['close'].pct_change()
            volatility = recent_returns.std()
        else:
            volatility = 0.01  # Default 1% daily vol

        # Classify volatility regime
        if volatility < 0.01:
            vol_regime = "low"
        elif volatility < 0.02:
            vol_regime = "normal"
        elif volatility < 0.04:
            vol_regime = "high"
        else:
            vol_regime = "extreme"

        # Estimate liquidity (simplified - would use volume in reality)
        if current_index > 0:
            avg_volume = market_data.iloc[max(0, current_index-10):current_index].get('volume', pd.Series([1000000])).mean()
            current_volume = market_data.iloc[current_index].get('volume', 1000000)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            volume_ratio = 1.0

        # Classify liquidity regime
        if volume_ratio > 1.5:
            liq_regime = "high"
        elif volume_ratio > 0.7:
            liq_regime = "normal"
        elif volume_ratio > 0.3:
            liq_regime = "low"
        else:
            liq_regime = "illiquid"

        # Estimate spread (simplified - would use actual bid/ask in reality)
        base_spread_bps = 5  # 5 bps base
        if vol_regime == "extreme":
            spread_bps = base_spread_bps * 3
        elif vol_regime == "high":
            spread_bps = base_spread_bps * 2
        else:
            spread_bps = base_spread_bps

        if liq_regime == "illiquid":
            spread_bps *= 2

        return {
            "volatility": volatility,
            "volatility_regime": vol_regime,
            "liquidity_regime": liq_regime,
            "volume_ratio": volume_ratio,
            "spread_bps": spread_bps
        }

    def _calculate_market_impact(self,
                                 quantity: int,
                                 decision_price: float,
                                 volatility_regime: str,
                                 liquidity_regime: str) -> float:
        """
        Calculate market impact in basis points.

        Uses state-dependent impact model.
        """
        # Base impact (simplified square root model)
        order_value = quantity * decision_price
        base_impact_bps = 2.0 * np.sqrt(order_value / 1000000)  # 2 bps per $1M

        # State-dependent multipliers
        if self.config.enable_state_impact:
            # Volatility adjustment
            if volatility_regime == "extreme":
                base_impact_bps *= self.config.high_vol_multiplier * 1.5
            elif volatility_regime == "high":
                base_impact_bps *= self.config.high_vol_multiplier

            # Liquidity adjustment
            if liquidity_regime == "illiquid":
                base_impact_bps *= self.config.low_liquidity_multiplier
            elif liquidity_regime == "low":
                base_impact_bps *= self.config.low_liquidity_multiplier * 0.5

        return base_impact_bps

    def _check_fill_probability(self,
                                order_type: OrderType,
                                market_state: Dict) -> bool:
        """
        Check if order fills based on order type and market state.

        Returns:
            True if order fills, False otherwise
        """
        # Market orders always fill
        if order_type == OrderType.MARKET:
            return True

        # IOC has high but not 100% fill rate
        if order_type == OrderType.IOC:
            if market_state['liquidity_regime'] == "illiquid":
                fill_prob = 0.7
            else:
                fill_prob = 0.95
            return np.random.random() < fill_prob

        # Mid-peg has good fill rate in normal conditions
        if order_type == OrderType.MID_PEG:
            if market_state['liquidity_regime'] == "illiquid":
                fill_prob = 0.6
            elif market_state['volatility_regime'] == "extreme":
                fill_prob = 0.75
            else:
                fill_prob = 0.9
            return np.random.random() < fill_prob

        # Limit orders have lower fill probability
        if order_type == OrderType.LIMIT:
            return np.random.random() < 0.6

        # Default: fill
        return True

    def get_execution_statistics(self) -> Dict:
        """
        Get statistics on execution quality.

        Returns:
            Dictionary with execution stats
        """
        if not self.execution_history:
            return {
                "total_executions": 0,
                "fill_rate": 0.0,
                "avg_latency_ms": 0.0,
                "avg_slippage_bps": 0.0,
                "avg_impact_bps": 0.0
            }

        filled_orders = [e for e in self.execution_history if e.filled]

        stats = {
            "total_executions": len(self.execution_history),
            "filled_orders": len(filled_orders),
            "fill_rate": len(filled_orders) / len(self.execution_history) * 100,
            "avg_latency_ms": np.mean([e.latency_ms for e in self.execution_history]),
            "p95_latency_ms": np.percentile([e.latency_ms for e in self.execution_history], 95),
            "p99_latency_ms": np.percentile([e.latency_ms for e in self.execution_history], 99),
            "avg_slippage_bps": np.mean([abs(e.slippage_bps) for e in filled_orders]) if filled_orders else 0,
            "avg_impact_bps": np.mean([e.market_impact_bps for e in filled_orders]) if filled_orders else 0,
            "total_slippage_cost_bps": np.sum([abs(e.slippage_bps) for e in filled_orders]) if filled_orders else 0,
            "total_impact_cost_bps": np.sum([e.market_impact_bps for e in filled_orders]) if filled_orders else 0
        }

        # Breakdown by order type
        for order_type in OrderType:
            type_orders = [e for e in self.execution_history if e.order_type == order_type]
            if type_orders:
                stats[f"{order_type.value}_count"] = len(type_orders)
                stats[f"{order_type.value}_fill_rate"] = sum(1 for e in type_orders if e.filled) / len(type_orders) * 100
                stats[f"{order_type.value}_avg_slippage"] = np.mean([abs(e.slippage_bps) for e in type_orders if e.filled])

        return stats

    def compare_live_vs_sim_slippage(self,
                                     live_slippage_data: List[float]) -> Dict:
        """
        Compare live vs simulated slippage.

        Args:
            live_slippage_data: List of actual slippage values (bps) from live trading

        Returns:
            Comparison statistics
        """
        sim_slippage = [abs(e.slippage_bps) for e in self.execution_history if e.filled]

        if not sim_slippage or not live_slippage_data:
            return {"error": "Insufficient data for comparison"}

        # Calculate gap
        live_avg = np.mean(live_slippage_data)
        sim_avg = np.mean(sim_slippage)
        gap_pct = abs(live_avg - sim_avg) / live_avg * 100 if live_avg > 0 else 0

        # Count trades within 10% gap
        total_trades = min(len(live_slippage_data), len(sim_slippage))
        within_10pct = 0

        for i in range(total_trades):
            live_val = live_slippage_data[i]
            sim_val = sim_slippage[i] if i < len(sim_slippage) else sim_avg

            if live_val > 0:
                trade_gap = abs(live_val - sim_val) / live_val * 100
                if trade_gap < 10:
                    within_10pct += 1

        within_10pct_rate = within_10pct / total_trades * 100 if total_trades > 0 else 0

        return {
            "live_avg_slippage_bps": live_avg,
            "sim_avg_slippage_bps": sim_avg,
            "gap_percent": gap_pct,
            "trades_within_10pct": within_10pct,
            "total_trades_compared": total_trades,
            "within_10pct_rate": within_10pct_rate,
            "meets_acceptance": within_10pct_rate >= 80.0  # ≥80% trades within 10% gap
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Configure latency modeling
    config = LatencyConfig(
        exec_latency_ms=300,
        enable_ioc=True,
        enable_mid_peg=True,
        enable_state_impact=True
    )

    latency_sim = BacktestWithLatency(config)

    # Mock market data
    dates = pd.date_range(start='2025-01-01', periods=100, freq='1min')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
    market_data = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(10000, 100000, 100)
    }, index=dates)

    # Simulate execution
    result = latency_sim.simulate_execution(
        symbol="AAPL",
        order_type=OrderType.MARKET,
        quantity=100,
        side="buy",
        decision_time=dates[50],
        decision_price=market_data.iloc[50]['close'],
        market_data=market_data,
        current_index=50
    )

    print(f"Execution Result:")
    print(f"  Intended price: ${result.intended_price:.2f}")
    print(f"  Fill price: ${result.fill_price:.2f}")
    print(f"  Latency: {result.latency_ms:.1f}ms")
    print(f"  Slippage: {result.slippage_bps:.2f} bps")
    print(f"  Market impact: {result.market_impact_bps:.2f} bps")
    print(f"  Filled: {result.filled}")

    # Get statistics
    stats = latency_sim.get_execution_statistics()
    print(f"\nExecution Statistics:")
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Fill rate: {stats['fill_rate']:.1f}%")
    print(f"  Avg latency: {stats['avg_latency_ms']:.1f}ms")
    print(f"  Avg slippage: {stats['avg_slippage_bps']:.2f} bps")
