"""
Queue Position & Adverse Selection Proxies

This module implements sophisticated microstructure proxies for execution modeling:
- Trade-to-book ratio analysis
- Order book imbalance calculations
- Lambda (adverse selection) proxy estimation
- Queue position modeling
- Fill rate prediction and validation

Key Features:
- Real-time microstructure signal calculation
- Shadow vs simulated fill comparison
- Slippage model integration
- Execution quality validation (≥80% accuracy target)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import minimize_scalar
import asyncio
from collections import deque
import warnings

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enum."""
    BUY = "buy"
    SELL = "sell"


class FillType(Enum):
    """Fill type classification."""
    SHADOW = "shadow"  # Actual market fill
    SIMULATED = "simulated"  # Model predicted fill


@dataclass
class MarketData:
    """Market microstructure data snapshot."""
    timestamp: datetime
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    last_price: float
    last_size: int
    volume: int
    trade_count: int
    
    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask_price - self.bid_price
    
    @property
    def mid_price(self) -> float:
        """Mid price."""
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        if self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return 0


@dataclass
class TradeData:
    """Individual trade information."""
    timestamp: datetime
    symbol: str
    price: float
    size: int
    side: OrderSide
    trade_id: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.side, str):
            self.side = OrderSide(self.side.lower())


@dataclass
class OrderBookSnapshot:
    """Order book snapshot with levels."""
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, int]]  # (price, size) sorted descending
    asks: List[Tuple[float, int]]  # (price, size) sorted ascending
    
    @property
    def best_bid(self) -> Tuple[float, int]:
        """Best bid price and size."""
        return self.bids[0] if self.bids else (0.0, 0)
    
    @property
    def best_ask(self) -> Tuple[float, int]:
        """Best ask price and size."""
        return self.asks[0] if self.asks else (0.0, 0)
    
    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.best_ask[0] - self.best_bid[0]
    
    @property
    def total_bid_size(self) -> int:
        """Total size on bid side."""
        return sum(size for _, size in self.bids)
    
    @property
    def total_ask_size(self) -> int:
        """Total size on ask side."""
        return sum(size for _, size in self.asks)


@dataclass
class MicrostructureSignals:
    """Calculated microstructure signals."""
    timestamp: datetime
    symbol: str
    
    # Trade-to-book ratios
    trade_to_book_ratio: float
    buy_trade_to_book_ratio: float
    sell_trade_to_book_ratio: float
    
    # Imbalance measures
    order_book_imbalance: float  # (bid_size - ask_size) / (bid_size + ask_size)
    price_impact_imbalance: float
    volume_imbalance: float
    
    # Lambda (adverse selection) proxy
    lambda_proxy: float
    effective_spread: float
    realized_spread: float
    price_impact: float
    
    # Queue position estimates
    queue_position_bid: float  # 0-1, position in bid queue
    queue_position_ask: float  # 0-1, position in ask queue
    
    # Flow toxicity measures
    toxicity_score: float
    volume_participation_rate: float
    
    # Execution difficulty
    execution_difficulty: float  # 0-1, higher = more difficult


class MicrostructureAnalyzer:
    """Calculates microstructure signals from market data."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.trade_history = deque(maxlen=window_size)
        self.book_history = deque(maxlen=window_size)
        self.signals_history = deque(maxlen=window_size)
        
    def add_trade(self, trade: TradeData):
        """Add a trade to the history."""
        self.trade_history.append(trade)
        
    def add_book_snapshot(self, book: OrderBookSnapshot):
        """Add an order book snapshot."""
        self.book_history.append(book)
        
    def calculate_signals(self, current_book: OrderBookSnapshot) -> MicrostructureSignals:
        """Calculate all microstructure signals."""
        
        timestamp = current_book.timestamp
        symbol = current_book.symbol
        
        # Get recent trades
        recent_trades = [t for t in self.trade_history 
                        if (timestamp - t.timestamp).total_seconds() <= 300]  # 5 minutes
        
        # Calculate trade-to-book ratios
        tb_ratios = self._calculate_trade_to_book_ratios(recent_trades, current_book)
        
        # Calculate imbalance measures
        imbalances = self._calculate_imbalances(recent_trades, current_book)
        
        # Calculate lambda proxy
        lambda_metrics = self._calculate_lambda_proxy(recent_trades, current_book)
        
        # Calculate queue positions
        queue_positions = self._calculate_queue_positions(recent_trades, current_book)
        
        # Calculate flow toxicity
        toxicity_metrics = self._calculate_toxicity_measures(recent_trades, current_book)
        
        # Calculate execution difficulty
        exec_difficulty = self._calculate_execution_difficulty(
            tb_ratios, imbalances, lambda_metrics, queue_positions, toxicity_metrics
        )
        
        signals = MicrostructureSignals(
            timestamp=timestamp,
            symbol=symbol,
            **tb_ratios,
            **imbalances,
            **lambda_metrics,
            **queue_positions,
            **toxicity_metrics,
            execution_difficulty=exec_difficulty
        )
        
        self.signals_history.append(signals)
        return signals
    
    def _calculate_trade_to_book_ratios(self, trades: List[TradeData], 
                                      book: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate trade-to-book ratios."""
        
        if not trades or not book.bids or not book.asks:
            return {
                'trade_to_book_ratio': 0.0,
                'buy_trade_to_book_ratio': 0.0,
                'sell_trade_to_book_ratio': 0.0
            }
        
        # Total trade volume
        total_trade_volume = sum(t.size for t in trades)
        buy_trade_volume = sum(t.size for t in trades if t.side == OrderSide.BUY)
        sell_trade_volume = sum(t.size for t in trades if t.side == OrderSide.SELL)
        
        # Book size (top 5 levels)
        bid_book_size = sum(size for _, size in book.bids[:5])
        ask_book_size = sum(size for _, size in book.asks[:5])
        total_book_size = bid_book_size + ask_book_size
        
        if total_book_size == 0:
            return {
                'trade_to_book_ratio': 0.0,
                'buy_trade_to_book_ratio': 0.0,
                'sell_trade_to_book_ratio': 0.0
            }
        
        return {
            'trade_to_book_ratio': total_trade_volume / total_book_size,
            'buy_trade_to_book_ratio': buy_trade_volume / ask_book_size if ask_book_size > 0 else 0.0,
            'sell_trade_to_book_ratio': sell_trade_volume / bid_book_size if bid_book_size > 0 else 0.0
        }
    
    def _calculate_imbalances(self, trades: List[TradeData], 
                            book: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate various imbalance measures."""
        
        # Order book imbalance
        bid_size = book.total_bid_size
        ask_size = book.total_ask_size
        total_size = bid_size + ask_size
        
        order_book_imbalance = 0.0
        if total_size > 0:
            order_book_imbalance = (bid_size - ask_size) / total_size
        
        # Price impact imbalance (based on recent price changes)
        price_impact_imbalance = 0.0
        if len(self.book_history) >= 2:
            prev_mid = (self.book_history[-2].best_bid[0] + self.book_history[-2].best_ask[0]) / 2
            curr_mid = (book.best_bid[0] + book.best_ask[0]) / 2
            if prev_mid > 0:
                price_impact_imbalance = (curr_mid - prev_mid) / prev_mid
        
        # Volume imbalance (recent trades)
        volume_imbalance = 0.0
        if trades:
            buy_volume = sum(t.size for t in trades if t.side == OrderSide.BUY)
            sell_volume = sum(t.size for t in trades if t.side == OrderSide.SELL)
            total_volume = buy_volume + sell_volume
            
            if total_volume > 0:
                volume_imbalance = (buy_volume - sell_volume) / total_volume
        
        return {
            'order_book_imbalance': order_book_imbalance,
            'price_impact_imbalance': price_impact_imbalance,
            'volume_imbalance': volume_imbalance
        }
    
    def _calculate_lambda_proxy(self, trades: List[TradeData], 
                               book: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate lambda (adverse selection) proxy."""
        
        if not trades:
            return {
                'lambda_proxy': 0.0,
                'effective_spread': 0.0,
                'realized_spread': 0.0,
                'price_impact': 0.0
            }
        
        # Calculate effective spread
        mid_price = (book.best_bid[0] + book.best_ask[0]) / 2
        effective_spreads = []
        
        for trade in trades[-10:]:  # Last 10 trades
            if trade.side == OrderSide.BUY:
                eff_spread = 2 * (trade.price - mid_price) / mid_price
            else:
                eff_spread = 2 * (mid_price - trade.price) / mid_price
            effective_spreads.append(abs(eff_spread))
        
        effective_spread = np.mean(effective_spreads) if effective_spreads else 0.0
        
        # Calculate realized spread (simplified)
        realized_spread = effective_spread * 0.5  # Rough approximation
        
        # Price impact = effective spread - realized spread
        price_impact = effective_spread - realized_spread
        
        # Lambda proxy (adverse selection component)
        lambda_proxy = price_impact / effective_spread if effective_spread > 0 else 0.0
        
        return {
            'lambda_proxy': lambda_proxy,
            'effective_spread': effective_spread,
            'realized_spread': realized_spread,
            'price_impact': price_impact
        }
    
    def _calculate_queue_positions(self, trades: List[TradeData], 
                                 book: OrderBookSnapshot) -> Dict[str, float]:
        """Estimate queue positions."""
        
        # Simple queue position estimation based on recent activity
        # In practice, this would use more sophisticated order flow analysis
        
        recent_buy_volume = sum(t.size for t in trades[-5:] if t.side == OrderSide.BUY)
        recent_sell_volume = sum(t.size for t in trades[-5:] if t.side == OrderSide.SELL)
        
        # Estimate position based on recent flow and book size
        bid_queue_pos = 0.5  # Default middle of queue
        ask_queue_pos = 0.5
        
        if book.best_bid[1] > 0:
            # Higher recent sell volume = likely better position in bid queue
            bid_queue_pos = min(0.9, 0.1 + (recent_sell_volume / book.best_bid[1]))
        
        if book.best_ask[1] > 0:
            # Higher recent buy volume = likely better position in ask queue
            ask_queue_pos = min(0.9, 0.1 + (recent_buy_volume / book.best_ask[1]))
        
        return {
            'queue_position_bid': bid_queue_pos,
            'queue_position_ask': ask_queue_pos
        }
    
    def _calculate_toxicity_measures(self, trades: List[TradeData], 
                                   book: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate flow toxicity measures."""
        
        if not trades:
            return {
                'toxicity_score': 0.0,
                'volume_participation_rate': 0.0
            }
        
        # Volume-weighted price trend (toxicity indicator)
        if len(trades) >= 3:
            recent_prices = [t.price for t in trades[-3:]]
            recent_sizes = [t.size for t in trades[-3:]]
            
            # Calculate volume-weighted trend
            vwap_recent = np.average(recent_prices, weights=recent_sizes)
            mid_price = (book.best_bid[0] + book.best_ask[0]) / 2
            
            toxicity_score = abs(vwap_recent - mid_price) / mid_price if mid_price > 0 else 0.0
        else:
            toxicity_score = 0.0
        
        # Volume participation rate
        total_volume = sum(t.size for t in trades)
        book_volume = book.total_bid_size + book.total_ask_size
        
        participation_rate = 0.0
        if book_volume > 0:
            participation_rate = min(1.0, total_volume / book_volume)
        
        return {
            'toxicity_score': toxicity_score,
            'volume_participation_rate': participation_rate
        }
    
    def _calculate_execution_difficulty(self, tb_ratios: Dict, imbalances: Dict,
                                      lambda_metrics: Dict, queue_pos: Dict,
                                      toxicity: Dict) -> float:
        """Calculate overall execution difficulty score."""
        
        # Combine various factors into a single difficulty score
        factors = [
            tb_ratios['trade_to_book_ratio'] * 0.3,  # Higher = more difficult
            abs(imbalances['order_book_imbalance']) * 0.2,  # Imbalance = difficulty
            lambda_metrics['lambda_proxy'] * 0.2,  # Adverse selection
            (1 - min(queue_pos['queue_position_bid'], queue_pos['queue_position_ask'])) * 0.2,  # Poor queue position
            toxicity['toxicity_score'] * 0.1  # Flow toxicity
        ]
        
        difficulty = sum(factors)
        return min(1.0, difficulty)  # Cap at 1.0


class FillSimulator:
    """Simulates order fills based on microstructure signals."""
    
    def __init__(self, base_fill_probability: float = 0.8):
        self.base_fill_probability = base_fill_probability
        self.fill_history = []
        
    def predict_fill_probability(self, order_size: int, order_side: OrderSide,
                               signals: MicrostructureSignals,
                               book: OrderBookSnapshot) -> float:
        """Predict probability of fill for an order."""
        
        # Base probability
        fill_prob = self.base_fill_probability
        
        # Adjust based on queue position
        if order_side == OrderSide.BUY:
            queue_factor = signals.queue_position_ask
            available_size = book.best_ask[1]
        else:
            queue_factor = signals.queue_position_bid
            available_size = book.best_bid[1]
        
        # Size factor - larger orders relative to available size are less likely to fill
        size_factor = 1.0
        if available_size > 0:
            size_ratio = order_size / available_size
            size_factor = max(0.1, 1.0 - min(0.8, size_ratio))
        
        # Market conditions factor
        conditions_factor = 1.0 - signals.execution_difficulty
        
        # Toxicity factor - toxic flow is less likely to get good fills
        toxicity_factor = max(0.2, 1.0 - signals.toxicity_score)
        
        # Combined probability
        fill_prob *= queue_factor * size_factor * conditions_factor * toxicity_factor
        
        return max(0.01, min(0.99, fill_prob))
    
    def simulate_fill(self, order_size: int, order_side: OrderSide,
                     signals: MicrostructureSignals,
                     book: OrderBookSnapshot) -> Tuple[bool, int, float]:
        """
        Simulate an order fill.
        
        Returns:
            (filled, fill_size, fill_price)
        """
        
        fill_prob = self.predict_fill_probability(order_size, order_side, signals, book)
        
        # Determine if order fills
        if np.random.random() > fill_prob:
            return False, 0, 0.0
        
        # Determine fill size (partial fills possible)
        if order_side == OrderSide.BUY:
            available_size = book.best_ask[1]
            fill_price = book.best_ask[0]
        else:
            available_size = book.best_bid[1]
            fill_price = book.best_bid[0]
        
        # Fill size based on available liquidity and queue position
        max_fill_size = min(order_size, available_size)
        
        # Random partial fill factor
        if signals.execution_difficulty > 0.7:  # Difficult market conditions
            partial_factor = np.random.uniform(0.3, 0.8)
        else:
            partial_factor = np.random.uniform(0.8, 1.0)
        
        fill_size = int(max_fill_size * partial_factor)
        fill_size = max(1, fill_size) if max_fill_size > 0 else 0
        
        return True, fill_size, fill_price


@dataclass
class FillRecord:
    """Record of an actual or simulated fill."""
    timestamp: datetime
    symbol: str
    order_size: int
    order_side: OrderSide
    fill_type: FillType
    filled: bool
    fill_size: int
    fill_price: float
    signals: MicrostructureSignals
    predicted_probability: float


class FillValidator:
    """Validates simulated fills against shadow (actual) fills."""
    
    def __init__(self, target_accuracy: float = 0.80):
        self.target_accuracy = target_accuracy
        self.fill_records = []
        
    def add_fill_record(self, record: FillRecord):
        """Add a fill record for validation."""
        self.fill_records.append(record)
        
    def calculate_mae(self, lookback_days: int = 30) -> Dict[str, float]:
        """Calculate Mean Absolute Error for fill predictions."""
        
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        recent_records = [r for r in self.fill_records if r.timestamp >= cutoff_time]
        
        if not recent_records:
            return {'mae': 0.0, 'accuracy': 0.0, 'sample_size': 0}
        
        # Group by symbol and order characteristics for comparison
        shadow_fills = [r for r in recent_records if r.fill_type == FillType.SHADOW]
        sim_fills = [r for r in recent_records if r.fill_type == FillType.SIMULATED]
        
        if not shadow_fills or not sim_fills:
            return {'mae': 1.0, 'accuracy': 0.0, 'sample_size': len(recent_records)}
        
        # Calculate prediction accuracy
        prediction_errors = []
        
        for shadow_fill in shadow_fills:
            # Find matching simulated fills
            matching_sims = [
                s for s in sim_fills 
                if (s.symbol == shadow_fill.symbol and
                    s.order_side == shadow_fill.order_side and
                    abs(s.order_size - shadow_fill.order_size) <= shadow_fill.order_size * 0.1 and
                    abs((s.timestamp - shadow_fill.timestamp).total_seconds()) <= 300)
            ]
            
            if matching_sims:
                sim_fill = matching_sims[0]  # Take closest match
                
                # Compare predicted vs actual fill
                predicted_fill = sim_fill.filled
                actual_fill = shadow_fill.filled
                
                # Binary accuracy (did we predict fill correctly?)
                error = abs(int(predicted_fill) - int(actual_fill))
                prediction_errors.append(error)
                
                # Fill size accuracy if both filled
                if predicted_fill and actual_fill:
                    size_error = abs(sim_fill.fill_size - shadow_fill.fill_size) / shadow_fill.fill_size
                    prediction_errors.append(size_error)
        
        if not prediction_errors:
            return {'mae': 1.0, 'accuracy': 0.0, 'sample_size': 0}
        
        mae = np.mean(prediction_errors)
        accuracy = 1.0 - mae
        
        return {
            'mae': mae,
            'accuracy': accuracy,
            'sample_size': len(prediction_errors),
            'meets_target': accuracy >= self.target_accuracy
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        
        if not self.fill_records:
            return {'error': 'No fill records available'}
        
        # Overall statistics
        total_records = len(self.fill_records)
        shadow_records = len([r for r in self.fill_records if r.fill_type == FillType.SHADOW])
        sim_records = len([r for r in self.fill_records if r.fill_type == FillType.SIMULATED])
        
        # MAE calculation
        mae_metrics = self.calculate_mae()
        
        # Symbol-level breakdown
        symbols = list(set(r.symbol for r in self.fill_records))
        symbol_metrics = {}
        
        for symbol in symbols:
            symbol_records = [r for r in self.fill_records if r.symbol == symbol]
            symbol_mae = self._calculate_symbol_mae(symbol_records)
            symbol_metrics[symbol] = symbol_mae
        
        return {
            'overall_metrics': mae_metrics,
            'symbol_metrics': symbol_metrics,
            'record_counts': {
                'total': total_records,
                'shadow': shadow_records,
                'simulated': sim_records
            },
            'target_accuracy': self.target_accuracy,
            'validation_status': 'PASS' if mae_metrics.get('meets_target', False) else 'FAIL'
        }
    
    def _calculate_symbol_mae(self, records: List[FillRecord]) -> Dict[str, float]:
        """Calculate MAE for a specific symbol."""
        
        shadow_fills = [r for r in records if r.fill_type == FillType.SHADOW]
        sim_fills = [r for r in records if r.fill_type == FillType.SIMULATED]
        
        if not shadow_fills or not sim_fills:
            return {'mae': 1.0, 'accuracy': 0.0, 'sample_size': 0}
        
        errors = []
        for shadow in shadow_fills:
            for sim in sim_fills:
                if (abs((shadow.timestamp - sim.timestamp).total_seconds()) <= 300 and
                    shadow.order_side == sim.order_side):
                    error = abs(int(shadow.filled) - int(sim.filled))
                    errors.append(error)
                    break
        
        if not errors:
            return {'mae': 1.0, 'accuracy': 0.0, 'sample_size': 0}
        
        mae = np.mean(errors)
        return {'mae': mae, 'accuracy': 1.0 - mae, 'sample_size': len(errors)}


class MicrostructureProxyEngine:
    """Main engine for microstructure proxies and fill simulation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        self.analyzer = MicrostructureAnalyzer(
            window_size=config.get('window_size', 100)
        )
        self.simulator = FillSimulator(
            base_fill_probability=config.get('base_fill_probability', 0.8)
        )
        self.validator = FillValidator(
            target_accuracy=config.get('target_accuracy', 0.80)
        )
        
        self.current_signals = None
        self.last_book = None
        
    async def process_market_data(self, market_data: MarketData) -> MicrostructureSignals:
        """Process new market data and calculate signals."""
        
        # Convert to order book snapshot (simplified)
        book = OrderBookSnapshot(
            timestamp=market_data.timestamp,
            symbol=market_data.symbol,
            bids=[(market_data.bid_price, market_data.bid_size)],
            asks=[(market_data.ask_price, market_data.ask_size)]
        )
        
        self.analyzer.add_book_snapshot(book)
        self.last_book = book
        
        # Calculate signals
        signals = self.analyzer.calculate_signals(book)
        self.current_signals = signals
        
        return signals
    
    async def process_trade(self, trade_data: TradeData):
        """Process a new trade."""
        self.analyzer.add_trade(trade_data)
    
    async def simulate_order_fill(self, order_size: int, order_side: OrderSide,
                                symbol: str) -> Dict[str, Any]:
        """Simulate an order fill and return results."""
        
        if not self.current_signals or not self.last_book:
            return {'error': 'No market data available'}
        
        # Predict fill
        fill_prob = self.simulator.predict_fill_probability(
            order_size, order_side, self.current_signals, self.last_book
        )
        
        # Simulate execution
        filled, fill_size, fill_price = self.simulator.simulate_fill(
            order_size, order_side, self.current_signals, self.last_book
        )
        
        # Create fill record
        fill_record = FillRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            order_size=order_size,
            order_side=order_side,
            fill_type=FillType.SIMULATED,
            filled=filled,
            fill_size=fill_size,
            fill_price=fill_price,
            signals=self.current_signals,
            predicted_probability=fill_prob
        )
        
        self.validator.add_fill_record(fill_record)
        
        return {
            'order_size': order_size,
            'order_side': order_side.value,
            'predicted_fill_probability': fill_prob,
            'simulated_fill': {
                'filled': filled,
                'fill_size': fill_size,
                'fill_price': fill_price
            },
            'microstructure_signals': {
                'trade_to_book_ratio': self.current_signals.trade_to_book_ratio,
                'order_book_imbalance': self.current_signals.order_book_imbalance,
                'lambda_proxy': self.current_signals.lambda_proxy,
                'execution_difficulty': self.current_signals.execution_difficulty,
                'queue_position_bid': self.current_signals.queue_position_bid,
                'queue_position_ask': self.current_signals.queue_position_ask
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def add_shadow_fill(self, symbol: str, order_size: int, order_side: OrderSide,
                            filled: bool, fill_size: int, fill_price: float):
        """Add a shadow (actual market) fill for validation."""
        
        if not self.current_signals:
            return
        
        fill_record = FillRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            order_size=order_size,
            order_side=order_side,
            fill_type=FillType.SHADOW,
            filled=filled,
            fill_size=fill_size,
            fill_price=fill_price,
            signals=self.current_signals,
            predicted_probability=0.0  # Not applicable for shadow fills
        )
        
        self.validator.add_fill_record(fill_record)
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get current validation metrics."""
        return self.validator.get_performance_metrics()
    
    def get_current_signals(self) -> Optional[MicrostructureSignals]:
        """Get the most recent microstructure signals."""
        return self.current_signals


# Example usage and testing functions

async def example_usage():
    """Example of using the microstructure proxy engine."""
    
    # Create engine
    config = {
        'window_size': 50,
        'base_fill_probability': 0.8,
        'target_accuracy': 0.80
    }
    
    engine = MicrostructureProxyEngine(config)
    
    # Simulate market data stream
    symbol = "AAPL"
    base_price = 150.0
    
    print("Simulating market data and order flow...")
    
    for i in range(20):
        # Generate market data
        spread = 0.02
        bid_price = base_price - spread/2 + np.random.normal(0, 0.01)
        ask_price = bid_price + spread
        
        market_data = MarketData(
            timestamp=datetime.now(),
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=np.random.randint(100, 1000),
            ask_size=np.random.randint(100, 1000),
            last_price=bid_price + spread/2,
            last_size=np.random.randint(10, 100),
            volume=np.random.randint(1000, 5000),
            trade_count=np.random.randint(10, 50)
        )
        
        # Process market data
        signals = await engine.process_market_data(market_data)
        
        # Simulate some trades
        if i > 5:  # After some history
            for _ in range(2):
                side = OrderSide.BUY if np.random.random() > 0.5 else OrderSide.SELL
                size = np.random.randint(50, 300)
                
                # Simulate order
                result = await engine.simulate_order_fill(size, side, symbol)
                
                if i % 5 == 0:  # Print some results
                    print(f"Order: {size} {side.value}")
                    print(f"Fill probability: {result['predicted_fill_probability']:.3f}")
                    print(f"Simulated fill: {result['simulated_fill']}")
                    print(f"Execution difficulty: {result['microstructure_signals']['execution_difficulty']:.3f}")
                    print("---")
        
        # Update base price (random walk)
        base_price += np.random.normal(0, 0.05)
    
    # Get validation metrics
    metrics = engine.get_validation_metrics()
    print(f"\nValidation Metrics:")
    print(f"Overall accuracy: {metrics.get('overall_metrics', {}).get('accuracy', 0):.3f}")
    print(f"Target accuracy: {metrics.get('target_accuracy', 0.8)}")
    print(f"Validation status: {metrics.get('validation_status', 'UNKNOWN')}")


if __name__ == "__main__":
    asyncio.run(example_usage())