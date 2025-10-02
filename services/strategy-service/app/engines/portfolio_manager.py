"""
Portfolio Manager - Manages portfolio state during backtesting and live trading
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass
import pandas as pd
import uuid

from ..schemas import Trade, Position, TradeType
from ..execution.venue_rules import VenueRuleEngine
from ..decisions import DecisionExplainer, TradeDecision, create_trade_decision, finalize_trade_decision
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from infrastructure.monitoring.latency_timer import (
    PipelineStage, PipelineTimer, StageTimer, latency_timer
)

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Current state of the portfolio"""
    cash: Decimal
    positions: Dict[str, Position]
    total_value: Decimal
    timestamp: datetime


class PortfolioManager:
    """Manages portfolio positions and cash during backtesting and live trading"""
    
    def __init__(self, initial_cash: Decimal = Decimal('100000')):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.transaction_costs = Decimal('0.001')  # 0.1% transaction cost
        self.venue_rule_engine = VenueRuleEngine()
        
    async def initialize_venue_rules(self, symbols: List[str]) -> None:
        """Initialize venue rule engine with symbols for halt/LULD monitoring"""
        await self.venue_rule_engine.initialize(symbols)
        logger.info(f"Venue rule engine initialized for symbols: {symbols}")
        
    async def execute_trade(self, trade: Trade, current_price: Decimal) -> bool:
        """Execute a trade and update portfolio state with full latency monitoring and decision tracing"""
        # Generate unique trace ID for this execution
        trace_id = str(uuid.uuid4())
        side = "buy" if trade.trade_type == TradeType.BUY else "sell"
        quantity = int(trade.quantity)
        
        # Start pipeline timing
        with PipelineTimer(trace_id, trade.symbol, side, quantity) as pipeline:
            try:
                # Create initial trade decision for tracking
                decision = create_trade_decision(
                    symbol=trade.symbol,
                    side=side,
                    quantity=quantity,
                    trade_id=trace_id,
                    trace_id=trace_id
                )
                
                # Stage 1: Signal Generation (simulated - already generated)
                with StageTimer(trace_id, PipelineStage.SIGNAL_GENERATION, {"signal_strength": 0.75}):
                    # Signal generation already completed (external)
                    pass
                
                # Stage 2: Regime Filter (simulated check)
                with StageTimer(trace_id, PipelineStage.REGIME_FILTER, {"market_regime": "bullish"}):
                    # Simulate regime filter check
                    regime_check = True  # Would check actual regime here
                    if not regime_check:
                        decision.add_rejection("regime_bearish", "Market regime is bearish, blocking trade")
                        finalize_trade_decision(decision)
                        return False
                
                # Stage 3: VaR Calculation
                with StageTimer(trace_id, PipelineStage.VAR_CALCULATION, {"current_var": 0.15}):
                    # Simulate VaR check
                    portfolio_value = float(self.get_portfolio_value())
                    trade_value = float(quantity * current_price)
                    var_check = trade_value < portfolio_value * 0.2  # Max 20% of portfolio
                    if not var_check:
                        decision.add_rejection("var_exceeded", f"Trade value ${trade_value:,.0f} exceeds VaR limit")
                        finalize_trade_decision(decision)
                        return False
                
                # Stage 4: Borrow Check (for short sales)
                with StageTimer(trace_id, PipelineStage.BORROW_CHECK, {"borrow_rate": 0.02}):
                    if trade.trade_type == TradeType.SELL:
                        # Check if we have position to sell
                        position = self.get_position(trade.symbol)
                        if not position or position.quantity < trade.quantity:
                            decision.add_rejection("borrow_unavailable", f"Insufficient position: have {position.quantity if position else 0}, need {trade.quantity}")
                            finalize_trade_decision(decision)
                            return False
                
                # Stage 5: Venue Rules (HALT/LULD COMPLIANCE)
                with StageTimer(trace_id, PipelineStage.VENUE_RULES, {"venue": "NASDAQ"}):
                    price = float(current_price) if current_price else None
                    allowed, reason = await self.venue_rule_engine.validate_order(
                        symbol=trade.symbol,
                        side=side,
                        quantity=quantity,
                        price=price,
                        order_type="market"
                    )
                    
                    if not allowed:
                        decision.add_rejection("halted" if "halt" in reason.lower() else "luld_violation", reason)
                        finalize_trade_decision(decision)
                        logger.warning(f"Order blocked by venue rules: {trade.symbol} {side} {quantity} - {reason}")
                        return False
                
                # Stage 6: Position Sizing
                with StageTimer(trace_id, PipelineStage.POSITION_SIZING, {"max_position_size": 10000}):
                    # Check position sizing limits
                    if trade.trade_type == TradeType.BUY:
                        total_cost = trade.quantity * current_price
                        commission = total_cost * self.transaction_costs
                        total_cost_with_commission = total_cost + commission
                        
                        if self.cash < total_cost_with_commission:
                            decision.add_rejection("insufficient_funds", f"Insufficient cash: need ${total_cost_with_commission}, have ${self.cash}")
                            finalize_trade_decision(decision)
                            return False
                
                # Stage 7: Execution Prep
                with StageTimer(trace_id, PipelineStage.EXECUTION_PREP, {"order_type": "market"}):
                    # Final pre-execution checks and order preparation
                    if trade.trade_type not in [TradeType.BUY, TradeType.SELL]:
                        decision.add_rejection("invalid_trade_type", f"Unknown trade type: {trade.trade_type}")
                        finalize_trade_decision(decision)
                        return False
                
                # Stage 8: Trade Execution
                with StageTimer(trace_id, PipelineStage.TRADE_EXECUTION, {"execution_venue": "primary"}):
                    if trade.trade_type == TradeType.BUY:
                        success = self._execute_buy(trade, current_price)
                    else:
                        success = self._execute_sell(trade, current_price)
                    
                    # Finalize decision with success/failure
                    finalize_trade_decision(decision)
                    
                    if success:
                        logger.info(f"Trade executed successfully: {trace_id}")
                    else:
                        logger.warning(f"Trade execution failed: {trace_id}")
                    
                    return success
                    
            except Exception as e:
                # Mark decision as failed due to error
                if 'decision' in locals():
                    decision.add_rejection("execution_error", "execution", str(e))
                    finalize_trade_decision(decision)
                logger.error(f"Failed to execute trade {trace_id}: {e}")
                return False
    
    def _execute_buy(self, trade: Trade, current_price: Decimal) -> bool:
        """Execute buy order"""
        total_cost = trade.quantity * current_price
        commission = total_cost * self.transaction_costs
        total_cost_with_commission = total_cost + commission
        
        if self.cash < total_cost_with_commission:
            logger.warning(f"Insufficient funds for buy order: {total_cost_with_commission}")
            return False
        
        # Update cash
        self.cash -= total_cost_with_commission
        
        # Update or create position
        if trade.symbol in self.positions:
            position = self.positions[trade.symbol]
            # Calculate new average price
            total_quantity = position.quantity + trade.quantity
            total_cost_basis = (position.quantity * position.average_price + 
                              trade.quantity * current_price)
            new_average_price = total_cost_basis / total_quantity
            
            position.quantity = total_quantity
            position.average_price = new_average_price
            position.current_price = current_price
            position.market_value = total_quantity * current_price
            position.unrealized_pnl = position.market_value - total_cost_basis
        else:
            # Create new position
            self.positions[trade.symbol] = Position(
                symbol=trade.symbol,
                quantity=trade.quantity,
                average_price=current_price,
                current_price=current_price,
                market_value=trade.quantity * current_price,
                unrealized_pnl=Decimal('0')
            )
        
        # Record trade
        executed_trade = Trade(
            symbol=trade.symbol,
            trade_type=trade.trade_type,
            quantity=trade.quantity,
            price=current_price,
            timestamp=trade.timestamp,
            commission=commission
        )
        self.trades.append(executed_trade)
        
        logger.info(f"Executed BUY: {trade.quantity} {trade.symbol} @ {current_price}")
        return True
    
    def _execute_sell(self, trade: Trade, current_price: Decimal) -> bool:
        """Execute sell order"""
        if trade.symbol not in self.positions:
            logger.warning(f"Cannot sell {trade.symbol}: No position found")
            return False
        
        position = self.positions[trade.symbol]
        if position.quantity < trade.quantity:
            logger.warning(f"Cannot sell {trade.quantity} {trade.symbol}: Only have {position.quantity}")
            return False
        
        # Calculate proceeds
        proceeds = trade.quantity * current_price
        commission = proceeds * self.transaction_costs
        net_proceeds = proceeds - commission
        
        # Update cash
        self.cash += net_proceeds
        
        # Update position
        position.quantity -= trade.quantity
        if position.quantity == 0:
            # Close position
            del self.positions[trade.symbol]
        else:
            # Update remaining position
            position.current_price = current_price
            position.market_value = position.quantity * current_price
            cost_basis = position.quantity * position.average_price
            position.unrealized_pnl = position.market_value - cost_basis
        
        # Record trade
        executed_trade = Trade(
            symbol=trade.symbol,
            trade_type=trade.trade_type,
            quantity=trade.quantity,
            price=current_price,
            timestamp=trade.timestamp,
            commission=commission
        )
        self.trades.append(executed_trade)
        
        logger.info(f"Executed SELL: {trade.quantity} {trade.symbol} @ {current_price}")
        return True
    
    def update_positions_price(self, price_data: Dict[str, Decimal]) -> None:
        """Update current prices for all positions"""
        for symbol, position in self.positions.items():
            if symbol in price_data:
                position.current_price = price_data[symbol]
                position.market_value = position.quantity * position.current_price
                cost_basis = position.quantity * position.average_price
                position.unrealized_pnl = position.market_value - cost_basis
    
    def get_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value"""
        positions_value = sum(pos.market_value or Decimal('0') 
                            for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """Get all current positions"""
        return list(self.positions.values())
    
    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state snapshot"""
        return PortfolioState(
            cash=self.cash,
            positions=self.positions.copy(),
            total_value=self.get_portfolio_value(),
            timestamp=datetime.now()
        )
    
    def reset(self, initial_cash: Optional[Decimal] = None) -> None:
        """Reset portfolio to initial state"""
        if initial_cash is not None:
            self.initial_cash = initial_cash
        
        self.cash = self.initial_cash
        self.positions.clear()
        self.trades.clear()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        current_value = float(self.get_portfolio_value())
        initial_value = float(self.initial_cash)
        
        total_return = current_value - initial_value
        total_return_pct = (total_return / initial_value) * 100 if initial_value > 0 else 0
        
        return {
            'initial_value': initial_value,
            'current_value': current_value,
            'total_return': total_return,
            'total_return_percent': total_return_pct,
            'cash': float(self.cash),
            'positions_count': len(self.positions),
            'total_trades': len(self.trades)
        }