"""
Portfolio Manager - Manages portfolio state during backtesting and live trading
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass
import pandas as pd

from ..schemas import Trade, Position, TradeType

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
        
    def execute_trade(self, trade: Trade, current_price: Decimal) -> bool:
        """Execute a trade and update portfolio state"""
        try:
            if trade.trade_type == TradeType.BUY:
                return self._execute_buy(trade, current_price)
            elif trade.trade_type == TradeType.SELL:
                return self._execute_sell(trade, current_price)
            else:
                logger.warning(f"Unknown trade type: {trade.trade_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
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