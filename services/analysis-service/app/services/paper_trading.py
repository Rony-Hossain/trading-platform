"""
Paper Trading Engine for Strategy Validation

This module implements a comprehensive paper trading system for validating
strategies in a realistic but risk-free environment before production deployment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging
import uuid
from decimal import Decimal, ROUND_HALF_UP
import asyncpg
from ..core.database import get_database_url
from .execution_modeling import (
    AdvancedExecutionEngine, ExchangeType, OrderExecutionType, 
    MarketDepth, create_execution_engine
)

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types for paper trading"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class PositionSide(Enum):
    """Position sides"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

@dataclass
class MarketData:
    """Real-time market data snapshot"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    bid_size: int
    ask_size: int

@dataclass
class PaperOrder:
    """Paper trading order"""
    order_id: str
    strategy_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PaperPosition:
    """Paper trading position"""
    strategy_id: str
    symbol: str
    side: PositionSide
    quantity: int
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class PaperAccount:
    """Paper trading account"""
    account_id: str
    strategy_id: str
    initial_balance: float
    cash_balance: float
    total_value: float
    buying_power: float
    positions: Dict[str, PaperPosition] = field(default_factory=dict)
    open_orders: Dict[str, PaperOrder] = field(default_factory=dict)
    order_history: List[PaperOrder] = field(default_factory=list)
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ExecutionDetails:
    """Order execution details"""
    execution_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    commission: float
    execution_time: datetime
    market_impact: float = 0.0
    slippage: float = 0.0
    exchange_type: str = "NYSE"
    maker_fee: float = 0.0
    taker_fee: float = 0.0
    sec_fee: float = 0.0
    taf_fee: float = 0.0
    finra_orf: float = 0.0
    clearing_fee: float = 0.0
    total_fees: float = 0.0
    fill_probability: float = 1.0
    depth_level: int = 1


class PaperTradingEngine:
    """Comprehensive paper trading engine"""
    
    def __init__(self, initial_balance: float = 100000.0, exchange_type: ExchangeType = ExchangeType.NYSE):
        self.initial_balance = initial_balance
        self.accounts: Dict[str, PaperAccount] = {}
        self.execution_engine = create_execution_engine(exchange_type)
        self.market_data_cache: Dict[str, MarketData] = {}
        
    async def create_paper_account(self, strategy_id: str) -> PaperAccount:
        """Create new paper trading account"""
        
        account_id = f"paper_{strategy_id}_{uuid.uuid4().hex[:8]}"
        
        account = PaperAccount(
            account_id=account_id,
            strategy_id=strategy_id,
            initial_balance=self.initial_balance,
            cash_balance=self.initial_balance,
            total_value=self.initial_balance,
            buying_power=self.initial_balance * 4.0  # 4:1 margin
        )
        
        self.accounts[account_id] = account
        
        # Persist to database
        await self._persist_account(account)
        
        logger.info(f"Created paper account {account_id} for strategy {strategy_id}")
        return account
    
    async def place_order(
        self,
        account_id: str,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PaperOrder:
        """Place paper trading order"""
        
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} not found")
        
        account = self.accounts[account_id]
        
        # Create order
        order = PaperOrder(
            order_id=f"order_{uuid.uuid4().hex[:12]}",
            strategy_id=account.strategy_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            metadata=metadata or {}
        )
        
        # Validate order
        validation_result = await self._validate_order(order, account)
        if not validation_result["valid"]:
            order.status = OrderStatus.REJECTED
            order.metadata["rejection_reason"] = validation_result["reason"]
            await self._persist_order(order)
            raise ValueError(f"Order rejected: {validation_result['reason']}")
        
        # Add to account
        account.open_orders[order.order_id] = order
        
        # Attempt immediate execution for market orders
        if order.order_type == OrderType.MARKET:
            await self._attempt_execution(order, account)
        
        await self._persist_order(order)
        logger.info(f"Placed order {order.order_id}: {side.value} {quantity} {symbol}")
        
        return order
    
    async def cancel_order(self, account_id: str, order_id: str) -> bool:
        """Cancel pending order"""
        
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} not found")
        
        account = self.accounts[account_id]
        
        if order_id not in account.open_orders:
            return False
        
        order = account.open_orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
        
        order.status = OrderStatus.CANCELLED
        del account.open_orders[order_id]
        account.order_history.append(order)
        
        await self._persist_order(order)
        logger.info(f"Cancelled order {order_id}")
        
        return True
    
    async def update_market_data(self, symbol: str, market_data: MarketData):
        """Update market data and trigger order processing"""
        
        self.market_data_cache[symbol] = market_data
        
        # Process pending orders for this symbol
        for account in self.accounts.values():
            pending_orders = [
                order for order in account.open_orders.values()
                if order.symbol == symbol and order.status == OrderStatus.PENDING
            ]
            
            for order in pending_orders:
                await self._attempt_execution(order, account)
        
        # Update position values
        await self._update_positions_for_symbol(symbol, market_data)
    
    async def _validate_order(self, order: PaperOrder, account: PaperAccount) -> Dict[str, Any]:
        """Validate order before placement"""
        
        # Check buying power for buy orders
        if order.side == OrderSide.BUY:
            if order.symbol not in self.market_data_cache:
                return {"valid": False, "reason": f"No market data available for {order.symbol}"}
            
            market_data = self.market_data_cache[order.symbol]
            estimated_cost = order.quantity * market_data.ask
            estimated_commission = self.execution_engine.calculate_total_costs(
                order.symbol, order.quantity, market_data.ask, order.side.value.lower()
            )["total_fees"]
            total_cost = estimated_cost + estimated_commission
            
            if total_cost > account.buying_power:
                return {"valid": False, "reason": "Insufficient buying power"}
        
        # Check position for sell orders
        elif order.side == OrderSide.SELL:
            current_position = account.positions.get(order.symbol)
            if not current_position or current_position.quantity < order.quantity:
                return {"valid": False, "reason": "Insufficient shares to sell"}
        
        # Check order size limits
        if order.quantity <= 0:
            return {"valid": False, "reason": "Order quantity must be positive"}
        
        if order.quantity > 100000:  # Reasonable upper limit
            return {"valid": False, "reason": "Order size too large"}
        
        return {"valid": True, "reason": None}
    
    async def _attempt_execution(self, order: PaperOrder, account: PaperAccount):
        """Attempt to execute order against market data"""
        
        if order.symbol not in self.market_data_cache:
            return
        
        market_data = self.market_data_cache[order.symbol]
        
        # Convert to market depth format
        market_depth = MarketDepth(
            symbol=order.symbol,
            timestamp=market_data.timestamp,
            bid_price=market_data.bid,
            ask_price=market_data.ask,
            bid_size=market_data.bid_size,
            ask_size=market_data.ask_size,
            last_price=market_data.last,
            volume=market_data.volume
        )
        
        # Calculate execution using advanced engine
        try:
            execution_result = await self.execution_engine.execute_order_async(
                symbol=order.symbol,
                side=order.side.value.lower(),
                quantity=order.quantity,
                order_type=OrderExecutionType.MARKET if order.order_type == OrderType.MARKET else OrderExecutionType.LIMIT,
                limit_price=order.price,
                market_depth=market_depth
            )
            
            if not execution_result["success"]:
                logger.warning(f"Order execution failed: {execution_result['reason']}")
                return
            
            execution_price = execution_result["avg_fill_price"]
            total_costs = execution_result["total_costs"]
            commission = total_costs["total_fees"]
            slippage = execution_result["total_slippage"]
            market_impact = execution_result.get("market_impact", 0.0)
            
        except Exception as e:
            logger.error(f"Advanced execution failed, falling back to simple execution: {e}")
            # Fallback to simple execution
            if order.side == OrderSide.BUY:
                execution_price = market_data.ask
            else:
                execution_price = market_data.bid
            
            # Simple commission calculation
            commission = max(0.005 * order.quantity, 1.0)
            slippage = abs(execution_price - market_data.last)
            market_impact = 0.0
        
        # Extract comprehensive fee breakdown if available
        fee_breakdown = total_costs.get("fee_breakdown", {}) if "total_costs" in locals() else {}
        exchange_type = self.execution_engine.__class__.__name__.replace("ExecutionEngine", "").upper()
        
        # Execute order
        execution = ExecutionDetails(
            execution_id=f"exec_{uuid.uuid4().hex[:12]}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            commission=commission,
            execution_time=datetime.now(),
            slippage=slippage,
            market_impact=market_impact,
            exchange_type=exchange_type,
            maker_fee=fee_breakdown.get("maker_fee", 0.0),
            taker_fee=fee_breakdown.get("taker_fee", 0.0),
            sec_fee=fee_breakdown.get("sec_fee", 0.0),
            taf_fee=fee_breakdown.get("taf_fee", 0.0),
            finra_orf=fee_breakdown.get("finra_orf", 0.0),
            clearing_fee=fee_breakdown.get("clearing_fee", 0.0),
            total_fees=commission,
            fill_probability=execution_result.get("fill_probability", 1.0) if "execution_result" in locals() else 1.0,
            depth_level=execution_result.get("depth_level", 1) if "execution_result" in locals() else 1
        )
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = execution_price
        order.commission = commission
        order.filled_at = execution.execution_time
        
        # Update account
        await self._process_execution(execution, account)
        
        # Move order from open to history
        if order.order_id in account.open_orders:
            del account.open_orders[order.order_id]
        account.order_history.append(order)
        
        await self._persist_execution(execution)
        logger.info(f"Executed order {order.order_id}: {order.quantity} @ ${execution_price:.2f}")
    
    async def _process_execution(self, execution: ExecutionDetails, account: PaperAccount):
        """Process order execution and update account"""
        
        symbol = execution.symbol
        
        if execution.side == OrderSide.BUY:
            # Buying shares
            total_cost = execution.quantity * execution.price + execution.commission
            account.cash_balance -= total_cost
            
            # Update position
            if symbol in account.positions:
                position = account.positions[symbol]
                total_shares = position.quantity + execution.quantity
                total_cost_basis = (position.quantity * position.avg_cost) + (execution.quantity * execution.price)
                position.quantity = total_shares
                position.avg_cost = total_cost_basis / total_shares if total_shares > 0 else 0
                position.side = PositionSide.LONG
            else:
                account.positions[symbol] = PaperPosition(
                    strategy_id=account.strategy_id,
                    symbol=symbol,
                    side=PositionSide.LONG,
                    quantity=execution.quantity,
                    avg_cost=execution.price,
                    market_value=execution.quantity * execution.price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0
                )
        
        else:  # SELL
            # Selling shares
            total_proceeds = execution.quantity * execution.price - execution.commission
            account.cash_balance += total_proceeds
            
            # Update position and calculate realized P&L
            if symbol in account.positions:
                position = account.positions[symbol]
                realized_pnl = execution.quantity * (execution.price - position.avg_cost) - execution.commission
                position.realized_pnl += realized_pnl
                position.quantity -= execution.quantity
                
                if position.quantity == 0:
                    position.side = PositionSide.FLAT
                    # Remove position if flat
                    # del account.positions[symbol]
                
                # Update total realized P&L
                account.total_pnl += realized_pnl
        
        # Update account totals
        await self._update_account_totals(account)
    
    async def _update_positions_for_symbol(self, symbol: str, market_data: MarketData):
        """Update position values for specific symbol"""
        
        for account in self.accounts.values():
            if symbol in account.positions:
                position = account.positions[symbol]
                if position.quantity > 0:
                    position.market_value = position.quantity * market_data.last
                    position.unrealized_pnl = position.quantity * (market_data.last - position.avg_cost)
                    position.last_updated = datetime.now()
            
            await self._update_account_totals(account)
    
    async def _update_account_totals(self, account: PaperAccount):
        """Update account total values"""
        
        # Calculate total position value
        total_position_value = sum(
            position.market_value for position in account.positions.values()
            if position.quantity > 0
        )
        
        # Calculate total unrealized P&L
        total_unrealized_pnl = sum(
            position.unrealized_pnl for position in account.positions.values()
            if position.quantity > 0
        )
        
        # Update account totals
        account.total_value = account.cash_balance + total_position_value
        
        # Calculate daily P&L (would need previous day's total_value in production)
        account.daily_pnl = total_unrealized_pnl  # Simplified
        
        # Update max drawdown
        drawdown = (account.total_value - account.initial_balance) / account.initial_balance
        if drawdown < account.max_drawdown:
            account.max_drawdown = drawdown
        
        # Update buying power (simplified calculation)
        account.buying_power = account.cash_balance * 2.0  # 2:1 margin for long positions
    
    def get_account_summary(self, account_id: str) -> Dict[str, Any]:
        """Get comprehensive account summary"""
        
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} not found")
        
        account = self.accounts[account_id]
        
        # Position summary
        positions_summary = []
        for position in account.positions.values():
            if position.quantity > 0:
                positions_summary.append({
                    "symbol": position.symbol,
                    "side": position.side.value,
                    "quantity": position.quantity,
                    "avg_cost": position.avg_cost,
                    "market_value": position.market_value,
                    "unrealized_pnl": position.unrealized_pnl,
                    "unrealized_pnl_pct": position.unrealized_pnl / (position.quantity * position.avg_cost) if position.avg_cost > 0 else 0
                })
        
        # Open orders summary
        open_orders_summary = []
        for order in account.open_orders.values():
            open_orders_summary.append({
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "order_type": order.order_type.value,
                "price": order.price,
                "status": order.status.value,
                "created_at": order.created_at.isoformat()
            })
        
        # Performance metrics
        total_return = (account.total_value - account.initial_balance) / account.initial_balance
        
        return {
            "account_id": account.account_id,
            "strategy_id": account.strategy_id,
            "account_value": {
                "total_value": account.total_value,
                "cash_balance": account.cash_balance,
                "buying_power": account.buying_power,
                "initial_balance": account.initial_balance
            },
            "performance": {
                "total_return": total_return,
                "total_pnl": account.total_pnl,
                "daily_pnl": account.daily_pnl,
                "max_drawdown": account.max_drawdown
            },
            "positions": positions_summary,
            "open_orders": open_orders_summary,
            "order_count": len(account.order_history),
            "last_updated": datetime.now().isoformat()
        }
    
    async def get_performance_metrics(self, account_id: str, start_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} not found")
        
        account = self.accounts[account_id]
        
        # Get historical performance data
        # In production, this would query time-series data from database
        # For now, we'll calculate based on current state
        
        total_return = (account.total_value - account.initial_balance) / account.initial_balance
        
        # Calculate trade statistics
        filled_orders = [order for order in account.order_history if order.status == OrderStatus.FILLED]
        
        # Group buy/sell pairs to calculate trade returns
        trade_returns = []
        positions_tracker = {}
        
        for order in sorted(filled_orders, key=lambda x: x.filled_at):
            symbol = order.symbol
            
            if symbol not in positions_tracker:
                positions_tracker[symbol] = {"quantity": 0, "avg_cost": 0.0}
            
            pos = positions_tracker[symbol]
            
            if order.side == OrderSide.BUY:
                # Add to position
                new_quantity = pos["quantity"] + order.filled_quantity
                new_avg_cost = (pos["quantity"] * pos["avg_cost"] + order.filled_quantity * order.avg_fill_price) / new_quantity
                positions_tracker[symbol] = {"quantity": new_quantity, "avg_cost": new_avg_cost}
            
            else:  # SELL
                # Calculate trade return
                if pos["quantity"] >= order.filled_quantity:
                    trade_return = (order.avg_fill_price - pos["avg_cost"]) / pos["avg_cost"]
                    trade_returns.append(trade_return)
                    positions_tracker[symbol]["quantity"] -= order.filled_quantity
        
        # Calculate metrics
        win_rate = sum(1 for ret in trade_returns if ret > 0) / len(trade_returns) if trade_returns else 0
        avg_win = np.mean([ret for ret in trade_returns if ret > 0]) if trade_returns else 0
        avg_loss = np.mean([ret for ret in trade_returns if ret < 0]) if trade_returns else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Sharpe ratio (simplified - would need daily returns time series)
        sharpe_ratio = total_return / 0.1 if total_return > 0 else 0  # Simplified
        
        return {
            "total_return": total_return,
            "total_trades": len(trade_returns),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": account.max_drawdown,
            "current_positions": len([p for p in account.positions.values() if p.quantity > 0])
        }
    
    async def _persist_account(self, account: PaperAccount):
        """Persist account to database"""
        try:
            conn = await asyncpg.connect(get_database_url())
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_accounts (
                    account_id VARCHAR(255) PRIMARY KEY,
                    strategy_id VARCHAR(255) NOT NULL,
                    initial_balance FLOAT NOT NULL,
                    cash_balance FLOAT NOT NULL,
                    total_value FLOAT NOT NULL,
                    buying_power FLOAT NOT NULL,
                    total_pnl FLOAT DEFAULT 0,
                    max_drawdown FLOAT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await conn.execute("""
                INSERT INTO paper_accounts 
                (account_id, strategy_id, initial_balance, cash_balance, total_value, buying_power)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (account_id) DO UPDATE SET
                    cash_balance = EXCLUDED.cash_balance,
                    total_value = EXCLUDED.total_value,
                    buying_power = EXCLUDED.buying_power,
                    total_pnl = EXCLUDED.total_pnl,
                    max_drawdown = EXCLUDED.max_drawdown,
                    updated_at = CURRENT_TIMESTAMP
            """, account.account_id, account.strategy_id, account.initial_balance,
                account.cash_balance, account.total_value, account.buying_power)
            
            await conn.close()
        except Exception as e:
            logger.error(f"Failed to persist account: {e}")
    
    async def _persist_order(self, order: PaperOrder):
        """Persist order to database"""
        try:
            conn = await asyncpg.connect(get_database_url())
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_orders (
                    order_id VARCHAR(255) PRIMARY KEY,
                    strategy_id VARCHAR(255) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    order_type VARCHAR(20) NOT NULL,
                    quantity INTEGER NOT NULL,
                    price FLOAT,
                    stop_price FLOAT,
                    status VARCHAR(20) NOT NULL,
                    filled_quantity INTEGER DEFAULT 0,
                    avg_fill_price FLOAT DEFAULT 0,
                    commission FLOAT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    filled_at TIMESTAMP,
                    metadata JSONB
                )
            """)
            
            await conn.execute("""
                INSERT INTO paper_orders 
                (order_id, strategy_id, symbol, side, order_type, quantity, price, 
                 stop_price, status, filled_quantity, avg_fill_price, commission, 
                 created_at, filled_at, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (order_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    filled_quantity = EXCLUDED.filled_quantity,
                    avg_fill_price = EXCLUDED.avg_fill_price,
                    commission = EXCLUDED.commission,
                    filled_at = EXCLUDED.filled_at
            """, order.order_id, order.strategy_id, order.symbol, order.side.value,
                order.order_type.value, order.quantity, order.price, order.stop_price,
                order.status.value, order.filled_quantity, order.avg_fill_price,
                order.commission, order.created_at, order.filled_at, order.metadata)
            
            await conn.close()
        except Exception as e:
            logger.error(f"Failed to persist order: {e}")
    
    async def _persist_execution(self, execution: ExecutionDetails):
        """Persist execution details to database"""
        try:
            conn = await asyncpg.connect(get_database_url())
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_executions (
                    execution_id VARCHAR(255) PRIMARY KEY,
                    order_id VARCHAR(255) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    quantity INTEGER NOT NULL,
                    price FLOAT NOT NULL,
                    commission FLOAT NOT NULL,
                    execution_time TIMESTAMP NOT NULL,
                    market_impact FLOAT DEFAULT 0,
                    slippage FLOAT DEFAULT 0,
                    exchange_type VARCHAR(20) DEFAULT 'NYSE',
                    maker_fee FLOAT DEFAULT 0,
                    taker_fee FLOAT DEFAULT 0,
                    sec_fee FLOAT DEFAULT 0,
                    taf_fee FLOAT DEFAULT 0,
                    finra_orf FLOAT DEFAULT 0,
                    clearing_fee FLOAT DEFAULT 0,
                    total_fees FLOAT DEFAULT 0,
                    fill_probability FLOAT DEFAULT 1.0,
                    depth_level INTEGER DEFAULT 1
                )
            """)
            
            await conn.execute("""
                INSERT INTO paper_executions 
                (execution_id, order_id, symbol, side, quantity, price, commission, 
                 execution_time, market_impact, slippage, exchange_type,
                 maker_fee, taker_fee, sec_fee, taf_fee, finra_orf, clearing_fee,
                 total_fees, fill_probability, depth_level)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
            """, execution.execution_id, execution.order_id, execution.symbol,
                execution.side.value, execution.quantity, execution.price,
                execution.commission, execution.execution_time,
                execution.market_impact, execution.slippage, execution.exchange_type,
                execution.maker_fee, execution.taker_fee, execution.sec_fee,
                execution.taf_fee, execution.finra_orf, execution.clearing_fee,
                execution.total_fees, execution.fill_probability, execution.depth_level)
            
            await conn.close()
        except Exception as e:
            logger.error(f"Failed to persist execution: {e}")

# Factory function
async def create_paper_trading_engine(
    initial_balance: float = 100000.0, 
    exchange_type: ExchangeType = ExchangeType.NYSE
) -> PaperTradingEngine:
    """Create paper trading engine with initial balance and exchange type"""
    return PaperTradingEngine(initial_balance, exchange_type)