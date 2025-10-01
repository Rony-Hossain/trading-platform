"""
Portfolio Service - Business logic for portfolio management
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from decimal import Decimal
import httpx
import logging

from ..models.portfolio import (
    Portfolio, Position, Watchlist, WatchlistItem
)
from ..models.schemas import (
    PortfolioCreate, PortfolioUpdate, TransactionCreate,
    PerformanceMetrics, MarketDataUpdate, WatchlistItemCreate
)

logger = logging.getLogger(__name__)

class PortfolioService:
    """Service for portfolio management operations"""
    
    def __init__(self):
        self.market_data_url = "http://localhost:8002"
    
    # Portfolio CRUD operations
    async def create_portfolio(self, db: Session, portfolio: PortfolioCreate) -> Portfolio:
        """Create a new portfolio"""
        db_portfolio = Portfolio(
            user_id=portfolio.user_id,
            name=portfolio.name,
            description=portfolio.description,
            initial_balance=portfolio.initial_balance,
            cash_balance=portfolio.initial_balance
        )
        db.add(db_portfolio)
        db.commit()
        db.refresh(db_portfolio)
        
        # Create initial cash deposit transaction if balance > 0
        if portfolio.initial_balance > 0:
            await self.add_transaction(
                db, 
                db_portfolio.id,
                TransactionCreate(
                    transaction_type=TransactionType.DEPOSIT,
                    asset_type=AssetType.CASH,
                    total_amount=portfolio.initial_balance,
                    transaction_date=datetime.now()
                )
            )
        
        return db_portfolio
    
    def get_portfolio(self, db: Session, portfolio_id: int, user_id: str) -> Optional[Portfolio]:
        """Get portfolio by ID and user"""
        return db.query(Portfolio).filter(
            and_(Portfolio.id == portfolio_id, Portfolio.user_id == user_id)
        ).first()
    
    def get_user_portfolios(self, db: Session, user_id: str, skip: int = 0, limit: int = 100) -> List[Portfolio]:
        """Get all portfolios for a user"""
        return db.query(Portfolio).filter(
            Portfolio.user_id == user_id
        ).offset(skip).limit(limit).all()
    
    async def update_portfolio(self, db: Session, portfolio_id: int, user_id: str, 
                              portfolio_update: PortfolioUpdate) -> Optional[Portfolio]:
        """Update portfolio"""
        db_portfolio = self.get_portfolio(db, portfolio_id, user_id)
        if not db_portfolio:
            return None
        
        update_data = portfolio_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_portfolio, field, value)
        
        db.commit()
        db.refresh(db_portfolio)
        return db_portfolio
    
    def delete_portfolio(self, db: Session, portfolio_id: int, user_id: str) -> bool:
        """Delete portfolio (soft delete by marking inactive)"""
        db_portfolio = self.get_portfolio(db, portfolio_id, user_id)
        if not db_portfolio:
            return False
        
        db_portfolio.is_active = False
        db.commit()
        return True
    
    # Transaction operations
    async def add_transaction(self, db: Session, portfolio_id: int, 
                             transaction: TransactionCreate) -> Transaction:
        """Add a new transaction and update positions"""
        # Create transaction record
        db_transaction = Transaction(
            portfolio_id=portfolio_id,
            **transaction.dict()
        )
        db.add(db_transaction)
        db.flush()  # Get the transaction ID
        
        # Update portfolio and positions based on transaction type
        await self._process_transaction(db, db_transaction)
        
        db.commit()
        db.refresh(db_transaction)
        return db_transaction
    
    async def _process_transaction(self, db: Session, transaction: Transaction):
        """Process transaction and update portfolio/positions"""
        portfolio = db.query(Portfolio).filter(Portfolio.id == transaction.portfolio_id).first()
        
        if transaction.transaction_type == TransactionType.BUY:
            await self._process_buy_transaction(db, transaction, portfolio)
        elif transaction.transaction_type == TransactionType.SELL:
            await self._process_sell_transaction(db, transaction, portfolio)
        elif transaction.transaction_type == TransactionType.DEPOSIT:
            portfolio.cash_balance += transaction.total_amount
        elif transaction.transaction_type == TransactionType.WITHDRAWAL:
            portfolio.cash_balance -= transaction.total_amount
        elif transaction.transaction_type == TransactionType.DIVIDEND:
            portfolio.cash_balance += transaction.total_amount
    
    async def _process_buy_transaction(self, db: Session, transaction: Transaction, portfolio: Portfolio):
        """Process buy transaction"""
        # Find or create position
        position = db.query(Position).filter(
            and_(
                Position.portfolio_id == transaction.portfolio_id,
                Position.symbol == transaction.symbol,
                Position.asset_type == transaction.asset_type
            )
        ).first()
        
        total_cost = transaction.total_amount + transaction.fees
        
        if position:
            # Update existing position
            old_total_cost = position.total_cost_basis
            old_quantity = position.quantity
            
            new_quantity = old_quantity + transaction.quantity
            new_total_cost = old_total_cost + total_cost
            new_average_cost = new_total_cost / new_quantity
            
            position.quantity = new_quantity
            position.total_cost_basis = new_total_cost
            position.average_cost = new_average_cost
        else:
            # Create new position
            position = Position(
                portfolio_id=transaction.portfolio_id,
                symbol=transaction.symbol,
                asset_type=transaction.asset_type,
                quantity=transaction.quantity,
                total_cost_basis=total_cost,
                average_cost=transaction.price,
                first_purchase_date=transaction.transaction_date
            )
            db.add(position)
        
        # Update portfolio cash balance
        portfolio.cash_balance -= total_cost
        
        # Link transaction to position
        db.flush()
        transaction.position_id = position.id
    
    async def _process_sell_transaction(self, db: Session, transaction: Transaction, portfolio: Portfolio):
        """Process sell transaction"""
        position = db.query(Position).filter(
            and_(
                Position.portfolio_id == transaction.portfolio_id,
                Position.symbol == transaction.symbol,
                Position.asset_type == transaction.asset_type
            )
        ).first()
        
        if not position or position.quantity < transaction.quantity:
            raise ValueError("Insufficient shares to sell")
        
        # Calculate realized P&L
        cost_basis_per_share = position.average_cost
        proceeds = transaction.total_amount - transaction.fees
        cost_basis = cost_basis_per_share * transaction.quantity
        realized_pnl = proceeds - cost_basis
        
        # Update position
        position.quantity -= transaction.quantity
        position.total_cost_basis -= cost_basis
        position.realized_pnl += realized_pnl
        
        # Update portfolio cash balance
        portfolio.cash_balance += proceeds
        
        # If position is completely sold, mark it
        if position.quantity == 0:
            position.average_cost = 0
        
        # Link transaction to position
        transaction.position_id = position.id
    
    # Position operations
    def get_portfolio_positions(self, db: Session, portfolio_id: int, user_id: str) -> List[Position]:
        """Get all positions for a portfolio"""
        portfolio = self.get_portfolio(db, portfolio_id, user_id)
        if not portfolio:
            return []
        
        return db.query(Position).filter(
            and_(Position.portfolio_id == portfolio_id, Position.quantity > 0)
        ).all()
    
    async def update_positions_market_data(self, db: Session, portfolio_id: int, 
                                          market_updates: List[MarketDataUpdate]):
        """Update positions with current market data"""
        for update in market_updates:
            positions = db.query(Position).filter(
                and_(
                    Position.portfolio_id == portfolio_id,
                    Position.symbol == update.symbol,
                    Position.quantity > 0
                )
            ).all()
            
            for position in positions:
                position.current_price = update.current_price
                position.market_value = position.quantity * update.current_price
                position.unrealized_pnl = position.market_value - position.total_cost_basis
                position.last_updated = update.last_updated
        
        db.commit()
    
    # Performance calculation
    async def calculate_performance_metrics(self, db: Session, portfolio_id: int, user_id: str) -> Optional[PerformanceMetrics]:
        """Calculate comprehensive performance metrics"""
        portfolio = self.get_portfolio(db, portfolio_id, user_id)
        if not portfolio:
            return None
        
        positions = self.get_portfolio_positions(db, portfolio_id, user_id)
        
        # Update positions with current market data
        await self._fetch_and_update_market_data(db, positions)
        
        # Calculate metrics
        total_positions_value = sum(p.market_value or 0 for p in positions)
        total_value = portfolio.cash_balance + total_positions_value
        total_unrealized_pnl = sum(p.unrealized_pnl or 0 for p in positions)
        total_realized_pnl = sum(p.realized_pnl or 0 for p in positions)
        
        total_return = total_value - portfolio.initial_balance
        total_return_percent = (total_return / portfolio.initial_balance * 100) if portfolio.initial_balance > 0 else 0
        
        # Get recent performance for daily return
        yesterday = datetime.now() - timedelta(days=1)
        recent_snapshot = db.query(PerformanceSnapshot).filter(
            and_(
                PerformanceSnapshot.portfolio_id == portfolio_id,
                PerformanceSnapshot.snapshot_date >= yesterday
            )
        ).order_by(desc(PerformanceSnapshot.snapshot_date)).first()
        
        daily_return = None
        daily_return_percent = None
        if recent_snapshot:
            daily_return = total_value - recent_snapshot.total_value
            daily_return_percent = (daily_return / recent_snapshot.total_value * 100) if recent_snapshot.total_value > 0 else 0
        
        # Calculate position breakdown
        largest_position_percent = None
        if positions and total_positions_value > 0:
            largest_position_value = max(p.market_value or 0 for p in positions)
            largest_position_percent = (largest_position_value / total_positions_value * 100)
        
        return PerformanceMetrics(
            total_value=total_value,
            total_return=total_return,
            total_return_percent=total_return_percent,
            daily_return=daily_return,
            daily_return_percent=daily_return_percent,
            realized_pnl=total_realized_pnl,
            unrealized_pnl=total_unrealized_pnl,
            cash_balance=portfolio.cash_balance,
            positions_value=total_positions_value,
            positions_count=len(positions),
            largest_position_percent=largest_position_percent,
            volatility_30d=None,  # Would require historical data
            max_drawdown=None,    # Would require historical snapshots
            sharpe_ratio=None,    # Would require risk-free rate and volatility
            benchmark_return=None,
            alpha=None,
            beta=None,
            sector_allocation={}
        )
    
    async def _fetch_and_update_market_data(self, db: Session, positions: List[Position]):
        """Fetch current market data for positions"""
        if not positions:
            return
        
        symbols = list(set(p.symbol for p in positions))
        
        try:
            async with httpx.AsyncClient() as client:
                for symbol in symbols:
                    try:
                        response = await client.get(
                            f"{self.market_data_url}/stocks/{symbol}/price",
                            timeout=5.0
                        )
                        if response.status_code == 200:
                            data = response.json()
                            current_price = data.get("current_price", 0)
                            
                            # Update all positions for this symbol
                            for position in positions:
                                if position.symbol == symbol:
                                    position.current_price = current_price
                                    position.market_value = position.quantity * current_price
                                    position.unrealized_pnl = position.market_value - position.total_cost_basis
                                    position.last_updated = datetime.now()
                    except Exception as e:
                        logger.warning(f"Failed to fetch price for {symbol}: {e}")
            
            db.commit()
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
    
    # Watchlist operations
    async def add_to_watchlist(self, db: Session, watchlist_item: WatchlistItemCreate) -> WatchlistItem:
        """Add symbol to user's watchlist"""
        # Check if already exists
        existing = db.query(WatchlistItem).filter(
            and_(
                WatchlistItem.user_id == watchlist_item.user_id,
                WatchlistItem.symbol == watchlist_item.symbol
            )
        ).first()
        
        if existing:
            raise ValueError(f"Symbol {watchlist_item.symbol} already in watchlist")
        
        db_item = WatchlistItem(**watchlist_item.dict())
        db.add(db_item)
        db.commit()
        db.refresh(db_item)
        return db_item
    
    def get_user_watchlist(self, db: Session, user_id: str) -> List[WatchlistItem]:
        """Get user's watchlist"""
        return db.query(WatchlistItem).filter(WatchlistItem.user_id == user_id).all()

# Global service instance
portfolio_service = PortfolioService()