"""
Portfolio Service - FastAPI application for portfolio management
"""

from fastapi import FastAPI, HTTPException, Depends, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import logging

from .core.database import get_db, create_tables
from .services.portfolio_service import portfolio_service
from .models.schemas import (
    Portfolio, PortfolioCreate, PortfolioUpdate, PortfolioDetail,
    Transaction, TransactionCreate, TransactionListResponse,
    Position, PositionListResponse,
    PerformanceMetrics, PerformanceSnapshot,
    WatchlistItem, WatchlistItemCreate,
    ErrorResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Portfolio Service",
    description="Portfolio management and performance tracking service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    # create_tables()  # Skip table creation since tables already exist
    logger.info("Portfolio Service started")

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Portfolio Service",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "portfolios": "/portfolios",
            "transactions": "/portfolios/{portfolio_id}/transactions",
            "positions": "/portfolios/{portfolio_id}/positions",
            "performance": "/portfolios/{portfolio_id}/performance",
            "watchlist": "/watchlist",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "portfolio-service",
        "timestamp": datetime.now().isoformat()
    }

# Portfolio endpoints
@app.post("/portfolios", response_model=Portfolio)
async def create_portfolio(
    portfolio: PortfolioCreate,
    db: Session = Depends(get_db)
):
    """Create a new portfolio"""
    try:
        return await portfolio_service.create_portfolio(db, portfolio)
    except Exception as e:
        logger.error(f"Error creating portfolio: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/portfolios", response_model=List[Portfolio])
async def get_user_portfolios(
    user_id: str = Query(..., description="User ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get all portfolios for a user"""
    return portfolio_service.get_user_portfolios(db, user_id, skip, limit)

@app.get("/portfolios/{portfolio_id}", response_model=PortfolioDetail)
async def get_portfolio_detail(
    portfolio_id: int = Path(..., description="Portfolio ID"),
    user_id: str = Query(..., description="User ID"),
    db: Session = Depends(get_db)
):
    """Get detailed portfolio information including positions and performance"""
    portfolio = portfolio_service.get_portfolio(db, portfolio_id, user_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Get positions
    positions = portfolio_service.get_portfolio_positions(db, portfolio_id, user_id)
    
    # Get recent transactions (last 10)
    recent_transactions = db.query(Transaction).filter(
        Transaction.portfolio_id == portfolio_id
    ).order_by(Transaction.created_at.desc()).limit(10).all()
    
    # Get performance metrics
    performance_metrics = await portfolio_service.calculate_performance_metrics(
        db, portfolio_id, user_id
    )
    
    return PortfolioDetail(
        **portfolio.__dict__,
        positions=positions,
        recent_transactions=recent_transactions,
        performance_metrics=performance_metrics
    )

@app.put("/portfolios/{portfolio_id}", response_model=Portfolio)
async def update_portfolio(
    portfolio_id: int = Path(..., description="Portfolio ID"),
    portfolio_update: PortfolioUpdate = ...,
    user_id: str = Query(..., description="User ID"),
    db: Session = Depends(get_db)
):
    """Update portfolio"""
    portfolio = await portfolio_service.update_portfolio(
        db, portfolio_id, user_id, portfolio_update
    )
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return portfolio

@app.delete("/portfolios/{portfolio_id}")
async def delete_portfolio(
    portfolio_id: int = Path(..., description="Portfolio ID"),
    user_id: str = Query(..., description="User ID"),
    db: Session = Depends(get_db)
):
    """Delete (deactivate) portfolio"""
    success = portfolio_service.delete_portfolio(db, portfolio_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return {"message": "Portfolio deleted successfully"}

# Transaction endpoints
@app.post("/portfolios/{portfolio_id}/transactions", response_model=Transaction)
async def add_transaction(
    portfolio_id: int = Path(..., description="Portfolio ID"),
    transaction: TransactionCreate = ...,
    user_id: str = Query(..., description="User ID"),
    db: Session = Depends(get_db)
):
    """Add a new transaction to portfolio"""
    # Verify portfolio ownership
    portfolio = portfolio_service.get_portfolio(db, portfolio_id, user_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    try:
        return await portfolio_service.add_transaction(db, portfolio_id, transaction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding transaction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/portfolios/{portfolio_id}/transactions", response_model=TransactionListResponse)
async def get_portfolio_transactions(
    portfolio_id: int = Path(..., description="Portfolio ID"),
    user_id: str = Query(..., description="User ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    transaction_type: Optional[str] = Query(None, description="Filter by transaction type"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    db: Session = Depends(get_db)
):
    """Get portfolio transactions with filtering and pagination"""
    # Verify portfolio ownership
    portfolio = portfolio_service.get_portfolio(db, portfolio_id, user_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    query = db.query(Transaction).filter(Transaction.portfolio_id == portfolio_id)
    
    if transaction_type:
        query = query.filter(Transaction.transaction_type == transaction_type)
    if symbol:
        query = query.filter(Transaction.symbol.ilike(f"%{symbol}%"))
    
    total_count = query.count()
    transactions = query.order_by(Transaction.created_at.desc()).offset(skip).limit(limit).all()
    
    return TransactionListResponse(
        transactions=transactions,
        total_count=total_count,
        total_pages=(total_count + limit - 1) // limit,
        current_page=(skip // limit) + 1
    )

# Position endpoints
@app.get("/portfolios/{portfolio_id}/positions", response_model=PositionListResponse)
async def get_portfolio_positions(
    portfolio_id: int = Path(..., description="Portfolio ID"),
    user_id: str = Query(..., description="User ID"),
    db: Session = Depends(get_db)
):
    """Get all positions for a portfolio"""
    # Verify portfolio ownership
    portfolio = portfolio_service.get_portfolio(db, portfolio_id, user_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    positions = portfolio_service.get_portfolio_positions(db, portfolio_id, user_id)
    
    total_value = sum(p.market_value or 0 for p in positions)
    total_unrealized_pnl = sum(p.unrealized_pnl or 0 for p in positions)
    
    return PositionListResponse(
        positions=positions,
        total_value=total_value,
        total_unrealized_pnl=total_unrealized_pnl
    )

# Performance endpoints
@app.get("/portfolios/{portfolio_id}/performance", response_model=PerformanceMetrics)
async def get_portfolio_performance(
    portfolio_id: int = Path(..., description="Portfolio ID"),
    user_id: str = Query(..., description="User ID"),
    db: Session = Depends(get_db)
):
    """Get comprehensive portfolio performance metrics"""
    # Verify portfolio ownership
    portfolio = portfolio_service.get_portfolio(db, portfolio_id, user_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    performance = await portfolio_service.calculate_performance_metrics(
        db, portfolio_id, user_id
    )
    
    if not performance:
        raise HTTPException(status_code=404, detail="Performance data not available")
    
    return performance

# Watchlist endpoints
@app.post("/watchlist", response_model=WatchlistItem)
async def add_to_watchlist(
    watchlist_item: WatchlistItemCreate,
    db: Session = Depends(get_db)
):
    """Add symbol to user's watchlist"""
    try:
        return await portfolio_service.add_to_watchlist(db, watchlist_item)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/watchlist", response_model=List[WatchlistItem])
async def get_user_watchlist(
    user_id: str = Query(..., description="User ID"),
    db: Session = Depends(get_db)
):
    """Get user's watchlist"""
    return portfolio_service.get_user_watchlist(db, user_id)

@app.delete("/watchlist/{item_id}")
async def remove_from_watchlist(
    item_id: int = Path(..., description="Watchlist item ID"),
    user_id: str = Query(..., description="User ID"),
    db: Session = Depends(get_db)
):
    """Remove symbol from user's watchlist"""
    item = db.query(WatchlistItem).filter(
        WatchlistItem.id == item_id,
        WatchlistItem.user_id == user_id
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Watchlist item not found")
    
    db.delete(item)
    db.commit()
    return {"message": "Item removed from watchlist"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info")