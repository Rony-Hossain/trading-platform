"""
Positions API Endpoint
GET /api/v1/positions - Get user positions with plain-English context
"""
from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
import structlog

from ...upstream.portfolio_client import PortfolioClient
from ...config import settings

logger = structlog.get_logger(__name__)

router = APIRouter()


class PositionEnriched(BaseModel):
    """Enriched position with beginner-friendly context"""
    symbol: str
    shares: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    cost_basis: float
    weight: float  # Portfolio weight

    # Beginner-friendly fields
    status: str  # "winning", "losing", "break_even"
    status_message: str  # Plain English explanation
    suggestion: Optional[str] = None  # Optional action suggestion


class PositionsResponse(BaseModel):
    """Positions response"""
    positions: List[PositionEnriched]
    total_positions: int
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    concentration_risk: str  # "low", "medium", "high"


def get_portfolio_client() -> PortfolioClient:
    """Get portfolio client instance"""
    return PortfolioClient(
        base_url=settings.PORTFOLIO_SERVICE_URL,
        timeout_ms=settings.PORTFOLIO_TIMEOUT_MS
    )


@router.get("/positions", response_model=PositionsResponse)
async def get_positions(
    user_id: str = Header(..., alias="X-User-ID")
):
    """
    Get user positions with beginner-friendly context

    **Request:**
    - Header: `X-User-ID` - User identifier

    **Response:**
    - `positions` - List of positions with plain-English context
    - `total_positions` - Number of positions
    - `total_value` - Total market value
    - `total_pnl` - Total unrealized P&L
    - `total_pnl_pct` - Total unrealized P&L percentage
    - `concentration_risk` - Portfolio concentration level

    **Position Fields:**
    - **status**: "winning" (positive P&L), "losing" (negative P&L), "break_even"
    - **status_message**: Plain English explanation
    - **suggestion**: Optional action suggestion

    **Example:**
    ```
    GET /api/v1/positions
    X-User-ID: user123
    ```

    **Response:**
    ```json
    {
      "positions": [
        {
          "symbol": "AAPL",
          "shares": 100,
          "avg_cost": 170.50,
          "current_price": 175.50,
          "market_value": 17550.00,
          "unrealized_pnl": 500.00,
          "unrealized_pnl_pct": 2.93,
          "weight": 0.35,
          "status": "winning",
          "status_message": "Up $500 (2.9%) - holding steady",
          "suggestion": null
        }
      ],
      "total_positions": 5,
      "total_value": 50000.00,
      "total_pnl": 1250.00,
      "total_pnl_pct": 2.56,
      "concentration_risk": "low"
    }
    ```
    """
    logger.info("positions_request_received", user_id=user_id)

    try:
        # Fetch positions from portfolio service
        portfolio_client = get_portfolio_client()
        positions_data = await portfolio_client.get_positions(user_id)
        portfolio_summary = await portfolio_client.get_portfolio(user_id)

        # Enrich positions with beginner-friendly context
        enriched_positions = []
        total_pnl = 0.0

        for position in positions_data.get("positions", []):
            enriched = _enrich_position(position)
            enriched_positions.append(enriched)
            total_pnl += position.get("unrealized_pnl", 0)

        # Calculate totals
        total_value = portfolio_summary.get("positions_value", 0)
        total_pnl_pct = (total_pnl / (total_value - total_pnl) * 100) if total_value > 0 else 0

        response = PositionsResponse(
            positions=enriched_positions,
            total_positions=len(enriched_positions),
            total_value=total_value,
            total_pnl=total_pnl,
            total_pnl_pct=round(total_pnl_pct, 2),
            concentration_risk=positions_data.get("concentration_risk", "low")
        )

        logger.info(
            "positions_request_completed",
            user_id=user_id,
            position_count=len(enriched_positions)
        )

        return response

    except Exception as e:
        logger.error(
            "positions_request_failed",
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "POSITIONS_FETCH_FAILED",
                    "message": "Failed to fetch positions. Please try again."
                }
            }
        )


def _enrich_position(position: dict) -> PositionEnriched:
    """Enrich position with beginner-friendly context"""
    unrealized_pnl = position.get("unrealized_pnl", 0)
    unrealized_pnl_pct = position.get("unrealized_pnl_pct", 0)

    # Determine status
    if unrealized_pnl_pct > 1:
        status = "winning"
    elif unrealized_pnl_pct < -1:
        status = "losing"
    else:
        status = "break_even"

    # Build status message
    if status == "winning":
        status_message = f"Up ${abs(unrealized_pnl):.0f} ({unrealized_pnl_pct:.1f}%) - holding steady"
        suggestion = None
        if unrealized_pnl_pct > 20:
            suggestion = "Consider taking some profit - this position is up significantly"
    elif status == "losing":
        status_message = f"Down ${abs(unrealized_pnl):.0f} ({abs(unrealized_pnl_pct):.1f}%) - monitor closely"
        suggestion = None
        if unrealized_pnl_pct < -10:
            suggestion = "This position is down significantly - review your exit strategy"
    else:
        status_message = "Around break-even point"
        suggestion = None

    return PositionEnriched(
        symbol=position.get("symbol"),
        shares=position.get("shares"),
        avg_cost=position.get("avg_cost"),
        current_price=position.get("current_price"),
        market_value=position.get("market_value"),
        unrealized_pnl=unrealized_pnl,
        unrealized_pnl_pct=unrealized_pnl_pct,
        cost_basis=position.get("cost_basis"),
        weight=position.get("weight"),
        status=status,
        status_message=status_message,
        suggestion=suggestion
    )
