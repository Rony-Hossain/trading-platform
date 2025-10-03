"""
Plan API Endpoint
GET /api/v1/plan - Get personalized trading plan
"""
from fastapi import APIRouter, Query, Header, HTTPException, status
from typing import Optional, List
import structlog

from ...core.contracts import PlanResponse
from ...aggregation.plan_aggregator import PlanAggregator
from ...upstream.inference_client import InferenceClient
from ...upstream.forecast_client import ForecastClient
from ...upstream.sentiment_client import SentimentClient
from ...upstream.portfolio_client import PortfolioClient
from ...config import settings

logger = structlog.get_logger(__name__)

router = APIRouter()


# Initialize upstream clients (will be replaced by dependency injection)
def get_plan_aggregator() -> PlanAggregator:
    """Get plan aggregator instance"""
    inference_client = InferenceClient(
        base_url=settings.INFERENCE_SERVICE_URL,
        timeout_ms=settings.INFERENCE_TIMEOUT_MS
    )
    forecast_client = ForecastClient(
        base_url=settings.FORECAST_SERVICE_URL,
        timeout_ms=settings.FORECAST_TIMEOUT_MS
    )
    sentiment_client = SentimentClient(
        base_url=settings.SENTIMENT_SERVICE_URL,
        timeout_ms=settings.SENTIMENT_TIMEOUT_MS
    )
    portfolio_client = PortfolioClient(
        base_url=settings.PORTFOLIO_SERVICE_URL,
        timeout_ms=settings.PORTFOLIO_TIMEOUT_MS
    )

    return PlanAggregator(
        inference_client=inference_client,
        forecast_client=forecast_client,
        sentiment_client=sentiment_client,
        portfolio_client=portfolio_client
    )


@router.get("/plan", response_model=PlanResponse)
async def get_plan(
    user_id: str = Header(..., alias="X-User-ID"),
    watchlist: Optional[str] = Query(None, description="Comma-separated list of symbols"),
    mode: str = Query("beginner", description="Mode: beginner or expert")
):
    """
    Get personalized trading plan

    **Request:**
    - Header: `X-User-ID` - User identifier
    - Query: `watchlist` - Optional comma-separated symbols (e.g., "AAPL,MSFT,GOOGL")
    - Query: `mode` - "beginner" or "expert" (default: beginner)

    **Response:**
    - `request_id` - Unique request identifier for tracking
    - `picks` - List of trading recommendations
    - `daily_cap_reached` - Whether daily trade limit is reached
    - `degraded_fields` - List of degraded upstream services (if any)
    - `metadata` - Additional context (model version, timestamp, etc.)

    **Example:**
    ```
    GET /api/v1/plan?watchlist=AAPL,MSFT&mode=beginner
    X-User-ID: user123
    ```

    **Beginner Mode:**
    - Max 3 picks per day
    - Only high-confidence, low-risk recommendations
    - Stop loss required for all buys
    - Position size limited to 10% of portfolio

    **Expert Mode:**
    - More picks with detailed technical analysis
    - Progressive disclosure of advanced metrics
    - No artificial constraints
    """
    logger.info(
        "plan_request_received",
        user_id=user_id,
        watchlist=watchlist,
        mode=mode
    )

    # Validate mode
    if mode not in ["beginner", "expert"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "INVALID_MODE",
                    "message": "Mode must be 'beginner' or 'expert'"
                }
            }
        )

    # Parse watchlist
    watchlist_symbols = None
    if watchlist:
        watchlist_symbols = [s.strip().upper() for s in watchlist.split(",")]
        if len(watchlist_symbols) > 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "code": "WATCHLIST_TOO_LARGE",
                        "message": "Watchlist cannot exceed 20 symbols"
                    }
                }
            )

    try:
        # Generate plan
        aggregator = get_plan_aggregator()
        plan = await aggregator.generate_plan(
            user_id=user_id,
            watchlist=watchlist_symbols,
            mode=mode
        )

        logger.info(
            "plan_request_completed",
            user_id=user_id,
            request_id=plan.request_id,
            pick_count=len(plan.picks)
        )

        return plan

    except Exception as e:
        logger.error(
            "plan_request_failed",
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "PLAN_GENERATION_FAILED",
                    "message": "Failed to generate trading plan. Please try again."
                }
            }
        )
