"""
Alerts API Endpoint
GET /api/v1/alerts - Get user alerts and daily caps
"""
from fastapi import APIRouter, Header, Query, HTTPException, status
from typing import List
from pydantic import BaseModel
import structlog

from ...core.contracts import Alert, DailyCap
from ...aggregation.alert_aggregator import AlertAggregator
from ...upstream.portfolio_client import PortfolioClient
from ...config import settings

logger = structlog.get_logger(__name__)

router = APIRouter()


class AlertResponse(BaseModel):
    """Alert response model"""
    alerts: List[Alert]
    daily_cap: DailyCap


def get_alert_aggregator() -> AlertAggregator:
    """Get alert aggregator instance"""
    portfolio_client = PortfolioClient(
        base_url=settings.PORTFOLIO_SERVICE_URL,
        timeout_ms=settings.PORTFOLIO_TIMEOUT_MS
    )
    return AlertAggregator(portfolio_client=portfolio_client)


@router.get("/alerts", response_model=AlertResponse)
async def get_alerts(
    user_id: str = Header(..., alias="X-User-ID"),
    mode: str = Query("beginner", description="Mode: beginner or expert")
):
    """
    Get user alerts and daily trading caps

    **Request:**
    - Header: `X-User-ID` - User identifier
    - Query: `mode` - "beginner" or "expert" (default: beginner)

    **Response:**
    - `alerts` - List of active alerts
    - `daily_cap` - Daily trading cap information

    **Alert Types:**
    - **Error**: Blocking issues (daily limit reached, loss limit hit)
    - **Warning**: Important notices (approaching limits, concentration risk)
    - **Info**: Helpful tips (market conditions, profit opportunities)

    **Example:**
    ```
    GET /api/v1/alerts?mode=beginner
    X-User-ID: user123
    ```

    **Response:**
    ```json
    {
      "alerts": [
        {
          "severity": "warning",
          "title": "Approaching Daily Trade Limit",
          "message": "You have 1 trade remaining today (limit: 3 trades).",
          "action_required": false
        }
      ],
      "daily_cap": {
        "trades_today": 2,
        "trades_remaining": 1,
        "cap_reason": null
      }
    }
    ```
    """
    logger.info("alerts_request_received", user_id=user_id, mode=mode)

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

    try:
        # Get alerts
        aggregator = get_alert_aggregator()
        alerts, daily_cap = await aggregator.get_alerts(user_id=user_id, mode=mode)

        logger.info(
            "alerts_request_completed",
            user_id=user_id,
            alert_count=len(alerts),
            trades_remaining=daily_cap.trades_remaining
        )

        return AlertResponse(alerts=alerts, daily_cap=daily_cap)

    except Exception as e:
        logger.error(
            "alerts_request_failed",
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "ALERTS_FETCH_FAILED",
                    "message": "Failed to fetch alerts. Please try again."
                }
            }
        )
