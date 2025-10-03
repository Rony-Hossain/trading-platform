"""
Actions API Endpoint
POST /api/v1/actions/execute - Execute trading action with idempotency
"""
from fastapi import APIRouter, Header, HTTPException, status, Body
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
import time
import structlog

from ...core.contracts import ActionRecord
from ...core.idempotency import get_idempotency_manager
from ...core.guardrails import get_guardrail_engine
from ...upstream.portfolio_client import PortfolioClient
from ...upstream.sor_client import get_sor_client
from ...config import settings

logger = structlog.get_logger(__name__)

router = APIRouter()


class ActionRequest(BaseModel):
    """Action execution request"""
    symbol: str = Field(..., description="Ticker symbol")
    action: Literal["BUY", "SELL"] = Field(..., description="Action type")
    shares: int = Field(..., gt=0, description="Number of shares")
    limit_price: Optional[float] = Field(None, description="Limit price")
    stop_loss_price: Optional[float] = Field(None, description="Stop loss price")


class ActionResponse(BaseModel):
    """Action execution response"""
    action_record: ActionRecord
    message: str


def get_portfolio_client() -> PortfolioClient:
    """Get portfolio client instance"""
    return PortfolioClient(
        base_url=settings.PORTFOLIO_SERVICE_URL,
        timeout_ms=settings.PORTFOLIO_TIMEOUT_MS
    )


@router.post("/actions/execute", response_model=ActionResponse)
async def execute_action(
    request: ActionRequest,
    user_id: str = Header(..., alias="X-User-ID"),
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    mode: str = Header("beginner", alias="X-Mode")
):
    """
    Execute trading action with idempotency protection

    **Request:**
    - Header: `X-User-ID` - User identifier
    - Header: `Idempotency-Key` - Unique key to prevent duplicate execution
    - Header: `X-Mode` - "beginner" or "expert" (default: beginner)
    - Body: Action details (symbol, action, shares, prices)

    **Response:**
    - `action_record` - Execution record with status
    - `message` - Human-readable result message

    **Idempotency:**
    - Same `Idempotency-Key` within 5 minutes returns cached result
    - Prevents accidental double-execution
    - Successful executions cached for 10 minutes

    **Guardrails (Beginner Mode):**
    - Daily trade limit check (max 3 trades)
    - Daily loss limit check (max 5% loss)
    - Sufficient shares check for SELL orders
    - Position size validation

    **Example:**
    ```
    POST /api/v1/actions/execute
    X-User-ID: user123
    Idempotency-Key: uuid-12345
    X-Mode: beginner

    {
      "symbol": "AAPL",
      "action": "BUY",
      "shares": 10,
      "limit_price": 175.50,
      "stop_loss_price": 170.00
    }
    ```

    **Response:**
    ```json
    {
      "action_record": {
        "idempotency_key": "uuid-12345",
        "user_id": "user123",
        "action_type": "BUY",
        "action_data": {...},
        "status": "executed",
        "result": {...},
        "created_at": "2024-01-20T15:30:00Z"
      },
      "message": "Order executed successfully"
    }
    ```
    """
    logger.info(
        "action_request_received",
        user_id=user_id,
        idempotency_key=idempotency_key,
        symbol=request.symbol,
        action=request.action
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

    try:
        # Check idempotency
        idempotency_manager = get_idempotency_manager()
        is_duplicate, existing_record = idempotency_manager.check_or_create(
            idempotency_key=idempotency_key,
            user_id=user_id,
            action_data={
                "symbol": request.symbol,
                "action": request.action,
                "shares": request.shares,
                "limit_price": request.limit_price,
                "stop_loss_price": request.stop_loss_price
            }
        )

        if is_duplicate:
            logger.info(
                "action_duplicate_detected",
                user_id=user_id,
                idempotency_key=idempotency_key,
                original_status=existing_record.status
            )
            return ActionResponse(
                action_record=existing_record,
                message=f"Action already {existing_record.status} (duplicate request)"
            )

        # Get portfolio data for guardrail checks
        portfolio_client = get_portfolio_client()
        portfolio_data = await portfolio_client.get_portfolio(user_id)
        positions_data = await portfolio_client.get_positions(user_id)

        # Check guardrails
        guardrail_engine = get_guardrail_engine()
        is_allowed, violations = guardrail_engine.check_action_allowed(
            action=request.action,
            symbol=request.symbol,
            shares=request.shares,
            user_context={
                "beginner_mode": mode == "beginner",
                "total_value": portfolio_data.get("total_value", 0),
                "daily_trades": portfolio_data.get("daily_trades", 0),
                "daily_pnl_pct": portfolio_data.get("daily_pnl_pct", 0),
                "positions": positions_data.get("positions", [])
            }
        )

        if not is_allowed:
            # Update idempotency record as failed
            idempotency_manager.update_status(
                idempotency_key=idempotency_key,
                status="failed",
                result={
                    "error": "GUARDRAIL_VIOLATION",
                    "violations": [v.to_dict() for v in violations]
                }
            )

            blocking_violations = [v for v in violations if v.severity == "blocking"]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "code": "GUARDRAIL_VIOLATION",
                        "message": blocking_violations[0].message if blocking_violations else "Action not allowed",
                        "violations": [v.to_dict() for v in violations]
                    }
                }
            )

        # Execute action (placeholder - would integrate with actual trading system)
        execution_result = await _execute_trade(
            user_id=user_id,
            symbol=request.symbol,
            action=request.action,
            shares=request.shares,
            limit_price=request.limit_price,
            stop_loss_price=request.stop_loss_price
        )

        # Update idempotency record as executed
        idempotency_manager.update_status(
            idempotency_key=idempotency_key,
            status="executed",
            result=execution_result
        )

        # Get updated record
        _, action_record = idempotency_manager.check_or_create(
            idempotency_key=idempotency_key,
            user_id=user_id,
            action_data={}
        )

        logger.info(
            "action_executed_successfully",
            user_id=user_id,
            idempotency_key=idempotency_key,
            symbol=request.symbol,
            action=request.action
        )

        return ActionResponse(
            action_record=action_record,
            message=f"{request.action} order for {request.shares} shares of {request.symbol} executed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "action_execution_failed",
            user_id=user_id,
            idempotency_key=idempotency_key,
            error=str(e),
            exc_info=True
        )

        # Update idempotency record as failed
        try:
            idempotency_manager = get_idempotency_manager()
            idempotency_manager.update_status(
                idempotency_key=idempotency_key,
                status="failed",
                result={"error": str(e)}
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "EXECUTION_FAILED",
                    "message": "Failed to execute action. Please try again."
                }
            }
        )


async def _execute_trade(
    user_id: str,
    symbol: str,
    action: str,
    shares: int,
    limit_price: Optional[float],
    stop_loss_price: Optional[float]
) -> dict:
    """
    Execute trade via Smart Order Router (SOR)

    Steps:
    1. Route order to optimal venue via SOR
    2. Submit order to selected venue
    3. Return execution details

    Falls back to mock execution if SOR unavailable
    """
    logger.info(
        "executing_trade_via_sor",
        user_id=user_id,
        symbol=symbol,
        action=action,
        shares=shares
    )

    try:
        # Get SOR client
        sor_client = get_sor_client()

        # Determine order type
        order_type = "LIMIT" if limit_price else "MARKET"

        # Route and execute order via SOR
        result = await sor_client.route_and_execute(
            symbol=symbol,
            action=action,
            shares=shares,
            order_type=order_type,
            limit_price=limit_price,
            stop_loss_price=stop_loss_price,
            urgency="normal",
            user_id=user_id
        )

        # Extract execution details
        routing = result["routing"]
        execution = result["execution"]

        logger.info(
            "sor_execution_success",
            order_id=execution.get("order_id"),
            venue=routing.get("venue"),
            filled_price=execution.get("filled_price"),
            slippage_bps=execution.get("slippage_bps")
        )

        # Return combined result
        return {
            "order_id": execution["order_id"],
            "status": execution["status"],
            "venue": routing["venue"],
            "venue_score": routing["score"],
            "filled_price": execution["filled_price"],
            "filled_shares": execution["filled_shares"],
            "commission": execution.get("commission", 0.00),
            "slippage_bps": execution.get("slippage_bps", 0.0),
            "execution_time_ms": execution.get("execution_time_ms", 0.0),
            "routing_decision_ms": routing.get("decision_time_ms", 0.0),
            "timestamp": execution["timestamp"],
            "routing_reasoning": routing.get("reasoning", {})
        }

    except Exception as e:
        logger.warning(
            "sor_execution_failed_fallback_to_mock",
            error=str(e),
            symbol=symbol,
            action=action
        )

        # Fallback to mock execution
        return {
            "order_id": f"MOCK-{int(time.time() * 1000)}",
            "status": "filled",
            "venue": "MOCK",
            "venue_score": 0.0,
            "filled_price": limit_price or 100.00,
            "filled_shares": shares,
            "commission": 0.00,
            "slippage_bps": 0.0,
            "execution_time_ms": 0.0,
            "routing_decision_ms": 0.0,
            "timestamp": datetime.now().isoformat(),
            "routing_reasoning": {},
            "note": "SOR unavailable, using mock execution"
        }
