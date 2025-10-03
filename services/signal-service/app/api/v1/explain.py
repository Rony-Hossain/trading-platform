"""
Explain API Endpoint
POST /api/v1/explain - Get detailed explanation of a recommendation
"""
from fastapi import APIRouter, Header, HTTPException, status, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import structlog

from ...core.contracts import ExplainResponse, ReasonCode
from ...core.fitness_checker import get_fitness_checker
from ...core.decision_store import get_decision_store
from ...translation.beginner_translator import BeginnerTranslator

logger = structlog.get_logger(__name__)

router = APIRouter()


class ExplainRequest(BaseModel):
    """Explain request"""
    request_id: str = Field(..., description="Request ID from /plan endpoint")
    symbol: str = Field(..., description="Symbol to explain")


def _build_glossary() -> List[Dict[str, str]]:
    """Build glossary of trading terms"""
    return [
        {
            "term": "Support",
            "definition": "A price level where a stock tends to find buying interest and stop falling"
        },
        {
            "term": "Resistance",
            "definition": "A price level where a stock tends to find selling pressure and stop rising"
        },
        {
            "term": "RSI (Relative Strength Index)",
            "definition": "A momentum indicator that measures if a stock is overbought (>70) or oversold (<30)"
        },
        {
            "term": "MACD",
            "definition": "A trend-following momentum indicator that shows the relationship between two moving averages"
        },
        {
            "term": "Volatility",
            "definition": "How much a stock's price fluctuates - higher volatility means bigger price swings"
        },
        {
            "term": "Liquidity",
            "definition": "How easily you can buy or sell a stock - measured by trading volume"
        },
        {
            "term": "Stop Loss",
            "definition": "A safety exit price that automatically sells your position to limit losses"
        },
        {
            "term": "Sentiment",
            "definition": "Overall market mood toward a stock based on news, social media, and analyst opinions"
        }
    ]


@router.post("/explain", response_model=ExplainResponse)
async def explain_recommendation(
    request: ExplainRequest,
    user_id: str = Header(..., alias="X-User-ID")
):
    """
    Get detailed explanation of a recommendation

    **Request:**
    - Header: `X-User-ID` - User identifier
    - Body: `request_id` - Request ID from /plan endpoint
    - Body: `symbol` - Symbol to explain

    **Response:**
    - `symbol` - Ticker symbol
    - `plain_english` - Beginner-friendly explanation
    - `glossary` - Definitions of technical terms
    - `decision_tree` - Step-by-step decision breakdown
    - `confidence_breakdown` - What contributed to confidence score
    - `risk_factors` - Identified risks

    **Example:**
    ```
    POST /api/v1/explain
    X-User-ID: user123

    {
      "request_id": "req_abc123",
      "symbol": "AAPL"
    }
    ```

    **Response:**
    ```json
    {
      "symbol": "AAPL",
      "plain_english": "We recommend buying AAPL because...",
      "glossary": [...],
      "decision_tree": [
        "1. AI model predicted 2.3% expected return (high confidence)",
        "2. Price holding above support level at $172",
        "3. Positive sentiment from recent earnings",
        "4. Passed beginner safety checks"
      ],
      "confidence_breakdown": {...},
      "risk_factors": [...]
    }
    ```
    """
    logger.info(
        "explain_request_received",
        user_id=user_id,
        request_id=request.request_id,
        symbol=request.symbol
    )

    try:
        # Retrieve decision snapshot
        decision_store = get_decision_store()
        snapshot = decision_store.get_snapshot(request.request_id)

        if not snapshot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "code": "REQUEST_NOT_FOUND",
                        "message": f"No plan found for request_id: {request.request_id}"
                    }
                }
            )

        # Find the pick for this symbol
        picks = snapshot.picks
        symbol_pick = next((p for p in picks if p.get("symbol") == request.symbol), None)

        if not symbol_pick:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "code": "SYMBOL_NOT_FOUND",
                        "message": f"Symbol {request.symbol} not found in plan"
                    }
                }
            )

        # Build explanation
        translator = BeginnerTranslator()
        fitness_checker = get_fitness_checker()

        # Plain English explanation
        plain_english = symbol_pick.get("reason", "")

        # Decision tree breakdown
        decision_tree = _build_decision_tree(symbol_pick, snapshot)

        # Confidence breakdown
        from ...core.contracts import Pick
        pick_obj = Pick(**symbol_pick)
        confidence_breakdown = fitness_checker.calculate_pick_score(pick_obj)

        # Risk factors
        risk_factors = _extract_risk_factors(symbol_pick)

        # Glossary
        glossary = _build_glossary()

        response = ExplainResponse(
            symbol=request.symbol,
            plain_english=plain_english,
            glossary=glossary,
            decision_tree=decision_tree,
            confidence_breakdown=confidence_breakdown,
            risk_factors=risk_factors
        )

        logger.info(
            "explain_request_completed",
            user_id=user_id,
            request_id=request.request_id,
            symbol=request.symbol
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "explain_request_failed",
            user_id=user_id,
            request_id=request.request_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "EXPLAIN_FAILED",
                    "message": "Failed to generate explanation. Please try again."
                }
            }
        )


def _build_decision_tree(pick: Dict[str, Any], snapshot: Any) -> List[str]:
    """Build step-by-step decision tree"""
    steps = []

    # Step 1: Model prediction
    metadata = pick.get("metadata", {})
    expected_return = metadata.get("expected_return")
    if expected_return:
        steps.append(
            f"1. AI model predicted {expected_return*100:.1f}% expected return ({pick.get('confidence')} confidence)"
        )

    # Step 2: Reason codes
    reason_codes = pick.get("reason_codes", [])
    for idx, code in enumerate(reason_codes[:3], start=2):  # Top 3 reasons
        translator = BeginnerTranslator()
        code_enum = ReasonCode(code) if isinstance(code, str) else code
        description = translator.REASON_TEMPLATES.get(code_enum, code)
        steps.append(f"{idx}. {description}")

    # Step 3: Guardrails
    steps.append(f"{len(steps)+1}. Passed beginner safety checks (position sizing, stop loss, volatility)")

    # Step 4: Final recommendation
    action = pick.get("action")
    shares = pick.get("shares")
    steps.append(f"{len(steps)+1}. Final recommendation: {action} {shares} shares")

    return steps


def _extract_risk_factors(pick: Dict[str, Any]) -> List[str]:
    """Extract risk factors from pick"""
    risk_factors = []

    metadata = pick.get("metadata", {})

    # High volatility
    volatility = metadata.get("volatility", 0)
    if volatility > 0.03:  # > 3%
        risk_factors.append(f"Stock has higher than average volatility ({volatility*100:.1f}%)")

    # Sector risk
    sector = metadata.get("sector")
    if sector in ["TECH", "CRYPTO"]:
        risk_factors.append(f"{sector} sector can be more volatile than broader market")

    # Low confidence
    confidence = pick.get("confidence")
    if confidence == "low":
        risk_factors.append("Model has low confidence in this prediction")

    # Stop loss
    stop_loss = pick.get("stop_loss_price")
    limit_price = pick.get("limit_price")
    if stop_loss and limit_price:
        stop_distance_pct = abs(stop_loss - limit_price) / limit_price * 100
        risk_factors.append(f"Maximum potential loss: {stop_distance_pct:.1f}% (stop loss at ${stop_loss:.2f})")

    if not risk_factors:
        risk_factors.append("No significant risk factors identified")

    return risk_factors
