"""
Smart Order Router (SOR) Client
Routes orders to optimal venues for execution
"""
from typing import Dict, Any, Optional
import structlog
import httpx

from ..config import settings

logger = structlog.get_logger(__name__)


class SORClient:
    """
    Client for Smart Order Router (SOR) service

    Routes orders to optimal venues based on:
    - Spread and liquidity
    - Latency
    - Fees
    - Fill probability
    """

    def __init__(self, timeout_ms: int = 500):
        # SOR is part of strategy-service
        self.base_url = "http://localhost:8005"  # Strategy service URL
        self.timeout_ms = timeout_ms

    async def route_order(
        self,
        symbol: str,
        action: str,
        shares: int,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        urgency: str = "normal",
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Route order to optimal venue

        Args:
            symbol: Ticker symbol
            action: "BUY" or "SELL"
            shares: Number of shares
            order_type: "MARKET", "LIMIT", "IOC", "POST_ONLY"
            limit_price: Optional limit price
            urgency: "low", "normal", "high"
            user_id: Optional user ID for tracking

        Returns:
            {
                "venue": "NASDAQ",
                "venue_id": "XNAS",
                "score": 0.92,
                "decision_time_ms": 3.5,
                "reasoning": {
                    "spread_score": 0.90,
                    "latency_score": 0.95,
                    "fee_score": 0.88,
                    "fill_probability": 0.97
                },
                "order": {
                    "symbol": "AAPL",
                    "side": "BUY",
                    "quantity": 100,
                    "order_type": "MARKET"
                }
            }
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/execution/route-order",
                    json={
                        "symbol": symbol,
                        "side": action,
                        "quantity": shares,
                        "order_type": order_type,
                        "limit_price": limit_price,
                        "urgency": urgency,
                        "user_id": user_id
                    },
                    timeout=self.timeout_ms / 1000.0
                )
                response.raise_for_status()

                result = response.json()

                logger.info(
                    "sor_routing_success",
                    symbol=symbol,
                    action=action,
                    venue=result.get("venue"),
                    score=result.get("score"),
                    decision_time_ms=result.get("decision_time_ms")
                )

                return result

        except httpx.HTTPError as e:
            logger.error(
                "sor_routing_failed",
                symbol=symbol,
                action=action,
                error=str(e)
            )
            raise

    async def submit_order(
        self,
        symbol: str,
        action: str,
        shares: int,
        venue: str,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit order to specific venue after routing

        Args:
            symbol: Ticker symbol
            action: "BUY" or "SELL"
            shares: Number of shares
            venue: Venue name (from routing decision)
            order_type: Order type
            limit_price: Optional limit price
            stop_loss_price: Optional stop loss price
            user_id: User ID

        Returns:
            {
                "order_id": "ORD-123456",
                "status": "filled",
                "venue": "NASDAQ",
                "filled_price": 175.50,
                "filled_shares": 100,
                "commission": 1.00,
                "slippage_bps": 2.5,
                "execution_time_ms": 45.2,
                "timestamp": "2024-01-20T15:30:00Z"
            }
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/execution/submit-order",
                    json={
                        "symbol": symbol,
                        "side": action,
                        "quantity": shares,
                        "venue": venue,
                        "order_type": order_type,
                        "limit_price": limit_price,
                        "stop_loss_price": stop_loss_price,
                        "user_id": user_id
                    },
                    timeout=5.0  # Longer timeout for actual execution
                )
                response.raise_for_status()

                result = response.json()

                logger.info(
                    "order_submission_success",
                    order_id=result.get("order_id"),
                    symbol=symbol,
                    action=action,
                    status=result.get("status"),
                    filled_price=result.get("filled_price")
                )

                return result

        except httpx.HTTPError as e:
            logger.error(
                "order_submission_failed",
                symbol=symbol,
                action=action,
                venue=venue,
                error=str(e)
            )
            raise

    async def route_and_execute(
        self,
        symbol: str,
        action: str,
        shares: int,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        urgency: str = "normal",
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Route order and execute in one call (convenience method)

        Args:
            symbol: Ticker symbol
            action: "BUY" or "SELL"
            shares: Number of shares
            order_type: Order type
            limit_price: Optional limit price
            stop_loss_price: Optional stop loss price
            urgency: Urgency level
            user_id: User ID

        Returns:
            Combined routing + execution result:
            {
                "routing": {...},
                "execution": {...}
            }
        """
        # Step 1: Route order to optimal venue
        routing_result = await self.route_order(
            symbol=symbol,
            action=action,
            shares=shares,
            order_type=order_type,
            limit_price=limit_price,
            urgency=urgency,
            user_id=user_id
        )

        # Step 2: Execute order on selected venue
        execution_result = await self.submit_order(
            symbol=symbol,
            action=action,
            shares=shares,
            venue=routing_result["venue"],
            order_type=order_type,
            limit_price=limit_price,
            stop_loss_price=stop_loss_price,
            user_id=user_id
        )

        return {
            "routing": routing_result,
            "execution": execution_result
        }

    async def get_order_status(
        self,
        order_id: str
    ) -> Dict[str, Any]:
        """
        Get order status

        Args:
            order_id: Order ID

        Returns:
            {
                "order_id": "ORD-123456",
                "status": "filled",
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 100,
                "filled_quantity": 100,
                "avg_fill_price": 175.52,
                "venue": "NASDAQ",
                "created_at": "2024-01-20T15:30:00Z",
                "updated_at": "2024-01-20T15:30:01Z"
            }
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/execution/orders/{order_id}",
                    timeout=2.0
                )
                response.raise_for_status()

                return response.json()

        except httpx.HTTPError as e:
            logger.error(
                "order_status_failed",
                order_id=order_id,
                error=str(e)
            )
            raise


# Singleton instance
_sor_client: Optional[SORClient] = None


def get_sor_client() -> SORClient:
    """Get or create SOR client singleton"""
    global _sor_client
    if _sor_client is None:
        _sor_client = SORClient()
    return _sor_client
