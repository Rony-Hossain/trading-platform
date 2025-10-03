"""
Portfolio Service Client
Fetches user portfolio data and position information

NOTE: This client can optionally integrate with Trade Journal service
for real position data. Set use_trade_journal=True to enable.
"""
from typing import Dict, Any, List, Optional
import structlog
import httpx

from .base_client import UpstreamClient
from ..config import settings

logger = structlog.get_logger(__name__)


class PortfolioClient(UpstreamClient):
    """
    Client for Portfolio Service

    Endpoints:
    - GET /portfolio/{user_id}: Get user portfolio
    - GET /positions/{user_id}: Get current positions
    - POST /validate-order: Validate order against portfolio constraints

    Optional Trade Journal Integration:
    - When use_trade_journal=True, fetches positions from Trade Journal service
    - Falls back to Portfolio Service on error
    """

    def __init__(
        self,
        base_url: str,
        timeout_ms: int = 5000,
        use_trade_journal: bool = False
    ):
        super().__init__(
            service_name="portfolio",
            base_url=base_url,
            timeout_ms=timeout_ms,
            circuit_breaker_config={
                "failure_threshold": 5,
                "recovery_timeout": 60,
                "success_threshold": 2
            }
        )
        self.use_trade_journal = use_trade_journal
        self.trade_journal_url = settings.TRADE_JOURNAL_URL

    async def get_portfolio(self, user_id: str) -> Dict[str, Any]:
        """
        Get user portfolio summary

        Args:
            user_id: User ID

        Returns:
            {
                "user_id": "user123",
                "total_value": 50000.00,
                "cash": 10000.00,
                "buying_power": 15000.00,
                "positions_value": 40000.00,
                "daily_pnl": 523.50,
                "daily_pnl_pct": 1.05,
                "risk_level": "moderate",
                "diversification_score": 0.75
            }
        """
        try:
            response = await self.get(f"/portfolio/{user_id}")
            logger.info(
                "portfolio_get_success",
                user_id=user_id,
                total_value=response.get("total_value")
            )
            return response

        except Exception as e:
            logger.error("portfolio_get_failed", user_id=user_id, error=str(e))
            raise

    async def get_positions(self, user_id: str) -> Dict[str, Any]:
        """
        Get user's current positions

        If use_trade_journal=True, fetches from Trade Journal service
        and transforms into Signal Service format.

        Args:
            user_id: User ID

        Returns:
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
                        "cost_basis": 17050.00,
                        "weight": 0.35  # 35% of portfolio
                    }
                ],
                "total_positions": 5,
                "concentration_risk": "low"
            }
        """
        # Try Trade Journal first if enabled
        if self.use_trade_journal:
            try:
                positions = await self._get_positions_from_trade_journal(user_id)
                logger.info(
                    "positions_from_trade_journal",
                    user_id=user_id,
                    position_count=len(positions.get("positions", []))
                )
                return positions
            except Exception as e:
                logger.warning(
                    "trade_journal_fallback",
                    user_id=user_id,
                    error=str(e),
                    msg="Falling back to Portfolio Service"
                )
                # Fall through to Portfolio Service

        # Use Portfolio Service (original implementation or fallback)
        try:
            response = await self.get(f"/positions/{user_id}")
            logger.info(
                "portfolio_positions_success",
                user_id=user_id,
                position_count=len(response.get("positions", []))
            )
            return response

        except Exception as e:
            logger.error("portfolio_positions_failed", user_id=user_id, error=str(e))
            raise

    async def validate_order(
        self,
        user_id: str,
        symbol: str,
        action: str,
        shares: int,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Validate order against portfolio constraints

        Args:
            user_id: User ID
            symbol: Ticker symbol
            action: "BUY" or "SELL"
            shares: Number of shares
            price: Optional limit price

        Returns:
            {
                "valid": true,
                "warnings": [
                    "This will increase concentration in TECH sector to 45%"
                ],
                "constraints": {
                    "buying_power_sufficient": true,
                    "position_limit_ok": true,
                    "concentration_ok": false,
                    "risk_level_ok": true
                },
                "estimated_impact": {
                    "new_cash": 8500.00,
                    "new_position_value": 42000.00,
                    "new_concentration": {"TECH": 0.45, "FINANCE": 0.30}
                }
            }
        """
        try:
            response = await self.post(
                "/validate-order",
                json_data={
                    "user_id": user_id,
                    "symbol": symbol,
                    "action": action,
                    "shares": shares,
                    "price": price
                }
            )

            logger.info(
                "portfolio_validate_order_success",
                user_id=user_id,
                symbol=symbol,
                action=action,
                valid=response.get("valid")
            )

            return response

        except Exception as e:
            logger.error(
                "portfolio_validate_order_failed",
                user_id=user_id,
                symbol=symbol,
                error=str(e)
            )
            raise

    async def get_risk_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get user's risk profile and constraints

        Args:
            user_id: User ID

        Returns:
            {
                "user_id": "user123",
                "risk_tolerance": "moderate",
                "constraints": {
                    "max_position_size_pct": 20,
                    "max_sector_concentration_pct": 40,
                    "max_single_trade_pct": 10,
                    "stop_loss_required": true,
                    "max_daily_loss_pct": 5
                },
                "trading_level": "beginner",
                "beginner_mode_enabled": true
            }
        """
        try:
            response = await self.get(f"/risk-profile/{user_id}")
            logger.info(
                "portfolio_risk_profile_success",
                user_id=user_id,
                risk_tolerance=response.get("risk_tolerance")
            )
            return response

        except Exception as e:
            logger.error("portfolio_risk_profile_failed", user_id=user_id, error=str(e))
            raise

    async def _get_positions_from_trade_journal(self, user_id: str) -> Dict[str, Any]:
        """
        Fetch positions from Trade Journal service and transform to Signal Service format

        Trade Journal endpoint: GET /positions?symbol={symbol}&user_id={user_id}
        Returns: List of position records

        Args:
            user_id: User ID

        Returns:
            Transformed positions dict matching Signal Service format
        """
        try:
            async with httpx.AsyncClient() as client:
                # Fetch all positions for user
                # Note: Trade Journal doesn't have user_id filter yet,
                # so we fetch all and filter (TODO: add user_id to Trade Journal)
                response = await client.get(
                    f"{self.trade_journal_url}/positions",
                    timeout=5.0
                )
                response.raise_for_status()

                trade_journal_positions = response.json()

                # Transform Trade Journal format to Signal Service format
                positions = []
                total_value = 0

                for pos in trade_journal_positions:
                    symbol = pos.get("symbol")
                    quantity = pos.get("quantity", 0)
                    avg_cost = pos.get("avg_cost", 0)
                    current_price = pos.get("current_price", avg_cost)  # Use avg_cost if no current price

                    # Calculate derived values
                    cost_basis = quantity * avg_cost
                    market_value = quantity * current_price
                    unrealized_pnl = market_value - cost_basis
                    unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0

                    positions.append({
                        "symbol": symbol,
                        "shares": quantity,
                        "avg_cost": avg_cost,
                        "current_price": current_price,
                        "market_value": market_value,
                        "unrealized_pnl": unrealized_pnl,
                        "unrealized_pnl_pct": unrealized_pnl_pct,
                        "cost_basis": cost_basis,
                        "weight": 0.0  # Will calculate after totals
                    })

                    total_value += market_value

                # Calculate position weights
                for pos in positions:
                    pos["weight"] = (pos["market_value"] / total_value) if total_value > 0 else 0

                # Calculate concentration risk
                max_weight = max([p["weight"] for p in positions]) if positions else 0
                if max_weight > 0.40:  # 40%+
                    concentration_risk = "high"
                elif max_weight > 0.25:  # 25-40%
                    concentration_risk = "medium"
                else:
                    concentration_risk = "low"

                return {
                    "positions": positions,
                    "total_positions": len(positions),
                    "concentration_risk": concentration_risk
                }

        except httpx.HTTPError as e:
            logger.error(
                "trade_journal_http_error",
                user_id=user_id,
                error=str(e)
            )
            raise
        except Exception as e:
            logger.error(
                "trade_journal_transform_error",
                user_id=user_id,
                error=str(e)
            )
            raise
