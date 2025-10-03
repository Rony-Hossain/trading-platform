"""
Halt Detection Client
Checks if symbols are halted before allowing trades
"""
from typing import Dict, Any, Optional
import structlog
import httpx

logger = structlog.get_logger(__name__)


class HaltClient:
    """
    Client for halt detection service

    Checks:
    - LULD (Limit Up Limit Down) halts
    - Volatility halts
    - News pending halts
    - Circuit breaker status
    - Auction periods
    """

    def __init__(self, base_url: str = "http://localhost:8005", timeout_ms: int = 200):
        self.base_url = base_url  # Strategy service URL
        self.timeout_ms = timeout_ms

    async def check_halt_status(self, symbol: str) -> Dict[str, Any]:
        """
        Check if symbol is currently halted

        Args:
            symbol: Ticker symbol

        Returns:
            {
                "symbol": "AAPL",
                "is_halted": false,
                "halt_type": null,
                "halt_start_time": null,
                "expected_resume_time": null,
                "restrictions": [],
                "message": null
            }

        If service unavailable, returns:
            {
                "symbol": symbol,
                "is_halted": false,
                "service_unavailable": true
            }
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/venue-rules/halt-status/{symbol}",
                    timeout=self.timeout_ms / 1000.0
                )
                response.raise_for_status()

                result = response.json()

                if result.get("is_halted"):
                    logger.warning(
                        "symbol_halted",
                        symbol=symbol,
                        halt_type=result.get("halt_type"),
                        message=result.get("message")
                    )
                else:
                    logger.debug(
                        "symbol_not_halted",
                        symbol=symbol
                    )

                return result

        except httpx.HTTPError as e:
            logger.error(
                "halt_check_failed",
                symbol=symbol,
                error=str(e)
            )

            # Return safe default (assume not halted, but mark service unavailable)
            return {
                "symbol": symbol,
                "is_halted": False,
                "service_unavailable": True,
                "error": str(e)
            }

    async def check_circuit_breaker(self) -> Dict[str, Any]:
        """
        Check market-wide circuit breaker status

        Returns:
            {
                "circuit_breaker_active": false,
                "circuit_breaker_level": null,
                "restrictions": []
            }
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/venue-rules/circuit-breaker-status",
                    timeout=self.timeout_ms / 1000.0
                )
                response.raise_for_status()

                return response.json()

        except httpx.HTTPError as e:
            logger.error(
                "circuit_breaker_check_failed",
                error=str(e)
            )

            # Return safe default
            return {
                "circuit_breaker_active": False,
                "service_unavailable": True,
                "error": str(e)
            }

    async def check_auction_period(self) -> Dict[str, Any]:
        """
        Check if currently in auction period

        Returns:
            {
                "in_auction": false,
                "auction_type": null,
                "start_time": null,
                "expected_end_time": null
            }
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/venue-rules/auction-status",
                    timeout=self.timeout_ms / 1000.0
                )
                response.raise_for_status()

                return response.json()

        except httpx.HTTPError as e:
            logger.error(
                "auction_check_failed",
                error=str(e)
            )

            # Return safe default
            return {
                "in_auction": False,
                "service_unavailable": True,
                "error": str(e)
            }


# Singleton instance
_halt_client: Optional[HaltClient] = None


def get_halt_client() -> HaltClient:
    """Get or create halt client singleton"""
    global _halt_client
    if _halt_client is None:
        _halt_client = HaltClient()
    return _halt_client
