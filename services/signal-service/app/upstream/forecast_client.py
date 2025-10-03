"""
Forecast Service Client
Fetches price forecasts and technical indicators
"""
from typing import Dict, Any, List
import structlog

from .base_client import UpstreamClient

logger = structlog.get_logger(__name__)


class ForecastClient(UpstreamClient):
    """
    Client for Forecast Service

    Endpoints:
    - POST /forecast: Get price forecasts and technical analysis
    """

    def __init__(self, base_url: str, timeout_ms: int = 5000):
        super().__init__(
            service_name="forecast",
            base_url=base_url,
            timeout_ms=timeout_ms,
            circuit_breaker_config={
                "failure_threshold": 5,
                "recovery_timeout": 60,
                "success_threshold": 2
            }
        )

    async def get_forecast(
        self,
        symbols: List[str],
        horizons: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get price forecasts for symbols

        Args:
            symbols: List of ticker symbols
            horizons: Forecast horizons (e.g., ["1d", "5d", "1m"])

        Returns:
            {
                "forecasts": [
                    {
                        "symbol": "AAPL",
                        "current_price": 175.50,
                        "predictions": {
                            "1d": {"price": 176.20, "confidence": 0.75},
                            "5d": {"price": 178.00, "confidence": 0.60}
                        },
                        "technicals": {
                            "rsi": 62.5,
                            "macd": {"value": 1.2, "signal": 0.8},
                            "support": 172.00,
                            "resistance": 180.00,
                            "trend": "bullish"
                        }
                    }
                ],
                "timestamp": "2024-01-20T15:30:00Z"
            }
        """
        horizons = horizons or ["1d", "5d"]

        try:
            response = await self.post(
                "/forecast",
                json_data={
                    "symbols": symbols,
                    "horizons": horizons
                }
            )

            logger.info(
                "forecast_get_success",
                symbols=symbols,
                horizons=horizons,
                forecast_count=len(response.get("forecasts", []))
            )

            return response

        except Exception as e:
            logger.error(
                "forecast_get_failed",
                symbols=symbols,
                error=str(e)
            )
            raise

    async def get_technicals(self, symbol: str) -> Dict[str, Any]:
        """
        Get technical indicators for a symbol

        Args:
            symbol: Ticker symbol

        Returns:
            {
                "symbol": "AAPL",
                "indicators": {
                    "rsi": 62.5,
                    "macd": {"value": 1.2, "signal": 0.8, "histogram": 0.4},
                    "bollinger_bands": {"upper": 180, "middle": 175, "lower": 170},
                    "moving_averages": {
                        "sma_20": 174.5,
                        "sma_50": 172.0,
                        "ema_12": 175.2
                    }
                },
                "support_resistance": {
                    "support_levels": [172.0, 170.5],
                    "resistance_levels": [180.0, 182.5]
                },
                "trend": "bullish"
            }
        """
        try:
            response = await self.get(f"/technicals/{symbol}")
            logger.info("forecast_technicals_success", symbol=symbol)
            return response

        except Exception as e:
            logger.error("forecast_technicals_failed", symbol=symbol, error=str(e))
            raise
