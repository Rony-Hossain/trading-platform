"""
Sentiment Service Client
Fetches market sentiment and social signals
"""
from typing import Dict, Any, List
import structlog

from .base_client import UpstreamClient

logger = structlog.get_logger(__name__)


class SentimentClient(UpstreamClient):
    """
    Client for Sentiment Service

    Endpoints:
    - POST /sentiment: Get sentiment analysis for symbols
    """

    def __init__(self, base_url: str, timeout_ms: int = 5000):
        super().__init__(
            service_name="sentiment",
            base_url=base_url,
            timeout_ms=timeout_ms,
            circuit_breaker_config={
                "failure_threshold": 5,
                "recovery_timeout": 60,
                "success_threshold": 2
            }
        )

    async def get_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get sentiment analysis for symbols

        Args:
            symbols: List of ticker symbols

        Returns:
            {
                "sentiments": [
                    {
                        "symbol": "AAPL",
                        "overall_score": 0.65,  # -1 to 1
                        "classification": "positive",  # negative/neutral/positive
                        "confidence": 0.82,
                        "sources": {
                            "news": {"score": 0.70, "article_count": 45},
                            "social": {"score": 0.60, "mention_count": 1200},
                            "analyst": {"score": 0.75, "rating_count": 12}
                        },
                        "trending": true,
                        "momentum": "increasing"
                    }
                ],
                "market_sentiment": {
                    "overall": "neutral",
                    "fear_greed_index": 52
                },
                "timestamp": "2024-01-20T15:30:00Z"
            }
        """
        try:
            response = await self.post(
                "/sentiment",
                json_data={"symbols": symbols}
            )

            logger.info(
                "sentiment_get_success",
                symbols=symbols,
                sentiment_count=len(response.get("sentiments", []))
            )

            return response

        except Exception as e:
            logger.error(
                "sentiment_get_failed",
                symbols=symbols,
                error=str(e)
            )
            raise

    async def get_market_sentiment(self) -> Dict[str, Any]:
        """
        Get overall market sentiment

        Returns:
            {
                "overall": "neutral",
                "fear_greed_index": 52,
                "vix": 18.5,
                "put_call_ratio": 0.95,
                "trending_sectors": [
                    {"sector": "TECH", "sentiment": "positive"},
                    {"sector": "ENERGY", "sentiment": "neutral"}
                ],
                "timestamp": "2024-01-20T15:30:00Z"
            }
        """
        try:
            response = await self.get("/market-sentiment")
            logger.info("sentiment_market_success", overall=response.get("overall"))
            return response

        except Exception as e:
            logger.error("sentiment_market_failed", error=str(e))
            raise
