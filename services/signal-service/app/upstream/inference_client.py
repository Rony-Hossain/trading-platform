"""
Inference Service Client
Fetches model predictions and confidence scores
"""
from typing import Dict, Any, List, Optional
import structlog

from .base_client import UpstreamClient

logger = structlog.get_logger(__name__)


class InferenceClient(UpstreamClient):
    """
    Client for Inference Service

    Endpoints:
    - POST /predict: Get model predictions for symbols
    """

    def __init__(self, base_url: str, timeout_ms: int = 5000):
        super().__init__(
            service_name="inference",
            base_url=base_url,
            timeout_ms=timeout_ms,
            circuit_breaker_config={
                "failure_threshold": 5,
                "recovery_timeout": 60,
                "success_threshold": 2
            }
        )

    async def predict(
        self,
        symbols: List[str],
        user_id: str,
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get model predictions for symbols

        Args:
            symbols: List of ticker symbols
            user_id: User ID for personalization
            model_version: Optional specific model version

        Returns:
            {
                "predictions": [
                    {
                        "symbol": "AAPL",
                        "action": "BUY",
                        "confidence": 0.85,
                        "expected_return": 0.023,
                        "risk_score": 0.42,
                        "model_version": "v1.2.3"
                    }
                ],
                "model_metadata": {
                    "version": "v1.2.3",
                    "trained_at": "2024-01-15T10:00:00Z"
                }
            }
        """
        try:
            response = await self.post(
                "/predict",
                json_data={
                    "symbols": symbols,
                    "user_id": user_id,
                    "model_version": model_version
                }
            )

            logger.info(
                "inference_predict_success",
                symbols=symbols,
                user_id=user_id,
                prediction_count=len(response.get("predictions", []))
            )

            return response

        except Exception as e:
            logger.error(
                "inference_predict_failed",
                symbols=symbols,
                user_id=user_id,
                error=str(e)
            )
            raise

    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get current model information

        Returns:
            {
                "version": "v1.2.3",
                "trained_at": "2024-01-15T10:00:00Z",
                "features": ["price", "volume", "sentiment"],
                "performance": {
                    "accuracy": 0.72,
                    "precision": 0.68,
                    "recall": 0.75
                }
            }
        """
        try:
            response = await self.get("/model/info")
            logger.info("inference_model_info_retrieved", version=response.get("version"))
            return response

        except Exception as e:
            logger.error("inference_model_info_failed", error=str(e))
            raise
