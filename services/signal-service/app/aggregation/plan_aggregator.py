"""
Plan Aggregator
Combines signals from upstream services into actionable trading plan
"""
from typing import Dict, Any, List, Optional
import asyncio
import structlog

from ..core.contracts import PlanResponse, Pick, ReasonCode
from ..core.observability import generate_request_id, set_request_id, log_plan_decision, log_degradation
from ..core.policy_manager import get_policy_manager
from ..core.decision_store import get_decision_store
from ..core.swr_cache import get_swr_cache_manager
from ..upstream.inference_client import InferenceClient
from ..upstream.forecast_client import ForecastClient
from ..upstream.sentiment_client import SentimentClient
from ..upstream.portfolio_client import PortfolioClient
from ..translation.beginner_translator import BeginnerTranslator
from ..core.guardrails import get_guardrail_engine
from ..core.fitness_checker import get_fitness_checker

logger = structlog.get_logger(__name__)


class PlanAggregator:
    """
    Aggregates upstream signals into trading plan

    Flow:
    1. Fetch data from upstream services (parallel, with caching)
    2. Translate predictions into plain English
    3. Apply guardrails and fitness checks
    4. Generate picks with metadata
    5. Save decision snapshot
    6. Return plan response
    """

    def __init__(
        self,
        inference_client: InferenceClient,
        forecast_client: ForecastClient,
        sentiment_client: SentimentClient,
        portfolio_client: PortfolioClient
    ):
        self.inference_client = inference_client
        self.forecast_client = forecast_client
        self.sentiment_client = sentiment_client
        self.portfolio_client = portfolio_client

        self.translator = BeginnerTranslator()
        self.policy_manager = get_policy_manager()
        self.guardrail_engine = get_guardrail_engine()
        self.fitness_checker = get_fitness_checker()
        self.decision_store = get_decision_store()
        self.cache_manager = get_swr_cache_manager()

        logger.info("plan_aggregator_initialized")

    async def generate_plan(
        self,
        user_id: str,
        watchlist: Optional[List[str]] = None,
        mode: str = "beginner"
    ) -> PlanResponse:
        """
        Generate trading plan for user

        Args:
            user_id: User ID
            watchlist: Optional list of symbols to analyze (defaults to user's watchlist)
            mode: "beginner" or "expert"

        Returns:
            PlanResponse with picks and metadata
        """
        request_id = generate_request_id()
        set_request_id(request_id)

        logger.info(
            "plan_generation_started",
            request_id=request_id,
            user_id=user_id,
            mode=mode,
            watchlist=watchlist
        )

        try:
            # Fetch all upstream data in parallel
            (
                portfolio_data,
                inference_data,
                forecast_data,
                sentiment_data,
                degraded_fields
            ) = await self._fetch_upstream_data(user_id, watchlist)

            # Determine symbols to analyze
            symbols = watchlist or self._get_default_watchlist(portfolio_data, inference_data)

            # Generate picks from aggregated data
            picks = await self._generate_picks(
                symbols,
                user_id,
                portfolio_data,
                inference_data,
                forecast_data,
                sentiment_data,
                mode
            )

            # Build response
            response = PlanResponse(
                request_id=request_id,
                picks=picks,
                daily_cap_reached=self._check_daily_cap(portfolio_data),
                degraded_fields=degraded_fields,
                metadata={
                    "model_version": inference_data.get("model_metadata", {}).get("version"),
                    "generated_at": self._get_timestamp(),
                    "mode": mode,
                    "symbols_analyzed": len(symbols)
                }
            )

            # Save decision snapshot
            await self._save_decision_snapshot(
                request_id,
                user_id,
                {
                    "watchlist": watchlist,
                    "mode": mode
                },
                response,
                {
                    "inference": inference_data,
                    "forecast": forecast_data,
                    "sentiment": sentiment_data,
                    "portfolio": portfolio_data
                }
            )

            log_plan_decision(
                user_id=user_id,
                request_id=request_id,
                pick_count=len(picks),
                degraded=len(degraded_fields) > 0
            )

            logger.info(
                "plan_generation_completed",
                request_id=request_id,
                user_id=user_id,
                pick_count=len(picks),
                degraded_fields=degraded_fields
            )

            return response

        except Exception as e:
            logger.error(
                "plan_generation_failed",
                request_id=request_id,
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise

    async def _fetch_upstream_data(
        self,
        user_id: str,
        watchlist: Optional[List[str]]
    ) -> tuple:
        """Fetch data from all upstream services in parallel"""
        degraded_fields = []

        # Define fetch tasks
        async def fetch_portfolio():
            try:
                return await self.portfolio_client.get_portfolio(user_id)
            except Exception as e:
                logger.error("portfolio_fetch_failed", error=str(e))
                degraded_fields.append("portfolio")
                return {}

        async def fetch_inference():
            try:
                symbols = watchlist or []
                if not symbols:
                    return {}
                return await self.inference_client.predict(symbols, user_id)
            except Exception as e:
                logger.error("inference_fetch_failed", error=str(e))
                degraded_fields.append("inference")
                return {}

        async def fetch_forecast():
            try:
                symbols = watchlist or []
                if not symbols:
                    return {}
                return await self.forecast_client.get_forecast(symbols)
            except Exception as e:
                logger.error("forecast_fetch_failed", error=str(e))
                degraded_fields.append("forecast")
                return {}

        async def fetch_sentiment():
            try:
                symbols = watchlist or []
                if not symbols:
                    return {}
                return await self.sentiment_client.get_sentiment(symbols)
            except Exception as e:
                logger.error("sentiment_fetch_failed", error=str(e))
                degraded_fields.append("sentiment")
                return {}

        # Execute in parallel
        results = await asyncio.gather(
            fetch_portfolio(),
            fetch_inference(),
            fetch_forecast(),
            fetch_sentiment(),
            return_exceptions=False
        )

        portfolio_data, inference_data, forecast_data, sentiment_data = results

        if degraded_fields:
            log_degradation(degraded_fields)

        return portfolio_data, inference_data, forecast_data, sentiment_data, degraded_fields

    async def _generate_picks(
        self,
        symbols: List[str],
        user_id: str,
        portfolio_data: Dict[str, Any],
        inference_data: Dict[str, Any],
        forecast_data: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        mode: str
    ) -> List[Pick]:
        """Generate picks from aggregated data"""
        picks = []
        is_beginner = mode == "beginner"

        # Get predictions for each symbol
        predictions = {
            p["symbol"]: p
            for p in inference_data.get("predictions", [])
        }

        forecasts = {
            f["symbol"]: f
            for f in forecast_data.get("forecasts", [])
        }

        sentiments = {
            s["symbol"]: s
            for s in sentiment_data.get("sentiments", [])
        }

        for symbol in symbols:
            try:
                pick = await self._generate_pick(
                    symbol,
                    predictions.get(symbol, {}),
                    forecasts.get(symbol, {}),
                    sentiments.get(symbol, {}),
                    portfolio_data,
                    is_beginner
                )

                if pick:
                    picks.append(pick)

            except Exception as e:
                logger.error(
                    "pick_generation_failed",
                    symbol=symbol,
                    error=str(e)
                )
                continue

        # Sort picks by confidence (high to low)
        picks.sort(key=lambda p: {"high": 3, "medium": 2, "low": 1}.get(p.confidence, 0), reverse=True)

        # Limit pick count for beginners
        if is_beginner:
            max_picks = self.policy_manager.get("beginner_mode.max_picks_per_day", 3)
            picks = picks[:max_picks]

        return picks

    async def _generate_pick(
        self,
        symbol: str,
        prediction: Dict[str, Any],
        forecast: Dict[str, Any],
        sentiment: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        is_beginner: bool
    ) -> Optional[Pick]:
        """Generate single pick from symbol data"""
        if not prediction:
            return None

        action = prediction.get("action", "HOLD")
        if action == "HOLD":
            return None

        # Translate to plain English
        reason_text, reason_codes, confidence_value = self.translator.translate_action(
            action=action,
            symbol=symbol,
            prediction_data=prediction,
            technical_data=forecast.get("technicals", {}),
            sentiment_data=sentiment,
            portfolio_context={}
        )

        confidence_label = self.translator.translate_confidence(confidence_value)

        # Calculate shares (simplified)
        current_price = forecast.get("current_price", 100)
        shares = self._calculate_shares(
            symbol,
            action,
            current_price,
            portfolio_data,
            is_beginner
        )

        # Create pick
        pick = Pick(
            symbol=symbol,
            action=action,
            shares=shares,
            confidence=confidence_label,
            reason=reason_text,
            reason_codes=reason_codes,
            decision_path=f"inference:{prediction.get('model_version', 'unknown')}",
            limit_price=current_price,
            stop_loss_price=self._calculate_stop_loss(current_price, action, is_beginner),
            metadata={
                "sector": forecast.get("sector"),
                "volatility": forecast.get("volatility"),
                "expected_return": prediction.get("expected_return")
            }
        )

        # Apply guardrails
        is_allowed, violations = self.guardrail_engine.check_pick(
            pick,
            user_context={
                "beginner_mode": is_beginner,
                "total_value": portfolio_data.get("total_value", 0),
                "daily_trades": portfolio_data.get("daily_trades", 0),
                "daily_pnl_pct": portfolio_data.get("daily_pnl_pct", 0),
                "sector_allocations": portfolio_data.get("sector_allocations", {}),
                "positions": portfolio_data.get("positions", [])
            },
            market_context={
                "symbols": {
                    symbol: {
                        "volatility": forecast.get("volatility", 0),
                        "avg_daily_volume": forecast.get("avg_daily_volume", 0),
                        "sector": forecast.get("sector", "UNKNOWN")
                    }
                },
                "date": self._get_timestamp()
            }
        )

        if not is_allowed:
            logger.debug("pick_blocked_by_guardrails", symbol=symbol, action=action)
            return None

        # Check fitness for beginners
        if is_beginner:
            is_fit, quality_score, issues = self.fitness_checker.check_fitness(pick, is_beginner)
            if not is_fit:
                logger.debug("pick_failed_fitness", symbol=symbol, issues=issues)
                return None

        return pick

    def _calculate_shares(
        self,
        symbol: str,
        action: str,
        price: float,
        portfolio_data: Dict[str, Any],
        is_beginner: bool
    ) -> int:
        """Calculate number of shares for pick"""
        if action == "SELL":
            # Sell all shares
            positions = portfolio_data.get("positions", [])
            owned = next((p["shares"] for p in positions if p["symbol"] == symbol), 0)
            return owned

        # BUY: Calculate based on position sizing
        portfolio_value = portfolio_data.get("total_value", 10000)
        max_position_pct = 0.10 if is_beginner else 0.20  # 10% or 20%

        position_value = portfolio_value * max_position_pct
        shares = int(position_value / price)

        return max(shares, 1)  # At least 1 share

    def _calculate_stop_loss(
        self,
        price: float,
        action: str,
        is_beginner: bool
    ) -> Optional[float]:
        """Calculate stop loss price"""
        if action != "BUY" or not is_beginner:
            return None

        # 3% stop loss for beginners
        stop_distance_pct = self.policy_manager.get("beginner_mode.default_stop_loss_pct", 3.0)
        stop_loss = price * (1 - stop_distance_pct / 100)

        return round(stop_loss, 2)

    def _get_default_watchlist(
        self,
        portfolio_data: Dict[str, Any],
        inference_data: Dict[str, Any]
    ) -> List[str]:
        """Get default watchlist if none provided"""
        # Use current positions + top predictions
        positions = portfolio_data.get("positions", [])
        symbols = [p["symbol"] for p in positions[:5]]

        # Add top predictions
        predictions = inference_data.get("predictions", [])
        for pred in predictions[:5]:
            if pred["symbol"] not in symbols:
                symbols.append(pred["symbol"])

        return symbols[:10]  # Max 10 symbols

    def _check_daily_cap(self, portfolio_data: Dict[str, Any]) -> bool:
        """Check if daily trade cap is reached"""
        daily_trades = portfolio_data.get("daily_trades", 0)
        max_trades = self.policy_manager.get("beginner_mode.max_daily_trades", 3)
        return daily_trades >= max_trades

    async def _save_decision_snapshot(
        self,
        request_id: str,
        user_id: str,
        inputs: Dict[str, Any],
        response: PlanResponse,
        upstream_data: Dict[str, Any]
    ):
        """Save decision snapshot for audit trail"""
        try:
            self.decision_store.save_snapshot(
                request_id=request_id,
                user_id=user_id,
                inputs=inputs,
                picks=[p.dict() for p in response.picks],
                metadata={
                    "degraded_fields": response.degraded_fields,
                    "model_versions": {
                        "inference": upstream_data.get("inference", {}).get("model_metadata", {}).get("version")
                    },
                    "upstream_latencies": {}  # TODO: Track latencies
                }
            )
        except Exception as e:
            logger.error("decision_snapshot_save_failed", error=str(e))

    def _get_timestamp(self) -> str:
        """Get current ISO timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"
