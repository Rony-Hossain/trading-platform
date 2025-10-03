"""
Integration tests for Plan Aggregator
Tests full flow with mocked upstream services
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.aggregation.plan_aggregator import PlanAggregator
from app.upstream.inference_client import InferenceClient
from app.upstream.forecast_client import ForecastClient
from app.upstream.sentiment_client import SentimentClient
from app.upstream.portfolio_client import PortfolioClient


@pytest.fixture
def mock_inference_client():
    """Mock inference client"""
    client = MagicMock(spec=InferenceClient)
    client.predict = AsyncMock(return_value={
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
    })
    return client


@pytest.fixture
def mock_forecast_client():
    """Mock forecast client"""
    client = MagicMock(spec=ForecastClient)
    client.get_forecast = AsyncMock(return_value={
        "forecasts": [
            {
                "symbol": "AAPL",
                "current_price": 175.50,
                "predictions": {
                    "1d": {"price": 176.20, "confidence": 0.75}
                },
                "technicals": {
                    "rsi": 62.5,
                    "macd": {"value": 1.2, "signal": 0.8},
                    "support": 172.00,
                    "resistance": 180.00,
                    "trend": "bullish"
                },
                "sector": "TECH",
                "volatility": 0.025,
                "avg_daily_volume": 75000000
            }
        ]
    })
    return client


@pytest.fixture
def mock_sentiment_client():
    """Mock sentiment client"""
    client = MagicMock(spec=SentimentClient)
    client.get_sentiment = AsyncMock(return_value={
        "sentiments": [
            {
                "symbol": "AAPL",
                "overall_score": 0.65,
                "classification": "positive",
                "confidence": 0.82,
                "sources": {
                    "news": {"score": 0.70, "article_count": 45},
                    "social": {"score": 0.60, "mention_count": 1200},
                    "analyst": {"score": 0.75, "rating_count": 12}
                }
            }
        ]
    })
    return client


@pytest.fixture
def mock_portfolio_client():
    """Mock portfolio client"""
    client = MagicMock(spec=PortfolioClient)
    client.get_portfolio = AsyncMock(return_value={
        "user_id": "user123",
        "total_value": 50000.00,
        "cash": 10000.00,
        "buying_power": 15000.00,
        "daily_trades": 1,
        "daily_pnl_pct": 0.5,
        "sector_allocations": {"TECH": 0.25},
        "positions": []
    })
    client.get_positions = AsyncMock(return_value={
        "positions": [],
        "total_positions": 0,
        "concentration_risk": "low"
    })
    return client


@pytest.fixture
def plan_aggregator(
    mock_inference_client,
    mock_forecast_client,
    mock_sentiment_client,
    mock_portfolio_client
):
    """Plan aggregator with mocked clients"""
    return PlanAggregator(
        inference_client=mock_inference_client,
        forecast_client=mock_forecast_client,
        sentiment_client=mock_sentiment_client,
        portfolio_client=mock_portfolio_client
    )


class TestPlanAggregatorIntegration:
    """Test plan aggregator with full integration"""

    @pytest.mark.asyncio
    async def test_generate_plan_success(self, plan_aggregator):
        """Test successful plan generation with all upstreams"""
        with patch('app.aggregation.plan_aggregator.get_guardrail_engine'), \
             patch('app.aggregation.plan_aggregator.get_fitness_checker'), \
             patch('app.aggregation.plan_aggregator.get_decision_store'), \
             patch('app.aggregation.plan_aggregator.get_swr_cache_manager'):

            plan = await plan_aggregator.generate_plan(
                user_id="user123",
                watchlist=["AAPL"],
                mode="beginner"
            )

            assert plan.request_id is not None
            assert isinstance(plan.picks, list)
            assert plan.daily_cap_reached is False
            assert len(plan.degraded_fields) == 0

    @pytest.mark.asyncio
    async def test_generate_plan_with_degraded_upstream(
        self,
        mock_inference_client,
        mock_forecast_client,
        mock_sentiment_client,
        mock_portfolio_client
    ):
        """Test plan generation with degraded sentiment service"""
        # Make sentiment client fail
        mock_sentiment_client.get_sentiment = AsyncMock(
            side_effect=Exception("Service down")
        )

        aggregator = PlanAggregator(
            inference_client=mock_inference_client,
            forecast_client=mock_forecast_client,
            sentiment_client=mock_sentiment_client,
            portfolio_client=mock_portfolio_client
        )

        with patch('app.aggregation.plan_aggregator.get_guardrail_engine'), \
             patch('app.aggregation.plan_aggregator.get_fitness_checker'), \
             patch('app.aggregation.plan_aggregator.get_decision_store'), \
             patch('app.aggregation.plan_aggregator.get_swr_cache_manager'):

            plan = await aggregator.generate_plan(
                user_id="user123",
                watchlist=["AAPL"],
                mode="beginner"
            )

            # Should still succeed with degraded field
            assert "sentiment" in plan.degraded_fields

    @pytest.mark.asyncio
    async def test_generate_plan_empty_watchlist(
        self,
        plan_aggregator,
        mock_portfolio_client
    ):
        """Test plan generation with no watchlist (uses defaults)"""
        # Add some positions for default watchlist
        mock_portfolio_client.get_portfolio = AsyncMock(return_value={
            "user_id": "user123",
            "total_value": 50000.00,
            "positions": [
                {"symbol": "AAPL", "shares": 100}
            ],
            "daily_trades": 0,
            "daily_pnl_pct": 0,
            "sector_allocations": {}
        })

        with patch('app.aggregation.plan_aggregator.get_guardrail_engine'), \
             patch('app.aggregation.plan_aggregator.get_fitness_checker'), \
             patch('app.aggregation.plan_aggregator.get_decision_store'), \
             patch('app.aggregation.plan_aggregator.get_swr_cache_manager'):

            plan = await plan_aggregator.generate_plan(
                user_id="user123",
                watchlist=None,
                mode="beginner"
            )

            assert plan is not None

    @pytest.mark.asyncio
    async def test_generate_plan_daily_cap_reached(
        self,
        plan_aggregator,
        mock_portfolio_client
    ):
        """Test plan generation when daily cap is reached"""
        # Set daily trades to max
        mock_portfolio_client.get_portfolio = AsyncMock(return_value={
            "user_id": "user123",
            "total_value": 50000.00,
            "daily_trades": 3,  # Max trades for beginner
            "daily_pnl_pct": 0,
            "sector_allocations": {},
            "positions": []
        })

        with patch('app.aggregation.plan_aggregator.get_guardrail_engine'), \
             patch('app.aggregation.plan_aggregator.get_fitness_checker'), \
             patch('app.aggregation.plan_aggregator.get_decision_store'), \
             patch('app.aggregation.plan_aggregator.get_swr_cache_manager'):

            plan = await plan_aggregator.generate_plan(
                user_id="user123",
                watchlist=["AAPL"],
                mode="beginner"
            )

            assert plan.daily_cap_reached is True
