"""
Integration tests for upstream service degradation scenarios
Tests graceful degradation and circuit breaker behavior
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.aggregation.plan_aggregator import PlanAggregator
from app.upstream.base_client import CircuitBreakerError, CircuitState
from app.upstream.inference_client import InferenceClient
from app.upstream.forecast_client import ForecastClient
from app.upstream.sentiment_client import SentimentClient
from app.upstream.portfolio_client import PortfolioClient


@pytest.fixture
def working_clients():
    """All upstream clients working normally"""
    inference = MagicMock(spec=InferenceClient)
    inference.predict = AsyncMock(return_value={
        "predictions": [{"symbol": "AAPL", "action": "BUY", "confidence": 0.75}],
        "model_metadata": {"version": "v1.2.3"}
    })

    forecast = MagicMock(spec=ForecastClient)
    forecast.get_forecast = AsyncMock(return_value={
        "forecasts": [{
            "symbol": "AAPL",
            "current_price": 175.50,
            "technicals": {"rsi": 60, "trend": "bullish"},
            "volatility": 0.02,
            "avg_daily_volume": 75000000
        }]
    })

    sentiment = MagicMock(spec=SentimentClient)
    sentiment.get_sentiment = AsyncMock(return_value={
        "sentiments": [{"symbol": "AAPL", "overall_score": 0.6, "classification": "positive"}]
    })

    portfolio = MagicMock(spec=PortfolioClient)
    portfolio.get_portfolio = AsyncMock(return_value={
        "total_value": 50000, "daily_trades": 0, "daily_pnl_pct": 0,
        "sector_allocations": {}, "positions": []
    })

    return inference, forecast, sentiment, portfolio


class TestUpstreamDegradation:
    """Test various upstream degradation scenarios"""

    @pytest.mark.asyncio
    async def test_inference_service_down(self, working_clients):
        """Test when inference service is completely down"""
        inference, forecast, sentiment, portfolio = working_clients

        # Make inference service fail
        inference.predict = AsyncMock(side_effect=Exception("Service unavailable"))

        aggregator = PlanAggregator(inference, forecast, sentiment, portfolio)

        with patch('app.aggregation.plan_aggregator.get_guardrail_engine'), \
             patch('app.aggregation.plan_aggregator.get_fitness_checker'), \
             patch('app.aggregation.plan_aggregator.get_decision_store'), \
             patch('app.aggregation.plan_aggregator.get_swr_cache_manager'):

            plan = await aggregator.generate_plan(
                user_id="user123",
                watchlist=["AAPL"],
                mode="beginner"
            )

            # Should succeed with degraded field
            assert "inference" in plan.degraded_fields
            # May have no picks if inference is critical
            assert isinstance(plan.picks, list)

    @pytest.mark.asyncio
    async def test_sentiment_service_down(self, working_clients):
        """Test when sentiment service is down (non-critical)"""
        inference, forecast, sentiment, portfolio = working_clients

        # Make sentiment service fail
        sentiment.get_sentiment = AsyncMock(side_effect=Exception("Service unavailable"))

        aggregator = PlanAggregator(inference, forecast, sentiment, portfolio)

        with patch('app.aggregation.plan_aggregator.get_guardrail_engine'), \
             patch('app.aggregation.plan_aggregator.get_fitness_checker'), \
             patch('app.aggregation.plan_aggregator.get_decision_store'), \
             patch('app.aggregation.plan_aggregator.get_swr_cache_manager'):

            plan = await aggregator.generate_plan(
                user_id="user123",
                watchlist=["AAPL"],
                mode="beginner"
            )

            # Should succeed without sentiment
            assert "sentiment" in plan.degraded_fields
            # Should still have picks from inference + forecast
            assert isinstance(plan.picks, list)

    @pytest.mark.asyncio
    async def test_multiple_services_down(self, working_clients):
        """Test when multiple services are down"""
        inference, forecast, sentiment, portfolio = working_clients

        # Make multiple services fail
        sentiment.get_sentiment = AsyncMock(side_effect=Exception("Service unavailable"))
        forecast.get_forecast = AsyncMock(side_effect=Exception("Service unavailable"))

        aggregator = PlanAggregator(inference, forecast, sentiment, portfolio)

        with patch('app.aggregation.plan_aggregator.get_guardrail_engine'), \
             patch('app.aggregation.plan_aggregator.get_fitness_checker'), \
             patch('app.aggregation.plan_aggregator.get_decision_store'), \
             patch('app.aggregation.plan_aggregator.get_swr_cache_manager'):

            plan = await aggregator.generate_plan(
                user_id="user123",
                watchlist=["AAPL"],
                mode="beginner"
            )

            # Should have both in degraded fields
            assert "sentiment" in plan.degraded_fields
            assert "forecast" in plan.degraded_fields

    @pytest.mark.asyncio
    async def test_slow_service_timeout(self, working_clients):
        """Test when service is slow and times out"""
        inference, forecast, sentiment, portfolio = working_clients

        # Make sentiment service slow (timeout)
        import asyncio
        async def slow_response():
            await asyncio.sleep(10)  # Longer than timeout
            return {}

        sentiment.get_sentiment = AsyncMock(side_effect=slow_response)

        aggregator = PlanAggregator(inference, forecast, sentiment, portfolio)

        with patch('app.aggregation.plan_aggregator.get_guardrail_engine'), \
             patch('app.aggregation.plan_aggregator.get_fitness_checker'), \
             patch('app.aggregation.plan_aggregator.get_decision_store'), \
             patch('app.aggregation.plan_aggregator.get_swr_cache_manager'):

            # Should handle timeout gracefully
            plan = await aggregator.generate_plan(
                user_id="user123",
                watchlist=["AAPL"],
                mode="beginner"
            )

            # Should degrade gracefully
            assert isinstance(plan.degraded_fields, list)

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after repeated failures"""
        from app.upstream.base_client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        # Fail 3 times
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Circuit should be open
        assert cb.state == CircuitState.OPEN

        # Next call should be rejected
        with pytest.raises(CircuitBreakerError):
            cb.call(lambda: "success")

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery to half-open state"""
        from app.upstream.base_client import CircuitBreaker
        import time

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # Open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(1.1)

        # Next success should transition to HALF_OPEN
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_partial_data_from_inference(self, working_clients):
        """Test when inference returns partial/incomplete data"""
        inference, forecast, sentiment, portfolio = working_clients

        # Return incomplete prediction data
        inference.predict = AsyncMock(return_value={
            "predictions": [
                {
                    "symbol": "AAPL",
                    "action": "BUY",
                    # Missing confidence, expected_return
                }
            ],
            "model_metadata": {}
        })

        aggregator = PlanAggregator(inference, forecast, sentiment, portfolio)

        with patch('app.aggregation.plan_aggregator.get_guardrail_engine'), \
             patch('app.aggregation.plan_aggregator.get_fitness_checker'), \
             patch('app.aggregation.plan_aggregator.get_decision_store'), \
             patch('app.aggregation.plan_aggregator.get_swr_cache_manager'):

            # Should handle partial data gracefully
            plan = await aggregator.generate_plan(
                user_id="user123",
                watchlist=["AAPL"],
                mode="beginner"
            )

            assert plan is not None

    @pytest.mark.asyncio
    async def test_all_services_down_fallback(self, working_clients):
        """Test fallback behavior when all services are down"""
        inference, forecast, sentiment, portfolio = working_clients

        # Make all services fail
        inference.predict = AsyncMock(side_effect=Exception("Down"))
        forecast.get_forecast = AsyncMock(side_effect=Exception("Down"))
        sentiment.get_sentiment = AsyncMock(side_effect=Exception("Down"))

        aggregator = PlanAggregator(inference, forecast, sentiment, portfolio)

        with patch('app.aggregation.plan_aggregator.get_guardrail_engine'), \
             patch('app.aggregation.plan_aggregator.get_fitness_checker'), \
             patch('app.aggregation.plan_aggregator.get_decision_store'), \
             patch('app.aggregation.plan_aggregator.get_swr_cache_manager'):

            plan = await aggregator.generate_plan(
                user_id="user123",
                watchlist=["AAPL"],
                mode="beginner"
            )

            # Should return empty plan but not crash
            assert len(plan.degraded_fields) >= 3
            assert len(plan.picks) == 0  # No picks when all critical services down

    @pytest.mark.asyncio
    async def test_intermittent_failures(self, working_clients):
        """Test handling of intermittent service failures"""
        inference, forecast, sentiment, portfolio = working_clients

        call_count = 0

        async def intermittent_failure():
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise Exception("Intermittent failure")
            return {
                "sentiments": [{
                    "symbol": "AAPL",
                    "overall_score": 0.5,
                    "classification": "neutral"
                }]
            }

        sentiment.get_sentiment = AsyncMock(side_effect=intermittent_failure)

        aggregator = PlanAggregator(inference, forecast, sentiment, portfolio)

        with patch('app.aggregation.plan_aggregator.get_guardrail_engine'), \
             patch('app.aggregation.plan_aggregator.get_fitness_checker'), \
             patch('app.aggregation.plan_aggregator.get_decision_store'), \
             patch('app.aggregation.plan_aggregator.get_swr_cache_manager'):

            # First call might fail
            plan1 = await aggregator.generate_plan(
                user_id="user123",
                watchlist=["AAPL"],
                mode="beginner"
            )

            # Second call might succeed
            plan2 = await aggregator.generate_plan(
                user_id="user123",
                watchlist=["AAPL"],
                mode="beginner"
            )

            # Both should return valid plans (one with degradation)
            assert plan1 is not None
            assert plan2 is not None

    @pytest.mark.asyncio
    async def test_degraded_field_metadata(self, working_clients):
        """Test that degraded_fields metadata is properly populated"""
        inference, forecast, sentiment, portfolio = working_clients

        sentiment.get_sentiment = AsyncMock(side_effect=Exception("Down"))

        aggregator = PlanAggregator(inference, forecast, sentiment, portfolio)

        with patch('app.aggregation.plan_aggregator.get_guardrail_engine'), \
             patch('app.aggregation.plan_aggregator.get_fitness_checker'), \
             patch('app.aggregation.plan_aggregator.get_decision_store'), \
             patch('app.aggregation.plan_aggregator.get_swr_cache_manager'):

            plan = await aggregator.generate_plan(
                user_id="user123",
                watchlist=["AAPL"],
                mode="beginner"
            )

            # Verify degraded_fields is properly set
            assert isinstance(plan.degraded_fields, list)
            assert "sentiment" in plan.degraded_fields
            assert "inference" not in plan.degraded_fields  # Should still work
            assert "forecast" not in plan.degraded_fields  # Should still work
