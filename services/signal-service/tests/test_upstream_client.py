"""
Tests for Upstream Clients and Circuit Breaker
"""
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
import time

from app.upstream.base_client import (
    UpstreamClient,
    CircuitBreaker,
    CircuitState,
    CircuitBreakerError
)


class TestCircuitBreaker:
    """Test Circuit Breaker pattern"""

    def test_circuit_breaker_initial_state(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_stays_closed_on_success(self):
        cb = CircuitBreaker(failure_threshold=3)

        # Successful calls
        for _ in range(5):
            result = cb.call(lambda: "success")
            assert result == "success"
            assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3)

        # Fail 3 times
        for i in range(3):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Circuit should be open
        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_rejects_requests_when_open(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10)

        # Fail enough to open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.state == CircuitState.OPEN

        # Should reject requests
        with pytest.raises(CircuitBreakerError, match="Circuit breaker is OPEN"):
            cb.call(lambda: "success")

    def test_circuit_breaker_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # Open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(1.1)

        # Next call should transition to HALF_OPEN
        result = cb.call(lambda: "success")
        assert result == "success"
        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_breaker_closes_after_successful_half_open(self):
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=1,
            success_threshold=2
        )

        # Open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Wait and transition to HALF_OPEN
        time.sleep(1.1)

        # Succeed enough times to close
        for _ in range(2):
            cb.call(lambda: "success")

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_reopens_on_half_open_failure(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # Open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Wait and transition to HALF_OPEN
        time.sleep(1.1)
        cb.call(lambda: "success")
        assert cb.state == CircuitState.HALF_OPEN

        # Fail again - should immediately reopen
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_get_state(self):
        cb = CircuitBreaker(failure_threshold=3)

        state = cb.get_state()
        assert state["state"] == "closed"
        assert state["failure_count"] == 0
        assert state["success_count"] == 0


class TestUpstreamClient:
    """Test Upstream Client base functionality"""

    @pytest.fixture
    def mock_httpx_client(self):
        """Mock httpx AsyncClient"""
        client = MagicMock()
        client.get = AsyncMock()
        client.post = AsyncMock()
        client.aclose = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        client = UpstreamClient(
            service_name="test-service",
            base_url="http://localhost:8000",
            timeout_ms=5000
        )

        assert client.service_name == "test-service"
        assert client.base_url == "http://localhost:8000"
        assert client.timeout_ms == 5000
        assert client.circuit_breaker.state == CircuitState.CLOSED

        await client.close()

    @pytest.mark.asyncio
    async def test_successful_get_request(self, mock_httpx_client):
        client = UpstreamClient(
            service_name="test-service",
            base_url="http://localhost:8000"
        )

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, 'client', mock_httpx_client):
            mock_httpx_client.get.return_value = mock_response

            result = await client.get("/test", params={"foo": "bar"})

            assert result == {"result": "success"}
            mock_httpx_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_successful_post_request(self, mock_httpx_client):
        client = UpstreamClient(
            service_name="test-service",
            base_url="http://localhost:8000"
        )

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "created"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, 'client', mock_httpx_client):
            mock_httpx_client.post.return_value = mock_response

            result = await client.post("/test", json_data={"data": "value"})

            assert result == {"result": "created"}
            mock_httpx_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_timeout_with_retries(self, mock_httpx_client):
        client = UpstreamClient(
            service_name="test-service",
            base_url="http://localhost:8000",
            timeout_ms=100,
            max_retries=2
        )

        # Mock timeout on first 2 attempts, success on 3rd
        mock_httpx_client.get.side_effect = [
            httpx.TimeoutException("Timeout"),
            httpx.TimeoutException("Timeout"),
            MagicMock(status_code=200, json=lambda: {"result": "success"}, raise_for_status=MagicMock())
        ]

        with patch.object(client, 'client', mock_httpx_client):
            with patch.object(client, '_sleep', new_callable=AsyncMock):
                result = await client.get("/test")
                assert result == {"result": "success"}
                assert mock_httpx_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_request_exhausts_retries(self, mock_httpx_client):
        client = UpstreamClient(
            service_name="test-service",
            base_url="http://localhost:8000",
            max_retries=2
        )

        # Always timeout
        mock_httpx_client.get.side_effect = httpx.TimeoutException("Timeout")

        with patch.object(client, 'client', mock_httpx_client):
            with patch.object(client, '_sleep', new_callable=AsyncMock):
                with pytest.raises(httpx.TimeoutException):
                    await client.get("/test")

                assert mock_httpx_client.get.call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_http_status_error_no_retry(self, mock_httpx_client):
        client = UpstreamClient(
            service_name="test-service",
            base_url="http://localhost:8000",
            max_retries=2
        )

        # Mock 404 error
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=mock_response
        )

        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'client', mock_httpx_client):
            with pytest.raises(httpx.HTTPStatusError):
                await client.get("/test")

            # Should not retry on 4xx errors
            assert mock_httpx_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, mock_httpx_client):
        client = UpstreamClient(
            service_name="test-service",
            base_url="http://localhost:8000",
            max_retries=0,  # No retries for faster test
            circuit_breaker_config={
                "failure_threshold": 2,
                "recovery_timeout": 60
            }
        )

        # Mock failures
        mock_httpx_client.get.side_effect = Exception("Service down")

        with patch.object(client, 'client', mock_httpx_client):
            # Fail twice to open circuit
            for _ in range(2):
                with pytest.raises(Exception):
                    await client.get("/test")

            assert client.circuit_breaker.state == CircuitState.OPEN

            # Next request should be rejected by circuit breaker
            with pytest.raises(CircuitBreakerError):
                await client.get("/test")

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_httpx_client):
        client = UpstreamClient(
            service_name="test-service",
            base_url="http://localhost:8000"
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, 'client', mock_httpx_client):
            mock_httpx_client.get.return_value = mock_response

            is_healthy = await client.health_check()
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_httpx_client):
        client = UpstreamClient(
            service_name="test-service",
            base_url="http://localhost:8000"
        )

        mock_httpx_client.get.side_effect = Exception("Connection failed")

        with patch.object(client, 'client', mock_httpx_client):
            is_healthy = await client.health_check()
            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_base_url_stripping(self):
        client = UpstreamClient(
            service_name="test-service",
            base_url="http://localhost:8000/",  # Trailing slash
        )

        assert client.base_url == "http://localhost:8000"
        await client.close()

    def test_get_circuit_breaker_state(self):
        client = UpstreamClient(
            service_name="test-service",
            base_url="http://localhost:8000"
        )

        state = client.get_circuit_breaker_state()
        assert state["state"] == "closed"
        assert "failure_count" in state
