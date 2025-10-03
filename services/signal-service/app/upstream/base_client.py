"""
Base Upstream Client with Circuit Breaker Pattern
Prevents cascade failures when upstream services are degraded
"""
import time
import httpx
from typing import Optional, Dict, Any, Literal
from enum import Enum
import structlog

from ..core.observability import UpstreamTimer

logger = structlog.get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit Breaker Pattern Implementation

    States:
    - CLOSED: Normal operation, requests flow through
    - OPEN: Too many failures, reject all requests
    - HALF_OPEN: Testing if service recovered

    Transitions:
    - CLOSED -> OPEN: After failure_threshold consecutive failures
    - OPEN -> HALF_OPEN: After recovery_timeout seconds
    - HALF_OPEN -> CLOSED: After success_threshold consecutive successes
    - HALF_OPEN -> OPEN: On any failure
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout elapsed
            if self.last_failure_time and \
               time.time() - self.last_failure_time >= self.recovery_timeout:
                logger.info(
                    "circuit_breaker_half_open",
                    recovery_timeout=self.recovery_timeout,
                    message="Testing service recovery"
                )
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise CircuitBreakerError("Circuit breaker is OPEN")

        try:
            # Execute function
            result = func(*args, **kwargs)

            # Success
            self._on_success()
            return result

        except Exception as e:
            # Failure
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                logger.info(
                    "circuit_breaker_closed",
                    success_count=self.success_count,
                    message="Service recovered"
                )
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Immediately open on any failure in half-open
            logger.warning(
                "circuit_breaker_open",
                reason="failure_in_half_open",
                message="Service still degraded"
            )
            self.state = CircuitState.OPEN
            self.success_count = 0
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                logger.warning(
                    "circuit_breaker_open",
                    failure_count=self.failure_count,
                    failure_threshold=self.failure_threshold,
                    message="Too many failures"
                )
                self.state = CircuitState.OPEN

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time
        }


class UpstreamClient:
    """
    Base client for upstream service communication

    Features:
    - Circuit breaker pattern (prevent cascade failures)
    - Configurable timeouts
    - Automatic retries with exponential backoff
    - Request/response logging
    - Latency tracking
    """

    def __init__(
        self,
        service_name: str,
        base_url: str,
        timeout_ms: int = 5000,
        max_retries: int = 2,
        circuit_breaker_config: Optional[Dict[str, int]] = None
    ):
        self.service_name = service_name
        self.base_url = base_url.rstrip("/")
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries

        # HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_ms / 1000.0),
            follow_redirects=True
        )

        # Circuit breaker
        cb_config = circuit_breaker_config or {}
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=cb_config.get("failure_threshold", 5),
            recovery_timeout=cb_config.get("recovery_timeout", 60),
            success_threshold=cb_config.get("success_threshold", 2)
        )

        logger.info(
            "upstream_client_initialized",
            service=service_name,
            base_url=base_url,
            timeout_ms=timeout_ms
        )

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """GET request with circuit breaker"""
        return await self._request("GET", endpoint, params=params, headers=headers)

    async def post(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """POST request with circuit breaker"""
        return await self._request("POST", endpoint, json_data=json_data, headers=headers)

    async def _request(
        self,
        method: Literal["GET", "POST"],
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Execute HTTP request with circuit breaker and retries"""
        url = f"{self.base_url}{endpoint}"

        def make_request():
            """Sync wrapper for circuit breaker"""
            import asyncio
            return asyncio.run(self._make_request(method, url, params, json_data, headers))

        try:
            # Execute with circuit breaker
            response = self.circuit_breaker.call(make_request)
            return response

        except CircuitBreakerError as e:
            logger.error(
                "upstream_circuit_breaker_open",
                service=self.service_name,
                endpoint=endpoint,
                circuit_state=self.circuit_breaker.get_state()
            )
            raise

        except Exception as e:
            logger.error(
                "upstream_request_failed",
                service=self.service_name,
                method=method,
                endpoint=endpoint,
                error=str(e),
                exc_info=True
            )
            raise

    async def _make_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]],
        json_data: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Make actual HTTP request with retries"""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                with UpstreamTimer(self.service_name):
                    if method == "GET":
                        response = await self.client.get(url, params=params, headers=headers)
                    else:  # POST
                        response = await self.client.post(url, json=json_data, headers=headers)

                    response.raise_for_status()

                    logger.debug(
                        "upstream_request_success",
                        service=self.service_name,
                        method=method,
                        url=url,
                        status_code=response.status_code,
                        attempt=attempt + 1
                    )

                    return response.json()

            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(
                    "upstream_request_timeout",
                    service=self.service_name,
                    method=method,
                    url=url,
                    timeout_ms=self.timeout_ms,
                    attempt=attempt + 1,
                    max_retries=self.max_retries
                )

                if attempt < self.max_retries:
                    # Exponential backoff
                    backoff = (2 ** attempt) * 0.1  # 100ms, 200ms, 400ms...
                    await self._sleep(backoff)
                    continue
                else:
                    raise

            except httpx.HTTPStatusError as e:
                logger.error(
                    "upstream_http_error",
                    service=self.service_name,
                    method=method,
                    url=url,
                    status_code=e.response.status_code,
                    response_body=e.response.text[:500]
                )
                raise

            except Exception as e:
                last_exception = e
                logger.error(
                    "upstream_request_exception",
                    service=self.service_name,
                    method=method,
                    url=url,
                    error=str(e),
                    attempt=attempt + 1
                )

                if attempt < self.max_retries:
                    backoff = (2 ** attempt) * 0.1
                    await self._sleep(backoff)
                    continue
                else:
                    raise

        # All retries exhausted
        if last_exception:
            raise last_exception

    async def _sleep(self, seconds: float):
        """Async sleep (for testing)"""
        import asyncio
        await asyncio.sleep(seconds)

    async def health_check(self) -> bool:
        """
        Check if upstream service is healthy

        Returns:
            True if healthy, False otherwise
        """
        try:
            await self.get("/health")
            return True
        except Exception:
            return False

    def get_circuit_breaker_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return self.circuit_breaker.get_state()

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
        logger.info("upstream_client_closed", service=self.service_name)
