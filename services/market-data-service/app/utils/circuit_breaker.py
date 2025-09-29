"""
Circuit breaker implementation for provider resilience
"""

import time
import asyncio
import logging
from typing import Callable, Any, Optional
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"     # Normal operation
    OPEN = "open"         # Circuit is open, calls are blocked
    HALF_OPEN = "half_open"  # Testing if service is back

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5  # Number of failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds to wait before trying again
    expected_exception: type = Exception
    timeout: float = 30.0  # Request timeout in seconds

@dataclass
class CircuitBreakerStats:
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    circuit_opens: int = 0

class CircuitBreaker:
    """
    Circuit breaker pattern implementation for provider resilience
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            self.stats.total_requests += 1
            
            # Check if circuit is open
            if self.stats.state == CircuitState.OPEN:
                if time.time() - self.stats.last_failure_time < self.config.recovery_timeout:
                    raise CircuitOpenException(f"Circuit breaker {self.name} is OPEN")
                else:
                    # Try to recover
                    self.stats.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Success - reset circuit if it was half-open
            async with self._lock:
                if self.stats.state == CircuitState.HALF_OPEN:
                    self.stats.state = CircuitState.CLOSED
                    self.stats.failure_count = 0
                    logger.info(f"Circuit breaker {self.name} recovered to CLOSED state")
                
                self.stats.successful_requests += 1
            
            return result
            
        except asyncio.TimeoutError:
            async with self._lock:
                self.stats.timeouts += 1
                await self._record_failure()
            raise CircuitTimeoutException(f"Request to {self.name} timed out after {self.config.timeout}s")
            
        except self.config.expected_exception as e:
            async with self._lock:
                await self._record_failure()
            raise e
    
    async def _record_failure(self):
        """Record a failure and potentially open the circuit"""
        self.stats.failed_requests += 1
        self.stats.failure_count += 1
        self.stats.last_failure_time = time.time()
        
        if self.stats.failure_count >= self.config.failure_threshold:
            if self.stats.state != CircuitState.OPEN:
                self.stats.state = CircuitState.OPEN
                self.stats.circuit_opens += 1
                logger.warning(
                    f"Circuit breaker {self.name} opened after {self.stats.failure_count} failures"
                )
    
    def get_stats(self) -> dict:
        """Get current circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.stats.state.value,
            "failure_count": self.stats.failure_count,
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "timeouts": self.stats.timeouts,
            "circuit_opens": self.stats.circuit_opens,
            "success_rate": (
                self.stats.successful_requests / max(self.stats.total_requests, 1) * 100
            ),
            "last_failure_time": self.stats.last_failure_time,
            "is_healthy": self.stats.state == CircuitState.CLOSED
        }
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        self.stats = CircuitBreakerStats()
        logger.info(f"Circuit breaker {self.name} reset")

class CircuitOpenException(Exception):
    """Raised when circuit breaker is open"""
    pass

class CircuitTimeoutException(Exception):
    """Raised when request times out"""
    pass

class ProviderCircuitBreakers:
    """Manager for multiple provider circuit breakers"""
    
    def __init__(self):
        self.breakers: dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, provider_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for provider"""
        if provider_name not in self.breakers:
            config = CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                timeout=30.0
            )
            self.breakers[provider_name] = CircuitBreaker(provider_name, config)
        
        return self.breakers[provider_name]
    
    def get_all_stats(self) -> dict[str, dict]:
        """Get stats for all circuit breakers"""
        return {
            name: breaker.get_stats() 
            for name, breaker in self.breakers.items()
        }
    
    def get_health_summary(self) -> dict:
        """Get overall health summary"""
        total_breakers = len(self.breakers)
        healthy_breakers = sum(
            1 for breaker in self.breakers.values() 
            if breaker.get_stats()["is_healthy"]
        )
        
        return {
            "total_providers": total_breakers,
            "healthy_providers": healthy_breakers,
            "unhealthy_providers": total_breakers - healthy_breakers,
            "overall_health": "healthy" if healthy_breakers == total_breakers else "degraded"
        }

# Global instance
circuit_breakers = ProviderCircuitBreakers()