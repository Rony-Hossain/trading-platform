"""
Upstream service clients package
"""
from .base_client import UpstreamClient, CircuitBreakerError

__all__ = ["UpstreamClient", "CircuitBreakerError"]
