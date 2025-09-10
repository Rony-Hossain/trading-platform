import time
import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import asyncio
from typing import Dict

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
CACHE_HITS = Counter('cache_hits_total', 'Cache hits', ['cache_type'])
CACHE_MISSES = Counter('cache_misses_total', 'Cache misses', ['cache_type'])
EXTERNAL_API_CALLS = Counter('external_api_calls_total', 'External API calls', ['provider', 'status'])

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured logging"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info("Request started", 
                   method=request.method,
                   url=str(request.url),
                   client_ip=request.client.host)
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info("Request completed",
                       method=request.method,
                       url=str(request.url),
                       status_code=response.status_code,
                       duration=duration)
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            logger.error("Request failed",
                        method=request.method,
                        url=str(request.url),
                        error=str(e),
                        duration=duration)
            raise

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for Prometheus metrics"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Extract endpoint (remove query params)
        endpoint = request.url.path
        method = request.method
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=response.status_code).inc()
            REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
            
            return response
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=500).inc()
            REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
            raise

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""
    
    def __init__(self, app, calls_per_minute: int = 100):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.clients: Dict[str, list] = {}
        self.cleanup_task = None
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Initialize client tracking
        if client_ip not in self.clients:
            self.clients[client_ip] = []
        
        # Clean old requests (older than 1 minute)
        self.clients[client_ip] = [
            req_time for req_time in self.clients[client_ip]
            if current_time - req_time < 60
        ]
        
        # Check rate limit
        if len(self.clients[client_ip]) >= self.calls_per_minute:
            logger.warning("Rate limit exceeded", client_ip=client_ip)
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": "60"}
            )
        
        # Add current request
        self.clients[client_ip].append(current_time)
        
        return await call_next(request)

def setup_middleware(app: FastAPI):
    """Setup all middleware for the FastAPI app"""
    
    # Add custom middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RateLimitMiddleware, calls_per_minute=100)
    
    # Add metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    # Add cache metrics helpers
    def record_cache_hit(cache_type: str):
        CACHE_HITS.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(cache_type: str):
        CACHE_MISSES.labels(cache_type=cache_type).inc()
    
    def record_external_api_call(provider: str, status: str):
        EXTERNAL_API_CALLS.labels(provider=provider, status=status).inc()
    
    # Make metrics available globally
    app.state.metrics = {
        'cache_hit': record_cache_hit,
        'cache_miss': record_cache_miss,
        'external_api_call': record_external_api_call
    }