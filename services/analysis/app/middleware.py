import time
import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from typing import Dict

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('analysis_http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('analysis_http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
ANALYSIS_OPERATIONS = Counter('analysis_operations_total', 'Analysis operations', ['operation_type', 'status'])
CACHE_OPERATIONS = Counter('analysis_cache_operations_total', 'Cache operations', ['operation', 'result'])

class AnalysisLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured logging in Analysis API"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info("Analysis request started", 
                   method=request.method,
                   url=str(request.url),
                   client_ip=request.client.host)
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info("Analysis request completed",
                       method=request.method,
                       url=str(request.url),
                       status_code=response.status_code,
                       duration=duration)
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            logger.error("Analysis request failed",
                        method=request.method,
                        url=str(request.url),
                        error=str(e),
                        duration=duration)
            raise

class AnalysisMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for Prometheus metrics in Analysis API"""
    
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

class AnalysisRateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for Analysis API"""
    
    def __init__(self, app, calls_per_minute: int = 60):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.clients: Dict[str, list] = {}
    
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
            logger.warning("Analysis API rate limit exceeded", client_ip=client_ip)
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": "60"}
            )
        
        # Add current request
        self.clients[client_ip].append(current_time)
        
        return await call_next(request)

def setup_middleware(app: FastAPI):
    """Setup all middleware for the Analysis API FastAPI app"""
    
    # Add custom middleware
    app.add_middleware(AnalysisLoggingMiddleware)
    app.add_middleware(AnalysisMetricsMiddleware)
    app.add_middleware(AnalysisRateLimitMiddleware, calls_per_minute=60)
    
    # Add metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    # Add metrics helpers
    def record_analysis_operation(operation_type: str, status: str):
        ANALYSIS_OPERATIONS.labels(operation_type=operation_type, status=status).inc()
    
    def record_cache_operation(operation: str, result: str):
        CACHE_OPERATIONS.labels(operation=operation, result=result).inc()
    
    # Make metrics available globally
    app.state.metrics = {
        'analysis_operation': record_analysis_operation,
        'cache_operation': record_cache_operation
    }