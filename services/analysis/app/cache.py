import redis.asyncio as redis
import json
import structlog
from typing import Optional, Any, Dict
from datetime import datetime

from .config import settings

logger = structlog.get_logger(__name__)

# Global Redis connection
_redis_client: Optional[redis.Redis] = None

async def init_redis():
    """Initialize Redis connection"""
    global _redis_client
    try:
        _redis_client = redis.from_url(
            settings.REDIS_URL,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
            retry_on_timeout=True,
            health_check_interval=30
        )
        # Test connection
        await _redis_client.ping()
        logger.info("Analysis Redis connected successfully")
    except Exception as e:
        logger.error("Failed to connect to Analysis Redis", error=str(e))
        raise

async def get_redis() -> redis.Redis:
    """Get Redis client"""
    global _redis_client
    if _redis_client is None:
        await init_redis()
    return _redis_client

class AnalysisCacheService:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error("Analysis cache get error", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        try:
            ttl = ttl or settings.CACHE_DEFAULT_TTL
            serialized = json.dumps(value, default=self._json_serializer)
            await self.redis.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error("Analysis cache set error", key=key, error=str(e))
            return False
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for datetime and other objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'dict'):  # Pydantic models
            return obj.dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Cache key generators for analysis
class AnalysisCacheKeys:
    @staticmethod
    def technical_analysis(symbol: str, period: str) -> str:
        return f"analysis:technical:{symbol}:{period}"
    
    @staticmethod
    def chart_patterns(symbol: str, period: str) -> str:
        return f"analysis:patterns:{symbol}:{period}"
    
    @staticmethod
    def advanced_indicators(symbol: str, period: str, indicators: str = "all") -> str:
        return f"analysis:advanced:{symbol}:{period}:{indicators}"
    
    @staticmethod
    def forecast(symbol: str, model_type: str, horizon: int) -> str:
        return f"analysis:forecast:{symbol}:{model_type}:{horizon}"
    
    @staticmethod
    def comprehensive(symbol: str, period: str) -> str:
        return f"analysis:comprehensive:{symbol}:{period}"