import redis.asyncio as redis
import json
import structlog
from typing import Optional, Any, Dict
from datetime import datetime, timedelta

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
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.error("Failed to connect to Redis", error=str(e))
        raise

async def get_redis() -> redis.Redis:
    """Get Redis client"""
    global _redis_client
    if _redis_client is None:
        await init_redis()
    return _redis_client

class CacheService:
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
            logger.error("Cache get error", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        try:
            ttl = ttl or settings.CACHE_DEFAULT_TTL
            serialized = json.dumps(value, default=self._json_serializer)
            await self.redis.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error("Cache set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            result = await self.redis.delete(key)
            return bool(result)
        except Exception as e:
            logger.error("Cache delete error", key=key, error=str(e))
            return False
    
    async def get_or_set(self, key: str, fetch_func, ttl: Optional[int] = None) -> Any:
        """Get from cache or fetch and cache the result"""
        # Try to get from cache first
        cached_value = await self.get(key)
        if cached_value is not None:
            return cached_value
        
        # Cache miss - fetch and cache
        try:
            if asyncio.iscoroutinefunction(fetch_func):
                value = await fetch_func()
            else:
                value = fetch_func()
            
            if value is not None:
                await self.set(key, value, ttl)
            return value
        except Exception as e:
            logger.error("Cache get_or_set error", key=key, error=str(e))
            return None
    
    async def get_many(self, keys: list[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        try:
            values = await self.redis.mget(keys)
            result = {}
            for key, value in zip(keys, values):
                if value:
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        continue
            return result
        except Exception as e:
            logger.error("Cache get_many error", keys=keys, error=str(e))
            return {}
    
    async def set_many(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache"""
        try:
            ttl = ttl or settings.CACHE_DEFAULT_TTL
            pipe = self.redis.pipeline()
            
            for key, value in data.items():
                serialized = json.dumps(value, default=self._json_serializer)
                pipe.setex(key, ttl, serialized)
            
            await pipe.execute()
            return True
        except Exception as e:
            logger.error("Cache set_many error", error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            result = await self.redis.exists(key)
            return bool(result)
        except Exception as e:
            logger.error("Cache exists error", key=key, error=str(e))
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key"""
        try:
            result = await self.redis.expire(key, ttl)
            return bool(result)
        except Exception as e:
            logger.error("Cache expire error", key=key, error=str(e))
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error("Cache clear_pattern error", pattern=pattern, error=str(e))
            return 0
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for datetime and other objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'dict'):  # Pydantic models
            return obj.dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Cache key generators
class CacheKeys:
    @staticmethod
    def stock_price(symbol: str) -> str:
        return f"stock:price:{symbol}"
    
    @staticmethod
    def historical_data(symbol: str, period: str) -> str:
        return f"stock:history:{symbol}:{period}"
    
    @staticmethod
    def company_profile(symbol: str) -> str:
        return f"stock:profile:{symbol}"
    
    @staticmethod
    def search_results(query: str) -> str:
        return f"search:symbols:{query.lower()}"
    
    @staticmethod
    def batch_prices(symbols: list[str]) -> str:
        symbols_str = ",".join(sorted(symbols))
        return f"stock:batch:{symbols_str}"