"""Redis Caching Service for Event Data

High-performance caching layer for frequently accessed events and related data
to reduce database load and improve response times.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

import aioredis
from aioredis import Redis
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Cache strategy types."""
    LRU = "lru"              # Least Recently Used
    TTL = "ttl"              # Time To Live
    WRITE_THROUGH = "write_through"   # Write to cache and DB simultaneously
    WRITE_BEHIND = "write_behind"     # Write to cache first, DB later
    READ_THROUGH = "read_through"     # Read from cache, fallback to DB


class CacheKeyType(str, Enum):
    """Cache key types for namespacing."""
    EVENT = "event"
    EVENT_LIST = "event_list"
    HEADLINE = "headline"
    SEARCH_RESULT = "search"
    AGGREGATION = "agg"
    CLUSTER = "cluster"
    SENTIMENT = "sentiment"
    ENRICHMENT = "enrichment"


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""
    ttl_seconds: int = 3600  # 1 hour default
    max_size: int = 10000    # Maximum items in cache
    strategy: CacheStrategy = CacheStrategy.TTL
    compression: bool = True
    serialization: str = "json"  # json, pickle, msgpack


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_size_bytes: int = 0
    avg_response_time_ms: float = 0.0
    hit_rate: float = 0.0


class EventCacheService:
    """Redis-based caching service for Event Data Service."""
    
    def __init__(self):
        """Initialize the cache service."""
        # Redis configuration
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_db = int(os.getenv("EVENT_CACHE_DB", "1"))
        self.redis_password = os.getenv("REDIS_PASSWORD")
        
        # Cache configuration
        self.enabled = os.getenv("EVENT_CACHE_ENABLED", "true").lower() == "true"
        self.default_ttl = int(os.getenv("EVENT_CACHE_TTL_SECONDS", "3600"))
        self.max_connections = int(os.getenv("EVENT_CACHE_MAX_CONNECTIONS", "10"))
        self.key_prefix = os.getenv("EVENT_CACHE_KEY_PREFIX", "event_cache")
        
        # Performance settings
        self.compression_threshold = int(os.getenv("EVENT_CACHE_COMPRESSION_THRESHOLD", "1024"))
        self.batch_size = int(os.getenv("EVENT_CACHE_BATCH_SIZE", "100"))
        self.timeout = float(os.getenv("EVENT_CACHE_TIMEOUT", "5.0"))
        
        # Cache strategies per data type
        self.cache_configs = {
            CacheKeyType.EVENT: CacheConfig(ttl_seconds=self.default_ttl),
            CacheKeyType.EVENT_LIST: CacheConfig(ttl_seconds=300),  # 5 minutes for lists
            CacheKeyType.HEADLINE: CacheConfig(ttl_seconds=1800),   # 30 minutes
            CacheKeyType.SEARCH_RESULT: CacheConfig(ttl_seconds=600),  # 10 minutes
            CacheKeyType.AGGREGATION: CacheConfig(ttl_seconds=900),    # 15 minutes
            CacheKeyType.CLUSTER: CacheConfig(ttl_seconds=3600),       # 1 hour
            CacheKeyType.SENTIMENT: CacheConfig(ttl_seconds=1800),     # 30 minutes
            CacheKeyType.ENRICHMENT: CacheConfig(ttl_seconds=7200),    # 2 hours
        }
        
        # Redis client
        self.redis: Optional[Redis] = None
        
        # Statistics
        self.stats = CacheStats()
        self._stats_lock = asyncio.Lock()
        
        logger.info(f"EventCacheService initialized (enabled={self.enabled})")
    
    async def start(self):
        """Start the cache service."""
        if not self.enabled:
            logger.info("Event cache service disabled")
            return
        
        try:
            # Create Redis connection pool
            self.redis = await aioredis.from_url(
                self.redis_url,
                db=self.redis_db,
                password=self.redis_password,
                max_connections=self.max_connections,
                socket_timeout=self.timeout,
                decode_responses=True
            )
            
            # Test connection
            await self.redis.ping()
            logger.info("Connected to Redis cache")
            
            # Initialize cache statistics
            await self._init_cache_stats()
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.enabled = False
    
    async def stop(self):
        """Stop the cache service."""
        if self.redis:
            await self.redis.close()
            logger.info("Redis cache connection closed")
    
    async def _init_cache_stats(self):
        """Initialize cache statistics tracking."""
        try:
            # Reset daily stats if needed
            stats_key = f"{self.key_prefix}:stats:daily"
            today = datetime.utcnow().strftime("%Y-%m-%d")
            
            current_date = await self.redis.hget(stats_key, "date")
            if current_date != today:
                # Reset daily stats
                await self.redis.delete(stats_key)
                await self.redis.hset(stats_key, mapping={
                    "date": today,
                    "hits": 0,
                    "misses": 0,
                    "sets": 0,
                    "deletes": 0,
                    "errors": 0
                })
        except Exception as e:
            logger.warning(f"Failed to initialize cache stats: {e}")
    
    def _make_cache_key(self, key_type: CacheKeyType, identifier: str, **kwargs) -> str:
        """Create a standardized cache key."""
        parts = [self.key_prefix, key_type.value, identifier]
        
        # Add additional parameters for compound keys
        if kwargs:
            # Sort kwargs for consistent key generation
            sorted_params = sorted(kwargs.items())
            params_str = "&".join(f"{k}={v}" for k, v in sorted_params)
            # Hash long parameter strings
            if len(params_str) > 100:
                params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
                parts.append(params_hash)
            else:
                parts.append(params_str.replace(":", "_"))
        
        return ":".join(parts)
    
    async def _serialize_data(self, data: Any, compression: bool = True) -> str:
        """Serialize data for cache storage."""
        try:
            # Convert to JSON
            if hasattr(data, 'dict'):  # Pydantic model
                json_data = json.dumps(data.dict(), default=str)
            elif hasattr(data, '__dict__'):  # Regular object
                json_data = json.dumps(data.__dict__, default=str)
            else:
                json_data = json.dumps(data, default=str)
            
            # Compress if data is large enough
            if compression and len(json_data) > self.compression_threshold:
                import gzip
                import base64
                compressed = gzip.compress(json_data.encode('utf-8'))
                return f"gzip:{base64.b64encode(compressed).decode('ascii')}"
            
            return json_data
            
        except Exception as e:
            logger.error(f"Failed to serialize cache data: {e}")
            raise
    
    async def _deserialize_data(self, cached_data: str) -> Any:
        """Deserialize data from cache storage."""
        try:
            # Check if data is compressed
            if cached_data.startswith("gzip:"):
                import gzip
                import base64
                compressed_data = base64.b64decode(cached_data[5:])
                json_data = gzip.decompress(compressed_data).decode('utf-8')
            else:
                json_data = cached_data
            
            return json.loads(json_data)
            
        except Exception as e:
            logger.error(f"Failed to deserialize cache data: {e}")
            raise
    
    async def _update_stats(self, operation: str, hit: bool = False):
        """Update cache statistics."""
        if not self.enabled:
            return
        
        async with self._stats_lock:
            try:
                stats_key = f"{self.key_prefix}:stats:daily"
                if operation == "get":
                    if hit:
                        self.stats.hits += 1
                        await self.redis.hincrby(stats_key, "hits", 1)
                    else:
                        self.stats.misses += 1
                        await self.redis.hincrby(stats_key, "misses", 1)
                elif operation == "set":
                    self.stats.sets += 1
                    await self.redis.hincrby(stats_key, "sets", 1)
                elif operation == "delete":
                    self.stats.deletes += 1
                    await self.redis.hincrby(stats_key, "deletes", 1)
                elif operation == "error":
                    self.stats.errors += 1
                    await self.redis.hincrby(stats_key, "errors", 1)
                
                # Update hit rate
                total_requests = self.stats.hits + self.stats.misses
                if total_requests > 0:
                    self.stats.hit_rate = self.stats.hits / total_requests
                    
            except Exception as e:
                logger.warning(f"Failed to update cache stats: {e}")
    
    async def get(self, key_type: CacheKeyType, identifier: str, **kwargs) -> Optional[Any]:
        """Get item from cache."""
        if not self.enabled or not self.redis:
            return None
        
        try:
            cache_key = self._make_cache_key(key_type, identifier, **kwargs)
            start_time = datetime.utcnow()
            
            cached_data = await self.redis.get(cache_key)
            
            if cached_data:
                data = await self._deserialize_data(cached_data)
                await self._update_stats("get", hit=True)
                
                # Update access time for LRU
                if self.cache_configs[key_type].strategy == CacheStrategy.LRU:
                    await self.redis.expire(cache_key, self.cache_configs[key_type].ttl_seconds)
                
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                logger.debug(f"Cache HIT for {cache_key} ({response_time:.2f}ms)")
                return data
            else:
                await self._update_stats("get", hit=False)
                logger.debug(f"Cache MISS for {cache_key}")
                return None
                
        except Exception as e:
            logger.error(f"Cache get error for {key_type}:{identifier}: {e}")
            await self._update_stats("error")
            return None
    
    async def set(self, key_type: CacheKeyType, identifier: str, data: Any, 
                  ttl: Optional[int] = None, **kwargs) -> bool:
        """Set item in cache."""
        if not self.enabled or not self.redis:
            return False
        
        try:
            cache_key = self._make_cache_key(key_type, identifier, **kwargs)
            config = self.cache_configs[key_type]
            
            # Use provided TTL or default from config
            ttl_seconds = ttl or config.ttl_seconds
            
            # Serialize data
            serialized_data = await self._serialize_data(data, config.compression)
            
            # Set with expiration
            await self.redis.setex(cache_key, ttl_seconds, serialized_data)
            await self._update_stats("set")
            
            logger.debug(f"Cache SET for {cache_key} (TTL: {ttl_seconds}s)")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for {key_type}:{identifier}: {e}")
            await self._update_stats("error")
            return False
    
    async def delete(self, key_type: CacheKeyType, identifier: str, **kwargs) -> bool:
        """Delete item from cache."""
        if not self.enabled or not self.redis:
            return False
        
        try:
            cache_key = self._make_cache_key(key_type, identifier, **kwargs)
            result = await self.redis.delete(cache_key)
            await self._update_stats("delete")
            
            logger.debug(f"Cache DELETE for {cache_key}")
            return result > 0
            
        except Exception as e:
            logger.error(f"Cache delete error for {key_type}:{identifier}: {e}")
            await self._update_stats("error")
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete multiple keys matching a pattern."""
        if not self.enabled or not self.redis:
            return 0
        
        try:
            keys = await self.redis.keys(f"{self.key_prefix}:{pattern}")
            if keys:
                deleted_count = await self.redis.delete(*keys)
                await self._update_stats("delete")
                logger.debug(f"Cache DELETE pattern {pattern}: {deleted_count} keys")
                return deleted_count
            return 0
            
        except Exception as e:
            logger.error(f"Cache delete pattern error for {pattern}: {e}")
            await self._update_stats("error")
            return 0
    
    async def invalidate_event(self, event_id: str, symbol: str = None):
        """Invalidate all cache entries related to an event."""
        patterns_to_clear = [
            f"event:{event_id}",
            f"event_list:*",  # Clear all event lists
            f"search:*",      # Clear all search results
            f"agg:*"          # Clear all aggregations
        ]
        
        if symbol:
            patterns_to_clear.extend([
                f"event_list:*symbol={symbol}*",
                f"search:*symbol={symbol}*",
                f"cluster:*{symbol}*",
                f"sentiment:*{symbol}*"
            ])
        
        for pattern in patterns_to_clear:
            await self.delete_pattern(pattern)
    
    async def invalidate_symbol(self, symbol: str):
        """Invalidate all cache entries related to a symbol."""
        patterns_to_clear = [
            f"*symbol={symbol}*",
            f"*{symbol}*"
        ]
        
        for pattern in patterns_to_clear:
            await self.delete_pattern(pattern)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get current cache statistics."""
        if not self.enabled or not self.redis:
            return {"enabled": False}
        
        try:
            # Get Redis info
            redis_info = await self.redis.info("memory")
            redis_stats = await self.redis.info("stats")
            
            # Get daily stats
            stats_key = f"{self.key_prefix}:stats:daily"
            daily_stats = await self.redis.hgetall(stats_key)
            
            # Calculate cache size
            keys = await self.redis.keys(f"{self.key_prefix}:*")
            cache_size = len(keys)
            
            return {
                "enabled": True,
                "connection_status": "connected",
                "total_keys": cache_size,
                "memory_usage_bytes": redis_info.get("used_memory", 0),
                "daily_stats": {
                    "hits": int(daily_stats.get("hits", 0)),
                    "misses": int(daily_stats.get("misses", 0)),
                    "sets": int(daily_stats.get("sets", 0)),
                    "deletes": int(daily_stats.get("deletes", 0)),
                    "errors": int(daily_stats.get("errors", 0)),
                    "hit_rate": self.stats.hit_rate
                },
                "redis_stats": {
                    "total_commands_processed": redis_stats.get("total_commands_processed", 0),
                    "keyspace_hits": redis_stats.get("keyspace_hits", 0),
                    "keyspace_misses": redis_stats.get("keyspace_misses", 0),
                    "connected_clients": redis_stats.get("connected_clients", 0)
                },
                "configuration": {
                    "default_ttl": self.default_ttl,
                    "max_connections": self.max_connections,
                    "compression_threshold": self.compression_threshold,
                    "timeout": self.timeout
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"enabled": True, "error": str(e)}
    
    async def clear_all(self) -> bool:
        """Clear all cache data (use with caution)."""
        if not self.enabled or not self.redis:
            return False
        
        try:
            keys = await self.redis.keys(f"{self.key_prefix}:*")
            if keys:
                await self.redis.delete(*keys)
                logger.warning(f"Cleared all cache data: {len(keys)} keys")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False


def build_cache_service() -> EventCacheService:
    """Factory function to create cache service instance."""
    return EventCacheService()