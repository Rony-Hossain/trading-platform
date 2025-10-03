"""
SWR (Stale-While-Revalidate) Cache Manager
Serves stale data while asynchronously revalidating in background
"""
import json
import time
import asyncio
import hashlib
from typing import Optional, Dict, Any, Callable, Awaitable
from datetime import datetime, timedelta
from redis import Redis
import structlog

logger = structlog.get_logger(__name__)


class CacheEntry:
    """Cache entry with metadata"""
    def __init__(
        self,
        key: str,
        value: Any,
        etag: str,
        cached_at: float,
        ttl_seconds: int,
        stale_ttl_seconds: int
    ):
        self.key = key
        self.value = value
        self.etag = etag
        self.cached_at = cached_at
        self.ttl_seconds = ttl_seconds
        self.stale_ttl_seconds = stale_ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Age of cache entry in seconds"""
        return time.time() - self.cached_at

    @property
    def is_fresh(self) -> bool:
        """Check if cache entry is fresh"""
        return self.age_seconds < self.ttl_seconds

    @property
    def is_stale(self) -> bool:
        """Check if cache entry is stale but still valid"""
        return self.ttl_seconds <= self.age_seconds < self.stale_ttl_seconds

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is completely expired"""
        return self.age_seconds >= self.stale_ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "value": self.value,
            "etag": self.etag,
            "cached_at": self.cached_at,
            "ttl_seconds": self.ttl_seconds,
            "stale_ttl_seconds": self.stale_ttl_seconds
        }

    @classmethod
    def from_dict(cls, key: str, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary"""
        return cls(
            key=key,
            value=data["value"],
            etag=data["etag"],
            cached_at=data["cached_at"],
            ttl_seconds=data["ttl_seconds"],
            stale_ttl_seconds=data["stale_ttl_seconds"]
        )


class SWRCacheManager:
    """
    Stale-While-Revalidate Cache Manager

    Pattern:
    1. If cache is fresh: return immediately
    2. If cache is stale: return stale data + trigger background revalidation
    3. If cache is expired or missing: fetch fresh data (blocking)

    Features:
    - ETag support for conditional fetching
    - Background revalidation (non-blocking)
    - Graceful degradation (return stale on revalidation failure)
    - Request coalescing (prevent duplicate fetches)
    """

    def __init__(
        self,
        redis_client: Redis,
        default_ttl: int = 30,
        default_stale_ttl: int = 120
    ):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.default_stale_ttl = default_stale_ttl

        # In-flight revalidation requests (prevent duplicate fetches)
        self._revalidation_locks: Dict[str, asyncio.Lock] = {}
        self._revalidation_tasks: Dict[str, asyncio.Task] = {}

    async def get_or_fetch(
        self,
        key: str,
        fetch_fn: Callable[[], Awaitable[Any]],
        ttl_seconds: Optional[int] = None,
        stale_ttl_seconds: Optional[int] = None,
        etag: Optional[str] = None
    ) -> tuple[Any, bool]:
        """
        Get from cache or fetch fresh data

        Args:
            key: Cache key
            fetch_fn: Async function to fetch fresh data
            ttl_seconds: Fresh TTL (default: 30s)
            stale_ttl_seconds: Stale TTL (default: 120s)
            etag: Optional ETag for conditional fetching

        Returns:
            (value, is_stale) - value and whether it's stale data
        """
        ttl = ttl_seconds or self.default_ttl
        stale_ttl = stale_ttl_seconds or self.default_stale_ttl

        # Get from cache
        entry = await self._get_entry(key)

        # Cache miss or expired - fetch fresh (blocking)
        if entry is None or entry.is_expired:
            logger.debug(
                "swr_cache_miss",
                key=key,
                reason="expired" if entry else "not_found"
            )
            value = await self._fetch_and_cache(key, fetch_fn, ttl, stale_ttl)
            return value, False

        # Cache hit - fresh data
        if entry.is_fresh:
            logger.debug(
                "swr_cache_hit_fresh",
                key=key,
                age_seconds=entry.age_seconds
            )
            return entry.value, False

        # Cache hit - stale data (trigger background revalidation)
        if entry.is_stale:
            logger.info(
                "swr_cache_hit_stale",
                key=key,
                age_seconds=entry.age_seconds,
                triggering_revalidation=True
            )

            # Trigger background revalidation (non-blocking)
            self._trigger_revalidation(key, fetch_fn, ttl, stale_ttl)

            return entry.value, True

        # Fallback (should not reach here)
        return entry.value, False

    async def _get_entry(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry from Redis"""
        try:
            cache_key = f"swr:cache:{key}"
            data = self.redis.get(cache_key)

            if not data:
                return None

            entry_dict = json.loads(data)
            return CacheEntry.from_dict(key, entry_dict)

        except Exception as e:
            logger.error("swr_cache_get_failed", key=key, error=str(e))
            return None

    async def _fetch_and_cache(
        self,
        key: str,
        fetch_fn: Callable[[], Awaitable[Any]],
        ttl_seconds: int,
        stale_ttl_seconds: int
    ) -> Any:
        """Fetch fresh data and cache it"""
        try:
            # Fetch fresh data
            start_time = time.time()
            value = await fetch_fn()
            fetch_latency = (time.time() - start_time) * 1000

            # Generate ETag
            etag = self._generate_etag(value)

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                etag=etag,
                cached_at=time.time(),
                ttl_seconds=ttl_seconds,
                stale_ttl_seconds=stale_ttl_seconds
            )

            # Store in Redis with stale TTL
            cache_key = f"swr:cache:{key}"
            self.redis.setex(
                cache_key,
                stale_ttl_seconds,
                json.dumps(entry.to_dict())
            )

            logger.info(
                "swr_cache_fetched_and_stored",
                key=key,
                fetch_latency_ms=fetch_latency,
                ttl_seconds=ttl_seconds,
                stale_ttl_seconds=stale_ttl_seconds,
                etag=etag
            )

            return value

        except Exception as e:
            logger.error(
                "swr_cache_fetch_failed",
                key=key,
                error=str(e),
                exc_info=True
            )
            raise

    def _trigger_revalidation(
        self,
        key: str,
        fetch_fn: Callable[[], Awaitable[Any]],
        ttl_seconds: int,
        stale_ttl_seconds: int
    ):
        """Trigger background revalidation (non-blocking)"""
        # Check if revalidation already in progress
        if key in self._revalidation_tasks:
            task = self._revalidation_tasks[key]
            if not task.done():
                logger.debug("swr_revalidation_already_in_progress", key=key)
                return

        # Create revalidation task
        task = asyncio.create_task(
            self._revalidate(key, fetch_fn, ttl_seconds, stale_ttl_seconds)
        )
        self._revalidation_tasks[key] = task

        # Cleanup task when done
        task.add_done_callback(lambda t: self._cleanup_task(key))

    async def _revalidate(
        self,
        key: str,
        fetch_fn: Callable[[], Awaitable[Any]],
        ttl_seconds: int,
        stale_ttl_seconds: int
    ):
        """Background revalidation task"""
        try:
            logger.debug("swr_revalidation_started", key=key)

            # Fetch fresh data and update cache
            await self._fetch_and_cache(key, fetch_fn, ttl_seconds, stale_ttl_seconds)

            logger.info("swr_revalidation_completed", key=key)

        except Exception as e:
            logger.warning(
                "swr_revalidation_failed",
                key=key,
                error=str(e),
                message="Serving stale data on revalidation failure"
            )
            # Graceful degradation - keep serving stale data

    def _cleanup_task(self, key: str):
        """Cleanup completed revalidation task"""
        if key in self._revalidation_tasks:
            del self._revalidation_tasks[key]

    def _generate_etag(self, value: Any) -> str:
        """Generate ETag from value"""
        try:
            value_json = json.dumps(value, sort_keys=True)
            return hashlib.sha256(value_json.encode()).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(str(value).encode()).hexdigest()[:16]

    async def invalidate(self, key: str) -> bool:
        """
        Invalidate cache entry

        Args:
            key: Cache key

        Returns:
            True if invalidated
        """
        try:
            cache_key = f"swr:cache:{key}"
            result = self.redis.delete(cache_key)

            logger.info("swr_cache_invalidated", key=key)
            return result > 0

        except Exception as e:
            logger.error("swr_cache_invalidate_failed", key=key, error=str(e))
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache entries matching pattern

        Args:
            pattern: Redis pattern (e.g., "plan:*")

        Returns:
            Number of keys invalidated
        """
        try:
            cache_pattern = f"swr:cache:{pattern}"
            cursor = 0
            count = 0

            while True:
                cursor, keys = self.redis.scan(cursor, match=cache_pattern, count=100)

                if keys:
                    deleted = self.redis.delete(*keys)
                    count += deleted

                if cursor == 0:
                    break

            logger.info("swr_cache_pattern_invalidated", pattern=pattern, count=count)
            return count

        except Exception as e:
            logger.error("swr_cache_invalidate_pattern_failed", pattern=pattern, error=str(e))
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            pattern = "swr:cache:*"
            cursor = 0
            total_entries = 0
            fresh_entries = 0
            stale_entries = 0
            expired_entries = 0

            while True:
                cursor, keys = self.redis.scan(cursor, match=pattern, count=100)

                for key in keys:
                    data = self.redis.get(key)
                    if data:
                        total_entries += 1
                        try:
                            entry_dict = json.loads(data)
                            entry = CacheEntry.from_dict(
                                key.decode() if isinstance(key, bytes) else key,
                                entry_dict
                            )

                            if entry.is_fresh:
                                fresh_entries += 1
                            elif entry.is_stale:
                                stale_entries += 1
                            else:
                                expired_entries += 1
                        except Exception:
                            continue

                if cursor == 0:
                    break

            return {
                "total_entries": total_entries,
                "fresh_entries": fresh_entries,
                "stale_entries": stale_entries,
                "expired_entries": expired_entries,
                "active_revalidations": len(self._revalidation_tasks)
            }

        except Exception as e:
            logger.error("swr_cache_get_stats_failed", error=str(e))
            return {}


# Global SWR cache manager instance
_swr_cache_manager: Optional[SWRCacheManager] = None


def init_swr_cache_manager(
    redis_client: Redis,
    default_ttl: int = 30,
    default_stale_ttl: int = 120
) -> SWRCacheManager:
    """
    Initialize global SWR cache manager

    Args:
        redis_client: Redis client
        default_ttl: Default fresh TTL (default: 30s)
        default_stale_ttl: Default stale TTL (default: 120s)

    Returns:
        SWRCacheManager instance
    """
    global _swr_cache_manager
    _swr_cache_manager = SWRCacheManager(redis_client, default_ttl, default_stale_ttl)
    logger.info(
        "swr_cache_manager_initialized",
        default_ttl=default_ttl,
        default_stale_ttl=default_stale_ttl
    )
    return _swr_cache_manager


def get_swr_cache_manager() -> SWRCacheManager:
    """
    Get global SWR cache manager instance

    Returns:
        SWRCacheManager instance

    Raises:
        RuntimeError: If not initialized
    """
    if _swr_cache_manager is None:
        raise RuntimeError("SWRCacheManager not initialized. Call init_swr_cache_manager() first.")
    return _swr_cache_manager
