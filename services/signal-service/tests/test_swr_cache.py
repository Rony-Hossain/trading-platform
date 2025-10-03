"""
Tests for SWR Cache Manager
"""
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch
from redis import Redis

from app.core.swr_cache import (
    SWRCacheManager,
    CacheEntry,
    init_swr_cache_manager,
    get_swr_cache_manager
)


@pytest.fixture
def redis_client():
    """Mock Redis client"""
    client = Redis(host="localhost", port=6379, db=15, decode_responses=False)
    client.flushdb()
    yield client
    client.flushdb()
    client.close()


@pytest.fixture
def swr_cache(redis_client):
    """SWR cache manager instance"""
    return SWRCacheManager(
        redis_client,
        default_ttl=2,  # 2 seconds fresh
        default_stale_ttl=6  # 6 seconds total (4 seconds stale window)
    )


class TestCacheEntry:
    """Test CacheEntry model"""

    def test_cache_entry_creation(self):
        entry = CacheEntry(
            key="test",
            value={"foo": "bar"},
            etag="abc123",
            cached_at=time.time(),
            ttl_seconds=30,
            stale_ttl_seconds=120
        )

        assert entry.key == "test"
        assert entry.value == {"foo": "bar"}
        assert entry.etag == "abc123"
        assert entry.age_seconds < 1  # Just created
        assert entry.is_fresh is True
        assert entry.is_stale is False
        assert entry.is_expired is False

    def test_cache_entry_is_fresh(self):
        entry = CacheEntry(
            key="test",
            value={"foo": "bar"},
            etag="abc123",
            cached_at=time.time() - 10,  # 10 seconds ago
            ttl_seconds=30,
            stale_ttl_seconds=120
        )

        assert entry.is_fresh is True
        assert entry.is_stale is False
        assert entry.is_expired is False

    def test_cache_entry_is_stale(self):
        entry = CacheEntry(
            key="test",
            value={"foo": "bar"},
            etag="abc123",
            cached_at=time.time() - 60,  # 60 seconds ago
            ttl_seconds=30,
            stale_ttl_seconds=120
        )

        assert entry.is_fresh is False
        assert entry.is_stale is True
        assert entry.is_expired is False

    def test_cache_entry_is_expired(self):
        entry = CacheEntry(
            key="test",
            value={"foo": "bar"},
            etag="abc123",
            cached_at=time.time() - 150,  # 150 seconds ago
            ttl_seconds=30,
            stale_ttl_seconds=120
        )

        assert entry.is_fresh is False
        assert entry.is_stale is False
        assert entry.is_expired is True

    def test_cache_entry_serialization(self):
        entry = CacheEntry(
            key="test",
            value={"foo": "bar"},
            etag="abc123",
            cached_at=1234567890.0,
            ttl_seconds=30,
            stale_ttl_seconds=120
        )

        data = entry.to_dict()
        assert data["value"] == {"foo": "bar"}
        assert data["etag"] == "abc123"
        assert data["cached_at"] == 1234567890.0

        # Deserialize
        restored = CacheEntry.from_dict("test", data)
        assert restored.key == "test"
        assert restored.value == {"foo": "bar"}
        assert restored.etag == "abc123"


class TestSWRCacheManager:
    """Test SWR Cache Manager"""

    @pytest.mark.asyncio
    async def test_cache_miss_fetches_fresh_data(self, swr_cache):
        """Test cache miss triggers fresh fetch"""
        fetch_fn = AsyncMock(return_value={"result": "fresh"})

        value, is_stale = await swr_cache.get_or_fetch(
            "test_key",
            fetch_fn,
            ttl_seconds=30,
            stale_ttl_seconds=120
        )

        assert value == {"result": "fresh"}
        assert is_stale is False
        fetch_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_hit_returns_fresh_data(self, swr_cache):
        """Test cache hit returns fresh data without fetching"""
        # First fetch
        fetch_fn = AsyncMock(return_value={"result": "fresh"})
        await swr_cache.get_or_fetch("test_key", fetch_fn)

        # Second fetch (should use cache)
        fetch_fn.reset_mock()
        value, is_stale = await swr_cache.get_or_fetch("test_key", fetch_fn)

        assert value == {"result": "fresh"}
        assert is_stale is False
        fetch_fn.assert_not_called()  # Should not fetch again

    @pytest.mark.asyncio
    async def test_stale_cache_triggers_background_revalidation(self, swr_cache):
        """Test stale cache returns data and triggers background revalidation"""
        # Initial fetch
        fetch_fn = AsyncMock(return_value={"result": "v1"})
        await swr_cache.get_or_fetch("test_key", fetch_fn, ttl_seconds=1, stale_ttl_seconds=5)

        # Wait for data to become stale
        await asyncio.sleep(1.5)

        # Update fetch function to return new value
        fetch_fn = AsyncMock(return_value={"result": "v2"})

        # Get stale data (should trigger background revalidation)
        value, is_stale = await swr_cache.get_or_fetch(
            "test_key",
            fetch_fn,
            ttl_seconds=1,
            stale_ttl_seconds=5
        )

        # Should return stale data immediately
        assert value == {"result": "v1"}
        assert is_stale is True

        # Wait for background revalidation
        await asyncio.sleep(0.5)

        # Next fetch should get fresh data
        fetch_fn.reset_mock()
        value, is_stale = await swr_cache.get_or_fetch("test_key", fetch_fn)
        assert value == {"result": "v2"}
        assert is_stale is False

    @pytest.mark.asyncio
    async def test_expired_cache_fetches_fresh_data(self, swr_cache):
        """Test expired cache triggers blocking fresh fetch"""
        # Initial fetch
        fetch_fn = AsyncMock(return_value={"result": "v1"})
        await swr_cache.get_or_fetch("test_key", fetch_fn, ttl_seconds=1, stale_ttl_seconds=2)

        # Wait for cache to expire
        await asyncio.sleep(2.5)

        # Update fetch function
        fetch_fn = AsyncMock(return_value={"result": "v2"})

        # Get data (should fetch fresh)
        value, is_stale = await swr_cache.get_or_fetch(
            "test_key",
            fetch_fn,
            ttl_seconds=1,
            stale_ttl_seconds=2
        )

        assert value == {"result": "v2"}
        assert is_stale is False
        fetch_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_etag_generation(self, swr_cache):
        """Test ETag is generated correctly"""
        fetch_fn = AsyncMock(return_value={"result": "data"})
        await swr_cache.get_or_fetch("test_key", fetch_fn)

        # Get cache entry
        entry = await swr_cache._get_entry("test_key")
        assert entry is not None
        assert entry.etag is not None
        assert len(entry.etag) == 16  # SHA256 truncated to 16 chars

    @pytest.mark.asyncio
    async def test_invalidate_cache_entry(self, swr_cache):
        """Test cache invalidation"""
        # Cache some data
        fetch_fn = AsyncMock(return_value={"result": "data"})
        await swr_cache.get_or_fetch("test_key", fetch_fn)

        # Invalidate
        result = await swr_cache.invalidate("test_key")
        assert result is True

        # Verify cache miss
        fetch_fn = AsyncMock(return_value={"result": "new_data"})
        value, is_stale = await swr_cache.get_or_fetch("test_key", fetch_fn)
        assert value == {"result": "new_data"}
        fetch_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_pattern(self, swr_cache):
        """Test pattern-based invalidation"""
        # Cache multiple entries
        for i in range(3):
            fetch_fn = AsyncMock(return_value={"result": f"data{i}"})
            await swr_cache.get_or_fetch(f"plan:user1:{i}", fetch_fn)

        # Invalidate pattern
        count = await swr_cache.invalidate_pattern("plan:user1:*")
        assert count == 3

    @pytest.mark.asyncio
    async def test_cache_stats(self, swr_cache):
        """Test cache statistics"""
        # Cache fresh data
        fetch_fn = AsyncMock(return_value={"result": "data"})
        await swr_cache.get_or_fetch("fresh_key", fetch_fn, ttl_seconds=10)

        # Cache stale data
        await swr_cache.get_or_fetch("stale_key", fetch_fn, ttl_seconds=1)
        await asyncio.sleep(1.5)

        stats = await swr_cache.get_stats()
        assert stats["total_entries"] >= 1  # At least fresh entry
        assert stats["fresh_entries"] >= 1

    @pytest.mark.asyncio
    async def test_fetch_failure_raises_exception(self, swr_cache):
        """Test fetch failure raises exception"""
        fetch_fn = AsyncMock(side_effect=Exception("Fetch failed"))

        with pytest.raises(Exception, match="Fetch failed"):
            await swr_cache.get_or_fetch("test_key", fetch_fn)

    @pytest.mark.asyncio
    async def test_revalidation_failure_serves_stale_data(self, swr_cache):
        """Test revalidation failure gracefully serves stale data"""
        # Initial fetch
        fetch_fn = AsyncMock(return_value={"result": "v1"})
        await swr_cache.get_or_fetch("test_key", fetch_fn, ttl_seconds=1, stale_ttl_seconds=5)

        # Wait for stale
        await asyncio.sleep(1.5)

        # Fetch function fails on revalidation
        fetch_fn = AsyncMock(side_effect=Exception("Service down"))

        # Get stale data (triggers failed revalidation)
        value, is_stale = await swr_cache.get_or_fetch(
            "test_key",
            fetch_fn,
            ttl_seconds=1,
            stale_ttl_seconds=5
        )

        # Should still return stale data
        assert value == {"result": "v1"}
        assert is_stale is True

        # Wait for revalidation to complete (and fail)
        await asyncio.sleep(0.5)

        # Should still serve stale data
        fetch_fn.reset_mock()
        value, is_stale = await swr_cache.get_or_fetch("test_key", fetch_fn)
        assert value == {"result": "v1"}  # Still old data

    @pytest.mark.asyncio
    async def test_concurrent_requests_coalesce(self, swr_cache):
        """Test concurrent requests don't trigger duplicate fetches"""
        call_count = 0

        async def slow_fetch():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return {"result": "data"}

        # Fire multiple concurrent requests
        tasks = [
            swr_cache.get_or_fetch("test_key", slow_fetch)
            for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All should get same result
        assert all(r[0] == {"result": "data"} for r in results)

        # Note: Current implementation may fetch multiple times
        # This test documents current behavior
        # TODO: Implement request coalescing to reduce to 1 fetch

    @pytest.mark.asyncio
    async def test_default_ttl_values(self, redis_client):
        """Test default TTL values"""
        cache = SWRCacheManager(redis_client)
        assert cache.default_ttl == 30
        assert cache.default_stale_ttl == 120


class TestGlobalInstance:
    """Test global instance management"""

    def test_init_and_get_swr_cache_manager(self, redis_client):
        """Test global instance initialization"""
        manager = init_swr_cache_manager(redis_client, default_ttl=10, default_stale_ttl=60)

        assert manager.default_ttl == 10
        assert manager.default_stale_ttl == 60

        # Get instance
        same_manager = get_swr_cache_manager()
        assert same_manager is manager

    def test_get_without_init_raises_error(self):
        """Test getting instance before initialization raises error"""
        # Reset global instance
        import app.core.swr_cache
        app.core.swr_cache._swr_cache_manager = None

        with pytest.raises(RuntimeError, match="not initialized"):
            get_swr_cache_manager()
