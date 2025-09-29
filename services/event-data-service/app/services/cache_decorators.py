"""Cache Decorators for Event Data Service

Provides convenient decorators to add caching to database operations
and API endpoints with minimal code changes.
"""

import asyncio
import functools
import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Callable, Optional, Dict, List, Union

from .event_cache import EventCacheService, CacheKeyType

logger = logging.getLogger(__name__)


def cached_event(
    cache_service: EventCacheService,
    ttl: Optional[int] = None,
    key_builder: Optional[Callable] = None
):
    """
    Decorator for caching individual event data.
    
    Args:
        cache_service: EventCacheService instance
        ttl: Optional custom TTL in seconds
        key_builder: Optional function to build custom cache key
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default: use event_id from args/kwargs
                event_id = None
                if args and len(args) > 1:
                    event_id = args[1]  # Assuming session is first arg
                elif 'event_id' in kwargs:
                    event_id = kwargs['event_id']
                
                if not event_id:
                    # If no event_id, skip caching
                    return await func(*args, **kwargs)
                
                cache_key = str(event_id)
            
            # Try to get from cache
            cached_result = await cache_service.get(CacheKeyType.EVENT, cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            if result is not None:
                await cache_service.set(CacheKeyType.EVENT, cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def cached_event_list(
    cache_service: EventCacheService,
    ttl: Optional[int] = None,
    include_params: Optional[List[str]] = None,
    exclude_params: Optional[List[str]] = None
):
    """
    Decorator for caching event list queries.
    
    Args:
        cache_service: EventCacheService instance
        ttl: Optional custom TTL in seconds
        include_params: List of parameter names to include in cache key
        exclude_params: List of parameter names to exclude from cache key
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Build cache key from function parameters
            cache_params = {}
            
            # Include specified parameters
            if include_params:
                for param in include_params:
                    if param in kwargs:
                        cache_params[param] = kwargs[param]
            else:
                # Include all kwargs except excluded ones
                cache_params = {k: v for k, v in kwargs.items() 
                              if not exclude_params or k not in exclude_params}
            
            # Remove session from cache key
            cache_params.pop('session', None)
            
            # Create deterministic cache key
            cache_key_data = json.dumps(cache_params, sort_keys=True, default=str)
            cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
            
            # Try to get from cache
            cached_result = await cache_service.get(
                CacheKeyType.EVENT_LIST, 
                cache_key,
                **cache_params
            )
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            if result is not None:
                await cache_service.set(
                    CacheKeyType.EVENT_LIST, 
                    cache_key, 
                    result, 
                    ttl,
                    **cache_params
                )
            
            return result
        
        return wrapper
    return decorator


def cached_search_result(
    cache_service: EventCacheService,
    ttl: Optional[int] = None
):
    """
    Decorator for caching search results.
    
    Args:
        cache_service: EventCacheService instance
        ttl: Optional custom TTL in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Build cache key from search parameters
            search_params = {k: v for k, v in kwargs.items() 
                           if k not in ['session'] and v is not None}
            
            # Create deterministic cache key
            cache_key_data = json.dumps(search_params, sort_keys=True, default=str)
            cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
            
            # Try to get from cache
            cached_result = await cache_service.get(
                CacheKeyType.SEARCH_RESULT,
                cache_key,
                **search_params
            )
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            if result is not None:
                await cache_service.set(
                    CacheKeyType.SEARCH_RESULT,
                    cache_key,
                    result,
                    ttl,
                    **search_params
                )
            
            return result
        
        return wrapper
    return decorator


def cached_aggregation(
    cache_service: EventCacheService,
    ttl: Optional[int] = None,
    cache_key_prefix: str = "default"
):
    """
    Decorator for caching aggregation results.
    
    Args:
        cache_service: EventCacheService instance
        ttl: Optional custom TTL in seconds
        cache_key_prefix: Prefix for cache key to namespace different aggregations
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Build cache key
            agg_params = {k: v for k, v in kwargs.items() 
                         if k not in ['session'] and v is not None}
            
            cache_key_data = json.dumps(agg_params, sort_keys=True, default=str)
            cache_key = f"{cache_key_prefix}:{hashlib.md5(cache_key_data.encode()).hexdigest()}"
            
            # Try to get from cache
            cached_result = await cache_service.get(
                CacheKeyType.AGGREGATION,
                cache_key,
                **agg_params
            )
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            if result is not None:
                await cache_service.set(
                    CacheKeyType.AGGREGATION,
                    cache_key,
                    result,
                    ttl,
                    **agg_params
                )
            
            return result
        
        return wrapper
    return decorator


def cache_invalidate_on_write(
    cache_service: EventCacheService,
    invalidation_patterns: List[str] = None
):
    """
    Decorator that invalidates cache entries after write operations.
    
    Args:
        cache_service: EventCacheService instance
        invalidation_patterns: List of cache patterns to invalidate
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Execute the function first
            result = await func(*args, **kwargs)
            
            # Invalidate cache entries
            if invalidation_patterns:
                for pattern in invalidation_patterns:
                    await cache_service.delete_pattern(pattern)
            else:
                # Default invalidation based on function parameters
                if 'event_id' in kwargs:
                    await cache_service.invalidate_event(kwargs['event_id'])
                elif 'symbol' in kwargs:
                    await cache_service.invalidate_symbol(kwargs['symbol'])
                
                # Invalidate common list and search caches
                await cache_service.delete_pattern("event_list:*")
                await cache_service.delete_pattern("search:*")
            
            return result
        
        return wrapper
    return decorator


class CacheManager:
    """
    Context manager for cache operations with automatic invalidation.
    """
    
    def __init__(self, cache_service: EventCacheService):
        self.cache_service = cache_service
        self.invalidation_queue = []
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Only invalidate cache if no exception occurred
        if exc_type is None:
            for invalidation in self.invalidation_queue:
                if invalidation['type'] == 'event':
                    await self.cache_service.invalidate_event(
                        invalidation['event_id'],
                        invalidation.get('symbol')
                    )
                elif invalidation['type'] == 'symbol':
                    await self.cache_service.invalidate_symbol(invalidation['symbol'])
                elif invalidation['type'] == 'pattern':
                    await self.cache_service.delete_pattern(invalidation['pattern'])
    
    def invalidate_event(self, event_id: str, symbol: str = None):
        """Queue event invalidation."""
        self.invalidation_queue.append({
            'type': 'event',
            'event_id': event_id,
            'symbol': symbol
        })
    
    def invalidate_symbol(self, symbol: str):
        """Queue symbol invalidation."""
        self.invalidation_queue.append({
            'type': 'symbol',
            'symbol': symbol
        })
    
    def invalidate_pattern(self, pattern: str):
        """Queue pattern invalidation."""
        self.invalidation_queue.append({
            'type': 'pattern',
            'pattern': pattern
        })


def with_cache_invalidation(cache_service: EventCacheService):
    """
    Decorator that provides a CacheManager context to the decorated function.
    
    Usage:
        @with_cache_invalidation(cache_service)
        async def update_event(cache_manager, event_id, data):
            # Update event in database
            result = await db_update(event_id, data)
            
            # Queue cache invalidation
            cache_manager.invalidate_event(event_id)
            
            return result
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            async with CacheManager(cache_service) as cache_manager:
                return await func(cache_manager, *args, **kwargs)
        
        return wrapper
    return decorator