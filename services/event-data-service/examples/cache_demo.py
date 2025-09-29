#!/usr/bin/env python3
"""
Redis Cache Demo for Event Data Service

This script demonstrates the caching capabilities of the Event Data Service,
showing performance improvements and cache management features.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
import httpx


class EventCacheDemo:
    """Demonstrates Event Data Service caching capabilities."""
    
    def __init__(self, base_url: str = "http://localhost:8006"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def health_check(self):
        """Check if the Event Data Service is running with cache enabled."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            health = response.json()
            
            print("=== Event Data Service Health ===")
            print(f"Service: {health.get('service')}")
            print(f"Status: {health.get('status')}")
            print(f"Event Count: {health.get('event_count', 0)}")
            print(f"Headline Count: {health.get('headline_count', 0)}")
            
            cache_info = health.get('cache', {})
            print("\n=== Cache Status ===")
            print(f"Enabled: {cache_info.get('enabled', False)}")
            print(f"Status: {cache_info.get('status', 'unknown')}")
            print(f"Total Keys: {cache_info.get('total_keys', 0)}")
            print(f"Hit Rate: {cache_info.get('hit_rate', 0):.2%}")
            print(f"Memory Usage: {cache_info.get('memory_usage_mb', 0):.2f} MB")
            
            return cache_info.get('enabled', False)
            
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    async def create_sample_events(self, count: int = 10):
        """Create sample events for testing."""
        print(f"\n=== Creating {count} Sample Events ===")
        
        sample_events = []
        base_time = datetime.utcnow()
        
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA", "AMD", "INTC"]
        categories = ["earnings", "product_launch", "analyst_day", "regulatory", "m&a"]
        
        for i in range(count):
            event_data = {
                "symbol": symbols[i % len(symbols)],
                "title": f"Sample Event {i+1} for {symbols[i % len(symbols)]}",
                "category": categories[i % len(categories)],
                "scheduled_at": (base_time + timedelta(days=i)).isoformat(),
                "description": f"This is a sample event created for cache testing purposes. Event #{i+1}",
                "status": "scheduled",
                "metadata": {
                    "demo": True,
                    "created_for": "cache_testing",
                    "event_number": i + 1
                }
            }
            
            try:
                response = await self.client.post(f"{self.base_url}/events", json=event_data)
                response.raise_for_status()
                event = response.json()
                sample_events.append(event)
                print(f"  Created event {event['id']} for {event['symbol']}")
                
            except Exception as e:
                print(f"  Failed to create event {i+1}: {e}")
        
        print(f"Successfully created {len(sample_events)} events")
        return sample_events
    
    async def demonstrate_single_event_caching(self, event_id: str):
        """Demonstrate caching for single event retrieval."""
        print(f"\n=== Single Event Caching Demo ===")
        print(f"Testing event ID: {event_id}")
        
        # Clear cache for this event first
        await self.client.post(f"{self.base_url}/cache/invalidate/event/{event_id}")
        
        # First request (cache miss)
        print("\n1. First request (should be cache miss):")
        start_time = time.time()
        response = await self.client.get(f"{self.base_url}/events/{event_id}")
        response.raise_for_status()
        first_duration = time.time() - start_time
        print(f"   Duration: {first_duration*1000:.2f}ms")
        
        # Second request (cache hit)
        print("\n2. Second request (should be cache hit):")
        start_time = time.time()
        response = await self.client.get(f"{self.base_url}/events/{event_id}")
        response.raise_for_status()
        second_duration = time.time() - start_time
        print(f"   Duration: {second_duration*1000:.2f}ms")
        
        # Calculate improvement
        if first_duration > 0:
            improvement = ((first_duration - second_duration) / first_duration) * 100
            print(f"   Performance improvement: {improvement:.1f}%")
        
        return response.json()
    
    async def demonstrate_list_caching(self):
        """Demonstrate caching for event list queries."""
        print(f"\n=== Event List Caching Demo ===")
        
        # Clear list caches
        await self.client.post(f"{self.base_url}/cache/invalidate/pattern", params={"pattern": "event_list:*"})
        
        # Test different list queries
        queries = [
            {"symbol": "AAPL"},
            {"category": "earnings"},
            {"status": "scheduled"},
            {}  # No filters
        ]
        
        for i, params in enumerate(queries, 1):
            print(f"\n{i}. Testing query: {params or 'no filters'}")
            
            # First request (cache miss)
            start_time = time.time()
            response = await self.client.get(f"{self.base_url}/events", params=params)
            response.raise_for_status()
            first_duration = time.time() - start_time
            events = response.json()
            print(f"   First request: {first_duration*1000:.2f}ms ({len(events)} events)")
            
            # Second request (cache hit)
            start_time = time.time()
            response = await self.client.get(f"{self.base_url}/events", params=params)
            response.raise_for_status()
            second_duration = time.time() - start_time
            print(f"   Second request: {second_duration*1000:.2f}ms (cached)")
            
            if first_duration > 0:
                improvement = ((first_duration - second_duration) / first_duration) * 100
                print(f"   Performance improvement: {improvement:.1f}%")
    
    async def demonstrate_cache_invalidation(self, event_id: str, symbol: str):
        """Demonstrate cache invalidation scenarios."""
        print(f"\n=== Cache Invalidation Demo ===")
        
        # Cache some data first
        await self.client.get(f"{self.base_url}/events/{event_id}")
        await self.client.get(f"{self.base_url}/events", params={"symbol": symbol})
        
        print("1. Cached event and list data")
        stats_before = await self.get_cache_stats()
        print(f"   Cache keys before: {stats_before.get('total_keys', 0)}")
        
        # Update the event (should trigger invalidation)
        print(f"\n2. Updating event {event_id}...")
        update_data = {
            "description": f"Updated at {datetime.utcnow().isoformat()} for cache demo"
        }
        response = await self.client.patch(f"{self.base_url}/events/{event_id}", json=update_data)
        response.raise_for_status()
        print("   Event updated successfully")
        
        # Check cache stats after update
        stats_after = await self.get_cache_stats()
        print(f"   Cache keys after: {stats_after.get('total_keys', 0)}")
        
        # Test manual invalidation
        print(f"\n3. Manual cache invalidation for symbol {symbol}...")
        response = await self.client.post(f"{self.base_url}/cache/invalidate/symbol/{symbol}")
        response.raise_for_status()
        invalidation_result = response.json()
        print(f"   {invalidation_result.get('message', 'Invalidation completed')}")
    
    async def get_cache_stats(self):
        """Get current cache statistics."""
        try:
            response = await self.client.get(f"{self.base_url}/cache/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Failed to get cache stats: {e}")
            return {}
    
    async def show_detailed_cache_stats(self):
        """Display detailed cache statistics."""
        print(f"\n=== Detailed Cache Statistics ===")
        
        stats = await self.get_cache_stats()
        
        if not stats.get("enabled"):
            print("Cache is disabled")
            return
        
        daily_stats = stats.get("daily_stats", {})
        redis_stats = stats.get("redis_stats", {})
        config = stats.get("configuration", {})
        
        print(f"Connection Status: {stats.get('connection_status', 'unknown')}")
        print(f"Total Cache Keys: {stats.get('total_keys', 0)}")
        print(f"Memory Usage: {stats.get('memory_usage_bytes', 0) / 1024 / 1024:.2f} MB")
        
        print(f"\nDaily Performance:")
        print(f"  Cache Hits: {daily_stats.get('hits', 0)}")
        print(f"  Cache Misses: {daily_stats.get('misses', 0)}")
        print(f"  Cache Sets: {daily_stats.get('sets', 0)}")
        print(f"  Cache Deletes: {daily_stats.get('deletes', 0)}")
        print(f"  Hit Rate: {daily_stats.get('hit_rate', 0):.2%}")
        
        print(f"\nRedis Statistics:")
        print(f"  Total Commands: {redis_stats.get('total_commands_processed', 0)}")
        print(f"  Keyspace Hits: {redis_stats.get('keyspace_hits', 0)}")
        print(f"  Keyspace Misses: {redis_stats.get('keyspace_misses', 0)}")
        print(f"  Connected Clients: {redis_stats.get('connected_clients', 0)}")
        
        print(f"\nConfiguration:")
        print(f"  Default TTL: {config.get('default_ttl', 0)} seconds")
        print(f"  Max Connections: {config.get('max_connections', 0)}")
        print(f"  Compression Threshold: {config.get('compression_threshold', 0)} bytes")
        print(f"  Timeout: {config.get('timeout', 0)} seconds")
    
    async def cleanup_demo_data(self):
        """Clean up demo events created during testing."""
        print(f"\n=== Cleaning Up Demo Data ===")
        
        try:
            # Get all events with demo metadata
            response = await self.client.get(f"{self.base_url}/events")
            response.raise_for_status()
            events = response.json()
            
            demo_events = [
                event for event in events 
                if event.get('metadata', {}).get('demo') is True
            ]
            
            print(f"Found {len(demo_events)} demo events to clean up")
            
            for event in demo_events:
                try:
                    response = await self.client.delete(f"{self.base_url}/events/{event['id']}")
                    if response.status_code in [204, 404]:  # Success or already deleted
                        print(f"  Deleted event {event['id']} ({event['symbol']})")
                    else:
                        print(f"  Failed to delete event {event['id']}: {response.status_code}")
                except Exception as e:
                    print(f"  Error deleting event {event['id']}: {e}")
            
            # Clear any remaining cache
            await self.client.delete(f"{self.base_url}/cache/clear")
            print("Cleared all cache data")
            
        except Exception as e:
            print(f"Cleanup failed: {e}")


async def main():
    """Run the complete cache demonstration."""
    print("🚀 Event Data Service - Redis Cache Demo")
    print("=" * 50)
    
    async with EventCacheDemo() as demo:
        # Check service health
        cache_enabled = await demo.health_check()
        
        if not cache_enabled:
            print("\n❌ Cache is not enabled. Please check your Redis configuration.")
            return
        
        print("\n✅ Cache is enabled and ready for demo!")
        
        # Create sample data
        sample_events = await demo.create_sample_events(5)
        
        if not sample_events:
            print("\n❌ No sample events created. Cannot proceed with demo.")
            return
        
        # Demonstrate single event caching
        first_event = sample_events[0]
        await demo.demonstrate_single_event_caching(first_event['id'])
        
        # Demonstrate list caching
        await demo.demonstrate_list_caching()
        
        # Demonstrate cache invalidation
        await demo.demonstrate_cache_invalidation(first_event['id'], first_event['symbol'])
        
        # Show detailed cache statistics
        await demo.show_detailed_cache_stats()
        
        # Ask if user wants to clean up
        print(f"\n=== Demo Complete ===")
        print("The cache demo has completed successfully!")
        print("\nKey benefits demonstrated:")
        print("  • Significant performance improvements for repeated queries")
        print("  • Automatic cache invalidation on data updates")
        print("  • Flexible cache management via API endpoints")
        print("  • Comprehensive statistics and monitoring")
        
        try:
            cleanup = input("\nClean up demo data? (y/N): ").strip().lower()
            if cleanup in ['y', 'yes']:
                await demo.cleanup_demo_data()
                print("✅ Demo data cleaned up successfully!")
            else:
                print("Demo data preserved for further testing.")
        except KeyboardInterrupt:
            print("\nDemo data preserved.")


if __name__ == "__main__":
    asyncio.run(main())