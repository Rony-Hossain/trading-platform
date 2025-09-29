#!/usr/bin/env python3
"""
Historical Data Backfill Demo

This script demonstrates the historical data backfill capabilities
for new symbols in the Event Data Service.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

import httpx


class BackfillDemo:
    """Demo client for historical data backfill capabilities."""
    
    def __init__(self, base_url: str = "http://localhost:8006"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def get_backfill_stats(self) -> Dict[str, Any]:
        """Get backfill service statistics."""
        print("📊 Getting backfill service statistics...")
        
        response = await self.client.get(f"{self.base_url}/backfill/stats")
        
        if response.status_code == 200:
            stats = response.json()
            print("✅ Retrieved backfill statistics")
            print(f"   Service: {stats['service']}")
            print(f"   Enabled: {stats['enabled']}")
            print(f"   Configured Sources: {', '.join(stats['configured_sources'])}")
            print(f"   Total Events: {stats['statistics']['total_events']}")
            print(f"   Backfilled Events: {stats['statistics']['backfilled_events']}")
            print(f"   Recent Backfills (7d): {stats['statistics']['recent_backfills_7d']}")
            print(f"   Active Backfills: {stats['active_backfills']}")
            print(f"   Queue Size: {stats['queue_size']}")
            
            config = stats.get('configuration', {})
            print("\n   Configuration:")
            for key, value in config.items():
                print(f"     {key}: {value}")
            
            return stats
        else:
            print(f"❌ Failed to get backfill stats: {response.status_code}")
            print(response.text)
            return {}
    
    async def request_backfill(
        self, 
        symbol: str = "NVDA",
        start_date: str = None,
        end_date: str = None,
        categories: str = None,
        sources: str = None,
        priority: int = 1
    ) -> Dict[str, Any]:
        """Request backfill for a specific symbol."""
        print(f"\n🔄 Requesting backfill for {symbol}...")
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_dt = datetime.now() - timedelta(days=365)
            start_date = start_dt.strftime("%Y-%m-%d")
        
        params = {
            "priority": priority
        }
        
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if categories:
            params["categories"] = categories
        if sources:
            params["sources"] = sources
        
        response = await self.client.post(
            f"{self.base_url}/backfill/symbols/{symbol}",
            params=params
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Backfill request submitted")
            print(f"   Request ID: {result['request_id']}")
            print(f"   Symbol: {result['symbol']}")
            
            params = result.get('parameters', {})
            print(f"   Date Range: {params.get('start_date')} to {params.get('end_date')}")
            if params.get('categories'):
                print(f"   Categories: {', '.join(params['categories'])}")
            if params.get('sources'):
                print(f"   Sources: {', '.join(params['sources'])}")
            print(f"   Priority: {params.get('priority')}")
            
            return result
        else:
            print(f"❌ Failed to request backfill: {response.status_code}")
            print(response.text)
            return {}
    
    async def check_backfill_status(self, symbol: str) -> Dict[str, Any]:
        """Check the status of a backfill operation."""
        print(f"\n🔍 Checking backfill status for {symbol}...")
        
        response = await self.client.get(f"{self.base_url}/backfill/status/{symbol}")
        
        if response.status_code == 200:
            status = response.json()
            
            if status.get("status") == "not_found":
                print(f"ℹ️  No active backfill found for {symbol}")
            else:
                print(f"✅ Backfill status for {symbol}")
                progress = status.get('progress', {})
                print(f"   Status: {status['status']}")
                print(f"   Progress: {progress.get('completion_percentage', 0):.1f}%")
                print(f"   Requests: {progress.get('completed_requests', 0)}/{progress.get('total_requests', 0)}")
                print(f"   Current Source: {progress.get('current_source', 'N/A')}")
                print(f"   Events Processed: {progress.get('events_processed', 0)}")
                print(f"   Started: {progress.get('started_at', 'N/A')}")
                
                if progress.get('estimated_completion'):
                    print(f"   Estimated Completion: {progress['estimated_completion']}")
            
            return status
        else:
            print(f"❌ Failed to check status: {response.status_code}")
            print(response.text)
            return {}
    
    async def list_active_backfills(self) -> Dict[str, Any]:
        """List all active backfill operations."""
        print(f"\n📋 Listing active backfill operations...")
        
        response = await self.client.get(f"{self.base_url}/backfill/active")
        
        if response.status_code == 200:
            result = response.json()
            active_backfills = result.get('active_backfills', [])
            
            print(f"✅ Found {result.get('count', 0)} active backfill(s)")
            
            for i, backfill in enumerate(active_backfills, 1):
                print(f"\n   Backfill {i}:")
                print(f"     Symbol: {backfill['symbol']}")
                print(f"     Progress: {backfill.get('completion_percentage', 0):.1f}%")
                print(f"     Current Source: {backfill.get('current_source', 'N/A')}")
                print(f"     Events Processed: {backfill.get('events_processed', 0)}")
                print(f"     Started: {backfill.get('started_at', 'N/A')}")
            
            return result
        else:
            print(f"❌ Failed to list active backfills: {response.status_code}")
            print(response.text)
            return {}
    
    async def create_test_event_for_new_symbol(self, symbol: str = "TSLA") -> str:
        """Create a test event to trigger automatic backfill."""
        print(f"\n➕ Creating test event for {symbol} to demonstrate auto-backfill...")
        
        event_data = {
            "symbol": symbol,
            "title": f"{symbol} Q4 2024 Earnings Call",
            "category": "earnings",
            "scheduled_at": (datetime.utcnow() + timedelta(days=7)).isoformat(),
            "timezone": "US/Eastern",
            "description": f"{symbol} will report Q4 2024 financial results",
            "status": "scheduled",
            "source": "demo",
            "external_id": f"demo-{symbol.lower()}-earnings-2024-q4",
            "metadata": {
                "test_event": True,
                "auto_backfill_trigger": True
            }
        }
        
        response = await self.client.post(
            f"{self.base_url}/events",
            json=event_data
        )
        
        if response.status_code == 201:
            event = response.json()
            print(f"✅ Created test event: {event['id']}")
            print(f"   Symbol: {event['symbol']}")
            print(f"   Title: {event['title']}")
            print(f"   This should trigger automatic backfill if {symbol} is a new symbol")
            return event['id']
        else:
            print(f"❌ Failed to create test event: {response.status_code}")
            print(response.text)
            return None
    
    async def check_events_for_symbol(self, symbol: str) -> int:
        """Check how many events exist for a symbol."""
        print(f"\n📊 Checking existing events for {symbol}...")
        
        response = await self.client.get(
            f"{self.base_url}/events",
            params={"symbol": symbol, "limit": 1000}
        )
        
        if response.status_code == 200:
            events = response.json()
            count = len(events)
            print(f"✅ Found {count} existing events for {symbol}")
            
            if count > 0:
                print("   Recent events:")
                for event in events[:5]:  # Show first 5
                    print(f"     - {event['title']} ({event['scheduled_at'][:10]})")
                if count > 5:
                    print(f"     ... and {count - 5} more")
            
            return count
        else:
            print(f"❌ Failed to check events: {response.status_code}")
            return 0
    
    async def demonstrate_backfill_flow(self):
        """Demonstrate the complete backfill workflow."""
        print("🚀 Historical Data Backfill Demo")
        print("=" * 50)
        
        try:
            # 1. Get service statistics
            await self.get_backfill_stats()
            
            # 2. Check if we have active backfills
            await self.list_active_backfills()
            
            # 3. Pick a symbol to demonstrate with
            demo_symbol = "NVDA"
            
            # 4. Check existing events for the symbol
            existing_count = await self.check_events_for_symbol(demo_symbol)
            
            # 5. Request manual backfill
            backfill_result = await self.request_backfill(
                symbol=demo_symbol,
                categories="earnings,split,dividend",
                sources="financial_modeling_prep,polygon",
                priority=1
            )
            
            if backfill_result:
                # 6. Wait a moment for processing to start
                await asyncio.sleep(5)
                
                # 7. Check backfill status
                await self.check_backfill_status(demo_symbol)
                
                # 8. List active backfills again
                await self.list_active_backfills()
            
            # 9. Demonstrate automatic backfill trigger
            test_symbol = "RBLX"  # Pick a symbol that might not have many events
            print(f"\n🧪 Testing automatic backfill trigger with {test_symbol}...")
            
            # Check if symbol has existing events
            existing_test_count = await self.check_events_for_symbol(test_symbol)
            
            if existing_test_count == 0:
                print(f"   {test_symbol} has no events - creating one should trigger auto-backfill")
                await self.create_test_event_for_new_symbol(test_symbol)
                
                # Wait for auto-backfill to be queued
                await asyncio.sleep(3)
                
                # Check if backfill was triggered
                await self.check_backfill_status(test_symbol)
            else:
                print(f"   {test_symbol} already has {existing_test_count} events - auto-backfill won't trigger")
            
            # 10. Final status check
            print(f"\n🏁 Final backfill service status:")
            await self.get_backfill_stats()
            
            print("\n🎉 Demo completed!")
            print("\nKey Features Demonstrated:")
            print("✅ Manual backfill requests with configurable parameters")
            print("✅ Progress tracking and status monitoring")
            print("✅ Multiple data source integration")
            print("✅ Automatic backfill triggers for new symbols")
            print("✅ Service statistics and health monitoring")
            print("✅ Queue management and concurrent processing")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await self.client.aclose()
    
    async def test_specific_source(self, source: str = "financial_modeling_prep"):
        """Test backfill from a specific data source."""
        print(f"\n🔧 Testing backfill from {source}...")
        
        result = await self.request_backfill(
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            sources=source,
            priority=1
        )
        
        if result:
            await asyncio.sleep(5)
            await self.check_backfill_status("AAPL")


async def main():
    """Run the backfill demo."""
    demo = BackfillDemo()
    await demo.demonstrate_backfill_flow()


async def test_source():
    """Test a specific data source."""
    demo = BackfillDemo()
    await demo.test_specific_source("financial_modeling_prep")
    await demo.client.aclose()


if __name__ == "__main__":
    print("Historical Data Backfill Demo")
    print("Make sure the Event Data Service is running: http://localhost:8006")
    print("Configure API keys in .env for data sources:")
    print("- FMP_API_KEY (Financial Modeling Prep)")
    print("- ALPHA_VANTAGE_API_KEY (Alpha Vantage)")
    print("- POLYGON_API_KEY (Polygon.io)")
    print("- FINNHUB_API_KEY (Finnhub)")
    print()
    
    # Run the main demo
    asyncio.run(main())
    
    # Uncomment to test specific source
    # asyncio.run(test_source())