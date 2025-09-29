#!/usr/bin/env python3
"""
Event Sentiment Analysis Integration Demo

This script demonstrates the event sentiment analysis capabilities
integrated into the Event Data Service.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

import httpx


class EventSentimentDemo:
    """Demo client for event sentiment analysis integration."""
    
    def __init__(self, base_url: str = "http://localhost:8006"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def create_sample_event(self) -> str:
        """Create a sample event for sentiment analysis."""
        event_data = {
            "symbol": "AAPL",
            "title": "Apple Inc. Q4 2024 Earnings Call",
            "category": "earnings",
            "scheduled_at": (datetime.utcnow() + timedelta(days=1)).isoformat(),
            "timezone": "US/Eastern",
            "description": "Apple will report Q4 2024 financial results and host earnings call with analysts",
            "status": "scheduled",
            "source": "demo",
            "external_id": "demo-earnings-2024-q4",
            "metadata": {
                "importance": "high",
                "market_cap": 3000000000000,  # $3T
                "sector": "technology"
            }
        }
        
        response = await self.client.post(
            f"{self.base_url}/events",
            json=event_data
        )
        
        if response.status_code == 201:
            event = response.json()
            print(f"✅ Created sample event: {event['id']}")
            print(f"   Symbol: {event['symbol']}")
            print(f"   Title: {event['title']}")
            print(f"   Scheduled: {event['scheduled_at']}")
            return event['id']
        else:
            print(f"❌ Failed to create event: {response.status_code}")
            print(response.text)
            return None
    
    async def analyze_event_sentiment(self, event_id: str) -> Dict[str, Any]:
        """Analyze sentiment for a specific event."""
        print(f"\n🔍 Analyzing sentiment for event {event_id}...")
        
        response = await self.client.get(
            f"{self.base_url}/sentiment/events/{event_id}",
            params={"force_refresh": True}
        )
        
        if response.status_code == 200:
            analysis = response.json()
            print("✅ Sentiment analysis completed")
            print(f"   Overall Sentiment: {analysis['overall_sentiment']['label']} ({analysis['overall_sentiment']['compound']:.3f})")
            print(f"   Confidence: {analysis['overall_sentiment']['confidence']:.3f}")
            print(f"   Sentiment Momentum: {analysis['sentiment_momentum']:.3f}")
            print(f"   Prediction: {analysis['outcome_prediction']} ({analysis['prediction_confidence']:.3f})")
            
            # Show timeframe breakdown
            if analysis.get('timeframes'):
                print("\n   Timeframe Analysis:")
                for timeframe, sentiment in analysis['timeframes'].items():
                    print(f"     {timeframe}: {sentiment['label']} ({sentiment['compound']:.3f})")
            
            # Show source breakdown
            if analysis.get('sources'):
                print("\n   Source Analysis:")
                for source, sentiment in analysis['sources'].items():
                    print(f"     {source}: {sentiment['label']} ({sentiment['compound']:.3f}) - {sentiment['volume']} posts")
            
            return analysis
        else:
            print(f"❌ Failed to analyze sentiment: {response.status_code}")
            if response.status_code == 503:
                print("   Sentiment service may not be running")
            print(response.text)
            return {}
    
    async def analyze_outcome_sentiment(self, event_id: str) -> Dict[str, Any]:
        """Analyze outcome-specific sentiment for an event."""
        print(f"\n📈 Analyzing outcome sentiment for event {event_id}...")
        
        response = await self.client.get(
            f"{self.base_url}/sentiment/events/{event_id}/outcome"
        )
        
        if response.status_code == 200:
            outcome = response.json()
            if 'outcome_sentiment' in outcome:
                sentiment = outcome['outcome_sentiment']
                print("✅ Outcome sentiment analysis completed")
                print(f"   Outcome Sentiment: {sentiment['label']} ({sentiment['compound']:.3f})")
                print(f"   Confidence: {sentiment['confidence']:.3f}")
                print(f"   Source: {sentiment['source']}")
                print(f"   Volume: {sentiment['volume']} data points")
                
                if sentiment.get('metadata'):
                    print(f"   Metadata: {json.dumps(sentiment['metadata'], indent=2)}")
            else:
                print("ℹ️  No outcome sentiment data available yet")
            
            return outcome
        else:
            print(f"❌ Failed to analyze outcome sentiment: {response.status_code}")
            print(response.text)
            return {}
    
    async def get_sentiment_trends(self, symbol: str = "AAPL", days: int = 7) -> Dict[str, Any]:
        """Get sentiment trends for a symbol."""
        print(f"\n📊 Getting sentiment trends for {symbol} over {days} days...")
        
        response = await self.client.get(
            f"{self.base_url}/sentiment/trends/{symbol}",
            params={"days": days}
        )
        
        if response.status_code == 200:
            trends = response.json()
            print(f"✅ Retrieved sentiment trends for {trends['symbol']}")
            print(f"   Timeframe: {trends['timeframe_days']} days")
            
            if trends.get('trends'):
                print("   Trend data available")
                # Note: Actual trend data structure depends on sentiment service response
            else:
                print("   No trend data available")
            
            return trends
        else:
            print(f"❌ Failed to get sentiment trends: {response.status_code}")
            print(response.text)
            return {}
    
    async def get_sentiment_stats(self) -> Dict[str, Any]:
        """Get sentiment analysis statistics."""
        print(f"\n📈 Getting sentiment analysis statistics...")
        
        response = await self.client.get(f"{self.base_url}/sentiment/stats")
        
        if response.status_code == 200:
            stats = response.json()
            print("✅ Retrieved sentiment analysis stats")
            print(f"   Service: {stats['service']}")
            print(f"   Enabled: {stats['enabled']}")
            print(f"   Sentiment Service URL: {stats['sentiment_service_url']}")
            print(f"   Cached Analyses: {stats['cache_stats']['cached_analyses']}")
            print(f"   Cache TTL: {stats['cache_stats']['cache_ttl_seconds']}s")
            
            config = stats.get('configuration', {})
            print("\n   Configuration:")
            for key, value in config.items():
                print(f"     {key}: {value}")
            
            return stats
        else:
            print(f"❌ Failed to get sentiment stats: {response.status_code}")
            print(response.text)
            return {}
    
    async def update_event_for_sentiment_test(self, event_id: str):
        """Update an event to trigger sentiment re-analysis."""
        print(f"\n🔄 Updating event {event_id} to trigger sentiment re-analysis...")
        
        update_data = {
            "description": "Updated: Apple will report Q4 2024 financial results with expected strong iPhone sales and services growth"
        }
        
        response = await self.client.patch(
            f"{self.base_url}/events/{event_id}",
            json=update_data
        )
        
        if response.status_code == 200:
            event = response.json()
            print("✅ Event updated successfully")
            print(f"   Description updated: {event['description'][:100]}...")
            return True
        else:
            print(f"❌ Failed to update event: {response.status_code}")
            print(response.text)
            return False
    
    async def demonstrate_sentiment_integration(self):
        """Run a complete demonstration of sentiment integration."""
        print("🚀 Event Sentiment Analysis Integration Demo")
        print("=" * 50)
        
        try:
            # 1. Get sentiment stats to verify service is available
            await self.get_sentiment_stats()
            
            # 2. Create a sample event
            event_id = await self.create_sample_event()
            if not event_id:
                return
            
            # Small delay to let background processing complete
            await asyncio.sleep(2)
            
            # 3. Analyze event sentiment
            sentiment_analysis = await self.analyze_event_sentiment(event_id)
            
            # 4. Get sentiment trends for the symbol
            await self.get_sentiment_trends("AAPL", 7)
            
            # 5. Analyze outcome sentiment (may not have data yet)
            await self.analyze_outcome_sentiment(event_id)
            
            # 6. Update event and re-analyze
            if await self.update_event_for_sentiment_test(event_id):
                await asyncio.sleep(2)  # Let processing complete
                await self.analyze_event_sentiment(event_id)
            
            print("\n🎉 Demo completed successfully!")
            print(f"Event ID: {event_id}")
            print("You can continue to monitor sentiment analysis by calling the API endpoints.")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await self.client.aclose()


async def main():
    """Run the sentiment analysis demo."""
    demo = EventSentimentDemo()
    await demo.demonstrate_sentiment_integration()


if __name__ == "__main__":
    print("Event Sentiment Analysis Integration Demo")
    print("Make sure both Event Data Service and Sentiment Service are running")
    print("Event Data Service: http://localhost:8006")
    print("Sentiment Service: http://localhost:8007")
    print()
    
    asyncio.run(main())