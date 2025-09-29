#!/usr/bin/env python3
"""
Event Streaming Demo for Event Data Service

This script demonstrates the real-time event streaming capabilities of the Event Data Service,
showing WebSocket and Server-Sent Events connections with real-time event processing.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import websockets
import httpx
import aiohttp


class EventStreamingDemo:
    """Demonstrates Event Data Service real-time streaming capabilities."""
    
    def __init__(self, base_url: str = "http://localhost:8006"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")
        self.http_client = httpx.AsyncClient()
        self.websocket = None
        self.sse_session = None
        self.received_events = []
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
        if self.websocket:
            await self.websocket.close()
        if self.sse_session:
            await self.sse_session.close()
    
    async def health_check(self):
        """Check if the Event Data Service is running with streaming enabled."""
        try:
            response = await self.http_client.get(f"{self.base_url}/health")
            response.raise_for_status()
            health = response.json()
            
            print("=== Event Data Service Health ===")
            print(f"Service: {health.get('service')}")
            print(f"Status: {health.get('status')}")
            
            # Check streaming health
            response = await self.http_client.get(f"{self.base_url}/stream/health")
            response.raise_for_status()
            streaming_health = response.json()
            
            print("\n=== Streaming Service Status ===")
            print(f"Status: {streaming_health.get('status')}")
            print(f"Enabled: {streaming_health.get('enabled', False)}")
            print(f"Backends: {', '.join(streaming_health.get('backends', []))}")
            
            active_connections = streaming_health.get('active_connections', {})
            print(f"WebSocket Connections: {active_connections.get('websockets', 0)}")
            print(f"SSE Connections: {active_connections.get('sse', 0)}")
            
            message_stats = streaming_health.get('message_stats', {})
            print(f"Messages Sent: {message_stats.get('sent', 0)}")
            print(f"Messages Received: {message_stats.get('received', 0)}")
            print(f"Messages Failed: {message_stats.get('failed', 0)}")
            
            return streaming_health.get('enabled', False)
            
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    async def test_websocket_streaming(self):
        """Demonstrate WebSocket streaming functionality."""
        print(f"\n=== WebSocket Streaming Demo ===")
        
        try:
            # Connect to WebSocket
            websocket_url = f"{self.ws_url}/stream/ws"
            print(f"Connecting to: {websocket_url}")
            
            self.websocket = await websockets.connect(websocket_url)
            print("WebSocket connection established")
            
            # Send subscription message
            subscription_message = {
                "type": "subscribe",
                "topics": ["event.created", "event.updated", "event.deleted"],
                "filters": {
                    "symbols": ["AAPL", "MSFT", "TEST"],
                    "min_priority": 1
                }
            }
            
            await self.websocket.send(json.dumps(subscription_message))
            print("Sent subscription request")
            
            # Listen for initial response
            response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            message = json.loads(response)
            print(f"Received subscription confirmation: {message.get('type')}")
            
            # Start background listener
            listener_task = asyncio.create_task(self._websocket_listener())
            
            # Send some test events
            await self._send_test_events()
            
            # Wait for events to be received
            await asyncio.sleep(2)
            
            # Send ping test
            ping_message = {"type": "ping"}
            await self.websocket.send(json.dumps(ping_message))
            
            # Wait a bit more for all responses
            await asyncio.sleep(2)
            
            # Cancel listener and close connection
            listener_task.cancel()
            
            print(f"WebSocket demo completed. Received {len(self.received_events)} events")
            
        except Exception as e:
            print(f"WebSocket streaming failed: {e}")
    
    async def _websocket_listener(self):
        """Background task to listen for WebSocket messages."""
        try:
            while True:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                print(f"WebSocket received: {data.get('type')} - {data.get('message', data.get('data', {}).get('title', 'Unknown'))}")
                
                if data.get('type') not in ['connection.established', 'subscribed', 'pong']:
                    self.received_events.append(data)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"WebSocket listener error: {e}")
    
    async def test_sse_streaming(self):
        """Demonstrate Server-Sent Events streaming functionality."""
        print(f"\n=== Server-Sent Events Demo ===")
        
        try:
            # Prepare SSE connection parameters
            params = {
                "topics": "event.created,event.updated",
                "symbols": "AAPL,MSFT,TEST",
                "min_priority": 1
            }
            
            sse_url = f"{self.base_url}/stream/sse"
            print(f"Connecting to SSE: {sse_url}")
            print(f"Filters: {params}")
            
            # Connect to SSE endpoint
            self.sse_session = aiohttp.ClientSession()
            
            async with self.sse_session.get(sse_url, params=params) as response:
                print(f"SSE connection established (status: {response.status})")
                
                # Start background listener
                listener_task = asyncio.create_task(self._sse_listener(response))
                
                # Send some test events
                await self._send_test_events()
                
                # Wait for events to be received
                await asyncio.sleep(3)
                
                # Cancel listener
                listener_task.cancel()
                
                print(f"SSE demo completed. Received events through SSE")
                
        except Exception as e:
            print(f"SSE streaming failed: {e}")
    
    async def _sse_listener(self, response):
        """Background task to listen for SSE messages."""
        try:
            event_buffer = ""
            
            async for line in response.content:
                line_text = line.decode('utf-8').strip()
                
                if line_text.startswith('data: '):
                    event_data = line_text[6:]  # Remove 'data: ' prefix
                    
                    try:
                        data = json.loads(event_data)
                        event_type = data.get('type', 'unknown')
                        
                        if event_type == 'heartbeat':
                            print("SSE heartbeat received")
                        elif event_type == 'connection.established':
                            print("SSE connection established")
                        else:
                            event_title = data.get('data', {}).get('title', 'Unknown Event')
                            print(f"SSE received: {event_type} - {event_title}")
                    
                    except json.JSONDecodeError:
                        print(f"SSE received invalid JSON: {event_data[:100]}")
                
                elif line_text.startswith('event: '):
                    event_type = line_text[7:]  # Remove 'event: ' prefix
                    if event_type not in ['heartbeat', 'connection']:
                        print(f"SSE event type: {event_type}")
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"SSE listener error: {e}")
    
    async def _send_test_events(self):
        """Send test events to trigger streaming notifications."""
        print("Sending test events...")
        
        test_events = [
            {
                "symbol": "AAPL",
                "title": "Apple Q4 Earnings Streaming Test",
                "category": "earnings",
                "scheduled_at": datetime.utcnow().isoformat(),
                "description": "Test event for streaming demo",
                "metadata": {"streaming_test": True}
            },
            {
                "symbol": "MSFT",
                "title": "Microsoft Product Launch Streaming Test",
                "category": "product_launch",
                "scheduled_at": datetime.utcnow().isoformat(),
                "description": "Another test event for streaming demo",
                "metadata": {"streaming_test": True}
            },
            {
                "symbol": "TEST",
                "title": "Generic Test Event for Streaming",
                "category": "other",
                "scheduled_at": datetime.utcnow().isoformat(),
                "description": "Generic test event",
                "metadata": {"streaming_test": True}
            }
        ]
        
        created_event_ids = []
        
        # Create events
        for i, event_data in enumerate(test_events):
            try:
                response = await self.http_client.post(f"{self.base_url}/events", json=event_data)
                if response.status_code in [200, 201]:
                    event = response.json()
                    created_event_ids.append(event["id"])
                    print(f"  Created test event {i+1}: {event['symbol']}")
                    await asyncio.sleep(0.5)  # Small delay between events
                else:
                    print(f"  Failed to create test event {i+1}: {response.status_code}")
            except Exception as e:
                print(f"  Error creating test event {i+1}: {e}")
        
        # Update an event to trigger update streams
        if created_event_ids:
            try:
                update_data = {
                    "description": f"Updated description for streaming test at {datetime.utcnow().isoformat()}"
                }
                response = await self.http_client.patch(
                    f"{self.base_url}/events/{created_event_ids[0]}", 
                    json=update_data
                )
                if response.status_code == 200:
                    print(f"  Updated test event: {created_event_ids[0]}")
            except Exception as e:
                print(f"  Error updating test event: {e}")
        
        return created_event_ids
    
    async def test_streaming_api(self):
        """Test the streaming API endpoints."""
        print(f"\n=== Streaming API Tests ===")
        
        # Test streaming statistics
        try:
            response = await self.http_client.get(f"{self.base_url}/stream/stats")
            response.raise_for_status()
            stats = response.json()
            
            print("Streaming Statistics:")
            print(f"  Total Real-time Connections: {stats.get('total_real_time_connections', 0)}")
            
            streaming_service = stats.get('streaming_service', {})
            print(f"  Streaming Enabled: {streaming_service.get('enabled', False)}")
            print(f"  Active Backends: {', '.join(streaming_service.get('backends', []))}")
            
            metrics = streaming_service.get('metrics', {})
            print(f"  Messages Sent: {metrics.get('messages_sent', 0)}")
            print(f"  Throughput: {metrics.get('throughput_per_second', 0):.1f} msg/sec")
            
        except Exception as e:
            print(f"Failed to get streaming stats: {e}")
        
        # Test streaming test endpoint
        try:
            print("\nTesting streaming test endpoint...")
            
            test_params = {
                "event_type": "event.created",
                "symbol": "STREAM_TEST",
                "message": "API streaming test message"
            }
            
            response = await self.http_client.post(f"{self.base_url}/stream/test", params=test_params)
            response.raise_for_status()
            result = response.json()
            
            print(f"Test event published: {result.get('event_id')}")
            print(f"Event type: {result.get('event_type')}")
            
        except Exception as e:
            print(f"Failed to test streaming endpoint: {e}")
    
    async def performance_test(self):
        """Test streaming performance with multiple events."""
        print(f"\n=== Streaming Performance Test ===")
        
        try:
            # Connect WebSocket for monitoring
            websocket_url = f"{self.ws_url}/stream/ws"
            self.websocket = await websockets.connect(websocket_url)
            
            # Subscribe to all event types
            subscription_message = {
                "type": "subscribe",
                "topics": ["event.created", "event.updated", "event.deleted"]
            }
            await self.websocket.send(json.dumps(subscription_message))
            
            # Skip confirmation message
            await self.websocket.recv()
            
            # Start event counter
            event_count = 0
            start_time = time.time()
            
            # Send multiple test events rapidly
            print("Sending 10 rapid test events...")
            
            for i in range(10):
                try:
                    # Use streaming test endpoint for speed
                    test_params = {
                        "event_type": "event.created",
                        "symbol": f"PERF{i:02d}",
                        "message": f"Performance test event #{i+1}"
                    }
                    
                    response = await self.http_client.post(f"{self.base_url}/stream/test", params=test_params)
                    if response.status_code == 200:
                        event_count += 1
                
                except Exception as e:
                    print(f"Failed to send performance test event {i+1}: {e}")
            
            # Monitor WebSocket for events (with timeout)
            received_count = 0
            timeout_time = time.time() + 5  # 5 second timeout
            
            while time.time() < timeout_time and received_count < event_count:
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    if data.get('type') == 'event.created':
                        received_count += 1
                        if received_count % 5 == 0:
                            print(f"  Received {received_count} events via WebSocket")
                
                except asyncio.TimeoutError:
                    break
                except Exception as e:
                    print(f"WebSocket receive error: {e}")
                    break
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\nPerformance Test Results:")
            print(f"  Events sent: {event_count}")
            print(f"  Events received via WebSocket: {received_count}")
            print(f"  Total time: {total_time:.2f} seconds")
            print(f"  Events per second: {event_count / total_time:.1f}")
            print(f"  Streaming efficiency: {received_count / event_count:.1%}")
            
        except Exception as e:
            print(f"Performance test failed: {e}")
    
    async def cleanup_test_data(self):
        """Clean up test events created during the demo."""
        print(f"\n=== Cleaning Up Test Data ===")
        
        try:
            # Get all events
            response = await self.http_client.get(f"{self.base_url}/events")
            response.raise_for_status()
            events = response.json()
            
            # Find test events
            test_events = []
            for event in events:
                metadata = event.get('metadata', {})
                symbol = event.get('symbol', '')
                
                if (metadata.get('streaming_test') is True or 
                    symbol.startswith('PERF') or 
                    symbol.startswith('STREAM_TEST') or
                    symbol == 'TEST'):
                    test_events.append(event)
            
            print(f"Found {len(test_events)} test events to clean up")
            
            # Delete test events
            deleted_count = 0
            for event in test_events:
                try:
                    response = await self.http_client.delete(f"{self.base_url}/events/{event['id']}")
                    if response.status_code in [204, 404]:
                        deleted_count += 1
                        if deleted_count % 10 == 0:
                            print(f"  Deleted {deleted_count} events...")
                except Exception as e:
                    print(f"  Error deleting event {event['id']}: {e}")
            
            print(f"Successfully deleted {deleted_count} test events")
            
        except Exception as e:
            print(f"Cleanup failed: {e}")


async def main():
    """Run the complete event streaming demonstration."""
    print("🚀 Event Data Service - Real-Time Streaming Demo")
    print("=" * 55)
    
    async with EventStreamingDemo() as demo:
        # Check service health
        streaming_enabled = await demo.health_check()
        
        if not streaming_enabled:
            print("\n❌ Event streaming is not enabled. Please check your configuration.")
            return
        
        print("\n✅ Event streaming is enabled and ready for demo!")
        
        # Test streaming API endpoints
        await demo.test_streaming_api()
        
        # Test WebSocket streaming
        await demo.test_websocket_streaming()
        
        # Test Server-Sent Events streaming
        await demo.test_sse_streaming()
        
        # Performance test
        await demo.performance_test()
        
        print(f"\n=== Demo Complete ===")
        print("Real-time streaming demo completed successfully!")
        print("\nKey features demonstrated:")
        print("  • WebSocket real-time connections with bidirectional communication")
        print("  • Server-Sent Events for lightweight real-time updates")
        print("  • Event filtering and subscription management")
        print("  • Automatic event distribution on CRUD operations")
        print("  • Performance monitoring and statistics")
        print("  • Multiple streaming backend support (Redis Streams, WebSocket, SSE)")
        
        # Ask if user wants to clean up
        try:
            cleanup = input("\nClean up test data? (y/N): ").strip().lower()
            if cleanup in ['y', 'yes']:
                await demo.cleanup_test_data()
                print("✅ Test data cleaned up successfully!")
            else:
                print("Test data preserved for further testing.")
        except KeyboardInterrupt:
            print("\nTest data preserved.")


if __name__ == "__main__":
    # Install required dependencies first
    try:
        import websockets
        import aiohttp
    except ImportError:
        print("Missing required dependencies. Please install:")
        print("pip install websockets aiohttp")
        exit(1)
    
    asyncio.run(main())