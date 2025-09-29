"""Example strategy service client demonstrating event subscription usage."""

import asyncio
import json
import logging
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example strategy service that subscribes to event notifications
app = FastAPI(title="Strategy Service Example", version="1.0.0")

# Event Data Service configuration
EVENT_SERVICE_URL = "http://localhost:8006"
STRATEGY_SERVICE_URL = "http://localhost:8007"

class StrategyService:
    """Example strategy service that reacts to event notifications."""
    
    def __init__(self):
        self.subscription_id = None
        self.received_events = []
        
    async def subscribe_to_events(self):
        """Subscribe to high-impact events for trading strategies."""
        subscription_request = {
            "service_name": "strategy-service",
            "webhook_url": f"{STRATEGY_SERVICE_URL}/webhooks/events",
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],  # High-volume stocks
            "categories": ["earnings", "fda_approval", "mna", "guidance"],  # High-impact categories
            "min_impact_score": 7,  # Only high-impact events
            "event_types": ["event.created", "event.impact_changed"],  # React to new events and score changes
            "timeout": 3.0,
            "retry_count": 2
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{EVENT_SERVICE_URL}/subscriptions",
                    json=subscription_request,
                    timeout=10.0
                )
                response.raise_for_status()
                subscription_data = response.json()
                self.subscription_id = subscription_data["id"]
                logger.info(f"Created subscription: {self.subscription_id}")
                return subscription_data
            except Exception as e:
                logger.error(f"Failed to create subscription: {e}")
                return None
    
    async def unsubscribe(self):
        """Unsubscribe from event notifications."""
        if not self.subscription_id:
            return
            
        async with httpx.AsyncClient() as client:
            try:
                response = await client.delete(
                    f"{EVENT_SERVICE_URL}/subscriptions/{self.subscription_id}",
                    timeout=10.0
                )
                response.raise_for_status()
                logger.info(f"Deleted subscription: {self.subscription_id}")
                self.subscription_id = None
            except Exception as e:
                logger.error(f"Failed to delete subscription: {e}")
    
    def process_event_notification(self, event_data: Dict[str, Any]):
        """Process incoming event notification and execute trading strategy."""
        self.received_events.append(event_data)
        
        event_type = event_data.get("event_type")
        data = event_data.get("data", {})
        symbol = data.get("symbol")
        category = data.get("category")
        impact_score = data.get("impact_score")
        
        logger.info(f"Received {event_type} for {symbol} ({category}, impact: {impact_score})")
        
        # Example trading strategy logic
        if event_type == "event.created":
            self._handle_new_event(data)
        elif event_type == "event.impact_changed":
            self._handle_impact_change(data)
    
    def _handle_new_event(self, event_data: Dict[str, Any]):
        """Handle new event creation - prepare for potential trade."""
        symbol = event_data.get("symbol")
        category = event_data.get("category")
        impact_score = event_data.get("impact_score", 0)
        
        if category == "earnings" and impact_score >= 8:
            logger.info(f"🚨 High-impact earnings event for {symbol} - preparing earnings strategy")
            # Example: Set up pre-earnings position, prepare IV crush strategy
            
        elif category == "fda_approval" and impact_score >= 9:
            logger.info(f"🚨 FDA approval event for {symbol} - preparing biotech momentum strategy")
            # Example: Prepare for potential gap up/down, set up momentum trade
            
        elif category == "mna" and impact_score >= 8:
            logger.info(f"🚨 M&A event for {symbol} - preparing arbitrage strategy")
            # Example: Analyze merger arbitrage opportunity
            
        elif category == "guidance" and impact_score >= 7:
            logger.info(f"📊 Guidance update for {symbol} - analyzing sentiment shift")
            # Example: Adjust position based on guidance direction
    
    def _handle_impact_change(self, event_data: Dict[str, Any]):
        """Handle impact score changes - adjust strategy accordingly."""
        symbol = event_data.get("symbol")
        new_impact = event_data.get("impact_score", 0)
        
        if new_impact >= 9:
            logger.info(f"⚠️ Impact score increased to {new_impact} for {symbol} - increasing position size")
            # Example: Increase position size for higher impact events
        elif new_impact <= 5:
            logger.info(f"📉 Impact score decreased to {new_impact} for {symbol} - reducing exposure")
            # Example: Reduce position size for lower impact events

# Global strategy service instance
strategy_service = StrategyService()

@app.on_event("startup")
async def startup_event():
    """Subscribe to events on startup."""
    await strategy_service.subscribe_to_events()

@app.on_event("shutdown")
async def shutdown_event():
    """Unsubscribe on shutdown."""
    await strategy_service.unsubscribe()

@app.post("/webhooks/events")
async def receive_event_webhook(request: Request):
    """Webhook endpoint to receive event notifications from Event Data Service."""
    try:
        event_data = await request.json()
        logger.info(f"Received webhook: {json.dumps(event_data, indent=2)}")
        
        # Process the event notification
        strategy_service.process_event_notification(event_data)
        
        return {"status": "received", "message": "Event processed successfully"}
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return {"status": "error", "message": str(e)}, 400

@app.get("/")
async def root():
    """Service information."""
    return {
        "service": "strategy-service-example",
        "status": "running",
        "subscription_id": strategy_service.subscription_id,
        "events_received": len(strategy_service.received_events),
        "webhook_endpoint": "/webhooks/events"
    }

@app.get("/events")
async def list_received_events():
    """List all received events for debugging."""
    return {
        "events": strategy_service.received_events,
        "count": len(strategy_service.received_events)
    }

@app.get("/subscription")
async def get_subscription_status():
    """Get current subscription status."""
    if not strategy_service.subscription_id:
        return {"status": "not_subscribed"}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{EVENT_SERVICE_URL}/subscriptions/{strategy_service.subscription_id}",
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Strategy Service Example...")
    print(f"Event Data Service: {EVENT_SERVICE_URL}")
    print(f"Strategy Service: {STRATEGY_SERVICE_URL}")
    print("This service will subscribe to high-impact events and demonstrate trading strategy reactions.")
    
    uvicorn.run(app, host="0.0.0.0", port=8007)