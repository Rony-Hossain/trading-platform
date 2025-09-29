"""Demo script showing event lifecycle tracking capabilities."""

import asyncio
import json
from datetime import datetime, timedelta, timezone

# Mock event data for demonstration
SAMPLE_LIFECYCLE_EVENTS = [
    {
        "symbol": "AAPL",
        "title": "Apple Q4 Earnings Call",
        "category": "earnings",
        "scheduled_at": datetime.now(timezone.utc) + timedelta(hours=2),  # Future event
        "status": "scheduled",
        "impact_score": 8,
    },
    {
        "symbol": "TSLA", 
        "title": "Tesla FSD Beta Release",
        "category": "product_launch",
        "scheduled_at": datetime.now(timezone.utc) - timedelta(hours=6),  # Recent past event
        "status": "occurred",
        "impact_score": 7,
    },
    {
        "symbol": "PFE",
        "title": "Pfizer FDA Drug Approval",
        "category": "fda_approval", 
        "scheduled_at": datetime.now(timezone.utc) - timedelta(days=2),  # Older event
        "status": "occurred",
        "impact_score": 9,
    },
    {
        "symbol": "NVDA",
        "title": "NVIDIA AI Conference",
        "category": "product_launch",
        "scheduled_at": datetime.now(timezone.utc) + timedelta(days=5),  # Far future
        "status": "scheduled", 
        "impact_score": 6,
    }
]


async def demo_lifecycle_tracking():
    """Demonstrate the event lifecycle tracking capabilities."""
    from app.services.event_lifecycle import EventLifecycleTracker, EventStatus, LifecycleStage
    from app.models import EventORM
    
    print("🔄 Event Lifecycle Tracking Demo")
    print("=" * 50)
    
    # Mock session factory for demo
    class MockSessionFactory:
        def __call__(self):
            return MockAsyncSession()
    
    class MockAsyncSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def execute(self, stmt):
            # Mock database query results
            return MockResult()
        async def get(self, model, id):
            # Mock single record retrieval
            return MockEventORM()
        async def commit(self):
            pass
        async def rollback(self):
            pass
    
    class MockResult:
        def scalars(self):
            return MockScalars()
    
    class MockScalars:
        def all(self):
            # Return empty list for demo
            return []
        def first(self):
            return None
    
    class MockEventORM:
        def __init__(self):
            self.id = "mock-event-123"
            self.symbol = "AAPL"
            self.title = "Mock Event"
            self.category = "earnings"
            self.status = "scheduled"
            self.scheduled_at = datetime.now(timezone.utc)
            self.metadata_json = {}
    
    # Create lifecycle tracker
    session_factory = MockSessionFactory()
    lifecycle_tracker = EventLifecycleTracker(session_factory)
    
    # Don't actually start the monitoring loop for demo
    lifecycle_tracker._running = True
    
    print("\n1. Creating Lifecycle Events")
    print("-" * 30)
    
    # Create mock events and track their lifecycle
    for i, event_data in enumerate(SAMPLE_LIFECYCLE_EVENTS):
        event = MockEventORM()
        event.id = f"event-{i+1}"
        event.symbol = event_data["symbol"]
        event.title = event_data["title"]
        event.category = event_data["category"]
        event.scheduled_at = event_data["scheduled_at"]
        event.status = event_data["status"]
        
        lifecycle_event = await lifecycle_tracker._create_lifecycle_event(event)
        
        print(f"\n📊 Event {i+1}: {event.symbol} - {event.title}")
        print(f"    Scheduled: {event.scheduled_at.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"    Status: {lifecycle_event.current_status.value}")
        print(f"    Stage: {lifecycle_event.current_stage.value}")
        
        # Show stage determination logic
        now = datetime.now(timezone.utc)
        time_to_event = (event.scheduled_at - now).total_seconds() / 3600
        
        if time_to_event > 0:
            print(f"    Time to event: {time_to_event:.1f} hours")
        else:
            print(f"    Time since event: {-time_to_event:.1f} hours")
    
    print("\n\n2. Lifecycle Stage Progression")
    print("-" * 30)
    
    print("📈 Lifecycle Stages:")
    for stage in LifecycleStage:
        events_in_stage = lifecycle_tracker.get_events_by_stage(stage)
        print(f"  {stage.value}: {len(events_in_stage)} events")
        
        for event in events_in_stage:
            print(f"    - {event.symbol}: {event.title[:40]}...")
    
    print("\n\n3. Status Distribution")
    print("-" * 20)
    
    print("📊 Event Statuses:")
    for status in EventStatus:
        events_with_status = lifecycle_tracker.get_events_by_status(status)
        print(f"  {status.value}: {len(events_with_status)} events")
    
    print("\n\n4. Manual Status Updates")
    print("-" * 25)
    
    # Demonstrate manual status updates
    event_to_update = list(lifecycle_tracker._lifecycle_cache.values())[0]
    original_status = event_to_update.current_status
    
    print(f"🔄 Updating {event_to_update.symbol} status from {original_status.value} to 'occurred'")
    
    updated_event = await lifecycle_tracker.update_event_status(
        event_to_update.event_id, 
        EventStatus.OCCURRED, 
        "demo_manual_update"
    )
    
    print(f"  Status updated successfully!")
    print(f"  Status history entries: {len(updated_event.status_history)}")
    print(f"  Latest entry: {updated_event.status_history[-1]}")
    
    print("\n\n5. Impact Analysis Simulation")
    print("-" * 30)
    
    # Simulate impact analysis for an "occurred" event
    occurred_events = lifecycle_tracker.get_events_by_status(EventStatus.OCCURRED)
    
    if occurred_events:
        test_event = occurred_events[0]
        print(f"🎯 Simulating impact analysis for {test_event.symbol}")
        
        # Simulate the impact analysis process
        await lifecycle_tracker._analyze_event_impact(test_event)
        
        if test_event.impact_metrics:
            metrics = test_event.impact_metrics
            print(f"  Impact Analysis Results:")
            print(f"    Predicted Impact: {metrics.predicted_impact}")
            print(f"    Actual Impact: {metrics.actual_impact:.2f}")
            print(f"    Accuracy Score: {metrics.accuracy_score:.2f}")
            print(f"    Max Price Move: {metrics.max_move_pct}%")
            print(f"    Volume Change: {metrics.volume_change_pct}%")
            print(f"    Headlines Count: {metrics.headline_count}")
            print(f"    Analysis Time: {metrics.analysis_timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
        else:
            print("  No impact metrics generated")
    
    print("\n\n6. Lifecycle Statistics")
    print("-" * 20)
    
    stats = lifecycle_tracker.get_lifecycle_stats()
    print("📈 Lifecycle Tracking Stats:")
    print(f"  Total tracked events: {stats['total_tracked_events']}")
    
    print(f"\n  Status Distribution:")
    for status, count in stats['status_distribution'].items():
        print(f"    {status}: {count}")
    
    print(f"\n  Stage Distribution:")
    for stage, count in stats['stage_distribution'].items():
        print(f"    {stage}: {count}")
    
    print(f"\n  Analysis Performance:")
    print(f"    Analyzed events: {stats['analyzed_events']}")
    print(f"    Average accuracy: {stats['average_accuracy']:.2f}")
    
    print(f"\n  Configuration:")
    config = stats['config']
    for key, value in config.items():
        print(f"    {key}: {value}")
    
    print("\n\n7. Real-world Use Cases")
    print("-" * 25)
    
    print("🎯 Lifecycle Tracking Use Cases:")
    print("\n  📊 Performance Monitoring:")
    print("    - Track prediction accuracy across event types")
    print("    - Identify which categories are hardest to predict")
    print("    - Monitor model performance over time")
    
    print("\n  🚨 Operational Alerts:")
    print("    - Alert when high-impact events occur")
    print("    - Notify when predictions are significantly off")
    print("    - Track events that don't get properly analyzed")
    
    print("\n  📈 Strategy Optimization:")
    print("    - Analyze which event types generate best returns")
    print("    - Optimize position sizing based on prediction confidence")
    print("    - Identify optimal holding periods for different events")
    
    print("\n  🔍 Research & Analysis:")
    print("    - Study market reactions to different event types")
    print("    - Analyze seasonal patterns in event impact")
    print("    - Research correlation between events and volatility")
    
    print("\n\n8. Integration with Trading Strategies")
    print("-" * 35)
    
    print("⚡ Strategy Integration Examples:")
    
    # Show how a strategy service might use lifecycle data
    print("\n  📋 Pre-Event Preparation:")
    pre_event_events = lifecycle_tracker.get_events_by_stage(LifecycleStage.PRE_EVENT)
    for event in pre_event_events[:2]:  # Show first 2
        time_to_event = (event.scheduled_at - datetime.now(timezone.utc)).total_seconds() / 3600
        print(f"    {event.symbol}: {time_to_event:.1f}h until {event.category} event")
        print(f"      → Prepare position sizing strategy")
        print(f"      → Set up volatility alerts")
        print(f"      → Review historical patterns")
    
    print("\n  🎯 Active Event Monitoring:")
    event_window_events = lifecycle_tracker.get_events_by_stage(LifecycleStage.EVENT_WINDOW)
    for event in event_window_events:
        print(f"    {event.symbol}: {event.category} event in progress")
        print(f"      → Monitor real-time price action")
        print(f"      → Execute event-driven strategies")
        print(f"      → Adjust stop losses and targets")
    
    print("\n  📊 Post-Event Analysis:")
    post_event_events = lifecycle_tracker.get_events_by_stage(LifecycleStage.POST_EVENT)
    for event in post_event_events[:2]:  # Show first 2
        time_since_event = (datetime.now(timezone.utc) - event.scheduled_at).total_seconds() / 3600
        print(f"    {event.symbol}: {time_since_event:.1f}h since {event.category} event")
        print(f"      → Measure actual vs predicted impact")
        print(f"      → Analyze strategy performance")
        print(f"      → Update prediction models")
    
    print("\n" + "=" * 50)
    print("✅ Event Lifecycle Tracking Demo Complete!")
    print("\nKey Benefits:")
    print("• Automated tracking of event progression")
    print("• Comprehensive impact analysis and accuracy measurement")
    print("• Performance monitoring for prediction models")
    print("• Integration hooks for trading strategy optimization")
    print("• Historical data for research and backtesting")


if __name__ == "__main__":
    print("Event Data Service - Lifecycle Tracking Demo")
    print("This demo shows how events are tracked through their lifecycle")
    print("from scheduled → occurred → impact analyzed")
    print()
    
    asyncio.run(demo_lifecycle_tracking())