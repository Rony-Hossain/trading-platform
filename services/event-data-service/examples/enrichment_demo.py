"""Demo script showing event enrichment capabilities."""

import asyncio
import json
from typing import Dict, Any

# Mock event data for demonstration
SAMPLE_EVENTS = [
    {
        "symbol": "AAPL",
        "title": "Apple Q4 Earnings Call",
        "category": "earnings",
        "scheduled_at": "2025-01-28T21:00:00Z",
        "description": "Apple Inc. will report Q4 2024 financial results",
        "status": "scheduled",
        "impact_score": 8,
        "metadata": {
            "analyst_consensus": 1.25,
            "estimated_revenue": 89.5e9
        }
    },
    {
        "symbol": "TSLA",
        "title": "Tesla Full Self-Driving Beta Release",
        "category": "product_launch",
        "scheduled_at": "2025-01-30T10:00:00Z",
        "description": "Tesla announces major FSD beta update",
        "status": "scheduled",
        "impact_score": 7,
        "metadata": {
            "product_type": "software_update",
            "market_impact": "high"
        }
    },
    {
        "symbol": "PFE",
        "title": "Pfizer FDA Drug Approval Decision",
        "category": "fda_approval",
        "scheduled_at": "2025-02-01T16:00:00Z",
        "description": "FDA decision on new Alzheimer's drug",
        "status": "scheduled",
        "impact_score": 9,
        "metadata": {
            "drug_name": "PF-12345",
            "therapeutic_area": "neurology"
        }
    },
    {
        "symbol": "NVDA",
        "title": "NVIDIA AI Conference Keynote",
        "category": "product_launch",
        "scheduled_at": "2025-02-15T11:00:00Z",
        "description": "Jensen Huang keynote at AI conference",
        "status": "scheduled",
        "impact_score": 6,
        "metadata": {
            "conference": "AI Summit 2025",
            "expected_announcements": ["new_gpu", "ai_software"]
        }
    }
]


async def demo_enrichment_service():
    """Demonstrate the event enrichment service capabilities."""
    from app.services.event_enrichment import EventEnrichmentService, EnrichmentConfig
    
    print("🚀 Event Enrichment Service Demo")
    print("=" * 50)
    
    # Create enrichment service with demo config
    config = EnrichmentConfig(
        finnhub_api_key=None,  # Will use env var if available
        yahoo_finance_enabled=True,
        cache_duration_minutes=30,
        max_retries=2,
        timeout_seconds=5.0,
        batch_size=4
    )
    
    enrichment_service = EventEnrichmentService(config)
    await enrichment_service.start()
    
    try:
        print("\n1. Getting Market Context for Individual Symbols")
        print("-" * 45)
        
        symbols = ["AAPL", "TSLA", "PFE", "NVDA"]
        for symbol in symbols:
            print(f"\n📊 Fetching market context for {symbol}...")
            try:
                context = await enrichment_service.get_market_context(symbol)
                print(f"  Market Cap: ${context.market_cap/1e9:.1f}B ({context.market_cap_tier.value})" if context.market_cap else "  Market Cap: N/A")
                print(f"  Sector: {context.sector or 'Unknown'}")
                print(f"  Price: ${context.price:.2f}" if context.price else "  Price: N/A")
                print(f"  Beta: {context.beta:.2f}" if context.beta else "  Beta: N/A")
                print(f"  30d Volatility: {context.volatility_30d:.1f}% ({context.volatility_level.value})" if context.volatility_30d else "  Volatility: N/A")
                print(f"  Avg Volume: {context.avg_volume:,.0f}" if context.avg_volume else "  Volume: N/A")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print("\n\n2. Enriching Individual Events")
        print("-" * 35)
        
        for i, event in enumerate(SAMPLE_EVENTS[:2]):  # Demo with first 2 events
            print(f"\n🎯 Enriching event {i+1}: {event['symbol']} - {event['title']}")
            
            try:
                enriched_event = await enrichment_service.enrich_event(event.copy())
                
                # Show original vs enriched
                print(f"  Original metadata fields: {len(event.get('metadata', {}))}")
                enriched_metadata = enriched_event.get('metadata', {})
                print(f"  Enriched metadata fields: {len(enriched_metadata)}")
                
                # Show enrichment details
                enrichment = enriched_metadata.get('enrichment', {})
                if enrichment:
                    market_context = enrichment.get('market_context', {})
                    impact_modifiers = enrichment.get('impact_modifiers', {})
                    
                    print(f"  Market Context Added:")
                    print(f"    - Market Cap Tier: {market_context.get('market_cap_tier', 'N/A')}")
                    print(f"    - Sector: {market_context.get('sector', 'N/A')}")
                    print(f"    - Volatility Level: {market_context.get('volatility_level', 'N/A')}")
                    
                    if impact_modifiers:
                        print(f"  Impact Modifiers:")
                        for modifier, value in impact_modifiers.items():
                            print(f"    - {modifier}: {value:+.1f}")
                else:
                    print("  No enrichment data added")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print("\n\n3. Batch Enrichment Performance")
        print("-" * 35)
        
        print(f"📦 Batch enriching {len(SAMPLE_EVENTS)} events...")
        
        try:
            import time
            start_time = time.time()
            
            enriched_events = await enrichment_service.batch_enrich_events(SAMPLE_EVENTS.copy())
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"  ✅ Enriched {len(enriched_events)} events in {duration:.2f}s")
            print(f"  Average time per event: {duration/len(enriched_events):.3f}s")
            
            # Show enrichment statistics
            enriched_count = 0
            for event in enriched_events:
                if event.get('metadata', {}).get('enrichment'):
                    enriched_count += 1
            
            print(f"  Successfully enriched: {enriched_count}/{len(enriched_events)} events")
            
            # Show example enriched event structure
            if enriched_events:
                example_event = enriched_events[0]
                enrichment = example_event.get('metadata', {}).get('enrichment', {})
                if enrichment:
                    print(f"\n  Example enriched event structure:")
                    print(f"    Symbol: {example_event['symbol']}")
                    print(f"    Original impact score: {example_event.get('impact_score', 'N/A')}")
                    
                    market_context = enrichment.get('market_context', {})
                    impact_modifiers = enrichment.get('impact_modifiers', {})
                    
                    if market_context:
                        print(f"    Market cap: ${market_context.get('market_cap', 0)/1e9:.1f}B")
                        print(f"    Market cap tier: {market_context.get('market_cap_tier', 'N/A')}")
                        print(f"    Sector: {market_context.get('sector', 'N/A')}")
                        print(f"    Volatility: {market_context.get('volatility_30d', 'N/A')}%")
                    
                    if impact_modifiers:
                        total_modifier = sum(impact_modifiers.values())
                        print(f"    Impact modifiers: {impact_modifiers}")
                        print(f"    Total modifier: {total_modifier:+.1f}")
                        
        except Exception as e:
            print(f"  ❌ Batch enrichment failed: {e}")
        
        print("\n\n4. Service Statistics")
        print("-" * 20)
        
        stats = enrichment_service.get_enrichment_stats()
        print(f"📈 Enrichment Service Stats:")
        print(f"  Cached symbols: {stats['cached_symbols']}")
        print(f"  Configuration:")
        config_info = stats['config']
        for key, value in config_info.items():
            print(f"    {key}: {value}")
        
        print("\n\n5. Real-world Impact Scoring Demo")
        print("-" * 35)
        
        print("🎯 Demonstrating how enrichment improves impact scoring:")
        
        # Show how different company sizes get different impact modifiers
        test_cases = [
            ("AAPL", "mega_cap", "Apple - Mega cap technology stock"),
            ("AMD", "large_cap", "AMD - Large cap semiconductor"),
            ("RBLX", "mid_cap", "Roblox - Mid cap gaming platform"),
            ("PLUG", "small_cap", "Plug Power - Small cap hydrogen fuel"),
        ]
        
        for symbol, expected_tier, description in test_cases:
            try:
                context = await enrichment_service.get_market_context(symbol)
                
                print(f"\n  {description}")
                print(f"    Market Cap Tier: {context.market_cap_tier.value}")
                print(f"    Volatility Level: {context.volatility_level.value}")
                
                # Calculate impact modifiers
                modifiers = enrichment_service._calculate_impact_modifiers(context)
                if modifiers:
                    total_modifier = sum(modifiers.values())
                    print(f"    Impact Modifiers: {modifiers}")
                    print(f"    Net Modifier: {total_modifier:+.1f}")
                    print(f"    → A base 7/10 event would become: {min(10, max(1, 7 + total_modifier)):.1f}/10")
                else:
                    print(f"    No modifiers calculated")
                    
            except Exception as e:
                print(f"    ❌ Error for {symbol}: {e}")
    
    finally:
        await enrichment_service.stop()
    
    print("\n" + "=" * 50)
    print("✅ Event Enrichment Demo Complete!")
    print("\nKey Benefits:")
    print("• Automatic market context for all events")
    print("• Impact score adjustments based on company characteristics")
    print("• Sector and volatility classification for strategy targeting")
    print("• Cached data for performance optimization")
    print("• Batch processing for efficient bulk operations")


if __name__ == "__main__":
    print("Event Data Service - Enrichment Demo")
    print("This demo shows how events are enriched with market context")
    print("Note: Real market data requires API keys (Finnhub, Alpha Vantage)")
    print()
    
    asyncio.run(demo_enrichment_service())