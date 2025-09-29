#!/usr/bin/env python3
"""
Quick test script to verify unusual options activity detection is working
"""

import asyncio
import sys
import os

# Add the services path so we can import
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'market-data-service', 'app'))

from services.options_service import options_service

async def test_unusual_activity():
    """Test the unusual activity detection"""
    
    print("Testing unusual options activity detection...")
    
    # Test with a common symbol
    symbol = "AAPL"
    
    try:
        # Test 1: Get options chain
        print(f"\n1. Fetching options chain for {symbol}...")
        chain = await options_service.fetch_options_chain(symbol)
        print(f"   ✓ Got {len(chain.calls)} calls and {len(chain.puts)} puts")
        print(f"   ✓ Underlying price: ${chain.underlying_price}")
        
        # Test 2: Detect unusual activity
        print(f"\n2. Detecting unusual activity for {symbol}...")
        unusual_activities = await options_service.detect_unusual_activity(symbol)
        print(f"   ✓ Found {len(unusual_activities)} unusual activities")
        
        if unusual_activities:
            top_activity = unusual_activities[0]
            print(f"   ✓ Top unusual activity:")
            print(f"     - Contract: {top_activity.contract_symbol}")
            print(f"     - Volume ratio: {top_activity.volume_ratio:.2f}x")
            print(f"     - Unusual score: {top_activity.unusual_score:.1f}/100")
            print(f"     - Volume spike: {top_activity.volume_spike}")
            print(f"     - Large trades: {top_activity.large_single_trades}")
        
        # Test 3: Flow analysis
        print(f"\n3. Analyzing options flow for {symbol}...")
        flow_analysis = await options_service.analyze_options_flow(symbol)
        print(f"   ✓ Call volume: {flow_analysis.total_call_volume:,}")
        print(f"   ✓ Put volume: {flow_analysis.total_put_volume:,}")
        print(f"   ✓ Call/Put ratio: {flow_analysis.call_put_ratio:.2f}")
        print(f"   ✓ Flow sentiment: {flow_analysis.flow_sentiment}")
        print(f"   ✓ Smart money score: {flow_analysis.smart_money_score:.1f}/100")
        print(f"   ✓ Block trades value: ${flow_analysis.block_trades_value:,.0f}")
        
        print(f"\n✅ All tests passed! Unusual activity detection is working.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_unusual_activity())
    sys.exit(0 if success else 1)