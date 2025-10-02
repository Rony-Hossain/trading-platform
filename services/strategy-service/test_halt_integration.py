#!/usr/bin/env python3
"""
Integration test for halt/LULD functionality in portfolio execution
"""

import asyncio
import sys
import os
from decimal import Decimal
from datetime import datetime

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.engines.portfolio_manager import PortfolioManager
from app.schemas import Trade, TradeType
from app.execution.venue_rules import HaltReason

async def test_halt_integration():
    """Test that halt checks are properly integrated into portfolio execution"""
    print("*** Testing Halt/LULD Integration in Portfolio Manager ***")
    
    # Initialize portfolio manager
    portfolio_manager = PortfolioManager(initial_cash=Decimal('100000'))
    
    # Initialize venue rules for test symbols
    test_symbols = ["AAPL", "GOOGL", "MSFT"]
    await portfolio_manager.initialize_venue_rules(test_symbols)
    print(f"[OK] Venue rules initialized for symbols: {test_symbols}")
    
    # Test 1: Normal trading (should succeed)
    print("\n[TEST 1] Normal trading (no halt)")
    trade1 = Trade(
        symbol="AAPL",
        trade_type=TradeType.BUY,
        quantity=100,
        price=Decimal('150.00'),
        timestamp=datetime.now()
    )
    
    result1 = await portfolio_manager.execute_trade(trade1, Decimal('150.00'))
    print(f"   Trade execution result: {result1}")
    if result1:
        print("   [PASS] Normal trade executed successfully")
    else:
        print("   [FAIL] Normal trade failed unexpectedly")
    
    # Test 2: Trading during halt (should be blocked)
    print("\n[TEST 2] Trading during halt (should be blocked)")
    
    # Simulate a halt on AAPL
    portfolio_manager.venue_rule_engine.halt_monitor.simulate_halt("AAPL", HaltReason.VOLATILITY)
    print("   Simulated halt on AAPL for volatility")
    
    trade2 = Trade(
        symbol="AAPL",
        trade_type=TradeType.SELL,
        quantity=50,
        price=Decimal('145.00'),
        timestamp=datetime.now()
    )
    
    result2 = await portfolio_manager.execute_trade(trade2, Decimal('145.00'))
    print(f"   Trade execution result: {result2}")
    if not result2:
        print("   [PASS] Halted trade correctly blocked")
    else:
        print("   [FAIL] Halted trade was executed - COMPLIANCE VIOLATION!")
    
    # Test 3: Clear halt and trade again (should succeed)
    print("\n[TEST 3] Clear halt and trade again")
    
    # Clear the halt by simulating reopen
    portfolio_manager.venue_rule_engine.halt_monitor.simulate_reopen("AAPL", 149.00)
    print("   Cleared halt on AAPL")
    
    trade3 = Trade(
        symbol="AAPL",
        trade_type=TradeType.SELL,
        quantity=25,
        price=Decimal('148.00'),
        timestamp=datetime.now()
    )
    
    result3 = await portfolio_manager.execute_trade(trade3, Decimal('148.00'))
    print(f"   Trade execution result: {result3}")
    if result3:
        print("   [PASS] Trade after halt clear executed successfully")
    else:
        print("   [FAIL] Trade after halt clear failed unexpectedly")
    
    # Test 4: Multiple symbols - mixed halt status
    print("\n[TEST 4] Multiple symbols with mixed halt status")
    
    # Halt GOOGL but keep MSFT normal
    portfolio_manager.venue_rule_engine.halt_monitor.simulate_halt("GOOGL", HaltReason.NEWS_PENDING)
    print("   Simulated halt on GOOGL for news pending")
    
    # Try to trade GOOGL (should be blocked)
    trade4a = Trade(
        symbol="GOOGL",
        trade_type=TradeType.BUY,
        quantity=10,
        price=Decimal('2800.00'),
        timestamp=datetime.now()
    )
    
    result4a = await portfolio_manager.execute_trade(trade4a, Decimal('2800.00'))
    print(f"   GOOGL trade result: {result4a}")
    
    # Try to trade MSFT (should succeed)
    trade4b = Trade(
        symbol="MSFT",
        trade_type=TradeType.BUY,
        quantity=50,
        price=Decimal('350.00'),
        timestamp=datetime.now()
    )
    
    result4b = await portfolio_manager.execute_trade(trade4b, Decimal('350.00'))
    print(f"   MSFT trade result: {result4b}")
    
    if not result4a and result4b:
        print("   [PASS] Mixed halt status handled correctly")
    else:
        print("   [FAIL] Mixed halt status not handled correctly")
    
    # Summary
    print("\n[SUMMARY] Test Summary:")
    print(f"   Portfolio Cash: ${portfolio_manager.cash}")
    print(f"   Positions: {len(portfolio_manager.positions)}")
    for symbol, position in portfolio_manager.positions.items():
        print(f"     {symbol}: {position.quantity} shares @ ${position.average_price}")
    
    # Compliance verification
    compliance_passed = (
        result1 and  # Normal trade succeeded
        not result2 and  # Halted trade blocked
        result3 and  # Post-halt trade succeeded
        not result4a and result4b  # Mixed status handled correctly
    )
    
    print(f"\n[COMPLIANCE] {'PASSED' if compliance_passed else 'FAILED'}")
    if compliance_passed:
        print("   Zero entries executed during halts - REQUIREMENT MET")
    else:
        print("   CRITICAL: Orders executed during halts - COMPLIANCE VIOLATION")
    
    return compliance_passed

if __name__ == "__main__":
    success = asyncio.run(test_halt_integration())
    sys.exit(0 if success else 1)