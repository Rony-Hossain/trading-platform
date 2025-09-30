"""
Comprehensive Examples for Advanced Execution Modeling

This module demonstrates the complete execution modeling capabilities including:
- Depth-aware slippage modeling with fill probability curves
- Comprehensive exchange fee structures (NYSE, NASDAQ, BATS, IEX)
- Realistic market microstructure simulation
- Paper trading with advanced execution costs

Key Features:
- SEC fees: $27.80 per $1M notional
- TAF fees: $0.119 per 100 shares (NMS stocks)
- FINRA ORF: 0.5 mils of dollar volume
- Exchange-specific maker/taker fees
- Depth-based slippage modeling
- Fill probability curves based on order size vs available liquidity
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Import execution modeling components
from ..services.execution_modeling import (
    AdvancedExecutionEngine, 
    ComprehensiveFeeCalculator,
    DepthAwareExecutionModel,
    ExchangeType,
    OrderExecutionType, 
    MarketDepth,
    create_execution_engine
)

# Import paper trading components
from ..services.paper_trading import (
    PaperTradingEngine,
    create_paper_trading_engine,
    OrderSide,
    OrderType
)

logger = logging.getLogger(__name__)

class ExecutionModelingDemo:
    """Comprehensive demonstration of execution modeling capabilities"""
    
    def __init__(self):
        self.engines: Dict[ExchangeType, AdvancedExecutionEngine] = {}
        self.paper_engine: Optional[PaperTradingEngine] = None
        
    async def initialize_engines(self):
        """Initialize execution engines for all supported exchanges"""
        print("🔧 Initializing Advanced Execution Engines...")
        
        for exchange in ExchangeType:
            self.engines[exchange] = create_execution_engine(exchange)
            print(f"   ✓ {exchange.value} engine ready")
        
        # Initialize paper trading with NYSE engine
        self.paper_engine = await create_paper_trading_engine(
            initial_balance=100000.0,
            exchange_type=ExchangeType.NYSE
        )
        print("   ✓ Paper trading engine ready")
        print()
    
    def demonstrate_fee_structures(self):
        """Demonstrate comprehensive fee structures across exchanges"""
        print("💰 COMPREHENSIVE EXCHANGE FEE STRUCTURES")
        print("=" * 60)
        
        test_scenarios = [
            {"symbol": "AAPL", "quantity": 100, "price": 150.00, "side": "buy"},
            {"symbol": "TSLA", "quantity": 500, "price": 200.00, "side": "sell"},
            {"symbol": "SPY", "quantity": 1000, "price": 400.00, "side": "buy"},
        ]
        
        for scenario in test_scenarios:
            print(f"\nScenario: {scenario['side'].upper()} {scenario['quantity']} {scenario['symbol']} @ ${scenario['price']}")
            print(f"Notional Value: ${scenario['quantity'] * scenario['price']:,.2f}")
            print("-" * 50)
            
            for exchange, engine in self.engines.items():
                costs = engine.calculate_total_costs(
                    scenario['symbol'],
                    scenario['quantity'],
                    scenario['price'],
                    scenario['side']
                )
                
                print(f"{exchange.value:8} | Total: ${costs['total_fees']:6.2f} | " +
                      f"SEC: ${costs['fee_breakdown']['sec_fee']:5.2f} | " +
                      f"TAF: ${costs['fee_breakdown']['taf_fee']:5.2f} | " +
                      f"ORF: ${costs['fee_breakdown']['finra_orf']:5.2f}")
        
        print("\n" + "=" * 60)
    
    async def demonstrate_depth_aware_execution(self):
        """Demonstrate depth-aware execution with slippage modeling"""
        print("📊 DEPTH-AWARE EXECUTION MODELING")
        print("=" * 60)
        
        # Create realistic market depth scenarios
        market_scenarios = [
            {
                "name": "Liquid Large Cap (AAPL)",
                "symbol": "AAPL",
                "bid_price": 149.95,
                "ask_price": 150.00,
                "bid_size": 5000,
                "ask_size": 4500,
                "last_price": 149.98
            },
            {
                "name": "Mid Cap Stock",
                "symbol": "XYZ",
                "bid_price": 99.85,
                "ask_price": 100.15,
                "bid_size": 800,
                "ask_size": 600,
                "last_price": 100.00
            },
            {
                "name": "Illiquid Small Cap",
                "symbol": "ABC",
                "bid_price": 24.90,
                "ask_price": 25.10,
                "bid_size": 200,
                "ask_size": 300,
                "last_price": 25.00
            }
        ]
        
        order_sizes = [100, 500, 1000, 2000]
        
        for scenario in market_scenarios:
            print(f"\n{scenario['name']}")
            print(f"Market: ${scenario['bid_price']:.2f} x ${scenario['ask_price']:.2f} " +
                  f"({scenario['bid_size']} x {scenario['ask_size']})")
            print(f"Spread: {((scenario['ask_price'] - scenario['bid_price']) / scenario['bid_price'] * 10000):.1f} bps")
            print("-" * 50)
            
            market_depth = MarketDepth(
                symbol=scenario['symbol'],
                timestamp=datetime.now(),
                bid_price=scenario['bid_price'],
                ask_price=scenario['ask_price'],
                bid_size=scenario['bid_size'],
                ask_size=scenario['ask_size'],
                last_price=scenario['last_price'],
                volume=100000
            )
            
            # Test different order sizes
            for size in order_sizes:
                engine = self.engines[ExchangeType.NYSE]  # Use NYSE for demo
                
                # Simulate BUY order
                result = await engine.execute_order_async(
                    symbol=scenario['symbol'],
                    side="buy",
                    quantity=size,
                    order_type=OrderExecutionType.MARKET,
                    market_depth=market_depth
                )
                
                if result['success']:
                    slippage_bps = (result['avg_fill_price'] - scenario['last_price']) / scenario['last_price'] * 10000
                    fill_prob = result.get('fill_probability', 1.0)
                    
                    print(f"  {size:4d} shares: ${result['avg_fill_price']:7.3f} " +
                          f"(+{slippage_bps:5.1f} bps) " +
                          f"Fill: {fill_prob:5.1%} " +
                          f"Fees: ${result['total_costs']['total_fees']:6.2f}")
        
        print("\n" + "=" * 60)
    
    async def demonstrate_paper_trading_integration(self):
        """Demonstrate paper trading with advanced execution modeling"""
        print("📝 PAPER TRADING WITH ADVANCED EXECUTION")
        print("=" * 60)
        
        # Create paper account
        account = await self.paper_engine.create_paper_account("advanced_execution_demo")
        print(f"Created paper account: {account.account_id}")
        print(f"Initial balance: ${account.initial_balance:,.2f}")
        print(f"Buying power: ${account.buying_power:,.2f}")
        print()
        
        # Execute series of orders with realistic market data
        orders = [
            {"symbol": "AAPL", "side": OrderSide.BUY, "quantity": 100, "note": "Large cap liquid"},
            {"symbol": "MSFT", "side": OrderSide.BUY, "quantity": 200, "note": "Tech stock"},
            {"symbol": "SPY", "side": OrderSide.BUY, "quantity": 500, "note": "ETF large order"},
            {"symbol": "AAPL", "side": OrderSide.SELL, "quantity": 50, "note": "Partial position close"},
        ]
        
        # Simulate realistic market data
        market_data_cache = {
            "AAPL": {"bid": 149.95, "ask": 150.00, "last": 149.98, "bid_size": 5000, "ask_size": 4500},
            "MSFT": {"bid": 299.80, "ask": 300.20, "last": 300.00, "bid_size": 2000, "ask_size": 1800},
            "SPY": {"bid": 399.95, "ask": 400.05, "last": 400.00, "bid_size": 10000, "ask_size": 9500},
        }
        
        total_fees = 0.0
        
        for i, order_spec in enumerate(orders, 1):
            symbol = order_spec["symbol"]
            market_data = market_data_cache[symbol]
            
            # Update market data cache in paper engine
            from ..services.paper_trading import MarketData
            self.paper_engine.market_data_cache[symbol] = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                bid=market_data["bid"],
                ask=market_data["ask"],
                last=market_data["last"],
                volume=100000,
                bid_size=market_data["bid_size"],
                ask_size=market_data["ask_size"]
            )
            
            # Place order
            try:
                order = await self.paper_engine.place_order(
                    account_id=account.account_id,
                    symbol=symbol,
                    side=order_spec["side"],
                    quantity=order_spec["quantity"],
                    order_type=OrderType.MARKET,
                    metadata={"demo": True, "note": order_spec["note"]}
                )
                
                print(f"Order {i}: {order_spec['side'].value.upper()} {order_spec['quantity']} {symbol}")
                print(f"  Status: {order.status.value}")
                print(f"  Fill Price: ${order.avg_fill_price:.3f}")
                print(f"  Commission: ${order.commission:.2f}")
                print(f"  Note: {order_spec['note']}")
                
                total_fees += order.commission
                
            except Exception as e:
                print(f"Order {i} failed: {e}")
            
            print()
        
        # Get account summary
        await self._update_account_summary(account)
        
        print("FINAL ACCOUNT SUMMARY")
        print("-" * 30)
        print(f"Cash Balance: ${account.cash_balance:,.2f}")
        print(f"Total Value: ${account.total_value:,.2f}")
        print(f"Total Fees Paid: ${total_fees:.2f}")
        print(f"Number of Positions: {len(account.positions)}")
        print()
        
        # Show position details
        if account.positions:
            print("POSITION DETAILS")
            print("-" * 30)
            for symbol, position in account.positions.items():
                market_value = position.quantity * market_data_cache[symbol]["last"]
                pnl = market_value - (position.quantity * position.avg_cost)
                
                print(f"{symbol}: {position.quantity} shares @ ${position.avg_cost:.3f}")
                print(f"  Market Value: ${market_value:,.2f}")
                print(f"  Unrealized P&L: ${pnl:+.2f}")
        
        print("\n" + "=" * 60)
    
    async def _update_account_summary(self, account):
        """Update account summary with current market values"""
        total_value = account.cash_balance
        
        for symbol, position in account.positions.items():
            if symbol in self.paper_engine.market_data_cache:
                market_data = self.paper_engine.market_data_cache[symbol]
                market_value = position.quantity * market_data.last
                total_value += market_value
        
        account.total_value = total_value
    
    def demonstrate_regulatory_fee_calculations(self):
        """Demonstrate detailed regulatory fee calculations"""
        print("🏛️  REGULATORY FEE CALCULATIONS")
        print("=" * 60)
        
        # Large transaction to show all fees clearly
        test_case = {
            "symbol": "AAPL",
            "quantity": 10000,  # Large order
            "price": 150.00,
            "notional": 10000 * 150.00  # $1.5M
        }
        
        print(f"Test Case: SELL {test_case['quantity']:,} {test_case['symbol']} @ ${test_case['price']}")
        print(f"Notional Value: ${test_case['notional']:,.2f}")
        print()
        
        engine = self.engines[ExchangeType.NYSE]
        costs = engine.calculate_total_costs(
            test_case['symbol'],
            test_case['quantity'], 
            test_case['price'],
            "sell"
        )
        
        breakdown = costs['fee_breakdown']
        
        print("DETAILED FEE BREAKDOWN")
        print("-" * 40)
        print(f"SEC Fee (0.00278%):      ${breakdown['sec_fee']:8.2f}")
        print(f"TAF Fee ($0.119/100):    ${breakdown['taf_fee']:8.2f}")
        print(f"FINRA ORF (0.5 mils):    ${breakdown['finra_orf']:8.2f}")
        print(f"Exchange Fee:            ${breakdown['taker_fee']:8.2f}")
        print(f"Clearing Fee:            ${breakdown['clearing_fee']:8.2f}")
        print("-" * 40)
        print(f"TOTAL FEES:              ${costs['total_fees']:8.2f}")
        print()
        
        # Show fee breakdown as percentage of notional
        fee_pct = (costs['total_fees'] / test_case['notional']) * 100
        print(f"Total fees as % of notional: {fee_pct:.4f}%")
        print(f"Effective rate per share: ${costs['total_fees'] / test_case['quantity']:.4f}")
        
        print("\n" + "=" * 60)
    
    async def run_comprehensive_demo(self):
        """Run complete demonstration of execution modeling capabilities"""
        print("🚀 ADVANCED EXECUTION MODELING DEMONSTRATION")
        print("=" * 80)
        print("This demo showcases comprehensive execution modeling with:")
        print("• Depth-aware slippage and fill probability curves")
        print("• Exchange-specific fee structures (NYSE, NASDAQ, BATS, IEX)")
        print("• Realistic regulatory fees (SEC, TAF, FINRA ORF)")
        print("• Paper trading integration with advanced execution")
        print("=" * 80)
        print()
        
        await self.initialize_engines()
        
        self.demonstrate_fee_structures()
        print()
        
        await self.demonstrate_depth_aware_execution()
        print()
        
        self.demonstrate_regulatory_fee_calculations()
        print()
        
        await self.demonstrate_paper_trading_integration()
        
        print("✅ DEMONSTRATION COMPLETE")
        print("The advanced execution modeling system is ready for production use!")

# Utility functions for testing and validation

async def test_execution_engine_accuracy():
    """Test execution engine accuracy against known benchmarks"""
    print("🧪 TESTING EXECUTION ENGINE ACCURACY")
    print("=" * 50)
    
    # Known test cases with expected results
    test_cases = [
        {
            "name": "SEC Fee Calculation",
            "params": {"symbol": "AAPL", "quantity": 1000, "price": 150.0, "side": "sell"},
            "exchange": ExchangeType.NYSE,
            "expected_sec_fee": 4.17,  # $1.5M * 0.0000278
            "tolerance": 0.01
        },
        {
            "name": "TAF Fee Calculation", 
            "params": {"symbol": "MSFT", "quantity": 500, "price": 300.0, "side": "buy"},
            "exchange": ExchangeType.NASDAQ,
            "expected_taf_fee": 0.595,  # 500 shares * $0.119/100
            "tolerance": 0.01
        }
    ]
    
    all_passed = True
    
    for test in test_cases:
        engine = create_execution_engine(test["exchange"])
        result = engine.calculate_total_costs(**test["params"])
        
        if "expected_sec_fee" in test:
            actual = result["fee_breakdown"]["sec_fee"]
            expected = test["expected_sec_fee"]
            passed = abs(actual - expected) <= test["tolerance"]
            
            print(f"{test['name']}: {'✓' if passed else '✗'}")
            print(f"  Expected: ${expected:.2f}, Actual: ${actual:.2f}")
            
            if not passed:
                all_passed = False
        
        if "expected_taf_fee" in test:
            actual = result["fee_breakdown"]["taf_fee"] 
            expected = test["expected_taf_fee"]
            passed = abs(actual - expected) <= test["tolerance"]
            
            print(f"{test['name']}: {'✓' if passed else '✗'}")
            print(f"  Expected: ${expected:.2f}, Actual: ${actual:.2f}")
            
            if not passed:
                all_passed = False
    
    print(f"\nOverall Test Result: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    return all_passed

# Main execution
async def main():
    """Main execution function for demonstration"""
    demo = ExecutionModelingDemo()
    
    try:
        await demo.run_comprehensive_demo()
        print()
        await test_execution_engine_accuracy()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())