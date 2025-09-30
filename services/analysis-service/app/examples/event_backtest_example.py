"""
Event-Aware Backtesting Engine Example and Demonstration

This example demonstrates comprehensive event-driven backtesting capabilities
with sophisticated stop-loss management and performance attribution.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

from ..services.event_aware_backtest_engine import (
    EventAwareBacktestEngine,
    BacktestEvent,
    EventType,
    StopLossConfig,
    StopLossType,
    TradeDirection
)

class EventBacktestDemo:
    """Comprehensive demonstration of event-aware backtesting capabilities"""
    
    def __init__(self):
        self.engine = EventAwareBacktestEngine()
        
    async def run_all_demos(self):
        """Run all event-aware backtesting demonstrations"""
        print("=== Event-Aware Backtesting Engine Demonstration ===\n")
        
        print("1. Running Earnings Strategy Backtest...")
        await self.demo_earnings_strategy()
        print("\n" + "="*60 + "\n")
        
        print("2. Running FDA Approval Strategy Backtest...")
        await self.demo_fda_strategy()
        print("\n" + "="*60 + "\n")
        
        print("3. Running M&A Arbitrage Strategy Backtest...")
        await self.demo_ma_arbitrage()
        print("\n" + "="*60 + "\n")
        
        print("4. Running Stop Loss Optimization Demo...")
        await self.demo_stop_loss_optimization()
        print("\n" + "="*60 + "\n")
        
        print("5. Running Mixed Event Portfolio Backtest...")
        await self.demo_mixed_event_portfolio()
        print("\n" + "="*60 + "\n")
        
        print("6. Running Risk Management Analysis...")
        await self.demo_risk_management()
        print("\n" + "="*60 + "\n")
        
        print("All event-aware backtesting demonstrations completed successfully!")
        
    async def demo_earnings_strategy(self):
        """Demonstrate earnings-focused event strategy"""
        print("Earnings Strategy: Conservative Approach with Tight Stops")
        print("-" * 50)
        
        # Create earnings events with various surprise magnitudes
        events = [
            BacktestEvent(
                timestamp=datetime(2023, 1, 15, 16, 30),  # After market close
                symbol="AAPL",
                event_type=EventType.EARNINGS,
                event_description="Q4 2022 earnings beat by 8%",
                surprise_magnitude=0.08,
                confidence_score=0.85,
                market_cap=2800000000000,  # $2.8T
                sector="Technology",
                expected_move=0.05,
                actual_move_1d=0.06,
                actual_move_3d=0.08,
                actual_move_5d=0.04
            ),
            BacktestEvent(
                timestamp=datetime(2023, 2, 2, 16, 30),
                symbol="MSFT",
                event_type=EventType.EARNINGS,
                event_description="Q1 2023 earnings miss by 3%",
                surprise_magnitude=-0.03,
                confidence_score=0.7,
                market_cap=2400000000000,
                sector="Technology",
                expected_move=0.04,
                actual_move_1d=-0.04,
                actual_move_3d=-0.06,
                actual_move_5d=-0.02
            ),
            BacktestEvent(
                timestamp=datetime(2023, 2, 15, 16, 30),
                symbol="GOOGL",
                event_type=EventType.EARNINGS,
                event_description="Q4 2022 strong earnings beat",
                surprise_magnitude=0.12,
                confidence_score=0.9,
                market_cap=1500000000000,
                sector="Technology",
                expected_move=0.06,
                actual_move_1d=0.09,
                actual_move_3d=0.11,
                actual_move_5d=0.08
            )
        ]
        
        # Custom stop loss for earnings (conservative)
        earnings_stop_config = {
            EventType.EARNINGS: StopLossConfig(
                stop_type=StopLossType.VOLATILITY_ADJUSTED,
                base_percentage=0.06,      # 6% base stop (tight)
                volatility_multiplier=1.2, # Conservative multiplier
                max_stop_loss=0.12,        # Max 12% stop
                min_stop_loss=0.03,        # Min 3% stop
                time_decay_factor=0.02     # Tighten 2% per day
            )
        }
        
        # Generate sample price data
        price_data = self.generate_realistic_price_data(events)
        
        # Run backtest
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 3, 31)
        
        results = await self.engine.run_backtest(
            events=events,
            price_data=price_data,
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            position_sizing='kelly',  # Kelly criterion for earnings
            max_positions=5,
            stop_loss_config=earnings_stop_config,
            include_costs=True,
            commission_rate=0.001,
            slippage_bps=5.0
        )
        
        self.print_backtest_results(results, "Earnings Strategy")
        
    async def demo_fda_strategy(self):
        """Demonstrate FDA approval event strategy"""
        print("FDA Approval Strategy: High-Risk/High-Reward Binary Events")
        print("-" * 50)
        
        # FDA approval events (binary outcomes)
        events = [
            BacktestEvent(
                timestamp=datetime(2023, 1, 20, 8, 0),   # Pre-market announcement
                symbol="MRNA",
                event_type=EventType.FDA_APPROVAL,
                event_description="COVID vaccine booster approval",
                surprise_magnitude=0.45,  # Unexpected approval
                confidence_score=0.75,
                market_cap=65000000000,
                sector="Biotech",
                expected_move=0.25,
                actual_move_1d=0.32,
                actual_move_3d=0.28,
                actual_move_5d=0.22
            ),
            BacktestEvent(
                timestamp=datetime(2023, 2, 10, 7, 30),
                symbol="GILD",
                event_type=EventType.FDA_APPROVAL,
                event_description="Cancer drug approval denied",
                surprise_magnitude=-0.60,  # Unexpected rejection
                confidence_score=0.8,
                market_cap=85000000000,
                sector="Biotech",
                expected_move=0.30,
                actual_move_1d=-0.18,
                actual_move_3d=-0.15,
                actual_move_5d=-0.12
            ),
            BacktestEvent(
                timestamp=datetime(2023, 3, 5, 9, 0),
                symbol="BNTX",
                event_type=EventType.FDA_APPROVAL,
                event_description="New vaccine platform approved",
                surprise_magnitude=0.55,
                confidence_score=0.65,
                market_cap=25000000000,
                sector="Biotech",
                expected_move=0.35,
                actual_move_1d=0.42,
                actual_move_3d=0.38,
                actual_move_5d=0.35
            )
        ]
        
        # FDA-specific stop loss (wider stops for binary events)
        fda_stop_config = {
            EventType.FDA_APPROVAL: StopLossConfig(
                stop_type=StopLossType.EVENT_SPECIFIC,
                base_percentage=0.15,       # 15% base stop (wide)
                volatility_multiplier=2.0,  # High volatility adjustment
                max_stop_loss=0.30,         # Max 30% stop
                min_stop_loss=0.08,         # Min 8% stop
                trailing_threshold=0.20     # Trail after 20% gain
            )
        }
        
        price_data = self.generate_realistic_price_data(events, base_volatility=0.04)  # Higher volatility
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 3, 31)
        
        results = await self.engine.run_backtest(
            events=events,
            price_data=price_data,
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            position_sizing='equal_weight',
            max_positions=3,  # Fewer positions due to higher risk
            stop_loss_config=fda_stop_config,
            include_costs=True,
            commission_rate=0.001,
            slippage_bps=8.0  # Higher slippage for biotech
        )
        
        self.print_backtest_results(results, "FDA Approval Strategy")
        
    async def demo_ma_arbitrage(self):
        """Demonstrate M&A arbitrage strategy"""
        print("M&A Arbitrage Strategy: Deal Risk Management")
        print("-" * 50)
        
        # M&A announcement events
        events = [
            BacktestEvent(
                timestamp=datetime(2023, 1, 12, 7, 45),
                symbol="ATVI",
                event_type=EventType.MERGER_ACQUISITION,
                event_description="Microsoft acquisition announcement",
                surprise_magnitude=0.38,  # Large premium
                confidence_score=0.95,
                market_cap=75000000000,
                sector="Gaming",
                expected_move=0.25,
                actual_move_1d=0.28,
                actual_move_3d=0.26,
                actual_move_5d=0.25
            ),
            BacktestEvent(
                timestamp=datetime(2023, 2, 8, 8, 30),
                symbol="CRM",
                event_type=EventType.MERGER_ACQUISITION,
                event_description="Private equity buyout rumor",
                surprise_magnitude=0.15,
                confidence_score=0.6,  # Lower confidence (rumor)
                market_cap=200000000000,
                sector="Software",
                expected_move=0.12,
                actual_move_1d=0.08,  # Rumor fades
                actual_move_3d=0.03,
                actual_move_5d=-0.02
            ),
            BacktestEvent(
                timestamp=datetime(2023, 3, 15, 9, 0),
                symbol="VMW",
                event_type=EventType.MERGER_ACQUISITION,
                event_description="Broadcom acquisition confirmed",
                surprise_magnitude=0.42,
                confidence_score=0.88,
                market_cap=50000000000,
                sector="Technology",
                expected_move=0.30,
                actual_move_1d=0.35,
                actual_move_3d=0.33,
                actual_move_5d=0.31
            )
        ]
        
        # M&A-specific stop loss (tight for deal risk)
        ma_stop_config = {
            EventType.MERGER_ACQUISITION: StopLossConfig(
                stop_type=StopLossType.FIXED_PERCENTAGE,
                base_percentage=0.04,       # 4% tight stop
                volatility_multiplier=1.0,
                max_stop_loss=0.08,         # Max 8% stop
                min_stop_loss=0.02,         # Min 2% stop
                time_decay_factor=0.005     # Slight daily tightening
            )
        }
        
        price_data = self.generate_realistic_price_data(events, base_volatility=0.015)  # Lower volatility
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 3, 31)
        
        results = await self.engine.run_backtest(
            events=events,
            price_data=price_data,
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            position_sizing='equal_weight',
            max_positions=8,  # More positions (lower individual risk)
            stop_loss_config=ma_stop_config,
            include_costs=True,
            commission_rate=0.0005,  # Lower commissions for arb
            slippage_bps=3.0
        )
        
        self.print_backtest_results(results, "M&A Arbitrage Strategy")
        
    async def demo_stop_loss_optimization(self):
        """Demonstrate stop loss parameter optimization"""
        print("Stop Loss Optimization: Finding Optimal Parameters")
        print("-" * 50)
        
        # Create test events for optimization
        events = [
            BacktestEvent(
                timestamp=datetime(2023, 1, 15, 16, 30),
                symbol="TEST1",
                event_type=EventType.EARNINGS,
                event_description="Test earnings event 1",
                surprise_magnitude=0.10,
                confidence_score=0.8,
                actual_move_1d=0.08,
                actual_move_3d=0.06
            ),
            BacktestEvent(
                timestamp=datetime(2023, 2, 15, 16, 30),
                symbol="TEST2",
                event_type=EventType.EARNINGS,
                event_description="Test earnings event 2",
                surprise_magnitude=-0.05,
                confidence_score=0.7,
                actual_move_1d=-0.06,
                actual_move_3d=-0.04
            ),
            BacktestEvent(
                timestamp=datetime(2023, 3, 15, 16, 30),
                symbol="TEST3",
                event_type=EventType.EARNINGS,
                event_description="Test earnings event 3",
                surprise_magnitude=0.15,
                confidence_score=0.9,
                actual_move_1d=0.12,
                actual_move_3d=0.14
            )
        ]
        
        price_data = self.generate_realistic_price_data(events)
        
        # Test different stop loss parameters
        stop_parameters = [
            {"base_percentage": 0.05, "volatility_multiplier": 1.0},
            {"base_percentage": 0.08, "volatility_multiplier": 1.5},
            {"base_percentage": 0.10, "volatility_multiplier": 2.0},
            {"base_percentage": 0.12, "volatility_multiplier": 2.5}
        ]
        
        print("Testing stop loss parameter combinations:")
        print("Base% | Vol Mult | Total Return | Sharpe | Max DD | Win Rate")
        print("-" * 55)
        
        best_sharpe = -float('inf')
        best_params = None
        
        for params in stop_parameters:
            stop_config = {
                EventType.EARNINGS: StopLossConfig(
                    stop_type=StopLossType.VOLATILITY_ADJUSTED,
                    base_percentage=params["base_percentage"],
                    volatility_multiplier=params["volatility_multiplier"],
                    max_stop_loss=0.20,
                    min_stop_loss=0.03
                )
            }
            
            results = await self.engine.run_backtest(
                events=events,
                price_data=price_data,
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 3, 31),
                initial_capital=100000.0,
                position_sizing='equal_weight',
                max_positions=5,
                stop_loss_config=stop_config,
                include_costs=True
            )
            
            print(f"{params['base_percentage']:.1%} | {params['volatility_multiplier']:.1f}     | "
                  f"{results.total_return:>8.1%} | {results.sharpe_ratio:>5.2f} | "
                  f"{results.max_drawdown:>6.1%} | {results.win_rate:>7.1%}")
            
            if results.sharpe_ratio > best_sharpe:
                best_sharpe = results.sharpe_ratio
                best_params = params
        
        print(f"\nOptimal Parameters:")
        print(f"  Base Stop Loss: {best_params['base_percentage']:.1%}")
        print(f"  Volatility Multiplier: {best_params['volatility_multiplier']:.1f}")
        print(f"  Best Sharpe Ratio: {best_sharpe:.2f}")
        
    async def demo_mixed_event_portfolio(self):
        """Demonstrate mixed event type portfolio"""
        print("Mixed Event Portfolio: Diversified Event-Driven Strategy")
        print("-" * 50)
        
        # Mix of different event types
        events = [
            # Earnings events
            BacktestEvent(
                timestamp=datetime(2023, 1, 15, 16, 30),
                symbol="AAPL",
                event_type=EventType.EARNINGS,
                event_description="Q4 earnings beat",
                surprise_magnitude=0.08,
                confidence_score=0.85,
                actual_move_1d=0.06
            ),
            BacktestEvent(
                timestamp=datetime(2023, 2, 15, 16, 30),
                symbol="MSFT",
                event_type=EventType.EARNINGS,
                event_description="Q1 earnings",
                surprise_magnitude=0.05,
                confidence_score=0.8,
                actual_move_1d=0.04
            ),
            # FDA events
            BacktestEvent(
                timestamp=datetime(2023, 1, 25, 8, 0),
                symbol="PFE",
                event_type=EventType.FDA_APPROVAL,
                event_description="Drug approval",
                surprise_magnitude=0.35,
                confidence_score=0.7,
                actual_move_1d=0.28
            ),
            # M&A events
            BacktestEvent(
                timestamp=datetime(2023, 2, 5, 9, 0),
                symbol="TWTR",
                event_type=EventType.MERGER_ACQUISITION,
                event_description="Acquisition announcement",
                surprise_magnitude=0.40,
                confidence_score=0.9,
                actual_move_1d=0.35
            ),
            # Product launch
            BacktestEvent(
                timestamp=datetime(2023, 3, 10, 10, 0),
                symbol="TSLA",
                event_type=EventType.PRODUCT_LAUNCH,
                event_description="New model launch",
                surprise_magnitude=0.12,
                confidence_score=0.75,
                actual_move_1d=0.09
            )
        ]
        
        # Use default stop loss configurations (different for each event type)
        price_data = self.generate_realistic_price_data(events)
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 3, 31)
        
        results = await self.engine.run_backtest(
            events=events,
            price_data=price_data,
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            position_sizing='equal_weight',
            max_positions=8,
            stop_loss_config=None,  # Use defaults
            include_costs=True
        )
        
        self.print_backtest_results(results, "Mixed Event Portfolio")
        self.print_event_type_analysis(results)
        
    async def demo_risk_management(self):
        """Demonstrate risk management features"""
        print("Risk Management Analysis: Drawdown and Risk Controls")
        print("-" * 50)
        
        # Create events with some losses to show risk management
        events = [
            BacktestEvent(
                timestamp=datetime(2023, 1, 10, 16, 30),
                symbol="RISK1",
                event_type=EventType.EARNINGS,
                event_description="Disappointing earnings",
                surprise_magnitude=-0.08,
                confidence_score=0.6,
                actual_move_1d=-0.12  # Worse than expected
            ),
            BacktestEvent(
                timestamp=datetime(2023, 1, 15, 16, 30),
                symbol="RISK2",
                event_type=EventType.EARNINGS,
                event_description="Major earnings miss",
                surprise_magnitude=-0.15,
                confidence_score=0.7,
                actual_move_1d=-0.20  # Large loss
            ),
            BacktestEvent(
                timestamp=datetime(2023, 2, 1, 8, 0),
                symbol="RISK3",
                event_type=EventType.FDA_APPROVAL,
                event_description="FDA rejection",
                surprise_magnitude=-0.50,
                confidence_score=0.8,
                actual_move_1d=-0.35  # Binary event loss
            ),
            BacktestEvent(
                timestamp=datetime(2023, 2, 10, 16, 30),
                symbol="RISK4",
                event_type=EventType.EARNINGS,
                event_description="Good earnings",
                surprise_magnitude=0.10,
                confidence_score=0.85,
                actual_move_1d=0.08   # Recovery trade
            ),
            BacktestEvent(
                timestamp=datetime(2023, 2, 20, 9, 0),
                symbol="RISK5",
                event_type=EventType.MERGER_ACQUISITION,
                event_description="Successful acquisition",
                surprise_magnitude=0.25,
                confidence_score=0.9,
                actual_move_1d=0.22   # Good trade
            )
        ]
        
        # Conservative stop loss for risk demo
        risk_stop_config = {
            EventType.EARNINGS: StopLossConfig(
                stop_type=StopLossType.VOLATILITY_ADJUSTED,
                base_percentage=0.07,  # Tighter stops
                volatility_multiplier=1.3,
                max_stop_loss=0.12,
                min_stop_loss=0.04
            ),
            EventType.FDA_APPROVAL: StopLossConfig(
                stop_type=StopLossType.EVENT_SPECIFIC,
                base_percentage=0.12,
                volatility_multiplier=1.8,
                max_stop_loss=0.25,
                min_stop_loss=0.06
            ),
            EventType.MERGER_ACQUISITION: StopLossConfig(
                stop_type=StopLossType.FIXED_PERCENTAGE,
                base_percentage=0.05,
                volatility_multiplier=1.0,
                max_stop_loss=0.08,
                min_stop_loss=0.03
            )
        }
        
        price_data = self.generate_realistic_price_data(events)
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 2, 28)
        
        results = await self.engine.run_backtest(
            events=events,
            price_data=price_data,
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            position_sizing='equal_weight',
            max_positions=5,
            stop_loss_config=risk_stop_config,
            include_costs=True
        )
        
        self.print_risk_analysis(results)
        
    def generate_realistic_price_data(self, events: List[BacktestEvent], base_volatility: float = 0.02) -> Dict[str, pd.DataFrame]:
        """Generate realistic price data for backtesting"""
        
        price_data = {}
        symbols = list(set(event.symbol for event in events))
        
        # Determine date range
        start_date = min(event.timestamp for event in events) - timedelta(days=30)
        end_date = max(event.timestamp for event in events) + timedelta(days=30)
        
        for symbol in symbols:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Simulate price movement
            np.random.seed(hash(symbol) % 2**32)
            initial_price = np.random.uniform(50, 300)  # Random starting price
            
            prices = [initial_price]
            
            for i, date in enumerate(dates[1:], 1):
                # Base random walk
                daily_return = np.random.normal(0.0005, base_volatility)  # Slight upward drift
                
                # Check for events on this date
                symbol_events = [e for e in events if e.symbol == symbol and e.timestamp.date() == date.date()]
                
                if symbol_events:
                    # Apply event impact
                    event = symbol_events[0]
                    if event.actual_move_1d:
                        # Use actual move if provided
                        daily_return = event.actual_move_1d
                    else:
                        # Simulate based on surprise magnitude
                        event_impact = event.surprise_magnitude * np.random.uniform(0.8, 1.2)
                        daily_return = event_impact
                
                new_price = prices[-1] * (1 + daily_return)
                prices.append(max(1.0, new_price))  # Prevent negative prices
            
            # Create OHLCV data
            df_data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                daily_vol = abs(np.random.normal(0, base_volatility * 0.5))
                high = price * (1 + daily_vol)
                low = price * (1 - daily_vol)
                
                # Ensure OHLC consistency
                open_price = price
                close_price = price
                
                # Adjust based on daily move
                if i > 0:
                    prev_close = df_data[i-1]['close']
                    actual_return = (price - prev_close) / prev_close
                    close_price = prev_close * (1 + actual_return)
                
                volume = max(100000, int(np.random.normal(1000000, 300000)))
                
                df_data.append({
                    'open': open_price,
                    'high': max(open_price, high, close_price),
                    'low': min(open_price, low, close_price),
                    'close': close_price,
                    'volume': volume
                })
            
            df = pd.DataFrame(df_data, index=dates)
            price_data[symbol] = df
        
        return price_data
    
    def print_backtest_results(self, results, strategy_name: str):
        """Print formatted backtest results"""
        print(f"\n{strategy_name} Results:")
        print("=" * 40)
        print(f"Total Return: {results.total_return:.1%}")
        print(f"Annual Return: {results.annual_return:.1%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {results.sortino_ratio:.2f}")
        print(f"Maximum Drawdown: {results.max_drawdown:.1%}")
        print(f"Win Rate: {results.win_rate:.1%}")
        print(f"Profit Factor: {results.profit_factor:.2f}")
        print()
        print("Trade Statistics:")
        print(f"  Total Trades: {results.total_trades}")
        print(f"  Winning Trades: {results.winning_trades}")
        print(f"  Losing Trades: {results.losing_trades}")
        print(f"  Average Win: ${results.average_win:,.2f}")
        print(f"  Average Loss: ${results.average_loss:,.2f}")
        print(f"  Largest Win: ${results.largest_win:,.2f}")
        print(f"  Largest Loss: ${results.largest_loss:,.2f}")
        print()
        print("Risk Metrics:")
        print(f"  VaR (95%): {results.var_95:.1%}")
        print(f"  Expected Shortfall: {results.expected_shortfall:.1%}")
        print(f"  Max Consecutive Losses: {results.maximum_consecutive_losses}")
        print(f"  Average Holding Period: {results.average_holding_period:.1f} hours")
        
    def print_event_type_analysis(self, results):
        """Print event type performance breakdown"""
        print("\nEvent Type Performance Analysis:")
        print("=" * 40)
        
        for event_type, performance in results.event_type_performance.items():
            print(f"\n{event_type.upper()}:")
            print(f"  Total Trades: {performance['trade_count']}")
            print(f"  Total P&L: ${performance['total_pnl']:,.2f}")
            print(f"  Win Rate: {performance['win_rate']:.1%}")
            print(f"  Average Return: {performance['average_return']:.1%}")
    
    def print_risk_analysis(self, results):
        """Print detailed risk analysis"""
        print("\nRisk Management Analysis:")
        print("=" * 40)
        print(f"Portfolio Performance: {results.total_return:.1%}")
        print(f"Maximum Drawdown: {results.max_drawdown:.1%}")
        print(f"Risk-Adjusted Return: {results.sharpe_ratio:.2f}")
        print()
        
        # Analyze trade exits
        stop_loss_exits = len([t for t in results.trades if t.exit_reason and 'stop_loss' in t.exit_reason.value])
        profit_exits = len([t for t in results.trades if t.exit_reason and 'profit' in t.exit_reason.value])
        time_exits = len([t for t in results.trades if t.exit_reason and 'time' in t.exit_reason.value])
        
        print("Exit Reason Analysis:")
        print(f"  Stop Loss Exits: {stop_loss_exits} ({stop_loss_exits/max(1,results.total_trades):.1%})")
        print(f"  Profit Taking Exits: {profit_exits} ({profit_exits/max(1,results.total_trades):.1%})")
        print(f"  Time Decay Exits: {time_exits} ({time_exits/max(1,results.total_trades):.1%})")
        print()
        
        # Risk control effectiveness
        if results.trades:
            avg_loss = abs(results.average_loss)
            avg_win = results.average_win
            
            print("Risk Control Effectiveness:")
            print(f"  Average Loss Magnitude: ${avg_loss:,.2f}")
            print(f"  Loss Control Rating: {'EXCELLENT' if avg_loss < 5000 else 'GOOD' if avg_loss < 8000 else 'NEEDS IMPROVEMENT'}")
            print(f"  Win/Loss Ratio: {avg_win/max(1,avg_loss):.2f}")

async def main():
    """Run event-aware backtesting demonstrations"""
    demo = EventBacktestDemo()
    await demo.run_all_demos()

if __name__ == "__main__":
    asyncio.run(main())