"""
Gap Trading Engine Example and Demonstration

This example demonstrates comprehensive gap trading analysis capabilities
including continuation vs fade detection and pre/post-market price handling.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional
import numpy as np

from ..services.gap_trading_engine import (
    GapTradingEngine,
    GapType,
    GapSize,
    GapDirection,
    MarketSession,
    PrePostMarketData
)

class GapTradingDemo:
    """Comprehensive demonstration of gap trading analysis capabilities"""
    
    def __init__(self):
        self.engine = GapTradingEngine()
        
    async def run_all_demos(self):
        """Run all gap trading demonstrations"""
        print("=== Gap Trading Engine Demonstration ===\n")
        
        print("1. Running Large Gap Up Continuation Demo...")
        await self.demo_large_gap_up_continuation()
        print("\n" + "="*60 + "\n")
        
        print("2. Running Small Gap Fade Demo...")
        await self.demo_small_gap_fade()
        print("\n" + "="*60 + "\n")
        
        print("3. Running Earnings Gap Analysis Demo...")
        await self.demo_earnings_gap_analysis()
        print("\n" + "="*60 + "\n")
        
        print("4. Running Pre-Market Context Demo...")
        await self.demo_pre_market_context()
        print("\n" + "="*60 + "\n")
        
        print("5. Running Overnight Gap Analysis Demo...")
        await self.demo_overnight_gap_analysis()
        print("\n" + "="*60 + "\n")
        
        print("6. Running Gap Monitoring Demo...")
        await self.demo_gap_monitoring()
        print("\n" + "="*60 + "\n")
        
        print("7. Running Volume Surge Gap Demo...")
        await self.demo_volume_surge_gap()
        print("\n" + "="*60 + "\n")
        
        print("All gap trading demonstrations completed successfully!")
        
    async def demo_large_gap_up_continuation(self):
        """Demonstrate large gap up with continuation pattern"""
        print("Large Gap Up Continuation: Tech Stock Breakthrough")
        print("-" * 50)
        
        # Large gap up scenario
        previous_close = 145.80
        current_open = 167.25  # 14.7% gap up
        
        # Strong volume surge
        volume_data = {
            'current_volume': 15000000,  # 10x normal volume
            'average_volume': 1500000
        }
        
        # Pre-market strength
        pre_market_data = {
            'open': 146.50,
            'high': 168.75,
            'low': 146.20,
            'close': 167.25,
            'volume': 2500000,
            'session_start': datetime.utcnow() - timedelta(hours=5, minutes=30),
            'session_end': datetime.utcnow(),
            'trade_count': 8500,
            'spread_avg': 0.02,
            'vwap': 158.90
        }
        
        # News catalyst
        news_events = [
            {
                'event_type': 'product_launch',
                'description': 'Revolutionary AI chip breakthrough announcement',
                'timestamp': datetime.utcnow() - timedelta(hours=12),
                'impact_score': 9.5,
                'source': 'company_announcement'
            }
        ]
        
        gap_analysis = await self.engine.analyze_gap(
            symbol="TECH_BREAKTHROUGH",
            current_price_data={'price': current_open},
            previous_close=previous_close,
            pre_market_data=pre_market_data,
            overnight_data=None,
            volume_data=volume_data,
            news_events=news_events
        )
        
        self.print_gap_analysis(gap_analysis, "Large gap up with strong fundamentals")
        
    async def demo_small_gap_fade(self):
        """Demonstrate small gap that tends to fade"""
        print("Small Gap Fade: Random Market Noise")
        print("-" * 50)
        
        # Small gap down scenario
        previous_close = 89.45
        current_open = 87.30  # 2.4% gap down
        
        # Normal volume
        volume_data = {
            'current_volume': 1200000,  # Only slightly above average
            'average_volume': 950000
        }
        
        # Weak pre-market activity
        pre_market_data = {
            'open': 89.20,
            'high': 89.30,
            'low': 87.15,
            'close': 87.30,
            'volume': 150000,  # Low pre-market volume
            'session_start': datetime.utcnow() - timedelta(hours=5, minutes=30),
            'session_end': datetime.utcnow(),
            'trade_count': 850,
            'spread_avg': 0.05,
            'vwap': 88.12
        }
        
        # No significant news
        news_events = []
        
        gap_analysis = await self.engine.analyze_gap(
            symbol="RANDOM_NOISE",
            current_price_data={'price': current_open},
            previous_close=previous_close,
            pre_market_data=pre_market_data,
            overnight_data=None,
            volume_data=volume_data,
            news_events=news_events
        )
        
        self.print_gap_analysis(gap_analysis, "Small gap likely to fade")
        
    async def demo_earnings_gap_analysis(self):
        """Demonstrate earnings-related gap analysis"""
        print("Earnings Gap Analysis: Biotech Earnings Beat")
        print("-" * 50)
        
        # Medium earnings gap
        previous_close = 78.90
        current_open = 92.15  # 16.8% gap up
        
        # Earnings-driven volume
        volume_data = {
            'current_volume': 8500000,  # 5x normal earnings volume
            'average_volume': 1700000
        }
        
        # Limited pre-market (earnings released after hours)
        pre_market_data = {
            'open': 79.50,
            'high': 93.80,
            'low': 79.45,
            'close': 92.15,
            'volume': 1800000,
            'session_start': datetime.utcnow() - timedelta(hours=5, minutes=30),
            'session_end': datetime.utcnow(),
            'trade_count': 4200,
            'spread_avg': 0.08,
            'vwap': 86.50
        }
        
        # Earnings announcement
        news_events = [
            {
                'event_type': 'earnings',
                'description': 'Q3 earnings beat by 45%, raised guidance',
                'timestamp': datetime.utcnow() - timedelta(hours=16),
                'impact_score': 8.7,
                'source': 'earnings_release'
            }
        ]
        
        gap_analysis = await self.engine.analyze_gap(
            symbol="BIOTECH_EARNINGS",
            current_price_data={'price': current_open},
            previous_close=previous_close,
            pre_market_data=pre_market_data,
            overnight_data=None,
            volume_data=volume_data,
            news_events=news_events
        )
        
        self.print_gap_analysis(gap_analysis, "Earnings-driven gap with strong fundamentals")
        
    async def demo_pre_market_context(self):
        """Demonstrate importance of pre-market context"""
        print("Pre-Market Context Analysis: Building Momentum")
        print("-" * 50)
        
        # Medium gap with strong pre-market buildup
        previous_close = 156.30
        current_open = 171.85  # 9.9% gap up
        
        # Good volume
        volume_data = {
            'current_volume': 4500000,
            'average_volume': 1800000
        }
        
        # Strong pre-market momentum
        pre_market_data = {
            'open': 158.25,
            'high': 172.50,
            'low': 157.90,
            'close': 171.85,
            'volume': 950000,  # Strong pre-market volume
            'session_start': datetime.utcnow() - timedelta(hours=5, minutes=30),
            'session_end': datetime.utcnow(),
            'trade_count': 3200,
            'spread_avg': 0.03,
            'vwap': 165.20
        }
        
        # Analyst upgrade
        news_events = [
            {
                'event_type': 'analyst_upgrade',
                'description': 'Major bank upgrades to Strong Buy, $200 target',
                'timestamp': datetime.utcnow() - timedelta(hours=8),
                'impact_score': 7.2,
                'source': 'analyst_research'
            }
        ]
        
        gap_analysis = await self.engine.analyze_gap(
            symbol="MOMENTUM_BUILD",
            current_price_data={'price': current_open},
            previous_close=previous_close,
            pre_market_data=pre_market_data,
            overnight_data=None,
            volume_data=volume_data,
            news_events=news_events
        )
        
        self.print_gap_analysis(gap_analysis, "Strong pre-market momentum supporting gap")
        
    async def demo_overnight_gap_analysis(self):
        """Demonstrate overnight session analysis"""
        print("Overnight Gap Analysis: International News Impact")
        print("-" * 50)
        
        # Gap based on overnight international developments
        previous_close = 234.60
        current_open = 218.75  # 6.8% gap down
        
        # Moderate volume
        volume_data = {
            'current_volume': 3200000,
            'average_volume': 2100000
        }
        
        # Overnight selling pressure
        overnight_data = {
            'open': 234.20,
            'high': 234.50,
            'low': 216.80,
            'close': 218.75,
            'volume': 420000,
            'session_start': datetime.utcnow() - timedelta(hours=12),
            'session_end': datetime.utcnow() - timedelta(hours=5, minutes=30),
            'trade_count': 1200,
            'spread_avg': 0.15,
            'vwap': 225.30
        }
        
        # International regulatory news
        news_events = [
            {
                'event_type': 'regulatory',
                'description': 'European regulators announce investigation',
                'timestamp': datetime.utcnow() - timedelta(hours=10),
                'impact_score': 6.8,
                'source': 'international_news'
            }
        ]
        
        gap_analysis = await self.engine.analyze_gap(
            symbol="INTERNATIONAL_IMPACT",
            current_price_data={'price': current_open},
            previous_close=previous_close,
            pre_market_data=None,
            overnight_data=overnight_data,
            volume_data=volume_data,
            news_events=news_events
        )
        
        self.print_gap_analysis(gap_analysis, "Overnight gap from international developments")
        
    async def demo_gap_monitoring(self):
        """Demonstrate real-time gap monitoring"""
        print("Gap Monitoring: Real-Time Behavior Tracking")
        print("-" * 50)
        
        # Create initial gap analysis
        previous_close = 125.40
        current_open = 139.85  # 11.5% gap up
        
        gap_analysis = await self.engine.analyze_gap(
            symbol="MONITORING_DEMO",
            current_price_data={'price': current_open},
            previous_close=previous_close,
            pre_market_data=None,
            overnight_data=None,
            volume_data={'current_volume': 5000000, 'average_volume': 2000000},
            news_events=None
        )
        
        print(f"Initial Gap: {gap_analysis.gap_percentage:.1%} up")
        print(f"Gap Fill Probability: {gap_analysis.gap_fill_probability:.1%}")
        print(f"Continuation Probability: {gap_analysis.continuation_probability:.1%}")
        print()
        
        # Monitor gap behavior over time
        time_intervals = [30, 60, 120, 240]  # minutes after open
        prices = [142.30, 136.75, 131.20, 128.90]  # price progression
        
        for i, (minutes, price) in enumerate(zip(time_intervals, prices)):
            print(f"After {minutes} minutes - Price: ${price:.2f}")
            
            # Simulate volume profile
            volume_profile = {
                'vwap': price + np.random.normal(0, 1),
                'poc': price - 0.50,
                'distribution': 0.6,
                'buying_pressure': 0.3 - (i * 0.1)  # Decreasing buying pressure
            }
            
            monitoring_result = await self.engine.monitor_gap_behavior(
                gap_analysis=gap_analysis,
                current_price_data={'price': price, 'volume': 1000000},
                volume_profile=volume_profile
            )
            
            print(f"  Direction: {monitoring_result.current_direction.value}")
            print(f"  Fill %: {monitoring_result.fill_percentage:.1%}")
            print(f"  Strength: {monitoring_result.price_action_strength:.2f}")
            print()
        
    async def demo_volume_surge_gap(self):
        """Demonstrate gap with extreme volume surge"""
        print("Volume Surge Gap: Massive Institutional Interest")
        print("-" * 50)
        
        # Medium gap with massive volume
        previous_close = 67.25
        current_open = 78.90  # 17.3% gap up
        
        # Extreme volume surge
        volume_data = {
            'current_volume': 45000000,  # 25x normal volume
            'average_volume': 1800000
        }
        
        # Strong pre-market with high volume
        pre_market_data = {
            'open': 68.50,
            'high': 82.15,
            'low': 68.20,
            'close': 78.90,
            'volume': 8500000,  # Massive pre-market volume
            'session_start': datetime.utcnow() - timedelta(hours=5, minutes=30),
            'session_end': datetime.utcnow(),
            'trade_count': 15000,
            'spread_avg': 0.01,  # Tight spreads due to liquidity
            'vwap': 74.80
        }
        
        # Major acquisition rumor
        news_events = [
            {
                'event_type': 'merger_acquisition',
                'description': 'Takeover rumors from major competitor',
                'timestamp': datetime.utcnow() - timedelta(hours=6),
                'impact_score': 9.2,
                'source': 'market_rumors'
            }
        ]
        
        gap_analysis = await self.engine.analyze_gap(
            symbol="VOLUME_SURGE",
            current_price_data={'price': current_open},
            previous_close=previous_close,
            pre_market_data=pre_market_data,
            overnight_data=None,
            volume_data=volume_data,
            news_events=news_events
        )
        
        self.print_gap_analysis(gap_analysis, "Massive volume surge supporting gap")
        
    def print_gap_analysis(self, gap_analysis, description: str):
        """Print gap analysis in a formatted way"""
        print(f"Analysis: {description}")
        print(f"Symbol: {gap_analysis.symbol}")
        print(f"Gap Type: {gap_analysis.gap_type.value}")
        print(f"Gap Size: {gap_analysis.gap_size.value}")
        print(f"Gap Percentage: {gap_analysis.gap_percentage:.1%}")
        print(f"Gap Points: ${gap_analysis.gap_points:.2f}")
        print()
        
        print("Price Levels:")
        print(f"  Previous Close: ${gap_analysis.previous_close:.2f}")
        print(f"  Current Open: ${gap_analysis.current_open:.2f}")
        print(f"  Gap High: ${gap_analysis.gap_high:.2f}")
        print(f"  Gap Low: ${gap_analysis.gap_low:.2f}")
        print()
        
        print("Volume Analysis:")
        print(f"  Volume Surge: {gap_analysis.volume_surge}")
        print(f"  Volume Ratio: {gap_analysis.volume_ratio:.1f}x")
        print()
        
        print("Catalyst Analysis:")
        print(f"  News Catalyst: {gap_analysis.news_catalyst}")
        print(f"  Earnings Gap: {gap_analysis.earnings_gap}")
        print()
        
        print("Probability Analysis:")
        print(f"  Gap Fill Probability: {gap_analysis.gap_fill_probability:.1%}")
        print(f"  Continuation Probability: {gap_analysis.continuation_probability:.1%}")
        print()
        
        print("Technical Levels:")
        print(f"  Support/Resistance: {[f'${level:.2f}' for level in gap_analysis.support_resistance_levels[:5]]}")
        print()
        
        print("Fibonacci Levels:")
        for level, price in list(gap_analysis.fibonacci_levels.items())[:4]:
            print(f"  {level}: ${price:.2f}")
        print()
        
        print("Trading Setup:")
        print(f"  Entry Price: ${gap_analysis.optimal_entry_price:.2f}")
        print(f"  Stop Loss: ${gap_analysis.stop_loss_level:.2f}")
        print(f"  Profit Targets: {[f'${target:.2f}' for target in gap_analysis.profit_targets]}")
        print(f"  Risk/Reward: {gap_analysis.risk_reward_ratio:.2f}")
        print()
        
        if gap_analysis.pre_market_data:
            print("Pre-Market Session:")
            print(f"  Volume: {gap_analysis.pre_market_data.volume:,}")
            print(f"  VWAP: ${gap_analysis.pre_market_data.vwap:.2f}")
            print(f"  Liquidity Score: {gap_analysis.pre_market_data.liquidity_score:.2f}")
            print()
        
        if gap_analysis.overnight_data:
            print("Overnight Session:")
            print(f"  Volume: {gap_analysis.overnight_data.volume:,}")
            print(f"  VWAP: ${gap_analysis.overnight_data.vwap:.2f}")
            print(f"  Liquidity Score: {gap_analysis.overnight_data.liquidity_score:.2f}")
            print()
        
        print("Risk Metrics:")
        print(f"  Volatility Context: {gap_analysis.volatility_context:.2f}")
        print(f"  Average Gap Size: {gap_analysis.average_gap_size:.1%}")
        print()
        
        # Trading recommendation
        if gap_analysis.continuation_probability > 0.7:
            recommendation = "STRONG CONTINUATION PLAY"
        elif gap_analysis.gap_fill_probability > 0.7:
            recommendation = "FADE/REVERSAL PLAY"
        elif gap_analysis.gap_size in [gap_analysis.gap_size.LARGE, gap_analysis.gap_size.MASSIVE]:
            recommendation = "MOMENTUM CONTINUATION"
        else:
            recommendation = "WAIT FOR CONFIRMATION"
            
        print(f"Trading Recommendation: {recommendation}")

async def main():
    """Run gap trading demonstrations"""
    demo = GapTradingDemo()
    await demo.run_all_demos()

if __name__ == "__main__":
    asyncio.run(main())