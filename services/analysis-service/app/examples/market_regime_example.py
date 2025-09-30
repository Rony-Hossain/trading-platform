"""
Market Regime Filter Example and Demonstration

This example demonstrates comprehensive market regime analysis and filtering
for event trade execution with various market conditions and scenarios.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

from ..services.market_regime_filter import (
    MarketRegimeFilter,
    MarketRegime,
    RegimeFavorability,
    EventType
)

class MarketRegimeDemo:
    """Comprehensive demonstration of market regime filtering capabilities"""
    
    def __init__(self):
        self.filter_engine = MarketRegimeFilter()
        
    async def run_all_demos(self):
        """Run all market regime filtering demonstrations"""
        print("=== Market Regime Filter Demonstration ===\n")
        
        print("1. Running Bull Market Demo...")
        await self.demo_bull_market_regime()
        print("\n" + "="*60 + "\n")
        
        print("2. Running Crisis Mode Demo...")
        await self.demo_crisis_mode_regime()
        print("\n" + "="*60 + "\n")
        
        print("3. Running High Volatility Demo...")
        await self.demo_high_volatility_regime()
        print("\n" + "="*60 + "\n")
        
        print("4. Running Low Volatility Demo...")
        await self.demo_low_volatility_regime()
        print("\n" + "="*60 + "\n")
        
        print("5. Running Sideways Market Demo...")
        await self.demo_sideways_market_regime()
        print("\n" + "="*60 + "\n")
        
        print("6. Running Event Type Filtering Demo...")
        await self.demo_event_type_filtering()
        print("\n" + "="*60 + "\n")
        
        print("7. Running Bulk Trade Evaluation Demo...")
        await self.demo_bulk_trade_evaluation()
        print("\n" + "="*60 + "\n")
        
        print("8. Running Risk Adjustment Demo...")
        await self.demo_risk_adjustment_scenarios()
        print("\n" + "="*60 + "\n")
        
        print("All market regime filtering demonstrations completed successfully!")
        
    async def demo_bull_market_regime(self):
        """Demonstrate bull market regime with favorable conditions"""
        print("Bull Market Regime: Highly Favorable Conditions")
        print("-" * 50)
        
        # Strong bull market conditions
        market_data = {
            'vix': 16.5,              # Low volatility
            'vix9d_vix_ratio': 1.05,  # Normal term structure
            'trend_score': 0.8,       # Strong uptrend
            'risk_appetite': 0.85,    # High risk appetite
            'vol_clustering': 0.2,    # Low clustering
            'sector_rotation_score': 0.7,  # Healthy rotation
            'credit_spreads': 85.0,   # Tight credit spreads
            'liquidity_score': 0.9,   # Excellent liquidity
            'correlation_level': 0.4, # Low correlation
            'regime_duration_days': 45
        }
        
        # Analyze regime
        regime_analysis = await self.filter_engine.analyze_market_regime(market_data)
        
        print(f"Primary Regime: {regime_analysis.primary_regime.value}")
        print(f"Regime Strength: {regime_analysis.regime_strength:.1%}")
        print(f"Overall Favorability: {regime_analysis.overall_favorability.value}")
        print(f"Volatility Percentile: {regime_analysis.volatility_percentile:.1%}")
        print(f"Tail Risk: {regime_analysis.tail_risk:.1%}")
        print()
        
        # Test different event types
        test_events = [
            (EventType.EARNINGS, "AAPL earnings beat"),
            (EventType.FDA_APPROVAL, "Biotech drug approval"),
            (EventType.MERGER_ACQUISITION, "Tech acquisition")
        ]
        
        for event_type, description in test_events:
            execution_decision = await self.filter_engine.evaluate_trade_execution(
                symbol="TEST",
                event_type=event_type,
                event_details={'description': description},
                regime_analysis=regime_analysis,
                risk_tolerance='moderate'
            )
            
            print(f"{event_type.value.upper()}: {description}")
            print(f"  Execution Approved: {execution_decision.execution_approved}")
            print(f"  Favorability Score: {execution_decision.favorability_score:.1%}")
            print(f"  Position Size Modifier: {execution_decision.position_size_modifier:.2f}x")
            if execution_decision.approval_reasons:
                print(f"  Reasons: {', '.join(execution_decision.approval_reasons[:2])}")
            print()
            
    async def demo_crisis_mode_regime(self):
        """Demonstrate crisis mode regime with unfavorable conditions"""
        print("Crisis Mode Regime: Highly Unfavorable Conditions")
        print("-" * 50)
        
        # Crisis market conditions
        market_data = {
            'vix': 42.0,              # Extreme volatility
            'vix9d_vix_ratio': 0.85,  # Inverted term structure
            'trend_score': -0.6,      # Strong downtrend
            'risk_appetite': 0.15,    # Flight to safety
            'vol_clustering': 0.9,    # High clustering
            'sector_rotation_score': 0.2,  # Poor rotation
            'credit_spreads': 350.0,  # Wide credit spreads
            'liquidity_score': 0.25,  # Poor liquidity
            'correlation_level': 0.85, # High correlation
            'regime_duration_days': 8
        }
        
        # Analyze regime
        regime_analysis = await self.filter_engine.analyze_market_regime(market_data)
        
        print(f"Primary Regime: {regime_analysis.primary_regime.value}")
        print(f"Regime Strength: {regime_analysis.regime_strength:.1%}")
        print(f"Overall Favorability: {regime_analysis.overall_favorability.value}")
        print(f"Volatility Percentile: {regime_analysis.volatility_percentile:.1%}")
        print(f"Tail Risk: {regime_analysis.tail_risk:.1%}")
        print(f"Correlation Risk: {regime_analysis.correlation_risk:.1%}")
        print()
        
        # Test event execution in crisis
        test_events = [
            (EventType.EARNINGS, "Strong earnings in crisis"),
            (EventType.FDA_APPROVAL, "Major drug approval"),
            (EventType.MERGER_ACQUISITION, "Defensive acquisition")
        ]
        
        for event_type, description in test_events:
            execution_decision = await self.filter_engine.evaluate_trade_execution(
                symbol="TEST",
                event_type=event_type,
                event_details={'description': description},
                regime_analysis=regime_analysis,
                risk_tolerance='moderate'
            )
            
            print(f"{event_type.value.upper()}: {description}")
            print(f"  Execution Approved: {execution_decision.execution_approved}")
            print(f"  Favorability Score: {execution_decision.favorability_score:.1%}")
            print(f"  Position Size Modifier: {execution_decision.position_size_modifier:.2f}x")
            if execution_decision.rejection_reasons:
                print(f"  Rejection Reasons: {', '.join(execution_decision.rejection_reasons[:2])}")
            if execution_decision.risk_mitigation_required:
                print(f"  Risk Mitigation: {', '.join(execution_decision.risk_mitigation_required[:2])}")
            print()
            
    async def demo_high_volatility_regime(self):
        """Demonstrate high volatility regime"""
        print("High Volatility Regime: Challenging Conditions")
        print("-" * 50)
        
        # High volatility conditions
        market_data = {
            'vix': 28.5,              # High volatility
            'vix9d_vix_ratio': 0.92,  # Backwardation
            'trend_score': 0.2,       # Weak uptrend
            'risk_appetite': 0.4,     # Mixed sentiment
            'vol_clustering': 0.75,   # High clustering
            'sector_rotation_score': 0.3,  # Poor rotation
            'credit_spreads': 180.0,  # Elevated spreads
            'liquidity_score': 0.6,   # Moderate liquidity
            'correlation_level': 0.7,  # High correlation
            'regime_duration_days': 12
        }
        
        regime_analysis = await self.filter_engine.analyze_market_regime(market_data)
        
        print(f"Primary Regime: {regime_analysis.primary_regime.value}")
        print(f"Overall Favorability: {regime_analysis.overall_favorability.value}")
        print(f"Breakout Potential: {regime_analysis.breakout_potential:.1%}")
        print(f"Mean Reversion Tendency: {regime_analysis.mean_reversion_tendency:.1%}")
        print()
        
        # Test biotech events (should handle volatility better)
        execution_decision = await self.filter_engine.evaluate_trade_execution(
            symbol="BIOTECH",
            event_type=EventType.FDA_APPROVAL,
            event_details={'description': 'Phase 3 trial success'},
            regime_analysis=regime_analysis,
            risk_tolerance='moderate'
        )
        
        print("FDA APPROVAL in High Vol Environment:")
        print(f"  Execution Approved: {execution_decision.execution_approved}")
        print(f"  Favorability Score: {execution_decision.favorability_score:.1%}")
        print(f"  Risk Adjustment Factor: {execution_decision.risk_adjustment_factor:.2f}")
        print(f"  Entry Timing: {execution_decision.recommended_entry_timing}")
        if execution_decision.regime_stop_loss:
            print(f"  Regime Stop Loss: {execution_decision.regime_stop_loss:.1%}")
        print()
        
    async def demo_low_volatility_regime(self):
        """Demonstrate low volatility regime with breakout potential"""
        print("Low Volatility Regime: Compressed Volatility Environment")
        print("-" * 50)
        
        # Low volatility with compression
        market_data = {
            'vix': 12.8,              # Very low volatility
            'vix9d_vix_ratio': 1.15,  # Contango
            'trend_score': 0.1,       # Sideways drift
            'risk_appetite': 0.65,    # Moderate risk appetite
            'vol_clustering': 0.8,    # High clustering (compression)
            'sector_rotation_score': 0.5,  # Normal rotation
            'credit_spreads': 95.0,   # Normal spreads
            'liquidity_score': 0.85,  # Good liquidity
            'correlation_level': 0.5,  # Moderate correlation
            'regime_duration_days': 25
        }
        
        regime_analysis = await self.filter_engine.analyze_market_regime(market_data)
        
        print(f"Primary Regime: {regime_analysis.primary_regime.value}")
        print(f"Overall Favorability: {regime_analysis.overall_favorability.value}")
        print(f"Breakout Potential: {regime_analysis.breakout_potential:.1%}")
        print(f"Volatility Percentile: {regime_analysis.volatility_percentile:.1%}")
        print()
        
        # Test M&A events (should be highly favorable in low vol)
        execution_decision = await self.filter_engine.evaluate_trade_execution(
            symbol="TARGET",
            event_type=EventType.MERGER_ACQUISITION,
            event_details={'description': 'Acquisition rumor'},
            regime_analysis=regime_analysis,
            risk_tolerance='moderate'
        )
        
        print("M&A EVENT in Low Vol Environment:")
        print(f"  Execution Approved: {execution_decision.execution_approved}")
        print(f"  Favorability Score: {execution_decision.favorability_score:.1%}")
        print(f"  Position Size Modifier: {execution_decision.position_size_modifier:.2f}x")
        if execution_decision.regime_profit_target_adjustment:
            print(f"  Profit Target Adjustment: +{(execution_decision.regime_profit_target_adjustment-1)*100:.0f}%")
        print()
        
    async def demo_sideways_market_regime(self):
        """Demonstrate sideways market regime"""
        print("Sideways Market Regime: Range-Bound Conditions")
        print("-" * 50)
        
        # Sideways market conditions
        market_data = {
            'vix': 19.5,              # Normal volatility
            'vix9d_vix_ratio': 1.02,  # Normal term structure
            'trend_score': 0.05,      # No clear trend
            'risk_appetite': 0.55,    # Neutral sentiment
            'vol_clustering': 0.4,    # Moderate clustering
            'sector_rotation_score': 0.6,  # Good rotation
            'credit_spreads': 120.0,  # Normal spreads
            'liquidity_score': 0.75,  # Good liquidity
            'correlation_level': 0.55, # Moderate correlation
            'regime_duration_days': 35
        }
        
        regime_analysis = await self.filter_engine.analyze_market_regime(market_data)
        
        print(f"Primary Regime: {regime_analysis.primary_regime.value}")
        print(f"Overall Favorability: {regime_analysis.overall_favorability.value}")
        print(f"Mean Reversion Tendency: {regime_analysis.mean_reversion_tendency:.1%}")
        print()
        
        # Test earnings events (neutral conditions)
        execution_decision = await self.filter_engine.evaluate_trade_execution(
            symbol="NEUTRAL",
            event_type=EventType.EARNINGS,
            event_details={'description': 'Quarterly earnings'},
            regime_analysis=regime_analysis,
            risk_tolerance='moderate'
        )
        
        print("EARNINGS EVENT in Sideways Market:")
        print(f"  Execution Approved: {execution_decision.execution_approved}")
        print(f"  Favorability Score: {execution_decision.favorability_score:.1%}")
        print(f"  Risk Adjustment Factor: {execution_decision.risk_adjustment_factor:.2f}")
        print()
        
    async def demo_event_type_filtering(self):
        """Demonstrate event-specific filtering across regimes"""
        print("Event Type Filtering: Different Events in Same Regime")
        print("-" * 50)
        
        # Moderate bull market
        market_data = {
            'vix': 18.0,
            'trend_score': 0.6,       # Good uptrend
            'risk_appetite': 0.7,
            'vol_clustering': 0.3,
            'regime_duration_days': 20
        }
        
        regime_analysis = await self.filter_engine.analyze_market_regime(market_data)
        
        print(f"Market Regime: {regime_analysis.primary_regime.value}")
        print(f"Overall Favorability: {regime_analysis.overall_favorability.value}")
        print()
        
        # Test all event types
        all_events = [
            (EventType.EARNINGS, "Q3 earnings release"),
            (EventType.FDA_APPROVAL, "Drug approval decision"),
            (EventType.MERGER_ACQUISITION, "Acquisition announced"),
            (EventType.PRODUCT_LAUNCH, "New product launch"),
            (EventType.REGULATORY, "Regulatory approval"),
            (EventType.ANALYST_UPGRADE, "Price target raised"),
            (EventType.GUIDANCE, "Management guidance update")
        ]
        
        results = []
        for event_type, description in all_events:
            execution_decision = await self.filter_engine.evaluate_trade_execution(
                symbol="MULTI",
                event_type=event_type,
                event_details={'description': description},
                regime_analysis=regime_analysis,
                risk_tolerance='moderate'
            )
            
            results.append({
                'event': event_type.value,
                'approved': execution_decision.execution_approved,
                'score': execution_decision.favorability_score,
                'size_modifier': execution_decision.position_size_modifier
            })
        
        # Sort by favorability score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print("Event Type Rankings (Best to Worst):")
        for i, result in enumerate(results, 1):
            status = "✓ APPROVED" if result['approved'] else "✗ REJECTED"
            print(f"{i}. {result['event'].upper()}: {status}")
            print(f"   Score: {result['score']:.1%}, Size: {result['size_modifier']:.2f}x")
        print()
        
    async def demo_bulk_trade_evaluation(self):
        """Demonstrate bulk trade evaluation with regime filtering"""
        print("Bulk Trade Evaluation: Portfolio-Level Filtering")
        print("-" * 50)
        
        # Mixed market conditions
        market_data = {
            'vix': 21.0,
            'trend_score': 0.3,
            'risk_appetite': 0.6,
            'vol_clustering': 0.5,
            'regime_duration_days': 15
        }
        
        regime_analysis = await self.filter_engine.analyze_market_regime(market_data)
        
        # Simulate multiple trade opportunities
        trade_opportunities = [
            {'symbol': 'AAPL', 'event_type': EventType.EARNINGS, 'expected_return': 0.08},
            {'symbol': 'MRNA', 'event_type': EventType.FDA_APPROVAL, 'expected_return': 0.25},
            {'symbol': 'DIS', 'event_type': EventType.PRODUCT_LAUNCH, 'expected_return': 0.12},
            {'symbol': 'XOM', 'event_type': EventType.GUIDANCE, 'expected_return': 0.06},
            {'symbol': 'TSLA', 'event_type': EventType.REGULATORY, 'expected_return': 0.15},
            {'symbol': 'AMZN', 'event_type': EventType.ANALYST_UPGRADE, 'expected_return': 0.05},
            {'symbol': 'NFLX', 'event_type': EventType.EARNINGS, 'expected_return': 0.10},
            {'symbol': 'GOOG', 'event_type': EventType.MERGER_ACQUISITION, 'expected_return': 0.20}
        ]
        
        approved_trades = []
        rejected_trades = []
        total_risk_budget = 0.0
        
        print(f"Evaluating {len(trade_opportunities)} trade opportunities...")
        print()
        
        for trade in trade_opportunities:
            execution_decision = await self.filter_engine.evaluate_trade_execution(
                symbol=trade['symbol'],
                event_type=trade['event_type'],
                event_details={'expected_return': trade['expected_return']},
                regime_analysis=regime_analysis,
                risk_tolerance='moderate'
            )
            
            trade_result = {
                'symbol': trade['symbol'],
                'event_type': trade['event_type'].value,
                'expected_return': trade['expected_return'],
                'favorability_score': execution_decision.favorability_score,
                'position_size': execution_decision.position_size_modifier,
                'risk_adjusted_return': trade['expected_return'] * execution_decision.position_size_modifier
            }
            
            if execution_decision.execution_approved:
                approved_trades.append(trade_result)
                total_risk_budget += execution_decision.position_size_modifier
            else:
                rejected_trades.append(trade_result)
        
        # Display results
        print(f"APPROVED TRADES ({len(approved_trades)}):")
        for trade in sorted(approved_trades, key=lambda x: x['favorability_score'], reverse=True):
            print(f"  {trade['symbol']} ({trade['event_type']}): "
                  f"Score {trade['favorability_score']:.1%}, "
                  f"Size {trade['position_size']:.2f}x, "
                  f"R-Adj Return {trade['risk_adjusted_return']:.1%}")
        
        print(f"\nREJECTED TRADES ({len(rejected_trades)}):")
        for trade in rejected_trades[:3]:  # Show top 3 rejected
            print(f"  {trade['symbol']} ({trade['event_type']}): "
                  f"Score {trade['favorability_score']:.1%}")
        
        print(f"\nPortfolio Metrics:")
        print(f"  Approval Rate: {len(approved_trades)/len(trade_opportunities):.1%}")
        print(f"  Risk Budget Used: {total_risk_budget:.1f} / 10.0")
        print(f"  Expected Portfolio Return: {sum(t['risk_adjusted_return'] for t in approved_trades):.1%}")
        print()
        
    async def demo_risk_adjustment_scenarios(self):
        """Demonstrate risk adjustment across different risk tolerances"""
        print("Risk Adjustment Scenarios: Conservative vs Aggressive")
        print("-" * 50)
        
        # Moderate risk market
        market_data = {
            'vix': 24.0,              # Elevated volatility
            'trend_score': 0.2,       # Weak trend
            'risk_appetite': 0.45,    # Below average appetite
            'vol_clustering': 0.6,
            'regime_duration_days': 8
        }
        
        regime_analysis = await self.filter_engine.analyze_market_regime(market_data)
        
        print(f"Market Conditions: {regime_analysis.primary_regime.value}")
        print(f"Volatility Percentile: {regime_analysis.volatility_percentile:.1%}")
        print(f"Tail Risk: {regime_analysis.tail_risk:.1%}")
        print()
        
        # Test same event with different risk tolerances
        test_event = {
            'symbol': 'RISKY',
            'event_type': EventType.EARNINGS,
            'event_details': {'description': 'High-stakes earnings'}
        }
        
        risk_tolerances = ['conservative', 'moderate', 'aggressive']
        
        print("Risk Tolerance Comparison:")
        for tolerance in risk_tolerances:
            execution_decision = await self.filter_engine.evaluate_trade_execution(
                symbol=test_event['symbol'],
                event_type=test_event['event_type'],
                event_details=test_event['event_details'],
                regime_analysis=regime_analysis,
                risk_tolerance=tolerance
            )
            
            print(f"\n{tolerance.upper()}:")
            print(f"  Execution Approved: {execution_decision.execution_approved}")
            print(f"  Favorability Score: {execution_decision.favorability_score:.1%}")
            print(f"  Position Size Modifier: {execution_decision.position_size_modifier:.2f}x")
            print(f"  Risk Adjustment Factor: {execution_decision.risk_adjustment_factor:.2f}")
            
            if execution_decision.risk_mitigation_required:
                print(f"  Risk Mitigation: {execution_decision.risk_mitigation_required[0]}")
        
        print()

async def main():
    """Run market regime filtering demonstrations"""
    demo = MarketRegimeDemo()
    await demo.run_all_demos()

if __name__ == "__main__":
    asyncio.run(main())