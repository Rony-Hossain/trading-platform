"""
Out-of-Sample (OOS) Validation Example and Testing Framework

This module demonstrates comprehensive OOS validation workflow with
rigorous statistical testing and paper trading integration.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any

from ..services.oos_validation import (
    OOSValidator, ValidationThresholds, ValidationStatus, 
    create_oos_validator
)
from ..services.paper_trading import (
    PaperTradingEngine, OrderType, OrderSide, MarketData,
    create_paper_trading_engine
)
from ..services.car_analysis import EventType, Sector, EventData

logger = logging.getLogger(__name__)

class OOSValidationWorkflow:
    """Comprehensive OOS validation workflow manager"""
    
    def __init__(self):
        self.validator = None
        self.paper_engine = None
        
    async def initialize(self):
        """Initialize validation components"""
        # Create validator with strict thresholds
        strict_thresholds = ValidationThresholds(
            min_sharpe_ratio=1.2,  # Higher than default
            min_t_statistic=2.5,   # Higher than default
            min_hit_rate=0.58,     # Higher than default
            max_drawdown_threshold=0.12,  # Lower than default
            min_calmar_ratio=0.8,
            min_information_ratio=0.4,
            min_validation_period_months=9,  # Longer period
            min_trades=30,         # More trades required
            significance_level=0.01  # More stringent p-value
        )
        
        self.validator = await create_oos_validator(strict_thresholds)
        self.paper_engine = await create_paper_trading_engine(initial_balance=250000.0)
        
    async def demonstrate_full_validation_workflow(self) -> Dict[str, Any]:
        """Demonstrate complete validation workflow from strategy to deployment"""
        
        logger.info("Starting comprehensive OOS validation demonstration")
        
        # Step 1: Generate a realistic trading strategy with signals
        strategy_data = await self._generate_realistic_strategy()
        
        # Step 2: Perform rigorous OOS validation
        validation_results = await self._perform_oos_validation(strategy_data)
        
        # Step 3: If validation passes, run paper trading
        paper_results = None
        if validation_results["validation_status"] == ValidationStatus.PASSED.value:
            paper_results = await self._run_paper_trading_validation(strategy_data)
        
        # Step 4: Generate deployment recommendation
        deployment_decision = self._generate_deployment_decision(
            validation_results, paper_results
        )
        
        return {
            "workflow_summary": {
                "strategy_id": strategy_data["strategy_id"],
                "validation_status": validation_results["validation_status"],
                "paper_trading_completed": paper_results is not None,
                "deployment_recommended": deployment_decision["deploy"],
                "total_validation_time": "45 minutes"  # Typical workflow time
            },
            "validation_results": validation_results,
            "paper_trading_results": paper_results,
            "deployment_decision": deployment_decision,
            "next_steps": self._generate_next_steps(validation_results, deployment_decision)
        }
    
    async def _generate_realistic_strategy(self) -> Dict[str, Any]:
        """Generate realistic strategy with various performance characteristics"""
        
        strategy_id = f"momentum_reversion_v2_{datetime.now().strftime('%Y%m%d')}"
        
        # Generate 18 months of signals (12 for training, 6 for OOS validation)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=540)  # 18 months
        oos_start_date = end_date - timedelta(days=180)  # Last 6 months for OOS
        
        # Simulate a momentum-mean reversion strategy
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "UNH"]
        
        signals = []
        signal_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        for date in signal_dates:
            # Generate 2-4 signals per day with varying strength
            num_signals = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
            selected_symbols = np.random.choice(symbols, num_signals, replace=False)
            
            for symbol in selected_symbols:
                # Create momentum-reversion signal logic
                momentum_signal = np.random.normal(0, 0.3)
                reversion_signal = np.random.normal(0, 0.2)
                
                # Combine signals with time-varying weights
                momentum_weight = 0.7 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                reversion_weight = 1 - momentum_weight
                
                combined_signal = momentum_weight * momentum_signal + reversion_weight * reversion_signal
                
                # Apply realistic constraints
                signal_strength = np.clip(combined_signal, -0.8, 0.8)
                confidence = min(0.5 + abs(signal_strength) * 0.6, 0.95)
                
                # Add noise and regime changes
                if date >= oos_start_date:
                    # Make OOS period slightly more challenging
                    signal_strength *= 0.85
                    confidence *= 0.9
                
                signals.append({
                    "date": date,
                    "symbol": symbol,
                    "signal": signal_strength,
                    "confidence": confidence,
                    "strategy_component": "momentum" if momentum_weight > 0.5 else "reversion",
                    "market_regime": "normal" if abs(signal_strength) < 0.4 else "trending"
                })
        
        return {
            "strategy_id": strategy_id,
            "description": "Advanced momentum-mean reversion strategy with regime adaptation",
            "signals": pd.DataFrame(signals),
            "oos_start_date": oos_start_date,
            "strategy_parameters": {
                "lookback_period": 20,
                "momentum_threshold": 0.02,
                "reversion_threshold": 0.015,
                "position_sizing": "kelly_optimized",
                "risk_management": "dynamic_stops"
            }
        }
    
    async def _perform_oos_validation(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive OOS validation"""
        
        logger.info(f"Validating strategy {strategy_data['strategy_id']}")
        
        # Generate realistic price data
        symbols = strategy_data["signals"]["symbol"].unique()
        price_data = self._generate_realistic_price_data(
            symbols,
            strategy_data["signals"]["date"].min(),
            strategy_data["signals"]["date"].max()
        )
        
        # Generate benchmark data (SPY)
        benchmark_data = self._generate_realistic_benchmark_data(
            strategy_data["signals"]["date"].min(),
            strategy_data["signals"]["date"].max()
        )
        
        # Perform validation
        validation_results = await self.validator.validate_strategy(
            strategy_id=strategy_data["strategy_id"],
            strategy_signals=strategy_data["signals"],
            price_data=price_data,
            benchmark_data=benchmark_data,
            oos_start_date=strategy_data["oos_start_date"],
            validation_period_months=6
        )
        
        return {
            "strategy_id": validation_results.strategy_id,
            "validation_status": validation_results.validation_status.value,
            "performance_metrics": {
                "annualized_return": validation_results.strategy_performance.annualized_return,
                "sharpe_ratio": validation_results.strategy_performance.sharpe_ratio,
                "max_drawdown": validation_results.strategy_performance.max_drawdown,
                "hit_rate": validation_results.strategy_performance.hit_rate,
                "calmar_ratio": validation_results.strategy_performance.calmar_ratio,
                "total_trades": validation_results.strategy_performance.total_trades
            },
            "statistical_tests": [
                {
                    "test": test.test_type,
                    "t_statistic": test.t_statistic,
                    "p_value": test.p_value,
                    "significant": test.is_significant
                }
                for test in validation_results.statistical_tests
            ],
            "benchmark_comparison": {
                "information_ratio": validation_results.information_ratio,
                "excess_return_t_stat": validation_results.excess_return_t_stat
            },
            "risk_assessment": {
                "overfitting_score": validation_results.overfitting_score,
                "risk_warnings": validation_results.risk_warnings
            },
            "recommendations": validation_results.recommendations,
            "detailed_summary": validation_results.validation_summary
        }
    
    async def _run_paper_trading_validation(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run paper trading validation for 30 days"""
        
        logger.info(f"Starting paper trading for {strategy_data['strategy_id']}")
        
        # Create paper account
        account = await self.paper_engine.create_paper_account(strategy_data["strategy_id"])
        
        # Simulate 30 days of paper trading
        validation_start = datetime.now() - timedelta(days=30)
        
        # Get recent signals for paper trading
        recent_signals = strategy_data["signals"][
            strategy_data["signals"]["date"] >= validation_start
        ].copy()
        
        # Simulate daily trading
        total_orders = 0
        successful_trades = 0
        
        for date in pd.date_range(validation_start, datetime.now(), freq='B')[:20]:  # 20 trading days
            day_signals = recent_signals[recent_signals["date"].dt.date == date.date()]
            
            for _, signal_row in day_signals.iterrows():
                symbol = signal_row["symbol"]
                signal_strength = signal_row["signal"]
                confidence = signal_row["confidence"]
                
                # Update market data
                market_price = np.random.uniform(150, 300)
                spread = market_price * np.random.uniform(0.0001, 0.0005)
                
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=date,
                    bid=market_price - spread/2,
                    ask=market_price + spread/2,
                    last=market_price,
                    volume=int(np.random.lognormal(10, 1)),
                    bid_size=100,
                    ask_size=100
                )
                
                await self.paper_engine.update_market_data(symbol, market_data)
                
                # Place order based on signal
                if abs(signal_strength) > 0.3 and confidence > 0.6:
                    try:
                        quantity = int(1000 * abs(signal_strength) * confidence)  # Position sizing
                        side = OrderSide.BUY if signal_strength > 0 else OrderSide.SELL
                        
                        order = await self.paper_engine.place_order(
                            account_id=account.account_id,
                            symbol=symbol,
                            side=side,
                            quantity=quantity,
                            order_type=OrderType.MARKET,
                            metadata={
                                "signal_strength": signal_strength,
                                "confidence": confidence,
                                "strategy": "momentum_reversion"
                            }
                        )
                        
                        total_orders += 1
                        if order.status.value == "filled":
                            successful_trades += 1
                            
                    except Exception as e:
                        logger.warning(f"Failed to place order: {e}")
        
        # Get final performance
        performance = await self.paper_engine.get_performance_metrics(account.account_id)
        account_summary = self.paper_engine.get_account_summary(account.account_id)
        
        return {
            "account_id": account.account_id,
            "trading_period": "30 days",
            "orders_placed": total_orders,
            "successful_trades": successful_trades,
            "execution_rate": successful_trades / total_orders if total_orders > 0 else 0,
            "performance": {
                "total_return": performance["total_return"],
                "win_rate": performance["win_rate"],
                "profit_factor": performance["profit_factor"],
                "sharpe_ratio": performance["sharpe_ratio"],
                "max_drawdown": performance["max_drawdown"]
            },
            "account_summary": {
                "final_value": account_summary["account_value"]["total_value"],
                "cash_balance": account_summary["account_value"]["cash_balance"],
                "positions_count": len(account_summary["positions"]),
                "total_pnl": account_summary["performance"]["total_pnl"]
            },
            "validation_metrics": {
                "meets_sharpe_threshold": performance["sharpe_ratio"] >= 1.0,
                "meets_drawdown_threshold": performance["max_drawdown"] <= 0.15,
                "sufficient_trades": total_orders >= 20,
                "positive_returns": performance["total_return"] > 0
            }
        }
    
    def _generate_deployment_decision(
        self, 
        validation_results: Dict[str, Any], 
        paper_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive deployment decision"""
        
        # Check validation criteria
        validation_passed = validation_results["validation_status"] == "passed"
        
        # Check paper trading criteria (if available)
        paper_criteria_met = True
        if paper_results:
            paper_criteria_met = all([
                paper_results["validation_metrics"]["meets_sharpe_threshold"],
                paper_results["validation_metrics"]["meets_drawdown_threshold"],
                paper_results["validation_metrics"]["sufficient_trades"],
                paper_results["performance"]["win_rate"] >= 0.55
            ])
        
        # Overall decision
        deploy = validation_passed and paper_criteria_met
        
        # Risk assessment
        risk_level = "low"
        if validation_results["risk_assessment"]["overfitting_score"] > 0.5:
            risk_level = "medium"
        if validation_results["risk_assessment"]["overfitting_score"] > 0.7:
            risk_level = "high"
        
        # Position sizing recommendation
        if deploy:
            base_allocation = 0.10  # 10% of capital
            
            # Adjust based on validation strength
            sharpe_adj = min(validation_results["performance_metrics"]["sharpe_ratio"] / 2.0, 1.5)
            risk_adj = max(0.5, 1.0 - validation_results["risk_assessment"]["overfitting_score"])
            
            recommended_allocation = base_allocation * sharpe_adj * risk_adj
            recommended_allocation = min(recommended_allocation, 0.25)  # Cap at 25%
        else:
            recommended_allocation = 0.0
        
        return {
            "deploy": deploy,
            "confidence": "high" if validation_passed and paper_criteria_met else "low",
            "risk_level": risk_level,
            "recommended_allocation": recommended_allocation,
            "deployment_phase": "gradual_ramp" if deploy else "not_recommended",
            "monitoring_requirements": {
                "daily_performance_review": True,
                "weekly_risk_assessment": True,
                "monthly_revalidation": True,
                "stop_loss_trigger": validation_results["performance_metrics"]["max_drawdown"] * 1.5
            },
            "rationale": self._generate_deployment_rationale(
                validation_passed, paper_criteria_met, risk_level, validation_results
            )
        }
    
    def _generate_deployment_rationale(
        self, 
        validation_passed: bool, 
        paper_criteria_met: bool, 
        risk_level: str,
        validation_results: Dict[str, Any]
    ) -> List[str]:
        """Generate detailed deployment rationale"""
        
        rationale = []
        
        if validation_passed:
            rationale.append(f"✓ Strategy passes OOS validation with Sharpe ratio of {validation_results['performance_metrics']['sharpe_ratio']:.2f}")
            rationale.append(f"✓ Statistical significance confirmed (t-stat: {validation_results['benchmark_comparison']['excess_return_t_stat']:.2f})")
        else:
            rationale.append("✗ Strategy fails OOS validation requirements")
            
            if validation_results["performance_metrics"]["sharpe_ratio"] < 1.0:
                rationale.append(f"✗ Sharpe ratio ({validation_results['performance_metrics']['sharpe_ratio']:.2f}) below minimum threshold (1.0)")
            
            if validation_results["performance_metrics"]["max_drawdown"] > 0.15:
                rationale.append(f"✗ Max drawdown ({validation_results['performance_metrics']['max_drawdown']:.2%}) exceeds threshold (15%)")
        
        if paper_criteria_met:
            rationale.append("✓ Paper trading validation successful")
        else:
            rationale.append("✗ Paper trading validation indicates execution challenges")
        
        if risk_level == "high":
            rationale.append("⚠ High overfitting risk detected - strategy may not generalize well")
        elif risk_level == "medium":
            rationale.append("⚠ Moderate overfitting risk - requires careful monitoring")
        else:
            rationale.append("✓ Low overfitting risk - strategy appears robust")
        
        return rationale
    
    def _generate_next_steps(
        self, 
        validation_results: Dict[str, Any], 
        deployment_decision: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable next steps"""
        
        next_steps = []
        
        if deployment_decision["deploy"]:
            next_steps.extend([
                "1. Prepare production deployment with gradual capital allocation",
                "2. Set up real-time monitoring dashboard",
                "3. Configure automated risk controls and stop-losses",
                "4. Schedule first performance review in 7 days",
                "5. Plan monthly strategy revalidation"
            ])
        else:
            next_steps.extend([
                "1. Analyze validation failures and identify improvement areas",
                "2. Consider strategy refinement or parameter optimization",
                "3. Extend validation period or increase sample size",
                "4. Review and update strategy logic based on OOS performance",
                "5. Re-run validation after improvements"
            ])
            
            # Specific recommendations based on failure reasons
            if validation_results["performance_metrics"]["sharpe_ratio"] < 1.0:
                next_steps.append("• Focus on improving risk-adjusted returns")
            
            if validation_results["risk_assessment"]["overfitting_score"] > 0.7:
                next_steps.append("• Simplify strategy to reduce overfitting risk")
            
            if validation_results["performance_metrics"]["total_trades"] < 30:
                next_steps.append("• Increase signal frequency for more robust statistics")
        
        return next_steps
    
    def _generate_realistic_price_data(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate realistic price data with volatility clustering and trends"""
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        price_data = []
        
        for symbol in symbols:
            # Different initial prices for different symbols
            initial_price = np.random.uniform(100, 400)
            prices = [initial_price]
            volatility = np.random.uniform(0.015, 0.035)  # Base volatility
            
            for i in range(len(date_range) - 1):
                # Volatility clustering
                if i > 0:
                    volatility = 0.95 * volatility + 0.05 * np.random.uniform(0.01, 0.04)
                
                # Add trends and mean reversion
                trend = np.random.normal(0.0002, 0.0001)  # Small positive drift
                mean_reversion = -0.1 * (np.log(prices[-1]) - np.log(initial_price))
                
                # Random shock
                shock = np.random.normal(0, volatility)
                
                # Combined return
                return_rate = trend + mean_reversion + shock
                new_price = max(prices[-1] * (1 + return_rate), 1.0)
                prices.append(new_price)
            
            # Calculate returns
            returns = [0.0] + [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            
            # Create data entries
            for i, (date, price, ret) in enumerate(zip(date_range, prices, returns)):
                price_data.append({
                    'date': date,
                    'symbol': symbol,
                    'price': price,
                    'return': ret,
                    'volume': int(np.random.lognormal(12, 0.6))
                })
        
        return pd.DataFrame(price_data)
    
    def _generate_realistic_benchmark_data(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate realistic benchmark (market index) data"""
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Market index parameters
        initial_value = 4000
        values = [initial_value]
        base_volatility = 0.012  # Lower vol than individual stocks
        
        for i in range(len(date_range) - 1):
            # Market trends
            trend = np.random.normal(0.0003, 0.0002)  # Slight positive bias
            
            # Volatility clustering for market
            current_vol = base_volatility * np.random.uniform(0.8, 1.5)
            
            # Market shock
            shock = np.random.normal(0, current_vol)
            
            return_rate = trend + shock
            new_value = max(values[-1] * (1 + return_rate), 1000)
            values.append(new_value)
        
        returns = [0.0] + [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        
        benchmark_data = []
        for date, value, ret in zip(date_range, values, returns):
            benchmark_data.append({
                'date': date,
                'symbol': 'SPY',
                'price': value,
                'return': ret
            })
        
        return pd.DataFrame(benchmark_data)

# Example usage and comprehensive testing
async def run_comprehensive_oos_validation_demo():
    """Run comprehensive OOS validation demonstration"""
    
    workflow = OOSValidationWorkflow()
    await workflow.initialize()
    
    print("=== Comprehensive OOS Validation Workflow Demo ===")
    
    try:
        # Run full workflow
        results = await workflow.demonstrate_full_validation_workflow()
        
        # Display results
        print(f"\n📊 Workflow Summary:")
        print(f"Strategy: {results['workflow_summary']['strategy_id']}")
        print(f"Validation Status: {results['workflow_summary']['validation_status']}")
        print(f"Paper Trading: {'✓' if results['workflow_summary']['paper_trading_completed'] else '✗'}")
        print(f"Deploy Recommended: {'✓' if results['workflow_summary']['deployment_recommended'] else '✗'}")
        
        print(f"\n📈 Validation Performance:")
        perf = results['validation_results']['performance_metrics']
        print(f"Annualized Return: {perf['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {perf['max_drawdown']:.2%}")
        print(f"Hit Rate: {perf['hit_rate']:.1%}")
        print(f"Total Trades: {perf['total_trades']}")
        
        if results['paper_trading_results']:
            print(f"\n🎯 Paper Trading Results:")
            paper = results['paper_trading_results']
            print(f"Orders Placed: {paper['orders_placed']}")
            print(f"Execution Rate: {paper['execution_rate']:.1%}")
            print(f"Paper Return: {paper['performance']['total_return']:.2%}")
            print(f"Paper Sharpe: {paper['performance']['sharpe_ratio']:.2f}")
        
        print(f"\n🚀 Deployment Decision:")
        decision = results['deployment_decision']
        print(f"Deploy: {decision['deploy']}")
        print(f"Confidence: {decision['confidence']}")
        print(f"Risk Level: {decision['risk_level']}")
        print(f"Recommended Allocation: {decision['recommended_allocation']:.1%}")
        
        print(f"\n💡 Next Steps:")
        for i, step in enumerate(results['next_steps'], 1):
            print(f"{i}. {step}")
        
        return results
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        return None

if __name__ == "__main__":
    # Run comprehensive demo
    asyncio.run(run_comprehensive_oos_validation_demo())