#!/usr/bin/env python3
"""
Strategy Statistical Significance Validation Script

Uses White's Reality Check and SPA tests to validate that trading strategies
show statistically significant outperformance, controlling for data mining bias.

Usage:
    python scripts/validate_strategy_significance.py
    python scripts/validate_strategy_significance.py --strategies SPY_MOMENTUM,VIX_MEAN_REVERT
    python scripts/validate_strategy_significance.py --benchmark SP500 --test-type spa_consistent
    python scripts/validate_strategy_significance.py --output-report strategy_validation_report.json
"""

import asyncio
import argparse
import logging
import sys
import os
import json
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add analysis service to path
analysis_service_path = project_root / 'services' / 'analysis-service'
sys.path.insert(0, str(analysis_service_path))

from app.statistics.spa_framework import (
    SPATestFramework, PerformanceMetrics, TestType, BootstrapMethod,
    create_performance_metrics_from_returns
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StrategySignificanceValidator:
    """Validates statistical significance of trading strategies"""
    
    def __init__(self,
                 bootstrap_iterations: int = 10000,
                 significance_level: float = 0.05,
                 output_dir: str = "artifacts/strategy_validation"):
        
        self.bootstrap_iterations = bootstrap_iterations
        self.significance_level = significance_level
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SPA framework
        self.spa_framework = SPATestFramework(
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_method=BootstrapMethod.STATIONARY,
            significance_levels=[significance_level, 0.01]
        )
    
    async def validate_strategies(self,
                                strategy_names: List[str],
                                benchmark_name: str = "SP500",
                                start_date: date = None,
                                end_date: date = None,
                                test_types: List[str] = None) -> Dict[str, Any]:
        """
        Validate statistical significance of specified strategies
        """
        if start_date is None:
            start_date = date.today() - timedelta(days=365*2)  # 2 years default
        if end_date is None:
            end_date = date.today()
        if test_types is None:
            test_types = ["reality_check", "spa_consistent"]
        
        logger.info(f"Validating {len(strategy_names)} strategies against {benchmark_name}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Bootstrap iterations: {self.bootstrap_iterations}")
        
        # Load strategy performance data
        strategy_metrics, benchmark_metrics = await self._load_strategy_data(
            strategy_names, benchmark_name, start_date, end_date
        )
        
        if not strategy_metrics:
            raise ValueError("No strategy data loaded")
        
        # Run comprehensive statistical testing
        validation_results = {
            "metadata": {
                "validation_date": datetime.utcnow().isoformat(),
                "strategies": strategy_names,
                "benchmark": benchmark_name,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "bootstrap_iterations": self.bootstrap_iterations,
                "significance_level": self.significance_level,
                "sample_size": len(strategy_metrics[0].returns)
            },
            "strategy_performance": self._calculate_performance_summary(strategy_metrics, benchmark_metrics),
            "statistical_tests": {},
            "conclusions": {}
        }
        
        # Run requested tests
        if "reality_check" in test_types:
            logger.info("Running White's Reality Check test...")
            rc_result = self.spa_framework.reality_check_test(
                strategy_metrics, benchmark_metrics, test_statistic="sharpe_ratio"
            )
            validation_results["statistical_tests"]["reality_check"] = self._format_test_result(rc_result)
        
        if "spa_consistent" in test_types:
            logger.info("Running SPA consistent test...")
            spa_result = self.spa_framework.spa_test(
                strategy_metrics, benchmark_metrics, 
                test_statistic="sharpe_ratio",
                test_type=TestType.SPA_CONSISTENT
            )
            validation_results["statistical_tests"]["spa_consistent"] = self._format_test_result(spa_result)
        
        if "spa_lower" in test_types:
            logger.info("Running SPA lower test...")
            spa_lower = self.spa_framework.spa_test(
                strategy_metrics, benchmark_metrics,
                test_type=TestType.SPA_LOWER
            )
            validation_results["statistical_tests"]["spa_lower"] = self._format_test_result(spa_lower)
        
        # Individual strategy testing and multiple testing correction
        if len(strategy_metrics) > 1:
            logger.info("Running individual strategy tests...")
            individual_results = []
            
            for strategy in strategy_metrics:
                individual_result = self.spa_framework.spa_test(
                    [strategy], benchmark_metrics, test_type=TestType.SPA_CONSISTENT
                )
                individual_result.strategy_name = strategy.strategy_name
                individual_results.append(individual_result)
            
            # Multiple testing correction
            logger.info("Applying multiple testing corrections...")
            mt_results = self.spa_framework.multiple_testing_correction(individual_results)
            
            validation_results["individual_tests"] = [
                self._format_test_result(result) for result in individual_results
            ]
            validation_results["multiple_testing"] = self._format_multiple_testing_results(mt_results)
        
        # Generate conclusions and recommendations
        validation_results["conclusions"] = self._generate_conclusions(validation_results)
        
        logger.info("Strategy significance validation completed")
        return validation_results
    
    async def _load_strategy_data(self,
                                strategy_names: List[str],
                                benchmark_name: str,
                                start_date: date,
                                end_date: date) -> tuple:
        """
        Load strategy and benchmark performance data
        
        In a real implementation, this would connect to your database
        and load actual strategy returns. For now, we'll simulate data.
        """
        logger.info("Loading strategy performance data...")
        
        # Simulate loading data (replace with actual database queries)
        date_range = pd.date_range(start_date, end_date, freq='D')
        trading_days = date_range[date_range.weekday < 5]  # Only weekdays
        n_days = len(trading_days)
        
        logger.info(f"Loading {n_days} trading days of data")
        
        # Simulate strategy returns with different performance characteristics
        strategy_metrics = []
        np.random.seed(42)  # For reproducible results
        
        for i, strategy_name in enumerate(strategy_names):
            # Simulate different strategy performance levels
            if "MOMENTUM" in strategy_name.upper():
                mean_return = 0.0008 + i * 0.0002  # Momentum strategies
                volatility = 0.018
            elif "MEAN_REVERT" in strategy_name.upper():
                mean_return = 0.0006 + i * 0.0001  # Mean reversion
                volatility = 0.015
            elif "VIX" in strategy_name.upper():
                mean_return = 0.0012  # Volatility trading
                volatility = 0.025
            else:
                mean_return = 0.0005 + i * 0.0003  # Generic strategies
                volatility = 0.020
            
            # Generate correlated returns (some autocorrelation)
            returns = self._generate_realistic_returns(mean_return, volatility, n_days)
            
            metrics = PerformanceMetrics(
                strategy_name=strategy_name,
                returns=returns
            )
            strategy_metrics.append(metrics)
            
            logger.info(f"Loaded {strategy_name}: Sharpe {metrics.sharpe_ratio:.3f}")
        
        # Generate benchmark returns
        benchmark_returns = self._generate_realistic_returns(0.0004, 0.016, n_days)
        benchmark_metrics = PerformanceMetrics(
            strategy_name=benchmark_name,
            returns=benchmark_returns
        )
        
        logger.info(f"Loaded {benchmark_name}: Sharpe {benchmark_metrics.sharpe_ratio:.3f}")
        
        return strategy_metrics, benchmark_metrics
    
    def _generate_realistic_returns(self, mean_return: float, volatility: float, n_days: int) -> np.ndarray:
        """Generate realistic return series with some autocorrelation"""
        # Generate base random returns
        innovations = np.random.normal(0, volatility, n_days)
        
        # Add some autocorrelation (AR(1) process)
        returns = np.zeros(n_days)
        returns[0] = mean_return + innovations[0]
        
        ar_coeff = 0.05  # Small autocorrelation
        for i in range(1, n_days):
            returns[i] = mean_return + ar_coeff * returns[i-1] + innovations[i]
        
        return returns
    
    def _calculate_performance_summary(self,
                                     strategy_metrics: List[PerformanceMetrics],
                                     benchmark_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Calculate comprehensive performance summary"""
        
        summary = {
            "benchmark": {
                "name": benchmark_metrics.strategy_name,
                "sharpe_ratio": benchmark_metrics.sharpe_ratio,
                "annualized_return": np.mean(benchmark_metrics.returns) * 252,
                "annualized_volatility": np.std(benchmark_metrics.returns, ddof=1) * np.sqrt(252),
                "max_drawdown": self._calculate_max_drawdown(benchmark_metrics.returns)
            },
            "strategies": []
        }
        
        for strategy in strategy_metrics:
            excess_returns = strategy.returns - benchmark_metrics.returns
            
            strategy_summary = {
                "name": strategy.strategy_name,
                "sharpe_ratio": strategy.sharpe_ratio,
                "information_ratio": strategy.information_ratio,
                "annualized_return": np.mean(strategy.returns) * 252,
                "annualized_volatility": np.std(strategy.returns, ddof=1) * np.sqrt(252),
                "max_drawdown": self._calculate_max_drawdown(strategy.returns),
                "excess_return": np.mean(excess_returns) * 252,
                "tracking_error": np.std(excess_returns, ddof=1) * np.sqrt(252),
                "win_rate": np.mean(strategy.returns > 0),
                "best_day": np.max(strategy.returns),
                "worst_day": np.min(strategy.returns)
            }
            summary["strategies"].append(strategy_summary)
        
        return summary
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _format_test_result(self, result) -> Dict[str, Any]:
        """Format test result for JSON serialization"""
        return {
            "test_type": result.test_type.value,
            "test_statistic": float(result.test_statistic),
            "p_value": float(result.p_value),
            "critical_value_95": float(result.critical_value_95),
            "critical_value_99": float(result.critical_value_99),
            "is_significant_95": bool(result.is_significant_95),
            "is_significant_99": bool(result.is_significant_99),
            "bootstrap_iterations": int(result.bootstrap_iterations),
            "strategy_name": result.strategy_name,
            "benchmark_name": result.benchmark_name,
            "sample_size": int(result.sample_size)
        }
    
    def _format_multiple_testing_results(self, mt_results) -> Dict[str, Any]:
        """Format multiple testing results for JSON serialization"""
        return {
            "strategy_names": mt_results.strategy_names,
            "raw_p_values": mt_results.raw_p_values.tolist(),
            "bonferroni_p_values": mt_results.bonferroni_p_values.tolist(),
            "fdr_p_values": mt_results.fdr_p_values.tolist(),
            "significant_strategies_raw": mt_results.significant_strategies_raw,
            "significant_strategies_bonferroni": mt_results.significant_strategies_bonferroni,
            "significant_strategies_fdr": mt_results.significant_strategies_fdr,
            "family_wise_error_rate": float(mt_results.family_wise_error_rate),
            "false_discovery_rate": float(mt_results.false_discovery_rate)
        }
    
    def _generate_conclusions(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate conclusions and recommendations"""
        conclusions = {
            "overall_assessment": "",
            "significant_strategies": [],
            "recommended_actions": [],
            "risk_warnings": [],
            "statistical_summary": {}
        }
        
        # Check Reality Check results
        reality_check_significant = False
        if "reality_check" in validation_results["statistical_tests"]:
            rc = validation_results["statistical_tests"]["reality_check"]
            reality_check_significant = rc["is_significant_95"]
        
        # Check SPA test results
        spa_significant = False
        if "spa_consistent" in validation_results["statistical_tests"]:
            spa = validation_results["statistical_tests"]["spa_consistent"]
            spa_significant = spa["is_significant_95"]
        
        # Check multiple testing results
        significant_after_correction = []
        if "multiple_testing" in validation_results:
            mt = validation_results["multiple_testing"]
            significant_after_correction = mt["significant_strategies_fdr"]
        
        # Overall assessment
        if reality_check_significant or spa_significant:
            if len(significant_after_correction) > 0:
                conclusions["overall_assessment"] = "STRONG_EVIDENCE"
                conclusions["significant_strategies"] = significant_after_correction
                conclusions["recommended_actions"].append(
                    "Strategies show statistically significant outperformance even after correction for multiple testing"
                )
            else:
                conclusions["overall_assessment"] = "MODERATE_EVIDENCE"
                conclusions["recommended_actions"].append(
                    "Some evidence of outperformance, but significance disappears after multiple testing correction"
                )
                conclusions["risk_warnings"].append(
                    "Results may be due to data mining bias - use with caution"
                )
        else:
            conclusions["overall_assessment"] = "NO_EVIDENCE"
            conclusions["recommended_actions"].extend([
                "No statistically significant outperformance detected",
                "Consider alternative strategies or longer evaluation periods",
                "Avoid deploying strategies that may be result of data mining"
            ])
        
        # Statistical summary
        n_strategies = len(validation_results["strategy_performance"]["strategies"])
        best_strategy = max(
            validation_results["strategy_performance"]["strategies"],
            key=lambda x: x["sharpe_ratio"]
        )
        
        conclusions["statistical_summary"] = {
            "n_strategies_tested": n_strategies,
            "best_strategy_name": best_strategy["name"],
            "best_strategy_sharpe": best_strategy["sharpe_ratio"],
            "bootstrap_iterations": validation_results["metadata"]["bootstrap_iterations"],
            "family_wise_error_controlled": len(significant_after_correction) > 0
        }
        
        # Risk warnings based on statistical properties
        if n_strategies > 10:
            conclusions["risk_warnings"].append(
                f"Large number of strategies tested ({n_strategies}) increases risk of false discoveries"
            )
        
        sample_size = validation_results["metadata"]["sample_size"]
        if sample_size < 252:
            conclusions["risk_warnings"].append(
                f"Small sample size ({sample_size} observations) may lead to unreliable results"
            )
        
        return conclusions
    
    def save_validation_report(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save validation results to JSON report"""
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_validation_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to: {output_path}")
        return str(output_path)
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable summary report"""
        
        summary_lines = [
            "TRADING STRATEGY STATISTICAL SIGNIFICANCE VALIDATION REPORT",
            "=" * 60,
            "",
            f"Validation Date: {results['metadata']['validation_date']}",
            f"Strategies Tested: {', '.join(results['metadata']['strategies'])}",
            f"Benchmark: {results['metadata']['benchmark']}",
            f"Period: {results['metadata']['start_date']} to {results['metadata']['end_date']}",
            f"Sample Size: {results['metadata']['sample_size']} observations",
            f"Bootstrap Iterations: {results['metadata']['bootstrap_iterations']:,}",
            "",
            "PERFORMANCE SUMMARY",
            "-" * 30,
        ]
        
        # Benchmark performance
        benchmark = results["strategy_performance"]["benchmark"]
        summary_lines.extend([
            f"Benchmark ({benchmark['name']}):",
            f"  Sharpe Ratio: {benchmark['sharpe_ratio']:.3f}",
            f"  Annual Return: {benchmark['annualized_return']:.2%}",
            f"  Annual Volatility: {benchmark['annualized_volatility']:.2%}",
            f"  Max Drawdown: {benchmark['max_drawdown']:.2%}",
            ""
        ])
        
        # Strategy performance
        for strategy in results["strategy_performance"]["strategies"]:
            summary_lines.extend([
                f"{strategy['name']}:",
                f"  Sharpe Ratio: {strategy['sharpe_ratio']:.3f}",
                f"  Information Ratio: {strategy['information_ratio']:.3f}",
                f"  Annual Return: {strategy['annualized_return']:.2%}",
                f"  Excess Return: {strategy['excess_return']:.2%}",
                f"  Tracking Error: {strategy['tracking_error']:.2%}",
                f"  Win Rate: {strategy['win_rate']:.2%}",
                ""
            ])
        
        # Statistical test results
        summary_lines.extend([
            "STATISTICAL TEST RESULTS",
            "-" * 30,
        ])
        
        for test_name, test_result in results["statistical_tests"].items():
            summary_lines.extend([
                f"{test_name.upper()}:",
                f"  Test Statistic: {test_result['test_statistic']:.4f}",
                f"  P-value: {test_result['p_value']:.4f}",
                f"  Significant (5%): {'YES' if test_result['is_significant_95'] else 'NO'}",
                f"  Significant (1%): {'YES' if test_result['is_significant_99'] else 'NO'}",
                ""
            ])
        
        # Multiple testing results
        if "multiple_testing" in results:
            mt = results["multiple_testing"]
            summary_lines.extend([
                "MULTIPLE TESTING CORRECTION",
                "-" * 30,
                f"Strategies significant (raw): {len(mt['significant_strategies_raw'])}",
                f"Strategies significant (Bonferroni): {len(mt['significant_strategies_bonferroni'])}",
                f"Strategies significant (FDR): {len(mt['significant_strategies_fdr'])}",
                f"Family-wise error rate: {mt['family_wise_error_rate']:.4f}",
                f"False discovery rate: {mt['false_discovery_rate']:.4f}",
                ""
            ])
        
        # Conclusions
        conclusions = results["conclusions"]
        summary_lines.extend([
            "CONCLUSIONS & RECOMMENDATIONS",
            "-" * 30,
            f"Overall Assessment: {conclusions['overall_assessment']}",
            ""
        ])
        
        if conclusions["significant_strategies"]:
            summary_lines.extend([
                f"Significant Strategies: {', '.join(conclusions['significant_strategies'])}",
                ""
            ])
        
        summary_lines.extend([
            "Recommended Actions:",
        ])
        for action in conclusions["recommended_actions"]:
            summary_lines.append(f"  • {action}")
        
        if conclusions["risk_warnings"]:
            summary_lines.extend([
                "",
                "Risk Warnings:",
            ])
            for warning in conclusions["risk_warnings"]:
                summary_lines.append(f"  ⚠ {warning}")
        
        return "\n".join(summary_lines)

async def main():
    """Main entry point for strategy validation script"""
    parser = argparse.ArgumentParser(
        description="Validate statistical significance of trading strategies"
    )
    parser.add_argument(
        '--strategies',
        default='SPY_MOMENTUM,QQQ_MOMENTUM,VIX_MEAN_REVERT,SECTOR_ROTATION,EARNINGS_MOMENTUM',
        help='Comma-separated list of strategy names to validate'
    )
    parser.add_argument(
        '--benchmark',
        default='SP500',
        help='Benchmark strategy name'
    )
    parser.add_argument(
        '--start-date',
        type=date.fromisoformat,
        default=date.today() - timedelta(days=730),  # 2 years
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=date.fromisoformat,
        default=date.today(),
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--test-types',
        default='reality_check,spa_consistent',
        help='Comma-separated list of tests to run'
    )
    parser.add_argument(
        '--bootstrap-iterations',
        type=int,
        default=10000,
        help='Number of bootstrap iterations'
    )
    parser.add_argument(
        '--output-report',
        help='Output filename for validation report'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse arguments
    strategy_names = [s.strip() for s in args.strategies.split(',')]
    test_types = [t.strip() for t in args.test_types.split(',')]
    
    logger.info(f"Starting strategy significance validation for {len(strategy_names)} strategies")
    
    try:
        # Initialize validator
        validator = StrategySignificanceValidator(
            bootstrap_iterations=args.bootstrap_iterations
        )
        
        # Run validation
        results = await validator.validate_strategies(
            strategy_names=strategy_names,
            benchmark_name=args.benchmark,
            start_date=args.start_date,
            end_date=args.end_date,
            test_types=test_types
        )
        
        # Save detailed report
        report_path = validator.save_validation_report(results, args.output_report)
        
        # Generate and display summary
        summary_report = validator.generate_summary_report(results)
        print("\n" + summary_report)
        
        # Exit with appropriate code
        overall_assessment = results["conclusions"]["overall_assessment"]
        if overall_assessment == "STRONG_EVIDENCE":
            logger.info("Validation completed: Strong evidence of outperformance found")
            return 0
        elif overall_assessment == "MODERATE_EVIDENCE":
            logger.warning("Validation completed: Moderate evidence, use caution")
            return 0
        else:
            logger.warning("Validation completed: No significant outperformance detected")
            return 1
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 2

if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code)