#!/usr/bin/env python3
"""
Strategy Overfitting Detection Script

This script provides a comprehensive analysis of trading strategies for potential 
overfitting using Deflated Sharpe Ratio and Probability of Backtest Overfitting (PBO) methods.

Usage:
    python scripts/detect_strategy_overfitting.py --help
    
Examples:
    # Analyze multiple strategies for overfitting
    python scripts/detect_strategy_overfitting.py \
        --strategies SPY_MOMENTUM,QQQ_MOMENTUM,VIX_MEAN_REVERT \
        --benchmark SP500 \
        --output-report overfitting_analysis.json
    
    # Quick analysis with reduced bootstrap iterations
    python scripts/detect_strategy_overfitting.py \
        --strategies MY_STRATEGY \
        --n-trials 100 \
        --n-pbo-splits 3
    
    # Comprehensive analysis with custom date range
    python scripts/detect_strategy_overfitting.py \
        --strategies MOMENTUM_STRATEGY,MEAN_REVERT_STRATEGY \
        --start-date 2020-01-01 \
        --end-date 2023-12-31 \
        --n-trials 1000 \
        --n-pbo-splits 5 \
        --significance-level 0.01

References:
- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio.
- Bailey, D. H., et al. (2017). The probability of backtest overfitting.
"""

import argparse
import json
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add the analysis service to Python path
sys.path.append(str(Path(__file__).parent.parent / "services" / "analysis-service"))

from app.statistics.overfitting_detection import (
    OverfittingDetector,
    BacktestConfiguration,
    create_backtest_configuration
)
from app.statistics.spa_framework import PerformanceMetrics


class OverfittingAnalysisRunner:
    """Main class for running overfitting detection analysis."""
    
    def __init__(self, config: Dict):
        """Initialize the analysis runner with configuration."""
        self.config = config
        self.detector = OverfittingDetector()
        
        # Setup logging/output
        self.verbose = config.get('verbose', False)
        
    def log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def load_strategy_returns(self, strategy_names: List[str], 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Load strategy returns from data sources.
        
        This is a placeholder implementation. In production, this would
        connect to your strategy database or data warehouse.
        """
        self.log(f"Loading returns for strategies: {strategy_names}")
        
        # Determine date range
        if start_date:
            start_dt = pd.to_datetime(start_date)
        else:
            start_dt = pd.to_datetime('2020-01-01')
            
        if end_date:
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = pd.to_datetime('2023-12-31')
        
        n_days = (end_dt - start_dt).days
        dates = pd.date_range(start_dt, end_dt, freq='D')
        
        strategy_returns = {}
        
        # Generate synthetic returns for demonstration
        # In production, replace this with actual data loading
        np.random.seed(42)  # For reproducible results
        
        for i, strategy_name in enumerate(strategy_names):
            self.log(f"  Loading {strategy_name}...")
            
            # Create different return patterns for different strategies
            if 'MOMENTUM' in strategy_name.upper():
                # Momentum strategy: trending periods with volatility
                base_return = 0.0008 + i * 0.0002
                volatility = 0.018 + i * 0.002
                returns = np.random.normal(base_return, volatility, len(dates))
                
                # Add some momentum periods
                for j in range(0, len(returns), 50):
                    end_idx = min(j + 25, len(returns))
                    trend = np.random.choice([-1, 1]) * 0.002
                    returns[j:end_idx] += trend
                    
            elif 'MEAN_REVERT' in strategy_name.upper():
                # Mean reversion strategy: choppy with reversals
                base_return = 0.0006
                returns = []
                current_return = base_return
                
                for _ in range(len(dates)):
                    # Mean reversion toward base return
                    current_return += np.random.normal(0, 0.01)
                    current_return = 0.7 * current_return + 0.3 * base_return
                    returns.append(current_return + np.random.normal(0, 0.015))
                
                returns = np.array(returns)
                
            elif 'VIX' in strategy_name.upper():
                # VIX strategy: volatile with crisis alpha
                base_return = 0.0004
                volatility = 0.025
                returns = np.random.normal(base_return, volatility, len(dates))
                
                # Add crisis periods with high returns
                crisis_periods = np.random.choice(len(returns), size=len(returns)//20, replace=False)
                returns[crisis_periods] += np.random.normal(0.005, 0.01, len(crisis_periods))
                
            else:
                # Generic strategy
                base_return = 0.0007
                volatility = 0.02
                returns = np.random.normal(base_return, volatility, len(dates))
            
            strategy_returns[strategy_name] = pd.Series(returns, index=dates)
        
        self.log(f"Loaded returns for {len(strategy_returns)} strategies")
        return strategy_returns
    
    def load_benchmark_returns(self, benchmark_name: str,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> pd.Series:
        """
        Load benchmark returns.
        
        This is a placeholder implementation. In production, this would
        connect to your market data sources.
        """
        self.log(f"Loading benchmark returns for: {benchmark_name}")
        
        # Use same date logic as strategy returns
        if start_date:
            start_dt = pd.to_datetime(start_date)
        else:
            start_dt = pd.to_datetime('2020-01-01')
            
        if end_date:
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = pd.to_datetime('2023-12-31')
        
        dates = pd.date_range(start_dt, end_dt, freq='D')
        
        # Generate synthetic benchmark returns
        np.random.seed(43)  # Different seed for benchmark
        
        if benchmark_name.upper() in ['SP500', 'SPY']:
            # S&P 500 characteristics
            annual_return = 0.10
            annual_vol = 0.16
            daily_return = annual_return / 252
            daily_vol = annual_vol / np.sqrt(252)
            
        elif benchmark_name.upper() in ['TREASURY', 'RISK_FREE']:
            # Risk-free rate
            annual_return = 0.02
            annual_vol = 0.01
            daily_return = annual_return / 252
            daily_vol = annual_vol / np.sqrt(252)
            
        else:
            # Default benchmark
            annual_return = 0.08
            annual_vol = 0.15
            daily_return = annual_return / 252
            daily_vol = annual_vol / np.sqrt(252)
        
        returns = np.random.normal(daily_return, daily_vol, len(dates))
        
        self.log(f"Generated {len(returns)} benchmark returns")
        return pd.Series(returns, index=dates)
    
    def run_analysis(self) -> Dict:
        """Run the complete overfitting detection analysis."""
        try:
            self.log("Starting overfitting detection analysis...")
            
            # Load data
            strategy_names = self.config['strategies']
            strategy_returns = self.load_strategy_returns(
                strategy_names,
                self.config.get('start_date'),
                self.config.get('end_date')
            )
            
            benchmark_returns = None
            if self.config.get('benchmark'):
                benchmark_returns = self.load_benchmark_returns(
                    self.config['benchmark'],
                    self.config.get('start_date'),
                    self.config.get('end_date')
                )
            
            # Run comprehensive analysis
            self.log("Running comprehensive overfitting analysis...")
            
            results = self.detector.comprehensive_overfitting_analysis(
                strategy_returns=strategy_returns,
                benchmark_returns=benchmark_returns,
                n_trials=self.config.get('n_trials', 1000),
                n_pbo_splits=self.config.get('n_pbo_splits', 5),
                significance_level=self.config.get('significance_level', 0.05)
            )
            
            # Add metadata
            results['metadata'] = {
                'analysis_date': datetime.now().isoformat(),
                'config': self.config,
                'n_strategies': len(strategy_names),
                'n_observations': len(next(iter(strategy_returns.values()))),
                'date_range': {
                    'start': str(next(iter(strategy_returns.values())).index[0]),
                    'end': str(next(iter(strategy_returns.values())).index[-1])
                }
            }
            
            self.log("Analysis completed successfully")
            return results
            
        except Exception as e:
            self.log(f"Error during analysis: {str(e)}")
            if self.verbose:
                traceback.print_exc()
            raise
    
    def format_results_for_output(self, results: Dict) -> Dict:
        """Format results for JSON serialization."""
        formatted = {}
        
        # Copy metadata
        formatted['metadata'] = results['metadata']
        
        # Format deflated Sharpe results
        formatted['deflated_sharpe_results'] = {}
        for strategy_name, ds_result in results['deflated_sharpe_results'].items():
            formatted['deflated_sharpe_results'][strategy_name] = {
                'sharpe_ratio': float(ds_result.sharpe_ratio) if ds_result.sharpe_ratio is not None else None,
                'deflated_sharpe': float(ds_result.deflated_sharpe) if ds_result.deflated_sharpe is not None else None,
                'p_value': float(ds_result.p_value) if ds_result.p_value is not None else None,
                'is_significant_95': ds_result.is_significant_95,
                'variance_inflation_factor': float(ds_result.variance_inflation_factor) if ds_result.variance_inflation_factor is not None else None,
                'expected_max_sharpe': float(ds_result.expected_max_sharpe) if ds_result.expected_max_sharpe is not None else None,
                'skewness': float(ds_result.skewness) if ds_result.skewness is not None else None,
                'kurtosis': float(ds_result.kurtosis) if ds_result.kurtosis is not None else None
            }
        
        # Format PBO results
        pbo_result = results['pbo_result']
        formatted['pbo_result'] = {
            'pbo_probability': float(pbo_result.pbo_probability),
            'n_strategies': pbo_result.n_strategies,
            'n_splits': pbo_result.n_splits,
            'is_rank_degradation': pbo_result.is_rank_degradation,
            'median_rank_degradation': float(pbo_result.median_rank_degradation) if pbo_result.median_rank_degradation is not None else None,
            'performance_degradation': float(pbo_result.performance_degradation) if pbo_result.performance_degradation is not None else None
        }
        
        # Copy risk assessment and recommendations
        formatted['risk_assessment'] = results['risk_assessment']
        formatted['recommendations'] = results['recommendations']
        
        return formatted
    
    def print_summary(self, results: Dict) -> None:
        """Print a human-readable summary of results."""
        print("\n" + "="*80)
        print("OVERFITTING DETECTION ANALYSIS SUMMARY")
        print("="*80)
        
        metadata = results['metadata']
        print(f"Analysis Date: {metadata['analysis_date']}")
        print(f"Strategies Analyzed: {metadata['n_strategies']}")
        print(f"Observations: {metadata['n_observations']}")
        print(f"Date Range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
        print()
        
        # Deflated Sharpe Results
        print("DEFLATED SHARPE RATIO RESULTS")
        print("-" * 40)
        ds_results = results['deflated_sharpe_results']
        
        for strategy_name, result in ds_results.items():
            print(f"\n{strategy_name}:")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}" if result['sharpe_ratio'] is not None else "  Sharpe Ratio: N/A")
            print(f"  Deflated Sharpe: {result['deflated_sharpe']:.4f}" if result['deflated_sharpe'] is not None else "  Deflated Sharpe: N/A")
            print(f"  P-Value: {result['p_value']:.4f}" if result['p_value'] is not None else "  P-Value: N/A")
            print(f"  Significant (95%): {'Yes' if result['is_significant_95'] else 'No'}")
        
        # PBO Results
        print(f"\n\nPROBABILITY OF BACKTEST OVERFITTING")
        print("-" * 40)
        pbo = results['pbo_result']
        print(f"PBO Probability: {pbo['pbo_probability']:.2%}")
        print(f"Performance Degradation: {pbo['performance_degradation']:.2%}" if pbo['performance_degradation'] is not None else "Performance Degradation: N/A")
        print(f"Median Rank Degradation: {pbo['median_rank_degradation']:.2f}" if pbo['median_rank_degradation'] is not None else "Median Rank Degradation: N/A")
        
        # Risk Assessment
        print(f"\n\nRISK ASSESSMENT")
        print("-" * 40)
        risk = results['risk_assessment']
        print(f"Overall Risk Level: {risk['overall_risk_level']}")
        print(f"High PBO Probability: {'Yes' if risk['high_pbo_probability'] else 'No'}")
        print(f"Multiple Non-Significant Strategies: {'Yes' if risk['multiple_non_significant_strategies'] else 'No'}")
        
        # Recommendations
        print(f"\n\nRECOMMENDATIONS")
        print("-" * 40)
        recommendations = results['recommendations']
        print(f"Primary Recommendation: {recommendations['primary_recommendation']}")
        
        if recommendations['specific_actions']:
            print("\nSpecific Actions:")
            for action in recommendations['specific_actions']:
                print(f"  • {action}")
        
        print("\n" + "="*80)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Detect trading strategy overfitting using statistical methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python scripts/detect_strategy_overfitting.py --strategies SPY_MOMENTUM,QQQ_MOMENTUM
  
  # Comprehensive analysis with benchmark
  python scripts/detect_strategy_overfitting.py \\
    --strategies MOMENTUM,MEAN_REVERT,VIX_HEDGE \\
    --benchmark SP500 \\
    --n-trials 2000 \\
    --output-report full_analysis.json
  
  # Quick test with reduced iterations
  python scripts/detect_strategy_overfitting.py \\
    --strategies TEST_STRATEGY \\
    --n-trials 100 \\
    --n-pbo-splits 3
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--strategies',
        required=True,
        help='Comma-separated list of strategy names to analyze'
    )
    
    # Optional arguments
    parser.add_argument(
        '--benchmark',
        help='Benchmark strategy name (e.g., SP500, TREASURY)'
    )
    
    parser.add_argument(
        '--start-date',
        help='Start date for analysis (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        help='End date for analysis (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=1000,
        help='Number of trials for deflated Sharpe ratio (default: 1000)'
    )
    
    parser.add_argument(
        '--n-pbo-splits',
        type=int,
        default=5,
        help='Number of splits for PBO analysis (default: 5)'
    )
    
    parser.add_argument(
        '--significance-level',
        type=float,
        default=0.05,
        help='Significance level for tests (default: 0.05)'
    )
    
    parser.add_argument(
        '--output-report',
        help='Output file path for detailed JSON report'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Parse strategy list
    strategies = [s.strip() for s in args.strategies.split(',')]
    
    # Build configuration
    config = {
        'strategies': strategies,
        'benchmark': args.benchmark,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'n_trials': args.n_trials,
        'n_pbo_splits': args.n_pbo_splits,
        'significance_level': args.significance_level,
        'verbose': args.verbose
    }
    
    try:
        # Run analysis
        runner = OverfittingAnalysisRunner(config)
        results = runner.run_analysis()
        
        # Print summary
        formatted_results = runner.format_results_for_output(results)
        runner.print_summary(formatted_results)
        
        # Save detailed report if requested
        if args.output_report:
            with open(args.output_report, 'w') as f:
                json.dump(formatted_results, f, indent=2)
            print(f"\nDetailed report saved to: {args.output_report}")
        
        # Exit with appropriate code based on risk level
        risk_level = results['risk_assessment']['overall_risk_level']
        if risk_level == 'HIGH':
            sys.exit(2)  # High risk detected
        elif risk_level == 'MEDIUM':
            sys.exit(1)  # Medium risk detected
        else:
            sys.exit(0)  # Low risk
            
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()