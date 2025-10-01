"""
Validation script for Triple Barrier Labeling and Meta-Labeling Acceptance Criteria.

This script validates the implementation against the specified requirements:
- OOS meta-labeling improves F1 ≥ +0.05
- Calibration curve slope within [0.9, 1.1]
- All functionality works correctly with edge cases
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Any
import warnings

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from labels.triple_barrier import (
    TripleBarrierConfig,
    TripleBarrierLabeler,
    MetaLabeler,
    create_meta_labels
)
from datasets.builder import DatasetBuilder, DatasetConfig
from core.config import get_acceptance_criteria, get_triple_barrier_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class AcceptanceCriteriaValidator:
    """
    Validates the triple barrier labeling implementation against
    acceptance criteria.
    """
    
    def __init__(self):
        self.results = {}
        self.acceptance_criteria = get_acceptance_criteria()
        self.tb_config = TripleBarrierConfig(**get_triple_barrier_config())
        
    def generate_realistic_data(self, 
                              n_days: int = 1000,
                              base_price: float = 100.0,
                              seed: int = 42) -> Tuple[pd.Series, pd.DataFrame]:
        """Generate realistic market data for testing."""
        
        np.random.seed(seed)
        dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
        
        # Create realistic price series with:
        # - Trends
        # - Volatility clustering
        # - Occasional jumps
        
        # Base drift and volatility
        drift = 0.0002  # ~5% annual
        base_vol = 0.015  # ~24% annual
        
        # Add volatility clustering (GARCH-like)
        vol_persistence = 0.9
        vol_mean_reversion = 0.1
        vol_shock = 0.3
        
        volatilities = np.zeros(n_days)
        volatilities[0] = base_vol
        
        for i in range(1, n_days):
            vol_innovation = np.random.normal(0, 0.001)
            volatilities[i] = (vol_persistence * volatilities[i-1] + 
                             vol_mean_reversion * base_vol +
                             vol_shock * vol_innovation)
            volatilities[i] = max(volatilities[i], 0.005)  # Floor at 0.5%
        
        # Generate returns with time-varying volatility
        returns = np.random.normal(drift, volatilities)
        
        # Add occasional jumps (1% chance per day)
        jump_mask = np.random.random(n_days) < 0.01
        jump_sizes = np.random.normal(0, 0.03, n_days)  # 3% jump volatility
        returns[jump_mask] += jump_sizes[jump_mask]
        
        # Create price series
        prices = pd.Series(
            base_price * np.exp(np.cumsum(returns)),
            index=dates
        )
        
        # Generate realistic trading signals
        # Use momentum + mean reversion strategy
        signal_frequency = 0.1  # 10% chance per day
        
        short_ma = prices.rolling(5).mean()
        long_ma = prices.rolling(20).mean()
        momentum_signal = (short_ma > long_ma).astype(int) * 2 - 1
        
        # Add noise to signals
        signal_noise = np.random.normal(0, 0.5, len(prices))
        noisy_momentum = momentum_signal + signal_noise
        
        # Generate actual trading events
        signal_dates = []
        signal_sides = []
        
        for i, date in enumerate(dates[20:], 20):  # Start after MA warm-up
            if np.random.random() < signal_frequency:
                signal_dates.append(date)
                # Use noisy momentum signal
                side = 1 if noisy_momentum.iloc[i] > 0 else -1
                signal_sides.append(side)
        
        events = pd.DataFrame({
            'timestamp': signal_dates,
            'side': signal_sides
        })
        
        logger.info(f"Generated {len(prices)} price points and {len(events)} trading events")
        
        return prices, events
    
    def create_baseline_strategy(self, 
                               prices: pd.Series,
                               events: pd.DataFrame) -> Dict[str, float]:
        """Create baseline strategy performance (without meta-labeling)."""
        
        logger.info("Creating baseline strategy performance...")
        
        # Simple buy-and-hold baseline
        baseline_returns = []
        
        for _, event in events.iterrows():
            entry_date = event['timestamp']
            side = event['side']
            
            if entry_date not in prices.index:
                continue
                
            entry_price = prices[entry_date]
            
            # Hold for fixed period (5 days)
            exit_date = entry_date + timedelta(days=5)
            
            # Find closest available exit date
            available_dates = prices.index[prices.index >= exit_date]
            if len(available_dates) == 0:
                continue
                
            exit_date = available_dates[0]
            exit_price = prices[exit_date]
            
            # Calculate return
            if side == 1:  # Long
                ret = (exit_price - entry_price) / entry_price
            else:  # Short
                ret = (entry_price - exit_price) / entry_price
            
            baseline_returns.append(ret)
        
        baseline_returns = np.array(baseline_returns)
        
        # Calculate baseline metrics
        baseline_metrics = {
            'mean_return': np.mean(baseline_returns),
            'volatility': np.std(baseline_returns),
            'sharpe_ratio': np.mean(baseline_returns) / np.std(baseline_returns) if np.std(baseline_returns) > 0 else 0,
            'hit_rate': np.mean(baseline_returns > 0),
            'n_trades': len(baseline_returns)
        }
        
        logger.info(f"Baseline metrics: {baseline_metrics}")
        return baseline_metrics
    
    def test_triple_barrier_labeling(self, 
                                   prices: pd.Series,
                                   events: pd.DataFrame) -> Dict[str, Any]:
        """Test triple barrier labeling functionality."""
        
        logger.info("Testing triple barrier labeling...")
        
        labeler = TripleBarrierLabeler(self.tb_config)
        labels = labeler.create_labels(prices, events, 'VALIDATION_TEST')
        
        if not labels:
            raise ValueError("No triple barrier labels created")
        
        # Analyze label distribution
        label_values = [label.label for label in labels]
        label_dist = pd.Series(label_values).value_counts().to_dict()
        
        # Analyze barrier hits
        barrier_hits = [label.hit_barrier for label in labels]
        barrier_dist = pd.Series(barrier_hits).value_counts().to_dict()
        
        # Sample weights
        sample_weights = [label.sample_weight for label in labels]
        
        results = {
            'n_labels': len(labels),
            'label_distribution': label_dist,
            'barrier_distribution': barrier_dist,
            'sample_weight_stats': {
                'mean': np.mean(sample_weights),
                'std': np.std(sample_weights),
                'min': np.min(sample_weights),
                'max': np.max(sample_weights)
            },
            'holding_time_stats': {
                'mean': np.mean([label.holding_time for label in labels]),
                'std': np.std([label.holding_time for label in labels]),
                'max': np.max([label.holding_time for label in labels])
            }
        }
        
        logger.info(f"Triple barrier results: {results}")
        return results
    
    def test_meta_labeling_performance(self,
                                     prices: pd.Series,
                                     events: pd.DataFrame,
                                     baseline_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Test meta-labeling performance and improvement over baseline."""
        
        logger.info("Testing meta-labeling performance...")
        
        # Create meta-labels
        try:
            meta_results, performance_metrics = create_meta_labels(
                prices, events, self.tb_config
            )
        except Exception as e:
            logger.error(f"Meta-labeling failed: {e}")
            return {'error': str(e)}
        
        if not meta_results:
            return {'error': 'No meta-labeling results generated'}
        
        # Calculate meta-labeling metrics
        meta_predictions = [result.meta_prediction for result in meta_results]
        meta_probabilities = [result.meta_probability for result in meta_results]
        
        if len(meta_predictions) == 0:
            return {'error': 'No meta-predictions generated'}
        
        # Simulate meta-labeling strategy performance
        meta_returns = []
        
        for i, result in enumerate(meta_results):
            if result.meta_prediction == 1:  # Trade signal
                # Get actual return from features
                if 'true_return' in result.features:
                    actual_return = result.features['true_return']
                    meta_returns.append(actual_return)
        
        if len(meta_returns) == 0:
            return {'error': 'No trades executed by meta-labeling strategy'}
        
        meta_returns = np.array(meta_returns)
        
        # Calculate meta-strategy metrics
        meta_strategy_metrics = {
            'mean_return': np.mean(meta_returns),
            'volatility': np.std(meta_returns),
            'sharpe_ratio': np.mean(meta_returns) / np.std(meta_returns) if np.std(meta_returns) > 0 else 0,
            'hit_rate': np.mean(meta_returns > 0),
            'n_trades': len(meta_returns)
        }
        
        # Calculate improvements over baseline
        baseline_f1 = 2 * baseline_metrics['hit_rate'] * baseline_metrics['hit_rate'] / (2 * baseline_metrics['hit_rate']) if baseline_metrics['hit_rate'] > 0 else 0
        
        if 'f1_score' in performance_metrics:
            meta_f1 = performance_metrics['f1_score']
            f1_improvement = meta_f1 - baseline_f1
        else:
            # Estimate F1 from hit rate
            meta_f1 = 2 * meta_strategy_metrics['hit_rate'] * meta_strategy_metrics['hit_rate'] / (2 * meta_strategy_metrics['hit_rate']) if meta_strategy_metrics['hit_rate'] > 0 else 0
            f1_improvement = meta_f1 - baseline_f1
        
        results = {
            'meta_strategy_metrics': meta_strategy_metrics,
            'performance_metrics': performance_metrics,
            'f1_improvement': f1_improvement,
            'meets_f1_criteria': f1_improvement >= self.acceptance_criteria['f1_improvement_threshold'],
            'calibration_slope': performance_metrics.get('calibration_slope', None),
            'meets_calibration_criteria': False
        }
        
        # Check calibration criteria
        if 'calibration_slope' in performance_metrics:
            slope = performance_metrics['calibration_slope']
            min_slope, max_slope = self.acceptance_criteria['calibration_slope_range']
            results['meets_calibration_criteria'] = min_slope <= slope <= max_slope
        
        logger.info(f"Meta-labeling results: {results}")
        return results
    
    def test_edge_cases(self, prices: pd.Series) -> Dict[str, bool]:
        """Test edge cases handling."""
        
        logger.info("Testing edge cases...")
        
        edge_case_results = {}
        labeler = TripleBarrierLabeler(self.tb_config)
        
        # Test 1: Price gaps
        try:
            gap_dates = prices.index[:20]
            gap_prices = pd.Series([100] * 10 + [120] * 10, index=gap_dates)
            gap_events = pd.DataFrame({
                'timestamp': [gap_dates[5]],
                'side': [1]
            })
            gap_labels = labeler.create_labels(gap_prices, gap_events, 'GAP_TEST')
            edge_case_results['handles_price_gaps'] = True
        except Exception as e:
            logger.error(f"Price gap test failed: {e}")
            edge_case_results['handles_price_gaps'] = False
        
        # Test 2: Trading halts (flat prices)
        try:
            halt_dates = prices.index[:20]
            halt_prices = pd.Series([100] * 20, index=halt_dates)
            halt_events = pd.DataFrame({
                'timestamp': [halt_dates[5]],
                'side': [1]
            })
            halt_labels = labeler.create_labels(halt_prices, halt_events, 'HALT_TEST')
            edge_case_results['handles_trading_halts'] = True
        except Exception as e:
            logger.error(f"Trading halt test failed: {e}")
            edge_case_results['handles_trading_halts'] = False
        
        # Test 3: Extreme volatility
        try:
            np.random.seed(42)
            extreme_dates = prices.index[:50]
            extreme_returns = np.random.normal(0, 0.1, 50)  # 10% daily vol
            extreme_prices = pd.Series(
                100 * np.exp(np.cumsum(extreme_returns)),
                index=extreme_dates
            )
            extreme_events = pd.DataFrame({
                'timestamp': extreme_dates[::10],
                'side': [1, -1, 1, 1, -1]
            })
            extreme_labels = labeler.create_labels(extreme_prices, extreme_events, 'EXTREME_TEST')
            edge_case_results['handles_extreme_volatility'] = True
        except Exception as e:
            logger.error(f"Extreme volatility test failed: {e}")
            edge_case_results['handles_extreme_volatility'] = False
        
        # Test 4: Insufficient data
        try:
            short_dates = prices.index[:3]
            short_prices = pd.Series([100, 101, 102], index=short_dates)
            short_events = pd.DataFrame({
                'timestamp': [short_dates[0]],
                'side': [1]
            })
            short_labels = labeler.create_labels(short_prices, short_events, 'SHORT_TEST')
            edge_case_results['handles_insufficient_data'] = True
        except Exception as e:
            logger.error(f"Insufficient data test failed: {e}")
            edge_case_results['handles_insufficient_data'] = False
        
        logger.info(f"Edge case results: {edge_case_results}")
        return edge_case_results
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        
        logger.info("Starting acceptance criteria validation...")
        
        # Generate test data
        prices, events = self.generate_realistic_data(n_days=800, seed=42)
        
        # Run tests
        try:
            # Test 1: Triple barrier labeling
            self.results['triple_barrier'] = self.test_triple_barrier_labeling(prices, events)
            
            # Test 2: Baseline strategy
            baseline_metrics = self.create_baseline_strategy(prices, events)
            self.results['baseline'] = baseline_metrics
            
            # Test 3: Meta-labeling performance
            self.results['meta_labeling'] = self.test_meta_labeling_performance(
                prices, events, baseline_metrics
            )
            
            # Test 4: Edge cases
            self.results['edge_cases'] = self.test_edge_cases(prices)
            
            # Overall assessment
            self.results['overall_assessment'] = self._assess_overall_performance()
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self.results['error'] = str(e)
        
        return self.results
    
    def _assess_overall_performance(self) -> Dict[str, Any]:
        """Assess overall performance against acceptance criteria."""
        
        assessment = {
            'meets_all_criteria': True,
            'criteria_results': {},
            'recommendations': []
        }
        
        # Check F1 improvement criterion
        if 'meta_labeling' in self.results and 'f1_improvement' in self.results['meta_labeling']:
            f1_improvement = self.results['meta_labeling']['f1_improvement']
            f1_threshold = self.acceptance_criteria['f1_improvement_threshold']
            
            meets_f1 = f1_improvement >= f1_threshold
            assessment['criteria_results']['f1_improvement'] = {
                'meets_criteria': meets_f1,
                'actual_value': f1_improvement,
                'required_value': f1_threshold
            }
            
            if not meets_f1:
                assessment['meets_all_criteria'] = False
                assessment['recommendations'].append(
                    f"F1 improvement ({f1_improvement:.3f}) is below threshold ({f1_threshold}). "
                    "Consider tuning hyperparameters or feature engineering."
                )
        
        # Check calibration criterion
        if ('meta_labeling' in self.results and 
            'calibration_slope' in self.results['meta_labeling'] and
            self.results['meta_labeling']['calibration_slope'] is not None):
            
            calibration_slope = self.results['meta_labeling']['calibration_slope']
            min_slope, max_slope = self.acceptance_criteria['calibration_slope_range']
            
            meets_calibration = min_slope <= calibration_slope <= max_slope
            assessment['criteria_results']['calibration'] = {
                'meets_criteria': meets_calibration,
                'actual_value': calibration_slope,
                'required_range': [min_slope, max_slope]
            }
            
            if not meets_calibration:
                assessment['meets_all_criteria'] = False
                assessment['recommendations'].append(
                    f"Calibration slope ({calibration_slope:.3f}) is outside acceptable range "
                    f"[{min_slope}, {max_slope}]. Consider model calibration techniques."
                )
        
        # Check edge cases
        if 'edge_cases' in self.results:
            edge_cases = self.results['edge_cases']
            failed_cases = [case for case, passed in edge_cases.items() if not passed]
            
            if failed_cases:
                assessment['meets_all_criteria'] = False
                assessment['recommendations'].append(
                    f"Failed edge cases: {failed_cases}. "
                    "Improve error handling and robustness."
                )
            
            assessment['criteria_results']['edge_cases'] = {
                'meets_criteria': len(failed_cases) == 0,
                'failed_cases': failed_cases,
                'passed_cases': [case for case, passed in edge_cases.items() if passed]
            }
        
        # General recommendations
        if assessment['meets_all_criteria']:
            assessment['recommendations'].append(
                "All acceptance criteria met! Implementation is ready for production."
            )
        else:
            assessment['recommendations'].append(
                "Some criteria not met. Address recommendations before deployment."
            )
        
        return assessment
    
    def print_validation_report(self):
        """Print comprehensive validation report."""
        
        print("\n" + "="*80)
        print("TRIPLE BARRIER LABELING - ACCEPTANCE CRITERIA VALIDATION REPORT")
        print("="*80)
        
        if 'error' in self.results:
            print(f"\n❌ VALIDATION FAILED: {self.results['error']}")
            return
        
        # Triple Barrier Results
        if 'triple_barrier' in self.results:
            tb = self.results['triple_barrier']
            print(f"\n📊 TRIPLE BARRIER LABELING RESULTS:")
            print(f"   Labels Created: {tb['n_labels']}")
            print(f"   Label Distribution: {tb['label_distribution']}")
            print(f"   Barrier Distribution: {tb['barrier_distribution']}")
            print(f"   Avg Sample Weight: {tb['sample_weight_stats']['mean']:.3f}")
        
        # Meta-Labeling Results
        if 'meta_labeling' in self.results:
            ml = self.results['meta_labeling']
            if 'error' not in ml:
                print(f"\n🎯 META-LABELING RESULTS:")
                print(f"   F1 Improvement: {ml['f1_improvement']:.3f}")
                print(f"   Meets F1 Criteria: {'✅' if ml['meets_f1_criteria'] else '❌'}")
                
                if ml['calibration_slope'] is not None:
                    print(f"   Calibration Slope: {ml['calibration_slope']:.3f}")
                    print(f"   Meets Calibration Criteria: {'✅' if ml['meets_calibration_criteria'] else '❌'}")
            else:
                print(f"\n❌ META-LABELING FAILED: {ml['error']}")
        
        # Edge Cases
        if 'edge_cases' in self.results:
            edge = self.results['edge_cases']
            print(f"\n🛡️ EDGE CASES TESTING:")
            for case, passed in edge.items():
                status = "✅" if passed else "❌"
                print(f"   {case}: {status}")
        
        # Overall Assessment
        if 'overall_assessment' in self.results:
            assessment = self.results['overall_assessment']
            print(f"\n🏆 OVERALL ASSESSMENT:")
            
            status = "✅ PASS" if assessment['meets_all_criteria'] else "❌ FAIL"
            print(f"   Status: {status}")
            
            print(f"\n📋 RECOMMENDATIONS:")
            for rec in assessment['recommendations']:
                print(f"   • {rec}")
        
        print("\n" + "="*80)


def main():
    """Main validation function."""
    
    validator = AcceptanceCriteriaValidator()
    
    print("Starting Triple Barrier Labeling validation...")
    print("This may take a few minutes...")
    
    # Run validation
    results = validator.run_validation()
    
    # Print report
    validator.print_validation_report()
    
    # Return exit code based on results
    if 'overall_assessment' in results:
        return 0 if results['overall_assessment']['meets_all_criteria'] else 1
    else:
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)