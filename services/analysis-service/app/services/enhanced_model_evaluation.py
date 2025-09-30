"""
Enhanced Model Evaluation with Statistical Significance Testing

Extends the standard model evaluation framework with sophisticated statistical tests:
- Superior Predictive Ability (SPA) testing
- Deflated Sharpe Ratio calculation
- Probability of Backtest Overfitting (PBO) estimation
- Deployment gate validation
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from .model_evaluation import (
    ModelEvaluator, EvaluationResult, ModelMetrics, 
    FinancialMetricsCalculator
)
from .significance_tests import (
    SignificanceTestSuite, validate_deployment_gates,
    format_test_results_for_api
)

logger = logging.getLogger(__name__)


@dataclass
class SignificanceMetrics:
    """Container for statistical significance metrics."""
    spa_p_value: float
    spa_is_significant: bool
    deflated_sharpe: float
    deflated_sharpe_p_value: float
    deflated_sharpe_significant: bool
    pbo_estimate: float
    pbo_is_overfitted: bool
    deployment_gate_passed: bool
    confidence_level: str
    n_strategies_tested: int
    interpretation: str


@dataclass
class EnhancedEvaluationResult:
    """Enhanced evaluation results with significance testing."""
    # Original evaluation results
    base_evaluation: EvaluationResult
    
    # Significance testing results
    significance_metrics: SignificanceMetrics
    deployment_recommendation: Dict[str, Any]
    
    # Enhanced metrics
    risk_adjusted_ranking: Dict[str, float]
    overfitting_risk_score: float
    statistical_confidence: str
    
    # Deployment gates
    spa_gate_passed: bool
    pbo_gate_passed: bool
    overall_deployment_approved: bool
    
    # Additional analysis
    model_comparison_analysis: Dict[str, Any]
    sensitivity_analysis: Dict[str, Any]
    
    timestamp: datetime


class EnhancedModelEvaluator:
    """
    Enhanced model evaluator with statistical significance testing.
    
    Combines traditional model evaluation metrics with sophisticated
    statistical tests to provide robust deployment recommendations.
    """
    
    def __init__(self, 
                 spa_threshold: float = 0.05,
                 pbo_threshold: float = 0.2,
                 min_strategies_for_testing: int = 3):
        """
        Initialize enhanced evaluator.
        
        Args:
            spa_threshold: SPA test significance threshold
            pbo_threshold: Maximum acceptable PBO estimate
            min_strategies_for_testing: Minimum strategies needed for significance testing
        """
        self.base_evaluator = ModelEvaluator()
        self.spa_threshold = spa_threshold
        self.pbo_threshold = pbo_threshold
        self.min_strategies_for_testing = min_strategies_for_testing
        
    async def evaluate_with_significance_testing(self,
                                               data: pd.DataFrame,
                                               target_column: str,
                                               benchmark_returns: Optional[pd.Series] = None,
                                               **kwargs) -> EnhancedEvaluationResult:
        """
        Run comprehensive model evaluation with significance testing.
        
        Args:
            data: Input data for model training/evaluation
            target_column: Target variable column name
            benchmark_returns: Benchmark returns for comparison
            **kwargs: Additional parameters for base evaluation
            
        Returns:
            EnhancedEvaluationResult with comprehensive analysis
        """
        logger.info("Starting enhanced model evaluation with significance testing")
        
        # 1. Run base model evaluation
        base_result = await self.base_evaluator.evaluate_models(
            data, target_column, **kwargs
        )
        
        # 2. Extract strategy returns for significance testing
        strategy_returns = self._extract_strategy_returns(base_result, data)
        
        # 3. Prepare benchmark returns
        if benchmark_returns is None:
            benchmark_returns = self._create_benchmark_returns(data, target_column)
        
        # 4. Run significance testing if we have enough strategies
        if len(strategy_returns) >= self.min_strategies_for_testing:
            significance_results = await self._run_significance_tests(
                strategy_returns, benchmark_returns, base_result
            )
        else:
            logger.warning(f"Only {len(strategy_returns)} strategies available, "
                         f"minimum {self.min_strategies_for_testing} required for significance testing")
            significance_results = self._create_default_significance_results()
        
        # 5. Generate deployment recommendation
        deployment_recommendation = self._generate_deployment_recommendation(
            base_result, significance_results
        )
        
        # 6. Calculate enhanced metrics
        enhanced_metrics = self._calculate_enhanced_metrics(
            base_result, significance_results
        )
        
        # 7. Create final result
        enhanced_result = EnhancedEvaluationResult(
            base_evaluation=base_result,
            significance_metrics=significance_results,
            deployment_recommendation=deployment_recommendation,
            risk_adjusted_ranking=enhanced_metrics['risk_adjusted_ranking'],
            overfitting_risk_score=enhanced_metrics['overfitting_risk_score'],
            statistical_confidence=enhanced_metrics['statistical_confidence'],
            spa_gate_passed=significance_results.spa_is_significant,
            pbo_gate_passed=not significance_results.pbo_is_overfitted,
            overall_deployment_approved=deployment_recommendation['deploy'],
            model_comparison_analysis=enhanced_metrics['model_comparison'],
            sensitivity_analysis=enhanced_metrics['sensitivity_analysis'],
            timestamp=datetime.now()
        )
        
        logger.info(f"Enhanced evaluation completed. Deployment approved: "
                   f"{enhanced_result.overall_deployment_approved}")
        
        return enhanced_result
    
    def _extract_strategy_returns(self, 
                                base_result: EvaluationResult,
                                data: pd.DataFrame) -> np.ndarray:
        """Extract strategy returns from model evaluation results."""
        strategy_returns = []
        
        for model_name, metrics in base_result.model_metrics.items():
            # Calculate returns based on model predictions
            # This is a simplified approach - in practice, you'd use actual backtest returns
            returns = self._calculate_model_returns(metrics, data)
            if returns is not None:
                strategy_returns.append(returns)
        
        return np.array(strategy_returns)
    
    def _calculate_model_returns(self, 
                               metrics: ModelMetrics,
                               data: pd.DataFrame) -> Optional[np.ndarray]:
        """Calculate returns for a specific model based on its performance."""
        # Simplified return calculation based on model metrics
        # In practice, you'd use actual backtest or out-of-sample predictions
        
        try:
            # Generate synthetic returns based on model performance
            n_periods = min(252, len(data))  # Use up to 1 year of data
            
            # Base return level from Sharpe ratio and volatility
            annual_return = metrics.sharpe_ratio * metrics.volatility
            daily_return = annual_return / 252
            daily_vol = metrics.volatility / np.sqrt(252)
            
            # Generate returns with some randomness
            np.random.seed(hash(metrics.model_name) % (2**32))
            returns = np.random.normal(daily_return, daily_vol, n_periods)
            
            # Apply hit rate and directional accuracy
            if hasattr(metrics, 'hit_rate') and metrics.hit_rate > 0:
                # Adjust returns based on hit rate
                success_mask = np.random.random(n_periods) < metrics.hit_rate
                returns = np.where(success_mask, np.abs(returns), -np.abs(returns))
            
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating returns for {metrics.model_name}: {e}")
            return None
    
    def _create_benchmark_returns(self, 
                                data: pd.DataFrame,
                                target_column: str) -> pd.Series:
        """Create benchmark returns from data."""
        # Simple benchmark: use target variable returns or market proxy
        if target_column in data.columns:
            # Use actual target returns as benchmark
            target_data = data[target_column].dropna()
            if len(target_data) > 0:
                return target_data
        
        # Fallback: generate market-like returns
        n_periods = min(252, len(data))
        np.random.seed(42)  # Fixed seed for reproducibility
        market_returns = np.random.normal(0.0004, 0.015, n_periods)  # ~10% annual, 15% vol
        
        return pd.Series(market_returns, name='benchmark_returns')
    
    async def _run_significance_tests(self,
                                    strategy_returns: np.ndarray,
                                    benchmark_returns: pd.Series,
                                    base_result: EvaluationResult) -> SignificanceMetrics:
        """Run comprehensive significance testing."""
        try:
            # Align data lengths
            min_length = min(strategy_returns.shape[1], len(benchmark_returns))
            strategy_returns = strategy_returns[:, :min_length]
            benchmark_returns = benchmark_returns.iloc[:min_length]
            
            # Create strategy names
            strategy_names = list(base_result.model_metrics.keys())
            
            # Run comprehensive significance testing
            test_suite = SignificanceTestSuite(
                strategy_returns=strategy_returns,
                benchmark_returns=benchmark_returns.values,
                strategy_names=strategy_names
            )
            
            results = test_suite.run_comprehensive_test(
                confidence_level=0.95,
                bootstrap_iterations=5000  # Reduced for faster execution
            )
            
            # Extract key metrics
            spa_result = results.get('spa_test')
            pbo_result = results.get('pbo_test')
            deflated_sharpe_results = results.get('deflated_sharpe', [])
            
            # Get best performing strategy metrics
            best_strategy = base_result.best_model
            best_dsr_result = None
            
            for dsr in deflated_sharpe_results:
                if dsr['strategy'] == best_strategy:
                    best_dsr_result = dsr['result']
                    break
            
            if best_dsr_result is None and deflated_sharpe_results:
                best_dsr_result = deflated_sharpe_results[0]['result']
            
            # Validate deployment gates
            spa_p_value = spa_result.spa_p_value if spa_result else 1.0
            pbo_estimate = pbo_result.pbo_estimate if pbo_result else 0.5
            
            gate_results = validate_deployment_gates(
                spa_p_value=spa_p_value,
                pbo_estimate=pbo_estimate,
                spa_threshold=self.spa_threshold,
                pbo_threshold=self.pbo_threshold
            )
            
            # Generate interpretation
            interpretation = self._generate_significance_interpretation(
                spa_result, pbo_result, best_dsr_result, gate_results
            )
            
            return SignificanceMetrics(
                spa_p_value=spa_p_value,
                spa_is_significant=spa_result.is_significant if spa_result else False,
                deflated_sharpe=best_dsr_result.deflated_sharpe if best_dsr_result else 0.0,
                deflated_sharpe_p_value=best_dsr_result.p_value if best_dsr_result else 1.0,
                deflated_sharpe_significant=best_dsr_result.is_significant if best_dsr_result else False,
                pbo_estimate=pbo_estimate,
                pbo_is_overfitted=pbo_result.is_overfitted if pbo_result else True,
                deployment_gate_passed=gate_results['overall_gate_passed'],
                confidence_level='95%',
                n_strategies_tested=len(strategy_names),
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.error(f"Significance testing failed: {e}")
            return self._create_default_significance_results()
    
    def _create_default_significance_results(self) -> SignificanceMetrics:
        """Create default significance results when testing fails."""
        return SignificanceMetrics(
            spa_p_value=1.0,
            spa_is_significant=False,
            deflated_sharpe=0.0,
            deflated_sharpe_p_value=1.0,
            deflated_sharpe_significant=False,
            pbo_estimate=0.5,
            pbo_is_overfitted=True,
            deployment_gate_passed=False,
            confidence_level='N/A',
            n_strategies_tested=0,
            interpretation="Insufficient data for significance testing"
        )
    
    def _generate_significance_interpretation(self,
                                           spa_result, pbo_result, 
                                           dsr_result, gate_results) -> str:
        """Generate human-readable interpretation of significance results."""
        interpretations = []
        
        if spa_result and spa_result.is_significant:
            interpretations.append(f"SPA test confirms significant outperformance (p={spa_result.spa_p_value:.4f})")
        elif spa_result:
            interpretations.append(f"SPA test suggests potential data snooping (p={spa_result.spa_p_value:.4f})")
        
        if pbo_result:
            if pbo_result.is_overfitted:
                interpretations.append(f"High overfitting risk (PBO={pbo_result.pbo_estimate:.3f})")
            else:
                interpretations.append(f"Low overfitting risk (PBO={pbo_result.pbo_estimate:.3f})")
        
        if dsr_result:
            if dsr_result.is_significant:
                interpretations.append(f"Deflated Sharpe ratio confirms performance significance")
            else:
                interpretations.append(f"Deflated Sharpe ratio suggests inflated performance metrics")
        
        if gate_results['overall_gate_passed']:
            interpretations.append("All deployment gates passed - recommended for production")
        else:
            interpretations.append("Deployment gates failed - not recommended for production")
        
        return "; ".join(interpretations) if interpretations else "No significant findings"
    
    def _generate_deployment_recommendation(self,
                                          base_result: EvaluationResult,
                                          significance_result: SignificanceMetrics) -> Dict[str, Any]:
        """Generate comprehensive deployment recommendation."""
        # Base recommendation from traditional metrics
        base_confidence = self._assess_base_confidence(base_result)
        
        # Significance-based recommendation
        sig_confidence = self._assess_significance_confidence(significance_result)
        
        # Combined recommendation
        overall_confidence = min(base_confidence, sig_confidence)
        
        # Deployment decision
        deploy = (
            significance_result.deployment_gate_passed and
            overall_confidence >= 0.7 and
            base_result.best_score > 0.6  # Minimum performance threshold
        )
        
        # Risk factors
        risk_factors = []
        if significance_result.pbo_is_overfitted:
            risk_factors.append("High probability of backtest overfitting")
        if not significance_result.spa_is_significant:
            risk_factors.append("No evidence of genuine predictive ability")
        if not significance_result.deflated_sharpe_significant:
            risk_factors.append("Performance metrics may be inflated")
        
        # Benefits
        benefits = []
        if significance_result.spa_is_significant:
            benefits.append("Statistically significant outperformance confirmed")
        if not significance_result.pbo_is_overfitted:
            benefits.append("Low risk of overfitting")
        if significance_result.deflated_sharpe_significant:
            benefits.append("Robust performance after multiple testing correction")
        
        return {
            'deploy': deploy,
            'confidence': overall_confidence,
            'confidence_level': self._confidence_to_label(overall_confidence),
            'risk_factors': risk_factors,
            'benefits': benefits,
            'recommended_actions': self._generate_recommended_actions(
                deploy, risk_factors, base_result
            ),
            'monitoring_requirements': self._generate_monitoring_requirements(
                significance_result
            )
        }
    
    def _assess_base_confidence(self, base_result: EvaluationResult) -> float:
        """Assess confidence based on traditional evaluation metrics."""
        best_metrics = base_result.model_metrics[base_result.best_model]
        
        confidence_factors = []
        
        # Sharpe ratio confidence
        if best_metrics.sharpe_ratio > 2.0:
            confidence_factors.append(0.9)
        elif best_metrics.sharpe_ratio > 1.0:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        # R-squared confidence
        if best_metrics.r2 > 0.3:
            confidence_factors.append(0.8)
        elif best_metrics.r2 > 0.1:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        # Cross-validation stability
        if best_metrics.cv_std < 0.1:
            confidence_factors.append(0.8)
        elif best_metrics.cv_std < 0.2:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        return np.mean(confidence_factors)
    
    def _assess_significance_confidence(self, sig_result: SignificanceMetrics) -> float:
        """Assess confidence based on significance testing."""
        confidence_factors = []
        
        # SPA test confidence
        if sig_result.spa_is_significant:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.3)
        
        # PBO confidence
        if not sig_result.pbo_is_overfitted:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.2)
        
        # Deflated Sharpe confidence
        if sig_result.deflated_sharpe_significant:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        return np.mean(confidence_factors)
    
    def _confidence_to_label(self, confidence: float) -> str:
        """Convert confidence score to label."""
        if confidence >= 0.8:
            return "High"
        elif confidence >= 0.6:
            return "Medium"
        elif confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_recommended_actions(self,
                                    deploy: bool,
                                    risk_factors: List[str],
                                    base_result: EvaluationResult) -> List[str]:
        """Generate recommended actions based on evaluation results."""
        actions = []
        
        if deploy:
            actions.append("Proceed with deployment to production")
            actions.append("Implement gradual rollout with monitoring")
            actions.append("Set up real-time performance tracking")
        else:
            actions.append("Do not deploy to production")
            
            if "overfitting" in str(risk_factors).lower():
                actions.append("Collect more out-of-sample data for validation")
                actions.append("Simplify model complexity to reduce overfitting")
            
            if "predictive ability" in str(risk_factors).lower():
                actions.append("Investigate additional features or data sources")
                actions.append("Consider ensemble methods or alternative algorithms")
            
            actions.append("Repeat evaluation with extended test period")
        
        # Model-specific recommendations
        best_model = base_result.best_model
        if "RandomForest" in best_model:
            actions.append("Consider feature selection to improve interpretability")
        elif "LightGBM" in best_model or "XGBoost" in best_model:
            actions.append("Fine-tune regularization parameters")
            actions.append("Monitor for concept drift in production")
        
        return actions
    
    def _generate_monitoring_requirements(self, 
                                        sig_result: SignificanceMetrics) -> List[str]:
        """Generate monitoring requirements for deployed models."""
        requirements = [
            "Monitor daily Sharpe ratio vs. expected performance",
            "Track cumulative returns vs. benchmark",
            "Alert on maximum drawdown exceeding 10%"
        ]
        
        if sig_result.pbo_estimate > 0.1:
            requirements.append("Enhanced overfitting monitoring with out-of-sample validation")
        
        if not sig_result.spa_is_significant:
            requirements.append("Weekly significance testing of live performance")
        
        requirements.extend([
            "Monthly model performance review",
            "Quarterly revalidation of statistical assumptions",
            "Annual comprehensive model evaluation"
        ])
        
        return requirements
    
    def _calculate_enhanced_metrics(self,
                                  base_result: EvaluationResult,
                                  sig_result: SignificanceMetrics) -> Dict[str, Any]:
        """Calculate enhanced evaluation metrics."""
        # Risk-adjusted ranking
        risk_adjusted_ranking = {}
        for model_name, metrics in base_result.model_metrics.items():
            # Combine traditional metrics with significance-based adjustments
            base_score = metrics.sharpe_ratio * 0.4 + metrics.r2 * 0.3 + metrics.hit_rate * 0.3
            
            # Apply significance penalties
            significance_penalty = 1.0
            if sig_result.pbo_is_overfitted:
                significance_penalty *= 0.7
            if not sig_result.spa_is_significant:
                significance_penalty *= 0.8
            
            risk_adjusted_ranking[model_name] = base_score * significance_penalty
        
        # Overfitting risk score
        overfitting_risk = sig_result.pbo_estimate * 100  # Convert to percentage
        
        # Statistical confidence
        statistical_confidence = self._assess_significance_confidence(sig_result)
        
        # Model comparison analysis
        model_comparison = {
            'best_traditional': base_result.best_model,
            'best_risk_adjusted': max(risk_adjusted_ranking.items(), key=lambda x: x[1])[0],
            'performance_gap': max(risk_adjusted_ranking.values()) - min(risk_adjusted_ranking.values()),
            'consistency_score': 1.0 - np.std(list(risk_adjusted_ranking.values()))
        }
        
        # Sensitivity analysis
        sensitivity_analysis = {
            'spa_threshold_sensitivity': self._analyze_spa_sensitivity(sig_result),
            'pbo_threshold_sensitivity': self._analyze_pbo_sensitivity(sig_result),
            'confidence_level_impact': self._analyze_confidence_impact(sig_result)
        }
        
        return {
            'risk_adjusted_ranking': risk_adjusted_ranking,
            'overfitting_risk_score': overfitting_risk,
            'statistical_confidence': self._confidence_to_label(statistical_confidence),
            'model_comparison': model_comparison,
            'sensitivity_analysis': sensitivity_analysis
        }
    
    def _analyze_spa_sensitivity(self, sig_result: SignificanceMetrics) -> Dict[str, Any]:
        """Analyze sensitivity to SPA threshold changes."""
        current_p = sig_result.spa_p_value
        
        thresholds = [0.01, 0.05, 0.10, 0.15, 0.20]
        sensitivity = {}
        
        for threshold in thresholds:
            sensitivity[f"threshold_{threshold}"] = current_p < threshold
        
        return {
            'current_p_value': current_p,
            'threshold_analysis': sensitivity,
            'recommendation': "Robust" if current_p < 0.01 else "Borderline" if current_p < 0.10 else "Weak"
        }
    
    def _analyze_pbo_sensitivity(self, sig_result: SignificanceMetrics) -> Dict[str, Any]:
        """Analyze sensitivity to PBO threshold changes."""
        current_pbo = sig_result.pbo_estimate
        
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
        sensitivity = {}
        
        for threshold in thresholds:
            sensitivity[f"threshold_{threshold}"] = current_pbo <= threshold
        
        return {
            'current_pbo': current_pbo,
            'threshold_analysis': sensitivity,
            'recommendation': "Low Risk" if current_pbo < 0.1 else "Medium Risk" if current_pbo < 0.2 else "High Risk"
        }
    
    def _analyze_confidence_impact(self, sig_result: SignificanceMetrics) -> Dict[str, Any]:
        """Analyze impact of different confidence levels."""
        return {
            'current_level': sig_result.confidence_level,
            'robustness': "High" if sig_result.spa_p_value < 0.01 else "Medium" if sig_result.spa_p_value < 0.05 else "Low",
            'recommendation': "Use 99% confidence for critical deployments" if sig_result.spa_p_value < 0.01 else "Standard 95% confidence acceptable"
        }
    
    def export_enhanced_results(self, result: EnhancedEvaluationResult) -> Dict[str, Any]:
        """Export enhanced results in API-friendly format."""
        return {
            'evaluation_summary': {
                'best_model': result.base_evaluation.best_model,
                'deployment_approved': result.overall_deployment_approved,
                'confidence_level': result.statistical_confidence,
                'overfitting_risk': result.overfitting_risk_score,
                'timestamp': result.timestamp.isoformat()
            },
            'significance_testing': {
                'spa_test': {
                    'p_value': result.significance_metrics.spa_p_value,
                    'is_significant': result.significance_metrics.spa_is_significant,
                    'gate_passed': result.spa_gate_passed
                },
                'deflated_sharpe': {
                    'value': result.significance_metrics.deflated_sharpe,
                    'p_value': result.significance_metrics.deflated_sharpe_p_value,
                    'is_significant': result.significance_metrics.deflated_sharpe_significant
                },
                'pbo_test': {
                    'estimate': result.significance_metrics.pbo_estimate,
                    'is_overfitted': result.significance_metrics.pbo_is_overfitted,
                    'gate_passed': result.pbo_gate_passed
                },
                'interpretation': result.significance_metrics.interpretation
            },
            'deployment_recommendation': result.deployment_recommendation,
            'model_rankings': {
                'traditional': {model: metrics.sharpe_ratio for model, metrics in result.base_evaluation.model_metrics.items()},
                'risk_adjusted': result.risk_adjusted_ranking
            },
            'model_comparison': result.model_comparison_analysis,
            'sensitivity_analysis': result.sensitivity_analysis
        }