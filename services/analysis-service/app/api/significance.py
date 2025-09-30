"""
API endpoints for statistical significance testing.

Provides REST API access to sophisticated statistical tests for trading strategies:
- Superior Predictive Ability (SPA) tests
- Deflated Sharpe Ratio calculations
- Probability of Backtest Overfitting (PBO) estimation
- Deployment gate validation
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from services.significance_tests import (
    SignificanceTestSuite, WhiteRealityCheck, DeflatedSharpe, PBOEstimator,
    validate_deployment_gates, format_test_results_for_api
)

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class StrategyReturnsRequest(BaseModel):
    """Request model for strategy returns data."""
    strategy_returns: List[List[float]] = Field(..., description="Matrix of strategy returns (strategies x time periods)")
    benchmark_returns: List[float] = Field(..., description="Benchmark returns time series")
    strategy_names: Optional[List[str]] = Field(None, description="Optional names for strategies")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99, description="Confidence level for tests")
    bootstrap_iterations: int = Field(10000, ge=1000, le=50000, description="Bootstrap iterations for SPA test")


class SPATestRequest(BaseModel):
    """Request model for SPA test."""
    strategy_returns: List[List[float]] = Field(..., description="Matrix of strategy returns")
    benchmark_returns: List[float] = Field(..., description="Benchmark returns")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99)
    bootstrap_iterations: int = Field(10000, ge=1000, le=50000)
    block_length: Optional[int] = Field(None, description="Block length for bootstrap (auto if None)")


class DeflatedSharpeRequest(BaseModel):
    """Request model for Deflated Sharpe Ratio test."""
    returns: List[float] = Field(..., description="Strategy returns")
    trials: int = Field(..., ge=1, description="Number of trials/strategies tested")
    sample_length: Optional[int] = Field(None, description="Sample length override")


class PBORequest(BaseModel):
    """Request model for PBO estimation."""
    returns_matrix: List[List[float]] = Field(..., description="Matrix of strategy returns")
    threshold: float = Field(0.0, description="Performance threshold")
    n_splits: int = Field(16, ge=4, le=64, description="Number of combinatorial splits")


class DeploymentGateRequest(BaseModel):
    """Request model for deployment gate validation."""
    spa_p_value: float = Field(..., ge=0.0, le=1.0, description="SPA test p-value")
    pbo_estimate: float = Field(..., ge=0.0, le=1.0, description="PBO estimate")
    spa_threshold: float = Field(0.05, ge=0.01, le=0.1, description="SPA p-value threshold")
    pbo_threshold: float = Field(0.2, ge=0.1, le=0.5, description="Maximum allowed PBO")


class SignificanceTestResponse(BaseModel):
    """Response model for significance tests."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime


# API Endpoints

@router.post("/spa", response_model=SignificanceTestResponse)
async def run_spa_test(request: SPATestRequest):
    """
    Run Superior Predictive Ability (SPA) test.
    
    Tests whether the best performing strategy is significantly better than benchmark
    after accounting for data snooping bias across multiple strategies.
    """
    try:
        # Validate input data
        strategy_returns = np.array(request.strategy_returns)
        benchmark_returns = np.array(request.benchmark_returns)
        
        if strategy_returns.ndim == 1:
            strategy_returns = strategy_returns.reshape(1, -1)
        
        if strategy_returns.shape[1] != len(benchmark_returns):
            raise HTTPException(
                status_code=400,
                detail="Strategy returns and benchmark returns must have the same time length"
            )
        
        # Run SPA test
        spa_test = WhiteRealityCheck(
            benchmark_returns=benchmark_returns,
            strategy_returns=strategy_returns,
            bootstrap_iterations=request.bootstrap_iterations,
            block_length=request.block_length
        )
        
        result = spa_test.run_test(confidence_level=request.confidence_level)
        
        # Format response
        response_data = {
            'spa_statistic': result.spa_statistic,
            'spa_p_value': result.spa_p_value,
            'rc_statistic': result.rc_statistic,
            'rc_p_value': result.rc_p_value,
            'bootstrap_iterations': result.bootstrap_iterations,
            'benchmark_performance': result.benchmark_performance,
            'best_strategy_performance': result.best_strategy_performance,
            'num_strategies': result.num_strategies,
            'is_significant': result.is_significant,
            'interpretation': result.interpretation,
            'deployment_gate_passed': result.spa_p_value < 0.05  # Standard threshold
        }
        
        return SignificanceTestResponse(
            success=True,
            message="SPA test completed successfully",
            data=response_data,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"SPA test failed: {e}")
        raise HTTPException(status_code=500, detail=f"SPA test failed: {str(e)}")


@router.post("/deflated-sharpe", response_model=SignificanceTestResponse)
async def calculate_deflated_sharpe(request: DeflatedSharpeRequest):
    """
    Calculate Deflated Sharpe Ratio.
    
    Corrects Sharpe ratio for multiple testing and non-normal returns,
    providing a more conservative estimate of strategy performance.
    """
    try:
        returns = np.array(request.returns)
        
        if len(returns) < 30:
            raise HTTPException(
                status_code=400,
                detail="Minimum 30 observations required for Deflated Sharpe Ratio calculation"
            )
        
        # Calculate deflated Sharpe ratio
        result = DeflatedSharpe.calculate_deflated_sharpe(
            returns=returns,
            trials=request.trials,
            length=request.sample_length
        )
        
        # Format response
        response_data = {
            'observed_sharpe': result.observed_sharpe,
            'deflated_sharpe': result.deflated_sharpe,
            'p_value': result.p_value,
            'trials': result.trials,
            'sample_length': result.length,
            'skewness': result.skewness,
            'kurtosis': result.kurtosis,
            'is_significant': result.is_significant,
            'interpretation': result.interpretation
        }
        
        return SignificanceTestResponse(
            success=True,
            message="Deflated Sharpe Ratio calculated successfully",
            data=response_data,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Deflated Sharpe calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deflated Sharpe calculation failed: {str(e)}")


@router.post("/pbo", response_model=SignificanceTestResponse)
async def estimate_pbo(request: PBORequest):
    """
    Estimate Probability of Backtest Overfitting (PBO).
    
    Estimates the probability that observed backtest performance
    is due to overfitting rather than genuine predictive ability.
    """
    try:
        returns_matrix = np.array(request.returns_matrix)
        
        if returns_matrix.ndim == 1:
            returns_matrix = returns_matrix.reshape(1, -1)
        
        if returns_matrix.shape[1] < 100:
            raise HTTPException(
                status_code=400,
                detail="Minimum 100 time periods required for PBO estimation"
            )
        
        # Estimate PBO
        pbo_estimator = PBOEstimator(
            returns_matrix=returns_matrix,
            threshold=request.threshold
        )
        
        result = pbo_estimator.estimate_pbo(n_splits=request.n_splits)
        
        # Format response
        response_data = {
            'pbo_estimate': result.pbo_estimate,
            'phi_estimate': result.phi_estimate,
            'is_overfitted': result.is_overfitted,
            'max_pbo_threshold': result.max_pbo_threshold,
            'n_strategies': returns_matrix.shape[0],
            'n_periods': returns_matrix.shape[1],
            'n_splits': request.n_splits,
            'interpretation': result.interpretation,
            'deployment_gate_passed': not result.is_overfitted
        }
        
        return SignificanceTestResponse(
            success=True,
            message="PBO estimation completed successfully",
            data=response_data,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"PBO estimation failed: {e}")
        raise HTTPException(status_code=500, detail=f"PBO estimation failed: {str(e)}")


@router.post("/comprehensive", response_model=SignificanceTestResponse)
async def run_comprehensive_tests(request: StrategyReturnsRequest, background_tasks: BackgroundTasks):
    """
    Run comprehensive significance testing suite.
    
    Combines SPA test, Deflated Sharpe Ratio, and PBO estimation
    to provide robust evaluation of strategy performance.
    """
    try:
        # Validate input data
        strategy_returns = np.array(request.strategy_returns)
        benchmark_returns = np.array(request.benchmark_returns)
        
        if strategy_returns.ndim == 1:
            strategy_returns = strategy_returns.reshape(1, -1)
        
        if strategy_returns.shape[1] != len(benchmark_returns):
            raise HTTPException(
                status_code=400,
                detail="Strategy returns and benchmark returns must have the same time length"
            )
        
        # Run comprehensive test suite
        test_suite = SignificanceTestSuite(
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            strategy_names=request.strategy_names
        )
        
        results = test_suite.run_comprehensive_test(
            confidence_level=request.confidence_level,
            bootstrap_iterations=request.bootstrap_iterations
        )
        
        # Format results for API response
        formatted_results = format_test_results_for_api(results)
        
        return SignificanceTestResponse(
            success=True,
            message="Comprehensive significance testing completed successfully",
            data=formatted_results,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Comprehensive testing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comprehensive testing failed: {str(e)}")


@router.post("/deployment-gates", response_model=SignificanceTestResponse)
async def validate_deployment(request: DeploymentGateRequest):
    """
    Validate deployment gates based on significance tests.
    
    Checks if strategy performance meets statistical significance thresholds
    required for production deployment.
    """
    try:
        # Validate deployment gates
        gate_results = validate_deployment_gates(
            spa_p_value=request.spa_p_value,
            pbo_estimate=request.pbo_estimate,
            spa_threshold=request.spa_threshold,
            pbo_threshold=request.pbo_threshold
        )
        
        # Add deployment recommendation
        if gate_results['overall_gate_passed']:
            recommendation = {
                'deploy': True,
                'confidence': 'high' if gate_results['spa_gate_passed'] and gate_results['pbo_gate_passed'] else 'medium',
                'message': 'Strategy meets statistical significance requirements for deployment'
            }
        else:
            recommendation = {
                'deploy': False,
                'confidence': 'low',
                'message': 'Strategy does not meet statistical significance requirements'
            }
        
        gate_results['recommendation'] = recommendation
        
        return SignificanceTestResponse(
            success=True,
            message="Deployment gate validation completed",
            data=gate_results,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Deployment gate validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deployment gate validation failed: {str(e)}")


@router.get("/gates/status")
async def get_deployment_gates_status():
    """Get current deployment gate thresholds and status."""
    return {
        'gates': {
            'spa_threshold': 0.05,
            'pbo_threshold': 0.2,
            'description': 'Statistical significance thresholds for deployment'
        },
        'requirements': {
            'spa_test': 'p-value must be < 0.05 (95% confidence)',
            'pbo_test': 'PBO estimate must be <= 0.2 (20% overfitting risk)',
            'deflated_sharpe': 'Deflated Sharpe ratio should be positive and significant'
        },
        'documentation': {
            'spa_test': 'Tests for genuine predictive ability after correcting for data snooping',
            'pbo_test': 'Estimates probability that backtest performance is due to overfitting',
            'deflated_sharpe': 'Corrects Sharpe ratio for multiple testing and non-normality'
        }
    }


@router.get("/health")
async def health_check():
    """Health check endpoint for significance testing service."""
    return {
        'status': 'healthy',
        'service': 'significance-testing',
        'timestamp': datetime.now(),
        'available_tests': [
            'spa',
            'deflated-sharpe', 
            'pbo',
            'comprehensive',
            'deployment-gates'
        ]
    }


# Utility endpoints for testing and validation

@router.post("/validate-returns")
async def validate_returns_data(
    strategy_returns: List[List[float]] = Query(..., description="Strategy returns matrix"),
    benchmark_returns: List[float] = Query(..., description="Benchmark returns")
):
    """Validate returns data format and quality."""
    try:
        strategy_returns = np.array(strategy_returns)
        benchmark_returns = np.array(benchmark_returns)
        
        if strategy_returns.ndim == 1:
            strategy_returns = strategy_returns.reshape(1, -1)
        
        validation_results = {
            'data_shape': {
                'n_strategies': strategy_returns.shape[0],
                'n_periods': strategy_returns.shape[1],
                'benchmark_length': len(benchmark_returns)
            },
            'data_quality': {
                'has_missing_values': np.isnan(strategy_returns).any() or np.isnan(benchmark_returns).any(),
                'has_infinite_values': np.isinf(strategy_returns).any() or np.isinf(benchmark_returns).any(),
                'has_zero_variance': np.any(np.std(strategy_returns, axis=1) == 0),
                'length_match': strategy_returns.shape[1] == len(benchmark_returns)
            },
            'descriptive_stats': {
                'strategy_means': np.mean(strategy_returns, axis=1).tolist(),
                'strategy_stds': np.std(strategy_returns, axis=1, ddof=1).tolist(),
                'benchmark_mean': np.mean(benchmark_returns),
                'benchmark_std': np.std(benchmark_returns, ddof=1)
            },
            'recommendations': []
        }
        
        # Add recommendations based on validation
        if validation_results['data_quality']['has_missing_values']:
            validation_results['recommendations'].append("Remove or impute missing values before testing")
        
        if validation_results['data_quality']['has_infinite_values']:
            validation_results['recommendations'].append("Remove infinite values before testing")
        
        if validation_results['data_quality']['has_zero_variance']:
            validation_results['recommendations'].append("Remove strategies with zero variance")
        
        if not validation_results['data_quality']['length_match']:
            validation_results['recommendations'].append("Ensure strategy and benchmark returns have same length")
        
        if strategy_returns.shape[1] < 100:
            validation_results['recommendations'].append("Consider using more historical data (minimum 100 observations recommended)")
        
        return {
            'valid': len(validation_results['recommendations']) == 0,
            'validation_results': validation_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data validation failed: {str(e)}")


@router.get("/documentation")
async def get_api_documentation():
    """Get comprehensive API documentation for significance testing endpoints."""
    return {
        'overview': 'Statistical significance testing API for trading strategy evaluation',
        'endpoints': {
            '/spa': {
                'method': 'POST',
                'description': 'Superior Predictive Ability test',
                'purpose': 'Tests for genuine outperformance after correcting for data snooping',
                'key_output': 'spa_p_value (< 0.05 indicates significant performance)'
            },
            '/deflated-sharpe': {
                'method': 'POST', 
                'description': 'Deflated Sharpe Ratio calculation',
                'purpose': 'Corrects Sharpe ratio for multiple testing and non-normality',
                'key_output': 'deflated_sharpe (positive values indicate good performance)'
            },
            '/pbo': {
                'method': 'POST',
                'description': 'Probability of Backtest Overfitting estimation',
                'purpose': 'Estimates risk that performance is due to overfitting',
                'key_output': 'pbo_estimate (< 0.2 indicates low overfitting risk)'
            },
            '/comprehensive': {
                'method': 'POST',
                'description': 'Run all significance tests together',
                'purpose': 'Comprehensive evaluation with deployment recommendation',
                'key_output': 'deployment_recommendation with confidence level'
            },
            '/deployment-gates': {
                'method': 'POST',
                'description': 'Validate deployment gates',
                'purpose': 'Check if strategy meets significance thresholds for deployment',
                'key_output': 'overall_gate_passed boolean'
            }
        },
        'deployment_criteria': {
            'spa_p_value': '< 0.05 (95% confidence of genuine performance)',
            'pbo_estimate': '<= 0.2 (20% maximum overfitting risk)',
            'deflated_sharpe': 'Positive and statistically significant'
        },
        'best_practices': [
            'Use at least 100 observations for reliable results',
            'Test multiple strategies to account for selection bias',
            'Consider transaction costs and market impact in returns',
            'Validate results with out-of-sample testing',
            'Monitor live performance for model degradation'
        ]
    }