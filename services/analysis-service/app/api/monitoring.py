"""
API endpoints for Monitoring & Attribution System.

Provides REST API access to comprehensive monitoring and attribution framework:
- Alpha-decay detection and tracking
- Multi-factor P&L attribution analysis
- Performance monitoring with SLOs
- Model health assessment and scoring
- Real-time alerting and notifications
- Risk attribution and factor exposure tracking
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from services.monitoring_attribution import (
    ComprehensiveMonitoringSystem, AlphaDecayAnalyzer, FactorAttributionEngine,
    SLOMonitor, PerformanceMonitor, ModelHealthAnalyzer, SLOTarget,
    AlphaDecayMetrics, AttributionResult, PerformanceMetrics, ModelHealthScore
)

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class TimeSeriesData(BaseModel):
    """Time series data point."""
    timestamp: datetime
    value: float


class MonitoringRequest(BaseModel):
    """Request model for comprehensive monitoring."""
    strategy_returns: List[TimeSeriesData] = Field(..., description="Strategy returns time series")
    benchmark_returns: List[TimeSeriesData] = Field(..., description="Benchmark returns time series")
    factor_returns: List[Dict[str, Any]] = Field(..., description="Factor returns data")
    factor_names: List[str] = Field(..., description="Factor names for attribution")
    
    analysis_date: Optional[datetime] = Field(None, description="Analysis date (default: now)")
    lookback_days: int = Field(252, ge=60, le=1000, description="Lookback period in days")


class AlphaDecayRequest(BaseModel):
    """Request model for alpha decay analysis."""
    strategy_returns: List[TimeSeriesData] = Field(..., description="Strategy returns")
    benchmark_returns: List[TimeSeriesData] = Field(..., description="Benchmark returns")
    lookback_periods: Optional[List[int]] = Field(None, description="Custom lookback periods")
    confidence_level: float = Field(0.95, ge=0.8, le=0.99, description="Confidence level")


class AttributionRequest(BaseModel):
    """Request model for factor attribution analysis."""
    portfolio_returns: List[TimeSeriesData] = Field(..., description="Portfolio returns")
    factor_returns: List[Dict[str, Any]] = Field(..., description="Factor returns data")
    factor_names: List[str] = Field(..., description="Factor names")
    attribution_method: str = Field("regression", description="Attribution method")
    rolling_analysis: bool = Field(False, description="Enable rolling attribution")
    rolling_window: int = Field(60, ge=30, le=252, description="Rolling window size")
    
    @validator('attribution_method')
    def validate_attribution_method(cls, v):
        allowed_methods = ['regression', 'ridge']
        if v not in allowed_methods:
            raise ValueError(f"Attribution method must be one of {allowed_methods}")
        return v


class SLOConfigRequest(BaseModel):
    """Request model for SLO configuration."""
    metric_name: str = Field(..., description="Metric name to monitor")
    target_value: float = Field(..., description="Target value for the metric")
    tolerance: float = Field(..., gt=0, description="Tolerance around target")
    operator: str = Field(..., description="Comparison operator")
    frequency: str = Field("daily", description="Monitoring frequency")
    description: str = Field(..., description="SLO description")
    
    @validator('operator')
    def validate_operator(cls, v):
        allowed_ops = ['gt', 'lt', 'gte', 'lte', 'eq']
        if v not in allowed_ops:
            raise ValueError(f"Operator must be one of {allowed_ops}")
        return v
    
    @validator('frequency')
    def validate_frequency(cls, v):
        allowed_freq = ['daily', 'weekly', 'monthly']
        if v not in allowed_freq:
            raise ValueError(f"Frequency must be one of {allowed_freq}")
        return v


class PerformanceAnalysisRequest(BaseModel):
    """Request model for performance analysis."""
    returns: List[TimeSeriesData] = Field(..., description="Return time series")
    prices: Optional[List[TimeSeriesData]] = Field(None, description="Price time series")
    trades: Optional[List[Dict[str, Any]]] = Field(None, description="Trade data")
    benchmark_returns: Optional[List[TimeSeriesData]] = Field(None, description="Benchmark for comparison")


class HealthAssessmentRequest(BaseModel):
    """Request model for model health assessment."""
    strategy_returns: List[TimeSeriesData] = Field(..., description="Strategy returns")
    benchmark_returns: List[TimeSeriesData] = Field(..., description="Benchmark returns")
    factor_returns: List[Dict[str, Any]] = Field(..., description="Factor returns")
    factor_names: List[str] = Field(..., description="Factor names")
    assessment_date: Optional[datetime] = Field(None, description="Assessment date")


class MonitoringResponse(BaseModel):
    """Response model for monitoring operations."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime
    processing_time_ms: Optional[float] = None


# Utility functions
def convert_timeseries_to_pandas(data: List[TimeSeriesData]) -> pd.Series:
    """Convert TimeSeriesData list to pandas Series."""
    timestamps = [point.timestamp for point in data]
    values = [point.value for point in data]
    return pd.Series(values, index=timestamps).sort_index()


def convert_factor_returns_to_dataframe(
    factor_data: List[Dict[str, Any]], 
    factor_names: List[str]
) -> pd.DataFrame:
    """Convert factor returns data to pandas DataFrame."""
    
    # Assuming factor_data contains dicts with timestamp and factor values
    df_data = []
    for record in factor_data:
        row = {'timestamp': record['timestamp']}
        for factor_name in factor_names:
            row[factor_name] = record.get(factor_name, 0.0)
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    return df


# API Endpoints

@router.post("/comprehensive", response_model=MonitoringResponse)
async def run_comprehensive_monitoring(request: MonitoringRequest):
    """
    Run comprehensive monitoring analysis including alpha decay, attribution, 
    performance metrics, SLO compliance, and model health assessment.
    """
    start_time = datetime.now()
    
    try:
        # Convert input data
        strategy_returns = convert_timeseries_to_pandas(request.strategy_returns)
        benchmark_returns = convert_timeseries_to_pandas(request.benchmark_returns)
        factor_returns = convert_factor_returns_to_dataframe(
            request.factor_returns, request.factor_names
        )
        
        # Filter by lookback period
        if request.lookback_days:
            cutoff_date = datetime.now() - timedelta(days=request.lookback_days)
            strategy_returns = strategy_returns[strategy_returns.index >= cutoff_date]
            benchmark_returns = benchmark_returns[benchmark_returns.index >= cutoff_date]
            factor_returns = factor_returns[factor_returns.index >= cutoff_date]
        
        # Validate data alignment
        if len(strategy_returns) < 30:
            raise HTTPException(
                status_code=400,
                detail="Insufficient strategy return data (minimum 30 observations)"
            )
        
        # Initialize monitoring system
        monitoring_system = ComprehensiveMonitoringSystem(request.factor_names)
        
        # Run comprehensive monitoring
        analysis_date = request.analysis_date or datetime.now()
        results = monitoring_system.run_comprehensive_monitoring(
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            factor_returns=factor_returns,
            timestamp=analysis_date
        )
        
        # Convert results to serializable format
        monitoring_data = {}
        
        # Alpha decay results
        if results['monitoring_results']['alpha_decay']:
            alpha_decay = results['monitoring_results']['alpha_decay']
            monitoring_data['alpha_decay'] = {
                'lookback_periods': alpha_decay.lookback_periods,
                'alpha_estimates': alpha_decay.alpha_estimates,
                'decay_rate': alpha_decay.decay_rate,
                'half_life_days': alpha_decay.half_life_days,
                'is_significant_decay': alpha_decay.is_significant_decay,
                'decay_confidence': alpha_decay.decay_confidence,
                'decay_r_squared': alpha_decay.decay_r_squared
            }
        
        # Attribution results
        if results['monitoring_results']['attribution']:
            attribution = results['monitoring_results']['attribution']
            monitoring_data['attribution'] = {
                'total_pnl': attribution.total_pnl,
                'factor_attributions': attribution.factor_attributions,
                'specific_return': attribution.specific_return,
                'attribution_r_squared': attribution.attribution_r_squared,
                'explained_variance': attribution.explained_variance,
                'factor_exposures': attribution.factor_exposures
            }
        
        # Performance metrics
        if results['monitoring_results']['performance']:
            performance = results['monitoring_results']['performance']
            monitoring_data['performance'] = {
                'total_return': performance.total_return,
                'annualized_return': performance.annualized_return,
                'volatility': performance.volatility,
                'sharpe_ratio': performance.sharpe_ratio,
                'max_drawdown': performance.max_drawdown,
                'calmar_ratio': performance.calmar_ratio,
                'win_rate': performance.win_rate,
                'var_95': performance.var_95,
                'skewness': performance.skewness,
                'kurtosis': performance.kurtosis
            }
        
        # SLO violations
        slo_violations = results['monitoring_results'].get('slo_violations', [])
        monitoring_data['slo_violations'] = [
            {
                'slo_name': v.slo_name,
                'actual_value': v.actual_value,
                'target_value': v.target_value,
                'severity': v.severity,
                'description': v.description
            } for v in slo_violations
        ]
        
        # Model health score
        if results['monitoring_results']['health_score']:
            health = results['monitoring_results']['health_score']
            monitoring_data['health_score'] = {
                'overall_score': health.overall_score,
                'alpha_decay_score': health.alpha_decay_score,
                'attribution_quality_score': health.attribution_quality_score,
                'risk_score': health.risk_score,
                'performance_score': health.performance_score,
                'data_quality_score': health.data_quality_score,
                'alerts': health.alerts,
                'recommendations': health.recommendations
            }
        
        # Get monitoring summary
        summary = monitoring_system.get_monitoring_summary()
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return MonitoringResponse(
            success=True,
            message="Comprehensive monitoring analysis completed",
            data={
                'monitoring_results': monitoring_data,
                'monitoring_summary': summary,
                'analysis_period': {
                    'start_date': strategy_returns.index[0],
                    'end_date': strategy_returns.index[-1],
                    'total_observations': len(strategy_returns)
                },
                'factor_names': request.factor_names
            },
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Comprehensive monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Monitoring analysis failed: {str(e)}")


@router.post("/alpha-decay", response_model=MonitoringResponse)
async def analyze_alpha_decay(request: AlphaDecayRequest):
    """
    Analyze alpha decay patterns in strategy performance.
    
    Detects degradation of strategy alpha over different time horizons
    and estimates decay rates and half-life.
    """
    start_time = datetime.now()
    
    try:
        # Convert input data
        strategy_returns = convert_timeseries_to_pandas(request.strategy_returns)
        benchmark_returns = convert_timeseries_to_pandas(request.benchmark_returns)
        
        # Initialize analyzer
        analyzer = AlphaDecayAnalyzer(confidence_level=request.confidence_level)
        
        # Run alpha decay analysis
        decay_metrics = analyzer.analyze_alpha_decay(
            returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            lookback_periods=request.lookback_periods
        )
        
        # Format results
        results = {
            'alpha_decay_analysis': {
                'lookback_periods': decay_metrics.lookback_periods,
                'alpha_estimates': decay_metrics.alpha_estimates,
                't_statistics': decay_metrics.t_statistics,
                'p_values': decay_metrics.p_values,
                'decay_rate': decay_metrics.decay_rate,
                'half_life_days': decay_metrics.half_life_days,
                'decay_r_squared': decay_metrics.decay_r_squared,
                'is_significant_decay': decay_metrics.is_significant_decay,
                'decay_confidence': decay_metrics.decay_confidence
            },
            'interpretation': {
                'alpha_trend': 'decreasing' if decay_metrics.is_significant_decay else 'stable',
                'decay_severity': 'high' if decay_metrics.decay_confidence > 0.8 else 'low',
                'recommended_action': 'model_retrain' if decay_metrics.is_significant_decay else 'continue_monitoring'
            }
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return MonitoringResponse(
            success=True,
            message="Alpha decay analysis completed",
            data=results,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Alpha decay analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Alpha decay analysis failed: {str(e)}")


@router.post("/attribution", response_model=MonitoringResponse)
async def factor_attribution_analysis(request: AttributionRequest):
    """
    Perform multi-factor P&L attribution analysis.
    
    Decomposes portfolio returns into factor contributions and specific returns
    to understand sources of performance.
    """
    start_time = datetime.now()
    
    try:
        # Convert input data
        portfolio_returns = convert_timeseries_to_pandas(request.portfolio_returns)
        factor_returns = convert_factor_returns_to_dataframe(
            request.factor_returns, request.factor_names
        )
        
        # Initialize attribution engine
        attribution_engine = FactorAttributionEngine(
            factor_names=request.factor_names,
            attribution_method=request.attribution_method
        )
        
        # Perform attribution analysis
        if request.rolling_analysis:
            # Rolling attribution
            rolling_attribution = attribution_engine.rolling_attribution(
                returns=portfolio_returns,
                factor_returns=factor_returns,
                window=request.rolling_window
            )
            
            # Get latest attribution
            latest_attribution = attribution_engine.attribute_returns(
                portfolio_returns, factor_returns
            )
            
            attribution_data = {
                'latest_attribution': {
                    'total_pnl': latest_attribution.total_pnl,
                    'factor_attributions': latest_attribution.factor_attributions,
                    'specific_return': latest_attribution.specific_return,
                    'attribution_r_squared': latest_attribution.attribution_r_squared,
                    'factor_exposures': latest_attribution.factor_exposures
                },
                'rolling_attribution': rolling_attribution.to_dict('records'),
                'attribution_stability': {
                    'r_squared_mean': rolling_attribution['r_squared'].mean(),
                    'r_squared_std': rolling_attribution['r_squared'].std(),
                    'specific_return_volatility': rolling_attribution['specific_return'].std()
                }
            }
        else:
            # Single period attribution
            attribution_result = attribution_engine.attribute_returns(
                portfolio_returns, factor_returns
            )
            
            attribution_data = {
                'attribution_result': {
                    'total_pnl': attribution_result.total_pnl,
                    'factor_attributions': attribution_result.factor_attributions,
                    'specific_return': attribution_result.specific_return,
                    'attribution_r_squared': attribution_result.attribution_r_squared,
                    'explained_variance': attribution_result.explained_variance,
                    'residual_risk': attribution_result.residual_risk,
                    'factor_exposures': attribution_result.factor_exposures
                }
            }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return MonitoringResponse(
            success=True,
            message="Factor attribution analysis completed",
            data=attribution_data,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Attribution analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Attribution analysis failed: {str(e)}")


@router.post("/performance", response_model=MonitoringResponse)
async def performance_analysis(request: PerformanceAnalysisRequest):
    """
    Comprehensive performance analysis with risk metrics and alerts.
    
    Calculates performance metrics, generates alerts based on thresholds,
    and provides detailed risk analysis.
    """
    start_time = datetime.now()
    
    try:
        # Convert input data
        returns = convert_timeseries_to_pandas(request.returns)
        prices = convert_timeseries_to_pandas(request.prices) if request.prices else None
        
        # Convert trades data if provided
        trades_df = None
        if request.trades:
            trades_df = pd.DataFrame(request.trades)
        
        # Initialize performance monitor
        performance_monitor = PerformanceMonitor()
        
        # Calculate performance metrics
        performance_metrics = performance_monitor.calculate_performance_metrics(
            returns=returns,
            prices=prices,
            trades=trades_df
        )
        
        # Generate alerts
        alerts = performance_monitor.generate_alerts(performance_metrics)
        
        # Benchmark comparison if provided
        benchmark_comparison = None
        if request.benchmark_returns:
            benchmark_returns = convert_timeseries_to_pandas(request.benchmark_returns)
            benchmark_metrics = performance_monitor.calculate_performance_metrics(benchmark_returns)
            
            benchmark_comparison = {
                'excess_return': performance_metrics.annualized_return - benchmark_metrics.annualized_return,
                'tracking_error': (returns - benchmark_returns).std() * np.sqrt(252),
                'information_ratio': ((returns - benchmark_returns).mean() * 252) / ((returns - benchmark_returns).std() * np.sqrt(252)),
                'beta': np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns),
                'correlation': np.corrcoef(returns, benchmark_returns)[0, 1]
            }
        
        # Format results
        performance_data = {
            'performance_metrics': {
                'total_return': performance_metrics.total_return,
                'annualized_return': performance_metrics.annualized_return,
                'volatility': performance_metrics.volatility,
                'sharpe_ratio': performance_metrics.sharpe_ratio,
                'max_drawdown': performance_metrics.max_drawdown,
                'calmar_ratio': performance_metrics.calmar_ratio,
                'win_rate': performance_metrics.win_rate,
                'profit_factor': performance_metrics.profit_factor,
                'var_95': performance_metrics.var_95,
                'cvar_95': performance_metrics.cvar_95,
                'skewness': performance_metrics.skewness,
                'kurtosis': performance_metrics.kurtosis
            },
            'risk_analysis': {
                'downside_deviation': returns[returns < 0].std() * np.sqrt(252),
                'upside_deviation': returns[returns > 0].std() * np.sqrt(252),
                'sortino_ratio': performance_metrics.annualized_return / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0,
                'tail_ratio': abs(returns.quantile(0.95) / returns.quantile(0.05)) if returns.quantile(0.05) != 0 else 0
            },
            'alerts': alerts,
            'benchmark_comparison': benchmark_comparison,
            'alert_thresholds': performance_monitor.alert_thresholds
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return MonitoringResponse(
            success=True,
            message="Performance analysis completed",
            data=performance_data,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance analysis failed: {str(e)}")


@router.post("/health-assessment", response_model=MonitoringResponse)
async def model_health_assessment(request: HealthAssessmentRequest):
    """
    Comprehensive model health assessment and scoring.
    
    Evaluates model health across multiple dimensions including alpha decay,
    attribution quality, risk metrics, and data quality.
    """
    start_time = datetime.now()
    
    try:
        # Convert input data
        strategy_returns = convert_timeseries_to_pandas(request.strategy_returns)
        benchmark_returns = convert_timeseries_to_pandas(request.benchmark_returns)
        factor_returns = convert_factor_returns_to_dataframe(
            request.factor_returns, request.factor_names
        )
        
        # Initialize health analyzer
        health_analyzer = ModelHealthAnalyzer()
        
        # Assess model health
        assessment_date = request.assessment_date or datetime.now()
        health_score = health_analyzer.assess_model_health(
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            factor_returns=factor_returns,
            timestamp=assessment_date
        )
        
        # Format results
        health_data = {
            'health_score': {
                'overall_score': health_score.overall_score,
                'alpha_decay_score': health_score.alpha_decay_score,
                'attribution_quality_score': health_score.attribution_quality_score,
                'risk_score': health_score.risk_score,
                'performance_score': health_score.performance_score,
                'data_quality_score': health_score.data_quality_score
            },
            'health_status': {
                'status': 'healthy' if health_score.overall_score >= 75 else 'warning' if health_score.overall_score >= 60 else 'critical',
                'alerts': health_score.alerts,
                'recommendations': health_score.recommendations
            },
            'score_interpretation': {
                'excellent': 'Score >= 85',
                'good': 'Score 75-84',
                'warning': 'Score 60-74',
                'critical': 'Score < 60'
            }
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return MonitoringResponse(
            success=True,
            message="Model health assessment completed",
            data=health_data,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Health assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health assessment failed: {str(e)}")


@router.post("/configure-slo")
async def configure_slo(request: SLOConfigRequest):
    """Configure Service Level Objective for monitoring."""
    
    try:
        # Create SLO target
        slo_target = SLOTarget(
            metric_name=request.metric_name,
            target_value=request.target_value,
            tolerance=request.tolerance,
            operator=request.operator,
            frequency=request.frequency,
            description=request.description
        )
        
        # In practice, this would be stored in a database
        # For now, we return the configuration
        
        return MonitoringResponse(
            success=True,
            message=f"SLO configured for {request.metric_name}",
            data={
                'slo_configuration': {
                    'metric_name': request.metric_name,
                    'target_value': request.target_value,
                    'tolerance': request.tolerance,
                    'operator': request.operator,
                    'frequency': request.frequency,
                    'description': request.description
                }
            },
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"SLO configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"SLO configuration failed: {str(e)}")


@router.get("/monitoring-metrics")
async def get_monitoring_metrics():
    """Get available monitoring metrics and their descriptions."""
    
    return {
        'performance_metrics': {
            'sharpe_ratio': 'Risk-adjusted return measure',
            'max_drawdown': 'Maximum peak-to-trough decline',
            'volatility': 'Annualized return volatility',
            'win_rate': 'Percentage of profitable periods',
            'var_95': '95% Value at Risk',
            'calmar_ratio': 'Annual return / max drawdown',
            'sortino_ratio': 'Downside risk-adjusted return'
        },
        'alpha_decay_metrics': {
            'decay_rate': 'Rate of alpha degradation over time',
            'half_life': 'Time for alpha to decay by 50%',
            'decay_confidence': 'Statistical confidence in decay trend',
            'alpha_significance': 'Statistical significance of alpha estimates'
        },
        'attribution_metrics': {
            'factor_contributions': 'Individual factor P&L contributions',
            'specific_return': 'Alpha not explained by factors',
            'attribution_r_squared': 'Explanatory power of factor model',
            'factor_exposures': 'Portfolio loadings on each factor'
        },
        'health_score_components': {
            'alpha_decay_score': 'Alpha stability over time (0-100)',
            'attribution_quality_score': 'Factor model explanatory power (0-100)',
            'risk_score': 'Risk management effectiveness (0-100)',
            'performance_score': 'Risk-adjusted performance (0-100)',
            'data_quality_score': 'Data completeness and quality (0-100)'
        }
    }


@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring service."""
    return {
        'status': 'healthy',
        'service': 'monitoring-attribution',
        'timestamp': datetime.now(),
        'available_endpoints': [
            'comprehensive',
            'alpha-decay',
            'attribution',
            'performance',
            'health-assessment',
            'configure-slo',
            'monitoring-metrics'
        ]
    }


@router.get("/documentation")
async def get_api_documentation():
    """Get comprehensive API documentation for monitoring and attribution endpoints."""
    
    return {
        'overview': 'Monitoring & Attribution System for comprehensive strategy performance tracking',
        'purpose': 'Monitor strategy health, detect alpha decay, and attribute performance to factors',
        'components': {
            'alpha_decay_tracking': {
                'description': 'Detects degradation of strategy alpha over time',
                'metrics': ['Decay rate', 'Half-life', 'Statistical significance'],
                'applications': ['Model retraining alerts', 'Strategy lifecycle management']
            },
            'factor_attribution': {
                'description': 'Decomposes returns into factor contributions',
                'methods': ['Linear regression', 'Ridge regression'],
                'outputs': ['Factor exposures', 'Factor contributions', 'Specific returns']
            },
            'performance_monitoring': {
                'description': 'Real-time performance tracking with alerts',
                'metrics': ['Sharpe ratio', 'Drawdown', 'VaR', 'Win rate'],
                'features': ['Threshold alerts', 'Risk decomposition', 'Benchmark comparison']
            },
            'slo_monitoring': {
                'description': 'Service Level Objectives for operational monitoring',
                'capabilities': ['Custom thresholds', 'Violation tracking', 'Compliance reporting'],
                'use_cases': ['Production monitoring', 'SLA compliance', 'Performance guarantees']
            },
            'health_scoring': {
                'description': 'Comprehensive model health assessment',
                'dimensions': ['Alpha decay', 'Attribution quality', 'Risk', 'Performance', 'Data quality'],
                'scoring': '0-100 scale with interpretive guidance'
            }
        },
        'endpoints': {
            '/comprehensive': {
                'method': 'POST',
                'description': 'Run complete monitoring analysis',
                'use_cases': ['Daily monitoring reports', 'Strategy health checks', 'Performance reviews']
            },
            '/alpha-decay': {
                'method': 'POST',
                'description': 'Analyze alpha decay patterns',
                'use_cases': ['Model retraining decisions', 'Strategy lifecycle management']
            },
            '/attribution': {
                'method': 'POST',
                'description': 'Multi-factor P&L attribution',
                'use_cases': ['Performance explanation', 'Risk factor analysis', 'Portfolio optimization']
            },
            '/performance': {
                'method': 'POST',
                'description': 'Comprehensive performance analysis',
                'use_cases': ['Risk monitoring', 'Performance reporting', 'Alert generation']
            },
            '/health-assessment': {
                'method': 'POST',
                'description': 'Model health scoring and diagnostics',
                'use_cases': ['Model validation', 'Production readiness', 'Quality assurance']
            }
        },
        'best_practices': [
            'Monitor alpha decay regularly to detect model degradation',
            'Use factor attribution to understand return sources',
            'Set appropriate SLO thresholds based on strategy characteristics',
            'Combine multiple health score dimensions for comprehensive assessment',
            'Automate alerts for critical performance thresholds',
            'Review monitoring results in context of market conditions'
        ],
        'typical_workflow': [
            '1. Configure SLOs and monitoring thresholds',
            '2. Run daily comprehensive monitoring',
            '3. Analyze alpha decay trends weekly',
            '4. Perform factor attribution monthly',
            '5. Assess model health quarterly',
            '6. Take corrective actions based on alerts and scores'
        ]
    }