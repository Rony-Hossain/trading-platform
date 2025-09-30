"""
API endpoints for Advanced Portfolio Construction.

Provides REST API access to sophisticated portfolio optimization:
- Equal Risk Contribution (ERC) portfolios
- Risk Parity allocation with constraints
- Volatility targeting and dynamic scaling
- Hierarchical Risk Parity (HRP)
- Portfolio backtesting and performance analysis
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from services.portfolio_construction import (
    AdvancedPortfolioConstructor, CovarianceEstimator, ERCOptimizer,
    VolatilityTargeting, HierarchicalRiskParity, PortfolioWeights,
    RiskBudget, PortfolioConstraints
)

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class AssetReturnData(BaseModel):
    """Asset return data point."""
    timestamp: datetime
    symbol: str
    return_value: float


class PortfolioConstructionRequest(BaseModel):
    """Request model for portfolio construction."""
    returns_data: List[AssetReturnData] = Field(..., description="Historical asset returns")
    method: str = Field("erc", description="Portfolio method: 'erc', 'hrp', 'risk_parity'")
    target_volatility: Optional[float] = Field(None, ge=0.01, le=1.0, description="Target annual volatility")
    
    # Risk budget specification
    risk_budget: Optional[Dict[str, float]] = Field(None, description="Asset risk budget allocation")
    
    # Constraints
    min_weights: Optional[Dict[str, float]] = Field(None, description="Minimum asset weights")
    max_weights: Optional[Dict[str, float]] = Field(None, description="Maximum asset weights")
    max_leverage: Optional[float] = Field(1.0, ge=0.1, le=5.0, description="Maximum portfolio leverage")
    
    # Covariance estimation
    cov_method: str = Field("ledoit_wolf", description="Covariance estimation method")
    lookback_days: int = Field(252, ge=60, le=1000, description="Lookback period for estimation")
    
    @validator('method')
    def validate_method(cls, v):
        allowed_methods = ['erc', 'hrp', 'risk_parity']
        if v not in allowed_methods:
            raise ValueError(f"Method must be one of {allowed_methods}")
        return v


class BacktestRequest(BaseModel):
    """Request model for portfolio backtesting."""
    returns_data: List[AssetReturnData] = Field(..., description="Historical asset returns")
    method: str = Field("erc", description="Portfolio construction method")
    rebalance_frequency: str = Field("monthly", description="Rebalancing frequency")
    target_volatility: Optional[float] = Field(None, description="Target volatility for vol targeting")
    start_date: Optional[datetime] = Field(None, description="Backtest start date")
    end_date: Optional[datetime] = Field(None, description="Backtest end date")
    
    # Advanced parameters
    transaction_costs: float = Field(0.001, ge=0.0, le=0.01, description="Transaction costs (bps)")
    risk_budget: Optional[Dict[str, float]] = Field(None, description="Asset risk budget")
    
    @validator('rebalance_frequency')
    def validate_rebalance_frequency(cls, v):
        allowed_freq = ['daily', 'weekly', 'monthly', 'quarterly']
        if v not in allowed_freq:
            raise ValueError(f"Rebalance frequency must be one of {allowed_freq}")
        return v


class VolatilityTargetingRequest(BaseModel):
    """Request model for volatility targeting analysis."""
    portfolio_returns: List[Dict[str, Any]] = Field(..., description="Historical portfolio returns")
    target_volatility: float = Field(..., ge=0.01, le=1.0, description="Target annual volatility")
    lookback_days: int = Field(60, ge=20, le=252, description="Lookback for vol estimation")
    max_leverage: float = Field(2.0, ge=1.0, le=5.0, description="Maximum allowed leverage")


class RiskBudgetRequest(BaseModel):
    """Request model for risk budget optimization."""
    returns_data: List[AssetReturnData] = Field(..., description="Historical asset returns")
    target_risk_budget: Dict[str, float] = Field(..., description="Target risk contributions by asset")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Portfolio constraints")
    
    @validator('target_risk_budget')
    def validate_risk_budget(cls, v):
        if abs(sum(v.values()) - 1.0) > 1e-6:
            raise ValueError("Target risk budget must sum to 1.0")
        return v


class PortfolioResponse(BaseModel):
    """Response model for portfolio operations."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime
    processing_time_ms: Optional[float] = None


class PortfolioWeightsDict(BaseModel):
    """Dictionary representation of portfolio weights."""
    weights: Dict[str, float]
    timestamp: datetime
    portfolio_type: str
    target_volatility: Optional[float]
    actual_volatility: Optional[float]
    risk_contributions: Optional[Dict[str, float]]
    leverage: Optional[float]
    expected_return: Optional[float]
    sharpe_ratio: Optional[float]


# Utility functions
def convert_returns_data_to_dataframe(returns_data: List[AssetReturnData]) -> pd.DataFrame:
    """Convert return data points to pandas DataFrame."""
    
    # Group by symbol and timestamp
    data_dict = {}
    for point in returns_data:
        if point.symbol not in data_dict:
            data_dict[point.symbol] = {}
        data_dict[point.symbol][point.timestamp] = point.return_value
    
    # Create DataFrame
    df = pd.DataFrame(data_dict)
    df.sort_index(inplace=True)
    
    # Handle missing values
    df = df.fillna(method='ffill').dropna()
    
    return df


def convert_portfolio_weights_to_dict(portfolio: PortfolioWeights) -> Dict[str, Any]:
    """Convert PortfolioWeights object to dictionary."""
    return {
        'weights': portfolio.weights,
        'timestamp': portfolio.timestamp,
        'portfolio_type': portfolio.portfolio_type,
        'target_volatility': portfolio.target_volatility,
        'actual_volatility': portfolio.actual_volatility,
        'risk_contributions': portfolio.risk_contributions,
        'leverage': portfolio.leverage,
        'expected_return': portfolio.expected_return,
        'sharpe_ratio': portfolio.sharpe_ratio,
        'max_drawdown': portfolio.max_drawdown,
        'turnover': portfolio.turnover
    }


# API Endpoints

@router.post("/construct", response_model=PortfolioResponse)
async def construct_portfolio(request: PortfolioConstructionRequest):
    """
    Construct optimal portfolio using specified method.
    
    Supports Equal Risk Contribution (ERC), Hierarchical Risk Parity (HRP),
    and traditional Risk Parity allocation methods.
    """
    start_time = datetime.now()
    
    try:
        # Convert input data
        returns_df = convert_returns_data_to_dataframe(request.returns_data)
        
        # Validate data
        if len(returns_df) < 60:
            raise HTTPException(
                status_code=400,
                detail="Minimum 60 observations required for portfolio construction"
            )
        
        if len(returns_df.columns) < 2:
            raise HTTPException(
                status_code=400,
                detail="Minimum 2 assets required for portfolio construction"
            )
        
        # Initialize components
        cov_estimator = CovarianceEstimator(
            method=request.cov_method,
            lookback_days=request.lookback_days
        )
        
        erc_optimizer = ERCOptimizer()
        vol_targeting = VolatilityTargeting(target_volatility=request.target_volatility or 0.15)
        hrp = HierarchicalRiskParity()
        
        # Initialize portfolio constructor
        constructor = AdvancedPortfolioConstructor(
            cov_estimator=cov_estimator,
            erc_optimizer=erc_optimizer,
            vol_targeting=vol_targeting,
            hrp=hrp
        )
        
        # Create constraints if specified
        constraints = None
        if request.min_weights or request.max_weights or request.max_leverage != 1.0:
            constraints = PortfolioConstraints(
                min_weights=request.min_weights,
                max_weights=request.max_weights,
                max_leverage=request.max_leverage
            )
        
        # Create risk budget if specified
        risk_budget = None
        if request.risk_budget:
            risk_budget = RiskBudget(asset_budgets=request.risk_budget)
        
        # Construct portfolio
        portfolio = constructor.construct_portfolio(
            returns=returns_df,
            method=request.method,
            risk_budget=risk_budget,
            constraints=constraints,
            target_volatility=request.target_volatility
        )
        
        # Convert to response format
        portfolio_dict = convert_portfolio_weights_to_dict(portfolio)
        
        # Add additional analytics
        portfolio_stats = constructor.get_portfolio_statistics()
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PortfolioResponse(
            success=True,
            message=f"Portfolio constructed using {request.method.upper()} method",
            data={
                'portfolio': portfolio_dict,
                'statistics': portfolio_stats,
                'method': request.method,
                'n_assets': len(returns_df.columns),
                'data_points': len(returns_df)
            },
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Portfolio construction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio construction failed: {str(e)}")


@router.post("/backtest", response_model=PortfolioResponse)
async def backtest_portfolio(request: BacktestRequest):
    """
    Backtest portfolio construction methodology.
    
    Simulates portfolio construction and rebalancing over historical period
    with realistic transaction costs and constraints.
    """
    start_time = datetime.now()
    
    try:
        # Convert input data
        returns_df = convert_returns_data_to_dataframe(request.returns_data)
        
        # Validate data
        if len(returns_df) < 252:
            raise HTTPException(
                status_code=400,
                detail="Minimum 1 year of data required for backtesting"
            )
        
        # Initialize portfolio constructor
        constructor = AdvancedPortfolioConstructor()
        
        # Run backtest
        backtest_results = constructor.backtest_portfolio(
            returns=returns_df,
            method=request.method,
            rebalance_frequency=request.rebalance_frequency,
            target_volatility=request.target_volatility,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Calculate performance metrics
        total_return = backtest_results['cumulative_return'].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(backtest_results)) - 1
        volatility = backtest_results['return'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = backtest_results['drawdown'].min()
        
        # Calculate other metrics
        win_rate = (backtest_results['return'] > 0).mean()
        avg_return = backtest_results['return'].mean() * 252
        
        # Rebalancing analysis
        rebalance_count = len(constructor.portfolio_history)
        avg_turnover = constructor.get_portfolio_statistics().get('avg_turnover', 0)
        
        performance_metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'rebalance_count': rebalance_count,
            'avg_turnover': avg_turnover
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PortfolioResponse(
            success=True,
            message=f"Backtest completed for {request.method.upper()} portfolio",
            data={
                'performance_metrics': performance_metrics,
                'backtest_period': {
                    'start': backtest_results.index[0],
                    'end': backtest_results.index[-1],
                    'duration_days': len(backtest_results)
                },
                'method': request.method,
                'rebalance_frequency': request.rebalance_frequency,
                'transaction_costs': request.transaction_costs,
                'target_volatility': request.target_volatility
            },
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Portfolio backtesting failed: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio backtesting failed: {str(e)}")


@router.post("/volatility-targeting", response_model=PortfolioResponse)
async def analyze_volatility_targeting(request: VolatilityTargetingRequest):
    """
    Analyze volatility targeting strategy performance.
    
    Calculates dynamic scaling factors to maintain target volatility
    and analyzes the impact on portfolio performance.
    """
    start_time = datetime.now()
    
    try:
        # Convert portfolio returns data
        returns_data = []
        for point in request.portfolio_returns:
            returns_data.append({
                'timestamp': pd.Timestamp(point['timestamp']),
                'return': point['return']
            })
        
        portfolio_returns = pd.DataFrame(returns_data)
        portfolio_returns.set_index('timestamp', inplace=True)
        portfolio_returns.sort_index(inplace=True)
        
        # Initialize volatility targeting
        vol_targeting = VolatilityTargeting(
            target_volatility=request.target_volatility,
            lookback_days=request.lookback_days,
            max_leverage=request.max_leverage
        )
        
        # Calculate scaling factors over time
        scaling_factors = []
        realized_vols = []
        
        for date in portfolio_returns.index[request.lookback_days:]:
            # Calculate scaling factor
            scale = vol_targeting.calculate_vol_scaling(
                portfolio_returns['return'], date
            )
            
            # Calculate realized volatility
            window_returns = portfolio_returns['return'].loc[
                portfolio_returns.index <= date
            ].tail(request.lookback_days)
            realized_vol = window_returns.std() * np.sqrt(252)
            
            scaling_factors.append({
                'date': date,
                'scaling_factor': scale,
                'realized_volatility': realized_vol,
                'target_volatility': request.target_volatility
            })
            realized_vols.append(realized_vol)
        
        # Calculate performance metrics
        avg_scaling = np.mean([sf['scaling_factor'] for sf in scaling_factors])
        vol_stability = np.std(realized_vols)
        max_leverage = max([sf['scaling_factor'] for sf in scaling_factors])
        min_leverage = min([sf['scaling_factor'] for sf in scaling_factors])
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PortfolioResponse(
            success=True,
            message="Volatility targeting analysis completed",
            data={
                'scaling_analysis': {
                    'avg_scaling_factor': avg_scaling,
                    'max_leverage': max_leverage,
                    'min_leverage': min_leverage,
                    'volatility_stability': vol_stability,
                    'target_volatility': request.target_volatility
                },
                'scaling_factors': scaling_factors[-50:],  # Last 50 periods
                'performance_impact': {
                    'avg_realized_vol': np.mean(realized_vols),
                    'vol_target_hit_rate': np.mean([
                        abs(rv - request.target_volatility) < 0.02 
                        for rv in realized_vols
                    ])
                }
            },
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Volatility targeting analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Volatility targeting analysis failed: {str(e)}")


@router.post("/risk-budget", response_model=PortfolioResponse)
async def optimize_risk_budget(request: RiskBudgetRequest):
    """
    Optimize portfolio for specific risk budget allocation.
    
    Creates portfolio where each asset contributes specified amount
    to total portfolio risk (Equal Risk Contribution variant).
    """
    start_time = datetime.now()
    
    try:
        # Convert input data
        returns_df = convert_returns_data_to_dataframe(request.returns_data)
        
        # Validate risk budget
        asset_names = list(returns_df.columns)
        risk_budget_assets = set(request.target_risk_budget.keys())
        
        if not risk_budget_assets.issubset(set(asset_names)):
            missing_assets = risk_budget_assets - set(asset_names)
            raise HTTPException(
                status_code=400,
                detail=f"Risk budget contains unknown assets: {missing_assets}"
            )
        
        # Create risk budget object
        risk_budget = RiskBudget(asset_budgets=request.target_risk_budget)
        
        # Initialize components
        cov_estimator = CovarianceEstimator()
        erc_optimizer = ERCOptimizer()
        
        # Estimate covariance matrix
        cov_matrix, _ = cov_estimator.estimate(returns_df)
        
        # Create risk budget array aligned with asset order
        risk_budget_array = np.array([
            request.target_risk_budget.get(asset, 0.0) 
            for asset in asset_names
        ])
        
        # Optimize portfolio
        weights, opt_info = erc_optimizer.optimize(
            cov_matrix, risk_budget_array, None
        )
        
        # Calculate final risk contributions
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        marginal_risk = np.dot(cov_matrix, weights)
        actual_risk_contribs = weights * marginal_risk / portfolio_var
        
        # Create results
        portfolio_weights = dict(zip(asset_names, weights))
        actual_risk_budget = dict(zip(asset_names, actual_risk_contribs))
        
        # Calculate tracking error vs target
        risk_tracking_error = np.sqrt(np.mean(
            (actual_risk_contribs - risk_budget_array) ** 2
        ))
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PortfolioResponse(
            success=True,
            message="Risk budget optimization completed",
            data={
                'portfolio_weights': portfolio_weights,
                'target_risk_budget': request.target_risk_budget,
                'actual_risk_budget': actual_risk_budget,
                'optimization_info': {
                    'success': opt_info['success'],
                    'iterations': opt_info['iterations'],
                    'risk_tracking_error': risk_tracking_error,
                    'concentration_ratio': opt_info['concentration_ratio'],
                    'effective_n_assets': opt_info['effective_n_assets']
                }
            },
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Risk budget optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Risk budget optimization failed: {str(e)}")


@router.get("/methods")
async def get_portfolio_methods():
    """Get available portfolio construction methods and their descriptions."""
    
    return {
        'methods': {
            'erc': {
                'name': 'Equal Risk Contribution',
                'description': 'Each asset contributes equally to portfolio risk',
                'benefits': ['Risk diversification', 'Reduced concentration', 'Stable allocations'],
                'parameters': ['risk_budget', 'constraints']
            },
            'hrp': {
                'name': 'Hierarchical Risk Parity',
                'description': 'Risk parity using hierarchical clustering',
                'benefits': ['Handles large universes', 'Reduces estimation error', 'Intuitive clustering'],
                'parameters': ['distance_metric', 'linkage_method']
            },
            'risk_parity': {
                'name': 'Traditional Risk Parity',
                'description': 'Equal risk contribution across all assets',
                'benefits': ['Simple implementation', 'Equal risk allocation', 'Well-studied'],
                'parameters': ['constraints']
            }
        },
        'features': {
            'volatility_targeting': 'Dynamic scaling to maintain target volatility',
            'risk_budgeting': 'Custom risk allocation across assets',
            'constraints': 'Weight bounds and leverage limits',
            'backtesting': 'Historical performance simulation',
            'transaction_costs': 'Realistic cost modeling'
        },
        'covariance_methods': {
            'ledoit_wolf': 'Shrinkage estimator (recommended)',
            'empirical': 'Sample covariance matrix',
            'exponential': 'Exponentially weighted covariance'
        }
    }


@router.get("/health")
async def health_check():
    """Health check endpoint for portfolio construction service."""
    return {
        'status': 'healthy',
        'service': 'portfolio-construction',
        'timestamp': datetime.now(),
        'available_endpoints': [
            'construct',
            'backtest',
            'volatility-targeting',
            'risk-budget',
            'methods'
        ]
    }


@router.get("/documentation")
async def get_api_documentation():
    """Get comprehensive API documentation for portfolio construction endpoints."""
    
    return {
        'overview': 'Advanced Portfolio Construction API with ERC, HRP, and Risk Parity methods',
        'methodology': {
            'equal_risk_contribution': {
                'description': 'Allocates capital so each asset contributes equally to portfolio risk',
                'advantages': ['Better diversification', 'Reduced tail risk', 'More stable allocations'],
                'use_cases': ['Multi-asset portfolios', 'Risk-focused allocation', 'Defensive strategies']
            },
            'hierarchical_risk_parity': {
                'description': 'Uses hierarchical clustering to build risk parity portfolios',
                'advantages': ['Handles large universes', 'Reduces estimation error', 'Intuitive structure'],
                'use_cases': ['Large asset universes', 'Factor-based investing', 'Alternative data']
            },
            'volatility_targeting': {
                'description': 'Dynamically scales portfolio to maintain constant volatility',
                'advantages': ['Risk control', 'Leverage efficiency', 'Performance stability'],
                'use_cases': ['Managed volatility', 'Risk budgeting', 'Institutional mandates']
            }
        },
        'endpoints': {
            '/construct': {
                'method': 'POST',
                'description': 'Construct optimal portfolio using specified method',
                'input': 'Asset returns, method, constraints, risk budget',
                'output': 'Portfolio weights with risk analysis'
            },
            '/backtest': {
                'method': 'POST',
                'description': 'Backtest portfolio construction methodology',
                'input': 'Historical returns, rebalancing frequency, constraints',
                'output': 'Performance metrics and rebalancing analysis'
            },
            '/volatility-targeting': {
                'method': 'POST',
                'description': 'Analyze volatility targeting strategy',
                'input': 'Portfolio returns, target volatility, parameters',
                'output': 'Scaling factors and performance impact'
            },
            '/risk-budget': {
                'method': 'POST',
                'description': 'Optimize for specific risk budget allocation',
                'input': 'Asset returns, target risk contributions',
                'output': 'Optimized weights with risk tracking'
            }
        },
        'best_practices': [
            'Use sufficient historical data (minimum 1 year)',
            'Consider transaction costs in backtesting',
            'Validate results with out-of-sample testing',
            'Monitor risk contributions over time',
            'Adjust rebalancing frequency based on transaction costs'
        ],
        'common_applications': [
            'Multi-asset portfolio construction',
            'Risk budgeting and allocation',
            'Alternative risk premia strategies',
            'Institutional portfolio management',
            'Factor-based investing'
        ]
    }