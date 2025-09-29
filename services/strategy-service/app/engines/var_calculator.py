"""
Value at Risk (VaR) and Conditional VaR (CVaR) Calculator

Implements multiple VaR methodologies for sophisticated portfolio risk management:
- Historical VaR (non-parametric)
- Parametric VaR (normal distribution assumption)
- Monte Carlo VaR simulation
- Conditional VaR (Expected Shortfall)
- Portfolio-level risk aggregation with correlation
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.optimize import minimize
import asyncio

logger = logging.getLogger(__name__)

class VaRMethod(Enum):
    """VaR calculation methods"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"

class RiskHorizon(Enum):
    """Risk horizon periods"""
    DAILY = 1
    WEEKLY = 5
    MONTHLY = 22
    QUARTERLY = 66
    YEARLY = 252

@dataclass
class VaRConfiguration:
    """Configuration for VaR calculations"""
    confidence_level: float = 0.95  # 95% confidence level
    risk_horizon: RiskHorizon = RiskHorizon.DAILY
    lookback_period: int = 252  # Trading days for historical data
    monte_carlo_simulations: int = 10000
    cornish_fisher_adjustment: bool = True  # Account for skewness/kurtosis
    bootstrap_samples: int = 1000
    
    def __post_init__(self):
        if not 0.5 <= self.confidence_level <= 0.999:
            raise ValueError("confidence_level must be between 0.5 and 0.999")

@dataclass
class PositionInfo:
    """Information about a trading position"""
    symbol: str
    quantity: float
    current_price: float
    market_value: float
    weight: float  # Portfolio weight
    returns_data: Optional[pd.Series] = None
    
    def __post_init__(self):
        if self.market_value is None:
            self.market_value = abs(self.quantity * self.current_price)

@dataclass
class VaRResult:
    """Results from VaR calculation"""
    symbol: str
    method: VaRMethod
    confidence_level: float
    risk_horizon: int
    var_absolute: float  # Absolute VaR in currency
    var_percentage: float  # VaR as percentage of position value
    cvar_absolute: float  # Conditional VaR (Expected Shortfall)
    cvar_percentage: float
    position_value: float
    calculation_timestamp: datetime
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'method': self.method.value,
            'confidence_level': self.confidence_level,
            'risk_horizon': self.risk_horizon,
            'var_absolute': self.var_absolute,
            'var_percentage': self.var_percentage,
            'cvar_absolute': self.cvar_absolute,
            'cvar_percentage': self.cvar_percentage,
            'position_value': self.position_value,
            'calculation_timestamp': self.calculation_timestamp.isoformat(),
            'additional_metrics': self.additional_metrics
        }

@dataclass
class PortfolioVaRResult:
    """Portfolio-level VaR results"""
    portfolio_value: float
    var_absolute: float
    var_percentage: float
    cvar_absolute: float
    cvar_percentage: float
    diversification_benefit: float  # Reduction due to correlation
    individual_var_sum: float  # Sum of individual VaRs
    confidence_level: float
    risk_horizon: int
    method: VaRMethod
    calculation_timestamp: datetime
    position_contributions: Dict[str, float]  # VaR contribution by position
    correlation_matrix: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'portfolio_value': self.portfolio_value,
            'var_absolute': self.var_absolute,
            'var_percentage': self.var_percentage,
            'cvar_absolute': self.cvar_absolute,
            'cvar_percentage': self.cvar_percentage,
            'diversification_benefit': self.diversification_benefit,
            'individual_var_sum': self.individual_var_sum,
            'confidence_level': self.confidence_level,
            'risk_horizon': self.risk_horizon,
            'method': self.method.value,
            'calculation_timestamp': self.calculation_timestamp.isoformat(),
            'position_contributions': self.position_contributions,
            'correlation_matrix': self.correlation_matrix.to_dict() if self.correlation_matrix is not None else None
        }

class VaRCalculator:
    """Comprehensive VaR calculator with multiple methodologies"""
    
    def __init__(self, config: Optional[VaRConfiguration] = None):
        self.config = config or VaRConfiguration()
        self.returns_cache = {}  # Cache for returns data
        
    def calculate_position_var(self, position: PositionInfo, 
                              method: VaRMethod = VaRMethod.HISTORICAL) -> VaRResult:
        """
        Calculate VaR for a single position using specified method.
        
        Args:
            position: Position information
            method: VaR calculation method
            
        Returns:
            VaRResult with comprehensive risk metrics
        """
        
        if position.returns_data is None or position.returns_data.empty:
            logger.warning(f"No returns data available for {position.symbol}")
            return self._create_default_var_result(position, method)
        
        returns = position.returns_data.dropna()
        
        if len(returns) < 30:  # Minimum data requirement
            logger.warning(f"Insufficient data for {position.symbol}: {len(returns)} observations")
            return self._create_default_var_result(position, method)
        
        # Calculate VaR based on method
        if method == VaRMethod.HISTORICAL:
            var_pct, cvar_pct = self._calculate_historical_var(returns)
        elif method == VaRMethod.PARAMETRIC:
            var_pct, cvar_pct = self._calculate_parametric_var(returns)
        elif method == VaRMethod.MONTE_CARLO:
            var_pct, cvar_pct = self._calculate_monte_carlo_var(returns)
        elif method == VaRMethod.CORNISH_FISHER:
            var_pct, cvar_pct = self._calculate_cornish_fisher_var(returns)
        else:
            raise ValueError(f"Unsupported VaR method: {method}")
        
        # Scale to risk horizon
        horizon_multiplier = np.sqrt(self.config.risk_horizon.value)
        var_pct *= horizon_multiplier
        cvar_pct *= horizon_multiplier
        
        # Convert to absolute values
        var_absolute = abs(var_pct * position.market_value)
        cvar_absolute = abs(cvar_pct * position.market_value)
        
        # Additional metrics
        additional_metrics = {
            'volatility_annualized': returns.std() * np.sqrt(252),
            'skewness': float(stats.skew(returns)),
            'kurtosis': float(stats.kurtosis(returns)),
            'worst_loss_observed': float(returns.min()),
            'best_gain_observed': float(returns.max()),
            'data_points': len(returns)
        }
        
        return VaRResult(
            symbol=position.symbol,
            method=method,
            confidence_level=self.config.confidence_level,
            risk_horizon=self.config.risk_horizon.value,
            var_absolute=var_absolute,
            var_percentage=abs(var_pct) * 100,
            cvar_absolute=cvar_absolute,
            cvar_percentage=abs(cvar_pct) * 100,
            position_value=position.market_value,
            calculation_timestamp=datetime.now(),
            additional_metrics=additional_metrics
        )
    
    def _calculate_historical_var(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate Historical VaR (non-parametric)"""
        alpha = 1 - self.config.confidence_level
        
        # VaR is the alpha-quantile of the return distribution
        var_percentile = returns.quantile(alpha)
        
        # CVaR is the expected value of returns below VaR
        tail_returns = returns[returns <= var_percentile]
        cvar_value = tail_returns.mean() if len(tail_returns) > 0 else var_percentile
        
        return var_percentile, cvar_value
    
    def _calculate_parametric_var(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate Parametric VaR (assumes normal distribution)"""
        alpha = 1 - self.config.confidence_level
        
        mean = returns.mean()
        std = returns.std()
        
        # Normal distribution quantile
        z_score = stats.norm.ppf(alpha)
        var_value = mean + z_score * std
        
        # CVaR for normal distribution
        # E[X | X <= VaR] = μ + σ * φ(z) / Φ(z)
        pdf_at_z = stats.norm.pdf(z_score)
        cdf_at_z = stats.norm.cdf(z_score)
        cvar_value = mean + std * (pdf_at_z / cdf_at_z)
        
        return var_value, cvar_value
    
    def _calculate_monte_carlo_var(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate Monte Carlo VaR"""
        alpha = 1 - self.config.confidence_level
        
        # Fit distribution to historical returns
        mean = returns.mean()
        std = returns.std()
        
        # Generate random scenarios
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(
            mean, std, self.config.monte_carlo_simulations
        )
        
        # Calculate VaR and CVaR from simulation
        var_value = np.percentile(simulated_returns, alpha * 100)
        tail_returns = simulated_returns[simulated_returns <= var_value]
        cvar_value = np.mean(tail_returns) if len(tail_returns) > 0 else var_value
        
        return var_value, cvar_value
    
    def _calculate_cornish_fisher_var(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate Cornish-Fisher VaR (adjusts for skewness and kurtosis)"""
        alpha = 1 - self.config.confidence_level
        
        mean = returns.mean()
        std = returns.std()
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)  # Excess kurtosis
        
        # Standard normal quantile
        z = stats.norm.ppf(alpha)
        
        # Cornish-Fisher adjustment
        adjusted_z = (z + 
                     (z**2 - 1) * skewness / 6 +
                     (z**3 - 3*z) * kurtosis / 24 -
                     (2*z**3 - 5*z) * skewness**2 / 36)
        
        var_value = mean + adjusted_z * std
        
        # For CVaR, use adjusted distribution (approximation)
        # This is a simplified approach; more sophisticated methods exist
        tail_threshold = var_value
        tail_returns = returns[returns <= tail_threshold]
        cvar_value = tail_returns.mean() if len(tail_returns) > 0 else var_value
        
        return var_value, cvar_value
    
    def calculate_portfolio_var(self, positions: List[PositionInfo], 
                               method: VaRMethod = VaRMethod.HISTORICAL,
                               consider_correlation: bool = True) -> PortfolioVaRResult:
        """
        Calculate portfolio-level VaR considering correlations.
        
        Args:
            positions: List of portfolio positions
            method: VaR calculation method
            consider_correlation: Whether to account for correlation in aggregation
            
        Returns:
            PortfolioVaRResult with portfolio-level risk metrics
        """
        
        if not positions:
            raise ValueError("No positions provided for portfolio VaR calculation")
        
        # Calculate individual position VaRs
        individual_vars = {}
        individual_cvars = {}
        position_values = {}
        
        for position in positions:
            var_result = self.calculate_position_var(position, method)
            individual_vars[position.symbol] = var_result.var_absolute
            individual_cvars[position.symbol] = var_result.cvar_absolute
            position_values[position.symbol] = position.market_value
        
        portfolio_value = sum(position_values.values())
        individual_var_sum = sum(individual_vars.values())
        
        # Portfolio VaR calculation
        if consider_correlation and len(positions) > 1:
            portfolio_var, portfolio_cvar, correlation_matrix = self._calculate_correlated_portfolio_var(
                positions, method
            )
        else:
            # Simple summation (worst case - perfect correlation)
            portfolio_var = individual_var_sum
            portfolio_cvar = sum(individual_cvars.values())
            correlation_matrix = None
        
        # Calculate diversification benefit
        diversification_benefit = individual_var_sum - portfolio_var
        
        # Calculate marginal VaR contributions
        position_contributions = {}
        for symbol in individual_vars:
            if portfolio_var > 0:
                # Approximate marginal contribution
                contribution = individual_vars[symbol] * (portfolio_var / individual_var_sum)
                position_contributions[symbol] = contribution
            else:
                position_contributions[symbol] = 0.0
        
        return PortfolioVaRResult(
            portfolio_value=portfolio_value,
            var_absolute=portfolio_var,
            var_percentage=(portfolio_var / portfolio_value * 100) if portfolio_value > 0 else 0,
            cvar_absolute=portfolio_cvar,
            cvar_percentage=(portfolio_cvar / portfolio_value * 100) if portfolio_value > 0 else 0,
            diversification_benefit=diversification_benefit,
            individual_var_sum=individual_var_sum,
            confidence_level=self.config.confidence_level,
            risk_horizon=self.config.risk_horizon.value,
            method=method,
            calculation_timestamp=datetime.now(),
            position_contributions=position_contributions,
            correlation_matrix=correlation_matrix
        )
    
    def _calculate_correlated_portfolio_var(self, positions: List[PositionInfo], 
                                          method: VaRMethod) -> Tuple[float, float, pd.DataFrame]:
        """Calculate portfolio VaR accounting for correlations"""
        
        # Collect returns data
        returns_dict = {}
        weights = {}
        
        for position in positions:
            if position.returns_data is not None and not position.returns_data.empty:
                returns_dict[position.symbol] = position.returns_data
                weights[position.symbol] = position.weight
        
        if len(returns_dict) < 2:
            # Fallback to simple summation
            individual_vars = [self.calculate_position_var(pos, method).var_absolute for pos in positions]
            individual_cvars = [self.calculate_position_var(pos, method).cvar_absolute for pos in positions]
            return sum(individual_vars), sum(individual_cvars), None
        
        # Align returns data
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            logger.warning("No aligned returns data available for correlation calculation")
            individual_vars = [self.calculate_position_var(pos, method).var_absolute for pos in positions]
            individual_cvars = [self.calculate_position_var(pos, method).cvar_absolute for pos in positions]
            return sum(individual_vars), sum(individual_cvars), None
        
        # Calculate portfolio returns
        weight_vector = np.array([weights.get(symbol, 0) for symbol in returns_df.columns])
        portfolio_returns = (returns_df * weight_vector).sum(axis=1)
        
        # Calculate portfolio VaR
        if method == VaRMethod.HISTORICAL:
            var_pct, cvar_pct = self._calculate_historical_var(portfolio_returns)
        elif method == VaRMethod.PARAMETRIC:
            var_pct, cvar_pct = self._calculate_parametric_var(portfolio_returns)
        elif method == VaRMethod.MONTE_CARLO:
            var_pct, cvar_pct = self._calculate_monte_carlo_var(portfolio_returns)
        elif method == VaRMethod.CORNISH_FISHER:
            var_pct, cvar_pct = self._calculate_cornish_fisher_var(portfolio_returns)
        else:
            raise ValueError(f"Unsupported VaR method: {method}")
        
        # Scale to risk horizon
        horizon_multiplier = np.sqrt(self.config.risk_horizon.value)
        var_pct *= horizon_multiplier
        cvar_pct *= horizon_multiplier
        
        # Convert to absolute values
        portfolio_value = sum(pos.market_value for pos in positions)
        portfolio_var = abs(var_pct * portfolio_value)
        portfolio_cvar = abs(cvar_pct * portfolio_value)
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        return portfolio_var, portfolio_cvar, correlation_matrix
    
    def _create_default_var_result(self, position: PositionInfo, method: VaRMethod) -> VaRResult:
        """Create default VaR result when calculation fails"""
        
        # Conservative fallback - assume 2% daily volatility
        default_daily_vol = 0.02
        alpha = 1 - self.config.confidence_level
        z_score = stats.norm.ppf(alpha)
        
        var_pct = z_score * default_daily_vol * np.sqrt(self.config.risk_horizon.value)
        cvar_pct = var_pct * 1.3  # Approximate CVaR multiplier
        
        var_absolute = abs(var_pct * position.market_value)
        cvar_absolute = abs(cvar_pct * position.market_value)
        
        return VaRResult(
            symbol=position.symbol,
            method=method,
            confidence_level=self.config.confidence_level,
            risk_horizon=self.config.risk_horizon.value,
            var_absolute=var_absolute,
            var_percentage=abs(var_pct) * 100,
            cvar_absolute=cvar_absolute,
            cvar_percentage=abs(cvar_pct) * 100,
            position_value=position.market_value,
            calculation_timestamp=datetime.now(),
            additional_metrics={'note': 'default_calculation_insufficient_data'}
        )

    def calculate_maximum_position_size(self, symbol: str, entry_price: float, 
                                       portfolio_value: float, max_var_contribution: float,
                                       returns_data: pd.Series,
                                       method: VaRMethod = VaRMethod.HISTORICAL) -> Dict[str, Any]:
        """
        Calculate maximum position size based on VaR contribution limit.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price for the position
            portfolio_value: Total portfolio value
            max_var_contribution: Maximum VaR contribution as fraction of portfolio
            returns_data: Historical returns for the symbol
            method: VaR calculation method
            
        Returns:
            Dictionary with position sizing recommendations
        """
        
        if returns_data is None or returns_data.empty:
            return {
                'error': 'No returns data available for VaR-based position sizing',
                'symbol': symbol,
                'fallback_recommendation': 'Use alternative position sizing method'
            }
        
        # Calculate single-unit VaR
        unit_position = PositionInfo(
            symbol=symbol,
            quantity=1.0,
            current_price=entry_price,
            market_value=entry_price,
            weight=entry_price / portfolio_value,
            returns_data=returns_data
        )
        
        unit_var_result = self.calculate_position_var(unit_position, method)
        var_per_unit = unit_var_result.var_absolute
        
        if var_per_unit <= 0:
            return {
                'error': 'Invalid VaR calculation for position sizing',
                'symbol': symbol,
                'var_per_unit': var_per_unit
            }
        
        # Calculate maximum allowable VaR for this position
        max_position_var = portfolio_value * max_var_contribution
        
        # Calculate maximum units
        max_units = max_position_var / var_per_unit
        max_notional = max_units * entry_price
        max_weight = max_notional / portfolio_value
        
        return {
            'symbol': symbol,
            'max_units': int(max_units),
            'max_notional': max_notional,
            'max_weight': max_weight,
            'var_per_unit': var_per_unit,
            'max_var_contribution': max_var_contribution,
            'entry_price': entry_price,
            'method': method.value,
            'confidence_level': self.config.confidence_level,
            'risk_horizon': self.config.risk_horizon.value,
            'calculation_timestamp': datetime.now().isoformat()
        }

    def stress_test_portfolio(self, positions: List[PositionInfo], 
                             stress_scenarios: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform stress testing on portfolio under different market scenarios.
        
        Args:
            positions: List of portfolio positions
            stress_scenarios: Dictionary of scenario names to market shock percentages
            
        Returns:
            Stress test results
        """
        
        portfolio_value = sum(pos.market_value for pos in positions)
        stress_results = {}
        
        for scenario_name, shock_pct in stress_scenarios.items():
            scenario_pnl = 0
            position_impacts = {}
            
            for position in positions:
                # Apply shock to position
                position_shock = position.market_value * (shock_pct / 100)
                scenario_pnl += position_shock
                position_impacts[position.symbol] = position_shock
            
            stress_results[scenario_name] = {
                'shock_percentage': shock_pct,
                'portfolio_pnl': scenario_pnl,
                'portfolio_pnl_percentage': (scenario_pnl / portfolio_value * 100) if portfolio_value > 0 else 0,
                'position_impacts': position_impacts
            }
        
        return {
            'portfolio_value': portfolio_value,
            'stress_scenarios': stress_results,
            'worst_case_scenario': min(stress_results.items(), key=lambda x: x[1]['portfolio_pnl'])[0],
            'best_case_scenario': max(stress_results.items(), key=lambda x: x[1]['portfolio_pnl'])[0],
            'calculation_timestamp': datetime.now().isoformat()
        }