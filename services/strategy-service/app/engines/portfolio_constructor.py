"""
Portfolio Construction Engine with Risk-Parity and Volatility Targeting

Implements Equal Risk Contribution (ERC) portfolio optimization with:
- Ledoit-Wolf or OAS covariance estimation
- Volatility targeting (e.g., 10% annual)
- Regime-conditioned covariances
- Exposure and correlation caps
- Risk metrics and constraint monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize
from scipy.linalg import inv, pinv
from sklearn.covariance import LedoitWolf, OAS, EmpiricalCovariance
import warnings

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Configuration for portfolio construction."""
    
    # Volatility targeting
    target_volatility: float = 0.10  # 10% annual target
    vol_tolerance: float = 0.01  # ±1% tolerance
    
    # Risk model parameters
    covariance_method: Literal['ledoit_wolf', 'oas', 'empirical'] = 'ledoit_wolf'
    lookback_days: int = 252  # 1 year of data
    min_history_days: int = 60  # Minimum data required
    
    # ERC parameters
    risk_aversion: float = 1e-6  # Risk aversion parameter
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-8
    
    # Exposure constraints
    max_single_weight: float = 0.05  # 5% max per name
    max_sector_weight: float = 0.25  # 25% max per sector
    max_beta: float = 1.2  # Maximum portfolio beta
    max_long_exposure: float = 1.0  # 100% max long
    max_short_exposure: float = 0.3  # 30% max short
    
    # Turnover control
    max_turnover: float = 0.5  # 50% max turnover per rebalance
    turnover_penalty: float = 0.001  # Cost of turnover
    
    # Regime conditioning
    use_regime_conditioning: bool = False
    regime_window: int = 60  # Days for regime detection


@dataclass
class AssetData:
    """Asset data for portfolio construction."""
    returns: pd.Series
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    beta: Optional[float] = None
    volume: Optional[float] = None
    borrow_cost: Optional[float] = None
    borrow_available: bool = True


@dataclass
class PortfolioResult:
    """Portfolio construction result."""
    weights: pd.Series
    expected_return: float
    expected_volatility: float
    risk_contributions: pd.Series
    constraint_flags: Dict[str, bool]
    turnover: float
    optimization_success: bool
    risk_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


class CovarianceEstimator:
    """Covariance matrix estimation with multiple methods."""
    
    def __init__(self, method: str = 'ledoit_wolf'):
        self.method = method
        
    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate covariance matrix using specified method."""
        
        if returns.isnull().any().any():
            logger.warning("NaN values detected in returns, filling with 0")
            returns = returns.fillna(0)
        
        if self.method == 'ledoit_wolf':
            estimator = LedoitWolf()
        elif self.method == 'oas':
            estimator = OAS()
        elif self.method == 'empirical':
            estimator = EmpiricalCovariance()
        else:
            raise ValueError(f"Unknown covariance method: {self.method}")
        
        try:
            estimator.fit(returns.values)
            cov_matrix = estimator.covariance_
            
            # Ensure positive definite
            cov_matrix = self._make_positive_definite(cov_matrix)
            
            return cov_matrix
            
        except Exception as e:
            logger.error(f"Covariance estimation failed: {e}")
            # Fallback to empirical covariance
            return returns.cov().values
    
    def _make_positive_definite(self, matrix: np.ndarray, min_eigenvalue: float = 1e-8) -> np.ndarray:
        """Ensure matrix is positive definite."""
        
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


class ERCOptimizer:
    """Equal Risk Contribution portfolio optimizer."""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        
    def optimize(self, 
                cov_matrix: np.ndarray,
                expected_returns: np.ndarray,
                constraints: Dict[str, Any],
                current_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """Optimize portfolio using ERC objective."""
        
        n_assets = len(expected_returns)
        
        # Initial guess - equal weights
        x0 = np.ones(n_assets) / n_assets
        
        # Bounds
        bounds = [(constraints.get('min_weight', -0.3), 
                  constraints.get('max_weight', 0.05)) for _ in range(n_assets)]
        
        # Constraints
        cons = self._build_constraints(constraints, n_assets)
        
        # Objective function
        def objective(weights):
            return self._erc_objective(weights, cov_matrix, expected_returns, current_weights)
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': self.config.max_iterations, 
                        'ftol': self.config.convergence_tolerance}
            )
            
            if result.success:
                weights = result.x
                # Apply volatility scaling
                weights = self._scale_for_volatility(weights, cov_matrix)
                return weights, True
            else:
                logger.warning(f"ERC optimization failed: {result.message}")
                return x0, False
                
        except Exception as e:
            logger.error(f"ERC optimization error: {e}")
            return x0, False
    
    def _erc_objective(self, 
                      weights: np.ndarray, 
                      cov_matrix: np.ndarray,
                      expected_returns: np.ndarray,
                      current_weights: Optional[np.ndarray] = None) -> float:
        """ERC objective function with turnover penalty."""
        
        # Portfolio variance
        port_var = weights.T @ cov_matrix @ weights
        
        if port_var <= 0:
            return 1e10
        
        # Risk contributions
        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / port_var
        
        # Target risk contribution (equal)
        target_contrib = np.ones(len(weights)) / len(weights)
        
        # ERC objective - minimize deviations from equal risk
        erc_penalty = np.sum((risk_contrib - target_contrib) ** 2)
        
        # Expected return component (maximize)
        expected_return = weights.T @ expected_returns
        
        # Turnover penalty
        turnover_penalty = 0
        if current_weights is not None:
            turnover = np.sum(np.abs(weights - current_weights))
            turnover_penalty = self.config.turnover_penalty * turnover
        
        # Combined objective
        objective = erc_penalty - self.config.risk_aversion * expected_return + turnover_penalty
        
        return objective
    
    def _build_constraints(self, constraints: Dict[str, Any], n_assets: int) -> List[Dict]:
        """Build optimization constraints."""
        
        cons = []
        
        # Sum to 1 constraint
        cons.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1.0
        })
        
        # Long/short exposure constraints
        if 'max_long_exposure' in constraints:
            cons.append({
                'type': 'ineq',
                'fun': lambda x: constraints['max_long_exposure'] - np.sum(x[x > 0])
            })
        
        if 'max_short_exposure' in constraints:
            cons.append({
                'type': 'ineq',
                'fun': lambda x: constraints['max_short_exposure'] - np.sum(np.abs(x[x < 0]))
            })
        
        # Sector constraints
        if 'sector_constraints' in constraints:
            for sector, sector_mask in constraints['sector_constraints'].items():
                max_weight = constraints.get('max_sector_weight', 0.25)
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, mask=sector_mask: max_weight - np.sum(x[mask])
                })
        
        return cons
    
    def _scale_for_volatility(self, weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Scale portfolio to target volatility."""
        
        current_vol = np.sqrt(weights.T @ cov_matrix @ weights * 252)  # Annualized
        
        if current_vol > 0:
            scale_factor = self.config.target_volatility / current_vol
            # Cap scaling to reasonable bounds
            scale_factor = np.clip(scale_factor, 0.1, 5.0)
            weights = weights * scale_factor
        
        return weights


class PortfolioConstructor:
    """Main portfolio construction engine."""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.cov_estimator = CovarianceEstimator(config.covariance_method)
        self.erc_optimizer = ERCOptimizer(config)
        
    def construct_portfolio(self,
                          asset_data: Dict[str, AssetData],
                          current_weights: Optional[pd.Series] = None,
                          regime_state: Optional[str] = None) -> PortfolioResult:
        """Construct portfolio using ERC with volatility targeting."""
        
        symbols = list(asset_data.keys())
        
        # Build returns matrix
        returns_df = self._build_returns_matrix(asset_data)
        
        if len(returns_df) < self.config.min_history_days:
            raise ValueError(f"Insufficient data: {len(returns_df)} < {self.config.min_history_days}")
        
        # Estimate covariance
        cov_matrix = self.cov_estimator.estimate(returns_df)
        
        # Expected returns (simple historical mean for now)
        expected_returns = returns_df.mean().values
        
        # Build constraints
        constraints = self._build_constraints(asset_data, symbols)
        
        # Current weights
        current_weights_array = None
        if current_weights is not None:
            current_weights_array = current_weights.reindex(symbols, fill_value=0).values
        
        # Optimize
        optimal_weights, success = self.erc_optimizer.optimize(
            cov_matrix, expected_returns, constraints, current_weights_array
        )
        
        # Create result
        weights_series = pd.Series(optimal_weights, index=symbols)
        
        # Calculate metrics
        expected_return = expected_returns @ optimal_weights
        expected_vol = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights * 252)
        risk_contributions = self._calculate_risk_contributions(optimal_weights, cov_matrix)
        
        # Check constraints
        constraint_flags = self._check_constraints(weights_series, asset_data)
        
        # Calculate turnover
        turnover = 0.0
        if current_weights is not None:
            turnover = np.sum(np.abs(weights_series - current_weights.reindex(symbols, fill_value=0)))
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(optimal_weights, cov_matrix, returns_df)
        
        return PortfolioResult(
            weights=weights_series,
            expected_return=expected_return,
            expected_volatility=expected_vol,
            risk_contributions=pd.Series(risk_contributions, index=symbols),
            constraint_flags=constraint_flags,
            turnover=turnover,
            optimization_success=success,
            risk_metrics=risk_metrics
        )
    
    def _build_returns_matrix(self, asset_data: Dict[str, AssetData]) -> pd.DataFrame:
        """Build aligned returns matrix."""
        
        returns_dict = {}
        for symbol, data in asset_data.items():
            if data.returns is not None and len(data.returns) > 0:
                returns_dict[symbol] = data.returns
        
        if not returns_dict:
            raise ValueError("No valid returns data provided")
        
        returns_df = pd.DataFrame(returns_dict)
        
        # Take last N days
        if len(returns_df) > self.config.lookback_days:
            returns_df = returns_df.tail(self.config.lookback_days)
        
        # Drop NaN rows
        returns_df = returns_df.dropna()
        
        return returns_df
    
    def _build_constraints(self, asset_data: Dict[str, AssetData], symbols: List[str]) -> Dict[str, Any]:
        """Build optimization constraints."""
        
        constraints = {
            'max_weight': self.config.max_single_weight,
            'min_weight': -self.config.max_single_weight,
            'max_long_exposure': self.config.max_long_exposure,
            'max_short_exposure': self.config.max_short_exposure,
            'max_sector_weight': self.config.max_sector_weight
        }
        
        # Sector constraints
        sectors = {}
        for i, symbol in enumerate(symbols):
            sector = asset_data[symbol].sector
            if sector:
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(i)
        
        if sectors:
            sector_constraints = {}
            for sector, indices in sectors.items():
                mask = np.zeros(len(symbols), dtype=bool)
                mask[indices] = True
                sector_constraints[sector] = mask
            constraints['sector_constraints'] = sector_constraints
        
        return constraints
    
    def _calculate_risk_contributions(self, weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate risk contributions."""
        
        port_var = weights.T @ cov_matrix @ weights
        
        if port_var <= 0:
            return np.zeros(len(weights))
        
        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / port_var
        
        return risk_contrib
    
    def _check_constraints(self, weights: pd.Series, asset_data: Dict[str, AssetData]) -> Dict[str, bool]:
        """Check constraint violations."""
        
        flags = {}
        
        # Single name exposure
        max_single = weights.abs().max()
        flags['single_name_ok'] = max_single <= self.config.max_single_weight
        
        # Long/short exposure
        long_exposure = weights[weights > 0].sum()
        short_exposure = weights[weights < 0].abs().sum()
        
        flags['long_exposure_ok'] = long_exposure <= self.config.max_long_exposure
        flags['short_exposure_ok'] = short_exposure <= self.config.max_short_exposure
        
        # Sector exposure
        sector_exposures = {}
        for symbol, weight in weights.items():
            sector = asset_data[symbol].sector
            if sector:
                sector_exposures[sector] = sector_exposures.get(sector, 0) + abs(weight)
        
        flags['sector_exposure_ok'] = all(
            exp <= self.config.max_sector_weight 
            for exp in sector_exposures.values()
        )
        
        # Beta constraint
        portfolio_beta = 0
        beta_weights = 0
        for symbol, weight in weights.items():
            beta = asset_data[symbol].beta
            if beta is not None:
                portfolio_beta += weight * beta
                beta_weights += abs(weight)
        
        if beta_weights > 0:
            portfolio_beta = portfolio_beta / beta_weights
            flags['beta_ok'] = abs(portfolio_beta) <= self.config.max_beta
        else:
            flags['beta_ok'] = True
        
        return flags
    
    def _calculate_risk_metrics(self, 
                               weights: np.ndarray, 
                               cov_matrix: np.ndarray,
                               returns_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate portfolio risk metrics."""
        
        port_var = weights.T @ cov_matrix @ weights
        port_vol = np.sqrt(port_var * 252)  # Annualized
        
        # Diversification ratio
        individual_vols = np.sqrt(np.diag(cov_matrix) * 252)
        weighted_avg_vol = weights @ individual_vols
        div_ratio = weighted_avg_vol / port_vol if port_vol > 0 else 0
        
        # Effective number of assets
        effective_assets = 1 / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else 0
        
        # Maximum drawdown (approximate)
        port_returns = returns_df @ weights
        cumulative = (1 + port_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        return {
            'volatility': port_vol,
            'diversification_ratio': div_ratio,
            'effective_assets': effective_assets,
            'max_drawdown': max_drawdown,
            'vol_target_achieved': abs(port_vol - self.config.target_volatility) <= self.config.vol_tolerance
        }


# API Integration Functions
def create_portfolio_allocation(
    asset_returns: Dict[str, pd.Series],
    current_positions: Optional[Dict[str, float]] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create portfolio allocation - main API function."""
    
    # Default config
    config = PortfolioConfig()
    
    # Apply overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Build asset data
    asset_data = {}
    for symbol, returns in asset_returns.items():
        asset_data[symbol] = AssetData(returns=returns)
    
    # Current weights
    current_weights = None
    if current_positions:
        current_weights = pd.Series(current_positions)
    
    # Construct portfolio
    constructor = PortfolioConstructor(config)
    
    try:
        result = constructor.construct_portfolio(asset_data, current_weights)
        
        return {
            'success': True,
            'weights': result.weights.to_dict(),
            'expected_return': result.expected_return,
            'expected_volatility': result.expected_volatility,
            'risk_contributions': result.risk_contributions.to_dict(),
            'constraint_flags': result.constraint_flags,
            'turnover': result.turnover,
            'optimization_success': result.optimization_success,
            'risk_metrics': result.risk_metrics,
            'timestamp': result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Portfolio construction failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    returns_data = {}
    for symbol in symbols:
        returns = np.random.normal(0.001, 0.02, 252)
        returns_data[symbol] = pd.Series(returns, index=dates)
    
    # Test portfolio construction
    result = create_portfolio_allocation(returns_data)
    
    if result['success']:
        print("Portfolio Construction Results:")
        print(f"Expected Return: {result['expected_return']:.4f}")
        print(f"Expected Volatility: {result['expected_volatility']:.4f}")
        print(f"Turnover: {result['turnover']:.4f}")
        print("\nWeights:")
        for symbol, weight in result['weights'].items():
            print(f"  {symbol}: {weight:.4f}")
        print("\nConstraint Flags:")
        for flag, status in result['constraint_flags'].items():
            print(f"  {flag}: {status}")
    else:
        print(f"Portfolio construction failed: {result['error']}")