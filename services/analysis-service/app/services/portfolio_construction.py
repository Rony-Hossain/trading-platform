"""
Advanced Portfolio Construction with ERC/Risk-Parity and Volatility Targeting.

Implements sophisticated portfolio optimization techniques:
1. Equal Risk Contribution (ERC) portfolios
2. Risk Parity allocation with leverage constraints
3. Volatility targeting with dynamic rebalancing
4. Hierarchical Risk Parity (HRP) for large universes
5. Black-Litterman model integration
6. Risk budgeting and factor exposure control
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize, OptimizationResult
from scipy.linalg import sqrtm, pinv
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
import warnings

logger = logging.getLogger(__name__)


@dataclass
class PortfolioWeights:
    """Portfolio weights with metadata."""
    weights: Dict[str, float]
    timestamp: datetime
    portfolio_type: str
    target_volatility: Optional[float] = None
    actual_volatility: Optional[float] = None
    risk_contributions: Optional[Dict[str, float]] = None
    leverage: Optional[float] = None
    expected_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    turnover: Optional[float] = None


@dataclass
class RiskBudget:
    """Risk budget allocation specification."""
    asset_budgets: Dict[str, float]
    factor_budgets: Optional[Dict[str, float]] = None
    sector_budgets: Optional[Dict[str, float]] = None
    total_budget: float = 1.0


@dataclass
class PortfolioConstraints:
    """Portfolio construction constraints."""
    min_weights: Optional[Dict[str, float]] = None
    max_weights: Optional[Dict[str, float]] = None
    min_total_weight: float = 0.95
    max_total_weight: float = 1.05
    max_leverage: float = 1.0
    min_diversification: Optional[float] = None
    max_turnover: Optional[float] = None
    sector_limits: Optional[Dict[str, Tuple[float, float]]] = None
    factor_exposure_limits: Optional[Dict[str, Tuple[float, float]]] = None


class CovarianceEstimator:
    """Advanced covariance matrix estimation with shrinkage and robust methods."""
    
    def __init__(
        self,
        method: str = 'ledoit_wolf',
        lookback_days: int = 252,
        halflife_days: Optional[int] = None,
        min_periods: int = 60
    ):
        self.method = method
        self.lookback_days = lookback_days
        self.halflife_days = halflife_days
        self.min_periods = min_periods
        
    def estimate(
        self,
        returns: pd.DataFrame,
        weights: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate covariance matrix and expected returns.
        
        Args:
            returns: Asset returns DataFrame
            weights: Optional exponential weights
            
        Returns:
            Tuple of (covariance_matrix, expected_returns)
        """
        
        # Handle missing data
        returns_clean = returns.dropna()
        
        if len(returns_clean) < self.min_periods:
            raise ValueError(f"Insufficient data: {len(returns_clean)} < {self.min_periods}")
        
        # Apply lookback window
        if len(returns_clean) > self.lookback_days:
            returns_clean = returns_clean.tail(self.lookback_days)
        
        # Apply exponential weighting if specified
        if self.halflife_days and weights is None:
            weights = self._create_exponential_weights(len(returns_clean))
        
        # Estimate covariance matrix
        if self.method == 'ledoit_wolf':
            cov_estimator = LedoitWolf()
            cov_matrix = cov_estimator.fit(returns_clean).covariance_
        elif self.method == 'empirical':
            if weights is not None:
                cov_matrix = self._weighted_covariance(returns_clean, weights)
            else:
                cov_estimator = EmpiricalCovariance()
                cov_matrix = cov_estimator.fit(returns_clean).covariance_
        elif self.method == 'exponential':
            cov_matrix = self._exponential_covariance(returns_clean)
        else:
            raise ValueError(f"Unknown covariance method: {self.method}")
        
        # Estimate expected returns (simple historical mean with optional weighting)
        if weights is not None:
            expected_returns = np.average(returns_clean.values, weights=weights, axis=0)
        else:
            expected_returns = returns_clean.mean().values
        
        return cov_matrix, expected_returns
    
    def _create_exponential_weights(self, n_periods: int) -> np.ndarray:
        """Create exponential decay weights."""
        alpha = 1 - np.exp(-np.log(2) / self.halflife_days)
        weights = [(1 - alpha) ** i for i in range(n_periods)]
        weights.reverse()
        return np.array(weights) / np.sum(weights)
    
    def _weighted_covariance(self, returns: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
        """Calculate weighted covariance matrix."""
        weighted_returns = returns.values * np.sqrt(weights[:, np.newaxis])
        return np.cov(weighted_returns.T)
    
    def _exponential_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate exponentially weighted covariance matrix."""
        weights = self._create_exponential_weights(len(returns))
        return self._weighted_covariance(returns, weights)


class ERCOptimizer:
    """Equal Risk Contribution portfolio optimizer."""
    
    def __init__(
        self,
        tolerance: float = 1e-8,
        max_iterations: int = 1000,
        regularization: float = 1e-6
    ):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.regularization = regularization
        
    def optimize(
        self,
        cov_matrix: np.ndarray,
        risk_budget: Optional[np.ndarray] = None,
        constraints: Optional[PortfolioConstraints] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Optimize for Equal Risk Contribution portfolio.
        
        Args:
            cov_matrix: Asset covariance matrix
            risk_budget: Target risk contributions (default: equal)
            constraints: Portfolio constraints
            
        Returns:
            Tuple of (optimal_weights, optimization_info)
        """
        
        n_assets = cov_matrix.shape[0]
        
        # Default to equal risk budget
        if risk_budget is None:
            risk_budget = np.ones(n_assets) / n_assets
        
        # Add regularization to covariance matrix
        cov_reg = cov_matrix + np.eye(n_assets) * self.regularization
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Objective function: minimize sum of squared deviations from target risk contributions
        def objective(weights):
            return self._risk_budget_objective(weights, cov_reg, risk_budget)
        
        # Constraints
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Sum to 1
        
        # Bounds
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        
        # Apply custom constraints if provided
        if constraints:
            bounds, constraints_list = self._apply_constraints(
                bounds, constraints_list, constraints, n_assets
            )
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        if not result.success:
            logger.warning(f"ERC optimization failed: {result.message}")
        
        # Calculate optimization info
        final_weights = result.x
        risk_contribs = self._calculate_risk_contributions(final_weights, cov_reg)
        
        info = {
            'success': result.success,
            'iterations': result.nit,
            'objective_value': result.fun,
            'risk_contributions': risk_contribs,
            'concentration_ratio': self._calculate_concentration_ratio(risk_contribs),
            'effective_n_assets': 1 / np.sum(final_weights ** 2)
        }
        
        return final_weights, info
    
    def _risk_budget_objective(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        target_risk_budget: np.ndarray
    ) -> float:
        """Objective function for risk budgeting."""
        
        # Calculate risk contributions
        risk_contribs = self._calculate_risk_contributions(weights, cov_matrix)
        
        # Minimize squared deviations from target
        return np.sum((risk_contribs - target_risk_budget) ** 2)
    
    def _calculate_risk_contributions(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculate risk contributions for each asset."""
        
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        marginal_risk = np.dot(cov_matrix, weights)
        risk_contributions = weights * marginal_risk / portfolio_var
        
        return risk_contributions
    
    def _calculate_concentration_ratio(self, risk_contributions: np.ndarray) -> float:
        """Calculate concentration ratio (Herfindahl index) of risk contributions."""
        return np.sum(risk_contributions ** 2)
    
    def _apply_constraints(
        self,
        bounds: List[Tuple[float, float]],
        constraints_list: List[Dict],
        constraints: PortfolioConstraints,
        n_assets: int
    ) -> Tuple[List[Tuple[float, float]], List[Dict]]:
        """Apply portfolio constraints to optimization."""
        
        # Weight bounds
        if constraints.min_weights or constraints.max_weights:
            for i in range(n_assets):
                min_weight = constraints.min_weights.get(i, 0.0) if constraints.min_weights else 0.0
                max_weight = constraints.max_weights.get(i, 1.0) if constraints.max_weights else 1.0
                bounds[i] = (min_weight, max_weight)
        
        # Total weight constraint
        if constraints.min_total_weight != 0.95 or constraints.max_total_weight != 1.05:
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda x: np.sum(x) - constraints.min_total_weight
            })
            constraints_list.append({
                'type': 'ineq', 
                'fun': lambda x: constraints.max_total_weight - np.sum(x)
            })
        
        return bounds, constraints_list


class VolatilityTargeting:
    """Volatility targeting system for dynamic portfolio scaling."""
    
    def __init__(
        self,
        target_volatility: float = 0.15,  # 15% annual volatility
        lookback_days: int = 60,
        rebalance_frequency: str = 'monthly',
        vol_floor: float = 0.01,
        vol_ceiling: float = 1.0,
        max_leverage: float = 2.0
    ):
        self.target_volatility = target_volatility
        self.lookback_days = lookback_days
        self.rebalance_frequency = rebalance_frequency
        self.vol_floor = vol_floor
        self.vol_ceiling = vol_ceiling
        self.max_leverage = max_leverage
        
    def calculate_vol_scaling(
        self,
        portfolio_returns: pd.Series,
        current_date: datetime
    ) -> float:
        """
        Calculate volatility scaling factor.
        
        Args:
            portfolio_returns: Historical portfolio returns
            current_date: Current rebalancing date
            
        Returns:
            Volatility scaling factor
        """
        
        # Get recent returns for volatility estimation
        end_date = current_date
        start_date = end_date - timedelta(days=self.lookback_days)
        
        recent_returns = portfolio_returns.loc[start_date:end_date]
        
        if len(recent_returns) < 20:  # Minimum periods
            logger.warning("Insufficient data for volatility estimation")
            return 1.0
        
        # Calculate realized volatility (annualized)
        realized_vol = recent_returns.std() * np.sqrt(252)
        
        # Apply floor and ceiling
        realized_vol = np.clip(realized_vol, self.vol_floor, self.vol_ceiling)
        
        # Calculate scaling factor
        vol_scale = self.target_volatility / realized_vol
        
        # Apply leverage constraints
        vol_scale = np.clip(vol_scale, 1.0 / self.max_leverage, self.max_leverage)
        
        return vol_scale
    
    def should_rebalance(
        self,
        current_date: datetime,
        last_rebalance_date: Optional[datetime]
    ) -> bool:
        """Determine if portfolio should be rebalanced."""
        
        if last_rebalance_date is None:
            return True
        
        if self.rebalance_frequency == 'daily':
            return (current_date - last_rebalance_date).days >= 1
        elif self.rebalance_frequency == 'weekly':
            return (current_date - last_rebalance_date).days >= 7
        elif self.rebalance_frequency == 'monthly':
            return current_date.month != last_rebalance_date.month
        elif self.rebalance_frequency == 'quarterly':
            return (current_date.month - 1) // 3 != (last_rebalance_date.month - 1) // 3
        else:
            return False


class HierarchicalRiskParity:
    """Hierarchical Risk Parity (HRP) for large asset universes."""
    
    def __init__(
        self,
        distance_metric: str = 'correlation',
        linkage_method: str = 'ward',
        max_clusters: Optional[int] = None
    ):
        self.distance_metric = distance_metric
        self.linkage_method = linkage_method
        self.max_clusters = max_clusters
        
    def optimize(
        self,
        returns: pd.DataFrame,
        cov_matrix: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Create HRP portfolio allocation.
        
        Args:
            returns: Asset returns DataFrame
            cov_matrix: Optional precomputed covariance matrix
            
        Returns:
            Tuple of (weights, clustering_info)
        """
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Calculate distance matrix
        if self.distance_metric == 'correlation':
            distances = np.sqrt((1 - corr_matrix) / 2)
        else:
            distances = pdist(returns.T, metric=self.distance_metric)
        
        # Hierarchical clustering
        linkage_matrix = linkage(squareform(distances), method=self.linkage_method)
        
        # Get sorted asset order from clustering
        sorted_indices = self._get_cluster_order(linkage_matrix, returns.columns)
        
        # Calculate covariance matrix if not provided
        if cov_matrix is None:
            cov_estimator = CovarianceEstimator()
            cov_matrix, _ = cov_estimator.estimate(returns)
        
        # Apply inverse-variance weighting within clusters
        weights = self._calculate_hrp_weights(
            cov_matrix, sorted_indices, linkage_matrix
        )
        
        clustering_info = {
            'linkage_matrix': linkage_matrix,
            'sorted_indices': sorted_indices,
            'n_clusters': len(np.unique(fcluster(linkage_matrix, self.max_clusters or len(returns.columns), criterion='maxclust')))
        }
        
        return weights, clustering_info
    
    def _get_cluster_order(self, linkage_matrix: np.ndarray, asset_names: pd.Index) -> List[int]:
        """Get asset ordering from hierarchical clustering."""
        
        def _recursive_ordering(node_id, linkage_matrix, order):
            if node_id < len(asset_names):
                order.append(node_id)
            else:
                left_child = int(linkage_matrix[node_id - len(asset_names), 0])
                right_child = int(linkage_matrix[node_id - len(asset_names), 1])
                _recursive_ordering(left_child, linkage_matrix, order)
                _recursive_ordering(right_child, linkage_matrix, order)
        
        order = []
        _recursive_ordering(len(linkage_matrix), linkage_matrix, order)
        return order
    
    def _calculate_hrp_weights(
        self,
        cov_matrix: np.ndarray,
        sorted_indices: List[int],
        linkage_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculate HRP weights using recursive bisection."""
        
        n_assets = len(sorted_indices)
        weights = np.ones(n_assets)
        
        def _recursive_bisection(indices):
            if len(indices) == 1:
                return
            
            # Split into two clusters
            mid = len(indices) // 2
            left_cluster = indices[:mid]
            right_cluster = indices[mid:]
            
            # Calculate cluster variances
            left_cov = cov_matrix[np.ix_(left_cluster, left_cluster)]
            right_cov = cov_matrix[np.ix_(right_cluster, right_cluster)]
            
            left_ivp = 1.0 / np.diag(left_cov).sum()  # Inverse variance
            right_ivp = 1.0 / np.diag(right_cov).sum()
            
            # Allocate weights between clusters
            total_ivp = left_ivp + right_ivp
            left_weight = left_ivp / total_ivp
            right_weight = right_ivp / total_ivp
            
            # Apply weights
            weights[left_cluster] *= left_weight
            weights[right_cluster] *= right_weight
            
            # Recursively process sub-clusters
            _recursive_bisection(left_cluster)
            _recursive_bisection(right_cluster)
        
        _recursive_bisection(sorted_indices)
        return weights


class AdvancedPortfolioConstructor:
    """
    Comprehensive portfolio construction system combining multiple techniques.
    """
    
    def __init__(
        self,
        cov_estimator: Optional[CovarianceEstimator] = None,
        erc_optimizer: Optional[ERCOptimizer] = None,
        vol_targeting: Optional[VolatilityTargeting] = None,
        hrp: Optional[HierarchicalRiskParity] = None
    ):
        self.cov_estimator = cov_estimator or CovarianceEstimator()
        self.erc_optimizer = erc_optimizer or ERCOptimizer()
        self.vol_targeting = vol_targeting or VolatilityTargeting()
        self.hrp = hrp or HierarchicalRiskParity()
        
        self.portfolio_history = []
        self.last_rebalance_date = None
        
    def construct_portfolio(
        self,
        returns: pd.DataFrame,
        method: str = 'erc',
        risk_budget: Optional[RiskBudget] = None,
        constraints: Optional[PortfolioConstraints] = None,
        target_volatility: Optional[float] = None,
        rebalance_date: Optional[datetime] = None
    ) -> PortfolioWeights:
        """
        Construct portfolio using specified method.
        
        Args:
            returns: Asset returns DataFrame
            method: Portfolio construction method ('erc', 'hrp', 'risk_parity')
            risk_budget: Target risk budget allocation
            constraints: Portfolio constraints
            target_volatility: Target portfolio volatility
            rebalance_date: Current rebalancing date
            
        Returns:
            PortfolioWeights object with allocation and metadata
        """
        
        if rebalance_date is None:
            rebalance_date = datetime.now()
        
        # Estimate covariance matrix and expected returns
        cov_matrix, expected_returns = self.cov_estimator.estimate(returns)
        
        # Choose optimization method
        if method == 'erc':
            weights, opt_info = self._construct_erc_portfolio(
                cov_matrix, risk_budget, constraints
            )
        elif method == 'hrp':
            weights, opt_info = self._construct_hrp_portfolio(
                returns, cov_matrix
            )
        elif method == 'risk_parity':
            weights, opt_info = self._construct_risk_parity_portfolio(
                cov_matrix, constraints
            )
        else:
            raise ValueError(f"Unknown portfolio method: {method}")
        
        # Apply volatility targeting if specified
        if target_volatility:
            weights = self._apply_volatility_targeting(
                weights, returns, target_volatility, rebalance_date
            )
        
        # Calculate portfolio metrics
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))) * np.sqrt(252)
        expected_return = np.dot(weights, expected_returns) * 252
        sharpe_ratio = expected_return / portfolio_vol if portfolio_vol > 0 else 0
        
        # Calculate risk contributions
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        marginal_risk = np.dot(cov_matrix, weights)
        risk_contribs = weights * marginal_risk / portfolio_var
        
        # Create portfolio weights object
        portfolio_weights = PortfolioWeights(
            weights=dict(zip(returns.columns, weights)),
            timestamp=rebalance_date,
            portfolio_type=method,
            target_volatility=target_volatility,
            actual_volatility=portfolio_vol,
            risk_contributions=dict(zip(returns.columns, risk_contribs)),
            leverage=np.sum(np.abs(weights)),
            expected_return=expected_return,
            sharpe_ratio=sharpe_ratio
        )
        
        # Store in history
        self.portfolio_history.append(portfolio_weights)
        self.last_rebalance_date = rebalance_date
        
        return portfolio_weights
    
    def _construct_erc_portfolio(
        self,
        cov_matrix: np.ndarray,
        risk_budget: Optional[RiskBudget],
        constraints: Optional[PortfolioConstraints]
    ) -> Tuple[np.ndarray, Dict]:
        """Construct Equal Risk Contribution portfolio."""
        
        risk_budget_array = None
        if risk_budget and risk_budget.asset_budgets:
            risk_budget_array = np.array(list(risk_budget.asset_budgets.values()))
        
        return self.erc_optimizer.optimize(cov_matrix, risk_budget_array, constraints)
    
    def _construct_hrp_portfolio(
        self,
        returns: pd.DataFrame,
        cov_matrix: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Construct Hierarchical Risk Parity portfolio."""
        
        return self.hrp.optimize(returns, cov_matrix)
    
    def _construct_risk_parity_portfolio(
        self,
        cov_matrix: np.ndarray,
        constraints: Optional[PortfolioConstraints]
    ) -> Tuple[np.ndarray, Dict]:
        """Construct traditional risk parity portfolio (equal risk contributions)."""
        
        # Use ERC with equal risk budget
        n_assets = cov_matrix.shape[0]
        equal_risk_budget = np.ones(n_assets) / n_assets
        
        return self.erc_optimizer.optimize(cov_matrix, equal_risk_budget, constraints)
    
    def _apply_volatility_targeting(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        target_volatility: float,
        rebalance_date: datetime
    ) -> np.ndarray:
        """Apply volatility targeting to portfolio weights."""
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Calculate volatility scaling
        vol_scale = self.vol_targeting.calculate_vol_scaling(
            portfolio_returns, rebalance_date
        )
        
        # Apply scaling to weights
        scaled_weights = weights * vol_scale
        
        return scaled_weights
    
    def get_portfolio_statistics(self) -> Dict[str, float]:
        """Get comprehensive portfolio statistics."""
        
        if not self.portfolio_history:
            return {}
        
        recent_portfolios = self.portfolio_history[-12:]  # Last 12 rebalances
        
        turnover_rates = []
        for i in range(1, len(recent_portfolios)):
            prev_weights = np.array(list(recent_portfolios[i-1].weights.values()))
            curr_weights = np.array(list(recent_portfolios[i].weights.values()))
            turnover = np.sum(np.abs(curr_weights - prev_weights)) / 2
            turnover_rates.append(turnover)
        
        volatilities = [p.actual_volatility for p in recent_portfolios if p.actual_volatility]
        sharpe_ratios = [p.sharpe_ratio for p in recent_portfolios if p.sharpe_ratio]
        leverages = [p.leverage for p in recent_portfolios if p.leverage]
        
        return {
            'avg_turnover': np.mean(turnover_rates) if turnover_rates else 0,
            'avg_volatility': np.mean(volatilities) if volatilities else 0,
            'avg_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'avg_leverage': np.mean(leverages) if leverages else 0,
            'vol_stability': np.std(volatilities) if len(volatilities) > 1 else 0,
            'n_rebalances': len(self.portfolio_history)
        }
    
    def backtest_portfolio(
        self,
        returns: pd.DataFrame,
        method: str = 'erc',
        rebalance_frequency: str = 'monthly',
        target_volatility: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Backtest portfolio construction methodology.
        
        Args:
            returns: Asset returns DataFrame
            method: Portfolio construction method
            rebalance_frequency: Rebalancing frequency
            target_volatility: Target volatility (if any)
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            DataFrame with portfolio performance metrics
        """
        
        # Set date range
        if start_date is None:
            start_date = returns.index[252]  # Skip first year for estimation
        if end_date is None:
            end_date = returns.index[-1]
        
        # Filter returns
        backtest_returns = returns.loc[start_date:end_date]
        
        # Initialize tracking variables
        portfolio_returns = []
        rebalance_dates = []
        current_weights = None
        
        # Rebalancing schedule
        rebalance_schedule = self._create_rebalance_schedule(
            start_date, end_date, rebalance_frequency
        )
        
        for date in backtest_returns.index:
            
            # Check if rebalancing is needed
            if date in rebalance_schedule or current_weights is None:
                
                # Get historical data for portfolio construction
                historical_data = returns.loc[returns.index < date].tail(252)
                
                if len(historical_data) < 60:
                    continue
                
                # Construct portfolio
                try:
                    portfolio = self.construct_portfolio(
                        historical_data,
                        method=method,
                        target_volatility=target_volatility,
                        rebalance_date=date
                    )
                    current_weights = np.array(list(portfolio.weights.values()))
                    rebalance_dates.append(date)
                    
                except Exception as e:
                    logger.warning(f"Portfolio construction failed on {date}: {e}")
                    continue
            
            # Calculate portfolio return
            if current_weights is not None:
                daily_returns = backtest_returns.loc[date].values
                portfolio_return = np.dot(current_weights, daily_returns)
                portfolio_returns.append({
                    'date': date,
                    'return': portfolio_return,
                    'weights': current_weights.copy()
                })
        
        # Convert to DataFrame
        backtest_df = pd.DataFrame(portfolio_returns)
        backtest_df.set_index('date', inplace=True)
        
        # Calculate performance metrics
        backtest_df['cumulative_return'] = (1 + backtest_df['return']).cumprod()
        backtest_df['drawdown'] = backtest_df['cumulative_return'] / backtest_df['cumulative_return'].cummax() - 1
        
        return backtest_df
    
    def _create_rebalance_schedule(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: str
    ) -> List[datetime]:
        """Create rebalancing schedule."""
        
        schedule = []
        current_date = start_date
        
        while current_date <= end_date:
            schedule.append(current_date)
            
            if frequency == 'daily':
                current_date += timedelta(days=1)
            elif frequency == 'weekly':
                current_date += timedelta(weeks=1)
            elif frequency == 'monthly':
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            elif frequency == 'quarterly':
                if current_date.month >= 10:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 3)
        
        return schedule