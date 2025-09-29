#!/usr/bin/env python3
"""
Advanced Time Series Cross-Validation Framework

Implements sophisticated time series cross-validation with walk-forward analysis,
regime-aware splitting, and intelligent sizing to prevent look-ahead bias in 
financial forecasting models.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Iterator, Union
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class CVSplit:
    """Single cross-validation split information."""
    fold_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_size: int
    test_size: int
    regime_label: Optional[str] = None
    market_volatility: Optional[float] = None


@dataclass
class CVConfiguration:
    """Time series cross-validation configuration."""
    method: str  # 'walk_forward', 'expanding_window', 'rolling_window', 'regime_aware'
    n_splits: int
    train_window_days: Optional[int] = None  # For rolling window
    test_window_days: int = 30
    min_train_days: int = 252  # Minimum 1 year of training data
    gap_days: int = 0  # Gap between train and test to prevent leakage
    step_days: int = 30  # Step size for walk forward
    regime_threshold: float = 0.25  # Volatility threshold for regime detection
    adaptive_sizing: bool = True
    preserve_proportion: float = 0.8  # Minimum proportion of data to use


@dataclass
class CVResults:
    """Cross-validation results."""
    configuration: CVConfiguration
    splits: List[CVSplit]
    fold_scores: List[Dict[str, float]]
    aggregate_scores: Dict[str, float]
    execution_time: float
    data_summary: Dict[str, Any]
    regime_analysis: Optional[Dict[str, Any]] = None


class AdvancedTimeSeriesCV:
    """
    Advanced Time Series Cross-Validation with multiple strategies.
    
    Supports walk-forward, expanding window, rolling window, and regime-aware
    cross-validation to prevent look-ahead bias in financial time series.
    """
    
    def __init__(self, 
                 default_config: Optional[CVConfiguration] = None):
        """
        Initialize TSCV with default configuration.
        
        Args:
            default_config: Default CV configuration
        """
        self.default_config = default_config or CVConfiguration(
            method='walk_forward',
            n_splits=5,
            train_window_days=1260,  # 5 years
            test_window_days=252,    # 1 year
            min_train_days=252,      # 1 year minimum
            gap_days=1,              # 1 day gap to prevent leakage
            step_days=126,           # 6 months step
            regime_threshold=0.25,
            adaptive_sizing=True,
            preserve_proportion=0.8
        )
        
        self.cv_history = []
        
    def detect_market_regimes(self, 
                            data: pd.DataFrame, 
                            price_col: str = 'close',
                            volatility_window: int = 30) -> pd.Series:
        """
        Detect market regimes based on volatility.
        
        Args:
            data: Time series data with price information
            price_col: Column name for price data
            volatility_window: Window for volatility calculation
            
        Returns:
            Series with regime labels ('low_vol', 'high_vol')
        """
        try:
            # Calculate rolling volatility
            returns = data[price_col].pct_change()
            volatility = returns.rolling(window=volatility_window).std() * np.sqrt(252)
            
            # Calculate volatility percentiles
            vol_median = volatility.median()
            vol_75th = volatility.quantile(0.75)
            
            # Assign regime labels
            regimes = pd.Series(index=data.index, dtype='object')
            regimes[volatility <= vol_median] = 'low_vol'
            regimes[(volatility > vol_median) & (volatility <= vol_75th)] = 'medium_vol'
            regimes[volatility > vol_75th] = 'high_vol'
            
            # Forward fill missing values
            regimes = regimes.fillna(method='ffill').fillna('medium_vol')
            
            return regimes
            
        except Exception as e:
            logger.warning(f"Error detecting market regimes: {e}")
            # Return default regime
            return pd.Series(['medium_vol'] * len(data), index=data.index)
    
    def calculate_adaptive_window_size(self, 
                                     data: pd.DataFrame,
                                     base_window_days: int,
                                     volatility_col: Optional[str] = None) -> int:
        """
        Calculate adaptive window size based on data characteristics.
        
        Args:
            data: Time series data
            base_window_days: Base window size in days
            volatility_col: Column name for volatility (optional)
            
        Returns:
            Adjusted window size
        """
        try:
            # Base adjustment for data availability
            available_days = len(data)
            max_window = int(available_days * self.default_config.preserve_proportion)
            
            # Volatility-based adjustment
            if volatility_col and volatility_col in data.columns:
                avg_volatility = data[volatility_col].mean()
                if avg_volatility > self.default_config.regime_threshold:
                    # Increase window size in high volatility periods
                    base_window_days = int(base_window_days * 1.2)
                elif avg_volatility < self.default_config.regime_threshold / 2:
                    # Decrease window size in low volatility periods  
                    base_window_days = int(base_window_days * 0.8)
            
            # Ensure minimum size and respect maximum
            adjusted_size = max(
                self.default_config.min_train_days,
                min(base_window_days, max_window)
            )
            
            logger.debug(f"Adaptive window size: {adjusted_size} days (base: {base_window_days})")
            return adjusted_size
            
        except Exception as e:
            logger.warning(f"Error calculating adaptive window size: {e}")
            return max(base_window_days, self.default_config.min_train_days)
    
    def generate_walk_forward_splits(self, 
                                   data: pd.DataFrame,
                                   config: CVConfiguration) -> List[CVSplit]:
        """
        Generate walk-forward cross-validation splits.
        
        Args:
            data: Time series data with datetime index
            config: CV configuration
            
        Returns:
            List of CV splits
        """
        splits = []
        data_index = data.index
        
        # Calculate window sizes
        if config.adaptive_sizing:
            train_window_days = self.calculate_adaptive_window_size(
                data, config.train_window_days or 1260
            )
        else:
            train_window_days = config.train_window_days or 1260
        
        # Calculate split positions
        total_days = len(data)
        min_end_position = train_window_days + config.test_window_days + config.gap_days
        
        if total_days < min_end_position:
            raise ValueError(f"Insufficient data: need {min_end_position} days, have {total_days}")
        
        # Generate splits
        for fold in range(config.n_splits):
            # Calculate split positions
            if config.train_window_days:
                # Rolling window
                train_start_idx = fold * config.step_days
                train_end_idx = train_start_idx + train_window_days
            else:
                # Expanding window
                train_start_idx = 0
                train_end_idx = train_window_days + (fold * config.step_days)
            
            test_start_idx = train_end_idx + config.gap_days
            test_end_idx = test_start_idx + config.test_window_days
            
            # Check bounds
            if test_end_idx > total_days:
                break
                
            # Create split
            split = CVSplit(
                fold_number=fold + 1,
                train_start=data_index[train_start_idx],
                train_end=data_index[train_end_idx - 1],
                test_start=data_index[test_start_idx],
                test_end=data_index[test_end_idx - 1],
                train_size=train_end_idx - train_start_idx,
                test_size=test_end_idx - test_start_idx
            )
            
            splits.append(split)
            
            logger.debug(f"Split {fold + 1}: Train {split.train_size} days, Test {split.test_size} days")
        
        return splits
    
    def generate_regime_aware_splits(self, 
                                   data: pd.DataFrame,
                                   config: CVConfiguration) -> List[CVSplit]:
        """
        Generate regime-aware cross-validation splits.
        
        Args:
            data: Time series data with datetime index
            config: CV configuration
            
        Returns:
            List of CV splits aligned with market regimes
        """
        # Detect market regimes
        regimes = self.detect_market_regimes(data)
        
        # Find regime transitions
        regime_changes = regimes != regimes.shift(1)
        regime_change_dates = data.index[regime_changes].tolist()
        
        splits = []
        
        # Generate splits around regime transitions
        for i, change_date in enumerate(regime_change_dates[1:], 1):
            if i >= config.n_splits:
                break
                
            # Calculate train period (before regime change)
            train_end_date = change_date - timedelta(days=config.gap_days)
            train_start_date = train_end_date - timedelta(days=config.train_window_days or 1260)
            
            # Calculate test period (after regime change)
            test_start_date = change_date
            test_end_date = test_start_date + timedelta(days=config.test_window_days)
            
            # Check if dates are within data range
            if (train_start_date >= data.index[0] and 
                test_end_date <= data.index[-1]):
                
                # Find indices
                train_start_idx = data.index.get_indexer([train_start_date], method='nearest')[0]
                train_end_idx = data.index.get_indexer([train_end_date], method='nearest')[0]
                test_start_idx = data.index.get_indexer([test_start_date], method='nearest')[0]
                test_end_idx = data.index.get_indexer([test_end_date], method='nearest')[0]
                
                # Get regime label for this period
                regime_label = regimes.iloc[test_start_idx]
                
                split = CVSplit(
                    fold_number=i,
                    train_start=data.index[train_start_idx],
                    train_end=data.index[train_end_idx],
                    test_start=data.index[test_start_idx],
                    test_end=data.index[test_end_idx],
                    train_size=train_end_idx - train_start_idx + 1,
                    test_size=test_end_idx - test_start_idx + 1,
                    regime_label=regime_label
                )
                
                splits.append(split)
        
        return splits
    
    def create_splits(self, 
                     data: pd.DataFrame,
                     config: Optional[CVConfiguration] = None) -> List[CVSplit]:
        """
        Create cross-validation splits based on configuration.
        
        Args:
            data: Time series data with datetime index
            config: CV configuration (uses default if None)
            
        Returns:
            List of CV splits
        """
        config = config or self.default_config
        
        logger.info(f"Creating {config.method} CV splits with {config.n_splits} folds")
        
        # Ensure data has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index")
        
        # Sort data by date
        data = data.sort_index()
        
        if config.method == 'walk_forward':
            return self.generate_walk_forward_splits(data, config)
        elif config.method == 'expanding_window':
            # Expanding window (no train_window_days limit)
            expanded_config = CVConfiguration(**asdict(config))
            expanded_config.train_window_days = None
            return self.generate_walk_forward_splits(data, expanded_config)
        elif config.method == 'rolling_window':
            # Rolling window (fixed train_window_days)
            return self.generate_walk_forward_splits(data, config)
        elif config.method == 'regime_aware':
            return self.generate_regime_aware_splits(data, config)
        else:
            raise ValueError(f"Unknown CV method: {config.method}")
    
    def evaluate_model_with_cv(self, 
                             model,
                             X: pd.DataFrame,
                             y: pd.Series,
                             config: Optional[CVConfiguration] = None,
                             scoring_metrics: Optional[List[str]] = None) -> CVResults:
        """
        Evaluate a model using time series cross-validation.
        
        Args:
            model: Scikit-learn compatible model
            X: Feature matrix with datetime index
            y: Target series with datetime index
            config: CV configuration
            scoring_metrics: List of metrics to calculate
            
        Returns:
            CV results with scores and analysis
        """
        start_time = datetime.now()
        config = config or self.default_config
        scoring_metrics = scoring_metrics or ['mse', 'mae', 'r2']
        
        logger.info(f"Starting {config.method} CV evaluation with {config.n_splits} folds")
        
        # Align X and y
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # Create CV splits
        splits = self.create_splits(X, config)
        
        if not splits:
            raise ValueError("No valid CV splits could be created")
        
        fold_scores = []
        regime_analysis = {}
        
        # Evaluate each fold
        for split in splits:
            try:
                # Get train and test data
                train_mask = (X.index >= split.train_start) & (X.index <= split.train_end)
                test_mask = (X.index >= split.test_start) & (X.index <= split.test_end)
                
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]
                
                if len(X_train) == 0 or len(X_test) == 0:
                    logger.warning(f"Empty data for fold {split.fold_number}, skipping")
                    continue
                
                # Train model
                model_clone = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                model_clone.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model_clone.predict(X_test)
                
                # Calculate scores
                scores = {}
                if 'mse' in scoring_metrics:
                    scores['mse'] = mean_squared_error(y_test, y_pred)
                if 'mae' in scoring_metrics:
                    scores['mae'] = mean_absolute_error(y_test, y_pred)
                if 'r2' in scoring_metrics:
                    scores['r2'] = r2_score(y_test, y_pred)
                if 'rmse' in scoring_metrics:
                    scores['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Add fold metadata
                scores['fold'] = split.fold_number
                scores['train_size'] = split.train_size
                scores['test_size'] = split.test_size
                scores['regime'] = split.regime_label
                
                fold_scores.append(scores)
                
                # Regime analysis
                if split.regime_label:
                    if split.regime_label not in regime_analysis:
                        regime_analysis[split.regime_label] = []
                    regime_analysis[split.regime_label].append(scores)
                
                logger.debug(f"Fold {split.fold_number}: R² = {scores.get('r2', 0):.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating fold {split.fold_number}: {e}")
                continue
        
        if not fold_scores:
            raise ValueError("No successful CV folds completed")
        
        # Calculate aggregate scores
        aggregate_scores = {}
        for metric in scoring_metrics:
            metric_scores = [s[metric] for s in fold_scores if metric in s]
            if metric_scores:
                aggregate_scores[f'{metric}_mean'] = np.mean(metric_scores)
                aggregate_scores[f'{metric}_std'] = np.std(metric_scores)
                aggregate_scores[f'{metric}_min'] = np.min(metric_scores)
                aggregate_scores[f'{metric}_max'] = np.max(metric_scores)
        
        # Execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Data summary
        data_summary = {
            'total_samples': len(X),
            'feature_count': X.shape[1],
            'date_range': {
                'start': X.index.min().isoformat(),
                'end': X.index.max().isoformat()
            },
            'successful_folds': len(fold_scores),
            'total_folds_attempted': len(splits)
        }
        
        results = CVResults(
            configuration=config,
            splits=splits,
            fold_scores=fold_scores,
            aggregate_scores=aggregate_scores,
            execution_time=execution_time,
            data_summary=data_summary,
            regime_analysis=regime_analysis if regime_analysis else None
        )
        
        self.cv_history.append(results)
        logger.info(f"CV evaluation completed: {len(fold_scores)} successful folds in {execution_time:.2f}s")
        
        return results
    
    async def save_cv_artifacts(self, 
                              results: CVResults,
                              output_dir: str = "artifacts/time_series_cv") -> Path:
        """Save CV results and artifacts."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts_path = Path(output_dir) / f"tscv_{results.configuration.method}_{timestamp}"
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        with open(artifacts_path / "cv_results.json", "w") as f:
            json.dump(asdict(results), f, indent=2, default=str)
        
        # Save configuration
        with open(artifacts_path / "cv_configuration.json", "w") as f:
            json.dump(asdict(results.configuration), f, indent=2)
        
        # Save fold scores
        fold_scores_df = pd.DataFrame(results.fold_scores)
        fold_scores_df.to_csv(artifacts_path / "fold_scores.csv", index=False)
        
        # Save splits information
        splits_data = [asdict(split) for split in results.splits]
        with open(artifacts_path / "cv_splits.json", "w") as f:
            json.dump(splits_data, f, indent=2, default=str)
        
        # Save summary
        summary = {
            "cv_timestamp": timestamp,
            "method": results.configuration.method,
            "successful_folds": len(results.fold_scores),
            "execution_time": results.execution_time,
            "aggregate_performance": results.aggregate_scores,
            "best_fold": max(results.fold_scores, key=lambda x: x.get('r2', -999))['fold'] if results.fold_scores else None
        }
        
        with open(artifacts_path / "cv_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"CV artifacts saved to {artifacts_path}")
        return artifacts_path


# Demo function for testing
async def run_time_series_cv_demo():
    """Demo of the time series cross-validation framework."""
    print("Advanced Time Series Cross-Validation Demo")
    print("=" * 60)
    
    # Generate synthetic financial time series data
    np.random.seed(42)
    n_samples = 1500  # ~6 years of daily data
    
    # Create datetime index
    start_date = datetime(2018, 1, 1)
    date_index = pd.date_range(start=start_date, periods=n_samples, freq='D')
    
    # Generate synthetic price series with regimes
    returns = []
    volatility_regime = np.random.choice(['low', 'high'], n_samples, p=[0.7, 0.3])
    
    for i, regime in enumerate(volatility_regime):
        if regime == 'low':
            ret = np.random.normal(0.0005, 0.01)  # Low volatility
        else:
            ret = np.random.normal(0.0, 0.03)     # High volatility
        returns.append(ret)
    
    # Create price series
    prices = [100]  # Starting price
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create features
    price_series = pd.Series(prices[1:], index=date_index)
    
    # Technical indicators as features
    X = pd.DataFrame(index=date_index)
    X['sma_5'] = price_series.rolling(5).mean()
    X['sma_20'] = price_series.rolling(20).mean()
    X['rsi'] = 50 + np.random.randn(n_samples) * 10  # Simplified RSI
    X['volume'] = np.random.lognormal(15, 0.5, n_samples)
    X['volatility'] = price_series.pct_change().rolling(20).std() * np.sqrt(252)
    
    # Target: next day return
    y = price_series.pct_change().shift(-1).dropna()
    
    # Align data
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index].fillna(method='ffill').dropna()
    y = y.loc[X.index]
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Date range: {X.index.min()} to {X.index.max()}")
    print()
    
    # Initialize CV framework
    tscv = AdvancedTimeSeriesCV()
    
    # Test different CV methods
    cv_methods = [
        ('walk_forward', CVConfiguration(
            method='walk_forward',
            n_splits=5,
            train_window_days=756,  # 3 years
            test_window_days=126,   # 6 months
            step_days=63            # 3 months step
        )),
        ('expanding_window', CVConfiguration(
            method='expanding_window',
            n_splits=4,
            test_window_days=126,
            step_days=126
        )),
        ('regime_aware', CVConfiguration(
            method='regime_aware',
            n_splits=3,
            train_window_days=504,  # 2 years
            test_window_days=126
        ))
    ]
    
    # Simple model for testing
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    
    for method_name, config in cv_methods:
        print(f"Testing {method_name.upper()} CV...")
        
        try:
            results = tscv.evaluate_model_with_cv(
                model, X, y, config, 
                scoring_metrics=['mse', 'mae', 'r2', 'rmse']
            )
            
            print(f"Configuration: {config.method}")
            print(f"Successful folds: {len(results.fold_scores)}/{len(results.splits)}")
            print(f"Execution time: {results.execution_time:.2f}s")
            
            # Print aggregate scores
            print("Aggregate Performance:")
            for metric, score in results.aggregate_scores.items():
                print(f"  {metric}: {score:.4f}")
            
            # Print fold details
            print("Fold Performance:")
            for i, fold_score in enumerate(results.fold_scores):
                regime_info = f" ({fold_score['regime']})" if fold_score.get('regime') else ""
                print(f"  Fold {fold_score['fold']}: R² = {fold_score.get('r2', 0):.4f}{regime_info}")
            
            # Regime analysis
            if results.regime_analysis:
                print("Regime Analysis:")
                for regime, regime_scores in results.regime_analysis.items():
                    avg_r2 = np.mean([s.get('r2', 0) for s in regime_scores])
                    print(f"  {regime}: {len(regime_scores)} folds, avg R² = {avg_r2:.4f}")
            
            print()
            
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            print()
    
    print("Time Series CV demo completed!")


if __name__ == "__main__":
    asyncio.run(run_time_series_cv_demo())