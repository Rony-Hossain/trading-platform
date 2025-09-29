"""Advanced Model Evaluation Framework

Comprehensive evaluation system that compares RandomForest, LightGBM, and XGBoost
models using proper time-series cross-validation and financial performance metrics.
"""

import asyncio
import logging
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from .time_series_cv import AdvancedTimeSeriesCV, CVConfiguration
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Try to import LightGBM and XGBoost with fallbacks
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    model_name: str
    mae: float
    rmse: float
    r2: float
    sharpe_ratio: float
    hit_rate: float
    max_drawdown: float
    volatility: float
    information_ratio: float
    calmar_ratio: float
    training_time: float
    prediction_time: float
    feature_importance: Dict[str, float]
    cv_scores: List[float]
    cv_std: float
    directional_accuracy: float = 0.0
    total_return: float = 0.0
    cv_method: str = 'time_series_split'
    cv_splits_used: int = 0


@dataclass
class ModelConfiguration:
    """Model hyperparameter configuration."""
    model_type: str
    parameters: Dict[str, Any]
    feature_engineering: Dict[str, Any]
    preprocessing: Dict[str, Any]


@dataclass
class EvaluationResult:
    """Complete model evaluation results."""
    best_model: str
    best_score: float
    model_metrics: Dict[str, ModelMetrics]
    model_configs: Dict[str, ModelConfiguration]
    feature_ranking: Dict[str, float]
    evaluation_summary: Dict[str, Any]
    recommendations: List[str]
    artifacts_path: str
    timestamp: datetime


class FinancialMetricsCalculator:
    """Calculate financial performance metrics for model evaluation."""
    
    @staticmethod
    def calculate_returns(predictions: np.ndarray, actual: np.ndarray, 
                         prices: np.ndarray) -> np.ndarray:
        """Calculate returns based on predictions vs actual."""
        # Simple strategy: long if prediction > actual, short if prediction < actual
        signals = np.sign(predictions - actual)
        
        # Calculate price returns
        price_returns = np.diff(prices) / prices[:-1]
        
        # Align signals with returns (signals predict next period return)
        if len(signals) > len(price_returns):
            signals = signals[:len(price_returns)]
        elif len(price_returns) > len(signals):
            price_returns = price_returns[:len(signals)]
        
        # Strategy returns
        strategy_returns = signals * price_returns
        return strategy_returns
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
    
    @staticmethod
    def hit_rate(predictions: np.ndarray, actual: np.ndarray) -> float:
        """Calculate directional accuracy (hit rate)."""
        pred_direction = np.sign(np.diff(predictions))
        actual_direction = np.sign(np.diff(actual))
        
        if len(pred_direction) == 0:
            return 0.0
        
        return np.mean(pred_direction == actual_direction)
    
    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    @staticmethod
    def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate information ratio vs benchmark."""
        excess_returns = returns - benchmark_returns
        if np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        annual_return = np.prod(1 + returns) ** (252 / len(returns)) - 1
        max_dd = abs(FinancialMetricsCalculator.max_drawdown(returns))
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / max_dd


class ModelEvaluationFramework:
    """Advanced framework for evaluating and selecting ML models."""
    
    def __init__(self, artifacts_dir: str = "artifacts/model_evaluation"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Time Series Cross-Validation
        self.tscv = AdvancedTimeSeriesCV()
        
        # Model configurations
        self.model_configs = {
            "random_forest": {
                "model_type": "sklearn",
                "class": RandomForestRegressor,
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "random_state": 42,
                    "n_jobs": -1
                },
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [8, 10, 12, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            }
        }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.model_configs["lightgbm"] = {
                "model_type": "lightgbm",
                "class": lgb.LGBMRegressor,
                "params": {
                    "objective": "regression",
                    "metric": "rmse",
                    "boosting_type": "gbdt",
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "feature_fraction": 0.9,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "verbose": -1,
                    "random_state": 42
                },
                "param_grid": {
                    "num_leaves": [15, 31, 63],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "feature_fraction": [0.8, 0.9, 1.0],
                    "min_child_samples": [20, 50, 100]
                }
            }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.model_configs["xgboost"] = {
                "model_type": "xgboost",
                "class": xgb.XGBRegressor,
                "params": {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "n_estimators": 100,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "n_jobs": -1
                },
                "param_grid": {
                    "max_depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [50, 100, 200],
                    "subsample": [0.7, 0.8, 0.9],
                    "colsample_bytree": [0.7, 0.8, 0.9]
                }
            }
        
        self.metrics_calculator = FinancialMetricsCalculator()
    
    async def evaluate_models(self, X: pd.DataFrame, y: pd.Series, 
                            prices: pd.Series = None,
                            cv_folds: int = 5,
                            cv_method: str = 'walk_forward',
                            cv_config: Optional[CVConfiguration] = None) -> EvaluationResult:
        """Comprehensive model evaluation with time-series cross-validation."""
        logger.info(f"Starting model evaluation with {len(self.model_configs)} models")
        
        # Prepare data
        X_processed, y_processed, prices_processed = self._prepare_data(X, y, prices)
        
        # Configure time series cross-validation
        if cv_config is None:
            cv_config = CVConfiguration(
                method=cv_method,
                n_splits=cv_folds,
                train_window_days=1260,  # 5 years
                test_window_days=252,    # 1 year
                min_train_days=252,      # 1 year minimum
                gap_days=1,              # 1 day gap to prevent leakage
                step_days=126,           # 6 months step
                adaptive_sizing=True
            )
        
        logger.info(f"Using {cv_config.method} CV with {cv_config.n_splits} folds")
        
        # Fallback to simple TimeSeriesSplit if advanced TSCV fails
        try:
            tscv_splits = self.tscv.create_splits(X_processed, cv_config)
            use_advanced_cv = True
        except Exception as e:
            logger.warning(f"Advanced TSCV failed, falling back to TimeSeriesSplit: {e}")
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            use_advanced_cv = False
        
        model_results = {}
        all_feature_importance = {}
        
        # Evaluate each model
        for model_name, config in self.model_configs.items():
            logger.info(f"Evaluating {model_name}")
            
            try:
                if use_advanced_cv:
                    metrics = await self._evaluate_single_model_advanced_cv(
                        model_name, config, X_processed, y_processed, 
                        prices_processed, tscv_splits, cv_config
                    )
                else:
                    metrics = await self._evaluate_single_model(
                        model_name, config, X_processed, y_processed, 
                        prices_processed, tscv
                    )
                model_results[model_name] = metrics
                all_feature_importance.update(metrics.feature_importance)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        if not model_results:
            raise ValueError("No models could be evaluated successfully")
        
        # Select best model
        best_model, best_score = self._select_best_model(model_results)
        
        # Rank features across all models
        feature_ranking = self._rank_features(all_feature_importance)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(model_results, best_model)
        
        # Save artifacts
        artifacts_path = await self._save_artifacts(model_results, feature_ranking)
        
        # Create evaluation summary
        summary = self._create_evaluation_summary(model_results, best_model)
        
        result = EvaluationResult(
            best_model=best_model,
            best_score=best_score,
            model_metrics=model_results,
            model_configs={name: ModelConfiguration(
                model_type=config["model_type"],
                parameters=config["params"],
                feature_engineering={},
                preprocessing={}
            ) for name, config in self.model_configs.items()},
            feature_ranking=feature_ranking,
            evaluation_summary=summary,
            recommendations=recommendations,
            artifacts_path=str(artifacts_path),
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"Model evaluation completed. Best model: {best_model} (score: {best_score:.4f})")
        return result
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                     prices: pd.Series = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for model training and evaluation."""
        # Handle missing values
        X_clean = X.fillna(method='ffill').fillna(method='bfill')
        y_clean = y.fillna(method='ffill').fillna(method='bfill')
        
        # Align data
        common_index = X_clean.index.intersection(y_clean.index)
        X_aligned = X_clean.loc[common_index]
        y_aligned = y_clean.loc[common_index]
        
        if prices is not None:
            prices_aligned = prices.loc[common_index] if hasattr(prices, 'loc') else prices
        else:
            # Create dummy prices if not provided
            prices_aligned = np.arange(len(y_aligned)) + 100
        
        # Convert to numpy arrays
        X_array = X_aligned.values
        y_array = y_aligned.values
        prices_array = np.array(prices_aligned)
        
        # Remove any remaining NaN values
        valid_mask = ~(np.isnan(X_array).any(axis=1) | np.isnan(y_array) | np.isnan(prices_array))
        
        return X_array[valid_mask], y_array[valid_mask], prices_array[valid_mask]
    
    async def _evaluate_single_model(self, model_name: str, config: Dict[str, Any],
                                   X: np.ndarray, y: np.ndarray, prices: np.ndarray,
                                   tscv: TimeSeriesSplit) -> ModelMetrics:
        """Evaluate a single model using time-series cross-validation."""
        
        # Initialize model
        model_class = config["class"]
        model_params = config["params"].copy()
        
        # Cross-validation results
        cv_scores = []
        cv_predictions = []
        cv_actuals = []
        cv_prices = []
        
        training_times = []
        prediction_times = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            prices_test = prices[test_idx]
            
            # Initialize model for this fold
            model = model_class(**model_params)
            
            # Training
            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            training_times.append(training_time)
            
            # Prediction
            start_time = datetime.now()
            y_pred = model.predict(X_test)
            prediction_time = (datetime.now() - start_time).total_seconds()
            prediction_times.append(prediction_time)
            
            # Store results
            cv_predictions.extend(y_pred)
            cv_actuals.extend(y_test)
            cv_prices.extend(prices_test)
            
            # Calculate fold score (RMSE)
            fold_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            cv_scores.append(fold_rmse)
        
        # Convert to numpy arrays
        cv_predictions = np.array(cv_predictions)
        cv_actuals = np.array(cv_actuals)
        cv_prices = np.array(cv_prices)
        
        # Calculate financial metrics
        strategy_returns = self.metrics_calculator.calculate_returns(
            cv_predictions, cv_actuals, cv_prices
        )
        
        # Standard ML metrics
        mae = mean_absolute_error(cv_actuals, cv_predictions)
        rmse = np.sqrt(mean_squared_error(cv_actuals, cv_predictions))
        r2 = r2_score(cv_actuals, cv_predictions)
        
        # Financial metrics
        directional_accuracy = self.metrics_calculator.hit_rate(cv_predictions, cv_actuals)
        sharpe = self.metrics_calculator.sharpe_ratio(strategy_returns)
        hit_rate = directional_accuracy
        max_dd = self.metrics_calculator.max_drawdown(strategy_returns)
        total_return = np.prod(1 + strategy_returns) - 1
        volatility = np.std(strategy_returns) * np.sqrt(252)
        
        # Benchmark (buy and hold)
        benchmark_returns = np.diff(cv_prices) / cv_prices[:-1]
        if len(benchmark_returns) > len(strategy_returns):
            benchmark_returns = benchmark_returns[:len(strategy_returns)]
        
        info_ratio = self.metrics_calculator.information_ratio(strategy_returns, benchmark_returns)
        calmar = self.metrics_calculator.calmar_ratio(strategy_returns)
        
        # Feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
            feature_importance = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # For linear models
            feature_names = [f"feature_{i}" for i in range(len(model.coef_))]
            feature_importance = dict(zip(feature_names, np.abs(model.coef_)))
        
        return ModelMetrics(
            model_name=model_name,
            mae=mae,
            rmse=rmse,
            r2=r2,
            directional_accuracy=directional_accuracy,
            sharpe_ratio=sharpe,
            hit_rate=hit_rate,
            max_drawdown=max_dd,
            total_return=total_return,
            volatility=volatility,
            information_ratio=info_ratio,
            calmar_ratio=calmar,
            training_time=np.mean(training_times),
            prediction_time=np.mean(prediction_times),
            feature_importance=feature_importance,
            cv_scores=cv_scores,
            cv_std=np.std(cv_scores)
        )
    
    async def _evaluate_single_model_advanced_cv(self, model_name: str, config: Dict[str, Any],
                                               X: pd.DataFrame, y: pd.Series, 
                                               prices: pd.Series, cv_splits: List,
                                               cv_config: CVConfiguration) -> ModelMetrics:
        """Evaluate single model using advanced time series cross-validation."""
        try:
            # Create model instance
            if config["model_type"] == "sklearn":
                model = config["class"](**config["params"])
            elif config["model_type"] == "lightgbm" and LIGHTGBM_AVAILABLE:
                model = lgb.LGBMRegressor(**config["params"])
            elif config["model_type"] == "xgboost" and XGBOOST_AVAILABLE:
                model = xgb.XGBRegressor(**config["params"])
            else:
                raise ValueError(f"Unsupported model type or library not available: {config['model_type']}")
            
            cv_scores = []
            feature_importance = {}
            training_times = []
            prediction_times = []
            
            # Use the TSCV framework for evaluation
            cv_results = self.tscv.evaluate_model_with_cv(
                model, X, y, cv_config,
                scoring_metrics=['mse', 'mae', 'r2', 'rmse']
            )
            
            # Extract results from TSCV
            cv_scores = [fold['r2'] for fold in cv_results.fold_scores if 'r2' in fold]
            avg_training_time = cv_results.execution_time / len(cv_results.fold_scores)
            
            # Get feature importance from final model
            model.fit(X, y)
            if hasattr(model, 'feature_importances_'):
                for i, feature in enumerate(X.columns):
                    feature_importance[feature] = float(model.feature_importances_[i])
            
            # Calculate financial metrics using last fold predictions
            if cv_scores and prices is not None:
                # For financial metrics, we need actual predictions vs actuals
                # Use the last CV fold for this calculation
                last_split = cv_splits[-1]
                test_mask = (X.index >= last_split.test_start) & (X.index <= last_split.test_end)
                
                if test_mask.sum() > 0:
                    X_test = X[test_mask]
                    y_test = y[test_mask]
                    prices_test = prices[test_mask] if prices is not None else y_test
                    
                    # Predict on test set
                    predictions = model.predict(X_test)
                    
                    # Calculate financial metrics
                    returns = self.metrics_calculator.calculate_returns(predictions, y_test.values)
                    sharpe = self.metrics_calculator.sharpe_ratio(returns)
                    hit_rate = self.metrics_calculator.hit_rate(predictions, y_test.values) 
                    max_dd = self.metrics_calculator.max_drawdown(returns)
                    volatility = np.std(returns) * np.sqrt(252)
                    info_ratio = self.metrics_calculator.information_ratio(returns, y_test.values)
                    calmar = self.metrics_calculator.calmar_ratio(returns)
                else:
                    # Fallback values if no test data
                    sharpe = hit_rate = max_dd = volatility = info_ratio = calmar = 0.0
            else:
                # Fallback values if no prices or CV scores
                sharpe = hit_rate = max_dd = volatility = info_ratio = calmar = 0.0
                avg_training_time = 0.1
            
            return ModelMetrics(
                model_name=model_name,
                rmse=np.sqrt(np.mean([fold.get('mse', 0) for fold in cv_results.fold_scores])),
                mae=np.mean([fold.get('mae', 0) for fold in cv_results.fold_scores]),
                r2=np.mean(cv_scores) if cv_scores else 0.0,
                sharpe_ratio=sharpe,
                hit_rate=hit_rate,
                max_drawdown=max_dd,
                volatility=volatility,
                information_ratio=info_ratio,
                calmar_ratio=calmar,
                training_time=avg_training_time,
                prediction_time=0.001,  # Simplified for advanced CV
                feature_importance=feature_importance,
                cv_scores=cv_scores,
                cv_std=np.std(cv_scores) if cv_scores else 0.0,
                cv_method=cv_config.method,
                cv_splits_used=len(cv_results.fold_scores)
            )
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name} with advanced CV: {e}")
            # Return default metrics
            return ModelMetrics(
                model_name=model_name,
                rmse=999.0, mae=999.0, r2=-999.0,
                sharpe_ratio=0.0, hit_rate=0.0, max_drawdown=0.0,
                volatility=0.0, information_ratio=0.0, calmar_ratio=0.0,
                training_time=0.0, prediction_time=0.0,
                feature_importance={}, cv_scores=[], cv_std=0.0,
                cv_method=cv_config.method, cv_splits_used=0
            )
    
    def _select_best_model(self, model_results: Dict[str, ModelMetrics]) -> Tuple[str, float]:
        """Select the best model based on composite financial score."""
        best_model = None
        best_score = -float('inf')
        
        for model_name, metrics in model_results.items():
            # Composite score: weighted combination of financial metrics
            score = (
                metrics.sharpe_ratio * 0.4 +           # Risk-adjusted returns
                metrics.information_ratio * 0.2 +      # Excess returns vs benchmark
                metrics.hit_rate * 0.2 +               # Directional accuracy
                (1 - abs(metrics.max_drawdown)) * 0.1 + # Drawdown control
                metrics.r2 * 0.1                       # Explanation power
            )
            
            logger.info(f"{model_name} composite score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model, best_score
    
    def _rank_features(self, all_feature_importance: Dict[str, float]) -> Dict[str, float]:
        """Rank features by average importance across models."""
        feature_scores = {}
        feature_counts = {}
        
        for feature, importance in all_feature_importance.items():
            if feature not in feature_scores:
                feature_scores[feature] = 0
                feature_counts[feature] = 0
            
            feature_scores[feature] += importance
            feature_counts[feature] += 1
        
        # Calculate average importance
        avg_importance = {
            feature: score / feature_counts[feature]
            for feature, score in feature_scores.items()
        }
        
        # Sort by importance
        return dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))
    
    def _generate_recommendations(self, model_results: Dict[str, ModelMetrics], 
                                best_model: str) -> List[str]:
        """Generate actionable recommendations based on evaluation results."""
        recommendations = []
        
        best_metrics = model_results[best_model]
        
        # Model selection recommendation
        recommendations.append(f"Deploy {best_model} as the primary model")
        
        # Performance assessment
        if best_metrics.sharpe_ratio > 1.0:
            recommendations.append("Model shows strong risk-adjusted performance (Sharpe > 1.0)")
        elif best_metrics.sharpe_ratio > 0.5:
            recommendations.append("Model shows moderate risk-adjusted performance")
        else:
            recommendations.append("Model performance below expectations - consider feature engineering")
        
        # Directional accuracy
        if best_metrics.hit_rate > 0.55:
            recommendations.append("Good directional accuracy for trading signals")
        else:
            recommendations.append("Low directional accuracy - review feature relevance")
        
        # Drawdown control
        if abs(best_metrics.max_drawdown) < 0.1:
            recommendations.append("Excellent drawdown control")
        elif abs(best_metrics.max_drawdown) < 0.2:
            recommendations.append("Acceptable drawdown levels")
        else:
            recommendations.append("High drawdown risk - implement additional risk controls")
        
        # Model comparison insights
        if len(model_results) > 1:
            model_scores = [(name, metrics.sharpe_ratio) for name, metrics in model_results.items()]
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            if model_scores[0][1] - model_scores[1][1] < 0.1:
                recommendations.append("Consider ensemble approach - models show similar performance")
        
        # Feature engineering suggestions
        if best_metrics.r2 < 0.3:
            recommendations.append("Low R² suggests need for additional features or feature engineering")
        
        return recommendations
    
    def _create_evaluation_summary(self, model_results: Dict[str, ModelMetrics], 
                                 best_model: str) -> Dict[str, Any]:
        """Create summary of evaluation results."""
        summary = {
            "total_models_evaluated": len(model_results),
            "best_model": best_model,
            "available_models": {
                "lightgbm": LIGHTGBM_AVAILABLE,
                "xgboost": XGBOOST_AVAILABLE,
                "random_forest": True
            },
            "performance_comparison": {},
            "model_rankings": {}
        }
        
        # Performance comparison
        for name, metrics in model_results.items():
            summary["performance_comparison"][name] = {
                "sharpe_ratio": metrics.sharpe_ratio,
                "hit_rate": metrics.hit_rate,
                "max_drawdown": metrics.max_drawdown,
                "rmse": metrics.rmse,
                "r2": metrics.r2
            }
        
        # Rank models by different metrics
        metrics_to_rank = ["sharpe_ratio", "hit_rate", "r2"]
        for metric in metrics_to_rank:
            ranked = sorted(
                model_results.items(),
                key=lambda x: getattr(x[1], metric),
                reverse=True
            )
            summary["model_rankings"][metric] = [name for name, _ in ranked]
        
        return summary
    
    async def _save_artifacts(self, model_results: Dict[str, ModelMetrics], 
                            feature_ranking: Dict[str, float]) -> Path:
        """Save evaluation artifacts to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts_path = self.artifacts_dir / f"evaluation_{timestamp}"
        artifacts_path.mkdir(exist_ok=True)
        
        # Save model metrics
        metrics_data = {name: asdict(metrics) for name, metrics in model_results.items()}
        with open(artifacts_path / "model_metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        # Save feature ranking (convert numpy types to Python types)
        feature_ranking_serializable = {k: float(v) for k, v in feature_ranking.items()}
        with open(artifacts_path / "feature_ranking.json", "w") as f:
            json.dump(feature_ranking_serializable, f, indent=2)
        
        # Save evaluation summary
        summary = {
            "evaluation_timestamp": timestamp,
            "best_model": max(model_results.items(), key=lambda x: x[1].sharpe_ratio)[0],
            "model_count": len(model_results),
            "top_features": {k: float(v) for k, v in list(feature_ranking.items())[:10]}
        }
        
        with open(artifacts_path / "evaluation_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Evaluation artifacts saved to {artifacts_path}")
        return artifacts_path


async def run_model_evaluation_demo():
    """Demo of the model evaluation framework."""
    print("Advanced Model Evaluation Framework Demo")
    print("=" * 60)
    
    # Generate synthetic financial data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create feature matrix
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    
    # Create synthetic target (price returns)
    y = pd.Series(
        0.1 * np.sum(X.iloc[:, :5], axis=1) + 0.01 * np.random.randn(n_samples),
        name="target_returns"
    )
    
    # Create synthetic prices
    prices = pd.Series(
        100 * np.cumprod(1 + y * 0.01),
        name="prices"
    )
    
    # Initialize evaluation framework
    evaluator = ModelEvaluationFramework()
    
    try:
        # Run evaluation
        print(f"Evaluating {len(evaluator.model_configs)} models...")
        print(f"   Available models: {list(evaluator.model_configs.keys())}")
        print(f"   Dataset: {len(X)} samples, {len(X.columns)} features")
        print()
        
        result = await evaluator.evaluate_models(X, y, prices, cv_folds=3)
        
        # Display results
        print("Evaluation Results:")
        print(f"   Best Model: {result.best_model}")
        print(f"   Best Score: {result.best_score:.4f}")
        print()
        
        print("Model Performance Summary:")
        for model_name, metrics in result.model_metrics.items():
            print(f"   {model_name.upper()}:")
            print(f"     Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
            print(f"     Hit Rate: {metrics.hit_rate:.3f}")
            print(f"     Max Drawdown: {metrics.max_drawdown:.3f}")
            print(f"     RMSE: {metrics.rmse:.4f}")
            print(f"     R²: {metrics.r2:.3f}")
            print()
        
        print("Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec}")
        print()
        
        print("Top Features:")
        top_features = list(result.feature_ranking.items())[:5]
        for feature, importance in top_features:
            print(f"   {feature}: {importance:.4f}")
        print()
        
        print(f"Artifacts saved to: {result.artifacts_path}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_model_evaluation_demo())