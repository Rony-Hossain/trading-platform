"""
Forecasting Service
High-level service for ML-based price forecasting and prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import json
from datetime import datetime, timedelta, timezone
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from ..models.ensemble_forecaster import EnsembleForecaster, ModelEvaluator
from ..models.forecasting_models import LSTMForecaster, RandomForestForecaster, ARIMAForecaster
from ..models.ensemble_stacking import (
    EnsembleStacking, StackingConfig, BaseModelConfig, BlenderType, 
    ValidationStrategy, LSTMBaseModel, GRUBaseModel
)
from ..models.regime_aware_stacking import (
    RegimeAwareStacking, RegimeStackingConfig, RegimeWeightingMode
)
from ..services.regime_analysis import RegimeAnalyzer, MarketRegime
from ..services.model_interpretability import (
    ModelExplainer, ExplainerType, AttributionScope, FeatureAttribution,
    GlobalAttribution, InterpretabilityReport
)
from ..core.data_pipeline import DataPipeline
from .model_evaluation import ModelEvaluationFramework
from .feature_selection import AdvancedFeatureSelector, SelectionResults
from .mlflow_tracking import MLflowTracker, ExperimentConfig, ModelMetrics
from .model_deployment import ModelDeploymentPipeline, DeploymentConfig, PromotionCriteria, DeploymentStage, DeploymentStrategy
from .model_drift_monitor import ModelDriftMonitor, DriftThresholds, ModelDriftReport, DriftStatus
from .latency_tracker import LatencyTracker, SignalEvent, SignalType, LatencyStage

logger = logging.getLogger(__name__)

class ForecastingService:
    """Comprehensive forecasting service with multiple ML models"""
    
    def __init__(self, market_data_url: str = "http://localhost:8002"):
        self.data_pipeline = DataPipeline(market_data_url)
        self.model_cache = {}  # Cache trained models by symbol
        self.model_evaluator = ModelEvaluator()
        self.feature_selector = AdvancedFeatureSelector(
            correlation_threshold=0.9,
            vif_threshold=10.0,
            shap_threshold=0.001,
            min_features=5,
            max_features=50
        )
        self.mlflow_tracker = MLflowTracker(tracking_uri="./mlruns")
        self.deployment_pipeline = ModelDeploymentPipeline(
            tracking_uri="./mlruns",
            staging_port=8004,
            production_port=8005
        )
        self.drift_monitor = ModelDriftMonitor(
            thresholds=DriftThresholds(
                psi_low=0.1,
                psi_medium=0.2,
                psi_high=0.25,
                ks_low=0.1,
                ks_medium=0.2,
                ks_high=0.3,
                min_samples=100
            )
        )
        self.latency_tracker = LatencyTracker(max_history=50000)
        
        # Initialize ensemble stacking configuration
        self.stacking_config = StackingConfig(
            blender_type=BlenderType.RIDGE,
            n_folds=5,
            validation_strategy=ValidationStrategy.TIME_SERIES_SPLIT,
            test_size=0.2,
            purge_gap=5,
            ridge_alpha=1.0,
            scale_features=True,
            validate_base_models=True,
            min_samples_per_fold=50
        )
        self.stacking_cache = {}  # Cache for trained stacking models
        
        # Initialize regime analysis
        self.regime_analyzer = RegimeAnalyzer()
        self.regime_aware_cache = {}  # Cache for regime-aware models
        
        # Regime-aware stacking configuration
        self.regime_stacking_config = RegimeStackingConfig(
            base_config=self.stacking_config,
            weighting_mode=RegimeWeightingMode.HYBRID,
            regime_lookback_days=60,
            performance_window=20,
            min_regime_samples=30,
            confidence_threshold=0.6
        )
        
        # Initialize model interpretability
        self.model_explainer = ModelExplainer(
            explainer_types=[ExplainerType.SHAP_TREE, ExplainerType.LIME_TABULAR, ExplainerType.PERMUTATION]
        )
        self.interpretability_cache = {}  # Cache for model explanations
        
    async def generate_forecast(self, symbol: str, model_type: str = 'ensemble',
                              prediction_horizon: int = 5, period: str = '2y',
                              retrain: bool = False) -> Dict[str, Any]:
        """
        Generate price forecast for a symbol
        
        Args:
            symbol: Stock symbol to forecast
            model_type: 'ensemble', 'stacking', 'lstm', 'random_forest', 'arima'
            prediction_horizon: Number of days to forecast
            period: Historical data period for training
            retrain: Whether to retrain the model
            
        Returns:
            Forecast results with predictions and confidence intervals
        """
        try:
            logger.info(f"Generating {model_type} forecast for {symbol}")
            
            # Get training data
            data = await self.data_pipeline.prepare_data_for_analysis(symbol, period, add_features=True)
            
            if data is None or len(data) < 100:
                return {
                    'error': f'Insufficient data for forecasting {symbol}',
                    'symbol': symbol,
                    'required_samples': 100,
                    'available_samples': len(data) if data is not None else 0
                }
            
            # Check cache for trained model
            model_key = f"{symbol}_{model_type}_{prediction_horizon}"
            
            if not retrain and model_key in self.model_cache:
                logger.info(f"Using cached model for {symbol}")
                forecaster = self.model_cache[model_key]
            else:
                # Train new model
                forecaster = await self._train_forecaster(data, model_type, prediction_horizon)
                
                if forecaster is None:
                    return {'error': f'Failed to train {model_type} model for {symbol}'}
                
                # Cache the model
                self.model_cache[model_key] = forecaster
            
            # Generate predictions
            forecast_result = forecaster.predict(data, 'close')
            
            if 'error' in forecast_result:
                return forecast_result
            
            # Enhance forecast with additional analysis
            enhanced_forecast = await self._enhance_forecast(forecast_result, data, symbol)
            
            logger.info(f"Forecast generated successfully for {symbol}")
            return enhanced_forecast
            
        except Exception as e:
            logger.error(f"Error generating forecast for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    async def _train_forecaster(self, data: pd.DataFrame, model_type: str, 
                              prediction_horizon: int) -> Optional[Any]:
        """Train the specified forecasting model"""
        try:
            if model_type == 'ensemble':
                forecaster = EnsembleForecaster(prediction_horizon=prediction_horizon)
            elif model_type == 'stacking':
                return await self._train_stacking_model(data, prediction_horizon)
            elif model_type == 'lstm':
                forecaster = LSTMForecaster(prediction_horizon=prediction_horizon)
            elif model_type == 'random_forest':
                forecaster = RandomForestForecaster(prediction_horizon=prediction_horizon)
            elif model_type == 'arima':
                forecaster = ARIMAForecaster(prediction_horizon=prediction_horizon)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
            
            # Train the model
            train_result = await asyncio.get_event_loop().run_in_executor(
                None, forecaster.train, data, 'close'
            )
            
            if 'error' in train_result:
                logger.error(f"Training failed: {train_result['error']}")
                return None
            
            logger.info(f"{model_type} model trained successfully")
            return forecaster
            
        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
            return None
    
    async def _enhance_forecast(self, forecast_result: Dict[str, Any], 
                              data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Enhance forecast with additional analysis and context"""
        try:
            enhanced = forecast_result.copy()
            
            # Add price movement analysis
            current_price = enhanced.get('current_price', 0)
            predictions = enhanced.get('predictions', [])
            
            if predictions and current_price > 0:
                # Price change analysis
                price_changes = [(pred - current_price) / current_price * 100 for pred in predictions]
                
                enhanced['price_analysis'] = {
                    'expected_return_1d': price_changes[0] if len(price_changes) > 0 else None,
                    'expected_return_5d': price_changes[-1] if len(price_changes) > 0 else None,
                    'max_expected_gain': max(price_changes) if price_changes else None,
                    'max_expected_loss': min(price_changes) if price_changes else None,
                    'price_volatility': np.std(price_changes) if len(price_changes) > 1 else None
                }
                
                # Trend analysis
                if len(predictions) >= 3:
                    trend_slope = (predictions[-1] - predictions[0]) / len(predictions)
                    if trend_slope > current_price * 0.01:  # > 1% over period
                        trend = "BULLISH"
                    elif trend_slope < -current_price * 0.01:  # < -1% over period
                        trend = "BEARISH"
                    else:
                        trend = "SIDEWAYS"
                else:
                    trend = "UNCERTAIN"
                
                enhanced['trend_forecast'] = {
                    'direction': trend,
                    'slope': float(trend_slope) if 'trend_slope' in locals() else None,
                    'confidence': self._calculate_trend_confidence(predictions)
                }
            
            # Risk analysis
            if 'prediction_intervals' in enhanced:
                intervals = enhanced['prediction_intervals']
                risks = []
                
                for i, (pred, interval) in enumerate(zip(predictions, intervals)):
                    if isinstance(interval, dict):
                        downside_risk = (current_price - interval.get('lower', pred)) / current_price * 100
                        upside_potential = (interval.get('upper', pred) - current_price) / current_price * 100
                        
                        risks.append({
                            'day': i + 1,
                            'downside_risk_pct': float(downside_risk),
                            'upside_potential_pct': float(upside_potential),
                            'risk_reward_ratio': float(upside_potential / abs(downside_risk)) if downside_risk != 0 else None
                        })
                
                enhanced['risk_analysis'] = {
                    'daily_risks': risks,
                    'max_downside_risk': max((r['downside_risk_pct'] for r in risks), default=0),
                    'max_upside_potential': max((r['upside_potential_pct'] for r in risks), default=0)
                }
            
            # Market context
            recent_volatility = data['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
            enhanced['market_context'] = {
                'recent_volatility_annualized': float(recent_volatility) if not pd.isna(recent_volatility) else None,
                'data_quality_score': self._calculate_data_quality(data),
                'forecast_reliability': self._assess_forecast_reliability(enhanced, data)
            }
            
            # Add timestamp and metadata
            enhanced['forecast_metadata'] = {
                'generated_at': datetime.now().isoformat(),
                'symbol': symbol,
                'training_period': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
                'training_samples': len(data),
                'model_version': '2.0'
            }
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing forecast: {e}")
            # Return original forecast if enhancement fails
            return forecast_result
    
    def _calculate_trend_confidence(self, predictions: List[float]) -> str:
        """Calculate confidence in trend direction"""
        if len(predictions) < 3:
            return "LOW"
        
        # Calculate consistency of trend
        diffs = np.diff(predictions)
        positive_moves = sum(1 for d in diffs if d > 0)
        negative_moves = sum(1 for d in diffs if d < 0)
        
        consistency = max(positive_moves, negative_moves) / len(diffs)
        
        if consistency >= 0.8:
            return "HIGH"
        elif consistency >= 0.6:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_data_quality(self, data: pd.DataFrame) -> float:
        """Calculate data quality score (0-1)"""
        try:
            score = 1.0
            
            # Check for missing values
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            score -= missing_ratio * 0.3
            
            # Check for data recency
            last_date = data.index[-1]
            days_old = (datetime.now().date() - last_date.date()).days
            if days_old > 5:
                score -= min(days_old / 30, 0.2)  # Reduce score for stale data
            
            # Check for data consistency
            if 'close' in data.columns:
                price_jumps = data['close'].pct_change().abs()
                extreme_moves = (price_jumps > 0.2).sum()  # Moves > 20%
                if extreme_moves > 0:
                    score -= min(extreme_moves / len(data) * 0.5, 0.2)
            
            return max(0, min(1, score))
            
        except Exception:
            return 0.5  # Default moderate quality
    
    def _assess_forecast_reliability(self, forecast: Dict[str, Any], data: pd.DataFrame) -> str:
        """Assess forecast reliability based on various factors"""
        try:
            reliability_score = 0.5  # Start with neutral
            
            # Model agreement (for ensemble)
            if forecast.get('model_type') == 'Ensemble':
                prediction_strength = forecast.get('prediction_strength', {})
                agreement = prediction_strength.get('agreement', 0.5)
                reliability_score += (agreement - 0.5) * 0.4
            
            # Data quality impact
            data_quality = self._calculate_data_quality(data)
            reliability_score += (data_quality - 0.5) * 0.3
            
            # Sample size impact
            sample_ratio = min(len(data) / 500, 1)  # Optimal around 500+ samples
            reliability_score += (sample_ratio - 0.5) * 0.2
            
            # Market volatility impact (higher vol = lower reliability)
            if 'market_context' in forecast:
                volatility = forecast['market_context'].get('recent_volatility_annualized', 0.3)
                vol_impact = max(0, (0.5 - volatility)) * 0.1
                reliability_score += vol_impact
            
            # Classify reliability
            reliability_score = max(0, min(1, reliability_score))
            
            if reliability_score >= 0.75:
                return "HIGH"
            elif reliability_score >= 0.55:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception:
            return "MEDIUM"
    
    async def retrain_random_forest_model(
        self,
        symbol: str,
        period: str = '2y',
        evaluation_holdout: int = 60,
        artifact_dir: str = 'artifacts/random_forest'
    ) -> Dict[str, Any]:
        """Retrain RandomForest with expanded features and log evaluation metadata."""
        try:
            logger.info("Retraining Random Forest for %s", symbol)
            data = await self.data_pipeline.prepare_data_for_analysis(symbol, period, add_features=True)
            if data is None or len(data) < 150:
                return {
                    'error': 'insufficient_data',
                    'message': f'Need at least 150 records, got {0 if data is None else len(data)}'
                }

            data = data.dropna()
            if len(data) < 150:
                return {
                    'error': 'insufficient_data',
                    'message': 'Not enough usable rows after cleaning'
                }

            holdout = max(20, min(evaluation_holdout, len(data) // 4))
            if len(data) - holdout < 100:
                holdout = max(20, len(data) // 5)
            train_df = data.iloc[:-holdout]
            eval_df = data.iloc[-(holdout + 5):].copy()

            forecaster = RandomForestForecaster(prediction_horizon=5)
            training_result = forecaster.train(train_df)
            if 'error' in training_result:
                return {'error': 'training_failed', 'details': training_result['error']}

            eval_metrics = {}
            baseline_metrics = {}
            predictions_detail = {}
            try:
                features_eval, targets_eval = forecaster.prepare_features_and_targets(eval_df)
                if not features_eval.empty and 1 in targets_eval:
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    target_series = targets_eval[1]
                    aligned_index = target_series.index
                    features_eval = features_eval.loc[aligned_index]
                    scaled_eval = forecaster.scaler.transform(features_eval)
                    model = forecaster.models.get(1)
                    if model:
                        preds = model.predict(scaled_eval)
                        y_true = target_series.values
                        rf_mae = float(mean_absolute_error(y_true, preds))
                        rf_rmse = float(mean_squared_error(y_true, preds, squared=False))
                        with np.errstate(divide='ignore', invalid='ignore'):
                            perc_errors = np.abs((y_true - preds) / y_true)
                            perc_errors = perc_errors[np.isfinite(perc_errors)]
                        rf_mape = float(np.mean(perc_errors) * 100) if perc_errors.size else None
                        eval_metrics = {
                            'mae': rf_mae,
                            'rmse': rf_rmse,
                            'mape': rf_mape,
                            'samples': int(len(y_true))
                        }
                        baseline_pred = eval_df.loc[aligned_index, 'close'].values
                        base_mae = float(mean_absolute_error(y_true, baseline_pred))
                        base_rmse = float(mean_squared_error(y_true, baseline_pred, squared=False))
                        with np.errstate(divide='ignore', invalid='ignore'):
                            base_perc_errors = np.abs((y_true - baseline_pred) / y_true)
                            base_perc_errors = base_perc_errors[np.isfinite(base_perc_errors)]
                        base_mape = float(np.mean(base_perc_errors) * 100) if base_perc_errors.size else None
                        baseline_metrics = {
                            'mae': base_mae,
                            'rmse': base_rmse,
                            'mape': base_mape,
                            'strategy': 'naive_last_close'
                        }
                        predictions_detail = {
                            'index': [idx.isoformat() if hasattr(idx, 'isoformat') else str(idx) for idx in aligned_index],
                            'predictions': [float(x) for x in preds],
                            'actuals': [float(x) for x in y_true],
                            'baseline': [float(x) for x in baseline_pred]
                        }
            except Exception as eval_exc:
                logger.warning("Evaluation failed for %s: %s", symbol, eval_exc)

            metadata = {
                'symbol': symbol,
                'period': period,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'training': training_result,
                'evaluation_holdout': holdout,
                'evaluation': {
                    'random_forest': eval_metrics,
                    'baseline': baseline_metrics
                },
                'predictions': predictions_detail
            }

            try:
                metadata['top_features'] = training_result.get('feature_importance', {})
            except Exception:
                metadata['top_features'] = {}

            # Log to MLflow instead of JSON files
            try:
                rf_experiment_config = ExperimentConfig(
                    experiment_name=f"random_forest_retrain_{symbol}",
                    run_name=f"{symbol}_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    tags={"symbol": symbol, "model_type": "random_forest", "retrain": "true"}
                )
                
                rf_metrics = ModelMetrics(
                    mae=eval_metrics.get('mae'),
                    rmse=eval_metrics.get('rmse'),
                    r2=None,
                    sharpe_ratio=None,
                    hit_rate=None,
                    max_drawdown=None,
                    total_return=None,
                    volatility=None,
                    information_ratio=None,
                    calmar_ratio=None,
                    training_time=training_result.get('training_time', 0.0),
                    prediction_time=None
                )
                
                # Log experiment to MLflow
                mlflow_result = await self.mlflow_tracker.log_complete_experiment(
                    experiment_config=rf_experiment_config,
                    model=forecaster,
                    parameters={
                        "symbol": symbol,
                        "period": period,
                        "evaluation_holdout": holdout,
                        "prediction_horizon": 5,
                        "model_type": "RandomForestForecaster"
                    },
                    metrics=rf_metrics,
                    artifacts={
                        "training_results": training_result,
                        "evaluation_metrics": eval_metrics,
                        "baseline_metrics": baseline_metrics,
                        "predictions_detail": predictions_detail,
                        "feature_importance": metadata.get('top_features', {})
                    },
                    model_name=f"{symbol}_random_forest_model"
                )
                
                metadata['mlflow_run_id'] = mlflow_result.run_id
                metadata['mlflow_experiment_id'] = mlflow_result.experiment_id
                logger.info(f"Random Forest retraining logged to MLflow: {mlflow_result.run_id}")
                
            except Exception as mlflow_error:
                logger.warning(f"MLflow logging failed for Random Forest retrain: {mlflow_error}")
                # Fallback to file logging if MLflow fails
                artifact_path = Path(artifact_dir)
                artifact_path.mkdir(parents=True, exist_ok=True)
                filename = artifact_path / f"{symbol.lower()}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
                with filename.open('w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Fallback: Random Forest retraining metadata written to {filename}")

            return metadata
        except Exception as exc:
            logger.error("Random Forest retraining workflow failed: %s", exc)
            return {'error': 'retrain_exception', 'details': str(exc)}

    async def evaluate_model_performance(self, symbol: str, model_type: str = 'ensemble',
                                       evaluation_period: str = '1y') -> Dict[str, Any]:
        """Evaluate model performance using walk-forward validation"""
        try:
            logger.info(f"Evaluating {model_type} model performance for {symbol}")
            
            # Get data for evaluation
            data = await self.data_pipeline.prepare_data_for_analysis(symbol, evaluation_period, add_features=True)
            
            if data is None or len(data) < 200:
                return {'error': f'Insufficient data for evaluation: {len(data) if data else 0} samples'}
            
            # Initialize forecaster
            forecaster = EnsembleForecaster() if model_type == 'ensemble' else None
            if forecaster is None:
                return {'error': f'Model type {model_type} not supported for evaluation'}
            
            # Perform walk-forward validation
            evaluation_result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.model_evaluator.walk_forward_validation,
                forecaster, data, 'close'
            )
            
            if 'error' in evaluation_result:
                return evaluation_result
            
            # Add evaluation metadata
            evaluation_result['evaluation_metadata'] = {
                'symbol': symbol,
                'model_type': model_type,
                'evaluation_period': evaluation_period,
                'evaluation_date': datetime.now().isoformat(),
                'data_samples': len(data)
            }
            
            logger.info(f"Model evaluation completed for {symbol}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            return {'error': str(e)}
    
    async def batch_forecast(self, symbols: List[str], model_type: str = 'ensemble',
                           prediction_horizon: int = 5) -> Dict[str, Any]:
        """Generate forecasts for multiple symbols"""
        try:
            if len(symbols) > 10:
                return {'error': 'Maximum 10 symbols allowed for batch forecasting'}
            
            logger.info(f"Generating batch forecasts for {len(symbols)} symbols")
            
            # Process forecasts concurrently
            tasks = [
                self.generate_forecast(symbol, model_type, prediction_horizon)
                for symbol in symbols
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Organize results
            forecasts = {}
            errors = {}
            
            for i, (symbol, result) in enumerate(zip(symbols, results)):
                if isinstance(result, Exception):
                    errors[symbol] = str(result)
                elif 'error' in result:
                    errors[symbol] = result['error']
                else:
                    forecasts[symbol] = result
            
            return {
                'batch_forecast_timestamp': datetime.now().isoformat(),
                'requested_symbols': symbols,
                'successful_forecasts': len(forecasts),
                'failed_forecasts': len(errors),
                'forecasts': forecasts,
                'errors': errors if errors else None,
                'model_type': model_type,
                'prediction_horizon': prediction_horizon
            }
            
        except Exception as e:
            logger.error(f"Error in batch forecasting: {e}")
            return {'error': str(e)}
    
    def clear_model_cache(self, symbol: str = None):
        """Clear cached models"""
        if symbol:
            # Clear models for specific symbol
            keys_to_remove = [key for key in self.model_cache.keys() if key.startswith(f"{symbol}_")]
            for key in keys_to_remove:
                del self.model_cache[key]
            logger.info(f"Cleared cached models for {symbol}")
        else:
            # Clear all cached models
            self.model_cache.clear()
            logger.info("Cleared all cached models")
    
    async def evaluate_advanced_models(
        self,
        symbol: str,
        period: str = '2y',
        cv_folds: int = 5,
        include_financial_metrics: bool = True,
        cv_method: str = 'walk_forward'
    ) -> Dict[str, Any]:
        """
        Enhanced model evaluation comparing RandomForest, LightGBM, and XGBoost
        with proper time-series cross-validation and financial metrics.
        
        Args:
            symbol: Stock symbol to evaluate models for
            period: Historical data period for training
            cv_folds: Number of time-series cross-validation folds
            include_financial_metrics: Whether to calculate financial performance metrics
            
        Returns:
            Comprehensive evaluation results with model selection recommendation
        """
        try:
            logger.info(f"Starting advanced model evaluation for {symbol}")
            
            # Prepare enhanced dataset
            data = await self.data_pipeline.prepare_data_for_analysis(
                symbol, period, add_features=True
            )
            
            if data is None or len(data) < 200:
                return {
                    'error': 'insufficient_data',
                    'message': f'Need at least 200 records for robust evaluation, got {0 if data is None else len(data)}',
                    'symbol': symbol
                }
            
            # Clean data
            data_clean = data.dropna()
            if len(data_clean) < 200:
                return {
                    'error': 'insufficient_data_after_cleaning',
                    'message': f'Not enough usable rows after cleaning: {len(data_clean)}',
                    'symbol': symbol
                }
            
            logger.info(f"Using {len(data_clean)} samples with {len(data_clean.columns)} features")
            
            # Prepare features and target
            # Use next day's return as target
            target_col = 'close'
            if target_col not in data_clean.columns:
                return {
                    'error': 'missing_target',
                    'message': f'Target column {target_col} not found in data',
                    'available_columns': list(data_clean.columns)
                }
            
            # Calculate returns as target
            prices = data_clean[target_col]
            returns = prices.pct_change().shift(-1)  # Next day return
            
            # Feature columns (exclude price columns and target)
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 'timestamp']
            feature_cols = [col for col in data_clean.columns if col not in exclude_cols]
            
            if len(feature_cols) < 5:
                return {
                    'error': 'insufficient_features',
                    'message': f'Need at least 5 features, got {len(feature_cols)}',
                    'available_features': feature_cols
                }
            
            # Prepare data for evaluation
            X = data_clean[feature_cols].iloc[:-1]  # Remove last row (no target)
            y = returns.iloc[:-1].dropna()  # Remove NaN returns
            prices_aligned = prices.iloc[:-1]
            
            # Align data
            common_index = X.index.intersection(y.index).intersection(prices_aligned.index)
            X_final = X.loc[common_index]
            y_final = y.loc[common_index]
            prices_final = prices_aligned.loc[common_index]
            
            logger.info(f"Final dataset: {len(X_final)} samples, {len(X_final.columns)} features")
            
            # Initialize evaluation framework
            evaluator = ModelEvaluationFramework(
                artifacts_dir=f"artifacts/model_evaluation/{symbol}"
            )
            
            # Run comprehensive evaluation
            evaluation_result = await evaluator.evaluate_models(
                X=X_final,
                y=y_final,
                prices=prices_final,
                cv_folds=cv_folds,
                cv_method=cv_method
            )
            
            # Log experiment to MLflow
            experiment_config = ExperimentConfig(
                experiment_name=f"model_evaluation_{symbol}",
                run_name=f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags={"symbol": symbol, "period": period, "cv_method": cv_method}
            )
            
            # Log each model's performance
            for model_name, metrics in evaluation_result.model_metrics.items():
                try:
                    mlflow_metrics = ModelMetrics(
                        mae=metrics.mae,
                        rmse=metrics.rmse,
                        r2=metrics.r2,
                        sharpe_ratio=metrics.sharpe_ratio,
                        hit_rate=metrics.hit_rate,
                        max_drawdown=metrics.max_drawdown,
                        total_return=metrics.total_return,
                        volatility=metrics.volatility,
                        information_ratio=metrics.information_ratio,
                        calmar_ratio=metrics.calmar_ratio,
                        training_time=metrics.training_time,
                        prediction_time=metrics.prediction_time
                    )
                    
                    # Create experiment for this model
                    model_run_config = ExperimentConfig(
                        experiment_name=f"model_evaluation_{symbol}",
                        run_name=f"{symbol}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        tags={"symbol": symbol, "model": model_name, "period": period}
                    )
                    
                    # Log model experiment
                    await self.mlflow_tracker.log_complete_experiment(
                        experiment_config=model_run_config,
                        model=evaluation_result.trained_models.get(model_name),
                        parameters={
                            "model_type": model_name,
                            "symbol": symbol,
                            "period": period,
                            "cv_folds": cv_folds,
                            "cv_method": cv_method,
                            "features_count": len(X_final.columns),
                            "samples_count": len(X_final)
                        },
                        metrics=mlflow_metrics,
                        artifacts={
                            "feature_importance": dict(evaluation_result.feature_ranking),
                            "evaluation_summary": evaluation_result.evaluation_summary
                        },
                        model_name=f"{symbol}_{model_name}_model"
                    )
                    
                except Exception as mlflow_error:
                    logger.warning(f"MLflow logging failed for {model_name}: {mlflow_error}")
                    # Continue without failing the entire evaluation
            
            # Prepare response
            result = {
                'symbol': symbol,
                'evaluation_timestamp': evaluation_result.timestamp.isoformat(),
                'best_model': evaluation_result.best_model,
                'best_score': evaluation_result.best_score,
                'data_summary': {
                    'total_samples': len(X_final),
                    'feature_count': len(X_final.columns),
                    'cv_folds': cv_folds,
                    'period': period
                },
                'model_performance': {},
                'recommendations': evaluation_result.recommendations,
                'top_features': dict(list(evaluation_result.feature_ranking.items())[:10]),
                'artifacts_path': evaluation_result.artifacts_path
            }
            
            # Add model performance details
            for model_name, metrics in evaluation_result.model_metrics.items():
                result['model_performance'][model_name] = {
                    'ml_metrics': {
                        'mae': metrics.mae,
                        'rmse': metrics.rmse,
                        'r2': metrics.r2,
                        'cv_mean': np.mean(metrics.cv_scores),
                        'cv_std': metrics.cv_std
                    },
                    'financial_metrics': {
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'hit_rate': metrics.hit_rate,
                        'max_drawdown': metrics.max_drawdown,
                        'total_return': metrics.total_return,
                        'volatility': metrics.volatility,
                        'information_ratio': metrics.information_ratio,
                        'calmar_ratio': metrics.calmar_ratio
                    },
                    'performance_metrics': {
                        'training_time': metrics.training_time,
                        'prediction_time': metrics.prediction_time
                    }
                }
            
            # Add model availability info
            result['model_availability'] = evaluation_result.evaluation_summary['available_models']
            
            # Cache the best model for future use
            cache_key = f"{symbol}_best_model"
            self.model_cache[cache_key] = {
                'model_type': evaluation_result.best_model,
                'evaluation_result': evaluation_result,
                'timestamp': datetime.now(),
                'symbol': symbol
            }
            
            logger.info(f"Advanced model evaluation completed for {symbol}. Best model: {evaluation_result.best_model}")
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced model evaluation for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': 'evaluation_failed',
                'message': str(e),
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_model_recommendation(self, symbol: str) -> Dict[str, Any]:
        """
        Get model recommendation based on recent evaluation results.
        
        Args:
            symbol: Stock symbol to get recommendation for
            
        Returns:
            Model recommendation with rationale
        """
        try:
            cache_key = f"{symbol}_best_model"
            
            if cache_key in self.model_cache:
                cached_result = self.model_cache[cache_key]
                evaluation_result = cached_result['evaluation_result']
                
                return {
                    'symbol': symbol,
                    'recommended_model': evaluation_result.best_model,
                    'confidence_score': evaluation_result.best_score,
                    'evaluation_date': cached_result['timestamp'].isoformat(),
                    'rationale': evaluation_result.recommendations,
                    'performance_summary': {
                        'sharpe_ratio': evaluation_result.model_metrics[evaluation_result.best_model].sharpe_ratio,
                        'hit_rate': evaluation_result.model_metrics[evaluation_result.best_model].hit_rate,
                        'r2_score': evaluation_result.model_metrics[evaluation_result.best_model].r2
                    },
                    'alternative_models': list(evaluation_result.model_metrics.keys())
                }
            else:
                return {
                    'symbol': symbol,
                    'recommended_model': 'random_forest',  # Default fallback
                    'confidence_score': 0.5,
                    'evaluation_date': None,
                    'rationale': ['No recent evaluation available - using default RandomForest'],
                    'recommendation': 'Run advanced model evaluation to get optimal model selection'
                }
                
        except Exception as e:
            logger.error(f"Error getting model recommendation for {symbol}: {e}")
            return {
                'error': 'recommendation_failed',
                'message': str(e),
                'symbol': symbol
            }

    def get_cache_status(self) -> Dict[str, Any]:
        """Get current model cache status"""
        return {
            'cached_models': len(self.model_cache),
            'cached_symbols': list(set(key.split('_')[0] for key in self.model_cache.keys())),
            'cache_keys': list(self.model_cache.keys()),
            'memory_usage_estimate': f"{len(self.model_cache) * 50}MB"  # Rough estimate
        }

    async def analyze_feature_importance(self, symbol: str, period: str = '2y',
                                       method: str = 'composite') -> Dict[str, Any]:
        """
        Analyze feature importance for a symbol using advanced feature selection.
        
        Args:
            symbol: Stock symbol to analyze
            period: Historical data period
            method: Selection method ('composite', 'shap', 'rfe', 'correlation')
            
        Returns:
            Feature importance analysis results
        """
        try:
            logger.info(f"Analyzing feature importance for {symbol} using {method}")
            
            # Get training data with full feature set
            data = await self.data_pipeline.prepare_data_for_analysis(symbol, period, add_features=True)
            
            if data is None or len(data) < 100:
                return {
                    'error': f'Insufficient data for feature analysis of {symbol}',
                    'symbol': symbol,
                    'required_samples': 100,
                    'available_samples': len(data) if data is not None else 0
                }
            
            # Prepare features and target
            feature_columns = [col for col in data.columns if col != 'close']
            if not feature_columns:
                return {
                    'error': 'No feature columns found in the data',
                    'symbol': symbol
                }
            
            X = data[feature_columns]
            y = data['close'].pct_change().dropna()
            
            # Align X and y
            X = X.iloc[1:]  # Remove first row to align with y
            
            if len(X) != len(y):
                min_len = min(len(X), len(y))
                X = X.iloc[:min_len]
                y = y.iloc[:min_len]
            
            # Perform feature selection analysis
            results = await self.feature_selector.select_features(
                X, y, method=method, cv_folds=3
            )
            
            # Format response
            return {
                'symbol': symbol,
                'method': method,
                'analysis_timestamp': datetime.now().isoformat(),
                'original_features': results.original_feature_count,
                'selected_features': results.selected_feature_count,
                'reduction_ratio': results.removed_feature_count / results.original_feature_count,
                'performance_metrics': results.performance_metrics,
                'execution_time': results.execution_time,
                'top_features': [
                    {
                        'name': fi.feature_name,
                        'composite_score': fi.composite_score,
                        'shap_importance': fi.shap_importance,
                        'model_importance': fi.model_importance,
                        'rfe_rank': fi.rfe_rank,
                        'selected': fi.selected
                    }
                    for fi in sorted(results.feature_importances, 
                                   key=lambda x: x.composite_score, reverse=True)[:20]
                ],
                'selected_feature_names': results.selected_features,
                'removed_feature_names': results.removed_features,
                'selection_params': results.selection_params
            }
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance for {symbol}: {e}")
            return {
                'error': f'Feature importance analysis failed: {str(e)}',
                'symbol': symbol
            }

    async def optimize_feature_set(self, symbol: str, period: str = '2y',
                                 method: str = 'composite',
                                 target_reduction: float = 0.5) -> Dict[str, Any]:
        """
        Optimize feature set by automatically selecting the best features.
        
        Args:
            symbol: Stock symbol to optimize
            period: Historical data period
            method: Selection method ('composite', 'shap', 'rfe', 'correlation') 
            target_reduction: Target feature reduction ratio (0.0-1.0)
            
        Returns:
            Feature optimization results with performance comparison
        """
        try:
            logger.info(f"Optimizing feature set for {symbol}")
            
            # Get training data
            data = await self.data_pipeline.prepare_data_for_analysis(symbol, period, add_features=True)
            
            if data is None or len(data) < 100:
                return {
                    'error': f'Insufficient data for feature optimization of {symbol}',
                    'symbol': symbol
                }
            
            # Prepare features and target
            feature_columns = [col for col in data.columns if col != 'close']
            X = data[feature_columns]
            y = data['close'].pct_change().dropna()
            X = X.iloc[1:]  # Align with y
            
            if len(X) != len(y):
                min_len = min(len(X), len(y))
                X = X.iloc[:min_len]
                y = y.iloc[:min_len]
            
            # Calculate target number of features
            original_count = X.shape[1]
            target_count = max(5, int(original_count * (1 - target_reduction)))
            
            # Update feature selector with target
            self.feature_selector.max_features = target_count
            
            # Perform optimization
            results = await self.feature_selector.select_features(
                X, y, method=method, cv_folds=5
            )
            
            # Save optimization artifacts
            artifacts_path = await self.feature_selector.save_selection_artifacts(results)
            
            # Log feature selection experiment to MLflow
            try:
                feature_experiment_config = ExperimentConfig(
                    experiment_name=f"feature_selection_{symbol}",
                    run_name=f"{symbol}_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    tags={"symbol": symbol, "method": method, "optimization": "true"}
                )
                
                feature_metrics = ModelMetrics(
                    mae=None,
                    rmse=None,
                    r2=results.performance_metrics.get('selected_features_r2', 0.0),
                    sharpe_ratio=None,
                    hit_rate=None,
                    max_drawdown=None,
                    total_return=None,
                    volatility=None,
                    information_ratio=None,
                    calmar_ratio=None,
                    training_time=results.execution_time,
                    prediction_time=None
                )
                
                await self.mlflow_tracker.log_complete_experiment(
                    experiment_config=feature_experiment_config,
                    model=None,  # No model to save for feature selection
                    parameters={
                        "symbol": symbol,
                        "method": method,
                        "target_reduction": target_reduction,
                        "original_features": results.original_feature_count,
                        "target_features": target_count,
                        "period": period
                    },
                    metrics=feature_metrics,
                    artifacts={
                        "feature_rankings": {fi.feature_name: fi.composite_score for fi in results.feature_importances},
                        "selected_features": results.selected_features,
                        "removed_features": results.removed_features,
                        "performance_metrics": results.performance_metrics
                    },
                    model_name=f"{symbol}_feature_selection"
                )
                
            except Exception as mlflow_error:
                logger.warning(f"MLflow logging failed for feature selection: {mlflow_error}")
            
            return {
                'symbol': symbol,
                'optimization_timestamp': datetime.now().isoformat(),
                'method': method,
                'target_reduction': target_reduction,
                'achieved_reduction': results.removed_feature_count / results.original_feature_count,
                'original_features': results.original_feature_count,
                'optimized_features': results.selected_feature_count,
                'performance_improvement': results.performance_metrics.get('r2_difference', 0.0),
                'complexity_reduction': results.performance_metrics.get('complexity_reduction', 0.0),
                'execution_time': results.execution_time,
                'selected_features': results.selected_features,
                'artifacts_path': str(artifacts_path),
                'optimization_summary': {
                    'features_removed': results.removed_feature_count,
                    'performance_retained': results.performance_metrics.get('selected_features_r2', 0.0),
                    'baseline_performance': results.performance_metrics.get('all_features_r2', 0.0),
                    'efficiency_gain': f"{results.performance_metrics.get('complexity_reduction', 0.0):.1%}"
                }
            }
            
        except Exception as e:
            logger.error(f"Error optimizing feature set for {symbol}: {e}")
            return {
                'error': f'Feature optimization failed: {str(e)}',
                'symbol': symbol
            }

    async def batch_feature_analysis(self, symbols: List[str], period: str = '2y',
                                   method: str = 'composite') -> Dict[str, Any]:
        """
        Perform batch feature analysis across multiple symbols.
        
        Args:
            symbols: List of symbols to analyze
            period: Historical data period
            method: Selection method
            
        Returns:
            Batch analysis results with aggregated insights
        """
        logger.info(f"Starting batch feature analysis for {len(symbols)} symbols")
        
        results = {}
        failed_symbols = []
        successful_analyses = 0
        
        # Process each symbol
        for symbol in symbols:
            try:
                analysis = await self.analyze_feature_importance(symbol, period, method)
                if 'error' not in analysis:
                    results[symbol] = analysis
                    successful_analyses += 1
                else:
                    failed_symbols.append(symbol)
                    logger.warning(f"Feature analysis failed for {symbol}: {analysis.get('error')}")
                    
            except Exception as e:
                failed_symbols.append(symbol)
                logger.error(f"Unexpected error analyzing {symbol}: {e}")
        
        # Calculate aggregate statistics
        if successful_analyses > 0:
            reduction_ratios = [r['reduction_ratio'] for r in results.values()]
            performance_improvements = [
                r['performance_metrics'].get('r2_difference', 0.0) 
                for r in results.values()
            ]
            
            # Find most important features across all symbols
            feature_importance_counts = {}
            for analysis in results.values():
                for feature_info in analysis['top_features'][:10]:
                    feature_name = feature_info['name']
                    feature_importance_counts[feature_name] = feature_importance_counts.get(feature_name, 0) + 1
            
            most_important_features = sorted(
                feature_importance_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:15]
        else:
            reduction_ratios = []
            performance_improvements = []
            most_important_features = []
        
        return {
            'batch_analysis_timestamp': datetime.now().isoformat(),
            'method': method,
            'symbols_analyzed': len(symbols),
            'successful_analyses': successful_analyses,
            'failed_symbols': failed_symbols,
            'aggregate_statistics': {
                'avg_reduction_ratio': np.mean(reduction_ratios) if reduction_ratios else 0.0,
                'avg_performance_improvement': np.mean(performance_improvements) if performance_improvements else 0.0,
                'best_performing_symbol': max(results.keys(), 
                    key=lambda s: results[s]['performance_metrics'].get('r2_difference', -999)) if results else None,
                'most_efficient_symbol': max(results.keys(),
                    key=lambda s: results[s]['performance_metrics'].get('complexity_reduction', -999)) if results else None
            },
            'most_important_features': [
                {'feature': name, 'frequency': count, 'percentage': count/successful_analyses*100}
                for name, count in most_important_features
            ],
            'individual_results': results
        }

    # Model Deployment Pipeline Methods

    async def deploy_model_to_staging(self, model_name: str, version: str,
                                     strategy: str = "blue_green") -> Dict[str, Any]:
        """
        Deploy model to staging environment for testing.
        
        Args:
            model_name: Name of the model to deploy
            version: Version of the model to deploy
            strategy: Deployment strategy (blue_green, canary, immediate)
            
        Returns:
            Deployment result with status and endpoint information
        """
        try:
            logger.info(f"Deploying {model_name} v{version} to staging")
            
            # Create deployment configuration
            config = DeploymentConfig(
                model_name=model_name,
                version=version,
                stage=DeploymentStage.STAGING,
                strategy=DeploymentStrategy(strategy.lower()) if strategy else DeploymentStrategy.BLUE_GREEN
            )
            
            # Deploy to staging
            result = await self.deployment_pipeline.deploy_to_staging(model_name, version, config)
            
            # Convert to serializable format
            return {
                'deployment_id': result.deployment_id,
                'model_name': result.model_name,
                'version': result.version,
                'stage': result.stage.value,
                'strategy': result.strategy.value,
                'status': result.status,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'duration_seconds': result.duration_seconds,
                'endpoints': [
                    {
                        'endpoint_id': ep.endpoint_id,
                        'url': ep.url,
                        'port': ep.port,
                        'health_url': ep.health_url,
                        'health_status': ep.health_status.value,
                        'traffic_weight': ep.traffic_weight
                    }
                    for ep in (result.endpoints or [])
                ],
                'error_message': result.error_message
            }
            
        except Exception as e:
            logger.error(f"Error deploying {model_name} v{version} to staging: {e}")
            return {
                'error': f'Staging deployment failed: {str(e)}',
                'model_name': model_name,
                'version': version,
                'stage': 'staging'
            }

    async def promote_model_to_production(self, model_name: str, version: str,
                                         strategy: str = "blue_green",
                                         promotion_criteria: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Promote model from staging to production with validation.
        
        Args:
            model_name: Name of the model to promote
            version: Version of the model to promote
            strategy: Deployment strategy (blue_green, canary, immediate)
            promotion_criteria: Custom promotion criteria
            
        Returns:
            Promotion result with validation status
        """
        try:
            logger.info(f"Promoting {model_name} v{version} to production")
            
            # Create promotion criteria if provided
            criteria = None
            if promotion_criteria:
                criteria = PromotionCriteria(
                    min_sharpe_ratio=promotion_criteria.get('min_sharpe_ratio', 1.0),
                    min_hit_rate=promotion_criteria.get('min_hit_rate', 0.55),
                    max_drawdown=promotion_criteria.get('max_drawdown', 0.15),
                    min_r2_score=promotion_criteria.get('min_r2_score', 0.4),
                    min_samples=promotion_criteria.get('min_samples', 100),
                    min_days_in_staging=promotion_criteria.get('min_days_in_staging', 7),
                    max_latency_ms=promotion_criteria.get('max_latency_ms', 500.0),
                    min_uptime_pct=promotion_criteria.get('min_uptime_pct', 99.0)
                )
            
            # Create deployment configuration
            config = DeploymentConfig(
                model_name=model_name,
                version=version,
                stage=DeploymentStage.PRODUCTION,
                strategy=DeploymentStrategy(strategy.lower()) if strategy else DeploymentStrategy.BLUE_GREEN
            )
            
            # Promote to production
            result = await self.deployment_pipeline.promote_to_production(model_name, version, criteria, config)
            
            # Convert to serializable format
            return {
                'deployment_id': result.deployment_id,
                'model_name': result.model_name,
                'version': result.version,
                'stage': result.stage.value,
                'strategy': result.strategy.value,
                'status': result.status,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'duration_seconds': result.duration_seconds,
                'endpoints': [
                    {
                        'endpoint_id': ep.endpoint_id,
                        'url': ep.url,
                        'port': ep.port,
                        'health_url': ep.health_url,
                        'health_status': ep.health_status.value,
                        'traffic_weight': ep.traffic_weight
                    }
                    for ep in (result.endpoints or [])
                ],
                'rollback_performed': result.rollback_performed,
                'promotion_eligible': result.promotion_eligible,
                'error_message': result.error_message
            }
            
        except Exception as e:
            logger.error(f"Error promoting {model_name} v{version} to production: {e}")
            return {
                'error': f'Production promotion failed: {str(e)}',
                'model_name': model_name,
                'version': version,
                'stage': 'production'
            }

    async def create_shadow_deployment(self, model_name: str, version: str,
                                      traffic_pct: float = 0.1) -> Dict[str, Any]:
        """
        Create shadow deployment for testing without affecting production traffic.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            traffic_pct: Percentage of traffic to shadow
            
        Returns:
            Shadow deployment result
        """
        try:
            logger.info(f"Creating shadow deployment for {model_name} v{version}")
            
            # Create shadow deployment
            result = await self.deployment_pipeline.create_shadow_deployment(
                model_name, version, traffic_pct
            )
            
            # Convert to serializable format
            return {
                'deployment_id': result.deployment_id,
                'model_name': result.model_name,
                'version': result.version,
                'stage': result.stage.value,
                'status': result.status,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'duration_seconds': result.duration_seconds,
                'traffic_pct': traffic_pct,
                'endpoints': [
                    {
                        'endpoint_id': ep.endpoint_id,
                        'url': ep.url,
                        'port': ep.port,
                        'health_url': ep.health_url,
                        'health_status': ep.health_status.value,
                        'traffic_weight': ep.traffic_weight
                    }
                    for ep in (result.endpoints or [])
                ],
                'error_message': result.error_message
            }
            
        except Exception as e:
            logger.error(f"Error creating shadow deployment for {model_name} v{version}: {e}")
            return {
                'error': f'Shadow deployment failed: {str(e)}',
                'model_name': model_name,
                'version': version,
                'stage': 'shadow'
            }

    async def rollback_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """
        Rollback a deployment to previous version.
        
        Args:
            deployment_id: ID of deployment to rollback
            
        Returns:
            Rollback result
        """
        try:
            logger.info(f"Rolling back deployment {deployment_id}")
            
            result = await self.deployment_pipeline.rollback_deployment(deployment_id)
            
            return {
                'rollback_id': deployment_id,
                'success': result.get('success', False),
                'rollback_time': result.get('rollback_time'),
                'error': result.get('error')
            }
            
        except Exception as e:
            logger.error(f"Error rolling back deployment {deployment_id}: {e}")
            return {
                'error': f'Rollback failed: {str(e)}',
                'deployment_id': deployment_id
            }

    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get status of a deployment.
        
        Args:
            deployment_id: ID of deployment to check
            
        Returns:
            Deployment status information
        """
        try:
            result = await self.deployment_pipeline.get_deployment_status(deployment_id)
            
            if result is None:
                return {
                    'error': f'Deployment {deployment_id} not found',
                    'deployment_id': deployment_id
                }
            
            return {
                'deployment_id': result.deployment_id,
                'model_name': result.model_name,
                'version': result.version,
                'stage': result.stage.value,
                'strategy': result.strategy.value,
                'status': result.status,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'duration_seconds': result.duration_seconds,
                'rollback_performed': result.rollback_performed,
                'promotion_eligible': result.promotion_eligible,
                'error_message': result.error_message
            }
            
        except Exception as e:
            logger.error(f"Error getting deployment status for {deployment_id}: {e}")
            return {
                'error': f'Status check failed: {str(e)}',
                'deployment_id': deployment_id
            }

    async def list_active_deployments(self) -> Dict[str, Any]:
        """
        List all active deployments.
        
        Returns:
            List of active deployment information
        """
        try:
            deployments = await self.deployment_pipeline.list_active_deployments()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_deployments': len(deployments),
                'deployments': [
                    {
                        'deployment_id': dep.deployment_id,
                        'model_name': dep.model_name,
                        'version': dep.version,
                        'stage': dep.stage.value,
                        'strategy': dep.strategy.value,
                        'status': dep.status,
                        'start_time': dep.start_time.isoformat(),
                        'duration_seconds': dep.duration_seconds,
                        'rollback_performed': dep.rollback_performed
                    }
                    for dep in deployments
                ]
            }
            
        except Exception as e:
            logger.error(f"Error listing active deployments: {e}")
            return {
                'error': f'Failed to list deployments: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }

    async def get_endpoint_health(self, stage: str = "all") -> Dict[str, Any]:
        """
        Get health status of endpoints for a stage or all stages.
        
        Args:
            stage: Stage to check (staging, production, shadow, all)
            
        Returns:
            Health status information
        """
        try:
            if stage.lower() == "all":
                health_results = await self.deployment_pipeline.health_check_all_endpoints()
                return health_results
            
            else:
                stage_enum = DeploymentStage(stage.capitalize())
                endpoints = await self.deployment_pipeline.get_endpoint_status(stage_enum)
                
                return {
                    'stage': stage,
                    'timestamp': datetime.now().isoformat(),
                    'endpoints': [
                        {
                            'endpoint_id': ep.endpoint_id,
                            'model_name': ep.model_name,
                            'version': ep.version,
                            'url': ep.url,
                            'health_status': ep.health_status.value,
                            'last_health_check': ep.last_health_check.isoformat() if ep.last_health_check else None,
                            'traffic_weight': ep.traffic_weight,
                            'deployment_time': ep.deployment_time.isoformat() if ep.deployment_time else None
                        }
                        for ep in endpoints
                    ]
                }
            
        except Exception as e:
            logger.error(f"Error getting endpoint health for stage {stage}: {e}")
            return {
                'error': f'Health check failed: {str(e)}',
                'stage': stage,
                'timestamp': datetime.now().isoformat()
            }

    # Model Drift Monitoring Methods

    async def store_model_baseline(self, model_name: str, symbol: str, 
                                  period: str = "2y") -> Dict[str, Any]:
        """
        Store baseline feature distributions for drift monitoring.
        
        Args:
            model_name: Name of the model
            symbol: Trading symbol
            period: Historical data period for baseline
            
        Returns:
            Baseline storage result
        """
        try:
            logger.info(f"Storing baseline data for {model_name}_{symbol}")
            
            # Get feature data
            data = await self.data_pipeline.prepare_data_for_analysis(
                symbol, period, add_features=True
            )
            
            if data is None or data.empty:
                return {
                    'error': f'No data available for {symbol}',
                    'model_name': model_name,
                    'symbol': symbol
                }
            
            # Store baseline
            success = await self.drift_monitor.store_baseline_data(
                model_name, symbol, data
            )
            
            if success:
                return {
                    'model_name': model_name,
                    'symbol': symbol,
                    'baseline_samples': len(data),
                    'baseline_features': len(data.columns),
                    'baseline_period': period,
                    'stored_at': datetime.now().isoformat(),
                    'status': 'success'
                }
            else:
                return {
                    'error': 'Failed to store baseline data',
                    'model_name': model_name,
                    'symbol': symbol,
                    'status': 'failed'
                }
                
        except Exception as e:
            logger.error(f"Error storing baseline for {model_name}_{symbol}: {e}")
            return {
                'error': f'Baseline storage failed: {str(e)}',
                'model_name': model_name,
                'symbol': symbol,
                'status': 'error'
            }

    async def analyze_model_drift(self, model_name: str, symbol: str,
                                 model_version: str = "latest",
                                 period: str = "30d") -> Dict[str, Any]:
        """
        Analyze model drift using PSI and KS tests.
        
        Args:
            model_name: Name of the model
            symbol: Trading symbol
            model_version: Model version
            period: Period for current data analysis
            
        Returns:
            Comprehensive drift analysis report
        """
        try:
            logger.info(f"Analyzing drift for {model_name}_{symbol}")
            
            # Get current feature data
            current_data = await self.data_pipeline.prepare_data_for_analysis(
                symbol, period, add_features=True
            )
            
            if current_data is None or current_data.empty:
                return {
                    'error': f'No current data available for {symbol}',
                    'model_name': model_name,
                    'symbol': symbol
                }
            
            # Perform drift analysis
            drift_report = await self.drift_monitor.analyze_model_drift(
                model_name, symbol, current_data, model_version
            )
            
            # Trigger retraining if critical drift detected
            retraining_workflow = None
            if drift_report.overall_drift_status == DriftStatus.CRITICAL_DRIFT:
                logger.warning(f"Critical drift detected for {model_name}_{symbol}, triggering retraining")
                retraining_workflow = await self.drift_monitor.trigger_retraining_workflow(
                    model_name, symbol, drift_report
                )
            
            # Convert to API response format
            response = drift_report.to_dict()
            if retraining_workflow:
                response['retraining_workflow'] = retraining_workflow
                
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing drift for {model_name}_{symbol}: {e}")
            return {
                'error': f'Drift analysis failed: {str(e)}',
                'model_name': model_name,
                'symbol': symbol,
                'analysis_timestamp': datetime.now().isoformat()
            }

    async def get_drift_monitoring_status(self) -> Dict[str, Any]:
        """
        Get overall drift monitoring status across all models.
        
        Returns:
            Drift monitoring system status
        """
        try:
            return await self.drift_monitor.get_drift_summary()
            
        except Exception as e:
            logger.error(f"Error getting drift monitoring status: {e}")
            return {
                'error': f'Status check failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }

    async def configure_drift_thresholds(self, thresholds: Dict[str, float]) -> Dict[str, Any]:
        """
        Configure drift detection thresholds.
        
        Args:
            thresholds: New threshold values
            
        Returns:
            Configuration result
        """
        try:
            success = self.drift_monitor.configure_thresholds(thresholds)
            
            if success:
                return {
                    'status': 'success',
                    'message': 'Drift thresholds updated successfully',
                    'new_thresholds': thresholds,
                    'updated_at': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'partial_success',
                    'message': 'Some thresholds could not be updated',
                    'attempted_thresholds': thresholds,
                    'updated_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error configuring drift thresholds: {e}")
            return {
                'error': f'Configuration failed: {str(e)}',
                'attempted_thresholds': thresholds,
                'status': 'error'
            }

    async def batch_drift_analysis(self, models: List[Dict[str, str]], 
                                  period: str = "30d") -> Dict[str, Any]:
        """
        Perform batch drift analysis across multiple models.
        
        Args:
            models: List of model dictionaries with 'model_name' and 'symbol' keys
            period: Analysis period for current data
            
        Returns:
            Batch drift analysis results
        """
        try:
            if len(models) > 10:
                return {
                    'error': 'Maximum 10 models allowed for batch drift analysis',
                    'models_requested': len(models)
                }
            
            logger.info(f"Starting batch drift analysis for {len(models)} models")
            
            results = {}
            alerts_triggered = 0
            critical_models = []
            
            for model_info in models:
                model_name = model_info.get('model_name')
                symbol = model_info.get('symbol')
                
                if not model_name or not symbol:
                    continue
                    
                try:
                    analysis = await self.analyze_model_drift(
                        model_name, symbol, "latest", period
                    )
                    
                    results[f"{model_name}_{symbol}"] = analysis
                    
                    if analysis.get('alert_triggered', False):
                        alerts_triggered += 1
                        
                    if analysis.get('overall_drift_status') == 'critical_drift':
                        critical_models.append(f"{model_name}_{symbol}")
                        
                except Exception as e:
                    logger.error(f"Drift analysis failed for {model_name}_{symbol}: {e}")
                    results[f"{model_name}_{symbol}"] = {
                        'error': str(e),
                        'status': 'analysis_failed'
                    }
            
            return {
                'batch_analysis_timestamp': datetime.now().isoformat(),
                'total_models': len(models),
                'successful_analyses': len([r for r in results.values() if 'error' not in r]),
                'alerts_triggered': alerts_triggered,
                'critical_models': critical_models,
                'summary': {
                    'stable_models': len([r for r in results.values() if r.get('overall_drift_status') == 'stable']),
                    'minor_drift_models': len([r for r in results.values() if r.get('overall_drift_status') == 'minor_drift']),
                    'significant_drift_models': len([r for r in results.values() if r.get('overall_drift_status') == 'significant_drift']),
                    'critical_drift_models': len(critical_models)
                },
                'individual_results': results
            }
            
        except Exception as e:
            logger.error(f"Error in batch drift analysis: {e}")
            return {
                'error': f'Batch analysis failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }

    # Signal-to-Execution Latency Tracking Methods

    async def generate_trading_signal(self, symbol: str, signal_type: str = "auto",
                                     model_confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Generate trading signal with latency tracking.
        
        Args:
            symbol: Trading symbol
            signal_type: Type of signal (buy, sell, auto)
            model_confidence_threshold: Minimum confidence for signal generation
            
        Returns:
            Trading signal with tracking ID
        """
        try:
            signal_start_time = datetime.now()
            await self.latency_tracker.log_latency_checkpoint(
                "signal_gen", LatencyStage.SIGNAL_GENERATED, "analysis_service"
            )
            
            # Get current forecast
            forecast = await self.generate_forecast(symbol)
            
            if 'error' in forecast:
                return {
                    'error': f'Forecast generation failed: {forecast["error"]}',
                    'symbol': symbol
                }
            
            # Extract signal information
            predictions = forecast.get('predictions', [])
            if not predictions:
                return {
                    'error': 'No predictions available for signal generation',
                    'symbol': symbol
                }
            
            # Determine signal based on forecast
            current_price = forecast.get('current_price', 0)
            next_prediction = predictions[0] if predictions else current_price
            confidence = forecast.get('ensemble_confidence', 0.5)
            
            # Generate signal
            if signal_type.lower() == "auto":
                if next_prediction > current_price * 1.01 and confidence > model_confidence_threshold:
                    signal_type_enum = SignalType.BUY
                    expected_return = (next_prediction - current_price) / current_price
                elif next_prediction < current_price * 0.99 and confidence > model_confidence_threshold:
                    signal_type_enum = SignalType.SELL
                    expected_return = (current_price - next_prediction) / current_price
                else:
                    signal_type_enum = SignalType.HOLD
                    expected_return = 0.0
            else:
                signal_type_enum = SignalType(signal_type.lower())
                expected_return = abs(next_prediction - current_price) / current_price
            
            # Create signal event
            signal = SignalEvent(
                signal_id="",  # Will be generated
                symbol=symbol,
                signal_type=signal_type_enum,
                signal_strength=confidence,
                expected_return=expected_return,
                target_price=next_prediction,
                metadata={
                    'signal_price': current_price,
                    'forecast_confidence': confidence,
                    'prediction_horizon': forecast.get('prediction_horizon', 5),
                    'model_type': forecast.get('model_type', 'ensemble'),
                    'generation_time': signal_start_time.isoformat()
                }
            )
            
            # Log signal generation
            signal_id = await self.latency_tracker.log_signal_generation(signal)
            
            # Log validation checkpoint
            await self.latency_tracker.log_latency_checkpoint(
                signal_id, LatencyStage.SIGNAL_VALIDATED, "analysis_service"
            )
            
            return {
                'signal_id': signal_id,
                'symbol': symbol,
                'signal_type': signal_type_enum.value,
                'signal_strength': confidence,
                'expected_return': expected_return,
                'current_price': current_price,
                'target_price': next_prediction,
                'timestamp': signal_start_time.isoformat(),
                'metadata': signal.metadata
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signal for {symbol}: {e}")
            return {
                'error': f'Signal generation failed: {str(e)}',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }

    async def track_signal_transmission(self, signal_id: str) -> Dict[str, Any]:
        """
        Track signal transmission to strategy service.
        
        Args:
            signal_id: Signal identifier
            
        Returns:
            Transmission tracking result
        """
        try:
            await self.latency_tracker.log_latency_checkpoint(
                signal_id, LatencyStage.SIGNAL_TRANSMITTED, "communication_layer"
            )
            
            return {
                'signal_id': signal_id,
                'transmission_logged': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error tracking signal transmission: {e}")
            return {
                'error': f'Transmission tracking failed: {str(e)}',
                'signal_id': signal_id
            }

    async def calculate_signal_alpha_decay(self, signal_id: str, current_price: float,
                                          holding_period_hours: float = 24.0) -> Dict[str, Any]:
        """
        Calculate alpha decay for a signal.
        
        Args:
            signal_id: Signal identifier
            current_price: Current market price
            holding_period_hours: Holding period for analysis
            
        Returns:
            Alpha decay analysis
        """
        try:
            alpha_decay = await self.latency_tracker.calculate_alpha_decay(
                signal_id, current_price, holding_period_hours
            )
            
            if alpha_decay is None:
                return {
                    'error': 'Insufficient data for alpha decay calculation',
                    'signal_id': signal_id
                }
            
            return alpha_decay.to_dict()
            
        except Exception as e:
            logger.error(f"Error calculating alpha decay: {e}")
            return {
                'error': f'Alpha decay calculation failed: {str(e)}',
                'signal_id': signal_id
            }

    async def get_latency_performance(self, lookback_minutes: int = 60) -> Dict[str, Any]:
        """
        Get latency performance analysis.
        
        Args:
            lookback_minutes: Analysis window
            
        Returns:
            Latency performance metrics
        """
        try:
            return await self.latency_tracker.get_latency_analysis(lookback_minutes)
            
        except Exception as e:
            logger.error(f"Error getting latency performance: {e}")
            return {
                'error': f'Latency analysis failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }

    async def get_alpha_decay_performance(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Get alpha decay performance analysis.
        
        Args:
            lookback_hours: Analysis window
            
        Returns:
            Alpha decay performance metrics
        """
        try:
            return await self.latency_tracker.get_alpha_decay_analysis(lookback_hours)
            
        except Exception as e:
            logger.error(f"Error getting alpha decay performance: {e}")
            return {
                'error': f'Alpha decay analysis failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }

    async def identify_execution_bottlenecks(self) -> Dict[str, Any]:
        """
        Identify execution bottlenecks in the signal pipeline.
        
        Returns:
            Bottleneck analysis with recommendations
        """
        try:
            return await self.latency_tracker.identify_bottlenecks()
            
        except Exception as e:
            logger.error(f"Error identifying bottlenecks: {e}")
            return {
                'error': f'Bottleneck analysis failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def _train_stacking_model(self, data: pd.DataFrame, prediction_horizon: int) -> Optional[EnsembleStacking]:
        """Train ensemble stacking model with formal blending"""
        try:
            logger.info("Training ensemble stacking model with formal blending")
            
            # Prepare features and target
            features = [col for col in data.columns if col not in ['close', 'date', 'timestamp']]
            if not features:
                logger.error("No features available for stacking model")
                return None
            
            X = data[features].values
            y = data['close'].values
            
            # Handle missing values
            if np.isnan(X).any() or np.isnan(y).any():
                # Forward fill missing values
                X = pd.DataFrame(X).fillna(method='ffill').fillna(method='bfill').values
                y = pd.Series(y).fillna(method='ffill').fillna(method='bfill').values
            
            # Create ensemble stacking model
            stacking_model = EnsembleStacking(self.stacking_config)
            
            # Add base models
            from sklearn.ensemble import RandomForestRegressor
            
            # Random Forest base model
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            rf_config = BaseModelConfig(
                name="random_forest",
                model_type="tree_based",
                enabled=True,
                weight=1.0,
                hyperparameters={
                    'n_estimators': 100,
                    'max_depth': 15,
                    'min_samples_split': 5
                }
            )
            stacking_model.add_base_model("random_forest", rf_model, rf_config)
            
            # LSTM base model (if TensorFlow is available)
            try:
                from ..models.ensemble_stacking import TF_AVAILABLE
                if TF_AVAILABLE:
                    lstm_model = LSTMBaseModel(
                        sequence_length=min(30, len(X) // 4),
                        units=50,
                        dropout=0.2,
                        epochs=30,
                        batch_size=min(32, len(X) // 10),
                        learning_rate=0.001
                    )
                    lstm_config = BaseModelConfig(
                        name="lstm",
                        model_type="deep_learning",
                        enabled=True,
                        weight=1.0,
                        hyperparameters={
                            'sequence_length': min(30, len(X) // 4),
                            'units': 50,
                            'epochs': 30
                        }
                    )
                    stacking_model.add_base_model("lstm", lstm_model, lstm_config)
                    
                    # GRU base model
                    gru_model = GRUBaseModel(
                        sequence_length=min(30, len(X) // 4),
                        units=50,
                        dropout=0.2,
                        epochs=30,
                        batch_size=min(32, len(X) // 10),
                        learning_rate=0.001
                    )
                    gru_config = BaseModelConfig(
                        name="gru",
                        model_type="deep_learning",
                        enabled=True,
                        weight=1.0,
                        hyperparameters={
                            'sequence_length': min(30, len(X) // 4),
                            'units': 50,
                            'epochs': 30
                        }
                    )
                    stacking_model.add_base_model("gru", gru_model, gru_config)
                else:
                    logger.warning("TensorFlow not available, using only Random Forest for stacking")
            except Exception as e:
                logger.warning(f"Deep learning models not available: {str(e)}")
            
            # Train the stacking model
            logger.info("Training stacking model with out-of-fold predictions")
            await asyncio.get_event_loop().run_in_executor(
                None, stacking_model.fit, X, y, features
            )
            
            logger.info("Ensemble stacking model trained successfully")
            
            # Log performance report
            try:
                performance_report = stacking_model.get_performance_report()
                logger.info(f"Stacking model performance: {performance_report['blender_score']}")
                logger.info(f"Model weights: {performance_report['model_weights']}")
            except Exception as e:
                logger.warning(f"Could not generate performance report: {str(e)}")
            
            return stacking_model
            
        except Exception as e:
            logger.error(f"Error training stacking model: {str(e)}")
            import traceback
            logger.error(f"Stacking model training traceback: {traceback.format_exc()}")
            return None
    
    async def generate_stacking_forecast(self, symbol: str, prediction_horizon: int = 5, 
                                       period: str = '2y', retrain: bool = False,
                                       blender_type: str = 'ridge') -> Dict[str, Any]:
        """
        Generate forecast using ensemble stacking with formal blending
        
        Args:
            symbol: Stock symbol to forecast
            prediction_horizon: Number of days to forecast
            period: Historical data period for training
            retrain: Whether to retrain the model
            blender_type: Type of blender ('ridge', 'linear', 'lasso', 'elastic_net', 'random_forest')
            
        Returns:
            Forecast results with stacking predictions and model importance
        """
        try:
            logger.info(f"Generating stacking forecast for {symbol}")
            
            # Update stacking configuration if different blender requested
            if blender_type.upper() in BlenderType.__members__:
                self.stacking_config.blender_type = BlenderType(blender_type.upper())
            
            # Get training data
            data = await self.data_pipeline.prepare_data_for_analysis(symbol, period, add_features=True)
            
            if data is None or len(data) < 200:
                return {
                    'error': f'Insufficient data for stacking forecast {symbol}',
                    'symbol': symbol,
                    'required_samples': 200,
                    'available_samples': len(data) if data is not None else 0
                }
            
            # Check cache for trained stacking model
            stacking_key = f"{symbol}_stacking_{prediction_horizon}_{blender_type}"
            
            if not retrain and stacking_key in self.stacking_cache:
                logger.info(f"Using cached stacking model for {symbol}")
                stacking_model = self.stacking_cache[stacking_key]
            else:
                # Train new stacking model
                stacking_model = await self._train_stacking_model(data, prediction_horizon)
                
                if stacking_model is None:
                    return {
                        'error': f'Failed to train stacking model for {symbol}',
                        'symbol': symbol
                    }
                
                # Cache the trained model
                self.stacking_cache[stacking_key] = stacking_model
            
            # Prepare recent data for prediction
            recent_data = data.tail(100)  # Use last 100 days for prediction
            features = [col for col in recent_data.columns if col not in ['close', 'date', 'timestamp']]
            
            if not features:
                return {
                    'error': f'No features available for prediction',
                    'symbol': symbol
                }
            
            X_recent = recent_data[features].values
            
            # Handle missing values
            if np.isnan(X_recent).any():
                X_recent = pd.DataFrame(X_recent).fillna(method='ffill').fillna(method='bfill').values
            
            # Generate predictions
            logger.info(f"Generating stacking predictions for {symbol}")
            predictions = await asyncio.get_event_loop().run_in_executor(
                None, stacking_model.predict, X_recent
            )
            
            if len(predictions) == 0:
                return {
                    'error': f'No predictions generated for {symbol}',
                    'symbol': symbol
                }
            
            # Get latest price for reference
            current_price = recent_data['close'].iloc[-1]
            
            # Format predictions for multiple horizons
            prediction_list = predictions[-prediction_horizon:].tolist() if len(predictions) >= prediction_horizon else predictions.tolist()
            
            # Generate confidence intervals (simple approach using recent volatility)
            recent_returns = recent_data['close'].pct_change().dropna()
            volatility = recent_returns.std()
            
            confidence_intervals = []
            for i, pred in enumerate(prediction_list):
                interval_width = volatility * np.sqrt(i + 1) * 1.96  # 95% confidence interval
                confidence_intervals.append({
                    'lower': pred * (1 - interval_width),
                    'upper': pred * (1 + interval_width)
                })
            
            # Get model importance and performance
            try:
                performance_report = stacking_model.get_performance_report()
                model_weights = performance_report['model_weights']
                blender_score = performance_report['blender_score']
                feature_importance = performance_report.get('feature_importance', {})
            except Exception as e:
                logger.warning(f"Could not get performance report: {str(e)}")
                model_weights = {}
                blender_score = {}
                feature_importance = {}
            
            # Calculate trend and confidence
            price_changes = np.diff(prediction_list)
            trend = "bullish" if np.mean(price_changes) > 0 else "bearish"
            trend_strength = abs(np.mean(price_changes) / current_price)
            
            if trend_strength > 0.02:
                confidence = "high"
            elif trend_strength > 0.01:
                confidence = "medium"
            else:
                confidence = "low"
            
            return {
                'symbol': symbol,
                'model_type': 'stacking',
                'blender_type': self.stacking_config.blender_type.value,
                'predictions': prediction_list,
                'confidence_intervals': confidence_intervals,
                'current_price': current_price,
                'predicted_change': {
                    'absolute': prediction_list[-1] - current_price,
                    'percentage': ((prediction_list[-1] - current_price) / current_price) * 100
                },
                'trend': {
                    'direction': trend,
                    'confidence': confidence,
                    'strength': trend_strength
                },
                'model_performance': {
                    'blender_score': blender_score,
                    'model_weights': model_weights,
                    'feature_importance': feature_importance
                },
                'metadata': {
                    'prediction_horizon': prediction_horizon,
                    'training_period': period,
                    'data_points': len(data),
                    'features_used': len(features),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating stacking forecast for {symbol}: {str(e)}")
            import traceback
            logger.error(f"Stacking forecast traceback: {traceback.format_exc()}")
            return {
                'error': f'Stacking forecast failed: {str(e)}',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
    
    async def compare_ensemble_models(self, symbol: str, prediction_horizon: int = 5, 
                                    period: str = '2y') -> Dict[str, Any]:
        """
        Compare traditional ensemble vs stacking ensemble performance
        
        Args:
            symbol: Stock symbol to compare
            prediction_horizon: Number of days to forecast
            period: Historical data period for comparison
            
        Returns:
            Comparison results between ensemble models
        """
        try:
            logger.info(f"Comparing ensemble models for {symbol}")
            
            # Generate forecasts from both models
            traditional_forecast = await self.generate_forecast(
                symbol, 'ensemble', prediction_horizon, period, retrain=True
            )
            
            stacking_forecast = await self.generate_stacking_forecast(
                symbol, prediction_horizon, period, retrain=True
            )
            
            # Get historical data for backtesting
            data = await self.data_pipeline.prepare_data_for_analysis(symbol, period, add_features=True)
            
            if data is None or len(data) < 200:
                return {
                    'error': f'Insufficient data for model comparison {symbol}',
                    'symbol': symbol
                }
            
            # Simple performance comparison based on trend consistency
            traditional_predictions = traditional_forecast.get('predictions', [])
            stacking_predictions = stacking_forecast.get('predictions', [])
            
            if not traditional_predictions or not stacking_predictions:
                return {
                    'error': 'Failed to generate predictions for comparison',
                    'symbol': symbol
                }
            
            # Calculate metrics
            traditional_trend = "bullish" if traditional_predictions[-1] > traditional_predictions[0] else "bearish"
            stacking_trend = "bullish" if stacking_predictions[-1] > stacking_predictions[0] else "bearish"
            
            traditional_volatility = np.std(traditional_predictions)
            stacking_volatility = np.std(stacking_predictions)
            
            return {
                'symbol': symbol,
                'comparison_period': period,
                'prediction_horizon': prediction_horizon,
                'traditional_ensemble': {
                    'predictions': traditional_predictions,
                    'trend': traditional_trend,
                    'volatility': traditional_volatility,
                    'model_type': 'traditional_ensemble'
                },
                'stacking_ensemble': {
                    'predictions': stacking_predictions,
                    'trend': stacking_trend,
                    'volatility': stacking_volatility,
                    'model_type': 'stacking_ensemble',
                    'blender_type': stacking_forecast.get('blender_type', 'ridge'),
                    'model_weights': stacking_forecast.get('model_performance', {}).get('model_weights', {})
                },
                'comparison_metrics': {
                    'trend_agreement': traditional_trend == stacking_trend,
                    'volatility_difference': abs(traditional_volatility - stacking_volatility),
                    'prediction_difference': abs(traditional_predictions[-1] - stacking_predictions[-1]),
                    'complexity_advantage': 'stacking' if len(stacking_forecast.get('model_performance', {}).get('model_weights', {})) > 2 else 'traditional'
                },
                'recommendation': {
                    'preferred_model': 'stacking' if stacking_volatility < traditional_volatility else 'traditional',
                    'reason': 'Lower prediction volatility' if stacking_volatility < traditional_volatility else 'Simpler model with similar performance'
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing ensemble models for {symbol}: {str(e)}")
            return {
                'error': f'Model comparison failed: {str(e)}',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_stacking_cache_status(self) -> Dict[str, Any]:
        """Get status of stacking model cache"""
        return {
            'cached_models': list(self.stacking_cache.keys()),
            'cache_size': len(self.stacking_cache),
            'stacking_config': {
                'blender_type': self.stacking_config.blender_type.value,
                'n_folds': self.stacking_config.n_folds,
                'validation_strategy': self.stacking_config.validation_strategy.value,
                'ridge_alpha': self.stacking_config.ridge_alpha,
                'scale_features': self.stacking_config.scale_features
            }
        }
    
    def clear_stacking_cache(self, symbol: str = None):
        """Clear stacking model cache"""
        if symbol:
            # Clear cache for specific symbol
            keys_to_remove = [key for key in self.stacking_cache.keys() if key.startswith(symbol)]
            for key in keys_to_remove:
                del self.stacking_cache[key]
            logger.info(f"Cleared {len(keys_to_remove)} stacking models for {symbol}")
        else:
            # Clear all cached models
            cache_size = len(self.stacking_cache)
            self.stacking_cache.clear()
            logger.info(f"Cleared all {cache_size} stacking models from cache")
    
    async def analyze_market_regime(self, symbol: str, period: str = '2y') -> Dict[str, Any]:
        """Analyze current market regime for a symbol"""
        try:
            logger.info(f"Analyzing market regime for {symbol}")
            
            # Get data
            data = await self.data_pipeline.prepare_data_for_analysis(symbol, period, add_features=True)
            
            if data is None or len(data) < 120:
                return {
                    'error': f'Insufficient data for regime analysis of {symbol}',
                    'symbol': symbol,
                    'required_samples': 120,
                    'available_samples': len(data) if data is not None else 0
                }
            
            # Perform regime analysis
            regime_results = self.regime_analyzer.analyze_regime_features(data)
            
            # Get recommended model weights
            current_regime = regime_results['current_state'].regime
            current_confidence = regime_results['current_state'].confidence
            
            recommended_weights = self.regime_analyzer.get_regime_based_model_weights(
                current_regime, current_confidence
            )
            
            # Format results
            return {
                'symbol': symbol,
                'current_regime': {
                    'regime_type': current_regime.value,
                    'confidence': current_confidence,
                    'volatility_percentile': regime_results['current_state'].volatility_percentile,
                    'trend_strength': regime_results['current_state'].trend_strength,
                    'atr_position': regime_results['current_state'].atr_position,
                    'duration_days': regime_results['current_state'].duration_days
                },
                'recommended_model_weights': recommended_weights,
                'regime_features': {
                    'atr_percentile': regime_results['atr_features']['atr_percentile'].iloc[-1] if len(regime_results['atr_features']) > 0 else None,
                    'volatility_rank': regime_results['volatility_features']['vol_rank'].iloc[-1] if len(regime_results['volatility_features']) > 0 else None,
                    'vol_term_slope': regime_results['volatility_features']['vol_term_slope'].iloc[-1] if len(regime_results['volatility_features']) > 0 else None,
                    'trend_strength_20d': regime_results['trend_features']['trend_strength_20d'].iloc[-1] if len(regime_results['trend_features']) > 0 else None
                },
                'regime_transitions': regime_results['transitions'][-5:] if regime_results['transitions'] else [],
                'hmm_analysis': regime_results.get('hmm_analysis', {}),
                'regime_statistics': regime_results['regime_statistics'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Regime analysis failed for {symbol}: {str(e)}")
            return {
                'error': f'Regime analysis failed: {str(e)}',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
    
    async def generate_regime_aware_forecast(self, symbol: str, prediction_horizon: int = 5,
                                           period: str = '2y', retrain: bool = False,
                                           weighting_mode: str = 'hybrid') -> Dict[str, Any]:
        """
        Generate forecast using regime-aware ensemble stacking
        
        Args:
            symbol: Stock symbol to forecast
            prediction_horizon: Number of days to forecast
            period: Historical data period for training
            retrain: Whether to retrain the model
            weighting_mode: Regime weighting mode ('fixed', 'adaptive', 'hybrid', 'confidence_weighted')
        """
        try:
            logger.info(f"Generating regime-aware forecast for {symbol}")
            
            # Validate weighting mode
            try:
                weighting_enum = RegimeWeightingMode(weighting_mode.upper())
                self.regime_stacking_config.weighting_mode = weighting_enum
            except ValueError:
                logger.warning(f"Invalid weighting mode {weighting_mode}, using HYBRID")
                weighting_enum = RegimeWeightingMode.HYBRID
            
            # Get training data
            data = await self.data_pipeline.prepare_data_for_analysis(symbol, period, add_features=True)
            
            if data is None or len(data) < 200:
                return {
                    'error': f'Insufficient data for regime-aware forecast {symbol}',
                    'symbol': symbol,
                    'required_samples': 200,
                    'available_samples': len(data) if data is not None else 0
                }
            
            # Check cache for trained regime-aware model
            regime_key = f"{symbol}_regime_aware_{prediction_horizon}_{weighting_mode}"
            
            if not retrain and regime_key in self.regime_aware_cache:
                logger.info(f"Using cached regime-aware model for {symbol}")
                regime_model = self.regime_aware_cache[regime_key]
            else:
                # Train new regime-aware model
                regime_model = await self._train_regime_aware_model(data, prediction_horizon)
                
                if regime_model is None:
                    return {
                        'error': f'Failed to train regime-aware model for {symbol}',
                        'symbol': symbol
                    }
                
                # Cache the trained model
                self.regime_aware_cache[regime_key] = regime_model
            
            # Prepare recent data for prediction
            recent_data = data.tail(100)
            features = [col for col in recent_data.columns if col not in ['close', 'date', 'timestamp']]
            
            if not features:
                return {
                    'error': f'No features available for prediction',
                    'symbol': symbol
                }
            
            X_recent = recent_data[features].values
            
            # Handle missing values
            if np.isnan(X_recent).any():
                X_recent = pd.DataFrame(X_recent).fillna(method='ffill').fillna(method='bfill').values
            
            # Generate predictions with regime awareness
            logger.info(f"Generating regime-aware predictions for {symbol}")
            predictions = await asyncio.get_event_loop().run_in_executor(
                None, regime_model.predict, X_recent, recent_data
            )
            
            if len(predictions) == 0:
                return {
                    'error': f'No predictions generated for {symbol}',
                    'symbol': symbol
                }
            
            # Get regime analysis
            regime_analysis = regime_model.get_regime_analysis()
            
            # Get latest price for reference
            current_price = recent_data['close'].iloc[-1]
            
            # Format predictions
            prediction_list = predictions[-prediction_horizon:].tolist() if len(predictions) >= prediction_horizon else predictions.tolist()
            
            # Calculate confidence intervals using regime volatility
            recent_returns = recent_data['close'].pct_change().dropna()
            volatility = recent_returns.std()
            
            confidence_intervals = []
            for i, pred in enumerate(prediction_list):
                interval_width = volatility * np.sqrt(i + 1) * 1.96
                confidence_intervals.append({
                    'lower': pred * (1 - interval_width),
                    'upper': pred * (1 + interval_width)
                })
            
            # Calculate trend and strength
            price_changes = np.diff(prediction_list)
            trend = "bullish" if np.mean(price_changes) > 0 else "bearish"
            trend_strength = abs(np.mean(price_changes) / current_price)
            
            confidence = "high" if trend_strength > 0.02 else "medium" if trend_strength > 0.01 else "low"
            
            return {
                'symbol': symbol,
                'model_type': 'regime_aware_stacking',
                'weighting_mode': weighting_enum.value,
                'predictions': prediction_list,
                'confidence_intervals': confidence_intervals,
                'current_price': current_price,
                'predicted_change': {
                    'absolute': prediction_list[-1] - current_price,
                    'percentage': ((prediction_list[-1] - current_price) / current_price) * 100
                },
                'trend': {
                    'direction': trend,
                    'confidence': confidence,
                    'strength': trend_strength
                },
                'regime_analysis': regime_analysis,
                'metadata': {
                    'prediction_horizon': prediction_horizon,
                    'training_period': period,
                    'data_points': len(data),
                    'features_used': len(features),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating regime-aware forecast for {symbol}: {str(e)}")
            import traceback
            logger.error(f"Regime-aware forecast traceback: {traceback.format_exc()}")
            return {
                'error': f'Regime-aware forecast failed: {str(e)}',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _train_regime_aware_model(self, data: pd.DataFrame, prediction_horizon: int) -> Optional[RegimeAwareStacking]:
        """Train regime-aware ensemble stacking model"""
        try:
            logger.info("Training regime-aware ensemble stacking model")
            
            # Prepare features and target
            features = [col for col in data.columns if col not in ['close', 'date', 'timestamp']]
            if not features:
                logger.error("No features available for regime-aware model")
                return None
            
            X = data[features].values
            y = data['close'].values
            
            # Handle missing values
            if np.isnan(X).any() or np.isnan(y).any():
                X = pd.DataFrame(X).fillna(method='ffill').fillna(method='bfill').values
                y = pd.Series(y).fillna(method='ffill').fillna(method='bfill').values
            
            # Create regime-aware stacking model
            regime_model = RegimeAwareStacking(self.regime_stacking_config)
            
            # Train the model
            logger.info("Training regime-aware model with feature enhancement")
            await asyncio.get_event_loop().run_in_executor(
                None, regime_model.fit, X, y, data, features
            )
            
            logger.info("Regime-aware ensemble stacking model trained successfully")
            
            # Log performance report
            try:
                performance_report = regime_model.get_performance_report()
                logger.info(f"Regime-aware model performance: {performance_report}")
            except Exception as e:
                logger.warning(f"Could not generate regime performance report: {str(e)}")
            
            return regime_model
            
        except Exception as e:
            logger.error(f"Error training regime-aware model: {str(e)}")
            import traceback
            logger.error(f"Regime-aware model training traceback: {traceback.format_exc()}")
            return None
    
    def get_regime_aware_cache_status(self) -> Dict[str, Any]:
        """Get status of regime-aware model cache"""
        return {
            'cached_models': list(self.regime_aware_cache.keys()),
            'cache_size': len(self.regime_aware_cache),
            'regime_stacking_config': {
                'weighting_mode': self.regime_stacking_config.weighting_mode.value,
                'confidence_threshold': self.regime_stacking_config.confidence_threshold,
                'performance_window': self.regime_stacking_config.performance_window,
                'min_regime_samples': self.regime_stacking_config.min_regime_samples
            }
        }
    
    def clear_regime_aware_cache(self, symbol: str = None):
        """Clear regime-aware model cache"""
        if symbol:
            keys_to_remove = [key for key in self.regime_aware_cache.keys() if key.startswith(symbol)]
            for key in keys_to_remove:
                del self.regime_aware_cache[key]
            logger.info(f"Cleared {len(keys_to_remove)} regime-aware models for {symbol}")
        else:
            cache_size = len(self.regime_aware_cache)
            self.regime_aware_cache.clear()
            logger.info(f"Cleared all {cache_size} regime-aware models from cache")
    
    async def validate_regime_detection(self, symbol: str, period: str = '2y', 
                                      validation_period: int = 60) -> Dict[str, Any]:
        """Validate regime detection accuracy across different market conditions"""
        try:
            logger.info(f"Validating regime detection for {symbol}")
            
            # Get data
            data = await self.data_pipeline.prepare_data_for_analysis(symbol, period, add_features=True)
            
            if data is None or len(data) < validation_period * 2:
                return {
                    'error': f'Insufficient data for regime validation of {symbol}',
                    'symbol': symbol,
                    'required_samples': validation_period * 2,
                    'available_samples': len(data) if data is not None else 0
                }
            
            # Perform validation
            validation_results = self.regime_analyzer.validate_regime_detection(data, validation_period)
            
            # Add symbol information
            validation_results['symbol'] = symbol
            validation_results['validation_period'] = validation_period
            validation_results['total_data_points'] = len(data)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Regime validation failed for {symbol}: {str(e)}")
            return {
                'error': f'Regime validation failed: {str(e)}',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }

    async def explain_model_predictions(self, symbol: str, explainer_type: str = 'shap_tree',
                                      sample_size: int = 100, period: str = '1y') -> Dict[str, Any]:
        """
        Generate explanations for model predictions using SHAP/LIME.
        
        Args:
            symbol: Stock symbol to analyze
            explainer_type: Type of explainer ('shap_tree', 'shap_linear', 'lime_tabular')
            sample_size: Number of samples to explain
            period: Period for training data
            
        Returns:
            Prediction explanations and feature attribution
        """
        try:
            logger.info(f"Explaining predictions for {symbol} using {explainer_type}")
            
            # Get best performing model for the symbol
            cache_key = f"{symbol}_best_model"
            if cache_key in self.model_cache:
                model = self.model_cache[cache_key]
            else:
                # Retrain to get best model
                await self.train_models(symbol, period=period)
                if cache_key not in self.model_cache:
                    return {
                        'error': f'No trained model available for {symbol}',
                        'symbol': symbol
                    }
                model = self.model_cache[cache_key]
            
            # Get training data
            data = await self.data_pipeline.prepare_data_for_analysis(symbol, period, add_features=True)
            if data is None or len(data) < sample_size:
                return {
                    'error': f'Insufficient data for explanation of {symbol}',
                    'symbol': symbol,
                    'required_samples': sample_size,
                    'available_samples': len(data) if data is not None else 0
                }
            
            # Prepare features
            feature_columns = [col for col in data.columns if col != 'close']
            X = data[feature_columns].fillna(0)
            
            # Sample data for explanation
            if len(X) > sample_size:
                X_sample = X.sample(n=sample_size, random_state=42)
            else:
                X_sample = X
            
            # Fit explainer if not already fitted
            await self.model_explainer.fit_explainers(model, X, explainer_types=[explainer_type])
            
            # Generate explanations
            explanations = []
            for idx, (index, row) in enumerate(X_sample.iterrows()):
                if idx >= 10:  # Limit to first 10 for performance
                    break
                    
                explanation = await self.model_explainer.explain_prediction(
                    row.values.reshape(1, -1), explainer_type
                )
                
                explanations.append({
                    'index': str(index),
                    'prediction': float(model.predict(row.values.reshape(1, -1))[0]),
                    'feature_attributions': {
                        feature_columns[i]: float(explanation.feature_attributions[i])
                        for i in range(len(feature_columns))
                    },
                    'base_value': float(explanation.base_value) if explanation.base_value is not None else None
                })
            
            # Get global feature importance
            global_importance = await self.model_explainer.global_feature_importance(explainer_type)
            
            return {
                'symbol': symbol,
                'explainer_type': explainer_type,
                'analysis_timestamp': datetime.now().isoformat(),
                'sample_size': len(explanations),
                'explanations': explanations,
                'global_feature_importance': {
                    feat.feature_name: float(feat.importance_score)
                    for feat in global_importance.feature_importances[:20]  # Top 20 features
                },
                'model_type': type(model).__name__
            }
            
        except Exception as e:
            logger.error(f"Model explanation failed for {symbol}: {str(e)}")
            return {
                'error': f'Model explanation failed: {str(e)}',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }

    async def analyze_feature_attribution_trends(self, symbol: str, 
                                               lookback_days: int = 90,
                                               regime_aware: bool = True) -> Dict[str, Any]:
        """
        Analyze feature attribution trends over time and across market regimes.
        
        Args:
            symbol: Stock symbol to analyze
            lookback_days: Number of days to look back for trend analysis
            regime_aware: Whether to include regime-specific analysis
            
        Returns:
            Feature attribution trends and regime-specific insights
        """
        try:
            logger.info(f"Analyzing feature attribution trends for {symbol}")
            
            # Get historical data
            data = await self.data_pipeline.prepare_data_for_analysis(
                symbol, f"{lookback_days}d", add_features=True
            )
            
            if data is None or len(data) < 30:
                return {
                    'error': f'Insufficient data for attribution analysis of {symbol}',
                    'symbol': symbol,
                    'required_samples': 30,
                    'available_samples': len(data) if data is not None else 0
                }
            
            # Get best model
            cache_key = f"{symbol}_best_model"
            if cache_key not in self.model_cache:
                await self.train_models(symbol, period=f"{lookback_days * 2}d")
                
            if cache_key not in self.model_cache:
                return {
                    'error': f'No trained model available for {symbol}',
                    'symbol': symbol
                }
                
            model = self.model_cache[cache_key]
            
            # Prepare features
            feature_columns = [col for col in data.columns if col != 'close']
            X = data[feature_columns].fillna(0)
            
            # Fit explainer
            await self.model_explainer.fit_explainers(model, X, explainer_types=['shap_tree'])
            
            # Analyze attribution over time windows
            window_size = min(20, len(X) // 3)
            attribution_trends = []
            
            for i in range(0, len(X) - window_size, window_size // 2):
                window_data = X.iloc[i:i + window_size]
                window_date = data.index[i + window_size - 1]
                
                # Get regime for this window if requested
                regime_info = None
                if regime_aware:
                    regime_data = data.iloc[i:i + window_size]
                    regime_analysis = self.regime_analyzer.analyze_regime_features(regime_data)
                    regime_info = {
                        'regime': regime_analysis['current_regime'].value,
                        'volatility_percentile': float(regime_analysis['volatility_percentile']),
                        'trend_strength': float(regime_analysis['trend_strength'])
                    }
                
                # Calculate average attribution for this window
                window_attributions = []
                sample_indices = window_data.sample(n=min(5, len(window_data)), random_state=42).index
                
                for idx in sample_indices:
                    row = window_data.loc[idx]
                    explanation = await self.model_explainer.explain_prediction(
                        row.values.reshape(1, -1), 'shap_tree'
                    )
                    window_attributions.append(explanation.feature_attributions)
                
                # Average attributions across samples
                avg_attributions = np.mean(window_attributions, axis=0)
                
                attribution_trends.append({
                    'date': window_date.isoformat(),
                    'feature_attributions': {
                        feature_columns[j]: float(avg_attributions[j])
                        for j in range(len(feature_columns))
                    },
                    'regime_info': regime_info,
                    'window_start': data.index[i].isoformat(),
                    'window_end': window_date.isoformat()
                })
            
            # Calculate attribution stability metrics
            all_attributions = np.array([
                list(trend['feature_attributions'].values()) 
                for trend in attribution_trends
            ])
            
            attribution_volatility = np.std(all_attributions, axis=0)
            attribution_means = np.mean(all_attributions, axis=0)
            
            stability_metrics = {
                feature_columns[i]: {
                    'mean_attribution': float(attribution_means[i]),
                    'attribution_volatility': float(attribution_volatility[i]),
                    'stability_score': float(1 / (1 + attribution_volatility[i])) if attribution_volatility[i] > 0 else 1.0
                }
                for i in range(len(feature_columns))
            }
            
            # Regime-specific analysis if requested
            regime_analysis = None
            if regime_aware and attribution_trends:
                regime_specific = await self.model_explainer.regime_specific_attribution(
                    model, X, data
                )
                regime_analysis = {
                    regime.value: {
                        'sample_count': info['sample_count'],
                        'top_features': {
                            feat.feature_name: float(feat.importance_score)
                            for feat in info['feature_importances'][:10]
                        }
                    }
                    for regime, info in regime_specific.items()
                }
            
            return {
                'symbol': symbol,
                'analysis_timestamp': datetime.now().isoformat(),
                'lookback_days': lookback_days,
                'attribution_trends': attribution_trends,
                'stability_metrics': stability_metrics,
                'regime_analysis': regime_analysis,
                'model_type': type(model).__name__,
                'total_windows': len(attribution_trends)
            }
            
        except Exception as e:
            logger.error(f"Attribution trend analysis failed for {symbol}: {str(e)}")
            return {
                'error': f'Attribution trend analysis failed: {str(e)}',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }

