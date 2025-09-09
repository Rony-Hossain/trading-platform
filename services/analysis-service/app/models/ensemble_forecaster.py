"""
Ensemble Forecasting Model
Combines multiple forecasting approaches for superior accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import joblib
import json

from .forecasting_models import LSTMForecaster, RandomForestForecaster, ARIMAForecaster

logger = logging.getLogger(__name__)

class EnsembleForecaster:
    """Ensemble model combining LSTM, Random Forest, and ARIMA"""
    
    def __init__(self, prediction_horizon: int = 5, sequence_length: int = 60):
        self.prediction_horizon = prediction_horizon
        self.sequence_length = sequence_length
        
        # Individual models
        self.lstm_model = None
        self.rf_model = None
        self.arima_model = None
        
        # Ensemble weights (learned during validation)
        self.model_weights = {'lstm': 0.4, 'random_forest': 0.4, 'arima': 0.2}
        self.adaptive_weights = True
        
        # Model performance tracking
        self.model_performance = {}
        self.is_trained = False
        
    def initialize_models(self):
        """Initialize individual forecasting models"""
        try:
            self.lstm_model = LSTMForecaster(
                sequence_length=self.sequence_length,
                prediction_horizon=self.prediction_horizon
            )
        except ImportError:
            logger.warning("LSTM model not available (TensorFlow required)")
            self.lstm_model = None
        
        self.rf_model = RandomForestForecaster(
            n_estimators=200,
            prediction_horizon=self.prediction_horizon
        )
        
        try:
            self.arima_model = ARIMAForecaster(
                prediction_horizon=self.prediction_horizon
            )
        except ImportError:
            logger.warning("ARIMA model not available (Statsmodels required)")
            self.arima_model = None
    
    def calculate_model_weights(self, validation_results: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate optimal ensemble weights based on validation performance"""
        weights = {}
        total_inverse_error = 0
        
        # Use inverse of validation error as weight
        for model_name, results in validation_results.items():
            if 'error' not in results and 'validation_mae' in results:
                inverse_error = 1 / (results['validation_mae'] + 1e-6)
                weights[model_name] = inverse_error
                total_inverse_error += inverse_error
        
        # Normalize weights
        if total_inverse_error > 0:
            for model_name in weights:
                weights[model_name] /= total_inverse_error
        else:
            # Default equal weights
            n_models = len(validation_results)
            weights = {model: 1/n_models for model in validation_results.keys()}
        
        return weights
    
    def train_model_async(self, model, model_name: str, data: pd.DataFrame, target_column: str) -> Tuple[str, Dict]:
        """Train a single model asynchronously"""
        try:
            logger.info(f"Training {model_name} model...")
            result = model.train(data, target_column)
            result['model_name'] = model_name
            return model_name, result
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return model_name, {'error': str(e), 'model_name': model_name}
    
    async def train(self, data: pd.DataFrame, target_column: str = 'close', 
                   validation_split: float = 0.2) -> Dict[str, Any]:
        """Train all models in the ensemble"""
        try:
            logger.info(f"Training ensemble model with {len(data)} data points")
            
            # Initialize models
            self.initialize_models()
            
            # Split data for validation
            split_idx = int(len(data) * (1 - validation_split))
            train_data = data.iloc[:split_idx]
            validation_data = data.iloc[split_idx:]
            
            # Train models in parallel
            training_results = {}
            validation_results = {}
            
            # Prepare training tasks
            training_tasks = []
            
            if self.lstm_model:
                training_tasks.append(('lstm', self.lstm_model))
            if self.rf_model:
                training_tasks.append(('random_forest', self.rf_model))
            if self.arima_model:
                training_tasks.append(('arima', self.arima_model))
            
            if not training_tasks:
                raise ValueError("No models available for training")
            
            # Train models
            with ThreadPoolExecutor(max_workers=len(training_tasks)) as executor:
                futures = [
                    executor.submit(self.train_model_async, model, name, train_data, target_column)
                    for name, model in training_tasks
                ]
                
                for future in futures:
                    model_name, result = future.result()
                    training_results[model_name] = result
            
            # Validate models on held-out data
            logger.info("Validating models...")
            for model_name, model in [('lstm', self.lstm_model), ('random_forest', self.rf_model), ('arima', self.arima_model)]:
                if model is None or model_name not in training_results:
                    continue
                
                if 'error' in training_results[model_name]:
                    validation_results[model_name] = training_results[model_name]
                    continue
                
                try:
                    # Make predictions on validation set
                    val_pred = model.predict(validation_data, target_column)
                    
                    if 'error' not in val_pred and 'predictions' in val_pred:
                        # Calculate validation metrics
                        actual_prices = validation_data[target_column].iloc[-self.prediction_horizon:].values
                        predicted_prices = val_pred['predictions'][:len(actual_prices)]
                        
                        if len(predicted_prices) > 0 and len(actual_prices) > 0:
                            val_mae = np.mean(np.abs(np.array(predicted_prices) - actual_prices))
                            val_mse = np.mean((np.array(predicted_prices) - actual_prices) ** 2)
                            val_mape = np.mean(np.abs((actual_prices - np.array(predicted_prices)) / actual_prices)) * 100
                            
                            validation_results[model_name] = {
                                'validation_mae': float(val_mae),
                                'validation_mse': float(val_mse),
                                'validation_mape': float(val_mape),
                                'model_name': model_name
                            }
                        else:
                            validation_results[model_name] = {'error': 'No valid predictions', 'model_name': model_name}
                    else:
                        validation_results[model_name] = {'error': val_pred.get('error', 'Prediction failed'), 'model_name': model_name}
                        
                except Exception as e:
                    validation_results[model_name] = {'error': str(e), 'model_name': model_name}
            
            # Calculate adaptive weights based on validation performance
            if self.adaptive_weights:
                valid_models = {k: v for k, v in validation_results.items() if 'error' not in v}
                if valid_models:
                    self.model_weights = self.calculate_model_weights(valid_models)
                    logger.info(f"Adaptive weights: {self.model_weights}")
            
            # Store performance metrics
            self.model_performance = {
                'training': training_results,
                'validation': validation_results,
                'weights': self.model_weights
            }
            
            self.is_trained = True
            
            # Ensemble training summary
            ensemble_result = {
                'ensemble_type': 'Multi-Model Ensemble',
                'models_trained': list(training_results.keys()),
                'successful_models': [k for k, v in validation_results.items() if 'error' not in v],
                'model_weights': self.model_weights,
                'training_samples': len(train_data),
                'validation_samples': len(validation_data),
                'prediction_horizon': self.prediction_horizon,
                'individual_results': {
                    'training': training_results,
                    'validation': validation_results
                }
            }
            
            # Calculate ensemble validation score
            successful_models = [k for k, v in validation_results.items() if 'error' not in v and 'validation_mae' in v]
            if successful_models:
                weighted_mae = sum(
                    validation_results[model]['validation_mae'] * self.model_weights.get(model, 0)
                    for model in successful_models
                )
                ensemble_result['ensemble_validation_mae'] = float(weighted_mae)
            
            logger.info(f"Ensemble training completed. Successful models: {len(successful_models)}")
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            return {'error': str(e)}
    
    def predict(self, data: pd.DataFrame, target_column: str = 'close') -> Dict[str, Any]:
        """Make ensemble predictions"""
        if not self.is_trained:
            return {'error': 'Ensemble model not trained'}
        
        try:
            logger.info("Making ensemble predictions...")
            
            # Get predictions from each model
            model_predictions = {}
            model_confidence = {}
            
            # LSTM predictions
            if self.lstm_model and self.lstm_model.is_trained:
                lstm_pred = self.lstm_model.predict(data, target_column)
                if 'error' not in lstm_pred:
                    model_predictions['lstm'] = lstm_pred['predictions']
                    model_confidence['lstm'] = lstm_pred
            
            # Random Forest predictions
            if self.rf_model and self.rf_model.is_trained:
                rf_pred = self.rf_model.predict(data, target_column)
                if 'error' not in rf_pred:
                    model_predictions['random_forest'] = rf_pred['predictions']
                    model_confidence['random_forest'] = rf_pred
            
            # ARIMA predictions
            if self.arima_model and self.arima_model.is_trained:
                arima_pred = self.arima_model.predict(data, target_column)
                if 'error' not in arima_pred:
                    model_predictions['arima'] = arima_pred['predictions']
                    model_confidence['arima'] = arima_pred
            
            if not model_predictions:
                return {'error': 'No models available for prediction'}
            
            # Calculate ensemble predictions
            ensemble_predictions = []
            max_horizon = max(len(predictions) for predictions in model_predictions.values())
            
            for i in range(max_horizon):
                weighted_pred = 0
                total_weight = 0
                
                for model_name, predictions in model_predictions.items():
                    if i < len(predictions):
                        weight = self.model_weights.get(model_name, 0)
                        weighted_pred += predictions[i] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    ensemble_predictions.append(weighted_pred / total_weight)
                else:
                    # Fallback to simple average
                    available_preds = [predictions[i] for predictions in model_predictions.values() if i < len(predictions)]
                    if available_preds:
                        ensemble_predictions.append(np.mean(available_preds))
            
            # Calculate ensemble confidence intervals
            ensemble_intervals = []
            for i in range(len(ensemble_predictions)):
                individual_preds = []
                intervals = []
                
                for model_name, predictions in model_predictions.items():
                    if i < len(predictions):
                        individual_preds.append(predictions[i])
                        
                        # Get confidence intervals if available
                        if model_name in model_confidence:
                            conf = model_confidence[model_name]
                            if 'prediction_intervals' in conf and i < len(conf['prediction_intervals']):
                                intervals.append(conf['prediction_intervals'][i])
                
                if individual_preds:
                    # Calculate ensemble confidence based on model spread
                    pred_std = np.std(individual_preds)
                    pred_mean = np.mean(individual_preds)
                    
                    ensemble_intervals.append({
                        'lower': float(pred_mean - 1.96 * pred_std),
                        'upper': float(pred_mean + 1.96 * pred_std),
                        'std': float(pred_std),
                        'individual_predictions': individual_preds
                    })
            
            # Generate future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=len(ensemble_predictions),
                freq='B'
            )
            
            # Prediction strength based on model agreement
            prediction_strength = self._calculate_prediction_strength(model_predictions)
            
            return {
                'predictions': [float(p) for p in ensemble_predictions],
                'prediction_intervals': ensemble_intervals,
                'dates': [date.strftime('%Y-%m-%d') for date in future_dates],
                'current_price': float(data[target_column].iloc[-1]),
                'prediction_horizon': len(ensemble_predictions),
                'model_type': 'Ensemble',
                'model_weights': self.model_weights,
                'individual_predictions': {k: v for k, v in model_predictions.items()},
                'prediction_strength': prediction_strength,
                'ensemble_confidence': float(np.mean([interval['std'] for interval in ensemble_intervals])) if ensemble_intervals else None
            }
            
        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            return {'error': str(e)}
    
    def _calculate_prediction_strength(self, model_predictions: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate prediction strength based on model agreement"""
        if len(model_predictions) < 2:
            return {'agreement': 1.0, 'confidence': 'HIGH'}
        
        # Calculate agreement for each time step
        agreements = []
        max_horizon = max(len(predictions) for predictions in model_predictions.values())
        
        for i in range(max_horizon):
            step_predictions = []
            for predictions in model_predictions.values():
                if i < len(predictions):
                    step_predictions.append(predictions[i])
            
            if len(step_predictions) >= 2:
                # Calculate coefficient of variation as disagreement measure
                mean_pred = np.mean(step_predictions)
                std_pred = np.std(step_predictions)
                cv = std_pred / abs(mean_pred) if mean_pred != 0 else 1
                agreement = max(0, 1 - cv)  # Higher agreement = lower CV
                agreements.append(agreement)
        
        overall_agreement = np.mean(agreements) if agreements else 0.5
        
        # Categorize confidence
        if overall_agreement > 0.8:
            confidence = 'HIGH'
        elif overall_agreement > 0.6:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        return {
            'agreement': float(overall_agreement),
            'confidence': confidence,
            'step_agreements': [float(a) for a in agreements]
        }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get detailed performance metrics for all models"""
        return self.model_performance
    
    def save_models(self, filepath_prefix: str):
        """Save trained models to disk"""
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        # Save ensemble metadata
        ensemble_metadata = {
            'model_weights': self.model_weights,
            'prediction_horizon': self.prediction_horizon,
            'sequence_length': self.sequence_length,
            'adaptive_weights': self.adaptive_weights,
            'performance': self.model_performance,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{filepath_prefix}_ensemble_metadata.json", 'w') as f:
            json.dump(ensemble_metadata, f, indent=2)
        
        # Save individual models
        if self.rf_model and self.rf_model.is_trained:
            joblib.dump(self.rf_model, f"{filepath_prefix}_random_forest.pkl")
        
        # Note: LSTM and ARIMA models have their own save mechanisms
        # This is a simplified version - in production, implement full model persistence
        
        logger.info(f"Models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix: str):
        """Load trained models from disk"""
        # Load ensemble metadata
        with open(f"{filepath_prefix}_ensemble_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.model_weights = metadata['model_weights']
        self.prediction_horizon = metadata['prediction_horizon']
        self.sequence_length = metadata['sequence_length']
        self.adaptive_weights = metadata['adaptive_weights']
        self.model_performance = metadata['performance']
        
        # Load individual models
        try:
            self.rf_model = joblib.load(f"{filepath_prefix}_random_forest.pkl")
        except FileNotFoundError:
            logger.warning("Random Forest model file not found")
        
        self.is_trained = True
        logger.info(f"Models loaded from prefix: {filepath_prefix}")

class ModelEvaluator:
    """Comprehensive model evaluation and backtesting"""
    
    @staticmethod
    def walk_forward_validation(forecaster, data: pd.DataFrame, target_column: str = 'close',
                              window_size: int = 252, step_size: int = 21) -> Dict[str, Any]:
        """Perform walk-forward validation"""
        logger.info("Performing walk-forward validation...")
        
        predictions = []
        actuals = []
        evaluation_dates = []
        
        for start_idx in range(window_size, len(data) - forecaster.prediction_horizon, step_size):
            end_idx = start_idx + window_size
            
            # Training data
            train_data = data.iloc[start_idx-window_size:start_idx]
            
            # Test data (next period for actual values)
            test_data = data.iloc[start_idx:start_idx + forecaster.prediction_horizon]
            
            try:
                # Train model on window
                train_result = forecaster.train(train_data, target_column)
                if 'error' in train_result:
                    continue
                
                # Make prediction
                pred_result = forecaster.predict(train_data, target_column)
                if 'error' in pred_result:
                    continue
                
                # Compare with actual
                pred_values = pred_result['predictions']
                actual_values = test_data[target_column].values[:len(pred_values)]
                
                predictions.extend(pred_values)
                actuals.extend(actual_values)
                evaluation_dates.extend(test_data.index[:len(pred_values)])
                
            except Exception as e:
                logger.warning(f"Validation step failed: {e}")
                continue
        
        # Calculate comprehensive metrics
        if predictions and actuals:
            return ModelEvaluator.calculate_comprehensive_metrics(actuals, predictions, evaluation_dates)
        else:
            return {'error': 'No valid predictions made during validation'}
    
    @staticmethod
    def calculate_comprehensive_metrics(actual: List[float], predicted: List[float], 
                                      dates: List = None) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Basic metrics
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Statistical metrics
        correlation = np.corrcoef(actual, predicted)[0, 1] if len(actual) > 1 else 0
        r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
        
        # Directional accuracy
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        directional_accuracy = np.mean(actual_direction == predicted_direction) if len(actual) > 1 else 0.5
        
        # Theil's U statistic
        numerator = np.sqrt(np.mean((predicted - actual) ** 2))
        denominator = np.sqrt(np.mean(actual ** 2)) + np.sqrt(np.mean(predicted ** 2))
        theil_u = numerator / denominator if denominator != 0 else float('inf')
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'correlation': float(correlation),
            'r2_score': float(r2),
            'directional_accuracy': float(directional_accuracy),
            'theil_u': float(theil_u),
            'sample_size': len(actual),
            'mean_actual': float(np.mean(actual)),
            'mean_predicted': float(np.mean(predicted)),
            'std_actual': float(np.std(actual)),
            'std_predicted': float(np.std(predicted))
        }