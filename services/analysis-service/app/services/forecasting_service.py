"""
Forecasting Service
High-level service for ML-based price forecasting and prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

from ..models.ensemble_forecaster import EnsembleForecaster, ModelEvaluator
from ..models.forecasting_models import LSTMForecaster, RandomForestForecaster, ARIMAForecaster
from ..core.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)

class ForecastingService:
    """Comprehensive forecasting service with multiple ML models"""
    
    def __init__(self, market_data_url: str = "http://localhost:8002"):
        self.data_pipeline = DataPipeline(market_data_url)
        self.model_cache = {}  # Cache trained models by symbol
        self.model_evaluator = ModelEvaluator()
        
    async def generate_forecast(self, symbol: str, model_type: str = 'ensemble',
                              prediction_horizon: int = 5, period: str = '2y',
                              retrain: bool = False) -> Dict[str, Any]:
        """
        Generate price forecast for a symbol
        
        Args:
            symbol: Stock symbol to forecast
            model_type: 'ensemble', 'lstm', 'random_forest', 'arima'
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
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get current model cache status"""
        return {
            'cached_models': len(self.model_cache),
            'cached_symbols': list(set(key.split('_')[0] for key in self.model_cache.keys())),
            'cache_keys': list(self.model_cache.keys()),
            'memory_usage_estimate': f"{len(self.model_cache) * 50}MB"  # Rough estimate
        }