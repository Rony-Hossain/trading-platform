"""
ML Forecasting Models
Advanced machine learning models for stock price prediction and forecasting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, MultiHeadAttention
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available, LSTM models will be disabled")

# Statistical Models
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("Statsmodels not available, ARIMA models will be disabled")

# Advanced ML
try:
    import xgboost as xgb
    import lightgbm as lgb
    BOOSTING_AVAILABLE = True
except ImportError:
    BOOSTING_AVAILABLE = False
    logging.warning("XGBoost/LightGBM not available, boosting models will be disabled")

logger = logging.getLogger(__name__)

class LSTMForecaster:
    """LSTM Neural Network for Time Series Forecasting"""
    
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 5):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
    
    def prepare_sequences(self, data: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data[[target_column]])
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(scaled_data) - self.prediction_horizon + 1):
            # Input sequence
            X.append(scaled_data[i-self.sequence_length:i, 0])
            # Target (next prediction_horizon values)
            y.append(scaled_data[i:i+self.prediction_horizon, 0])
        
        return np.array(X), np.array(y)
    
    def prepare_multivariate_sequences(self, data: pd.DataFrame, target_column: str = 'close', 
                                     feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare multivariate sequences with additional features"""
        if feature_columns is None:
            feature_columns = ['volume', 'returns', 'volatility_5d', 'rsi_14']
        
        # Select available features
        available_features = [col for col in feature_columns if col in data.columns]
        all_features = [target_column] + available_features
        
        # Scale target and features separately
        target_scaled = self.scaler.fit_transform(data[[target_column]])
        
        if available_features:
            features_scaled = self.feature_scaler.fit_transform(data[available_features])
            combined_data = np.hstack([target_scaled, features_scaled])
        else:
            combined_data = target_scaled
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(combined_data) - self.prediction_horizon + 1):
            # Input sequence (all features)
            X.append(combined_data[i-self.sequence_length:i])
            # Target (only price predictions)
            y.append(target_scaled[i:i+self.prediction_horizon, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build LSTM architecture"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(self.prediction_horizon)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_attention_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build LSTM with attention mechanism"""
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        lstm_out = LSTM(128, return_sequences=True)(inputs)
        lstm_out = Dropout(0.2)(lstm_out)
        lstm_out = LSTM(64, return_sequences=True)(lstm_out)
        lstm_out = Dropout(0.2)(lstm_out)
        
        # Attention mechanism
        attention = MultiHeadAttention(num_heads=4, key_dim=64)(lstm_out, lstm_out)
        attention = Dropout(0.1)(attention)
        
        # Global pooling and dense layers
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention)
        dense = Dense(32, activation='relu')(pooled)
        dense = Dropout(0.2)(dense)
        outputs = Dense(self.prediction_horizon)(dense)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, data: pd.DataFrame, target_column: str = 'close', 
              use_multivariate: bool = True, use_attention: bool = False,
              validation_split: float = 0.2, epochs: int = 100) -> Dict[str, Any]:
        """Train the LSTM model"""
        try:
            logger.info(f"Training LSTM model with {len(data)} data points")
            
            # Prepare data
            if use_multivariate:
                X, y = self.prepare_multivariate_sequences(data, target_column)
            else:
                X, y = self.prepare_sequences(data, target_column)
            
            if len(X) == 0:
                raise ValueError("Not enough data to create sequences")
            
            # Build model
            if use_attention:
                self.model = self.build_attention_model((X.shape[1], X.shape[2]))
            else:
                self.model = self.build_model((X.shape[1], X.shape[2]))
            
            # Callbacks
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
            
            # Train model
            history = self.model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            self.is_trained = True
            
            # Calculate metrics
            final_loss = min(history.history['val_loss'])
            final_mae = min(history.history['val_mae'])
            
            training_result = {
                'model_type': 'LSTM_Attention' if use_attention else 'LSTM',
                'training_samples': len(X),
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'final_val_loss': float(final_loss),
                'final_val_mae': float(final_mae),
                'epochs_trained': len(history.history['loss']),
                'multivariate': use_multivariate
            }
            
            logger.info(f"LSTM training completed: Loss={final_loss:.6f}, MAE={final_mae:.6f}")
            return training_result
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {'error': str(e)}
    
    def predict(self, data: pd.DataFrame, target_column: str = 'close') -> Dict[str, Any]:
        """Make predictions with trained LSTM model"""
        if not self.is_trained or self.model is None:
            return {'error': 'Model not trained'}
        
        try:
            # Prepare last sequence
            if hasattr(self, 'feature_scaler'):
                X, _ = self.prepare_multivariate_sequences(data, target_column)
            else:
                X, _ = self.prepare_sequences(data, target_column)
            
            if len(X) == 0:
                return {'error': 'Not enough data for prediction'}
            
            # Use last sequence for prediction
            last_sequence = X[-1:] 
            
            # Make prediction
            prediction_scaled = self.model.predict(last_sequence, verbose=0)
            
            # Inverse transform
            prediction_reshaped = prediction_scaled.reshape(-1, 1)
            prediction = self.scaler.inverse_transform(prediction_reshaped).flatten()
            
            # Generate future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=self.prediction_horizon,
                freq='B'  # Business days
            )
            
            return {
                'predictions': prediction.tolist(),
                'dates': [date.strftime('%Y-%m-%d') for date in future_dates],
                'current_price': float(data[target_column].iloc[-1]),
                'prediction_horizon': self.prediction_horizon,
                'model_type': 'LSTM'
            }
            
        except Exception as e:
            logger.error(f"Error making LSTM prediction: {e}")
            return {'error': str(e)}

class RandomForestForecaster:
    """Random Forest model for feature-based forecasting"""
    
    def __init__(self, n_estimators: int = 200, prediction_horizon: int = 5):
        self.n_estimators = n_estimators
        self.prediction_horizon = prediction_horizon
        self.models = {}  # One model per prediction step
        self.feature_importance = {}
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features_and_targets(self, data: pd.DataFrame, target_column: str = 'close') -> Tuple[pd.DataFrame, Dict[int, pd.Series]]:
        """Prepare features and multi-step targets"""
        # Select feature columns (exclude target and non-numeric)
        feature_cols = []
        for col in data.columns:
            if col != target_column and data[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                if not col.startswith('target_') and col != 'symbol':
                    feature_cols.append(col)
        
        # Create lagged features
        features_df = data[feature_cols].copy()
        
        # Add more predictive features
        if target_column in data.columns:
            # Price momentum features
            features_df[f'{target_column}_momentum_3'] = data[target_column].pct_change(3)
            features_df[f'{target_column}_momentum_5'] = data[target_column].pct_change(5)
            features_df[f'{target_column}_momentum_10'] = data[target_column].pct_change(10)
            
            # Moving average ratios
            for window in [5, 10, 20]:
                ma = data[target_column].rolling(window).mean()
                features_df[f'{target_column}_ma_ratio_{window}'] = data[target_column] / ma
        
        # Create targets for each prediction step
        targets = {}
        for step in range(1, self.prediction_horizon + 1):
            targets[step] = data[target_column].shift(-step)
        
        # Remove rows with NaN values
        valid_idx = features_df.notna().all(axis=1)
        for step in targets:
            valid_idx &= targets[step].notna()
        
        features_clean = features_df[valid_idx]
        targets_clean = {step: targets[step][valid_idx] for step in targets}
        
        return features_clean, targets_clean
    
    def train(self, data: pd.DataFrame, target_column: str = 'close') -> Dict[str, Any]:
        """Train Random Forest models"""
        try:
            logger.info(f"Training Random Forest model with {len(data)} data points")
            
            # Prepare data
            features, targets = self.prepare_features_and_targets(data, target_column)
            
            if len(features) < 50:
                raise ValueError("Not enough data for Random Forest training")
            
            # Scale features
            features_scaled = pd.DataFrame(
                self.scaler.fit_transform(features),
                index=features.index,
                columns=features.columns
            )
            
            # Train one model per prediction step
            training_scores = {}
            
            for step in range(1, self.prediction_horizon + 1):
                # Create time series split for validation
                tscv = TimeSeriesSplit(n_splits=3)
                
                # Train model
                model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, features_scaled, targets[step],
                    cv=tscv, scoring='neg_mean_squared_error'
                )
                
                # Train on full dataset
                model.fit(features_scaled, targets[step])
                
                # Store model and importance
                self.models[step] = model
                self.feature_importance[step] = dict(zip(features.columns, model.feature_importances_))
                
                # Calculate metrics
                train_pred = model.predict(features_scaled)
                train_r2 = r2_score(targets[step], train_pred)
                train_mae = mean_absolute_error(targets[step], train_pred)
                
                training_scores[step] = {
                    'cv_score': float(-cv_scores.mean()),
                    'cv_std': float(cv_scores.std()),
                    'train_r2': float(train_r2),
                    'train_mae': float(train_mae)
                }
                
                logger.info(f"Step {step}: R²={train_r2:.4f}, MAE={train_mae:.4f}")
            
            self.is_trained = True
            
            # Overall feature importance (average across all steps)
            all_features = set()
            for step_importance in self.feature_importance.values():
                all_features.update(step_importance.keys())
            
            overall_importance = {}
            for feature in all_features:
                importance_values = [self.feature_importance[step].get(feature, 0) for step in self.feature_importance]
                overall_importance[feature] = np.mean(importance_values)
            
            # Sort by importance
            sorted_importance = dict(sorted(overall_importance.items(), key=lambda x: x[1], reverse=True))
            
            training_result = {
                'model_type': 'RandomForest',
                'n_estimators': self.n_estimators,
                'training_samples': len(features),
                'n_features': len(features.columns),
                'prediction_horizon': self.prediction_horizon,
                'step_scores': training_scores,
                'feature_importance': dict(list(sorted_importance.items())[:20]),  # Top 20 features
                'feature_names': list(features.columns)
            }
            
            logger.info("Random Forest training completed")
            return training_result
            
        except Exception as e:
            logger.error(f"Error training Random Forest model: {e}")
            return {'error': str(e)}
    
    def predict(self, data: pd.DataFrame, target_column: str = 'close') -> Dict[str, Any]:
        """Make predictions with trained Random Forest model"""
        if not self.is_trained or not self.models:
            return {'error': 'Model not trained'}
        
        try:
            # Prepare features
            features, _ = self.prepare_features_and_targets(data, target_column)
            
            if len(features) == 0:
                return {'error': 'No valid features for prediction'}
            
            # Use last row for prediction
            last_features = features.iloc[-1:] 
            last_features_scaled = self.scaler.transform(last_features)
            
            # Make predictions for each step
            predictions = []
            prediction_intervals = []
            
            for step in range(1, self.prediction_horizon + 1):
                if step not in self.models:
                    continue
                
                model = self.models[step]
                
                # Point prediction
                pred = model.predict(last_features_scaled)[0]
                predictions.append(float(pred))
                
                # Prediction interval using tree predictions
                tree_predictions = []
                for estimator in model.estimators_:
                    tree_pred = estimator.predict(last_features_scaled)[0]
                    tree_predictions.append(tree_pred)
                
                # Calculate confidence interval
                tree_predictions = np.array(tree_predictions)
                lower_bound = np.percentile(tree_predictions, 5)
                upper_bound = np.percentile(tree_predictions, 95)
                
                prediction_intervals.append({
                    'lower': float(lower_bound),
                    'upper': float(upper_bound),
                    'std': float(np.std(tree_predictions))
                })
            
            # Generate future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=len(predictions),
                freq='B'
            )
            
            return {
                'predictions': predictions,
                'prediction_intervals': prediction_intervals,
                'dates': [date.strftime('%Y-%m-%d') for date in future_dates],
                'current_price': float(data[target_column].iloc[-1]),
                'prediction_horizon': len(predictions),
                'model_type': 'RandomForest',
                'feature_importance': dict(list(self.feature_importance[1].items())[:10])  # Top 10 for step 1
            }
            
        except Exception as e:
            logger.error(f"Error making Random Forest prediction: {e}")
            return {'error': str(e)}

class ARIMAForecaster:
    """ARIMA Statistical Model for Time Series Forecasting"""
    
    def __init__(self, prediction_horizon: int = 5):
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.model_fit = None
        self.order = None
        self.seasonal_order = None
        self.is_trained = False
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels is required for ARIMA models")
    
    def find_best_order(self, data: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """Find best ARIMA order using AIC"""
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        logger.info("Finding optimal ARIMA parameters...")
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted_model = model.fit()
                        aic = fitted_model.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        logger.info(f"Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
        return best_order
    
    def check_stationarity(self, data: pd.Series) -> Dict[str, Any]:
        """Check if series is stationary"""
        from statsmodels.tsa.stattools import adfuller
        
        result = adfuller(data.dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def train(self, data: pd.DataFrame, target_column: str = 'close', 
              auto_order: bool = True, order: Tuple[int, int, int] = None) -> Dict[str, Any]:
        """Train ARIMA model"""
        try:
            logger.info(f"Training ARIMA model with {len(data)} data points")
            
            # Extract time series
            series = data[target_column].dropna()
            
            if len(series) < 50:
                raise ValueError("Not enough data for ARIMA training")
            
            # Check stationarity
            stationarity = self.check_stationarity(series)
            
            # Find optimal order if not provided
            if auto_order:
                self.order = self.find_best_order(series)
            else:
                self.order = order or (1, 1, 1)
            
            # Fit ARIMA model
            self.model = ARIMA(series, order=self.order)
            self.model_fit = self.model.fit()
            
            self.is_trained = True
            
            # Model diagnostics
            residuals = self.model_fit.resid
            
            # Ljung-Box test for residual autocorrelation
            lb_stat, lb_p_value = acorr_ljungbox(residuals, lags=10, return_df=False)
            
            # Calculate metrics
            fitted_values = self.model_fit.fittedvalues
            mse = mean_squared_error(series[len(series)-len(fitted_values):], fitted_values)
            mae = mean_absolute_error(series[len(series)-len(fitted_values):], fitted_values)
            
            training_result = {
                'model_type': 'ARIMA',
                'order': self.order,
                'aic': float(self.model_fit.aic),
                'bic': float(self.model_fit.bic),
                'mse': float(mse),
                'mae': float(mae),
                'training_samples': len(series),
                'stationarity': stationarity,
                'ljung_box_p_value': float(lb_p_value[-1]) if isinstance(lb_p_value, np.ndarray) else float(lb_p_value),
                'residuals_autocorrelated': float(lb_p_value[-1]) < 0.05 if isinstance(lb_p_value, np.ndarray) else float(lb_p_value) < 0.05
            }
            
            logger.info(f"ARIMA training completed: AIC={self.model_fit.aic:.2f}")
            return training_result
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
            return {'error': str(e)}
    
    def predict(self, data: pd.DataFrame, target_column: str = 'close') -> Dict[str, Any]:
        """Make predictions with trained ARIMA model"""
        if not self.is_trained or self.model_fit is None:
            return {'error': 'Model not trained'}
        
        try:
            # Make forecast
            forecast = self.model_fit.forecast(steps=self.prediction_horizon)
            forecast_ci = self.model_fit.get_forecast(steps=self.prediction_horizon).conf_int()
            
            # Generate future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=self.prediction_horizon,
                freq='B'
            )
            
            # Prepare confidence intervals
            prediction_intervals = []
            for i in range(self.prediction_horizon):
                prediction_intervals.append({
                    'lower': float(forecast_ci.iloc[i, 0]),
                    'upper': float(forecast_ci.iloc[i, 1])
                })
            
            return {
                'predictions': forecast.tolist(),
                'prediction_intervals': prediction_intervals,
                'dates': [date.strftime('%Y-%m-%d') for date in future_dates],
                'current_price': float(data[target_column].iloc[-1]),
                'prediction_horizon': self.prediction_horizon,
                'model_type': 'ARIMA',
                'order': self.order
            }
            
        except Exception as e:
            logger.error(f"Error making ARIMA prediction: {e}")
            return {'error': str(e)}