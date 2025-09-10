import pandas as pd
import numpy as np
import structlog
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import httpx
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

from ..models import ForecastResponse, PriceAnalysis, TrendForecast
from ..cache import AnalysisCacheService, AnalysisCacheKeys
from ..config import settings

logger = structlog.get_logger(__name__)

class ForecastService:
    def __init__(self, db, redis_client):
        self.db = db
        self.cache = AnalysisCacheService(redis_client)
        self.market_data_client = httpx.AsyncClient(
            base_url=settings.MARKET_DATA_API_URL,
            timeout=30.0
        )
        self.models = {}
        self.scalers = {}
    
    async def get_forecast(
        self, 
        symbol: str, 
        model_type: str = "ensemble", 
        horizon: int = 5
    ) -> Optional[ForecastResponse]:
        """Generate ML forecast for stock price"""
        
        # Validate horizon
        if horizon > settings.MAX_FORECAST_HORIZON:
            horizon = settings.MAX_FORECAST_HORIZON
        
        cache_key = AnalysisCacheKeys.forecast(symbol, model_type, horizon)
        
        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.debug("Forecast cache hit", symbol=symbol, model_type=model_type)
            return ForecastResponse(**cached_result)
        
        try:
            # Get historical data (need more data for ML)
            historical_data = await self._fetch_historical_data(symbol, "2y")
            if not historical_data or len(historical_data) < 100:
                logger.warning("Insufficient data for forecast", symbol=symbol, data_points=len(historical_data) if historical_data else 0)
                return None
            
            # Prepare data
            df = self._prepare_dataframe(historical_data)
            
            # Feature engineering
            features_df = self._engineer_features(df)
            
            if len(features_df) < 50:
                logger.warning("Insufficient features for forecast", symbol=symbol)
                return None
            
            # Train model and make predictions
            predictions, confidence_score = await self._train_and_predict(
                features_df, model_type, horizon, symbol
            )
            
            if predictions is None:
                return None
            
            # Generate prediction dates
            last_date = df.index[-1]
            prediction_dates = [
                last_date + timedelta(days=i+1) for i in range(horizon)
            ]
            
            # Calculate price analysis
            current_price = float(df['close'].iloc[-1])
            price_analysis = self._calculate_price_analysis(predictions, current_price)
            
            # Generate trend forecast
            trend_forecast = self._generate_trend_forecast(predictions, current_price, confidence_score)
            
            result = ForecastResponse(
                symbol=symbol,
                predictions=predictions,
                dates=prediction_dates,
                current_price=current_price,
                model_type=model_type,
                price_analysis=price_analysis,
                trend_forecast=trend_forecast,
                confidence_score=confidence_score,
                calculated_at=datetime.now()
            )
            
            # Cache the result
            await self.cache.set(cache_key, result.dict(), settings.FORECAST_CACHE_TTL)
            
            return result
            
        except Exception as e:
            logger.error("Error generating forecast", symbol=symbol, error=str(e))
            return None
    
    async def _fetch_historical_data(self, symbol: str, period: str) -> Optional[List[Dict]]:
        """Fetch historical data from Market Data API"""
        try:
            response = await self.market_data_client.get(f"/stocks/{symbol}/history", params={"period": period})
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            logger.error("Failed to fetch historical data for forecast", symbol=symbol, error=str(e))
            return None
    
    def _prepare_dataframe(self, historical_data: List[Dict]) -> pd.DataFrame:
        """Convert historical data to pandas DataFrame"""
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.sort_index()
        
        # Ensure numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any NaN values
        df = df.dropna()
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML model"""
        try:
            features_df = pd.DataFrame(index=df.index)
            
            # Price-based features
            features_df['close'] = df['close']
            features_df['volume'] = df['volume']
            
            # Returns
            features_df['return_1d'] = df['close'].pct_change(1)
            features_df['return_5d'] = df['close'].pct_change(5)
            features_df['return_10d'] = df['close'].pct_change(10)
            features_df['return_20d'] = df['close'].pct_change(20)
            
            # Moving averages
            features_df['sma_5'] = df['close'].rolling(5).mean()
            features_df['sma_10'] = df['close'].rolling(10).mean()
            features_df['sma_20'] = df['close'].rolling(20).mean()
            features_df['sma_50'] = df['close'].rolling(50).mean()
            
            # Moving average ratios
            features_df['close_sma5_ratio'] = df['close'] / features_df['sma_5']
            features_df['close_sma20_ratio'] = df['close'] / features_df['sma_20']
            features_df['sma5_sma20_ratio'] = features_df['sma_5'] / features_df['sma_20']
            
            # Volatility features
            features_df['volatility_5d'] = features_df['return_1d'].rolling(5).std()
            features_df['volatility_20d'] = features_df['return_1d'].rolling(20).std()
            
            # Price position features
            features_df['high_low_ratio'] = df['high'] / df['low']
            features_df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Volume features
            features_df['volume_sma'] = df['volume'].rolling(20).mean()
            features_df['volume_ratio'] = df['volume'] / features_df['volume_sma']
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            features_df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            features_df['macd'] = ema12 - ema26
            features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
            
            # Bollinger Bands
            bb_sma = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            features_df['bb_upper'] = bb_sma + (bb_std * 2)
            features_df['bb_lower'] = bb_sma - (bb_std * 2)
            features_df['bb_position'] = (df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
            
            # Calendar features
            features_df['day_of_week'] = df.index.dayofweek
            features_df['month'] = df.index.month
            features_df['quarter'] = df.index.quarter
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                features_df[f'close_lag_{lag}'] = df['close'].shift(lag)
                features_df[f'return_lag_{lag}'] = features_df['return_1d'].shift(lag)
                features_df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            
            # Target variable (next day return)
            features_df['target'] = df['close'].shift(-1)
            
            # Remove NaN values
            features_df = features_df.dropna()
            
            return features_df
            
        except Exception as e:
            logger.error("Error engineering features", error=str(e))
            return pd.DataFrame()
    
    async def _train_and_predict(
        self, 
        features_df: pd.DataFrame, 
        model_type: str, 
        horizon: int, 
        symbol: str
    ) -> tuple[Optional[List[float]], float]:
        """Train model and make predictions"""
        try:
            # Prepare features and target
            target_col = 'target'
            feature_cols = [col for col in features_df.columns if col != target_col]
            
            X = features_df[feature_cols].iloc[:-1]  # Remove last row (no target)
            y = features_df[target_col].iloc[:-1]
            
            if len(X) < 30:
                return None, 0.0
            
            # Train-test split (time series split)
            split_point = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model based on type
            if model_type == "ensemble" or model_type == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)
            else:
                # Default to Random Forest
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Calculate confidence score based on model performance
            y_test_mean = y_test.mean()
            baseline_mse = mean_squared_error(y_test, [y_test_mean] * len(y_test))
            confidence_score = max(0.0, 1.0 - (mse / baseline_mse))
            confidence_score = min(0.9, confidence_score)  # Cap at 90%
            
            # Make multi-step predictions
            predictions = []
            last_features = features_df[feature_cols].iloc[-1:].copy()
            
            for step in range(horizon):
                # Scale current features
                current_features_scaled = scaler.transform(last_features)
                
                # Predict next price
                next_price = model.predict(current_features_scaled)[0]
                predictions.append(float(next_price))
                
                # Update features for next prediction
                # This is simplified - in practice, you'd update all time-dependent features
                current_price = float(last_features['close'].iloc[0])
                price_change = (next_price - current_price) / current_price
                
                # Update some key features for next iteration
                new_features = last_features.copy()
                new_features['close'] = next_price
                new_features['return_1d'] = price_change
                
                # Shift lag features
                for lag in [1, 2, 3, 5, 10]:
                    if f'close_lag_{lag}' in new_features.columns:
                        if lag == 1:
                            new_features[f'close_lag_{lag}'] = current_price
                        else:
                            new_features[f'close_lag_{lag}'] = last_features[f'close_lag_{lag-1}'].iloc[0]
                
                last_features = new_features
            
            # Store model and scaler for this symbol (simple in-memory cache)
            self.models[f"{symbol}_{model_type}"] = model
            self.scalers[f"{symbol}_{model_type}"] = scaler
            
            logger.info("Model trained successfully", 
                       symbol=symbol, 
                       model_type=model_type,
                       mse=mse, 
                       mae=mae, 
                       confidence_score=confidence_score)
            
            return predictions, confidence_score
            
        except Exception as e:
            logger.error("Error training model and predicting", symbol=symbol, error=str(e))
            return None, 0.0
    
    def _calculate_price_analysis(self, predictions: List[float], current_price: float) -> PriceAnalysis:
        """Calculate price analysis from predictions"""
        try:
            if not predictions:
                return PriceAnalysis()
            
            # Calculate returns
            expected_return_1d = (predictions[0] - current_price) / current_price * 100
            expected_return_5d = (predictions[-1] - current_price) / current_price * 100 if len(predictions) >= 5 else expected_return_1d
            
            # Calculate max gain and loss
            max_price = max(predictions)
            min_price = min(predictions)
            
            max_expected_gain = (max_price - current_price) / current_price * 100
            max_expected_loss = (min_price - current_price) / current_price * 100
            
            # Estimate volatility
            returns = [(predictions[i] - predictions[i-1]) / predictions[i-1] for i in range(1, len(predictions))]
            volatility_estimate = np.std(returns) * 100 if len(returns) > 1 else 0.0
            
            return PriceAnalysis(
                expected_return_1d=expected_return_1d,
                expected_return_5d=expected_return_5d,
                max_expected_gain=max_expected_gain,
                max_expected_loss=max_expected_loss,
                volatility_estimate=volatility_estimate
            )
            
        except Exception as e:
            logger.error("Error calculating price analysis", error=str(e))
            return PriceAnalysis()
    
    def _generate_trend_forecast(
        self, 
        predictions: List[float], 
        current_price: float, 
        confidence_score: float
    ) -> TrendForecast:
        """Generate trend forecast from predictions"""
        try:
            if not predictions:
                return TrendForecast(direction="SIDEWAYS", confidence="LOW", probability=0.5)
            
            # Calculate overall trend
            final_price = predictions[-1]
            price_change_pct = (final_price - current_price) / current_price * 100
            
            # Determine direction
            if price_change_pct > 2:
                direction = "UP"
            elif price_change_pct < -2:
                direction = "DOWN"
            else:
                direction = "SIDEWAYS"
            
            # Determine confidence level
            if confidence_score > 0.7:
                confidence = "HIGH"
            elif confidence_score > 0.4:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            
            # Calculate probability based on trend consistency
            trend_changes = 0
            for i in range(1, len(predictions)):
                if (predictions[i] > predictions[i-1]) != (predictions[0] > current_price):
                    trend_changes += 1
            
            trend_consistency = 1.0 - (trend_changes / len(predictions))
            probability = (confidence_score + trend_consistency) / 2
            
            return TrendForecast(
                direction=direction,
                confidence=confidence,
                probability=probability
            )
            
        except Exception as e:
            logger.error("Error generating trend forecast", error=str(e))
            return TrendForecast(direction="SIDEWAYS", confidence="LOW", probability=0.5)