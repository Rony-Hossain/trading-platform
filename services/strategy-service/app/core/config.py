"""
Configuration settings for the Strategy Service.

Includes triple barrier labeling settings and other ML configuration.
"""

import os
from typing import Optional
from pydantic import BaseSettings, Field


class TripleBarrierSettings(BaseSettings):
    """Triple Barrier Labeling Configuration."""
    
    # Core triple barrier parameters
    tb_horizon_days: int = Field(
        default=5,
        env="TB_HORIZON_DAYS",
        description="Maximum holding period in days"
    )
    
    tb_upper_sigma: float = Field(
        default=2.0,
        env="TB_UPPER_SIGMA", 
        description="Upper barrier threshold in volatility units"
    )
    
    tb_lower_sigma: float = Field(
        default=1.5,
        env="TB_LOWER_SIGMA",
        description="Lower barrier threshold in volatility units"
    )
    
    # Volatility and risk parameters
    volatility_lookback: int = Field(
        default=20,
        env="TB_VOLATILITY_LOOKBACK",
        description="Days for volatility calculation"
    )
    
    min_return_threshold: float = Field(
        default=0.005,
        env="TB_MIN_RETURN_THRESHOLD",
        description="Minimum return to consider significant (0.5%)"
    )
    
    max_holding_time: int = Field(
        default=10,
        env="TB_MAX_HOLDING_TIME",
        description="Maximum holding period in days"
    )
    
    # Meta-labeling parameters
    meta_train_ratio: float = Field(
        default=0.7,
        env="TB_META_TRAIN_RATIO",
        description="Training split for meta-labeling"
    )
    
    meta_cv_folds: int = Field(
        default=3,
        env="TB_META_CV_FOLDS",
        description="Cross-validation folds for meta-labeling"
    )
    
    # Performance thresholds
    f1_improvement_threshold: float = Field(
        default=0.05,
        env="TB_F1_IMPROVEMENT_THRESHOLD",
        description="Minimum F1 score improvement for meta-labeling acceptance"
    )
    
    calibration_slope_min: float = Field(
        default=0.9,
        env="TB_CALIBRATION_SLOPE_MIN",
        description="Minimum calibration slope for well-calibrated model"
    )
    
    calibration_slope_max: float = Field(
        default=1.1,
        env="TB_CALIBRATION_SLOPE_MAX", 
        description="Maximum calibration slope for well-calibrated model"
    )

    class Config:
        env_prefix = ""
        case_sensitive = False


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    database_url: str = Field(
        default="postgresql://trading_user:trading_pass@localhost:5432/trading_db",
        env="DATABASE_URL",
        description="Database connection URL"
    )
    
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL",
        description="Redis connection URL for caching"
    )

    class Config:
        env_prefix = ""
        case_sensitive = False


class MLSettings(BaseSettings):
    """Machine Learning configuration."""
    
    # Model settings
    default_n_estimators: int = Field(
        default=100,
        env="ML_N_ESTIMATORS",
        description="Default number of estimators for ensemble models"
    )
    
    max_depth: int = Field(
        default=10,
        env="ML_MAX_DEPTH",
        description="Maximum depth for tree-based models"
    )
    
    random_state: int = Field(
        default=42,
        env="ML_RANDOM_STATE",
        description="Random state for reproducibility"
    )
    
    # Feature engineering
    feature_lookback_days: int = Field(
        default=20,
        env="ML_FEATURE_LOOKBACK_DAYS",
        description="Days of historical data for feature engineering"
    )
    
    include_technical_indicators: bool = Field(
        default=True,
        env="ML_INCLUDE_TECHNICAL_INDICATORS",
        description="Include technical indicators in features"
    )
    
    include_sentiment_features: bool = Field(
        default=True,
        env="ML_INCLUDE_SENTIMENT_FEATURES",
        description="Include sentiment features"
    )
    
    include_macro_features: bool = Field(
        default=True,
        env="ML_INCLUDE_MACRO_FEATURES", 
        description="Include macro-economic features"
    )
    
    # Cross-validation
    cv_folds: int = Field(
        default=5,
        env="ML_CV_FOLDS",
        description="Number of cross-validation folds"
    )
    
    # Model persistence
    model_storage_path: str = Field(
        default="./models",
        env="ML_MODEL_STORAGE_PATH",
        description="Path to store trained models"
    )
    
    dataset_storage_path: str = Field(
        default="./datasets",
        env="ML_DATASET_STORAGE_PATH",
        description="Path to store datasets"
    )

    class Config:
        env_prefix = ""
        case_sensitive = False


class APISettings(BaseSettings):
    """API configuration."""
    
    host: str = Field(
        default="0.0.0.0",
        env="API_HOST",
        description="API host"
    )
    
    port: int = Field(
        default=8006,
        env="API_PORT",
        description="API port"
    )
    
    debug: bool = Field(
        default=False,
        env="API_DEBUG",
        description="Enable debug mode"
    )
    
    reload: bool = Field(
        default=False,
        env="API_RELOAD",
        description="Enable auto-reload in development"
    )

    class Config:
        env_prefix = ""
        case_sensitive = False


class Settings(BaseSettings):
    """Main application settings."""
    
    # Service identification
    service_name: str = "strategy-service"
    version: str = "1.0.0"
    
    # Component settings
    triple_barrier: TripleBarrierSettings = TripleBarrierSettings()
    database: DatabaseSettings = DatabaseSettings()
    ml: MLSettings = MLSettings()
    api: APISettings = APISettings()
    
    # Environment
    environment: str = Field(
        default="development",
        env="ENVIRONMENT",
        description="Application environment"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level"
    )

    class Config:
        env_prefix = ""
        case_sensitive = False


# Global settings instance
settings = Settings()


# Convenience functions to get specific settings
def get_triple_barrier_config():
    """Get triple barrier configuration as a dictionary."""
    return {
        'horizon_days': settings.triple_barrier.tb_horizon_days,
        'upper_sigma': settings.triple_barrier.tb_upper_sigma,
        'lower_sigma': settings.triple_barrier.tb_lower_sigma,
        'volatility_lookback': settings.triple_barrier.volatility_lookback,
        'min_return_threshold': settings.triple_barrier.min_return_threshold,
        'max_holding_time': settings.triple_barrier.max_holding_time
    }


def get_ml_config():
    """Get machine learning configuration as a dictionary."""
    return {
        'n_estimators': settings.ml.default_n_estimators,
        'max_depth': settings.ml.max_depth,
        'random_state': settings.ml.random_state,
        'feature_lookback_days': settings.ml.feature_lookback_days,
        'cv_folds': settings.ml.cv_folds,
        'model_storage_path': settings.ml.model_storage_path,
        'dataset_storage_path': settings.ml.dataset_storage_path
    }


def get_acceptance_criteria():
    """Get acceptance criteria for meta-labeling validation."""
    return {
        'f1_improvement_threshold': settings.triple_barrier.f1_improvement_threshold,
        'calibration_slope_range': (
            settings.triple_barrier.calibration_slope_min,
            settings.triple_barrier.calibration_slope_max
        )
    }


# Environment-specific overrides
if settings.environment == "production":
    settings.api.debug = False
    settings.api.reload = False
    settings.log_level = "WARNING"
elif settings.environment == "testing":
    settings.database.database_url = "sqlite:///./test.db"
    settings.ml.model_storage_path = "./test_models"
    settings.ml.dataset_storage_path = "./test_datasets"


# Validation functions
def validate_triple_barrier_config():
    """Validate triple barrier configuration."""
    tb = settings.triple_barrier
    
    errors = []
    
    if tb.tb_upper_sigma <= 0:
        errors.append("tb_upper_sigma must be positive")
        
    if tb.tb_lower_sigma <= 0:
        errors.append("tb_lower_sigma must be positive")
        
    if tb.tb_horizon_days <= 0:
        errors.append("tb_horizon_days must be positive")
        
    if tb.min_return_threshold < 0:
        errors.append("min_return_threshold must be non-negative")
        
    if not (0.0 < tb.meta_train_ratio < 1.0):
        errors.append("meta_train_ratio must be between 0 and 1")
        
    if tb.f1_improvement_threshold < 0:
        errors.append("f1_improvement_threshold must be non-negative")
        
    if tb.calibration_slope_min >= tb.calibration_slope_max:
        errors.append("calibration_slope_min must be less than calibration_slope_max")
    
    return errors


def validate_all_settings():
    """Validate all settings and return any errors."""
    errors = []
    
    # Validate triple barrier settings
    tb_errors = validate_triple_barrier_config()
    errors.extend([f"Triple Barrier: {err}" for err in tb_errors])
    
    # Validate ML settings
    ml = settings.ml
    if ml.cv_folds <= 1:
        errors.append("ML: cv_folds must be greater than 1")
        
    if ml.max_depth <= 0:
        errors.append("ML: max_depth must be positive")
    
    # Validate API settings
    api = settings.api
    if not (1 <= api.port <= 65535):
        errors.append("API: port must be between 1 and 65535")
    
    return errors


# Initialize and validate settings on import
validation_errors = validate_all_settings()
if validation_errors:
    raise ValueError(f"Configuration validation failed: {validation_errors}")


# Export main components
__all__ = [
    'Settings',
    'TripleBarrierSettings',
    'DatabaseSettings', 
    'MLSettings',
    'APISettings',
    'settings',
    'get_triple_barrier_config',
    'get_ml_config', 
    'get_acceptance_criteria',
    'validate_triple_barrier_config',
    'validate_all_settings'
]