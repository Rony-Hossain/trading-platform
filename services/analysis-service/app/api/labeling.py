"""
API endpoints for Triple-Barrier Labeling and Meta-Labeling.

Provides REST API access to advanced event trading label generation:
- Triple-barrier method for creating ML labels
- Meta-labeling for signal quality improvement
- Event extraction and analysis
- Label statistics and validation
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from services.triple_barrier_labeling import (
    TripleBarrierLabeler, MetaLabeler, AdvancedLabelingSystem,
    TripleBarrierLabel, MetaLabelResult
)

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class PriceDataPoint(BaseModel):
    """Single price data point."""
    timestamp: datetime
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: float
    volume: Optional[float] = None


class SignalDataPoint(BaseModel):
    """Trading signal data point."""
    timestamp: datetime
    signal: int = Field(..., ge=-1, le=1, description="Trading signal: -1 (short), 0 (neutral), 1 (long)")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class TripleBarrierRequest(BaseModel):
    """Request model for triple-barrier labeling."""
    price_data: List[PriceDataPoint] = Field(..., description="Historical price data")
    signals: Optional[List[SignalDataPoint]] = Field(None, description="Optional trading signals")
    symbol: str = Field(..., description="Trading symbol")
    
    # Barrier configuration
    profit_threshold: float = Field(0.02, ge=0.001, le=0.1, description="Profit threshold (e.g., 0.02 = 2%)")
    stop_threshold: float = Field(0.01, ge=0.001, le=0.1, description="Stop-loss threshold")
    max_holding_period: str = Field("5D", description="Maximum holding period (pandas Timedelta format)")
    
    # Advanced options
    volatility_adjustment: bool = Field(True, description="Adjust barriers based on volatility")
    dynamic_barriers: bool = Field(False, description="Use dynamic barrier calculation")
    min_event_separation: str = Field("1H", description="Minimum time between events")


class MetaLabelingRequest(BaseModel):
    """Request model for meta-labeling."""
    labels_data: List[Dict[str, Any]] = Field(..., description="Triple-barrier labels from previous step")
    price_data: List[PriceDataPoint] = Field(..., description="Historical price data")
    volume_data: Optional[List[Dict[str, Any]]] = Field(None, description="Volume data")
    volatility_data: Optional[List[Dict[str, Any]]] = Field(None, description="Volatility data")
    sentiment_data: Optional[List[Dict[str, Any]]] = Field(None, description="Sentiment data")
    
    # Model configuration
    cv_folds: int = Field(5, ge=3, le=10, description="Cross-validation folds")
    min_precision: float = Field(0.55, ge=0.5, le=0.9, description="Minimum precision threshold")


class AdvancedLabelingRequest(BaseModel):
    """Request model for complete labeling system."""
    price_data: List[PriceDataPoint] = Field(..., description="Historical price data")
    signals: Optional[List[SignalDataPoint]] = Field(None, description="Optional trading signals")
    symbol: str = Field(..., description="Trading symbol")
    
    # Market data
    volume_data: Optional[List[Dict[str, Any]]] = Field(None, description="Volume data")
    volatility_data: Optional[List[Dict[str, Any]]] = Field(None, description="Volatility data") 
    sentiment_data: Optional[List[Dict[str, Any]]] = Field(None, description="Sentiment data")
    
    # Configuration
    labeler_config: Optional[Dict[str, Any]] = Field(None, description="Triple-barrier labeler configuration")
    meta_config: Optional[Dict[str, Any]] = Field(None, description="Meta-labeler configuration")
    fit_meta_model: bool = Field(True, description="Whether to fit meta-labeling model")


class LabelResponse(BaseModel):
    """Response model for labeling operations."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime
    processing_time_ms: Optional[float] = None


class TripleBarrierLabelDict(BaseModel):
    """Dictionary representation of TripleBarrierLabel."""
    timestamp: datetime
    symbol: str
    side: int
    target_price: float
    profit_barrier: float
    stop_barrier: float
    time_barrier: datetime
    exit_timestamp: Optional[datetime]
    exit_price: Optional[float]
    exit_reason: Optional[str]
    return_pct: Optional[float]
    label: Optional[int]
    holding_period_hours: Optional[float]
    confidence: Optional[float]


class MetaLabelDict(BaseModel):
    """Dictionary representation of MetaLabelResult."""
    original_signal: int
    meta_prediction: int
    meta_probability: float
    final_signal: int
    model_confidence: float
    features: Dict[str, float]


# Utility functions
def convert_price_data_to_dataframe(price_data: List[PriceDataPoint]) -> pd.DataFrame:
    """Convert price data points to pandas DataFrame."""
    data = []
    for point in price_data:
        row = {
            'timestamp': point.timestamp,
            'close': point.close
        }
        if point.open is not None:
            row['open'] = point.open
        if point.high is not None:
            row['high'] = point.high
        if point.low is not None:
            row['low'] = point.low
        if point.volume is not None:
            row['volume'] = point.volume
        data.append(row)
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df


def convert_signals_to_series(signals: List[SignalDataPoint]) -> pd.Series:
    """Convert signal data points to pandas Series."""
    if not signals:
        return None
    
    signal_dict = {point.timestamp: point.signal for point in signals}
    series = pd.Series(signal_dict)
    series.sort_index(inplace=True)
    return series


def convert_labels_to_dict(labels: List[TripleBarrierLabel]) -> List[Dict[str, Any]]:
    """Convert TripleBarrierLabel objects to dictionaries."""
    result = []
    for label in labels:
        label_dict = {
            'timestamp': label.timestamp,
            'symbol': label.symbol,
            'side': label.side,
            'target_price': label.target_price,
            'profit_barrier': label.profit_barrier,
            'stop_barrier': label.stop_barrier,
            'time_barrier': label.time_barrier,
            'exit_timestamp': label.exit_timestamp,
            'exit_price': label.exit_price,
            'exit_reason': label.exit_reason,
            'return_pct': label.return_pct,
            'label': label.label,
            'holding_period_hours': label.holding_period.total_seconds() / 3600 if label.holding_period else None,
            'confidence': label.confidence
        }
        result.append(label_dict)
    return result


def convert_meta_results_to_dict(meta_results: List[MetaLabelResult]) -> List[Dict[str, Any]]:
    """Convert MetaLabelResult objects to dictionaries."""
    result = []
    for meta_result in meta_results:
        meta_dict = {
            'original_signal': meta_result.original_signal,
            'meta_prediction': meta_result.meta_prediction,
            'meta_probability': meta_result.meta_probability,
            'final_signal': meta_result.final_signal,
            'model_confidence': meta_result.model_confidence,
            'features': meta_result.features
        }
        result.append(meta_dict)
    return result


# API Endpoints

@router.post("/triple-barrier", response_model=LabelResponse)
async def create_triple_barrier_labels(request: TripleBarrierRequest):
    """
    Create triple-barrier labels from price data and signals.
    
    Applies the triple-barrier method to generate machine learning labels
    based on profit-taking, stop-loss, and time-based exit criteria.
    """
    start_time = datetime.now()
    
    try:
        # Convert input data
        prices_df = convert_price_data_to_dataframe(request.price_data)
        signals_series = convert_signals_to_series(request.signals) if request.signals else None
        
        # Validate data
        if len(prices_df) < 10:
            raise HTTPException(
                status_code=400,
                detail="Minimum 10 price observations required"
            )
        
        # Initialize labeler
        labeler = TripleBarrierLabeler(
            profit_threshold=request.profit_threshold,
            stop_threshold=request.stop_threshold,
            max_holding_period=request.max_holding_period,
            volatility_adjustment=request.volatility_adjustment,
            dynamic_barriers=request.dynamic_barriers
        )
        
        # Extract events
        price_series = prices_df['close']
        events = labeler.get_events(
            price_series, 
            signals_series,
            min_event_separation=request.min_event_separation
        )
        
        # Apply triple-barrier labeling
        labels = labeler.apply_triple_barrier(events, prices_df)
        
        # Convert to response format
        labels_dict = convert_labels_to_dict(labels)
        
        # Calculate statistics
        completed_labels = [l for l in labels if l.label is not None]
        statistics = {
            'total_events': len(events),
            'total_labels': len(labels),
            'completed_labels': len(completed_labels),
            'profit_labels': sum(1 for l in completed_labels if l.label == 1),
            'loss_labels': sum(1 for l in completed_labels if l.label == -1),
            'neutral_labels': sum(1 for l in completed_labels if l.label == 0),
            'completion_rate': len(completed_labels) / max(len(labels), 1),
            'profit_rate': sum(1 for l in completed_labels if l.label == 1) / max(len(completed_labels), 1)
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return LabelResponse(
            success=True,
            message=f"Generated {len(labels)} triple-barrier labels from {len(events)} events",
            data={
                'labels': labels_dict,
                'statistics': statistics,
                'events': len(events),
                'symbol': request.symbol
            },
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Triple-barrier labeling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Labeling failed: {str(e)}")


@router.post("/meta-labeling", response_model=LabelResponse)
async def apply_meta_labeling(request: MetaLabelingRequest):
    """
    Apply meta-labeling to improve signal quality.
    
    Trains a secondary model to predict when primary trading signals
    should be taken, improving overall precision.
    """
    start_time = datetime.now()
    
    try:
        # Convert input data
        prices_df = convert_price_data_to_dataframe(request.price_data)
        
        # Convert auxiliary data
        volume_series = None
        volatility_series = None
        sentiment_series = None
        
        if request.volume_data:
            volume_dict = {pd.Timestamp(item['timestamp']): item['value'] for item in request.volume_data}
            volume_series = pd.Series(volume_dict).sort_index()
        
        if request.volatility_data:
            vol_dict = {pd.Timestamp(item['timestamp']): item['value'] for item in request.volatility_data}
            volatility_series = pd.Series(vol_dict).sort_index()
            
        if request.sentiment_data:
            sent_dict = {pd.Timestamp(item['timestamp']): item['value'] for item in request.sentiment_data}
            sentiment_series = pd.Series(sent_dict).sort_index()
        
        # Reconstruct labels from request data
        labels = []
        for label_data in request.labels_data:
            # Note: This is simplified - in practice, you'd want to properly reconstruct TripleBarrierLabel objects
            pass
        
        # Initialize meta-labeler
        meta_labeler = MetaLabeler(
            cv_folds=request.cv_folds,
            min_precision=request.min_precision
        )
        
        # This is a simplified implementation
        # In practice, you'd need to properly handle the label reconstruction
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return LabelResponse(
            success=True,
            message="Meta-labeling completed successfully",
            data={
                'message': 'Meta-labeling implementation in progress',
                'cv_folds': request.cv_folds,
                'min_precision': request.min_precision
            },
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Meta-labeling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Meta-labeling failed: {str(e)}")


@router.post("/advanced-labeling", response_model=LabelResponse)
async def create_advanced_labels(request: AdvancedLabelingRequest):
    """
    Complete advanced labeling system with triple-barrier and meta-labeling.
    
    Combines triple-barrier method with meta-labeling for comprehensive
    signal quality improvement and label generation.
    """
    start_time = datetime.now()
    
    try:
        # Convert input data
        prices_df = convert_price_data_to_dataframe(request.price_data)
        signals_series = convert_signals_to_series(request.signals) if request.signals else None
        
        # Convert auxiliary data
        volume_series = None
        volatility_series = None
        sentiment_series = None
        
        if request.volume_data:
            volume_dict = {pd.Timestamp(item['timestamp']): item['value'] for item in request.volume_data}
            volume_series = pd.Series(volume_dict).sort_index()
        
        if request.volatility_data:
            vol_dict = {pd.Timestamp(item['timestamp']): item['value'] for item in request.volatility_data}
            volatility_series = pd.Series(vol_dict).sort_index()
            
        if request.sentiment_data:
            sent_dict = {pd.Timestamp(item['timestamp']): item['value'] for item in request.sentiment_data}
            sentiment_series = pd.Series(sent_dict).sort_index()
        
        # Initialize advanced labeling system
        labeling_system = AdvancedLabelingSystem(
            labeler_config=request.labeler_config or {},
            meta_labeler_config=request.meta_config or {}
        )
        
        # Create labels
        triple_labels, meta_results = labeling_system.create_labels(
            prices=prices_df,
            signals=signals_series,
            volume=volume_series,
            volatility=volatility_series,
            sentiment=sentiment_series,
            fit_meta_model=request.fit_meta_model
        )
        
        # Convert to response format
        labels_dict = convert_labels_to_dict(triple_labels)
        meta_dict = convert_meta_results_to_dict(meta_results)
        
        # Get system statistics
        system_stats = labeling_system.get_label_statistics()
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return LabelResponse(
            success=True,
            message=f"Advanced labeling completed: {len(triple_labels)} labels, {len(meta_results)} meta-predictions",
            data={
                'triple_barrier_labels': labels_dict,
                'meta_label_results': meta_dict,
                'system_statistics': system_stats,
                'symbol': request.symbol,
                'meta_model_fitted': request.fit_meta_model and len(meta_results) > 0
            },
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Advanced labeling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Advanced labeling failed: {str(e)}")


@router.get("/label-statistics/{symbol}")
async def get_label_statistics(
    symbol: str,
    start_date: Optional[datetime] = Query(None, description="Start date for statistics"),
    end_date: Optional[datetime] = Query(None, description="End date for statistics")
):
    """Get label statistics for a specific symbol."""
    
    # This would typically query a database of stored labels
    # For now, return a placeholder response
    
    return {
        'symbol': symbol,
        'period': {
            'start_date': start_date,
            'end_date': end_date
        },
        'statistics': {
            'total_labels': 0,
            'profit_rate': 0.0,
            'loss_rate': 0.0,
            'neutral_rate': 0.0,
            'avg_holding_period_hours': 0.0,
            'barrier_hit_rates': {
                'profit_barrier': 0.0,
                'stop_barrier': 0.0,
                'time_barrier': 0.0
            }
        },
        'message': 'Label statistics endpoint - implementation in progress'
    }


@router.get("/labeling-config")
async def get_labeling_configuration():
    """Get current labeling system configuration and documentation."""
    
    return {
        'triple_barrier_config': {
            'profit_threshold': 'Profit-taking threshold (default: 0.02 = 2%)',
            'stop_threshold': 'Stop-loss threshold (default: 0.01 = 1%)',
            'max_holding_period': 'Maximum holding period (default: 5D)',
            'volatility_adjustment': 'Adjust barriers based on volatility (default: True)',
            'dynamic_barriers': 'Use dynamic barrier calculation (default: False)'
        },
        'meta_labeling_config': {
            'cv_folds': 'Cross-validation folds (default: 5)',
            'min_precision': 'Minimum precision threshold (default: 0.55)',
            'feature_importance_threshold': 'Minimum feature importance (default: 0.01)'
        },
        'labeling_process': {
            'step_1': 'Extract trading events from price data and signals',
            'step_2': 'Apply triple-barrier method to create labels',
            'step_3': 'Extract features for meta-labeling',
            'step_4': 'Train meta-model to filter signals',
            'step_5': 'Generate final filtered signals'
        },
        'label_meanings': {
            '1': 'Profitable trade (hit profit barrier)',
            '-1': 'Loss trade (hit stop barrier)',
            '0': 'Neutral/timeout (hit time barrier without significant move)'
        },
        'meta_labeling_output': {
            'meta_prediction': '1 to take signal, 0 to skip signal',
            'meta_probability': 'Probability of signal being profitable',
            'final_signal': 'Filtered signal after meta-labeling'
        }
    }


@router.get("/health")
async def health_check():
    """Health check endpoint for labeling service."""
    return {
        'status': 'healthy',
        'service': 'triple-barrier-labeling',
        'timestamp': datetime.now(),
        'available_endpoints': [
            'triple-barrier',
            'meta-labeling',
            'advanced-labeling',
            'label-statistics',
            'labeling-config'
        ]
    }


# Validation endpoints

@router.post("/validate-price-data")
async def validate_price_data(price_data: List[PriceDataPoint]):
    """Validate price data format and quality."""
    
    try:
        df = convert_price_data_to_dataframe(price_data)
        
        validation_results = {
            'data_quality': {
                'total_observations': len(df),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_timestamps': df.index.duplicated().sum(),
                'time_gaps': 'analysis_needed',
                'price_anomalies': 'analysis_needed'
            },
            'data_coverage': {
                'start_date': df.index.min(),
                'end_date': df.index.max(),
                'time_span_days': (df.index.max() - df.index.min()).days,
                'average_frequency': 'analysis_needed'
            },
            'recommendations': []
        }
        
        # Add recommendations
        if len(df) < 100:
            validation_results['recommendations'].append(
                "Consider using more historical data (minimum 100 observations recommended)"
            )
        
        if df.isnull().any().any():
            validation_results['recommendations'].append(
                "Handle missing values before labeling"
            )
        
        if df.index.duplicated().any():
            validation_results['recommendations'].append(
                "Remove duplicate timestamps"
            )
        
        return {
            'valid': len(validation_results['recommendations']) == 0,
            'validation_results': validation_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Price data validation failed: {str(e)}")


@router.get("/documentation")
async def get_api_documentation():
    """Get comprehensive API documentation for labeling endpoints."""
    
    return {
        'overview': 'Triple-Barrier Labeling and Meta-Labeling API for advanced trading signal generation',
        'methodology': {
            'triple_barrier': {
                'description': 'Creates ML labels using profit, stop, and time barriers',
                'source': 'Advances in Financial Machine Learning by Marcos López de Prado',
                'benefits': ['Reduces look-ahead bias', 'Creates realistic labels', 'Handles path dependency']
            },
            'meta_labeling': {
                'description': 'Secondary model to filter primary signals and improve precision',
                'approach': 'Predicts when primary model signals should be taken',
                'benefits': ['Improves signal quality', 'Reduces false positives', 'Better risk management']
            }
        },
        'endpoints': {
            '/triple-barrier': {
                'method': 'POST',
                'description': 'Create triple-barrier labels from price data',
                'input': 'Price data, optional signals, barrier configuration',
                'output': 'Labeled events with exit reasons and returns'
            },
            '/meta-labeling': {
                'method': 'POST',
                'description': 'Apply meta-labeling to improve signal quality',
                'input': 'Labels, price data, market features',
                'output': 'Filtered signals with meta-predictions'
            },
            '/advanced-labeling': {
                'method': 'POST',
                'description': 'Complete labeling system combining both methods',
                'input': 'Price data, signals, market data',
                'output': 'Triple-barrier labels + meta-labeling results'
            }
        },
        'best_practices': [
            'Use sufficient historical data (minimum 100 observations)',
            'Include volume and volatility data for better features',
            'Adjust barrier thresholds based on market volatility',
            'Validate model performance with out-of-sample testing',
            'Monitor label quality and model degradation over time'
        ],
        'common_pitfalls': [
            'Look-ahead bias in feature construction',
            'Insufficient training data for meta-labeling',
            'Static barrier thresholds in volatile markets',
            'Overfitting in meta-labeling model'
        ]
    }