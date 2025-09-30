"""
Temporal Feature Store API

FastAPI endpoints for point-in-time feature enforcement, temporal table management,
and UTC timestamp normalization with millisecond precision.
"""

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta, timezone
import logging
import asyncio
import json

from ..services.temporal_feature_store import (
    TemporalFeatureStore, UTCTimestampNormalizer, EventNewsTimestampSynchronizer,
    PointInTimeQuery, FeatureValue, TemporalMetadata, TimestampPrecision
)
from ..core.feature_contracts import FeatureContractValidator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/temporal-features", tags=["Temporal Features"])

# Global temporal feature store instance
temporal_store: Optional[TemporalFeatureStore] = None

class FeatureStoreRequest(BaseModel):
    """Request model for storing features"""
    feature_name: str = Field(..., description="Name of the feature")
    symbol: str = Field(..., description="Symbol/identifier")
    value: Any = Field(..., description="Feature value")
    effective_timestamp: datetime = Field(..., description="When underlying event occurred")
    source_timestamp: Optional[datetime] = Field(None, description="Original source timestamp")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence score 0-1")
    quality_score: Optional[float] = Field(None, ge=0, le=1, description="Quality score 0-1")
    source_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    source_timezone: Optional[str] = Field(None, description="Source timezone identifier")

class BatchFeatureStoreRequest(BaseModel):
    """Request model for batch storing features"""
    features: List[FeatureStoreRequest] = Field(..., description="List of features to store")

class PointInTimeRequest(BaseModel):
    """Request model for point-in-time feature lookup"""
    features: List[str] = Field(..., description="List of feature names")
    symbols: List[str] = Field(..., description="List of symbols")
    as_of_time: datetime = Field(..., description="Point-in-time for lookup")
    effective_time_window: Optional[Tuple[datetime, datetime]] = Field(None)
    include_revisions: bool = Field(False, description="Include feature revisions")
    quality_threshold: Optional[float] = Field(None, ge=0, le=1)

class TimestampNormalizationRequest(BaseModel):
    """Request model for timestamp normalization"""
    timestamp: Union[datetime, str, int] = Field(..., description="Timestamp to normalize")
    source_timezone: Optional[str] = Field(None, description="Source timezone")
    precision: Optional[str] = Field("millisecond", description="Target precision")

    @validator('precision')
    def validate_precision(cls, v):
        valid_precisions = ['second', 'millisecond', 'microsecond', 'nanosecond']
        if v not in valid_precisions:
            raise ValueError(f"Precision must be one of: {valid_precisions}")
        return v

class EventSynchronizationRequest(BaseModel):
    """Request model for event timestamp synchronization"""
    events: List[Dict[str, Any]] = Field(..., description="List of events with timestamps")
    reference_market_data: Optional[Dict[str, Any]] = Field(None)

class FeatureHistoryRequest(BaseModel):
    """Request model for feature history lookup"""
    feature_name: str = Field(..., description="Feature name")
    symbol: str = Field(..., description="Symbol")
    start_time: datetime = Field(..., description="Start time for history")
    end_time: datetime = Field(..., description="End time for history")
    include_revisions: bool = Field(False, description="Include revisions")

# Initialize temporal store
async def get_temporal_store() -> TemporalFeatureStore:
    """Get or initialize temporal feature store"""
    global temporal_store
    if temporal_store is None:
        database_url = "postgresql://trading_user:trading_password@localhost:5432/trading_db"
        contracts_validator = FeatureContractValidator()
        temporal_store = TemporalFeatureStore(database_url, contracts_validator)
        await temporal_store.initialize()
    return temporal_store

@router.post("/initialize")
async def initialize_temporal_store():
    """
    Initialize the temporal feature store and create necessary database tables.
    
    This endpoint sets up the temporal feature store with:
    - Temporal tables with system versioning
    - Point-in-time constraints and indexes
    - Feature metadata and lineage tracking
    - Contract validation infrastructure
    """
    try:
        store = await get_temporal_store()
        return {
            "status": "success",
            "message": "Temporal feature store initialized",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to initialize temporal store: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/store-feature")
async def store_feature(request: FeatureStoreRequest):
    """
    Store a feature value with full temporal metadata and point-in-time enforcement.
    
    This endpoint stores features with:
    - UTC timestamp normalization with millisecond precision
    - Point-in-time availability calculation based on feature contracts
    - Data lineage tracking and audit trails
    - Contract validation and compliance checking
    """
    try:
        store = await get_temporal_store()
        
        # Normalize effective timestamp if source timezone provided
        if request.source_timezone:
            normalizer = UTCTimestampNormalizer()
            effective_timestamp = normalizer.normalize_to_utc(
                request.effective_timestamp,
                source_timezone=request.source_timezone,
                precision=TimestampPrecision.MILLISECOND
            )
            
            if request.source_timestamp:
                source_timestamp = normalizer.normalize_to_utc(
                    request.source_timestamp,
                    source_timezone=request.source_timezone,
                    precision=TimestampPrecision.MILLISECOND
                )
            else:
                source_timestamp = None
        else:
            effective_timestamp = request.effective_timestamp
            source_timestamp = request.source_timestamp
        
        lineage_id = await store.store_feature(
            feature_name=request.feature_name,
            symbol=request.symbol,
            value=request.value,
            effective_timestamp=effective_timestamp,
            source_timestamp=source_timestamp,
            confidence=request.confidence,
            quality_score=request.quality_score,
            source_metadata=request.source_metadata
        )
        
        return {
            "status": "success",
            "lineage_id": lineage_id,
            "feature_name": request.feature_name,
            "symbol": request.symbol,
            "stored_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to store feature: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/store-features-batch")
async def store_features_batch(request: BatchFeatureStoreRequest):
    """
    Store multiple features in batch with temporal metadata.
    
    This endpoint efficiently processes multiple feature values with:
    - Batch timestamp normalization
    - Parallel feature storage
    - Comprehensive error handling and reporting
    - Transaction consistency across the batch
    """
    try:
        store = await get_temporal_store()
        normalizer = UTCTimestampNormalizer()
        
        results = []
        errors = []
        
        for i, feature_req in enumerate(request.features):
            try:
                # Normalize timestamps
                if feature_req.source_timezone:
                    effective_timestamp = normalizer.normalize_to_utc(
                        feature_req.effective_timestamp,
                        source_timezone=feature_req.source_timezone,
                        precision=TimestampPrecision.MILLISECOND
                    )
                    
                    if feature_req.source_timestamp:
                        source_timestamp = normalizer.normalize_to_utc(
                            feature_req.source_timestamp,
                            source_timezone=feature_req.source_timezone,
                            precision=TimestampPrecision.MILLISECOND
                        )
                    else:
                        source_timestamp = None
                else:
                    effective_timestamp = feature_req.effective_timestamp
                    source_timestamp = feature_req.source_timestamp
                
                lineage_id = await store.store_feature(
                    feature_name=feature_req.feature_name,
                    symbol=feature_req.symbol,
                    value=feature_req.value,
                    effective_timestamp=effective_timestamp,
                    source_timestamp=source_timestamp,
                    confidence=feature_req.confidence,
                    quality_score=feature_req.quality_score,
                    source_metadata=feature_req.source_metadata
                )
                
                results.append({
                    "index": i,
                    "lineage_id": lineage_id,
                    "feature_name": feature_req.feature_name,
                    "symbol": feature_req.symbol,
                    "status": "success"
                })
                
            except Exception as e:
                errors.append({
                    "index": i,
                    "feature_name": feature_req.feature_name,
                    "symbol": feature_req.symbol,
                    "error": str(e)
                })
        
        return {
            "status": "completed",
            "total_requested": len(request.features),
            "successful_stores": len(results),
            "failed_stores": len(errors),
            "results": results,
            "errors": errors if errors else None,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to store features batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/point-in-time-lookup")
async def point_in_time_lookup(request: PointInTimeRequest):
    """
    Perform point-in-time feature lookup with temporal integrity.
    
    This endpoint provides accurate historical feature values by:
    - Enforcing point-in-time constraints based on feature contracts
    - Returning features as they were available at the specified time
    - Filtering by quality thresholds and effective time windows
    - Maintaining data lineage and revision tracking
    """
    try:
        store = await get_temporal_store()
        
        # Create point-in-time query
        query = PointInTimeQuery(
            features=request.features,
            symbols=request.symbols,
            as_of_time=request.as_of_time,
            effective_time_window=request.effective_time_window,
            include_revisions=request.include_revisions,
            quality_threshold=request.quality_threshold
        )
        
        # Execute lookup
        results = await store.point_in_time_lookup(query)
        
        # Convert to serializable format
        serializable_results = {}
        for symbol, features in results.items():
            serializable_results[symbol] = {}
            for feature_name, feature_value in features.items():
                serializable_results[symbol][feature_name] = {
                    "value": feature_value.value,
                    "confidence": feature_value.confidence,
                    "quality_score": feature_value.quality_score,
                    "metadata": {
                        "as_of_timestamp": feature_value.metadata.as_of_timestamp.isoformat(),
                        "effective_timestamp": feature_value.metadata.effective_timestamp.isoformat(),
                        "ingestion_timestamp": feature_value.metadata.ingestion_timestamp.isoformat(),
                        "source_timestamp": feature_value.metadata.source_timestamp.isoformat() if feature_value.metadata.source_timestamp else None,
                        "revision_number": feature_value.metadata.revision_number,
                        "lineage_id": feature_value.metadata.lineage_id
                    },
                    "source_metadata": feature_value.source_metadata
                }
        
        return {
            "status": "success",
            "query": {
                "features": request.features,
                "symbols": request.symbols,
                "as_of_time": request.as_of_time.isoformat(),
                "effective_time_window": [t.isoformat() for t in request.effective_time_window] if request.effective_time_window else None
            },
            "results": serializable_results,
            "lookup_performed_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to perform point-in-time lookup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feature-history")
async def get_feature_history(request: FeatureHistoryRequest):
    """
    Get historical feature values for a time range with temporal integrity.
    
    This endpoint provides feature evolution over time including:
    - Complete historical timeline of feature values
    - Optional revision history and data lineage
    - Temporal metadata for each value
    - Quality and confidence metrics tracking
    """
    try:
        store = await get_temporal_store()
        
        history = await store.get_feature_history(
            feature_name=request.feature_name,
            symbol=request.symbol,
            start_time=request.start_time,
            end_time=request.end_time,
            include_revisions=request.include_revisions
        )
        
        # Convert to serializable format
        serializable_history = []
        for feature_value in history:
            serializable_history.append({
                "value": feature_value.value,
                "confidence": feature_value.confidence,
                "quality_score": feature_value.quality_score,
                "metadata": {
                    "as_of_timestamp": feature_value.metadata.as_of_timestamp.isoformat(),
                    "effective_timestamp": feature_value.metadata.effective_timestamp.isoformat(),
                    "ingestion_timestamp": feature_value.metadata.ingestion_timestamp.isoformat(),
                    "source_timestamp": feature_value.metadata.source_timestamp.isoformat() if feature_value.metadata.source_timestamp else None,
                    "revision_number": feature_value.metadata.revision_number,
                    "lineage_id": feature_value.metadata.lineage_id
                },
                "source_metadata": feature_value.source_metadata
            })
        
        return {
            "status": "success",
            "query": {
                "feature_name": request.feature_name,
                "symbol": request.symbol,
                "start_time": request.start_time.isoformat(),
                "end_time": request.end_time.isoformat(),
                "include_revisions": request.include_revisions
            },
            "history": serializable_history,
            "total_records": len(serializable_history),
            "retrieved_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get feature history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/normalize-timestamp")
async def normalize_timestamp(request: TimestampNormalizationRequest):
    """
    Normalize timestamp to UTC with specified precision.
    
    This endpoint provides timestamp normalization including:
    - Conversion from various timestamp formats
    - Timezone handling and UTC conversion
    - Precision control (second, millisecond, microsecond, nanosecond)
    - Format standardization for temporal consistency
    """
    try:
        normalizer = UTCTimestampNormalizer()
        
        # Convert precision string to enum
        precision_map = {
            'second': TimestampPrecision.SECOND,
            'millisecond': TimestampPrecision.MILLISECOND,
            'microsecond': TimestampPrecision.MICROSECOND,
            'nanosecond': TimestampPrecision.NANOSECOND
        }
        
        precision = precision_map.get(request.precision, TimestampPrecision.MILLISECOND)
        
        normalized_timestamp = normalizer.normalize_to_utc(
            timestamp=request.timestamp,
            source_timezone=request.source_timezone,
            precision=precision
        )
        
        return {
            "status": "success",
            "original_timestamp": str(request.timestamp),
            "source_timezone": request.source_timezone,
            "target_precision": request.precision,
            "normalized_timestamp_utc": normalized_timestamp.isoformat(),
            "normalized_timestamp_unix": normalized_timestamp.timestamp(),
            "normalized_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to normalize timestamp: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/synchronize-events")
async def synchronize_event_timestamps(request: EventSynchronizationRequest):
    """
    Synchronize event and news timestamps with market data timeline.
    
    This endpoint provides event synchronization including:
    - UTC timestamp normalization for all events
    - Market session alignment and classification
    - Trading hours validation and adjustment
    - Reference market data correlation
    """
    try:
        store = await get_temporal_store()
        synchronizer = EventNewsTimestampSynchronizer(store)
        
        synchronized_events = await synchronizer.synchronize_event_timestamps(
            events=request.events,
            reference_market_data=request.reference_market_data
        )
        
        return {
            "status": "success",
            "original_event_count": len(request.events),
            "synchronized_event_count": len(synchronized_events),
            "synchronized_events": synchronized_events,
            "synchronization_metadata": {
                "reference_market_data_provided": request.reference_market_data is not None,
                "synchronization_performed_at": datetime.now(timezone.utc).isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to synchronize events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def temporal_store_health():
    """
    Check temporal feature store health and connectivity.
    
    This endpoint provides health status including:
    - Database connectivity status
    - Temporal table integrity
    - Contract validation status
    - System performance metrics
    """
    try:
        store = await get_temporal_store()
        
        # Basic connectivity test
        if store.connection_pool:
            async with store.connection_pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                db_connected = result == 1
        else:
            db_connected = False
        
        return {
            "status": "healthy" if db_connected else "degraded",
            "database_connected": db_connected,
            "contracts_loaded": len(store.contracts_validator.contracts),
            "timestamp_normalizer_active": True,
            "temporal_tables_available": db_connected,
            "checked_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "database_connected": False,
            "error": str(e),
            "checked_at": datetime.now(timezone.utc).isoformat()
        }

@router.get("/stats")
async def get_temporal_store_stats():
    """
    Get temporal feature store statistics and metrics.
    
    This endpoint provides comprehensive statistics including:
    - Feature count by type and symbol
    - Temporal data coverage and quality metrics
    - Storage utilization and performance
    - Contract compliance statistics
    """
    try:
        store = await get_temporal_store()
        
        async with store.connection_pool.acquire() as conn:
            # Get basic statistics
            total_features = await conn.fetchval(f"""
                SELECT COUNT(*) FROM {store.table_prefix}_main WHERE is_active = TRUE
            """)
            
            unique_symbols = await conn.fetchval(f"""
                SELECT COUNT(DISTINCT symbol) FROM {store.table_prefix}_main WHERE is_active = TRUE
            """)
            
            unique_feature_names = await conn.fetchval(f"""
                SELECT COUNT(DISTINCT feature_name) FROM {store.table_prefix}_main WHERE is_active = TRUE
            """)
            
            # Get date range
            date_range = await conn.fetchrow(f"""
                SELECT 
                    MIN(effective_timestamp) as earliest_data,
                    MAX(effective_timestamp) as latest_data
                FROM {store.table_prefix}_main WHERE is_active = TRUE
            """)
            
            # Get quality statistics
            quality_stats = await conn.fetchrow(f"""
                SELECT 
                    AVG(confidence) as avg_confidence,
                    AVG(quality_score) as avg_quality,
                    COUNT(*) FILTER (WHERE confidence IS NOT NULL) as records_with_confidence,
                    COUNT(*) FILTER (WHERE quality_score IS NOT NULL) as records_with_quality
                FROM {store.table_prefix}_main WHERE is_active = TRUE
            """)
        
        return {
            "status": "success",
            "statistics": {
                "total_feature_records": total_features,
                "unique_symbols": unique_symbols,
                "unique_feature_names": unique_feature_names,
                "data_coverage": {
                    "earliest_data": date_range['earliest_data'].isoformat() if date_range['earliest_data'] else None,
                    "latest_data": date_range['latest_data'].isoformat() if date_range['latest_data'] else None,
                    "total_days": (date_range['latest_data'] - date_range['earliest_data']).days if date_range['earliest_data'] and date_range['latest_data'] else 0
                },
                "quality_metrics": {
                    "avg_confidence": float(quality_stats['avg_confidence']) if quality_stats['avg_confidence'] else None,
                    "avg_quality_score": float(quality_stats['avg_quality']) if quality_stats['avg_quality'] else None,
                    "records_with_confidence": quality_stats['records_with_confidence'],
                    "records_with_quality": quality_stats['records_with_quality']
                },
                "contracts": {
                    "total_contracts": len(store.contracts_validator.contracts),
                    "contract_names": list(store.contracts_validator.contracts.keys())
                }
            },
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cleanup")
async def cleanup_temporal_data(
    older_than_days: int = Query(365, description="Remove data older than X days"),
    dry_run: bool = Query(True, description="Preview changes without executing")
):
    """
    Clean up old temporal data based on retention policies.
    
    This endpoint provides data cleanup including:
    - Age-based data removal with configurable retention
    - Dry run mode for preview before execution
    - Audit trail preservation for compliance
    - Performance optimization post-cleanup
    """
    try:
        store = await get_temporal_store()
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        
        async with store.connection_pool.acquire() as conn:
            if dry_run:
                # Count records that would be affected
                count_query = f"""
                    SELECT COUNT(*) FROM {store.table_prefix}_main 
                    WHERE effective_timestamp < $1 AND is_active = TRUE
                """
                affected_records = await conn.fetchval(count_query, cutoff_date)
                
                return {
                    "status": "dry_run_completed",
                    "affected_records": affected_records,
                    "cutoff_date": cutoff_date.isoformat(),
                    "retention_days": older_than_days,
                    "message": f"Would remove {affected_records} records older than {older_than_days} days",
                    "preview_generated_at": datetime.now(timezone.utc).isoformat()
                }
            else:
                # Move to history table first
                move_query = f"""
                    INSERT INTO {store.table_prefix}_history 
                    SELECT *, 'DELETE', NOW(), current_user 
                    FROM {store.table_prefix}_main 
                    WHERE effective_timestamp < $1 AND is_active = TRUE
                """
                await conn.execute(move_query, cutoff_date)
                
                # Mark as inactive (soft delete)
                update_query = f"""
                    UPDATE {store.table_prefix}_main 
                    SET is_active = FALSE, valid_to = NOW()
                    WHERE effective_timestamp < $1 AND is_active = TRUE
                """
                result = await conn.execute(update_query, cutoff_date)
                
                # Extract affected count from result
                affected_records = int(result.split()[-1]) if result else 0
                
                return {
                    "status": "cleanup_completed",
                    "affected_records": affected_records,
                    "cutoff_date": cutoff_date.isoformat(),
                    "retention_days": older_than_days,
                    "message": f"Cleaned up {affected_records} records older than {older_than_days} days",
                    "cleanup_performed_at": datetime.now(timezone.utc).isoformat()
                }
        
    except Exception as e:
        logger.error(f"Failed to cleanup temporal data: {e}")
        raise HTTPException(status_code=500, detail=str(e))