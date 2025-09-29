"""Bulk Ingestion Service for Historical Event Backlogs

High-performance bulk data ingestion service for processing large historical
event datasets with optimized database operations and parallel processing.
"""

import asyncio
import logging
import os
import tempfile
import csv
import json
import gzip
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib

from sqlalchemy import select, insert, update, delete, text, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ..models import EventORM, EventHeadlineORM
from .event_categorizer import EventCategorizer
from .event_impact import EventImpactScorer
from .event_enrichment import EventEnrichmentService
from .event_cache import EventCacheService

logger = logging.getLogger(__name__)


class IngestionFormat(str, Enum):
    """Supported ingestion file formats."""
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"  # JSON Lines
    PARQUET = "parquet"
    EXCEL = "excel"
    SQL = "sql"


class IngestionMode(str, Enum):
    """Data ingestion modes."""
    INSERT_ONLY = "insert_only"  # Only insert new records
    UPSERT = "upsert"  # Insert new, update existing
    REPLACE = "replace"  # Delete existing and insert new
    APPEND = "append"  # Append all records regardless


class ValidationLevel(str, Enum):
    """Data validation levels."""
    STRICT = "strict"  # Fail on any validation error
    PERMISSIVE = "permissive"  # Skip invalid records, continue processing
    NONE = "none"  # No validation, fastest processing


@dataclass
class IngestionConfig:
    """Configuration for bulk ingestion operations."""
    batch_size: int = 1000
    max_workers: int = 4
    validation_level: ValidationLevel = ValidationLevel.PERMISSIVE
    mode: IngestionMode = IngestionMode.UPSERT
    deduplicate: bool = True
    auto_categorize: bool = True
    auto_enrich: bool = False  # Expensive operation, default off
    skip_cache_invalidation: bool = False  # For performance during bulk ops
    progress_callback: Optional[callable] = None
    error_threshold: float = 0.1  # Stop if >10% errors


@dataclass
class IngestionStats:
    """Statistics for bulk ingestion operations."""
    total_records: int = 0
    processed_records: int = 0
    inserted_records: int = 0
    updated_records: int = 0
    skipped_records: int = 0
    failed_records: int = 0
    duplicate_records: int = 0
    validation_errors: int = 0
    processing_time_seconds: float = 0.0
    throughput_records_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    batch_count: int = 0


@dataclass
class IngestionResult:
    """Result of bulk ingestion operation."""
    operation_id: str
    status: str  # success, partial, failed
    stats: IngestionStats
    errors: List[Dict[str, Any]]
    warnings: List[str]
    file_info: Dict[str, Any]
    metadata: Dict[str, Any]


class BulkIngestionService:
    """High-performance bulk ingestion service for historical event data."""
    
    def __init__(
        self,
        session_factory,
        categorizer: Optional[EventCategorizer] = None,
        impact_scorer: Optional[EventImpactScorer] = None,
        enrichment_service: Optional[EventEnrichmentService] = None,
        cache_service: Optional[EventCacheService] = None
    ):
        self.session_factory = session_factory
        self.categorizer = categorizer
        self.impact_scorer = impact_scorer
        self.enrichment_service = enrichment_service
        self.cache_service = cache_service
        
        # Configuration
        self.enabled = os.getenv("BULK_INGESTION_ENABLED", "true").lower() == "true"
        self.max_file_size_mb = int(os.getenv("BULK_INGESTION_MAX_FILE_SIZE_MB", "500"))
        self.temp_dir = os.getenv("BULK_INGESTION_TEMP_DIR", tempfile.gettempdir())
        self.default_batch_size = int(os.getenv("BULK_INGESTION_BATCH_SIZE", "1000"))
        self.max_workers = int(os.getenv("BULK_INGESTION_MAX_WORKERS", "4"))
        self.connection_pool_size = int(os.getenv("BULK_INGESTION_POOL_SIZE", "20"))
        
        # Performance settings
        self.memory_limit_mb = int(os.getenv("BULK_INGESTION_MEMORY_LIMIT_MB", "1024"))
        self.chunk_size = int(os.getenv("BULK_INGESTION_CHUNK_SIZE", "10000"))
        self.vacuum_threshold = int(os.getenv("BULK_INGESTION_VACUUM_THRESHOLD", "100000"))
        
        # Active operations tracking
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self._operation_lock = asyncio.Lock()
        
        logger.info(f"BulkIngestionService initialized (enabled={self.enabled})")
    
    async def start(self):
        """Start the bulk ingestion service."""
        if not self.enabled:
            logger.info("Bulk ingestion service disabled")
            return
        
        logger.info("Bulk ingestion service started")
        
        # Ensure temp directory exists
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
    
    async def stop(self):
        """Stop the bulk ingestion service."""
        logger.info("Stopping bulk ingestion service")
        
        # Wait for active operations to complete
        while self.active_operations:
            logger.info(f"Waiting for {len(self.active_operations)} operations to complete...")
            await asyncio.sleep(1)
        
        logger.info("Bulk ingestion service stopped")
    
    def _generate_operation_id(self, file_info: Dict[str, Any]) -> str:
        """Generate unique operation ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_hash = hashlib.md5(str(file_info).encode()).hexdigest()[:8]
        return f"bulk_{timestamp}_{file_hash}"
    
    async def _validate_record(
        self, 
        record: Dict[str, Any], 
        validation_level: ValidationLevel
    ) -> Tuple[bool, Optional[str]]:
        """Validate a single record."""
        if validation_level == ValidationLevel.NONE:
            return True, None
        
        required_fields = ["symbol", "title", "scheduled_at"]
        
        # Check required fields
        for field in required_fields:
            if field not in record or not record[field]:
                if validation_level == ValidationLevel.STRICT:
                    return False, f"Missing required field: {field}"
                else:
                    return False, f"Missing required field: {field}"
        
        # Validate symbol format
        symbol = record["symbol"]
        if not isinstance(symbol, str) or len(symbol) > 32:
            error = f"Invalid symbol format: {symbol}"
            if validation_level == ValidationLevel.STRICT:
                return False, error
            else:
                return False, error
        
        # Validate scheduled_at
        try:
            if isinstance(record["scheduled_at"], str):
                datetime.fromisoformat(record["scheduled_at"].replace('Z', '+00:00'))
            elif not isinstance(record["scheduled_at"], datetime):
                return False, "scheduled_at must be datetime or ISO string"
        except Exception as e:
            error = f"Invalid scheduled_at format: {e}"
            if validation_level == ValidationLevel.STRICT:
                return False, error
            else:
                return False, error
        
        return True, None
    
    async def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and standardize record format."""
        normalized = {}
        
        # Required fields
        normalized["symbol"] = str(record["symbol"]).upper().strip()
        normalized["title"] = str(record["title"]).strip()
        
        # Handle datetime conversion
        scheduled_at = record["scheduled_at"]
        if isinstance(scheduled_at, str):
            # Parse ISO string
            normalized["scheduled_at"] = datetime.fromisoformat(
                scheduled_at.replace('Z', '+00:00')
            )
        else:
            normalized["scheduled_at"] = scheduled_at
        
        # Optional fields with defaults
        normalized["category"] = record.get("category", "other")
        normalized["description"] = record.get("description", "")
        normalized["status"] = record.get("status", "scheduled")
        normalized["timezone"] = record.get("timezone")
        normalized["source"] = record.get("source", "bulk_import")
        normalized["external_id"] = record.get("external_id")
        normalized["impact_score"] = record.get("impact_score")
        
        # Metadata handling
        metadata = record.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {"raw_metadata": metadata}
        normalized["metadata_json"] = metadata
        
        # Add bulk import metadata
        normalized["metadata_json"]["bulk_import"] = {
            "imported_at": datetime.utcnow().isoformat(),
            "import_source": "bulk_ingestion_service"
        }
        
        return normalized
    
    async def _process_batch(
        self,
        session: AsyncSession,
        batch: List[Dict[str, Any]],
        config: IngestionConfig,
        stats: IngestionStats
    ) -> None:
        """Process a batch of records."""
        try:
            if config.mode == IngestionMode.UPSERT:
                await self._upsert_batch(session, batch, config, stats)
            elif config.mode == IngestionMode.INSERT_ONLY:
                await self._insert_batch(session, batch, config, stats)
            elif config.mode == IngestionMode.REPLACE:
                await self._replace_batch(session, batch, config, stats)
            else:  # APPEND
                await self._append_batch(session, batch, config, stats)
            
            stats.batch_count += 1
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            stats.failed_records += len(batch)
            raise
    
    async def _upsert_batch(
        self,
        session: AsyncSession,
        batch: List[Dict[str, Any]],
        config: IngestionConfig,
        stats: IngestionStats
    ) -> None:
        """Upsert batch using PostgreSQL ON CONFLICT."""
        
        # Group by conflict key for efficient upserts
        insert_data = []
        update_data = []
        
        for record in batch:
            # Apply categorization if enabled
            if config.auto_categorize and self.categorizer:
                category_result = self.categorizer.categorize(
                    raw_category=record.get("category"),
                    title=record["title"],
                    description=record.get("description", ""),
                    metadata=record.get("metadata_json", {})
                )
                record["category"] = category_result.category
                
                # Update metadata with categorization info
                metadata = record.get("metadata_json", {})
                metadata["classification"] = {
                    "canonical_category": category_result.category,
                    "confidence": category_result.confidence,
                    "matched_keywords": category_result.matched_keywords,
                    "source": "bulk_import_categorization"
                }
                record["metadata_json"] = metadata
            
            # Apply impact scoring if enabled
            if config.auto_categorize and self.impact_scorer:
                impact_score = self.impact_scorer.calculate_impact_score({
                    "symbol": record["symbol"],
                    "category": record["category"],
                    "title": record["title"],
                    "description": record.get("description", ""),
                    "metadata": record.get("metadata_json", {})
                })
                if impact_score:
                    record["impact_score"] = impact_score
            
            insert_data.append(record)
        
        if insert_data:
            # Use PostgreSQL UPSERT (INSERT ... ON CONFLICT)
            stmt = pg_insert(EventORM).values(insert_data)
            
            # Define conflict resolution
            conflict_stmt = stmt.on_conflict_do_update(
                index_elements=["source", "external_id"],
                set_={
                    "title": stmt.excluded.title,
                    "category": stmt.excluded.category,
                    "description": stmt.excluded.description,
                    "metadata_json": stmt.excluded.metadata_json,
                    "impact_score": stmt.excluded.impact_score,
                    "updated_at": datetime.utcnow()
                }
            )
            
            # Execute upsert
            result = await session.execute(conflict_stmt)
            await session.commit()
            
            # Update statistics (approximate)
            affected_rows = result.rowcount
            stats.processed_records += len(batch)
            stats.inserted_records += affected_rows  # Some may be updates
    
    async def _insert_batch(
        self,
        session: AsyncSession,
        batch: List[Dict[str, Any]],
        config: IngestionConfig,
        stats: IngestionStats
    ) -> None:
        """Insert batch with duplicate handling."""
        try:
            stmt = insert(EventORM).values(batch)
            result = await session.execute(stmt)
            await session.commit()
            
            stats.processed_records += len(batch)
            stats.inserted_records += result.rowcount
            
        except IntegrityError as e:
            # Handle duplicates
            await session.rollback()
            
            if config.validation_level != ValidationLevel.STRICT:
                # Insert one by one to handle duplicates
                for record in batch:
                    try:
                        stmt = insert(EventORM).values(record)
                        await session.execute(stmt)
                        await session.commit()
                        stats.inserted_records += 1
                    except IntegrityError:
                        stats.duplicate_records += 1
                        await session.rollback()
                    
                    stats.processed_records += 1
            else:
                raise
    
    async def _replace_batch(
        self,
        session: AsyncSession,
        batch: List[Dict[str, Any]],
        config: IngestionConfig,
        stats: IngestionStats
    ) -> None:
        """Replace existing records with new data."""
        
        # Get symbols in batch
        symbols = set(record["symbol"] for record in batch)
        
        # Delete existing records for these symbols
        delete_stmt = delete(EventORM).where(EventORM.symbol.in_(symbols))
        delete_result = await session.execute(delete_stmt)
        
        # Insert new records
        insert_stmt = insert(EventORM).values(batch)
        insert_result = await session.execute(insert_stmt)
        await session.commit()
        
        stats.processed_records += len(batch)
        stats.inserted_records += insert_result.rowcount
    
    async def _append_batch(
        self,
        session: AsyncSession,
        batch: List[Dict[str, Any]],
        config: IngestionConfig,
        stats: IngestionStats
    ) -> None:
        """Append all records without conflict checking."""
        
        # Remove unique constraints temporarily by not setting external_id
        for record in batch:
            if not record.get("external_id"):
                record["external_id"] = f"bulk_{datetime.utcnow().timestamp()}_{hash(str(record))}"
        
        stmt = insert(EventORM).values(batch)
        result = await session.execute(stmt)
        await session.commit()
        
        stats.processed_records += len(batch)
        stats.inserted_records += result.rowcount
    
    async def _read_csv_file(self, file_path: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Read CSV file and yield records."""
        with open(file_path, 'r', encoding='utf-8') as file:
            # Auto-detect delimiter
            sample = file.read(1024)
            file.seek(0)
            
            delimiter = ','
            if '\t' in sample:
                delimiter = '\t'
            elif ';' in sample:
                delimiter = ';'
            
            reader = csv.DictReader(file, delimiter=delimiter)
            
            for row in reader:
                # Convert empty strings to None
                cleaned_row = {k: v if v != '' else None for k, v in row.items()}
                yield cleaned_row
    
    async def _read_json_file(self, file_path: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Read JSON file and yield records."""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            if isinstance(data, list):
                for record in data:
                    yield record
            elif isinstance(data, dict):
                # Handle single record or records under a key
                if "events" in data:
                    for record in data["events"]:
                        yield record
                elif "data" in data:
                    for record in data["data"]:
                        yield record
                else:
                    yield data
    
    async def _read_jsonl_file(self, file_path: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Read JSON Lines file and yield records."""
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        yield record
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON line: {line[:100]}... Error: {e}")
    
    async def ingest_file(
        self,
        file_path: str,
        format_type: IngestionFormat,
        config: Optional[IngestionConfig] = None
    ) -> IngestionResult:
        """Ingest data from a file."""
        
        if not self.enabled:
            raise RuntimeError("Bulk ingestion service is disabled")
        
        # Default configuration
        if config is None:
            config = IngestionConfig()
        
        # Validate file
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size_mb = file_path.stat().st_size / 1024 / 1024
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB")
        
        # Generate operation ID
        file_info = {
            "path": str(file_path),
            "size_mb": file_size_mb,
            "format": format_type.value,
            "modified": file_path.stat().st_mtime
        }
        operation_id = self._generate_operation_id(file_info)
        
        # Track operation
        async with self._operation_lock:
            self.active_operations[operation_id] = {
                "status": "starting",
                "started_at": datetime.utcnow(),
                "file_info": file_info,
                "config": asdict(config)
            }
        
        # Initialize statistics
        stats = IngestionStats()
        errors = []
        warnings = []
        start_time = datetime.utcnow()
        
        try:
            # Update operation status
            self.active_operations[operation_id]["status"] = "reading"
            
            # Select appropriate reader
            if format_type == IngestionFormat.CSV:
                record_generator = self._read_csv_file(str(file_path))
            elif format_type == IngestionFormat.JSON:
                record_generator = self._read_json_file(str(file_path))
            elif format_type == IngestionFormat.JSONL:
                record_generator = self._read_jsonl_file(str(file_path))
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            # Process records in batches
            batch = []
            record_count = 0
            error_count = 0
            
            async with self.session_factory() as session:
                async for raw_record in record_generator:
                    record_count += 1
                    stats.total_records += 1
                    
                    # Validate record
                    is_valid, validation_error = await self._validate_record(
                        raw_record, config.validation_level
                    )
                    
                    if not is_valid:
                        error_count += 1
                        stats.validation_errors += 1
                        errors.append({
                            "record_number": record_count,
                            "error": validation_error,
                            "record": raw_record
                        })
                        
                        # Check error threshold
                        error_rate = error_count / record_count
                        if error_rate > config.error_threshold:
                            raise RuntimeError(
                                f"Error rate {error_rate:.2%} exceeds threshold {config.error_threshold:.2%}"
                            )
                        
                        if config.validation_level == ValidationLevel.STRICT:
                            break
                        else:
                            continue
                    
                    # Normalize record
                    try:
                        normalized_record = await self._normalize_record(raw_record)
                        batch.append(normalized_record)
                    except Exception as e:
                        error_count += 1
                        stats.failed_records += 1
                        errors.append({
                            "record_number": record_count,
                            "error": f"Normalization failed: {e}",
                            "record": raw_record
                        })
                        continue
                    
                    # Process batch when full
                    if len(batch) >= config.batch_size:
                        self.active_operations[operation_id]["status"] = f"processing_batch_{stats.batch_count + 1}"
                        
                        try:
                            await self._process_batch(session, batch, config, stats)
                        except Exception as e:
                            logger.error(f"Batch processing failed: {e}")
                            errors.append({
                                "batch": stats.batch_count + 1,
                                "error": str(e),
                                "records_in_batch": len(batch)
                            })
                        
                        batch.clear()
                        
                        # Progress callback
                        if config.progress_callback:
                            await config.progress_callback(stats, record_count)
                
                # Process remaining records
                if batch:
                    self.active_operations[operation_id]["status"] = "processing_final_batch"
                    try:
                        await self._process_batch(session, batch, config, stats)
                    except Exception as e:
                        errors.append({
                            "batch": "final",
                            "error": str(e),
                            "records_in_batch": len(batch)
                        })
            
            # Invalidate cache if needed
            if not config.skip_cache_invalidation and self.cache_service:
                await self.cache_service.delete_pattern("event_list:*")
                await self.cache_service.delete_pattern("search:*")
            
            # Calculate final statistics
            end_time = datetime.utcnow()
            stats.processing_time_seconds = (end_time - start_time).total_seconds()
            
            if stats.processing_time_seconds > 0:
                stats.throughput_records_per_second = stats.processed_records / stats.processing_time_seconds
            
            # Determine final status
            if errors and config.validation_level == ValidationLevel.STRICT:
                status = "failed"
            elif stats.failed_records > 0 or stats.validation_errors > 0:
                status = "partial"
            else:
                status = "success"
            
            # Create result
            result = IngestionResult(
                operation_id=operation_id,
                status=status,
                stats=stats,
                errors=errors,
                warnings=warnings,
                file_info=file_info,
                metadata={
                    "config": asdict(config),
                    "started_at": start_time.isoformat(),
                    "completed_at": end_time.isoformat()
                }
            )
            
            logger.info(
                f"Bulk ingestion completed: {operation_id} - "
                f"{stats.processed_records}/{stats.total_records} records processed "
                f"in {stats.processing_time_seconds:.2f}s "
                f"({stats.throughput_records_per_second:.1f} rec/s)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Bulk ingestion failed: {operation_id} - {e}")
            
            # Update operation status
            self.active_operations[operation_id]["status"] = "failed"
            self.active_operations[operation_id]["error"] = str(e)
            
            # Create error result
            end_time = datetime.utcnow()
            stats.processing_time_seconds = (end_time - start_time).total_seconds()
            
            return IngestionResult(
                operation_id=operation_id,
                status="failed",
                stats=stats,
                errors=[{"error": str(e), "type": "system_error"}],
                warnings=warnings,
                file_info=file_info,
                metadata={
                    "config": asdict(config),
                    "started_at": start_time.isoformat(),
                    "failed_at": end_time.isoformat(),
                    "error": str(e)
                }
            )
        
        finally:
            # Clean up operation tracking
            async with self._operation_lock:
                if operation_id in self.active_operations:
                    del self.active_operations[operation_id]
    
    async def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of active operation."""
        return self.active_operations.get(operation_id)
    
    async def list_active_operations(self) -> List[Dict[str, Any]]:
        """List all active ingestion operations."""
        return list(self.active_operations.values())
    
    async def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "enabled": self.enabled,
            "active_operations": len(self.active_operations),
            "configuration": {
                "max_file_size_mb": self.max_file_size_mb,
                "default_batch_size": self.default_batch_size,
                "max_workers": self.max_workers,
                "memory_limit_mb": self.memory_limit_mb,
                "supported_formats": [fmt.value for fmt in IngestionFormat]
            }
        }


def build_bulk_ingestion_service(
    session_factory,
    categorizer: Optional[EventCategorizer] = None,
    impact_scorer: Optional[EventImpactScorer] = None,
    enrichment_service: Optional[EventEnrichmentService] = None,
    cache_service: Optional[EventCacheService] = None
) -> BulkIngestionService:
    """Factory function to create bulk ingestion service instance."""
    return BulkIngestionService(
        session_factory,
        categorizer,
        impact_scorer,
        enrichment_service,
        cache_service
    )