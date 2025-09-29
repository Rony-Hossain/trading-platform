"""Data Retention and Archival Service

Manages data lifecycle, retention policies, and archival operations
for the Event Data Service to optimize performance and ensure compliance.
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import zipfile
import tempfile
from pathlib import Path

from sqlalchemy import select, delete, and_, or_, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from ..models import EventORM, EventHeadlineORM

logger = logging.getLogger(__name__)


class RetentionPolicy(str, Enum):
    """Data retention policy types."""
    ACTIVE = "active"           # Recent and actively used data
    WARM = "warm"              # Less frequently accessed data
    COLD = "cold"              # Archived data, rarely accessed
    COMPLIANCE = "compliance"   # Long-term compliance storage
    DELETE = "delete"          # Data eligible for deletion


class ArchivalFormat(str, Enum):
    """Supported archival formats."""
    JSON = "json"
    PARQUET = "parquet"
    CSV = "csv"
    SQL_DUMP = "sql_dump"


class DataCategory(str, Enum):
    """Data categories for retention policies."""
    EVENTS = "events"
    HEADLINES = "headlines"
    METADATA = "metadata"
    LOGS = "logs"
    ANALYTICS = "analytics"


@dataclass
class RetentionRule:
    """Configuration for data retention rules."""
    name: str
    category: DataCategory
    policy: RetentionPolicy
    age_days: int
    conditions: Dict[str, Any]
    archive_format: Optional[ArchivalFormat] = None
    archive_location: Optional[str] = None
    enabled: bool = True
    priority: int = 1  # Lower number = higher priority


@dataclass
class ArchivalResult:
    """Result of archival operation."""
    rule_name: str
    category: DataCategory
    records_archived: int
    archive_size_bytes: int
    archive_location: str
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class RetentionStats:
    """Statistics for retention and archival operations."""
    total_events: int
    total_headlines: int
    active_events: int
    warm_events: int
    cold_events: int
    archived_events: int
    deleted_events: int
    storage_size_mb: float
    last_cleanup: Optional[datetime]
    next_cleanup: Optional[datetime]


class DataRetentionService:
    """Service for managing data retention and archival policies."""
    
    def __init__(self, session_factory):
        self.session_factory = session_factory
        
        # Configuration
        self.enabled = os.getenv("RETENTION_ENABLED", "true").lower() == "true"
        self.cleanup_interval_hours = int(os.getenv("RETENTION_CLEANUP_INTERVAL_HOURS", "24"))
        self.archive_base_path = os.getenv("RETENTION_ARCHIVE_PATH", "/data/archives")
        self.batch_size = int(os.getenv("RETENTION_BATCH_SIZE", "1000"))
        self.max_parallel_operations = int(os.getenv("RETENTION_MAX_PARALLEL", "3"))
        
        # Ensure archive directory exists
        Path(self.archive_base_path).mkdir(parents=True, exist_ok=True)
        
        # Load retention rules
        self.retention_rules = self._load_retention_rules()
        
        # Background task management
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"DataRetentionService initialized (enabled={self.enabled})")
    
    def _load_retention_rules(self) -> List[RetentionRule]:
        """Load retention rules from configuration."""
        default_rules = [
            # Active events (last 30 days)
            RetentionRule(
                name="active_events",
                category=DataCategory.EVENTS,
                policy=RetentionPolicy.ACTIVE,
                age_days=30,
                conditions={"status": ["scheduled", "occurred"]},
                priority=1
            ),
            
            # Warm events (30-180 days)
            RetentionRule(
                name="warm_events",
                category=DataCategory.EVENTS,
                policy=RetentionPolicy.WARM,
                age_days=180,
                conditions={"status": ["occurred", "cancelled", "impact_analyzed"]},
                priority=2
            ),
            
            # Cold events (180 days - 2 years)
            RetentionRule(
                name="cold_events",
                category=DataCategory.EVENTS,
                policy=RetentionPolicy.COLD,
                age_days=730,  # 2 years
                conditions={},
                archive_format=ArchivalFormat.JSON,
                archive_location="cold_storage",
                priority=3
            ),
            
            # Compliance events (2-7 years)
            RetentionRule(
                name="compliance_events",
                category=DataCategory.EVENTS,
                policy=RetentionPolicy.COMPLIANCE,
                age_days=2555,  # 7 years
                conditions={},
                archive_format=ArchivalFormat.PARQUET,
                archive_location="compliance_archive",
                priority=4
            ),
            
            # Delete very old events (7+ years)
            RetentionRule(
                name="delete_old_events",
                category=DataCategory.EVENTS,
                policy=RetentionPolicy.DELETE,
                age_days=2555,  # 7 years
                conditions={},
                priority=5
            ),
            
            # Active headlines (last 60 days)
            RetentionRule(
                name="active_headlines",
                category=DataCategory.HEADLINES,
                policy=RetentionPolicy.ACTIVE,
                age_days=60,
                conditions={},
                priority=1
            ),
            
            # Archive old headlines (60 days - 1 year)
            RetentionRule(
                name="archive_headlines",
                category=DataCategory.HEADLINES,
                policy=RetentionPolicy.COLD,
                age_days=365,
                conditions={},
                archive_format=ArchivalFormat.JSON,
                archive_location="headlines_archive",
                priority=2
            ),
            
            # Delete very old headlines (1+ year)
            RetentionRule(
                name="delete_old_headlines",
                category=DataCategory.HEADLINES,
                policy=RetentionPolicy.DELETE,
                age_days=365,
                conditions={},
                priority=3
            ),
        ]
        
        # Load custom rules from environment if available
        custom_rules_json = os.getenv("RETENTION_CUSTOM_RULES")
        if custom_rules_json:
            try:
                custom_rules_data = json.loads(custom_rules_json)
                for rule_data in custom_rules_data:
                    rule = RetentionRule(
                        name=rule_data["name"],
                        category=DataCategory(rule_data["category"]),
                        policy=RetentionPolicy(rule_data["policy"]),
                        age_days=rule_data["age_days"],
                        conditions=rule_data.get("conditions", {}),
                        archive_format=ArchivalFormat(rule_data["archive_format"]) if rule_data.get("archive_format") else None,
                        archive_location=rule_data.get("archive_location"),
                        enabled=rule_data.get("enabled", True),
                        priority=rule_data.get("priority", 10)
                    )
                    default_rules.append(rule)
            except Exception as e:
                logger.error(f"Failed to load custom retention rules: {e}")
        
        # Sort by priority
        return sorted(default_rules, key=lambda r: r.priority)
    
    async def start(self):
        """Start the retention service."""
        if not self.enabled:
            logger.info("Data retention service disabled")
            return
        
        logger.info("Starting data retention service")
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_worker())
    
    async def stop(self):
        """Stop the retention service."""
        logger.info("Stopping data retention service")
        
        self._shutdown_event.set()
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _cleanup_worker(self):
        """Background worker for periodic cleanup operations."""
        while not self._shutdown_event.is_set():
            try:
                logger.info("Starting scheduled data retention cleanup")
                await self.run_retention_cleanup()
                logger.info("Completed scheduled data retention cleanup")
                
                # Wait for next cleanup interval
                await asyncio.sleep(self.cleanup_interval_hours * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in retention cleanup worker: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def run_retention_cleanup(self) -> List[ArchivalResult]:
        """Run retention cleanup for all rules."""
        results = []
        
        for rule in self.retention_rules:
            if not rule.enabled:
                continue
            
            try:
                result = await self._apply_retention_rule(rule)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error applying retention rule {rule.name}: {e}")
                results.append(ArchivalResult(
                    rule_name=rule.name,
                    category=rule.category,
                    records_archived=0,
                    archive_size_bytes=0,
                    archive_location="",
                    duration_seconds=0.0,
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    async def _apply_retention_rule(self, rule: RetentionRule) -> Optional[ArchivalResult]:
        """Apply a specific retention rule."""
        start_time = datetime.utcnow()
        cutoff_date = datetime.utcnow() - timedelta(days=rule.age_days)
        
        logger.info(f"Applying retention rule: {rule.name} (cutoff: {cutoff_date})")
        
        async with self.session_factory() as session:
            if rule.category == DataCategory.EVENTS:
                return await self._process_events_retention(session, rule, cutoff_date, start_time)
            elif rule.category == DataCategory.HEADLINES:
                return await self._process_headlines_retention(session, rule, cutoff_date, start_time)
            else:
                logger.warning(f"Unsupported category for retention rule: {rule.category}")
                return None
    
    async def _process_events_retention(
        self,
        session: AsyncSession,
        rule: RetentionRule,
        cutoff_date: datetime,
        start_time: datetime
    ) -> ArchivalResult:
        """Process events according to retention rule."""
        
        # Build query based on rule conditions
        query = select(EventORM).where(EventORM.scheduled_at < cutoff_date)
        
        # Apply condition filters
        if rule.conditions:
            if "status" in rule.conditions:
                status_values = rule.conditions["status"]
                if isinstance(status_values, list):
                    query = query.where(EventORM.status.in_(status_values))
                else:
                    query = query.where(EventORM.status == status_values)
            
            if "category" in rule.conditions:
                category_values = rule.conditions["category"]
                if isinstance(category_values, list):
                    query = query.where(EventORM.category.in_(category_values))
                else:
                    query = query.where(EventORM.category == category_values)
            
            if "symbol" in rule.conditions:
                symbol_values = rule.conditions["symbol"]
                if isinstance(symbol_values, list):
                    query = query.where(EventORM.symbol.in_(symbol_values))
                else:
                    query = query.where(EventORM.symbol == symbol_values)
        
        # Count matching records
        count_result = await session.execute(
            select(func.count()).select_from(query.subquery())
        )
        total_records = count_result.scalar()
        
        if total_records == 0:
            return ArchivalResult(
                rule_name=rule.name,
                category=rule.category,
                records_archived=0,
                archive_size_bytes=0,
                archive_location="",
                duration_seconds=0.0,
                success=True
            )
        
        logger.info(f"Found {total_records} events for rule {rule.name}")
        
        # Process based on policy
        if rule.policy == RetentionPolicy.DELETE:
            return await self._delete_events(session, query, rule, start_time)
        elif rule.policy in [RetentionPolicy.COLD, RetentionPolicy.COMPLIANCE]:
            return await self._archive_events(session, query, rule, start_time)
        else:
            # For ACTIVE and WARM policies, just update metadata but don't archive/delete
            return ArchivalResult(
                rule_name=rule.name,
                category=rule.category,
                records_archived=total_records,
                archive_size_bytes=0,
                archive_location="in_place",
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                success=True
            )
    
    async def _process_headlines_retention(
        self,
        session: AsyncSession,
        rule: RetentionRule,
        cutoff_date: datetime,
        start_time: datetime
    ) -> ArchivalResult:
        """Process headlines according to retention rule."""
        
        query = select(EventHeadlineORM).where(EventHeadlineORM.published_at < cutoff_date)
        
        # Count matching records
        count_result = await session.execute(
            select(func.count()).select_from(query.subquery())
        )
        total_records = count_result.scalar()
        
        if total_records == 0:
            return ArchivalResult(
                rule_name=rule.name,
                category=rule.category,
                records_archived=0,
                archive_size_bytes=0,
                archive_location="",
                duration_seconds=0.0,
                success=True
            )
        
        logger.info(f"Found {total_records} headlines for rule {rule.name}")
        
        # Process based on policy
        if rule.policy == RetentionPolicy.DELETE:
            return await self._delete_headlines(session, query, rule, start_time)
        elif rule.policy in [RetentionPolicy.COLD, RetentionPolicy.COMPLIANCE]:
            return await self._archive_headlines(session, query, rule, start_time)
        else:
            return ArchivalResult(
                rule_name=rule.name,
                category=rule.category,
                records_archived=total_records,
                archive_size_bytes=0,
                archive_location="in_place",
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                success=True
            )
    
    async def _archive_events(
        self,
        session: AsyncSession,
        query,
        rule: RetentionRule,
        start_time: datetime
    ) -> ArchivalResult:
        """Archive events to external storage."""
        
        # Fetch events in batches
        events_data = []
        offset = 0
        
        while True:
            batch_query = query.offset(offset).limit(self.batch_size)
            result = await session.execute(batch_query)
            events = result.scalars().all()
            
            if not events:
                break
            
            # Convert to serializable format
            for event in events:
                event_dict = {
                    "id": event.id,
                    "symbol": event.symbol,
                    "title": event.title,
                    "category": event.category,
                    "status": event.status,
                    "scheduled_at": event.scheduled_at.isoformat(),
                    "timezone": event.timezone,
                    "description": event.description,
                    "metadata_json": event.metadata_json,
                    "source": event.source,
                    "external_id": event.external_id,
                    "impact_score": event.impact_score,
                    "created_at": event.created_at.isoformat(),
                    "updated_at": event.updated_at.isoformat(),
                }
                events_data.append(event_dict)
            
            offset += self.batch_size
        
        # Create archive file
        archive_path = await self._create_archive_file(
            events_data, 
            rule.archive_format or ArchivalFormat.JSON,
            rule.archive_location or "events",
            f"events_{rule.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Get file size
        archive_size = Path(archive_path).stat().st_size
        
        # Delete events from database after successful archive
        delete_query = delete(EventORM).where(
            EventORM.id.in_([event["id"] for event in events_data])
        )
        await session.execute(delete_query)
        await session.commit()
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Archived {len(events_data)} events to {archive_path}")
        
        return ArchivalResult(
            rule_name=rule.name,
            category=rule.category,
            records_archived=len(events_data),
            archive_size_bytes=archive_size,
            archive_location=archive_path,
            duration_seconds=duration,
            success=True
        )
    
    async def _archive_headlines(
        self,
        session: AsyncSession,
        query,
        rule: RetentionRule,
        start_time: datetime
    ) -> ArchivalResult:
        """Archive headlines to external storage."""
        
        # Fetch headlines in batches
        headlines_data = []
        offset = 0
        
        while True:
            batch_query = query.offset(offset).limit(self.batch_size)
            result = await session.execute(batch_query)
            headlines = result.scalars().all()
            
            if not headlines:
                break
            
            # Convert to serializable format
            for headline in headlines:
                headline_dict = {
                    "id": headline.id,
                    "symbol": headline.symbol,
                    "headline": headline.headline,
                    "summary": headline.summary,
                    "url": headline.url,
                    "published_at": headline.published_at.isoformat(),
                    "source": headline.source,
                    "external_id": headline.external_id,
                    "metadata_json": headline.metadata_json,
                    "created_at": headline.created_at.isoformat(),
                    "event_id": headline.event_id,
                }
                headlines_data.append(headline_dict)
            
            offset += self.batch_size
        
        # Create archive file
        archive_path = await self._create_archive_file(
            headlines_data, 
            rule.archive_format or ArchivalFormat.JSON,
            rule.archive_location or "headlines",
            f"headlines_{rule.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Get file size
        archive_size = Path(archive_path).stat().st_size
        
        # Delete headlines from database after successful archive
        delete_query = delete(EventHeadlineORM).where(
            EventHeadlineORM.id.in_([headline["id"] for headline in headlines_data])
        )
        await session.execute(delete_query)
        await session.commit()
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Archived {len(headlines_data)} headlines to {archive_path}")
        
        return ArchivalResult(
            rule_name=rule.name,
            category=rule.category,
            records_archived=len(headlines_data),
            archive_size_bytes=archive_size,
            archive_location=archive_path,
            duration_seconds=duration,
            success=True
        )
    
    async def _delete_events(
        self,
        session: AsyncSession,
        query,
        rule: RetentionRule,
        start_time: datetime
    ) -> ArchivalResult:
        """Delete events permanently."""
        
        # Count records to be deleted
        count_result = await session.execute(
            select(func.count()).select_from(query.subquery())
        )
        total_records = count_result.scalar()
        
        # Delete in batches to avoid long-running transactions
        deleted_count = 0
        while True:
            # Get a batch of IDs
            batch_result = await session.execute(
                query.with_only_columns(EventORM.id).limit(self.batch_size)
            )
            event_ids = [row[0] for row in batch_result.fetchall()]
            
            if not event_ids:
                break
            
            # Delete batch
            delete_query = delete(EventORM).where(EventORM.id.in_(event_ids))
            result = await session.execute(delete_query)
            deleted_count += result.rowcount
            await session.commit()
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Deleted {deleted_count} events")
        
        return ArchivalResult(
            rule_name=rule.name,
            category=rule.category,
            records_archived=deleted_count,
            archive_size_bytes=0,
            archive_location="deleted",
            duration_seconds=duration,
            success=True
        )
    
    async def _delete_headlines(
        self,
        session: AsyncSession,
        query,
        rule: RetentionRule,
        start_time: datetime
    ) -> ArchivalResult:
        """Delete headlines permanently."""
        
        # Count records to be deleted
        count_result = await session.execute(
            select(func.count()).select_from(query.subquery())
        )
        total_records = count_result.scalar()
        
        # Delete in batches to avoid long-running transactions
        deleted_count = 0
        while True:
            # Get a batch of IDs
            batch_result = await session.execute(
                query.with_only_columns(EventHeadlineORM.id).limit(self.batch_size)
            )
            headline_ids = [row[0] for row in batch_result.fetchall()]
            
            if not headline_ids:
                break
            
            # Delete batch
            delete_query = delete(EventHeadlineORM).where(EventHeadlineORM.id.in_(headline_ids))
            result = await session.execute(delete_query)
            deleted_count += result.rowcount
            await session.commit()
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Deleted {deleted_count} headlines")
        
        return ArchivalResult(
            rule_name=rule.name,
            category=rule.category,
            records_archived=deleted_count,
            archive_size_bytes=0,
            archive_location="deleted",
            duration_seconds=duration,
            success=True
        )
    
    async def _create_archive_file(
        self,
        data: List[Dict[str, Any]],
        format_type: ArchivalFormat,
        location: str,
        filename: str
    ) -> str:
        """Create archive file in specified format."""
        
        # Create location directory
        archive_dir = Path(self.archive_base_path) / location
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        if format_type == ArchivalFormat.JSON:
            archive_path = archive_dir / f"{filename}.json.gz"
            
            # Create compressed JSON file
            import gzip
            with gzip.open(archive_path, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        
        elif format_type == ArchivalFormat.PARQUET:
            archive_path = archive_dir / f"{filename}.parquet"
            
            # Convert to Parquet format (requires pandas and pyarrow)
            try:
                import pandas as pd
                df = pd.DataFrame(data)
                df.to_parquet(archive_path, compression='gzip')
            except ImportError:
                logger.warning("Pandas/PyArrow not available, falling back to JSON")
                return await self._create_archive_file(data, ArchivalFormat.JSON, location, filename)
        
        elif format_type == ArchivalFormat.CSV:
            archive_path = archive_dir / f"{filename}.csv.gz"
            
            import gzip
            import csv
            import io
            
            if data:
                fieldnames = data[0].keys()
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
                
                with gzip.open(archive_path, 'wt', encoding='utf-8') as f:
                    f.write(output.getvalue())
        
        else:  # Default to JSON
            return await self._create_archive_file(data, ArchivalFormat.JSON, location, filename)
        
        return str(archive_path)
    
    async def get_retention_stats(self) -> RetentionStats:
        """Get current retention and storage statistics."""
        async with self.session_factory() as session:
            # Count events by age
            now = datetime.utcnow()
            thirty_days_ago = now - timedelta(days=30)
            six_months_ago = now - timedelta(days=180)
            two_years_ago = now - timedelta(days=730)
            
            # Total events
            total_events = await session.scalar(select(func.count(EventORM.id)))
            
            # Events by age category
            active_events = await session.scalar(
                select(func.count(EventORM.id)).where(EventORM.scheduled_at >= thirty_days_ago)
            )
            
            warm_events = await session.scalar(
                select(func.count(EventORM.id)).where(
                    and_(
                        EventORM.scheduled_at < thirty_days_ago,
                        EventORM.scheduled_at >= six_months_ago
                    )
                )
            )
            
            cold_events = await session.scalar(
                select(func.count(EventORM.id)).where(
                    and_(
                        EventORM.scheduled_at < six_months_ago,
                        EventORM.scheduled_at >= two_years_ago
                    )
                )
            )
            
            archived_events = await session.scalar(
                select(func.count(EventORM.id)).where(EventORM.scheduled_at < two_years_ago)
            )
            
            # Total headlines
            total_headlines = await session.scalar(select(func.count(EventHeadlineORM.id)))
            
            # Estimate storage size (rough calculation)
            # This would be more accurate with actual database size queries
            avg_event_size = 2048  # bytes
            avg_headline_size = 1024  # bytes
            storage_size_mb = ((total_events or 0) * avg_event_size + 
                             (total_headlines or 0) * avg_headline_size) / (1024 * 1024)
            
            return RetentionStats(
                total_events=total_events or 0,
                total_headlines=total_headlines or 0,
                active_events=active_events or 0,
                warm_events=warm_events or 0,
                cold_events=cold_events or 0,
                archived_events=archived_events or 0,
                deleted_events=0,  # This would need to be tracked separately
                storage_size_mb=storage_size_mb,
                last_cleanup=None,  # This would be tracked in metadata
                next_cleanup=None   # Based on cleanup interval
            )
    
    async def get_retention_rules(self) -> List[Dict[str, Any]]:
        """Get current retention rules configuration."""
        return [asdict(rule) for rule in self.retention_rules]
    
    async def validate_retention_policy(self, rule: RetentionRule) -> Dict[str, Any]:
        """Validate a retention rule and estimate its impact."""
        async with self.session_factory() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=rule.age_days)
            
            if rule.category == DataCategory.EVENTS:
                query = select(func.count(EventORM.id)).where(EventORM.scheduled_at < cutoff_date)
                
                # Apply condition filters for estimation
                if rule.conditions:
                    if "status" in rule.conditions:
                        status_values = rule.conditions["status"]
                        if isinstance(status_values, list):
                            query = query.where(EventORM.status.in_(status_values))
                        else:
                            query = query.where(EventORM.status == status_values)
                
                affected_count = await session.scalar(query)
                
            elif rule.category == DataCategory.HEADLINES:
                affected_count = await session.scalar(
                    select(func.count(EventHeadlineORM.id)).where(
                        EventHeadlineORM.published_at < cutoff_date
                    )
                )
            else:
                affected_count = 0
            
            return {
                "rule_name": rule.name,
                "category": rule.category.value,
                "policy": rule.policy.value,
                "cutoff_date": cutoff_date.isoformat(),
                "affected_records": affected_count,
                "estimated_action": rule.policy.value,
                "archive_format": rule.archive_format.value if rule.archive_format else None,
                "archive_location": rule.archive_location
            }


def build_retention_service(session_factory) -> DataRetentionService:
    """Factory function to create retention service instance."""
    return DataRetentionService(session_factory)