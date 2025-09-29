"""External calendar ingestion utilities."""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import httpx
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import EventORM
from ..schemas import Event as EventSchema
from .feed_health import FeedHealthMonitor
from .event_impact import EventImpactScorer
from .event_categorizer import EventCategorizer
from .webhook_dispatcher import EventWebhookDispatcher

DEFAULT_POLL_INTERVAL_SECONDS = 900


class EventCalendarIngestor:
    """Periodically fetch scheduled events from external providers."""

    def __init__(self, session_factory, config: Optional[Dict[str, Any]] = None, health_monitor: Optional[FeedHealthMonitor] = None, webhook_dispatcher: Optional[EventWebhookDispatcher] = None, categorizer: Optional[EventCategorizer] = None, subscription_manager=None, enrichment_service=None) -> None:
        self._session_factory = session_factory
        self._config = config or {}
        self._poll_interval = int(
            self._config.get(
                "poll_interval_seconds",
                os.getenv("EVENT_CALENDAR_POLL_INTERVAL", DEFAULT_POLL_INTERVAL_SECONDS),
            )
        )
        self._providers = self._load_providers()
        self._max_failures = int(os.getenv("EVENT_CALENDAR_PROVIDER_MAX_FAILURES", 3))
        self._failback_seconds = int(os.getenv("EVENT_CALENDAR_PROVIDER_FAILBACK_SECONDS", 600))
        self._dedupe_window = timedelta(minutes=int(os.getenv("EVENT_CALENDAR_DEDUPE_WINDOW_MINUTES", 60)))
        self._max_horizon = timedelta(days=int(os.getenv("EVENT_CALENDAR_MAX_HORIZON_DAYS", 365)))
        self._min_symbol_length = int(os.getenv("EVENT_CALENDAR_MIN_SYMBOL_LENGTH", 1))
        allowed = self._config.get("allowed_categories") or os.getenv("EVENT_CALENDAR_ALLOWED_CATEGORIES")
        if isinstance(allowed, str):
            allowed = [item.strip() for item in allowed.split(',') if item.strip()]
        self._allowed_categories: Optional[Set[str]] = set(map(str.lower, allowed)) if allowed else None
        self._recent_event_cache: Dict[str, datetime] = {}
        self._provider_state: Dict[str, Dict[str, Any]] = {}
        for provider in self._providers:
            key = provider.get('name') or provider.get('url')
            self._provider_state[key] = {
                'failure_count': 0,
                'backoff_until': None,
                'last_error': None,
            }
        self._health_monitor = health_monitor
        self._webhook_dispatcher = webhook_dispatcher
        self._subscription_manager = subscription_manager
        self._enrichment_service = enrichment_service
        self._categorizer = categorizer or EventCategorizer()
        impact_overrides = self._config.get("impact_category_base")
        if not impact_overrides:
            env_overrides = os.getenv("EVENT_IMPACT_CATEGORY_BASE")
            if env_overrides:
                try:
                    decoded = json.loads(env_overrides)
                    if isinstance(decoded, dict):
                        impact_overrides = decoded
                except json.JSONDecodeError:
                    impact_overrides = None
        self._impact_scorer = EventImpactScorer(impact_overrides if isinstance(impact_overrides, dict) else None)
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    def _load_providers(self) -> List[Dict[str, Any]]:
        providers = self._config.get('providers') or []
        if not providers:
            providers_json = os.getenv('EVENT_CALENDAR_PROVIDERS_JSON')
            if providers_json:
                try:
                    data = json.loads(providers_json)
                    if isinstance(data, list):
                        providers = data
                except json.JSONDecodeError:
                    print('Failed to parse EVENT_CALENDAR_PROVIDERS_JSON')
        if not providers:
            providers = self._default_providers()
        normalized = []
        for provider in providers:
            provider = provider.copy()
            provider.setdefault('name', provider.get('url', 'provider'))
            normalized.append(provider)
        return normalized

    def _default_providers(self) -> List[Dict[str, Any]]:
        url = os.getenv("EVENT_CALENDAR_URL")
        api_key = os.getenv("EVENT_CALENDAR_API_KEY")
        if url:
            return [
                {
                    "name": os.getenv("EVENT_CALENDAR_PROVIDER", "external"),
                    "url": url,
                    "api_key": api_key,
                }
            ]
        return []

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._stop_event.set()
            await self._task
            self._task = None

    async def trigger_once(self) -> None:
        await self._process_once()

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            await self._process_once()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._poll_interval)
            except asyncio.TimeoutError:
                continue

    async def _process_once(self) -> None:
        if not self._providers:
            return
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            for provider in self._providers:
                provider_key = provider.get('name') or provider.get('url')
                state = self._provider_state.get(provider_key, {})
                now = datetime.utcnow()
                backoff_until = state.get('backoff_until')
                if backoff_until and now < backoff_until:
                    if self._health_monitor:
                        backoff_label = backoff_until.isoformat() if hasattr(backoff_until, "isoformat") else str(backoff_until)
                        await self._report_skip(provider_key, f"provider backoff until {backoff_label}")
                    continue
                try:
                    events = await self._fetch_from_provider(client, provider)
                    if events:
                        await self._store_events(events)
                    await self._report_success(provider_key, len(events) if events else 0)
                    state['failure_count'] = 0
                    state['backoff_until'] = None
                    state['last_error'] = None
                except Exception as exc:  # noqa: BLE001
                    await self._report_failure(provider_key, str(exc))
                    print(f"Event provider {provider.get('name')} failed: {exc}")
                    failure_count = state.get('failure_count', 0) + 1
                    state['failure_count'] = failure_count
                    state['last_error'] = str(exc)
                    if failure_count >= self._max_failures:
                        state['backoff_until'] = now + timedelta(seconds=self._failback_seconds)
                    else:
                        state['backoff_until'] = None
                finally:
                    self._provider_state[provider_key] = state

    async def _fetch_from_provider(
        self, client: httpx.AsyncClient, provider: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        url = provider.get("url")
        if not url:
            return []
        params = provider.get("params", {})
        headers = provider.get("headers", {})
        api_key = provider.get("api_key")
        if api_key and "X-API-KEY" not in headers:
            headers["X-API-KEY"] = api_key
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            events = payload.get("events") or payload.get("data") or []
        else:
            events = payload
        normalized: List[Dict[str, Any]] = []
        for raw in events:
            normalized_event = self._normalize_event(raw, provider.get("name"))
            if normalized_event:
                normalized.append(normalized_event)
        return normalized

    async def _report_success(self, feed_name: str, event_count: int) -> None:
        if self._health_monitor:
            await self._health_monitor.report_success(feed_name, event_count)

    async def _report_failure(self, feed_name: str, error: str) -> None:
        if self._health_monitor:
            await self._health_monitor.report_failure(feed_name, error)

    async def _report_skip(self, feed_name: str, reason: str) -> None:
        if self._health_monitor:
            await self._health_monitor.report_skip(feed_name, reason)

    def _normalize_event(self, raw: Dict[str, Any], source: Optional[str]) -> Optional[Dict[str, Any]]:
        symbol = raw.get("symbol") or raw.get("ticker")
        title = raw.get("title") or raw.get("name")
        category = raw.get("category") or raw.get("type")
        scheduled = raw.get("scheduled_at") or raw.get("date")
        if scheduled:
            try:
                scheduled_dt = datetime.fromisoformat(scheduled.replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None
        if not symbol or not title or not category:
            return None

        ext_id = raw.get("id") or raw.get("event_id")
        timezone = raw.get("timezone")
        description = raw.get("description")
        metadata = self._prepare_metadata(raw)
        raw_category_value = category
        classification = self._categorizer.categorize(
            raw_category=raw_category_value,
            title=title,
            description=description,
            metadata=metadata,
        )
        category = classification.category
        metadata.setdefault("raw_category", raw_category_value)
        classification_meta = metadata.setdefault("classification", {})
        classification_meta.update(
            {
                "raw_category": raw_category_value,
                "canonical_category": classification.category,
                "confidence": classification.confidence,
                "matched_keywords": classification.matched_keywords,
                "source": "heuristic",
            }
        )
        if classification.tags:
            existing_tags = metadata.get("tags")
            if not isinstance(existing_tags, list):
                existing_tags = []
            metadata["tags"] = sorted({*existing_tags, *classification.tags})
        impact_score = raw.get('impact_score') or raw.get('impact')
        if isinstance(impact_score, str) and impact_score.isdigit():
            impact_score = int(impact_score)
        elif isinstance(impact_score, (int, float)):
            impact_score = int(impact_score)
        else:
            impact_score = None
        if impact_score is None:
            score_result = self._impact_scorer.score_event(
                {
                    "symbol": symbol,
                    "category": category,
                    "metadata": metadata,
                }
            )
            impact_score = score_result.score
            metadata.setdefault("impact_analysis", score_result.components)
        if impact_score is None:
            defaults = self._config.get('category_impacts') or {}
            if not defaults:
                env_json = os.getenv('EVENT_CALENDAR_CATEGORY_IMPACTS')
                if env_json:
                    try:
                        defaults = json.loads(env_json)
                    except json.JSONDecodeError:
                        defaults = {}
            if defaults:
                normalized_defaults = {str(k).lower(): int(v) for k, v in defaults.items()}
                category_key = category.lower()
                if category_key in normalized_defaults:
                    impact_score = normalized_defaults[category_key]
            if impact_score is None:
                default_impact = os.getenv('EVENT_CALENDAR_DEFAULT_IMPACT_SCORE')
                if default_impact and default_impact.isdigit():
                    impact_score = int(default_impact)
        normalized = {
            "symbol": symbol,
            "title": title,
            "category": category,
            "scheduled_at": scheduled_dt,
            "timezone": timezone,
            "description": description,
            "metadata": metadata,
            "source": source,
            "external_id": ext_id,
            "impact_score": impact_score,
        }
        return normalized if self._is_event_valid(normalized) else None


    async def _notify_new_events(self, events: List[EventORM]) -> None:
        # Notify via webhooks
        if self._webhook_dispatcher and self._webhook_dispatcher.has_targets():
            for event in events:
                payload = self._serialize_event(event)
                await self._webhook_dispatcher.dispatch("event.created", payload)
        
        # Notify via subscriptions
        if self._subscription_manager:
            # Import here to avoid circular imports
            from .subscription_manager import EventType
            for event in events:
                payload = self._serialize_event(event)
                await self._subscription_manager.notify_event(payload, EventType.EVENT_CREATED)

    async def _enrich_new_events(self, session: AsyncSession, events: List[EventORM]) -> None:
        """Enrich newly created events with market context."""
        try:
            # Prepare events for enrichment
            event_payloads = [self._serialize_event(event) for event in events]
            
            # Batch enrich events
            enriched_payloads = await self._enrichment_service.batch_enrich_events(event_payloads)
            
            # Update events with enriched metadata
            for event, enriched_payload in zip(events, enriched_payloads):
                new_metadata = enriched_payload.get("metadata")
                if new_metadata and new_metadata != event.metadata_json:
                    event.metadata_json = new_metadata
                    
            # Commit enrichment updates
            await session.commit()
            
        except Exception as e:
            print(f"Failed to enrich events: {e}")
            # Don't fail the entire ingestion if enrichment fails
            await session.rollback()
            # Re-commit the original events without enrichment
            await session.commit()

    def _serialize_event(self, event: EventORM) -> Dict[str, Any]:
        schema = EventSchema.model_validate(event)
        data = schema.model_dump()
        if event.scheduled_at:
            data["scheduled_at"] = event.scheduled_at.isoformat()
        if event.created_at:
            data["created_at"] = event.created_at.isoformat()
        if event.updated_at:
            data["updated_at"] = event.updated_at.isoformat()
        return data

    def _prepare_metadata(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        raw_meta = raw.get("metadata")
        if isinstance(raw_meta, dict):
            metadata.update(raw_meta)
        capture_keys = (
            "market_cap",
            "marketCap",
            "market_capitalization",
            "market_cap_usd",
            "company_market_cap",
            "avg_daily_volume",
            "average_daily_volume",
            "adv",
            "implied_move",
            "implied_move_pct",
            "implied_move_bp",
            "expected_move",
            "expected_move_pct",
            "expected_price_move",
            "expected_move_percent",
            "historical_avg_move",
            "avg_post_move",
            "historical_move_pct",
            "historical_abs_move",
            "importance",
            "event_scope",
            "is_major",
            "is_critical",
            "is_flagship",
            "is_high_importance",
            "confidence",
            "is_preliminary",
            "tentative",
        )
        for key in capture_keys:
            if key in raw and raw[key] is not None and key not in metadata:
                metadata[key] = raw[key]
        return metadata

    def _dedupe_key(self, data: Dict[str, Any]) -> str:
        if data.get('source') and data.get('external_id'):
            return f"{data['source']}::{data['external_id']}"
        return f"{data['symbol'].upper()}::{data['category'].lower()}::{data['scheduled_at'].isoformat()}"

    def _prune_cache(self, now: datetime) -> None:
        threshold = now - self._dedupe_window
        keys_to_remove = [key for key, timestamp in self._recent_event_cache.items() if timestamp < threshold]
        for key in keys_to_remove:
            self._recent_event_cache.pop(key, None)

    def _is_event_valid(self, data: Dict[str, Any]) -> bool:
        if len(data.get('symbol', '')) < self._min_symbol_length:
            return False
        if self._allowed_categories and data.get('category', '').lower() not in self._allowed_categories:
            return False
        scheduled_at = data.get('scheduled_at')
        if not isinstance(scheduled_at, datetime):
            return False
        now = datetime.utcnow()
        if scheduled_at > now + self._max_horizon:
            return False
        if scheduled_at < now - timedelta(days=30):
            # ignore stale events older than 30 days
            return False
        return True

    async def _store_events(self, events: List[Dict[str, Any]]) -> None:
        if not events:
            return
        now = datetime.utcnow()
        self._prune_cache(now)
        async with self._session_factory() as session:  # type: ignore[call-arg]
            new_events: List[EventORM] = []
            for event_data in events:
                if not self._is_event_valid(event_data):
                    continue
                dedupe_key = self._dedupe_key(event_data)
                cached_at = self._recent_event_cache.get(dedupe_key)
                if cached_at and now - cached_at < self._dedupe_window:
                    continue
                created_event = await self._upsert_event(session, event_data)
                if created_event is not None:
                    new_events.append(created_event)
                self._recent_event_cache[dedupe_key] = now
            try:
                await session.commit()
            except SQLAlchemyError as exc:  # noqa: BLE001
                await session.rollback()
                print(f"Failed to store events: {exc}")
            else:
                if new_events:
                    # Enrich events with market context
                    if self._enrichment_service:
                        await self._enrich_new_events(session, new_events)
                    await self._notify_new_events(new_events)

    async def _upsert_event(self, session: AsyncSession, data: Dict[str, Any]) -> None:
        lookup_stmt = select(EventORM)
        if data.get("source") and data.get("external_id"):
            lookup_stmt = lookup_stmt.where(
                EventORM.source == data["source"],
                EventORM.external_id == data["external_id"],
            )
        else:
            lookup_stmt = lookup_stmt.where(
                EventORM.symbol == data["symbol"],
                EventORM.category == data["category"],
                EventORM.scheduled_at == data["scheduled_at"],
            )
        result = await session.execute(lookup_stmt)
        existing = result.scalars().first()
        now = datetime.utcnow()
        if existing:
            existing.title = data["title"]
            existing.timezone = data.get("timezone")
            existing.description = data.get("description")
            existing.metadata_json = data.get("metadata")
            existing.status = data.get("status", existing.status)
            existing.source = data.get("source", existing.source)
            existing.external_id = data.get("external_id", existing.external_id)
            if data.get('impact_score') is not None:
                existing.impact_score = data['impact_score']
            existing.updated_at = now
            return None
        else:
            event = EventORM(
                symbol=data["symbol"],
                title=data["title"],
                category=data["category"],
                scheduled_at=data["scheduled_at"],
                timezone=data.get("timezone"),
                description=data.get("description"),
                metadata_json=data.get("metadata"),
                status=data.get("status", "scheduled"),
                source=data.get("source"),
                external_id=data.get("external_id"),
                impact_score=data.get("impact_score"),
                created_at=now,
                updated_at=now,
            )
            session.add(event)
            await session.flush()
            return event


def build_calendar_ingestor(session_factory, health_monitor: Optional[FeedHealthMonitor] = None, webhook_dispatcher: Optional[EventWebhookDispatcher] = None, categorizer=None, subscription_manager=None, enrichment_service=None) -> EventCalendarIngestor:
    config: Dict[str, Any] = {
        "poll_interval_seconds": int(os.getenv("EVENT_CALENDAR_POLL_INTERVAL", DEFAULT_POLL_INTERVAL_SECONDS)),
    }
    return EventCalendarIngestor(session_factory, config=config, health_monitor=health_monitor, webhook_dispatcher=webhook_dispatcher, categorizer=categorizer, subscription_manager=subscription_manager, enrichment_service=enrichment_service)
