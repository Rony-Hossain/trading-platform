"""Headline ingestion utilities for linking outcomes to scheduled events."""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import EventHeadlineORM, EventORM
from ..schemas import EventHeadline as EventHeadlineSchema
from .feed_health import FeedHealthMonitor
from .webhook_dispatcher import EventWebhookDispatcher

DEFAULT_POLL_INTERVAL_SECONDS = 120
MATCH_WINDOW_MINUTES = 120


class HeadlineIngestor:
    """Fetch low-latency headlines and associate them with upcoming events."""

    def __init__(self, session_factory, config: Optional[Dict[str, Any]] = None, health_monitor: Optional[FeedHealthMonitor] = None, webhook_dispatcher: Optional[EventWebhookDispatcher] = None, subscription_manager=None) -> None:
        self._session_factory = session_factory
        self._config = config or {}
        self._poll_interval = int(
            self._config.get(
                "poll_interval_seconds",
                os.getenv("EVENT_HEADLINE_POLL_INTERVAL", DEFAULT_POLL_INTERVAL_SECONDS),
            )
        )
        self._providers = self._config.get("providers") or self._default_providers()
        self._health_monitor = health_monitor
        self._webhook_dispatcher = webhook_dispatcher
        self._subscription_manager = subscription_manager
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    def _default_providers(self) -> List[Dict[str, Any]]:
        url = os.getenv("EVENT_HEADLINE_URL")
        api_key = os.getenv("EVENT_HEADLINE_API_KEY")
        if url:
            return [
                {
                    "name": os.getenv("EVENT_HEADLINE_PROVIDER", "headline-feed"),
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
                try:
                    headlines = await self._fetch_from_provider(client, provider)
                    if headlines:
                        await self._store_headlines(headlines)
                    await self._report_success(provider.get("name") or provider.get("url", "provider"), len(headlines) if headlines else 0)
                except Exception as exc:  # noqa: BLE001
                    await self._report_failure(provider.get('name') or provider.get('url', "provider"), str(exc))
                    print(f"Headline provider {provider.get('name')} failed: {exc}")

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
        items = payload.get("headlines") if isinstance(payload, dict) else payload
        normalized: List[Dict[str, Any]] = []
        for raw in items or []:
            normalized_headline = self._normalize_headline(raw, provider.get("name"))
            if normalized_headline:
                normalized.append(normalized_headline)
        return normalized

    def _normalize_headline(self, raw: Dict[str, Any], source: Optional[str]) -> Optional[Dict[str, Any]]:
        symbol = raw.get("symbol") or raw.get("ticker")
        headline = raw.get("headline") or raw.get("title")
        if not symbol or not headline:
            return None
        published = raw.get("published_at") or raw.get("timestamp")
        if not published:
            return None
        try:
            published_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except ValueError:
            return None
        return {
            "symbol": symbol,
            "headline": headline,
            "summary": raw.get("summary"),
            "url": raw.get("url"),
            "published_at": published_dt,
            "source": source,
            "external_id": raw.get("id"),
            "metadata": raw,
        }

    async def _report_success(self, feed_name: str, count: int) -> None:
        if self._health_monitor:
            await self._health_monitor.report_success(feed_name, count)

    async def _report_failure(self, feed_name: str, error: str) -> None:
        if self._health_monitor:
            await self._health_monitor.report_failure(feed_name, error)


    async def _notify_new_headlines(self, headlines: List[EventHeadlineORM]) -> None:
        # Notify via webhooks
        if self._webhook_dispatcher and self._webhook_dispatcher.has_targets():
            for headline in headlines:
                payload = self._serialize_headline(headline)
                await self._webhook_dispatcher.dispatch("headline.created", payload)
        
        # Notify via subscriptions
        if self._subscription_manager:
            # Import here to avoid circular imports
            from .subscription_manager import EventType
            for headline in headlines:
                payload = self._serialize_headline(headline)
                await self._subscription_manager.notify_event(payload, EventType.HEADLINE_LINKED)

    def _serialize_headline(self, headline: EventHeadlineORM) -> Dict[str, Any]:
        schema = EventHeadlineSchema.model_validate(headline)
        data = schema.model_dump()
        if headline.published_at:
            data["published_at"] = headline.published_at.isoformat()
        if headline.created_at:
            data["created_at"] = headline.created_at.isoformat()
        return data

    async def _store_headlines(self, headlines: List[Dict[str, Any]]) -> None:
        async with self._session_factory() as session:  # type: ignore[call-arg]
            new_headlines: List[EventHeadlineORM] = []
            for data in headlines:
                created = await self._upsert_headline(session, data)
                if created is not None:
                    new_headlines.append(created)
            await session.commit()
            if new_headlines:
                await self._notify_new_headlines(new_headlines)

    async def _upsert_headline(self, session: AsyncSession, data: Dict[str, Any]) -> Optional[EventHeadlineORM]:
        lookup = select(EventHeadlineORM).where(EventHeadlineORM.symbol == data["symbol"])
        if data.get("external_id") and data.get("source"):
            lookup = lookup.where(
                EventHeadlineORM.external_id == data["external_id"],
                EventHeadlineORM.source == data["source"],
            )
        else:
            lookup = lookup.where(EventHeadlineORM.published_at == data["published_at"])
        result = await session.execute(lookup)
        existing = result.scalars().first()
        now = datetime.utcnow()
        if existing:
            existing.headline = data["headline"]
            existing.summary = data.get("summary")
            existing.url = data.get("url")
            existing.metadata_json = data.get("metadata")
            existing.source = data.get("source")
            existing.external_id = data.get("external_id")
            if not existing.event_id:
                existing.event_id = await self._match_event(session, data)
            return None
        else:
            headline_row = EventHeadlineORM(
                symbol=data["symbol"],
                headline=data["headline"],
                summary=data.get("summary"),
                url=data.get("url"),
                metadata_json=data.get("metadata"),
                published_at=data["published_at"],
                source=data.get("source"),
                external_id=data.get("external_id"),
                created_at=now,
                event_id=await self._match_event(session, data),
            )
            session.add(headline_row)
            await session.flush()
            return headline_row

    async def _match_event(self, session: AsyncSession, data: Dict[str, Any]) -> Optional[str]:
        window = int(os.getenv("EVENT_HEADLINE_MATCH_WINDOW_MINUTES", MATCH_WINDOW_MINUTES))
        start = data["published_at"] - timedelta(minutes=window)
        end = data["published_at"] + timedelta(minutes=window)
        stmt = (
            select(EventORM)
            .where(EventORM.symbol == data["symbol"])
            .where(EventORM.scheduled_at.between(start, end))
            .order_by(EventORM.scheduled_at.desc())
        )
        result = await session.execute(stmt)
        event = result.scalars().first()
        return event.id if event else None


def build_headline_ingestor(session_factory, health_monitor: Optional[FeedHealthMonitor] = None, webhook_dispatcher: Optional[EventWebhookDispatcher] = None, subscription_manager=None) -> HeadlineIngestor:
    config: Dict[str, Any] = {
        "poll_interval_seconds": int(os.getenv("EVENT_HEADLINE_POLL_INTERVAL", DEFAULT_POLL_INTERVAL_SECONDS)),
    }
    return HeadlineIngestor(session_factory, config=config, health_monitor=health_monitor, webhook_dispatcher=webhook_dispatcher, subscription_manager=subscription_manager)
