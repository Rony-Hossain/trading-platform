# repo: services/news-data-service
# ─────────────────────────────────────────────────────────────────────────────
# This is a slim, PIT-safe News-Data-Service with: 
# - exact dedupe prefilter (Redis SETNX)
# - identity cache (Redis)
# - async upsert buffer (micro-batching)
# - transactional outbox + DLQ publisher to Redis Streams
# - circuit breaker in adapter base
# - health + SSE endpoints
# - Flyway-style SQL migrations (see migrations/)
#
# NOTE: Replace provider stubs with real API calls. This runs as-is against
# Postgres + Redis (docker-compose included below) and exposes minimal endpoints.
#
# Files are delimited by "# === path: <file>" markers.

# === path: pyproject.toml
[tool.poetry]
name = "news-data-service"
version = "0.1.0"
description = "Slim, PIT-safe news ingest service"
authors = ["Rony Team <team@example.com>"]

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.115.0"
uvicorn = {extras = ["standard"], version = "^0.30.0"}
SQLAlchemy = "^2.0.32"
asyncpg = "^0.29.0"
redis = "^5.0.7"
pydantic = "^2.9.2"
pydantic-settings = "^2.4.0"
httpx = "^0.27.2"
python-dateutil = "^2.9.0.post0"
structlog = "^24.1.0"
prometheus-client = "^0.20.0"
beautifulsoup4 = "^4.12.3"
lxml = "^5.3.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3.3"
pytest-asyncio = "^0.24.0"
ruff = "^0.6.8"

[tool.ruff]
line-length = 100

# === path: Dockerfile
FROM python:3.11-slim
WORKDIR /app
ENV POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir poetry==1.8.3
COPY pyproject.toml /app/
RUN poetry install --no-root --only main
COPY app /app/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

# === path: docker-compose.yml
version: "3.9"
services:
  postgres:
    image: timescale/timescaledb-ha:pg16-latest
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_USER: postgres
      POSTGRES_DB: news
    ports: ["5432:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 10
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
  flyway:
    image: flyway/flyway:10
    command: -url=jdbc:postgresql://postgres:5432/news -user=postgres -password=postgres -connectRetries=30 migrate
    volumes:
      - ./migrations:/flyway/sql
    depends_on:
      - postgres
  news:
    build: .
    environment:
      NEWS_PG_DSN: postgresql+asyncpg://postgres:postgres@postgres:5432/news
      REDIS_URL: redis://redis:6379/0
      SSE_REDIS_STREAM: news.events
      PREFILTER_TTL_SECONDS: 172800
      UPSERT_BATCH_SIZE: 200
      UPSERT_FLUSH_MS: 20
      CIRCUIT_TRIP_ERROR_RATE: 0.35
      CIRCUIT_MIN_WINDOW: 50
      FINNHUB_API_KEY: ${FINNHUB_API_KEY:-}
      FINNHUB_NEWS_CATEGORY: ${FINNHUB_NEWS_CATEGORY:-general}
      FINNHUB_COMPANY_SYMBOLS: ${FINNHUB_COMPANY_SYMBOLS:-}
      FINNHUB_COMPANY_WATCHLIST_FILE: ${FINNHUB_COMPANY_WATCHLIST_FILE:-/app/watchlist/watchlist.txt}
      FINNHUB_COMPANY_WATCHLIST_REDIS_SET: ${FINNHUB_COMPANY_WATCHLIST_REDIS_SET:-}
      WATCHLIST_REFRESH_SECONDS: ${WATCHLIST_REFRESH_SECONDS:-60}
    volumes:
      - ./watchlist.txt:/app/watchlist/watchlist.txt:ro
    ports: ["8080:8080"]
    depends_on:
      - postgres
      - redis
      - flyway

# === path: app/config.py
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    NEWS_PG_DSN: str = Field(..., description="SQLAlchemy async DSN for Postgres")
    REDIS_URL: str = Field(..., description="Redis URL")

    # prefilter / cache
    PREFILTER_TTL_SECONDS: int = 48 * 3600
    IDENTITY_CACHE_TTL_SECONDS: int = 24 * 3600

    # upsert buffer
    UPSERT_BATCH_SIZE: int = 200
    UPSERT_FLUSH_MS: int = 20

    # redis streams
    SSE_REDIS_STREAM: str = "news.events"

    # circuit breaker
    CIRCUIT_TRIP_ERROR_RATE: float = 0.35
    CIRCUIT_MIN_WINDOW: int = 50
    CIRCUIT_OPEN_SECONDS: int = 60

    # watchlist config
    FINNHUB_COMPANY_WATCHLIST_FILE: str | None = None
    FINNHUB_COMPANY_WATCHLIST_REDIS_SET: str | None = None
    WATCHLIST_REFRESH_SECONDS: int = 60

settings = Settings()

# === path: app/db.py
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from .config import settings

engine = create_async_engine(settings.NEWS_PG_DSN, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

async def get_session() -> AsyncSession:
    async with SessionLocal() as session:
        yield session

# === path: app/models.py
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, Integer, JSON, TIMESTAMP, func, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
import uuid

class Base(DeclarativeBase):
    pass

class Content(Base):
    __tablename__ = "content"
    content_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    external_id: Mapped[str] = mapped_column(String(256), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[list[str] | None] = mapped_column(JSON)
    categories: Mapped[list[str] | None] = mapped_column(JSON)
    language: Mapped[str | None] = mapped_column(String(8))
    published_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    ingested_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    as_of_ts: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    revision_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    metadata: Mapped[dict | None] = mapped_column(JSON)
    exact_dedupe_key: Mapped[str] = mapped_column(String(128), nullable=False)

    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uq_source_external"),
        UniqueConstraint("exact_dedupe_key", name="uq_exact_dedupe"),
    )

class Outbox(Base):
    __tablename__ = "outbox"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING")
    retries: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

class OutboxDLQ(Base):
    __tablename__ = "outbox_dlq"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text)
    retries: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

class IngestAudit(Base):
    __tablename__ = "ingest_audit"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    source: Mapped[str] = mapped_column(String(32))
    external_id: Mapped[str] = mapped_column(String(256))
    reason: Mapped[str] = mapped_column(String(64))
    dedupe_key: Mapped[str | None] = mapped_column(String(128))
    details: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

# === path: app/schema.py
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime

class RawContentIn(BaseModel):
    source: Literal["finnhub","alpha_vantage","newsapi","rss","wire","sec","other"]
    external_id: str
    url: str
    title: str
    body: Optional[str] = None
    authors: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    language: Optional[str] = None
    published_at: datetime
    metadata: Optional[dict] = None

class ContentCreated(BaseModel):
    event: Literal["content.created"] = "content.created"
    type: Literal["news"] = "news"
    content_id: str
    source: str
    external_id: str
    url: str
    title: str
    body_present: bool
    language: Optional[str]
    published_at: datetime
    as_of_ts: datetime
    ingested_at: datetime
    revision_seq: int
    metadata: Optional[dict]

class ContentCorrected(BaseModel):
    event: Literal["content.corrected"] = "content.corrected"
    type: Literal["news"] = "news"
    content_id: str
    revision_seq: int
    published_at: datetime
    as_of_ts: datetime

class ContentRetracted(BaseModel):
    event: Literal["content.retracted"] = "content.retracted"
    type: Literal["news"] = "news"
    content_id: str
    reason: Optional[str]
    as_of_ts: datetime

# === path: app/utils.py
import hashlib, urllib.parse

def host_of(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""

def exact_dedupe_key(title: str, url: str) -> str:
    s = (title or "").lower() + "|" + host_of(url)
    return hashlib.sha256(s.encode()).hexdigest()

# === path: app/prefilter.py
import asyncio
from redis.asyncio import Redis
from .config import settings
from .utils import exact_dedupe_key as _dk

PREFILTER_PREFIX = "dedupe:"
IDENTITY_PREFIX = "id:"

class Prefilter:
    def __init__(self, redis: Redis):
        self.redis = redis
        self.ttl = settings.PREFILTER_TTL_SECONDS

    async def is_duplicate(self, title: str, url: str) -> bool:
        key = PREFILTER_PREFIX + _dk(title, url)
        # SETNX with TTL: use SET with NX + EX
        ok = await self.redis.set(key, 1, ex=self.ttl, nx=True)
        return not bool(ok)

class IdentityCache:
    def __init__(self, redis: Redis):
        self.redis = redis
        self.ttl = settings.IDENTITY_CACHE_TTL_SECONDS

    async def get(self, source: str, external_id: str) -> dict | None:
        v = await self.redis.get(IDENTITY_PREFIX + f"{source}:{external_id}")
        return None if v is None else __import__("json").loads(v)

    async def set(self, source: str, external_id: str, payload: dict) -> None:
        await self.redis.set(IDENTITY_PREFIX + f"{source}:{external_id}", __import__("json").dumps(payload), ex=self.ttl)

# === path: app/ccs.py
from bs4 import BeautifulSoup

def sanitize_html(text: str | None) -> str | None:
    if not text:
        return text
    try:
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(" ", strip=True)
    except Exception:
        return text

def canon_authors(authors: list[str] | None) -> list[str] | None:
    if not authors:
        return authors
    out = []
    for a in authors:
        a = (a or "").strip()
        if a:
            out.append(a)
    return out or None

CATEGORY_MAP = {
    # example mapping; extend as needed
    "Tech": "technology",
    "Technology": "technology",
}

def map_categories(cats: list[str] | None) -> list[str] | None:
    if not cats:
        return cats
    return [CATEGORY_MAP.get(c, c).lower() for c in cats]

# === path: app/batcher.py
import asyncio
from datetime import datetime, timezone
from typing import Literal
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Content, Outbox, IngestAudit
from .schema import ContentCreated, ContentCorrected
from .utils import exact_dedupe_key

Lane = Literal["REGULAR", "LIKELY_NOOP"]

class UpsertBuffer:
    def __init__(self, session_factory, batch_size: int, flush_ms: int):
        self._sf = session_factory
        self.batch_size = batch_size
        self.flush_ms = flush_ms
        self.queue: asyncio.Queue[tuple[Lane, dict]] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def enqueue(self, lane: Lane, item: dict):
        await self.queue.put((lane, item))

    async def _run(self):
        while True:
            items: list[tuple[Lane, dict]] = []
            try:
                lane_item = await asyncio.wait_for(self.queue.get(), timeout=self.flush_ms / 1000)
                items.append(lane_item)
            except asyncio.TimeoutError:
                pass
            # drain up to batch_size
            while len(items) < self.batch_size and not self.queue.empty():
                items.append(self.queue.get_nowait())
            if not items:
                continue
            await self._process_batch(items)

    async def _process_batch(self, items: list[tuple[Lane, dict]]):
        async with self._sf() as session:
            await self._batch_upsert(session, items)

    async def _batch_upsert(self, session: AsyncSession, items: list[tuple[Lane, dict]]):
        now = datetime.now(timezone.utc)
        for lane, r in items:
            # per-row transactional logic (kept simple for clarity)
            # SELECT FOR UPDATE SKIP LOCKED by identity
            stmt = select(Content).where(Content.source == r["source"], Content.external_id == r["external_id"]).with_for_update(skip_locked=True)
            res = await session.execute(stmt)
            row: Content | None = res.scalar_one_or_none()
            dedupe = exact_dedupe_key(r["title"], r["url"]) 
            if row is None:
                # new insert
                c = Content(
                    source=r["source"],
                    external_id=r["external_id"],
                    url=r["url"],
                    title=r["title"],
                    body=r.get("body"),
                    authors=r.get("authors"),
                    categories=r.get("categories"),
                    language=r.get("language"),
                    published_at=r["published_at"],
                    ingested_at=now,
                    as_of_ts=min(r["published_at"], now),
                    revision_seq=0,
                    status="active",
                    metadata=r.get("metadata"),
                    exact_dedupe_key=dedupe,
                )
                session.add(c)
                await session.flush()
                evt = ContentCreated(
                    content_id=str(c.content_id),
                    source=c.source,
                    external_id=c.external_id,
                    url=c.url,
                    title=c.title,
                    body_present=bool(c.body),
                    language=c.language,
                    published_at=c.published_at,
                    as_of_ts=c.as_of_ts,
                    ingested_at=c.ingested_at,
                    revision_seq=c.revision_seq,
                    metadata=c.metadata,
                )
                session.add(Outbox(payload=evt.model_dump()))
            else:
                # existing identity
                unchanged = (
                    row.title == r["title"] and row.url == r["url"] and row.published_at == r["published_at"]
                )
                if unchanged:
                    session.add(IngestAudit(
                        content_id=row.content_id,
                        source=row.source,
                        external_id=row.external_id,
                        reason="NOOP_UNCHANGED",
                        dedupe_key=row.exact_dedupe_key,
                        details=None,
                    ))
                else:
                    row.title = r["title"]
                    row.url = r["url"]
                    row.published_at = r["published_at"]
                    row.revision_seq = row.revision_seq + 1
                    # keep PIT monotone
                    row.as_of_ts = min(row.as_of_ts, min(r["published_at"], now))
                    evt2 = ContentCorrected(
                        content_id=str(row.content_id),
                        revision_seq=row.revision_seq,
                        published_at=row.published_at,
                        as_of_ts=row.as_of_ts,
                    )
                    session.add(Outbox(payload=evt2.model_dump()))
        try:
            await session.commit()
        except IntegrityError as e:
            await session.rollback()
            # exact_dedupe unique conflict or identity race → audit & continue
            for _, r in items:
                session.add(IngestAudit(
                    content_id=None,
                    source=r["source"],
                    external_id=r["external_id"],
                    reason="DEDUP_DB_CONFLICT",
                    dedupe_key=exact_dedupe_key(r["title"], r["url"]),
                    details={"err": str(e)}
                ))
            await session.commit()

# === path: app/outbox.py
import asyncio, json
from redis.asyncio import Redis
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Outbox, OutboxDLQ

class OutboxPublisher:
    def __init__(self, session_factory, redis: Redis, stream: str, max_retries: int = 20):
        self._sf = session_factory
        self.redis = redis
        self.stream = stream
        self.max_retries = max_retries
        self._task: asyncio.Task | None = None

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        while True:
            async with self._sf() as s:
                rows = (await s.execute(select(Outbox).where(Outbox.status == "PENDING").order_by(Outbox.id).limit(100))).scalars().all()
                if not rows:
                    await asyncio.sleep(0.2)
                    continue
                for r in rows:
                    try:
                        # publish to Redis Stream with idempotency key (outbox id)
                        await self.redis.xadd(self.stream, {"outbox_id": str(r.id), "payload": json.dumps(r.payload)})
                        await s.execute(update(Outbox).where(Outbox.id == r.id).values(status="SENT"))
                        await s.commit()
                    except Exception as e:
                        new_retries = r.retries + 1
                        if new_retries > self.max_retries:
                            await s.execute(update(Outbox).where(Outbox.id == r.id).values(status="ERROR"))
                            s.add(OutboxDLQ(payload=r.payload, last_error=str(e), retries=new_retries))
                        else:
                            await s.execute(update(Outbox).where(Outbox.id == r.id).values(retries=new_retries))
                        await s.commit()

# === path: app/health.py
from fastapi import APIRouter
from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

router = APIRouter()

@router.get("/health")
def health_root():
    return {"status": "ok"}

@router.get("/metrics")
def metrics():
    reg = CollectorRegistry()  # extend to register custom metrics
    data = generate_latest(reg)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

# === path: app/sse.py
import asyncio, json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis
from .config import settings

router = APIRouter()

async def _event_stream(redis: Redis, last_id: str | None):
    stream = settings.SSE_REDIS_STREAM
    start_id = last_id or "$"
    while True:
        results = await redis.xread({stream: start_id}, block=15000, count=100)
        if results:
            _, entries = results[0]
            for _id, fields in entries:
                payload = fields.get(b"payload") or fields.get("payload")
                if payload:
                    yield f"id: {_id}\n"
                    yield "event: news\n"
                    yield f"data: {payload.decode() if isinstance(payload, bytes) else payload}\n\n"
                start_id = _id
        else:
            # heartbeat
            yield ": keep-alive\n\n"

@router.get("/stream/news")
async def stream_news(request: Request):
    redis = request.app.state.redis
    last_event_id = request.headers.get("Last-Event-ID")
    return StreamingResponse(_event_stream(redis, last_event_id), media_type="text/event-stream")

# === path: app/adapters/base.py
import asyncio, time
from datetime import datetime, timezone, timedelta
from typing import AsyncIterator
import structlog

log = structlog.get_logger(__name__)

class CircuitBreaker:
    def __init__(self, trip_rate: float, min_window: int, open_seconds: int):
        self.trip_rate = trip_rate
        self.min_window = min_window
        self.open_seconds = open_seconds
        self.window = []  # list[bool] (True = error)
        self.state = "CLOSED"
        self.next_probe = 0.0

    def record(self, error: bool):
        self.window.append(error)
        if len(self.window) > 200:
            self.window = self.window[-200:]
        if self.state == "CLOSED" and len(self.window) >= self.min_window:
            rate = sum(self.window) / len(self.window)
            if rate >= self.trip_rate:
                self.state = "OPEN"
                self.next_probe = time.time() + self.open_seconds
                log.warning("circuit_open", rate=rate)
        elif self.state == "HALF_OPEN":
            if not error:
                # success probe → close
                self.state = "CLOSED"
                self.window.clear()
            else:
                self.state = "OPEN"
                self.next_probe = time.time() + self.open_seconds

    def can_call(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN" and time.time() >= self.next_probe:
            self.state = "HALF_OPEN"
            return True
        return self.state == "HALF_OPEN"

class AdapterBase:
    def __init__(self, name: str, out_queue, circuit: CircuitBreaker):
        self.name = name
        self.out_queue = out_queue
        self.circuit = circuit
        self.safety_lag = timedelta(seconds=60)
        self.since_ts = datetime.now(timezone.utc) - timedelta(minutes=5)

    async def fetch_window(self, start: datetime, end: datetime) -> AsyncIterator[dict]:
        """Override in child; yield provider items {source, external_id, url, title, body?, ...} """
        if False:
            yield {}

    async def run(self):
        while True:
            start = self.since_ts
            end = datetime.now(timezone.utc) - self.safety_lag
            if end <= start:
                await asyncio.sleep(0.5)
                continue
            try:
                if not self.circuit.can_call():
                    await asyncio.sleep(1.0)
                    continue
                async for item in self.fetch_window(start, end):
                    await self.out_queue.put(item)
                self.circuit.record(False)
                self.since_ts = end
            except Exception as e:
                log.error("adapter_error", source=self.name, err=str(e))
                self.circuit.record(True)
                await asyncio.sleep(1.0)

# === path: app/adapters/fake_feed.py
# A minimal fake adapter to prove the pipeline; replace with real providers.
import asyncio
from datetime import datetime, timezone, timedelta
from .base import AdapterBase

class FakeAdapter(AdapterBase):
    def __init__(self, out_queue, circuit):
        super().__init__("fake", out_queue, circuit)
        self._counter = 0

    async def fetch_window(self, start: datetime, end: datetime):
        # generate 3 fake items per tick
        for i in range(3):
            self._counter += 1
            yield {
                "source": "rss",
                "external_id": f"fake-{self._counter}",
                "url": f"https://example.com/{self._counter}",
                "title": f"Sample headline {self._counter}",
                "body": None,
                "authors": None,
                "categories": ["Tech"],
                "language": "en",
                "published_at": datetime.now(timezone.utc) - timedelta(seconds=5),
                "metadata": {"gen": True}
            }
        await asyncio.sleep(0.2)

# === path: app/normalizer.py
from datetime import datetime, timezone
from .schema import RawContentIn
from .ccs import sanitize_html, canon_authors, map_categories

async def normalize(p: dict) -> dict:
    r = RawContentIn(**p)
    body = sanitize_html(r.body)
    authors = canon_authors(r.authors)
    cats = map_categories(r.categories)
    return {
        "source": r.source,
        "external_id": r.external_id,
        "url": r.url,
        "title": r.title,
        "body": body,
        "authors": authors,
        "categories": cats,
        "language": r.language,
        "published_at": r.published_at,
        "metadata": r.metadata or {},
    }

# === path: app/main.py
import asyncio, contextlib, os
from fastapi import FastAPI, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from .config import settings
from .db import get_session, engine, SessionLocal
from .models import Base, Content, IngestAudit
from .health import router as health_router
from .sse import router as sse_router
from .prefilter import Prefilter, IdentityCache
from .batcher import UpsertBuffer
from .outbox import OutboxPublisher
from .adapters.base import CircuitBreaker
from .adapters.fake_feed import FakeAdapter
from .normalizer import normalize
from .utils import exact_dedupe_key
from .watchlist import WatchlistSource

app = FastAPI(title="News-Data-Service (slim)")
app.include_router(health_router)
app.include_router(sse_router)

@app.on_event("startup")
async def _startup():
    # DB init (migrations are applied by Flyway; this ensures models match)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)  # safe no-op with Flyway
    # Redis
    redis = Redis.from_url(settings.REDIS_URL)
    app.state.redis = redis
    # Prefilter + Identity Cache
    app.state.prefilter = Prefilter(redis)
    app.state.id_cache = IdentityCache(redis)
    # Upsert buffer
    app.state.buffer = UpsertBuffer(SessionLocal, settings.UPSERT_BATCH_SIZE, settings.UPSERT_FLUSH_MS)
    await app.state.buffer.start()
    # Outbox publisher
    app.state.publisher = OutboxPublisher(SessionLocal, redis, settings.SSE_REDIS_STREAM)
    await app.state.publisher.start()
    # Adapter pipelines (support multiple)
    app.state.ingest_q: asyncio.Queue[dict] = asyncio.Queue(maxsize=5000)
    circuit = CircuitBreaker(settings.CIRCUIT_TRIP_ERROR_RATE, settings.CIRCUIT_MIN_WINDOW, settings.CIRCUIT_OPEN_SECONDS)

    adapters = []
    if os.getenv("FINNHUB_API_KEY"):
        from .adapters.finnhub import FinnhubAdapter
        adapters.append(FinnhubAdapter(app.state.ingest_q, circuit, category=os.getenv("FINNHUB_NEWS_CATEGORY", "general")))

        # Optional company-news watchlist (env list + file + redis set, auto-refresh)
        static_symbols = [s.strip().upper() for s in os.getenv("FINNHUB_COMPANY_SYMBOLS", "").split(",") if s.strip()]
        wl_file = settings.FINNHUB_COMPANY_WATCHLIST_FILE
        wl_redis_set = settings.FINNHUB_COMPANY_WATCHLIST_REDIS_SET
        watchlist = WatchlistSource(redis=redis, static_symbols=static_symbols, file_path=wl_file, redis_set=wl_redis_set, refresh_seconds=settings.WATCHLIST_REFRESH_SECONDS)
        from .adapters.finnhub_company import FinnhubCompanyAdapter
        adapters.append(FinnhubCompanyAdapter(app.state.ingest_q, circuit, watchlist=watchlist))

    if not adapters:
        adapters = [FakeAdapter(app.state.ingest_q, circuit)]

    app.state.adapter_tasks = [asyncio.create_task(a.run()) for a in adapters]
    app.state.normalize_task = asyncio.create_task(_normalize_loop())

@app.on_event("shutdown")
async def _shutdown():
    with contextlib.suppress(Exception):
        await app.state.buffer.stop()
    with contextlib.suppress(Exception):
        [t.cancel() for t in getattr(app.state, "adapter_tasks", [])]
    with contextlib.suppress(Exception):
        app.state.normalize_task.cancel()
    with contextlib.suppress(Exception):
        await app.state.redis.aclose()

async def _normalize_loop():
    prefilter = app.state.prefilter
    idc = app.state.id_cache
    buf: UpsertBuffer = app.state.buffer
    q: asyncio.Queue = app.state.ingest_q
    while True:
        p = await q.get()
        r = await normalize(p)
        # prefilter exact duplicates
        if await prefilter.is_duplicate(r["title"], r["url"]):
            # audit-only entry
            async with SessionLocal() as s:
                s.add(IngestAudit(
                    content_id=None,
                    source=r["source"], external_id=r["external_id"],
                    reason="DEDUP_PREFILTER_HIT", dedupe_key=exact_dedupe_key(r["title"], r["url"]), details=None
                ))
                await s.commit()
            continue
        # identity cache lane decision
        lane = "REGULAR"
        id_hit = await idc.get(r["source"], r["external_id"])
        if id_hit:
            same = id_hit.get("hash") == exact_dedupe_key(r["title"], r["url"]) and id_hit.get("pub_ts") == r["published_at"].isoformat()
            if same:
                lane = "LIKELY_NOOP"
        await buf.enqueue(lane, r)
        await idc.set(r["source"], r["external_id"], {
            "hash": exact_dedupe_key(r["title"], r["url"]),
            "pub_ts": r["published_at"].isoformat(),
        })

@app.get("/news")
async def list_news(
    session: AsyncSession = Depends(get_session),
    q: str | None = Query(None, description="simple ilike filter on title"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    stmt = select(Content).order_by(Content.published_at.desc()).limit(limit).offset(offset)
    if q:
        stmt = select(Content).where(Content.title.ilike(f"%{q}%")).order_by(Content.published_at.desc()).limit(limit).offset(offset)
    rows = (await session.execute(stmt)).scalars().all()
    return [
        {
            "content_id": str(r.content_id),
            "source": r.source,
            "external_id": r.external_id,
            "url": r.url,
            "title": r.title,
            "language": r.language,
            "published_at": r.published_at,
            "as_of_ts": r.as_of_ts,
            "revision_seq": r.revision_seq,
            "status": r.status,
        }
        for r in rows
    ]

# === path: migrations/V1__init_schema.sql
-- Flyway SQL: core tables and constraints
CREATE TABLE IF NOT EXISTS content (
  content_id uuid primary key default gen_random_uuid(),
  source text not null,
  external_id text not null,
  url text not null,
  title text not null,
  body text null,
  authors jsonb null,
  categories jsonb null,
  language text null,
  published_at timestamptz not null,
  ingested_at timestamptz not null default now(),
  as_of_ts timestamptz not null,
  revision_seq int not null default 0,
  status text not null default 'active',
  metadata jsonb null,
  exact_dedupe_key text not null
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_source_external ON content(source, external_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_exact_dedupe ON content(exact_dedupe_key);
CREATE INDEX IF NOT EXISTS idx_content_published_at ON content(published_at desc);

CREATE TABLE IF NOT EXISTS outbox (
  id bigserial primary key,
  payload jsonb not null,
  status text not null default 'PENDING',
  retries int not null default 0,
  created_at timestamptz not null default now()
);

CREATE TABLE IF NOT EXISTS outbox_dlq (
  id bigserial primary key,
  payload jsonb not null,
  last_error text null,
  retries int not null,
  created_at timestamptz not null default now()
);

CREATE TABLE IF NOT EXISTS ingest_audit (
  id bigserial primary key,
  content_id uuid null,
  source text not null,
  external_id text not null,
  reason text not null,
  dedupe_key text null,
  details jsonb null,
  created_at timestamptz not null default now()
);

-- Timescale hypertable optional
-- SELECT create_hypertable('content','published_at', if_not_exists => TRUE);

# === path: migrations/V2__timescale_and_compression.sql
-- Optional: enable Timescale compression policy after 7d
-- ALTER TABLE content SET (timescaledb.compress, timescaledb.compress_orderby='published_at', timescaledb.compress_segmentby='source');
-- SELECT add_compression_policy('content', INTERVAL '7 days');

# === path: README.md
# News-Data-Service (slim)

## Quick start
```bash
docker compose up --build
```
Visit:
- API: http://localhost:8080/news
- Health: http://localhost:8080/health
- SSE: http://localhost:8080/stream/news (use a browser tab)

The service uses a **FakeAdapter** to generate sample headlines. Replace it with real provider adapters in `app/adapters/`.

## Replace FakeAdapter with real providers
- Create `app/adapters/finnhub.py` implementing `fetch_window(start,end)` and yielding dicts per `schema.RawContentIn`.
- Wire it in `app/main.py` instead of `FakeAdapter`.

## Design highlights
- Redis **SETNX** prefilter for exact duplicates (TTL=48h)
- Identity cache to route likely-noop updates to a lighter lane
- Async upsert buffer (batching N=200 or 20ms)
- Transactional outbox → Redis Streams (`news.events`)
- Circuit breaker around adapters
- PIT-safe fields: `as_of_ts=min(published_at, ingested_at)`; revisions bump `revision_seq`

## Migrations
Flyway runs automatically via docker-compose (`migrations/*.sql`).

## Environment
- `NEWS_PG_DSN` Postgres DSN (async)
- `REDIS_URL` Redis connection URL
- `SSE_REDIS_STREAM` Redis Stream for events (default `news.events`)

## What’s NOT here (by design)
- No ticker linking, sentiment, topics, story edges, novelty, near-dup ML. Those belong to sibling services.

# === path: app/adapters/finnhub.py
import os
import httpx
from datetime import datetime, timezone
from .base import AdapterBase

class FinnhubAdapter(AdapterBase):
    """Ingest Finnhub market news via /news?category=… using minId watermark.
    Docs: https://finnhub.io/docs/api (Get latest market news → /news?category=general; supports minId)
    """

    def __init__(self, out_queue, circuit, category: str = "general"):
        super().__init__("finnhub", out_queue, circuit)
        self.base = "https://finnhub.io/api/v1"
        self.token = os.getenv("FINNHUB_API_KEY", "")
        if not self.token:
            raise RuntimeError("FINNHUB_API_KEY not set")
        self.category = category
        self.min_id = 0  # Finnhub 'id' watermark

    async def fetch_window(self, start, end):
        # Finnhub market news doesn't use time windows for /news; it uses minId paging.
        # We still respect the adapter's cadence and safety lag by calling once per tick.
        params = {"category": self.category, "token": self.token}
        if self.min_id:
            params["minId"] = self.min_id
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{self.base}/news", params=params)
            r.raise_for_status()
            items = r.json() or []
        # Track max id as new watermark
        max_id = self.min_id
        for it in items:
            try:
                fid = int(it.get("id") or 0)
                if fid > max_id:
                    max_id = fid
            except Exception:
                pass
            dt = it.get("datetime")
            published_at = datetime.fromtimestamp(dt, tz=timezone.utc) if isinstance(dt, (int, float)) else datetime.now(timezone.utc)
            yield {
                "source": "finnhub",
                "external_id": str(it.get("id") or it.get("url")),
                "url": it.get("url") or "",
                "title": it.get("headline") or "",
                "body": it.get("summary") or None,
                "authors": None,
                "categories": [self.category],
                "language": "en",  # Finnhub market news is predominantly English
                "published_at": published_at,
                "metadata": {"source": it.get("source"), "image": it.get("image")},
            }
        self.min_id = max_id

# === path: tests/test_news_service.py
import asyncio
import os
import json
import pytest
from datetime import datetime, timezone, timedelta
from redis.asyncio import Redis

from app.db import SessionLocal
from app.models import Content, IngestAudit, Outbox
from app.batcher import UpsertBuffer
from app.prefilter import Prefilter, IdentityCache
from app.utils import exact_dedupe_key

pytestmark = pytest.mark.asyncio

async def _drain(session):
    # helper to count rows
    c = (await session.execute(Content.__table__.select())).all()
    a = (await session.execute(IngestAudit.__table__.select())).all()
    o = (await session.execute(Outbox.__table__.select())).all()
    return len(c), len(a), len(o)

async def _make_item(i=1, title=None):
    now = datetime.now(timezone.utc)
    return {
        "source": "rss",
        "external_id": f"x-{i}",
        "url": f"https://example.com/{i}",
        "title": title or f"Hello {i}",
        "body": None,
        "authors": None,
        "categories": ["Tech"],
        "language": "en",
        "published_at": now - timedelta(seconds=2),
        "metadata": {"t": i}
    }

@pytest.fixture(scope="module")
async def redis():
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    r = Redis.from_url(url)
    yield r
    await r.aclose()

@pytest.fixture(scope="function")
async def session():
    async with SessionLocal() as s:
        # clean tables between tests
        for t in [Outbox.__table__, IngestAudit.__table__, Content.__table__]:
            await s.execute(t.delete())
        await s.commit()
        yield s

async def test_prefilter_and_upsert(session, redis):
    pf = Prefilter(redis)
    idc = IdentityCache(redis)
    buf = UpsertBuffer(SessionLocal, batch_size=50, flush_ms=10)
    await buf.start()

    r1 = await _make_item(1)
    # First insert should pass prefilter
    dup1 = await pf.is_duplicate(r1["title"], r1["url"])
    assert dup1 is False
    await buf.enqueue("REGULAR", r1)

    # Second time same title/url → prefilter hit
    dup2 = await pf.is_duplicate(r1["title"], r1["url"])
    assert dup2 is True

    # wait for batcher
    await asyncio.sleep(0.1)

    # Only one content row
    c,a,o = await _drain(session)
    assert c == 1

async def test_revision_and_outbox(session, redis):
    buf = UpsertBuffer(SessionLocal, batch_size=10, flush_ms=5)
    await buf.start()

    r1 = await _make_item(2, title="Initial Title")
    await buf.enqueue("REGULAR", r1)
    await asyncio.sleep(0.05)

    # correction: same identity, new title
    r2 = dict(r1)
    r2["title"] = "Corrected Title"
    await buf.enqueue("REGULAR", r2)
    await asyncio.sleep(0.1)

    # one content row with revision_seq=1 and an outbox corrected event present
    rows = (await session.execute(Content.__table__.select())).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.revision_seq == 1

    out = (await session.execute(Outbox.__table__.select())).all()
    # expect 2 events: created + corrected
    assert len(out) == 2

# === path: README-Finnhub.md
# Finnhub Adapter

Set an API key in the environment and (optionally) a category to ingest Finnhub market news via `/news` with `minId` paging.

```bash
export FINNHUB_API_KEY=sk_...
export FINNHUB_NEWS_CATEGORY=general   # or forex|crypto|merger

# docker compose uses the env automatically
docker compose up --build
```

The adapter uses: `/api/v1/news?category=...&minId=...&token=...` and tracks the maximum `id` as a watermark per process.

**Notes**
- For company-specific news, create a separate adapter using `/company-news?symbol=…&from=YYYY-MM-DD&to=YYYY-MM-DD` keyed by your watchlist.
- Respect Finnhub rate limits and consider your central RLC if keys are shared across services.

# === path: app/adapters/finnhub_company.py
import os
import httpx
from datetime import datetime, timezone
from .base import AdapterBase

class FinnhubCompanyAdapter(AdapterBase):
    """Company news by symbol via /company-news?symbol=…&from=YYYY-MM-DD&to=YYYY-MM-DD.
    Symbols are pulled from a dynamic WatchlistSource (env list + file + Redis set) with periodic refresh.
    Docs: https://finnhub.io/docs/api/company-news
    """

    def __init__(self, out_queue, circuit, watchlist):
        super().__init__("finnhub-company", out_queue, circuit)
        self.base = "https://finnhub.io/api/v1"
        self.token = os.getenv("FINNHUB_API_KEY", "")
        if not self.token:
            raise RuntimeError("FINNHUB_API_KEY not set")
        self.watchlist = watchlist

    async def fetch_window(self, start, end):
        f = start.date().isoformat()
        t = end.date().isoformat()
        symbols = await self.watchlist.get_symbols()
        if not symbols:
            return
        async with httpx.AsyncClient(timeout=20) as client:
            for sym in symbols:
                params = {"symbol": sym, "from": f, "to": t, "token": self.token}
                r = await client.get(f"{self.base}/company-news", params=params)
                r.raise_for_status()
                items = r.json() or []
                for it in items:
                    dt = it.get("datetime")
                    published_at = datetime.fromtimestamp(dt, tz=timezone.utc) if isinstance(dt, (int, float)) else datetime.now(timezone.utc)
                    yield {
                        "source": "finnhub",
                        "external_id": str(it.get("id") or f"{sym}:{it.get('url')}") ,
                        "url": it.get("url") or "",
                        "title": it.get("headline") or "",
                        "body": it.get("summary") or None,
                        "authors": None,
                        "categories": ["company"],
                        "language": "en",
                        "published_at": published_at,
                        "metadata": {"symbol": sym, "source": it.get("source"), "image": it.get("image")},
                    }


# === path: app/watchlist.py
import os, time
from typing import Iterable, List, Optional
from redis.asyncio import Redis

class WatchlistSource:
    """Dynamic watchlist that merges: static env list + file + Redis set.
    - File: newline OR comma-separated, supports comments with '#'.
    - Redis set: SMEMBERS of a given key.
    - Auto-refreshes every `refresh_seconds`.
    """

    def __init__(self, redis: Redis, static_symbols: Optional[List[str]] = None, file_path: Optional[str] = None, redis_set: Optional[str] = None, refresh_seconds: int = 60):
        self.redis = redis
        self.static_symbols = [s.strip().upper() for s in (static_symbols or []) if s.strip()]
        self.file_path = file_path
        self.redis_set = redis_set
        self.refresh_seconds = max(5, int(refresh_seconds))
        self._cache: List[str] = []
        self._last = 0.0

    async def get_symbols(self) -> List[str]:
        now = time.time()
        if now - self._last < self.refresh_seconds and self._cache:
            return self._cache
        syms = set(self.static_symbols)
        # file symbols
        if self.file_path and os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    txt = f.read()
                for tok in self._parse_tokens(txt):
                    syms.add(tok)
            except Exception:
                pass
        # redis symbols
        if self.redis and self.redis_set:
            try:
                raw = await self.redis.smembers(self.redis_set)
                for b in raw:
                    tok = (b.decode() if isinstance(b, (bytes, bytearray)) else str(b)).strip().upper()
                    if tok:
                        syms.add(tok)
            except Exception:
                pass
        self._cache = sorted(syms)
        self._last = now
        return self._cache

    @staticmethod
    def _parse_tokens(text: str) -> Iterable[str]:
        # split by newline and commas; ignore comments
        for line in text.splitlines():
            line = line.split('#', 1)[0]
            for tok in line.replace('	', ',').split(','):
                tok = tok.strip().upper()
                if tok:
                    yield tok

# === path: README-Watchlist.md
# Dynamic Company Watchlist

You can feed **company-news** symbols via a static env list, a file, and/or a Redis set. The service merges them and auto-refreshes (default every 60s).

## Options
- `FINNHUB_COMPANY_SYMBOLS` – comma-separated (e.g., `AAPL,MSFT,TSLA`).
- `FINNHUB_COMPANY_WATCHLIST_FILE` – path to a text file (mounted by Docker). Default `/app/watchlist/watchlist.txt`.
- `FINNHUB_COMPANY_WATCHLIST_REDIS_SET` – Redis set key to read symbols from (e.g., `news:watchlist:company`).
- `WATCHLIST_REFRESH_SECONDS` – poll interval for changes (default 60).

## Examples
### File-based (edit without redeploy)
1. Put symbols in `watchlist.txt` (one per line or comma-separated):
   ```
   AAPL, MSFT
   TSLA
   # comments okay
   AMD
   ```
2. Run the stack:
   ```bash
   docker compose up --build
   ```
3. Update the file anytime; within ~60s the adapter will pick up the changes.

### Redis-based (live updates)
```bash
# add symbols
redis-cli SADD news:watchlist:company AAPL MSFT TSLA
# remove a symbol
redis-cli SREM news:watchlist:company TSLA
```
The adapter refreshes periodically and uses the new set on the next fetch cycle.

### Combined
You can use env + file + Redis together; the list is the **union** of all sources.
