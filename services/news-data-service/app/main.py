import asyncio
import contextlib
import os
from fastapi import FastAPI, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from .config import settings
from .db import get_session, engine, SessionLocal
from .models import Base, Content, IngestAudit
from .health import router as health_router
from .sse import router as sse_router
from .backfill import router as backfill_router
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
app.include_router(backfill_router)

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
        await app.state.publisher.stop()
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
