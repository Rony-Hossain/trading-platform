import asyncio
import contextlib
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
