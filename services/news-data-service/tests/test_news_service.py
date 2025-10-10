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
