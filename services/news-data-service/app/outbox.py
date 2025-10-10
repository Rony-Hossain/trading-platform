import asyncio
import json
from redis.asyncio import Redis
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Outbox, OutboxDLQ
from .config import settings

class OutboxPublisher:
    def __init__(self, session_factory, redis: Redis, stream: str, max_retries: int = 20):
        self._sf = session_factory
        self.redis = redis
        self.stream = stream
        self.max_retries = max_retries
        self._task: asyncio.Task | None = None

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

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
                        # ✅ Added MAXLEN to prevent unbounded growth
                        await self.redis.xadd(
                            self.stream,
                            {"outbox_id": str(r.id), "payload": json.dumps(r.payload)},
                            maxlen=settings.REDIS_STREAM_MAXLEN,
                            approximate=True
                        )
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
