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
