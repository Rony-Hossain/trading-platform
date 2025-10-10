import os
import time
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
            for tok in line.replace('\t', ',').split(','):
                tok = tok.strip().upper()
                if tok:
                    yield tok
