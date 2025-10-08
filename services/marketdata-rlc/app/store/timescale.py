import logging
from typing import Optional

import asyncpg

log = logging.getLogger("rlc.timescale")


class TsPool:
    """Asyncpg connection pool wrapper for TimescaleDB access."""

    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 5, timeout: int = 10) -> None:
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.timeout = timeout
        self.pool: Optional[asyncpg.Pool] = None

    async def start(self) -> None:
        self.pool = await asyncpg.create_pool(
            dsn=self.dsn,
            min_size=self.min_size,
            max_size=self.max_size,
            command_timeout=self.timeout,
        )
        log.info("Timescale pool initialised (max_size=%s)", self.max_size)

    async def stop(self) -> None:
        if self.pool:
            await self.pool.close()
            log.info("Timescale pool closed")


ts_pool: Optional[TsPool] = None
