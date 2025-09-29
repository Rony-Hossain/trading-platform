"""Database utilities for Event Data Service."""

import os
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def _build_database_url() -> str:
    return os.getenv(
        "EVENT_DB_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/event_data",
    )


DATABASE_URL = _build_database_url()
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
SessionFactory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionFactory() as session:
        yield session


async def init_db() -> None:
    from .models import EventORM  # noqa: F401 - ensure models are imported

    async with engine.begin() as conn:
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
        except Exception:
            # Extension creation may fail if not running on TimescaleDB; ignore gracefully
            pass
        await conn.run_sync(Base.metadata.create_all)
        try:
            await conn.execute(
                text(
                    "SELECT create_hypertable('events', 'scheduled_at', if_not_exists => TRUE, migrate_data => TRUE)"
                )
            )
        except Exception:
            # Hypertable creation may fail if TimescaleDB is unavailable or already configured
            pass
