from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
import structlog
from typing import AsyncGenerator

from .config import settings
from .models import Base

logger = structlog.get_logger(__name__)

# Create async engine
engine = create_async_engine(
    settings.async_database_url,
    echo=settings.DEBUG,
    poolclass=NullPool if settings.DEBUG else None,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# Create session maker
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def init_db():
    """Initialize database tables"""
    try:
        async with engine.begin() as conn:
            # Create tables if they don't exist
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", error=str(e))
            raise
        finally:
            await session.close()

# Database utilities
async def execute_raw_sql(query: str, params: dict = None):
    """Execute raw SQL query"""
    async with AsyncSessionLocal() as session:
        try:
            result = await session.execute(query, params or {})
            await session.commit()
            return result
        except Exception as e:
            await session.rollback()
            logger.error("Raw SQL execution failed", query=query, error=str(e))
            raise