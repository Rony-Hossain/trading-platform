"""
Database configuration and utilities for Analysis Service
"""

import os
from typing import Optional

def get_database_url() -> str:
    """Get database URL from environment or use default"""
    default_url = "postgresql://trading_user:trading_password@localhost:5432/trading_db"
    return os.getenv("DATABASE_URL", default_url)

async def test_database_connection() -> bool:
    """Test database connection"""
    try:
        import asyncpg
        conn = await asyncpg.connect(get_database_url())
        await conn.close()
        return True
    except Exception:
        return False