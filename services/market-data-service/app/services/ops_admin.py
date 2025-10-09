"""
DLQ (Dead Letter Queue) Admin Interface.

Provides endpoints for managing failed jobs:
- List failed backfill jobs
- Requeue failed jobs
- View error details
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from app.services.database import DatabaseService


router = APIRouter(prefix="/ops/dlq", tags=["ops-dlq"])


async def get_db() -> DatabaseService:
    """Dependency to get database service (to be overridden in main.py)."""
    raise NotImplementedError("Database dependency not configured")


@router.get("/backfills", response_model=List[Dict[str, Any]])
async def list_backfill_dlq(
    limit: int = 100,
    db: DatabaseService = Depends(get_db)
):
    """
    List failed backfill jobs in the DLQ.

    Args:
        limit: Maximum number of jobs to return (default: 100)
        db: Database service dependency

    Returns:
        List of failed backfill jobs with details
    """
    sql = """
        SELECT
            id,
            symbol,
            interval,
            start_ts,
            end_ts,
            priority,
            last_error,
            attempts,
            created_at,
            updated_at
        FROM backfill_jobs
        WHERE status = 'failed'
        ORDER BY updated_at DESC
        LIMIT $1
    """
    async with db.pool.acquire() as con:
        rows = await con.fetch(sql, limit)
        return [dict(r) for r in rows]


@router.get("/backfills/{job_id}", response_model=Dict[str, Any])
async def get_backfill_details(
    job_id: int,
    db: DatabaseService = Depends(get_db)
):
    """
    Get detailed information about a specific backfill job.

    Args:
        job_id: Backfill job ID
        db: Database service dependency

    Returns:
        Job details including full error trace

    Raises:
        HTTPException 404: Job not found
    """
    sql = """
        SELECT * FROM backfill_jobs WHERE id = $1
    """
    async with db.pool.acquire() as con:
        row = await con.fetchrow(sql, job_id)
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(row)


@router.post("/backfills/requeue/{job_id}")
async def requeue_backfill(
    job_id: int,
    reset_attempts: bool = True,
    db: DatabaseService = Depends(get_db)
):
    """
    Requeue a failed backfill job.

    Args:
        job_id: Backfill job ID to requeue
        reset_attempts: Reset attempt counter (default: True)
        db: Database service dependency

    Returns:
        Success confirmation with job ID

    Raises:
        HTTPException 404: Job not found or not failed
    """
    if reset_attempts:
        sql = """
            UPDATE backfill_jobs
            SET status = 'queued',
                attempts = 0,
                last_error = NULL,
                updated_at = NOW()
            WHERE id = $1 AND status = 'failed'
        """
    else:
        sql = """
            UPDATE backfill_jobs
            SET status = 'queued',
                last_error = NULL,
                updated_at = NOW()
            WHERE id = $1 AND status = 'failed'
        """

    async with db.pool.acquire() as con:
        result = await con.execute(sql, job_id)
        if result == "UPDATE 0":
            raise HTTPException(
                status_code=404,
                detail="Job not found or not in failed state"
            )
        return {"ok": True, "job_id": job_id, "reset_attempts": reset_attempts}


@router.post("/backfills/requeue-all")
async def requeue_all_failed(
    tier: str = None,
    limit: int = 100,
    db: DatabaseService = Depends(get_db)
):
    """
    Requeue multiple failed backfill jobs.

    Args:
        tier: Optional tier filter (T0, T1, T2)
        limit: Maximum number of jobs to requeue (default: 100)
        db: Database service dependency

    Returns:
        Number of jobs requeued
    """
    if tier:
        sql = """
            UPDATE backfill_jobs
            SET status = 'queued',
                attempts = 0,
                last_error = NULL,
                updated_at = NOW()
            WHERE status = 'failed'
                AND priority = $1
                AND id IN (
                    SELECT id FROM backfill_jobs
                    WHERE status = 'failed' AND priority = $1
                    ORDER BY updated_at ASC
                    LIMIT $2
                )
        """
        async with db.pool.acquire() as con:
            result = await con.execute(sql, tier, limit)
    else:
        sql = """
            UPDATE backfill_jobs
            SET status = 'queued',
                attempts = 0,
                last_error = NULL,
                updated_at = NOW()
            WHERE status = 'failed'
                AND id IN (
                    SELECT id FROM backfill_jobs
                    WHERE status = 'failed'
                    ORDER BY updated_at ASC
                    LIMIT $1
                )
        """
        async with db.pool.acquire() as con:
            result = await con.execute(sql, limit)

    # Parse UPDATE N result
    count = int(result.split()[-1]) if result.startswith("UPDATE") else 0
    return {"ok": True, "requeued": count, "tier": tier}


@router.delete("/backfills/{job_id}")
async def delete_failed_backfill(
    job_id: int,
    db: DatabaseService = Depends(get_db)
):
    """
    Permanently delete a failed backfill job from the DLQ.

    Args:
        job_id: Backfill job ID to delete
        db: Database service dependency

    Returns:
        Success confirmation

    Raises:
        HTTPException 404: Job not found or not failed
    """
    sql = "DELETE FROM backfill_jobs WHERE id = $1 AND status = 'failed'"
    async with db.pool.acquire() as con:
        result = await con.execute(sql, job_id)
        if result == "DELETE 0":
            raise HTTPException(
                status_code=404,
                detail="Job not found or not in failed state"
            )
        return {"ok": True, "job_id": job_id, "deleted": True}


@router.get("/stats")
async def get_dlq_stats(db: DatabaseService = Depends(get_db)):
    """
    Get DLQ statistics.

    Returns:
        Statistics about failed jobs by tier and interval
    """
    sql = """
        SELECT
            priority as tier,
            interval,
            COUNT(*) as count,
            MIN(created_at) as oldest,
            MAX(created_at) as newest
        FROM backfill_jobs
        WHERE status = 'failed'
        GROUP BY priority, interval
        ORDER BY priority, interval
    """
    async with db.pool.acquire() as con:
        rows = await con.fetch(sql)
        return {
            "failed_jobs": [dict(r) for r in rows],
            "total_failed": sum(r["count"] for r in rows)
        }
