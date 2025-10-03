"""
Internal API Endpoints
Administrative and monitoring endpoints
"""
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
import structlog

from ...core.slo_tracker import get_slo_tracker
from ...core.policy_manager import get_policy_manager
from ...core.decision_store import get_decision_store
from ...upstream.base_client import UpstreamClient
from ...config import settings

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/slo/status")
async def get_slo_status():
    """
    Get SLO status and error budget

    **Response:**
    - `overall_status`: "healthy", "warning", or "critical"
    - `error_budget`: Availability error budget details
    - `latency`: Latency percentiles for main endpoints

    **Example:**
    ```
    GET /internal/slo/status
    ```
    """
    try:
        slo_tracker = get_slo_tracker()
        status_data = slo_tracker.get_slo_status()

        logger.info(
            "slo_status_retrieved",
            overall_status=status_data["overall_status"]
        )

        return status_data

    except Exception as e:
        logger.error("slo_status_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "SLO_STATUS_FAILED", "message": str(e)}}
        )


@router.get("/slo/error-budget")
async def get_error_budget(window_days: int = 30):
    """
    Get error budget details

    **Query:**
    - `window_days`: Window size in days (default: 30)

    **Example:**
    ```
    GET /internal/slo/error-budget?window_days=7
    ```
    """
    try:
        slo_tracker = get_slo_tracker()
        error_budget = slo_tracker.get_error_budget(window_days=window_days)

        return error_budget

    except Exception as e:
        logger.error("error_budget_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "ERROR_BUDGET_FAILED", "message": str(e)}}
        )


@router.get("/policy/current")
async def get_current_policy():
    """
    Get current policy configuration

    **Example:**
    ```
    GET /internal/policy/current
    ```
    """
    try:
        policy_manager = get_policy_manager()
        policies = policy_manager.policies

        return {
            "version": policies.get("version"),
            "policies": policies
        }

    except Exception as e:
        logger.error("policy_get_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "POLICY_GET_FAILED", "message": str(e)}}
        )


@router.post("/policy/reload")
async def reload_policy():
    """
    Reload policy configuration from YAML

    **Example:**
    ```
    POST /internal/policy/reload
    ```
    """
    try:
        policy_manager = get_policy_manager()
        success = policy_manager.reload()

        if success:
            logger.info("policy_reloaded_via_api")
            return {
                "status": "success",
                "message": "Policy reloaded successfully",
                "version": policy_manager.get("version")
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"code": "POLICY_RELOAD_FAILED", "message": "Failed to reload policy"}}
            )

    except Exception as e:
        logger.error("policy_reload_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "POLICY_RELOAD_FAILED", "message": str(e)}}
        )


@router.get("/decision/{request_id}")
async def get_decision_snapshot(request_id: str):
    """
    Get decision snapshot for audit

    **Path:**
    - `request_id`: Request ID from /plan endpoint

    **Example:**
    ```
    GET /internal/decision/req_abc123
    ```
    """
    try:
        decision_store = get_decision_store()
        snapshot = decision_store.get_snapshot(request_id)

        if not snapshot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"code": "DECISION_NOT_FOUND", "message": f"No decision found for {request_id}"}}
            )

        return snapshot.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error("decision_get_failed", request_id=request_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "DECISION_GET_FAILED", "message": str(e)}}
        )


@router.get("/stats")
async def get_service_stats():
    """
    Get service statistics

    **Example:**
    ```
    GET /internal/stats
    ```
    """
    try:
        from ...core.swr_cache import get_swr_cache_manager

        swr_cache = get_swr_cache_manager()
        cache_stats = await swr_cache.get_stats()

        return {
            "service": settings.SERVICE_NAME,
            "version": settings.VERSION,
            "cache": cache_stats,
            "config": {
                "plan_cache_ttl": settings.PLAN_CACHE_TTL,
                "idempotency_ttl": settings.IDEMPOTENCY_TTL_SECONDS,
                "decision_snapshot_ttl_days": settings.DECISION_SNAPSHOT_TTL_DAYS
            }
        }

    except Exception as e:
        logger.error("stats_get_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "STATS_GET_FAILED", "message": str(e)}}
        )
