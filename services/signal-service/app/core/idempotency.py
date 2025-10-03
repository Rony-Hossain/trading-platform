"""
Idempotency Manager
Prevents duplicate execution of buy/sell actions
"""
import json
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from redis import Redis
from pydantic import BaseModel
import structlog

logger = structlog.get_logger(__name__)


class ActionRecord(BaseModel):
    """Record of an executed action"""
    idempotency_key: str
    user_id: str
    action_type: str  # "buy", "sell"
    symbol: str
    shares: int
    status: str  # "pending", "executed", "failed"
    result: Optional[Dict] = None
    created_at: str
    executed_at: Optional[str] = None
    error_message: Optional[str] = None


class IdempotencyManager:
    """
    Idempotency Manager
    Ensures actions are executed exactly once using Redis

    Features:
    - Prevent duplicate buy/sell actions
    - Store action records with TTL (5 minutes, 10 minutes after success)
    - Return same result for duplicate requests
    - Thread-safe using Redis atomic operations
    """

    def __init__(self, redis_client: Redis, ttl_seconds: int = 300):
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds
        self.success_ttl_seconds = ttl_seconds * 2  # Longer TTL for successful actions

    def check_or_create(
        self,
        idempotency_key: str,
        user_id: str,
        action_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[ActionRecord]]:
        """
        Check if action already executed, or create new record

        Args:
            idempotency_key: Unique key (ULID from client)
            user_id: User ID
            action_data: Action details (action_type, symbol, shares)

        Returns:
            (is_duplicate, existing_record)
            - is_duplicate: True if already processed
            - existing_record: ActionRecord if duplicate, otherwise new record
        """
        try:
            key = f"action:idem:{idempotency_key}"

            # Check if exists (atomic)
            existing = self.redis.get(key)

            if existing:
                # Already processed - return existing record
                record = ActionRecord(**json.loads(existing))
                logger.info(
                    "idempotent_action_duplicate_detected",
                    idempotency_key=idempotency_key,
                    user_id=user_id,
                    status=record.status,
                    action_type=action_data.get('action_type'),
                    symbol=action_data.get('symbol')
                )
                return True, record

            # Create new record
            record = ActionRecord(
                idempotency_key=idempotency_key,
                user_id=user_id,
                action_type=action_data['action_type'],
                symbol=action_data['symbol'],
                shares=action_data['shares'],
                status="pending",
                created_at=datetime.utcnow().isoformat()
            )

            # Store with TTL (using SET NX for atomic check-and-set)
            stored = self.redis.set(
                key,
                record.json(),
                nx=True,  # Only set if doesn't exist
                ex=self.ttl_seconds
            )

            if not stored:
                # Race condition - another request created it
                existing = self.redis.get(key)
                if existing:
                    record = ActionRecord(**json.loads(existing))
                    logger.warning(
                        "idempotent_action_race_condition",
                        idempotency_key=idempotency_key,
                        user_id=user_id
                    )
                    return True, record

            logger.info(
                "idempotent_action_created",
                idempotency_key=idempotency_key,
                user_id=user_id,
                action_type=record.action_type,
                symbol=record.symbol,
                shares=record.shares
            )

            return False, record

        except Exception as e:
            logger.error(
                "idempotency_check_failed",
                idempotency_key=idempotency_key,
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            # On error, assume not duplicate to allow action
            return False, None

    def update_status(
        self,
        idempotency_key: str,
        status: str,
        result: Optional[Dict] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update action status after execution

        Args:
            idempotency_key: Idempotency key
            status: New status ("executed", "failed")
            result: Execution result (for successful actions)
            error_message: Error message (for failed actions)

        Returns:
            True if updated successfully
        """
        try:
            key = f"action:idem:{idempotency_key}"
            existing = self.redis.get(key)

            if not existing:
                logger.warning(
                    "idempotent_action_not_found_for_update",
                    idempotency_key=idempotency_key,
                    status=status
                )
                return False

            # Parse existing record
            record = ActionRecord(**json.loads(existing))

            # Update fields
            record.status = status
            record.executed_at = datetime.utcnow().isoformat()
            if result:
                record.result = result
            if error_message:
                record.error_message = error_message

            # Determine TTL based on status
            ttl = self.success_ttl_seconds if status == "executed" else self.ttl_seconds

            # Store updated record
            self.redis.setex(
                key,
                ttl,
                record.json()
            )

            logger.info(
                "idempotent_action_updated",
                idempotency_key=idempotency_key,
                status=status,
                ttl_seconds=ttl,
                has_result=result is not None
            )

            return True

        except Exception as e:
            logger.error(
                "idempotency_update_failed",
                idempotency_key=idempotency_key,
                error=str(e),
                exc_info=True
            )
            return False

    def get_action(self, idempotency_key: str) -> Optional[ActionRecord]:
        """
        Get action record by idempotency key

        Args:
            idempotency_key: Idempotency key

        Returns:
            ActionRecord or None if not found
        """
        try:
            key = f"action:idem:{idempotency_key}"
            data = self.redis.get(key)

            if not data:
                return None

            record = ActionRecord(**json.loads(data))
            return record

        except Exception as e:
            logger.error(
                "get_action_failed",
                idempotency_key=idempotency_key,
                error=str(e)
            )
            return None

    def delete_action(self, idempotency_key: str) -> bool:
        """
        Delete action record (use with caution)

        Args:
            idempotency_key: Idempotency key

        Returns:
            True if deleted
        """
        try:
            key = f"action:idem:{idempotency_key}"
            result = self.redis.delete(key)

            logger.warning(
                "idempotent_action_deleted",
                idempotency_key=idempotency_key,
                message="Manual deletion - use only for cleanup"
            )

            return result > 0

        except Exception as e:
            logger.error(
                "delete_action_failed",
                idempotency_key=idempotency_key,
                error=str(e)
            )
            return False

    def get_user_actions(
        self,
        user_id: str,
        limit: int = 50
    ) -> list[ActionRecord]:
        """
        Get recent actions for user (for debugging/support)

        Note: This scans Redis keys - use sparingly in production

        Args:
            user_id: User ID
            limit: Maximum actions to return

        Returns:
            List of ActionRecord objects
        """
        try:
            pattern = "action:idem:*"
            cursor = 0
            actions = []

            # Scan for keys (expensive operation)
            while len(actions) < limit:
                cursor, keys = self.redis.scan(
                    cursor,
                    match=pattern,
                    count=100
                )

                for key in keys:
                    data = self.redis.get(key)
                    if data:
                        try:
                            record = ActionRecord(**json.loads(data))
                            if record.user_id == user_id:
                                actions.append(record)
                                if len(actions) >= limit:
                                    break
                        except Exception:
                            continue

                if cursor == 0:
                    break

            # Sort by created_at (newest first)
            actions.sort(key=lambda x: x.created_at, reverse=True)

            logger.info(
                "user_actions_retrieved",
                user_id=user_id,
                count=len(actions)
            )

            return actions[:limit]

        except Exception as e:
            logger.error(
                "get_user_actions_failed",
                user_id=user_id,
                error=str(e)
            )
            return []

    def cleanup_expired(self) -> int:
        """
        Cleanup expired idempotency keys (Redis handles this automatically via TTL)
        This method is for manual cleanup if needed

        Returns:
            Number of keys cleaned up
        """
        # Redis TTL handles automatic cleanup
        # This is a no-op, kept for interface compatibility
        logger.info("cleanup_expired_called", message="Redis TTL handles automatic cleanup")
        return 0

    def get_stats(self) -> Dict[str, int]:
        """
        Get idempotency manager statistics

        Returns:
            Dict with stats
        """
        try:
            pattern = "action:idem:*"
            cursor = 0
            total_actions = 0
            status_counts = {"pending": 0, "executed": 0, "failed": 0}

            # Count keys and statuses
            while True:
                cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
                total_actions += len(keys)

                for key in keys:
                    data = self.redis.get(key)
                    if data:
                        try:
                            record = ActionRecord(**json.loads(data))
                            status_counts[record.status] = status_counts.get(record.status, 0) + 1
                        except Exception:
                            continue

                if cursor == 0:
                    break

            return {
                "total_actions": total_actions,
                "pending": status_counts.get("pending", 0),
                "executed": status_counts.get("executed", 0),
                "failed": status_counts.get("failed", 0)
            }

        except Exception as e:
            logger.error("get_stats_failed", error=str(e))
            return {}


# Global idempotency manager instance
_idempotency_manager: Optional[IdempotencyManager] = None


def init_idempotency_manager(
    redis_client: Redis,
    ttl_seconds: int = 300
) -> IdempotencyManager:
    """
    Initialize global idempotency manager

    Args:
        redis_client: Redis client
        ttl_seconds: TTL for action records (default 5 minutes)

    Returns:
        IdempotencyManager instance
    """
    global _idempotency_manager
    _idempotency_manager = IdempotencyManager(redis_client, ttl_seconds)
    logger.info("idempotency_manager_initialized", ttl_seconds=ttl_seconds)
    return _idempotency_manager


def get_idempotency_manager() -> IdempotencyManager:
    """
    Get global idempotency manager instance

    Returns:
        IdempotencyManager instance

    Raises:
        RuntimeError: If not initialized
    """
    if _idempotency_manager is None:
        raise RuntimeError("IdempotencyManager not initialized. Call init_idempotency_manager() first.")
    return _idempotency_manager
