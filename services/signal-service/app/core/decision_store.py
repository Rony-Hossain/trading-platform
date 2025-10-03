"""
Decision Store
Immutable audit trail for all plan decisions
"""
import json
import hashlib
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from redis import Redis
import structlog

from .contracts import Pick, ResponseMetadata

logger = structlog.get_logger(__name__)


class DecisionSnapshot:
    """Immutable decision snapshot"""

    def __init__(
        self,
        request_id: str,
        user_id: str,
        inputs: Dict,
        picks: List[Dict],
        metadata: Dict,
        degraded_fields: Optional[List[str]] = None
    ):
        self.request_id = request_id
        self.user_id = user_id
        self.timestamp = datetime.utcnow().isoformat()
        self.inputs = inputs
        self.picks = picks
        self.metadata = metadata
        self.degraded_fields = degraded_fields or []
        self.snapshot_version = "1.0"

        # Calculate integrity hash
        self.snapshot_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash for integrity verification"""
        snapshot_dict = {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "inputs": self.inputs,
            "picks": self.picks,
            "metadata": self.metadata,
            "degraded_fields": self.degraded_fields,
            "snapshot_version": self.snapshot_version
        }

        snapshot_json = json.dumps(snapshot_dict, sort_keys=True)
        return hashlib.sha256(snapshot_json.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "inputs": self.inputs,
            "picks": self.picks,
            "metadata": self.metadata,
            "degraded_fields": self.degraded_fields,
            "snapshot_version": self.snapshot_version,
            "snapshot_hash": self.snapshot_hash
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DecisionSnapshot':
        """Create from dictionary"""
        snapshot = cls(
            request_id=data["request_id"],
            user_id=data["user_id"],
            inputs=data["inputs"],
            picks=data["picks"],
            metadata=data["metadata"],
            degraded_fields=data.get("degraded_fields", [])
        )
        snapshot.timestamp = data["timestamp"]
        snapshot.snapshot_version = data.get("snapshot_version", "1.0")
        snapshot.snapshot_hash = data.get("snapshot_hash", "")
        return snapshot


class DecisionStore:
    """
    Decision Store - Immutable Audit Trail

    Features:
    - Store every /plan decision snapshot in Redis
    - 30-day retention (configurable)
    - SHA-256 hash for integrity verification
    - User history index (last 100 decisions per user)
    - Fast retrieval by request_id
    """

    def __init__(self, redis_client: Redis, ttl_days: int = 30):
        self.redis = redis_client
        self.ttl_days = ttl_days
        self.ttl_seconds = ttl_days * 24 * 60 * 60

    def save_snapshot(
        self,
        request_id: str,
        user_id: str,
        inputs: Dict,
        picks: List[Pick],
        metadata: ResponseMetadata,
        degraded_fields: Optional[List[str]] = None
    ) -> str:
        """
        Save decision snapshot

        Args:
            request_id: Unique request ID (ULID)
            user_id: User ID
            inputs: Input parameters (watchlist, mode, risk_profile, cash_available)
            picks: List of Pick objects
            metadata: Response metadata
            degraded_fields: List of degraded services (if any)

        Returns:
            snapshot_hash: SHA-256 hash for integrity verification
        """
        try:
            # Serialize picks
            serialized_picks = [self._serialize_pick(p) for p in picks]

            # Serialize metadata
            serialized_metadata = {
                "source_models": [m.dict() for m in metadata.source_models],
                "version": metadata.version,
                "latency_ms": metadata.latency_ms,
                "generated_at": metadata.generated_at.isoformat()
            }

            # Create snapshot
            snapshot = DecisionSnapshot(
                request_id=request_id,
                user_id=user_id,
                inputs=inputs,
                picks=serialized_picks,
                metadata=serialized_metadata,
                degraded_fields=degraded_fields
            )

            # Store in Redis
            key = f"decision:snapshot:{request_id}"
            self.redis.setex(
                key,
                self.ttl_seconds,
                json.dumps(snapshot.to_dict())
            )

            # Update user index
            self._update_user_index(user_id, request_id)

            logger.info(
                "decision_snapshot_saved",
                request_id=request_id,
                user_id=user_id,
                picks_count=len(picks),
                snapshot_hash=snapshot.snapshot_hash,
                ttl_days=self.ttl_days
            )

            return snapshot.snapshot_hash

        except Exception as e:
            logger.error(
                "decision_snapshot_save_failed",
                request_id=request_id,
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise

    def _serialize_pick(self, pick: Pick) -> Dict:
        """Serialize pick for snapshot"""
        return {
            "symbol": pick.symbol,
            "action": pick.action,
            "shares": pick.shares,
            "entry_hint": pick.entry_hint,
            "safety_line": pick.safety_line,
            "target": pick.target,
            "confidence": pick.confidence,
            "reason": pick.reason,
            "reason_codes": [str(rc) for rc in pick.reason_codes],
            "decision_path": pick.decision_path,
            "constraints": pick.constraints.dict(),
            "limits_applied": pick.limits_applied.dict(),
            "max_risk_usd": pick.max_risk_usd,
            "reason_score": pick.reason_score
        }

    def _update_user_index(self, user_id: str, request_id: str):
        """Update user's recent decisions index"""
        user_key = f"decision:user:{user_id}:recent"

        # Add to list (left push)
        self.redis.lpush(user_key, request_id)

        # Keep only last 100
        self.redis.ltrim(user_key, 0, 99)

        # Set TTL
        self.redis.expire(user_key, self.ttl_seconds)

    def get_snapshot(self, request_id: str) -> Optional[DecisionSnapshot]:
        """
        Retrieve decision snapshot by request_id

        Args:
            request_id: Request ID

        Returns:
            DecisionSnapshot or None if not found
        """
        try:
            key = f"decision:snapshot:{request_id}"
            data = self.redis.get(key)

            if not data:
                logger.warning(
                    "decision_snapshot_not_found",
                    request_id=request_id
                )
                return None

            snapshot_dict = json.loads(data)
            snapshot = DecisionSnapshot.from_dict(snapshot_dict)

            logger.debug(
                "decision_snapshot_retrieved",
                request_id=request_id,
                snapshot_hash=snapshot.snapshot_hash
            )

            return snapshot

        except Exception as e:
            logger.error(
                "decision_snapshot_retrieval_failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            return None

    def get_user_decisions(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[DecisionSnapshot]:
        """
        Get recent decisions for user

        Args:
            user_id: User ID
            limit: Maximum number of decisions to return (max 100)

        Returns:
            List of DecisionSnapshot objects (newest first)
        """
        try:
            limit = min(limit, 100)  # Cap at 100
            user_key = f"decision:user:{user_id}:recent"

            # Get request IDs from user index
            request_ids = self.redis.lrange(user_key, 0, limit - 1)

            if not request_ids:
                logger.debug(
                    "no_user_decisions_found",
                    user_id=user_id
                )
                return []

            # Retrieve snapshots
            snapshots = []
            for req_id in request_ids:
                snapshot = self.get_snapshot(req_id.decode())
                if snapshot:
                    snapshots.append(snapshot)

            logger.info(
                "user_decisions_retrieved",
                user_id=user_id,
                count=len(snapshots),
                requested_limit=limit
            )

            return snapshots

        except Exception as e:
            logger.error(
                "user_decisions_retrieval_failed",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            return []

    def verify_integrity(self, snapshot: DecisionSnapshot) -> bool:
        """
        Verify snapshot integrity using hash

        Args:
            snapshot: DecisionSnapshot to verify

        Returns:
            True if integrity check passes
        """
        try:
            # Recalculate hash
            stored_hash = snapshot.snapshot_hash
            snapshot.snapshot_hash = ""  # Clear for recalculation
            calculated_hash = snapshot._calculate_hash()

            matches = stored_hash == calculated_hash

            if not matches:
                logger.warning(
                    "snapshot_integrity_check_failed",
                    request_id=snapshot.request_id,
                    stored_hash=stored_hash,
                    calculated_hash=calculated_hash
                )

            return matches

        except Exception as e:
            logger.error(
                "snapshot_integrity_verification_failed",
                request_id=snapshot.request_id,
                error=str(e)
            )
            return False

    def delete_snapshot(self, request_id: str) -> bool:
        """
        Delete decision snapshot (use with caution - violates immutability)

        Args:
            request_id: Request ID

        Returns:
            True if deleted
        """
        logger.warning(
            "decision_snapshot_deletion_requested",
            request_id=request_id,
            message="This violates immutability - use only for compliance/legal reasons"
        )

        key = f"decision:snapshot:{request_id}"
        result = self.redis.delete(key)
        return result > 0

    def get_stats(self) -> Dict:
        """
        Get decision store statistics

        Returns:
            Dict with stats (total snapshots, oldest, newest)
        """
        try:
            # Count total snapshots (expensive - only for monitoring)
            pattern = "decision:snapshot:*"
            cursor = 0
            total_snapshots = 0

            while True:
                cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
                total_snapshots += len(keys)
                if cursor == 0:
                    break

            return {
                "total_snapshots": total_snapshots,
                "ttl_days": self.ttl_days
            }

        except Exception as e:
            logger.error("decision_store_stats_failed", error=str(e))
            return {}


# Global decision store instance
_decision_store: Optional[DecisionStore] = None


def init_decision_store(redis_client: Redis, ttl_days: int = 30) -> DecisionStore:
    """
    Initialize global decision store

    Args:
        redis_client: Redis client
        ttl_days: Retention period in days

    Returns:
        DecisionStore instance
    """
    global _decision_store
    _decision_store = DecisionStore(redis_client, ttl_days)
    logger.info("decision_store_initialized", ttl_days=ttl_days)
    return _decision_store


def get_decision_store() -> DecisionStore:
    """
    Get global decision store instance

    Returns:
        DecisionStore instance

    Raises:
        RuntimeError: If not initialized
    """
    if _decision_store is None:
        raise RuntimeError("DecisionStore not initialized. Call init_decision_store() first.")
    return _decision_store
