"""
Rollback Controller
Manages model rollback and recovery procedures
Target: Can revert to previous model in < 5 minutes
"""
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json
import joblib
import shutil

logger = logging.getLogger(__name__)


class RollbackController:
    """
    Manages model rollback and recovery
    Ensures quick rollback capability (< 5 minutes)
    """

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.rollback_log_path = self.models_dir / "rollback_log.json"
        self.rollback_history = self._load_rollback_history()

    def _load_rollback_history(self) -> list:
        """Load rollback history"""
        if self.rollback_log_path.exists():
            with open(self.rollback_log_path, 'r') as f:
                return json.load(f)
        return []

    def _save_rollback_history(self):
        """Save rollback history"""
        with open(self.rollback_log_path, 'w') as f:
            json.dump(self.rollback_history, f, indent=2, default=str)

    def create_checkpoint(self, checkpoint_name: str = None) -> str:
        """
        Create a checkpoint of current champion model

        Args:
            checkpoint_name: Optional name for checkpoint

        Returns:
            Checkpoint ID
        """
        start_time = time.time()

        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Creating checkpoint: {checkpoint_name}")

        # Check if champion exists
        champion_path = self.models_dir / "champion.joblib"
        champion_meta_path = self.models_dir / "champion_metadata.json"

        if not champion_path.exists():
            raise FileNotFoundError("No champion model to checkpoint")

        # Create checkpoint directory
        checkpoint_dir = self.models_dir / "checkpoints" / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Copy model
        shutil.copy2(champion_path, checkpoint_dir / "champion.joblib")

        if champion_meta_path.exists():
            shutil.copy2(champion_meta_path, checkpoint_dir / "champion_metadata.json")

        # Save checkpoint metadata
        checkpoint_meta = {
            "checkpoint_id": checkpoint_name,
            "created_at": datetime.utcnow().isoformat(),
            "checkpoint_dir": str(checkpoint_dir)
        }

        checkpoint_meta_path = checkpoint_dir / "checkpoint_meta.json"
        with open(checkpoint_meta_path, 'w') as f:
            json.dump(checkpoint_meta, f, indent=2)

        elapsed_time = time.time() - start_time

        logger.info(f"✓ Checkpoint created in {elapsed_time:.2f}s: {checkpoint_name}")

        return checkpoint_name

    def rollback_to_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Rollback to a specific checkpoint

        Args:
            checkpoint_name: Name of checkpoint to rollback to

        Returns:
            Success status
        """
        start_time = time.time()

        logger.info(f"Rolling back to checkpoint: {checkpoint_name}")

        checkpoint_dir = self.models_dir / "checkpoints" / checkpoint_name

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")

        # Verify checkpoint has required files
        checkpoint_model_path = checkpoint_dir / "champion.joblib"

        if not checkpoint_model_path.exists():
            raise FileNotFoundError(f"Checkpoint model not found: {checkpoint_model_path}")

        # Backup current champion before rollback
        self.create_checkpoint("pre_rollback_backup")

        # Restore checkpoint
        champion_path = self.models_dir / "champion.joblib"
        champion_meta_path = self.models_dir / "champion_metadata.json"

        shutil.copy2(checkpoint_model_path, champion_path)

        checkpoint_meta_path = checkpoint_dir / "champion_metadata.json"
        if checkpoint_meta_path.exists():
            shutil.copy2(checkpoint_meta_path, champion_meta_path)

        elapsed_time = time.time() - start_time

        # Log rollback
        rollback_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "checkpoint_restored": checkpoint_name,
            "rollback_time_seconds": elapsed_time,
            "triggered_by": "manual"
        }

        self.rollback_history.append(rollback_entry)
        self._save_rollback_history()

        logger.info(f"✓ Rollback completed in {elapsed_time:.2f}s")

        # Assert rollback completed in < 5 minutes
        assert elapsed_time < 300, f"Rollback took {elapsed_time:.2f}s (> 5 minutes)"

        return True

    def rollback_to_previous(self) -> bool:
        """
        Quick rollback to most recent archived champion

        Returns:
            Success status
        """
        start_time = time.time()

        logger.info("Rolling back to previous champion...")

        # Find most recent archived champion
        archives = sorted(self.models_dir.glob("champion_archived_*.joblib"), reverse=True)

        if not archives:
            raise FileNotFoundError("No archived champion found for rollback")

        most_recent_archive = archives[0]
        logger.info(f"Found archived champion: {most_recent_archive.name}")

        # Backup current champion
        self.create_checkpoint("pre_quick_rollback_backup")

        # Restore archived champion
        champion_path = self.models_dir / "champion.joblib"
        shutil.copy2(most_recent_archive, champion_path)

        # Restore metadata if exists
        archive_meta = most_recent_archive.with_name(
            most_recent_archive.name.replace('.joblib', '_metadata.json')
        )

        if archive_meta.exists():
            champion_meta_path = self.models_dir / "champion_metadata.json"
            shutil.copy2(archive_meta, champion_meta_path)

        elapsed_time = time.time() - start_time

        # Log rollback
        rollback_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "rollback_type": "quick_rollback",
            "restored_from": str(most_recent_archive),
            "rollback_time_seconds": elapsed_time,
            "triggered_by": "auto"
        }

        self.rollback_history.append(rollback_entry)
        self._save_rollback_history()

        logger.info(f"✓ Quick rollback completed in {elapsed_time:.2f}s")

        # Assert quick rollback in < 5 minutes
        assert elapsed_time < 300, f"Rollback took {elapsed_time:.2f}s (> 5 minutes)"

        return True

    def list_checkpoints(self) -> list:
        """List all available checkpoints"""
        checkpoints_dir = self.models_dir / "checkpoints"

        if not checkpoints_dir.exists():
            return []

        checkpoints = []

        for checkpoint_dir in checkpoints_dir.iterdir():
            if checkpoint_dir.is_dir():
                meta_path = checkpoint_dir / "checkpoint_meta.json"

                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        checkpoints.append(meta)

        return sorted(checkpoints, key=lambda x: x['created_at'], reverse=True)

    def get_rollback_history(self) -> list:
        """Get rollback history"""
        return self.rollback_history

    def verify_rollback_capability(self) -> Dict[str, Any]:
        """
        Verify that rollback can be performed quickly

        Returns:
            Verification results
        """
        logger.info("Verifying rollback capability...")

        results = {
            "can_rollback": False,
            "estimated_time_seconds": None,
            "available_checkpoints": 0,
            "available_archives": 0,
            "issues": []
        }

        # Check for archived models
        archives = list(self.models_dir.glob("champion_archived_*.joblib"))
        results["available_archives"] = len(archives)

        if len(archives) == 0:
            results["issues"].append("No archived models available for rollback")

        # Check for checkpoints
        checkpoints = self.list_checkpoints()
        results["available_checkpoints"] = len(checkpoints)

        # Estimate rollback time (based on file sizes)
        if archives:
            # Get size of most recent archive
            archive_size = archives[0].stat().st_size / 1024 / 1024  # MB

            # Estimate time (assume 100 MB/s copy speed)
            estimated_time = archive_size / 100

            results["estimated_time_seconds"] = estimated_time
            results["can_rollback"] = estimated_time < 300  # < 5 minutes

            if estimated_time >= 300:
                results["issues"].append(f"Estimated rollback time {estimated_time:.2f}s > 5 minutes")

        else:
            results["can_rollback"] = False

        logger.info(f"Rollback capability: {'✓ Ready' if results['can_rollback'] else '✗ Not Ready'}")

        return results


if __name__ == "__main__":
    """Example usage"""
    logging.basicConfig(level=logging.INFO)

    controller = RollbackController(models_dir=Path("mlops/models"))

    # Verify rollback capability
    verification = controller.verify_rollback_capability()
    print(f"\nRollback verification: {json.dumps(verification, indent=2)}")

    # List checkpoints
    checkpoints = controller.list_checkpoints()
    print(f"\nAvailable checkpoints ({len(checkpoints)}):")
    for cp in checkpoints:
        print(f"  - {cp['checkpoint_id']} ({cp['created_at']})")

    # Get rollback history
    history = controller.get_rollback_history()
    print(f"\nRollback history ({len(history)} entries):")
    for entry in history[-5:]:  # Last 5 entries
        print(f"  - {entry['timestamp']}: {entry.get('rollback_type', 'N/A')}")
