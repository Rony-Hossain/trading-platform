"""
Champion/Challenger Deployment Manager
Manages shadow mode testing and model promotion
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json
import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a model version"""
    model_id: str
    model_type: str
    version: str
    trained_at: datetime
    promoted_at: Optional[datetime]
    metrics: Dict[str, float]
    status: str  # 'shadow', 'champion', 'archived'
    shadow_start: Optional[datetime]
    shadow_end: Optional[datetime]


class ChampionChallengerManager:
    """
    Manages champion and challenger models
    Handles shadow mode testing and promotion
    """

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.champion: Optional[Any] = None
        self.challenger: Optional[Any] = None

        self.champion_metadata: Optional[ModelMetadata] = None
        self.challenger_metadata: Optional[ModelMetadata] = None

        self._load_models()

    def _load_models(self):
        """Load champion and challenger models"""
        # Load champion
        champion_path = self.models_dir / "champion.joblib"
        champion_meta_path = self.models_dir / "champion_metadata.json"

        if champion_path.exists():
            self.champion = joblib.load(champion_path)

            if champion_meta_path.exists():
                with open(champion_meta_path, 'r') as f:
                    meta_dict = json.load(f)
                    self.champion_metadata = self._dict_to_metadata(meta_dict)

            logger.info(f"Loaded champion model: {champion_path}")

        # Load challenger if in shadow mode
        challenger_path = self.models_dir / "challenger_shadow.joblib"
        challenger_meta_path = self.models_dir / "challenger_shadow_metadata.json"

        if challenger_path.exists():
            self.challenger = joblib.load(challenger_path)

            if challenger_meta_path.exists():
                with open(challenger_meta_path, 'r') as f:
                    meta_dict = json.load(f)
                    self.challenger_metadata = self._dict_to_metadata(meta_dict)

            logger.info(f"Loaded challenger model in shadow mode: {challenger_path}")

    def _dict_to_metadata(self, meta_dict: Dict) -> ModelMetadata:
        """Convert dict to ModelMetadata"""
        return ModelMetadata(
            model_id=meta_dict.get('model_id', 'unknown'),
            model_type=meta_dict.get('model_type', 'unknown'),
            version=meta_dict.get('version', '1.0'),
            trained_at=datetime.fromisoformat(meta_dict['trained_at']) if 'trained_at' in meta_dict else datetime.utcnow(),
            promoted_at=datetime.fromisoformat(meta_dict['promoted_at']) if 'promoted_at' in meta_dict else None,
            metrics=meta_dict.get('training_metrics', {}),
            status=meta_dict.get('status', 'unknown'),
            shadow_start=datetime.fromisoformat(meta_dict['shadow_start']) if 'shadow_start' in meta_dict else None,
            shadow_end=datetime.fromisoformat(meta_dict['shadow_end']) if 'shadow_end' in meta_dict else None
        )

    def start_shadow_mode(
        self,
        challenger_model,
        metadata: Dict[str, Any],
        shadow_duration_days: int = 7
    ):
        """
        Start shadow mode for a challenger model

        Args:
            challenger_model: The challenger model
            metadata: Model metadata
            shadow_duration_days: Duration of shadow mode
        """
        logger.info(f"Starting shadow mode for {shadow_duration_days} days")

        now = datetime.utcnow()
        shadow_end = now + timedelta(days=shadow_duration_days)

        # Save challenger model
        challenger_path = self.models_dir / "challenger_shadow.joblib"
        joblib.dump(challenger_model, challenger_path)

        # Create metadata
        challenger_meta = {
            "model_id": f"challenger_{now.strftime('%Y%m%d_%H%M%S')}",
            "model_type": metadata.get('model_type', 'unknown'),
            "version": metadata.get('version', '1.0'),
            "trained_at": now.isoformat(),
            "status": "shadow",
            "shadow_start": now.isoformat(),
            "shadow_end": shadow_end.isoformat(),
            "training_metrics": metadata.get('training_metrics', {})
        }

        challenger_meta_path = self.models_dir / "challenger_shadow_metadata.json"
        with open(challenger_meta_path, 'w') as f:
            json.dump(challenger_meta, f, indent=2)

        self.challenger = challenger_model
        self.challenger_metadata = self._dict_to_metadata(challenger_meta)

        logger.info(f"Shadow mode started until {shadow_end.date()}")

    def predict_champion(self, features: np.ndarray) -> np.ndarray:
        """Get prediction from champion model"""
        if self.champion is None:
            raise ValueError("No champion model loaded")

        return self.champion.predict(features)

    def predict_challenger(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Get prediction from challenger model (if in shadow mode)"""
        if self.challenger is None:
            return None

        return self.challenger.predict(features)

    def predict_both(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from both models during shadow mode

        Args:
            features: Input features

        Returns:
            Dict with 'champion' and 'challenger' predictions
        """
        result = {
            "champion": self.predict_champion(features)
        }

        if self.challenger is not None:
            result["challenger"] = self.predict_challenger(features)

        return result

    def compare_models(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Compare champion and challenger performance

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Comparison results
        """
        if self.challenger is None:
            raise ValueError("No challenger model in shadow mode")

        from sklearn.metrics import mean_squared_error, mean_absolute_error

        # Get predictions
        champion_pred = self.predict_champion(X_test)
        challenger_pred = self.predict_challenger(X_test)

        # Calculate metrics
        champion_rmse = np.sqrt(mean_squared_error(y_test, champion_pred))
        challenger_rmse = np.sqrt(mean_squared_error(y_test, challenger_pred))

        champion_mae = mean_absolute_error(y_test, champion_pred)
        challenger_mae = mean_absolute_error(y_test, challenger_pred)

        # Calculate financial metrics
        champion_sharpe = self._calculate_sharpe(y_test, champion_pred)
        challenger_sharpe = self._calculate_sharpe(y_test, challenger_pred)

        comparison = {
            "champion": {
                "rmse": champion_rmse,
                "mae": champion_mae,
                "sharpe": champion_sharpe
            },
            "challenger": {
                "rmse": challenger_rmse,
                "mae": challenger_mae,
                "sharpe": challenger_sharpe
            },
            "improvement": {
                "rmse_delta": champion_rmse - challenger_rmse,
                "mae_delta": champion_mae - challenger_mae,
                "sharpe_delta": challenger_sharpe - champion_sharpe
            }
        }

        logger.info(f"Model comparison: {json.dumps(comparison, indent=2)}")

        return comparison

    def _calculate_sharpe(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Sharpe ratio from predictions"""
        returns = y_true * np.sign(y_pred)
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()

    def promote_challenger(self):
        """
        Promote challenger to champion

        Returns:
            Success status
        """
        if self.challenger is None:
            raise ValueError("No challenger model to promote")

        logger.info("Promoting challenger to champion")

        now = datetime.utcnow()

        # Archive current champion
        if self.champion is not None:
            archive_path = self.models_dir / f"champion_archived_{now.strftime('%Y%m%d_%H%M%S')}.joblib"
            joblib.dump(self.champion, archive_path)

            if self.champion_metadata:
                archive_meta_path = self.models_dir / f"champion_archived_{now.strftime('%Y%m%d_%H%M%S')}_metadata.json"
                with open(archive_meta_path, 'w') as f:
                    json.dump(asdict(self.champion_metadata), f, indent=2, default=str)

            logger.info(f"Archived previous champion: {archive_path}")

        # Promote challenger to champion
        champion_path = self.models_dir / "champion.joblib"
        joblib.dump(self.challenger, champion_path)

        # Update metadata
        promoted_meta = asdict(self.challenger_metadata)
        promoted_meta['status'] = 'champion'
        promoted_meta['promoted_at'] = now.isoformat()

        champion_meta_path = self.models_dir / "champion_metadata.json"
        with open(champion_meta_path, 'w') as f:
            json.dump(promoted_meta, f, indent=2, default=str)

        # Clear challenger
        challenger_path = self.models_dir / "challenger_shadow.joblib"
        challenger_meta_path = self.models_dir / "challenger_shadow_metadata.json"

        if challenger_path.exists():
            challenger_path.unlink()
        if challenger_meta_path.exists():
            challenger_meta_path.unlink()

        # Update internal state
        self.champion = self.challenger
        self.champion_metadata = self._dict_to_metadata(promoted_meta)
        self.challenger = None
        self.challenger_metadata = None

        logger.info("✓ Challenger successfully promoted to champion")

        return True

    def rollback_to_champion(self):
        """
        Rollback to champion (remove challenger from shadow mode)
        """
        logger.info("Rolling back to champion (removing challenger)")

        # Remove challenger files
        challenger_path = self.models_dir / "challenger_shadow.joblib"
        challenger_meta_path = self.models_dir / "challenger_shadow_metadata.json"

        if challenger_path.exists():
            challenger_path.unlink()
        if challenger_meta_path.exists():
            challenger_meta_path.unlink()

        self.challenger = None
        self.challenger_metadata = None

        logger.info("✓ Rolled back to champion")

    def get_active_model_info(self) -> Dict[str, Any]:
        """Get information about active models"""
        info = {
            "champion": None,
            "challenger": None,
            "shadow_mode_active": self.challenger is not None
        }

        if self.champion_metadata:
            info["champion"] = asdict(self.champion_metadata)

        if self.challenger_metadata:
            info["challenger"] = asdict(self.challenger_metadata)

        return info


if __name__ == "__main__":
    """Example usage"""
    logging.basicConfig(level=logging.INFO)

    manager = ChampionChallengerManager(models_dir=Path("mlops/models"))

    # Get model info
    info = manager.get_active_model_info()
    print(f"\nActive models: {json.dumps(info, indent=2, default=str)}")

    # Example: Make predictions
    if manager.champion:
        sample_features = np.random.randn(5, 50).astype(np.float32)
        predictions = manager.predict_both(sample_features)
        print(f"\nPredictions: {predictions}")
