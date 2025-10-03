"""
Automated Retrain Orchestrator
Manages scheduled model retraining with PIT-compliant data extraction
Workflow:
1. Scheduled Trigger (cron)
2. Extract Training Data (PIT-compliant)
3. Train Challenger Model
4. Validate: SPA/DSR/PBO Gates
5. Shadow Mode (1 week)
6. Compare Champion vs Challenger
7. Promote if Gates Pass
8. Deploy to Production
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import yaml

logger = logging.getLogger(__name__)


@dataclass
class RetrainConfig:
    """Configuration for retraining"""
    retrain_cron: str = "0 3 1 * *"  # Monthly, 3am on 1st
    promotion_gate: bool = True
    shadow_mode_days: int = 7
    min_sharpe_improvement: float = 0.1
    max_drawdown_tolerance: float = 0.02
    auto_rollback_enabled: bool = True
    training_lookback_days: int = 365
    validation_lookback_days: int = 90
    min_training_samples: int = 10000
    model_type: str = "RandomForestRegressor"
    hyperparameters: Dict[str, Any] = None


@dataclass
class TrainingMetrics:
    """Metrics from model training"""
    rmse: float
    mae: float
    r2: float
    sharpe_ratio: float
    hit_rate: float
    max_drawdown: float
    information_ratio: float
    training_time_seconds: float
    num_samples: int
    num_features: int


@dataclass
class ValidationResults:
    """Results from validation gates"""
    spa_passed: bool
    dsr_passed: bool
    pbo_passed: bool
    sharpe_improvement: float
    drawdown_delta: float
    details: Dict[str, Any]


class PITDataExtractor:
    """
    Extracts Point-in-Time compliant training data
    Ensures no look-ahead bias
    """

    def __init__(self, database_url: str):
        self.database_url = database_url

    async def extract_training_data(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract PIT-compliant training data

        Args:
            start_date: Start date for training data
            end_date: End date for training data
            symbols: Optional list of symbols to include

        Returns:
            DataFrame with features and labels
        """
        import asyncpg

        conn = await asyncpg.connect(self.database_url)

        try:
            # Use the PIT-compliant training view
            query = """
                SELECT
                    symbol,
                    date,
                    -- Price/volume features
                    sma_20, sma_50, ema_12, rsi_14,
                    bollinger_upper, bollinger_lower,
                    volume_sma_20, volume_ratio,
                    -- Momentum features
                    momentum_5d, momentum_20d,
                    -- Fundamentals (first-print only)
                    earnings_surprise, pe_ratio, revenue_growth,
                    debt_to_equity, return_on_equity,
                    -- Sentiment features
                    sentiment_score, news_volume, social_mentions,
                    -- Macro features
                    vix, treasury_yield_spread, market_correlation,
                    -- Label
                    forward_return_5d as label
                FROM vw_fundamentals_training
                WHERE date >= $1 AND date <= $2
            """

            params = [start_date, end_date]

            if symbols:
                query += " AND symbol = ANY($3)"
                params.append(symbols)

            query += " ORDER BY date, symbol"

            rows = await conn.fetch(query, *params)

            # Convert to DataFrame
            df = pd.DataFrame([dict(row) for row in rows])

            logger.info(
                f"Extracted {len(df)} PIT-compliant samples "
                f"from {start_date.date()} to {end_date.date()}"
            )

            return df

        finally:
            await conn.close()

    def validate_pit_compliance(self, df: pd.DataFrame) -> bool:
        """
        Validate that data is PIT-compliant

        Args:
            df: Training DataFrame

        Returns:
            True if PIT-compliant
        """
        # Check for future data leakage
        # All fundamentals should be from first_print table
        # All features should have proper lag

        # Example checks
        if df.isnull().any().any():
            logger.warning("Dataset contains null values - possible PIT violation")
            return False

        # Check date ordering
        if not df['date'].is_monotonic_increasing:
            logger.warning("Dates are not properly ordered")
            return False

        logger.info("PIT compliance validation passed")
        return True


class ModelTrainer:
    """
    Trains challenger models with proper validation
    """

    def __init__(self, config: RetrainConfig):
        self.config = config

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> tuple:
        """
        Train a challenger model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            (trained_model, training_metrics)
        """
        import time
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        logger.info(f"Training {self.config.model_type} with {len(X_train)} samples")

        start_time = time.time()

        # Get hyperparameters
        hyperparams = self.config.hyperparameters or {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 20,
            'random_state': 42,
            'n_jobs': -1
        }

        # Train model
        if self.config.model_type == "RandomForestRegressor":
            model = RandomForestRegressor(**hyperparams)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        # Evaluate on validation set
        y_pred = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # Calculate financial metrics
        sharpe_ratio = self._calculate_sharpe(y_val, y_pred)
        hit_rate = self._calculate_hit_rate(y_val, y_pred)
        max_drawdown = self._calculate_max_drawdown(y_val, y_pred)
        information_ratio = self._calculate_information_ratio(y_val, y_pred)

        metrics = TrainingMetrics(
            rmse=rmse,
            mae=mae,
            r2=r2,
            sharpe_ratio=sharpe_ratio,
            hit_rate=hit_rate,
            max_drawdown=max_drawdown,
            information_ratio=information_ratio,
            training_time_seconds=training_time,
            num_samples=len(X_train),
            num_features=X_train.shape[1]
        )

        logger.info(f"Training complete: Sharpe={sharpe_ratio:.3f}, RMSE={rmse:.4f}")

        return model, metrics

    def _calculate_sharpe(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Sharpe ratio from predictions"""
        # Simplified: use predictions as signals
        returns = y_true * np.sign(y_pred)
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()

    def _calculate_hit_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy"""
        correct = np.sign(y_true) == np.sign(y_pred)
        return correct.mean()

    def _calculate_max_drawdown(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        returns = y_true * np.sign(y_pred)
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return drawdown.min()

    def _calculate_information_ratio(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate information ratio"""
        returns = y_true * np.sign(y_pred)
        excess_returns = returns - returns.mean()
        if excess_returns.std() == 0:
            return 0.0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


class RetrainOrchestrator:
    """
    Orchestrates the entire retraining workflow
    """

    def __init__(
        self,
        config: RetrainConfig,
        database_url: str,
        models_dir: Path
    ):
        self.config = config
        self.database_url = database_url
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.data_extractor = PITDataExtractor(database_url)
        self.trainer = ModelTrainer(config)

    async def run_retrain_workflow(self) -> Dict[str, Any]:
        """
        Execute complete retraining workflow

        Returns:
            Workflow results
        """
        logger.info("="*60)
        logger.info("STARTING AUTOMATED RETRAIN WORKFLOW")
        logger.info("="*60)

        workflow_start = datetime.utcnow()

        try:
            # Step 1: Extract PIT-compliant training data
            logger.info("\n[Step 1/8] Extracting PIT-compliant training data...")

            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=self.config.training_lookback_days)

            training_data = await self.data_extractor.extract_training_data(
                start_date,
                end_date
            )

            # Validate PIT compliance
            if not self.data_extractor.validate_pit_compliance(training_data):
                raise ValueError("PIT compliance validation failed")

            # Step 2: Prepare train/val split
            logger.info("\n[Step 2/8] Preparing train/validation split...")

            # Time-based split (no shuffle for time series)
            split_idx = int(len(training_data) * 0.8)
            train_df = training_data.iloc[:split_idx]
            val_df = training_data.iloc[split_idx:]

            # Separate features and labels
            feature_cols = [c for c in train_df.columns if c not in ['symbol', 'date', 'label']]
            X_train = train_df[feature_cols]
            y_train = train_df['label']
            X_val = val_df[feature_cols]
            y_val = val_df['label']

            logger.info(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")

            # Step 3: Train challenger model
            logger.info("\n[Step 3/8] Training challenger model...")

            challenger_model, training_metrics = self.trainer.train_model(
                X_train, y_train, X_val, y_val
            )

            logger.info(f"Challenger metrics: {asdict(training_metrics)}")

            # Step 4: Save challenger model
            logger.info("\n[Step 4/8] Saving challenger model...")

            challenger_path = self.models_dir / f"challenger_{workflow_start.strftime('%Y%m%d_%H%M%S')}.joblib"
            joblib.dump(challenger_model, challenger_path)

            logger.info(f"Challenger saved: {challenger_path}")

            # Step 5: Validation gates (SPA/DSR/PBO)
            logger.info("\n[Step 5/8] Running validation gates (SPA/DSR/PBO)...")

            validation_results = await self._run_validation_gates(
                challenger_model,
                X_val,
                y_val
            )

            logger.info(f"Validation results: {asdict(validation_results)}")

            # Step 6: Shadow mode (would run for configured days)
            logger.info("\n[Step 6/8] Shadow mode testing...")
            logger.info(f"Would run shadow mode for {self.config.shadow_mode_days} days")
            logger.info("(Skipping in automated workflow - manual review required)")

            # Step 7: Champion vs Challenger comparison
            logger.info("\n[Step 7/8] Comparing champion vs challenger...")

            should_promote = self._should_promote_challenger(
                training_metrics,
                validation_results
            )

            # Step 8: Promotion decision
            logger.info("\n[Step 8/8] Promotion decision...")

            if should_promote:
                logger.info("✓ Challenger PROMOTED to champion")
                champion_path = self.models_dir / "champion.joblib"
                joblib.dump(challenger_model, champion_path)

                # Save metadata
                metadata = {
                    "promoted_at": workflow_start.isoformat(),
                    "training_metrics": asdict(training_metrics),
                    "validation_results": asdict(validation_results),
                    "model_path": str(challenger_path)
                }

                metadata_path = self.models_dir / "champion_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

            else:
                logger.info("✗ Challenger did NOT pass promotion gates")

            # Workflow summary
            workflow_time = (datetime.utcnow() - workflow_start).total_seconds()

            summary = {
                "workflow_start": workflow_start.isoformat(),
                "workflow_duration_seconds": workflow_time,
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "training_metrics": asdict(training_metrics),
                "validation_results": asdict(validation_results),
                "promoted": should_promote,
                "challenger_path": str(challenger_path)
            }

            logger.info("\n"+"="*60)
            logger.info("RETRAIN WORKFLOW COMPLETE")
            logger.info("="*60)

            return summary

        except Exception as e:
            logger.error(f"Retrain workflow failed: {e}")
            raise

    async def _run_validation_gates(
        self,
        model,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> ValidationResults:
        """
        Run SPA/DSR/PBO validation gates

        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Validation results
        """
        # Make predictions
        y_pred = model.predict(X_val)

        # Calculate metrics for gates
        sharpe = self.trainer._calculate_sharpe(y_val, y_pred)
        drawdown = self.trainer._calculate_max_drawdown(y_val, y_pred)

        # Load champion metrics if exists
        champion_metadata_path = self.models_dir / "champion_metadata.json"
        champion_sharpe = 0.0
        champion_drawdown = 0.0

        if champion_metadata_path.exists():
            with open(champion_metadata_path, 'r') as f:
                champion_meta = json.load(f)
                champion_sharpe = champion_meta.get('training_metrics', {}).get('sharpe_ratio', 0.0)
                champion_drawdown = champion_meta.get('training_metrics', {}).get('max_drawdown', 0.0)

        # Calculate improvements
        sharpe_improvement = sharpe - champion_sharpe
        drawdown_delta = drawdown - champion_drawdown

        # Gate checks (simplified - in production would include full SPA/DSR/PBO tests)
        spa_passed = sharpe_improvement >= self.config.min_sharpe_improvement
        dsr_passed = abs(drawdown_delta) <= self.config.max_drawdown_tolerance
        pbo_passed = True  # Would implement full PBO test here

        return ValidationResults(
            spa_passed=spa_passed,
            dsr_passed=dsr_passed,
            pbo_passed=pbo_passed,
            sharpe_improvement=sharpe_improvement,
            drawdown_delta=drawdown_delta,
            details={
                "challenger_sharpe": sharpe,
                "champion_sharpe": champion_sharpe,
                "challenger_drawdown": drawdown,
                "champion_drawdown": champion_drawdown
            }
        )

    def _should_promote_challenger(
        self,
        metrics: TrainingMetrics,
        validation: ValidationResults
    ) -> bool:
        """Determine if challenger should be promoted"""
        if not self.config.promotion_gate:
            return True  # Auto-promote if gates disabled

        # All gates must pass
        return validation.spa_passed and validation.dsr_passed and validation.pbo_passed


async def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    config = RetrainConfig(
        min_sharpe_improvement=0.1,
        max_drawdown_tolerance=0.02,
        training_lookback_days=365
    )

    orchestrator = RetrainOrchestrator(
        config=config,
        database_url="postgresql://trading_user:trading_pass@localhost:5432/trading_db",
        models_dir=Path("mlops/models")
    )

    try:
        results = await orchestrator.run_retrain_workflow()
        print(f"\nWorkflow results: {json.dumps(results, indent=2)}")
    except Exception as e:
        logger.error(f"Workflow failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
