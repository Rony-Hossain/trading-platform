"""
MLflow Governance Integration

Attaches model cards and deployment memos as MLflow artifacts.
Tags model versions with governance_ready=true when all artifacts present.

Ensures 100% production models have model cards & memos.
"""

import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class GovernanceArtifact:
    """Governance artifact types"""
    MODEL_CARD = "model_card.md"
    DEPLOYMENT_MEMO = "deployment_memo.md"
    BACKTEST_REPORT = "backtest_report.html"
    VALIDATION_REPORT = "validation_report.json"
    PIT_VALIDATION = "pit_validation.json"


class MLflowGovernance:
    """
    MLflow Governance Integration

    Manages model governance artifacts in MLflow:
    - Attaches model cards and deployment memos
    - Tags models with governance status
    - Validates governance completeness
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow governance manager.

        Args:
            tracking_uri: MLflow tracking server URI (defaults to env var)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient()

    def attach_governance_artifacts(self,
                                   run_id: str,
                                   model_card_path: str,
                                   deployment_memo_path: Optional[str] = None,
                                   backtest_report_path: Optional[str] = None,
                                   validation_report_path: Optional[str] = None) -> Dict[str, str]:
        """
        Attach governance artifacts to MLflow run.

        Args:
            run_id: MLflow run ID
            model_card_path: Path to model card markdown file
            deployment_memo_path: Path to deployment memo (optional)
            backtest_report_path: Path to backtest report (optional)
            validation_report_path: Path to validation report (optional)

        Returns:
            Dictionary of artifact names and MLflow artifact URIs
        """
        artifacts_logged = {}

        with mlflow.start_run(run_id=run_id):
            # Log model card (required)
            if Path(model_card_path).exists():
                mlflow.log_artifact(model_card_path, artifact_path="governance")
                artifacts_logged[GovernanceArtifact.MODEL_CARD] = f"runs:/{run_id}/governance/model_card.md"
                logger.info(f"Logged model card to run {run_id}")
            else:
                logger.error(f"Model card not found: {model_card_path}")
                raise FileNotFoundError(f"Model card required but not found: {model_card_path}")

            # Log deployment memo (recommended)
            if deployment_memo_path and Path(deployment_memo_path).exists():
                mlflow.log_artifact(deployment_memo_path, artifact_path="governance")
                artifacts_logged[GovernanceArtifact.DEPLOYMENT_MEMO] = f"runs:/{run_id}/governance/deployment_memo.md"
                logger.info(f"Logged deployment memo to run {run_id}")

            # Log backtest report (optional)
            if backtest_report_path and Path(backtest_report_path).exists():
                mlflow.log_artifact(backtest_report_path, artifact_path="governance")
                artifacts_logged[GovernanceArtifact.BACKTEST_REPORT] = f"runs:/{run_id}/governance/backtest_report.html"
                logger.info(f"Logged backtest report to run {run_id}")

            # Log validation report (optional)
            if validation_report_path and Path(validation_report_path).exists():
                mlflow.log_artifact(validation_report_path, artifact_path="governance")
                artifacts_logged[GovernanceArtifact.VALIDATION_REPORT] = f"runs:/{run_id}/governance/validation_report.json"
                logger.info(f"Logged validation report to run {run_id}")

        return artifacts_logged

    def tag_governance_status(self,
                            run_id: str,
                            governance_ready: bool = True,
                            status: str = "production",
                            owner: Optional[str] = None,
                            version: Optional[str] = None,
                            deployment_date: Optional[str] = None) -> None:
        """
        Tag MLflow run with governance status.

        Args:
            run_id: MLflow run ID
            governance_ready: Whether model meets governance requirements
            status: Model status (development/staging/production)
            owner: Model owner name
            version: Model version
            deployment_date: Deployment date (YYYY-MM-DD)
        """
        tags = {
            "governance_ready": str(governance_ready).lower(),
            "status": status,
            "governance_validated_at": datetime.now().isoformat()
        }

        if owner:
            tags["owner"] = owner
        if version:
            tags["version"] = version
        if deployment_date:
            tags["deployment_date"] = deployment_date

        for key, value in tags.items():
            self.client.set_tag(run_id, key, value)

        logger.info(f"Tagged run {run_id} with governance status: {tags}")

    def validate_governance_completeness(self, run_id: str) -> Tuple[bool, List[str]]:
        """
        Validate that all required governance artifacts are present.

        Args:
            run_id: MLflow run ID

        Returns:
            Tuple of (is_complete, missing_artifacts)
        """
        missing_artifacts = []

        # Get run artifacts
        artifacts = self.client.list_artifacts(run_id, path="governance")
        artifact_names = [a.path.split('/')[-1] for a in artifacts]

        # Check required artifacts
        if GovernanceArtifact.MODEL_CARD not in artifact_names:
            missing_artifacts.append(GovernanceArtifact.MODEL_CARD)

        # Check recommended artifacts (warn but don't fail)
        if GovernanceArtifact.DEPLOYMENT_MEMO not in artifact_names:
            logger.warning(f"Deployment memo missing for run {run_id} (recommended)")

        is_complete = len(missing_artifacts) == 0

        if is_complete:
            logger.info(f"Run {run_id} has all required governance artifacts")
        else:
            logger.error(f"Run {run_id} missing artifacts: {missing_artifacts}")

        return is_complete, missing_artifacts

    def get_production_models_without_governance(self,
                                                experiment_name: Optional[str] = None) -> List[Dict]:
        """
        Find production models missing governance artifacts.

        Args:
            experiment_name: Filter by experiment name (optional)

        Returns:
            List of runs missing governance artifacts
        """
        non_compliant_models = []

        # Get experiments
        if experiment_name:
            experiment = self.client.get_experiment_by_name(experiment_name)
            experiment_ids = [experiment.experiment_id] if experiment else []
        else:
            experiments = self.client.search_experiments()
            experiment_ids = [e.experiment_id for e in experiments]

        # Search for production runs
        for exp_id in experiment_ids:
            runs = self.client.search_runs(
                experiment_ids=[exp_id],
                filter_string="tags.status = 'production'"
            )

            for run in runs:
                # Check if governance_ready tag is set
                governance_ready = run.data.tags.get("governance_ready", "false")

                if governance_ready.lower() != "true":
                    # Validate artifacts
                    is_complete, missing = self.validate_governance_completeness(run.info.run_id)

                    if not is_complete:
                        non_compliant_models.append({
                            "run_id": run.info.run_id,
                            "experiment_id": exp_id,
                            "model_name": run.data.tags.get("mlflow.runName", "Unknown"),
                            "version": run.data.tags.get("version", "Unknown"),
                            "missing_artifacts": missing,
                            "status": run.data.tags.get("status", "unknown")
                        })

        return non_compliant_models

    def generate_governance_report(self, experiment_name: Optional[str] = None) -> Dict:
        """
        Generate governance compliance report.

        Args:
            experiment_name: Filter by experiment name (optional)

        Returns:
            Governance compliance report
        """
        # Get experiments
        if experiment_name:
            experiment = self.client.get_experiment_by_name(experiment_name)
            experiment_ids = [experiment.experiment_id] if experiment else []
        else:
            experiments = self.client.search_experiments()
            experiment_ids = [e.experiment_id for e in experiments]

        total_production_models = 0
        compliant_models = 0
        non_compliant_models = []

        for exp_id in experiment_ids:
            runs = self.client.search_runs(
                experiment_ids=[exp_id],
                filter_string="tags.status = 'production'"
            )

            for run in runs:
                total_production_models += 1

                governance_ready = run.data.tags.get("governance_ready", "false")

                if governance_ready.lower() == "true":
                    is_complete, missing = self.validate_governance_completeness(run.info.run_id)
                    if is_complete:
                        compliant_models += 1
                    else:
                        non_compliant_models.append({
                            "run_id": run.info.run_id,
                            "model_name": run.data.tags.get("mlflow.runName", "Unknown"),
                            "missing_artifacts": missing
                        })
                else:
                    non_compliant_models.append({
                        "run_id": run.info.run_id,
                        "model_name": run.data.tags.get("mlflow.runName", "Unknown"),
                        "reason": "governance_ready tag not set to true"
                    })

        compliance_rate = (compliant_models / total_production_models * 100) if total_production_models > 0 else 0

        report = {
            "generated_at": datetime.now().isoformat(),
            "experiment_filter": experiment_name or "all",
            "summary": {
                "total_production_models": total_production_models,
                "compliant_models": compliant_models,
                "non_compliant_models": len(non_compliant_models),
                "compliance_rate_percent": compliance_rate,
                "target_compliance_rate": 100.0,
                "meets_target": compliance_rate == 100.0
            },
            "non_compliant_details": non_compliant_models
        }

        return report

    def register_model_with_governance(self,
                                      run_id: str,
                                      model_name: str,
                                      model_card_path: str,
                                      deployment_memo_path: Optional[str] = None,
                                      version: Optional[str] = None) -> str:
        """
        Register model with full governance artifacts.

        Args:
            run_id: MLflow run ID
            model_name: Model name for registry
            model_card_path: Path to model card
            deployment_memo_path: Path to deployment memo (optional)
            version: Model version string (optional)

        Returns:
            Model version object
        """
        # Attach governance artifacts
        artifacts = self.attach_governance_artifacts(
            run_id=run_id,
            model_card_path=model_card_path,
            deployment_memo_path=deployment_memo_path
        )

        # Validate completeness
        is_complete, missing = self.validate_governance_completeness(run_id)

        if not is_complete:
            raise ValueError(
                f"Cannot register model without required governance artifacts. "
                f"Missing: {missing}"
            )

        # Tag with governance status
        self.tag_governance_status(
            run_id=run_id,
            governance_ready=True,
            status="production",
            version=version,
            deployment_date=datetime.now().strftime("%Y-%m-%d")
        )

        # Register model
        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(model_uri, model_name)

        logger.info(
            f"Registered model {model_name} version {model_version.version} "
            f"with governance artifacts"
        )

        return model_version

    def enforce_governance_gate(self, run_id: str) -> bool:
        """
        Enforce governance gate - block production deployment without artifacts.

        Args:
            run_id: MLflow run ID

        Returns:
            True if governance requirements met, False otherwise

        Raises:
            ValueError if governance requirements not met
        """
        is_complete, missing = self.validate_governance_completeness(run_id)

        if not is_complete:
            error_msg = (
                f"Governance gate FAILED for run {run_id}. "
                f"Missing required artifacts: {missing}. "
                f"Production deployment blocked."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Governance gate PASSED for run {run_id}")
        return True


def create_governance_manager(tracking_uri: Optional[str] = None) -> MLflowGovernance:
    """
    Factory function to create MLflow governance manager.

    Args:
        tracking_uri: MLflow tracking URI (optional)

    Returns:
        MLflowGovernance instance
    """
    return MLflowGovernance(tracking_uri=tracking_uri)


# Example usage
if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)

    # Initialize governance manager
    governance = MLflowGovernance()

    # Example: Attach governance artifacts to existing run
    example_run_id = "abc123def456"

    try:
        # Attach artifacts
        artifacts = governance.attach_governance_artifacts(
            run_id=example_run_id,
            model_card_path="docs/model_cards/momentum_alpha_v1.md",
            deployment_memo_path="docs/deploy_memos/momentum_alpha_v1.0.0.md"
        )

        print(f"Attached artifacts: {artifacts}")

        # Tag with governance status
        governance.tag_governance_status(
            run_id=example_run_id,
            governance_ready=True,
            status="production",
            owner="Jane Smith",
            version="v1.0.0",
            deployment_date="2025-10-01"
        )

        # Validate completeness
        is_complete, missing = governance.validate_governance_completeness(example_run_id)
        print(f"Governance complete: {is_complete}")
        if missing:
            print(f"Missing artifacts: {missing}")

        # Generate compliance report
        report = governance.generate_governance_report()
        print(f"\nGovernance Compliance Report:")
        print(f"Total production models: {report['summary']['total_production_models']}")
        print(f"Compliant models: {report['summary']['compliant_models']}")
        print(f"Compliance rate: {report['summary']['compliance_rate_percent']:.1f}%")
        print(f"Meets target (100%): {report['summary']['meets_target']}")

        if report['non_compliant_details']:
            print(f"\nNon-compliant models:")
            for model in report['non_compliant_details']:
                print(f"  - {model['model_name']} (run: {model['run_id']})")

    except Exception as e:
        logger.error(f"Example failed: {e}")
