"""
MLflow Model Registry for Trading Platform
Manages model lifecycle, versioning, and lineage tracking
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import RegisteredModel, ModelVersion
from mlflow.exceptions import RestException
import logging
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelStage(Enum):
    """Model lifecycle stages"""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"

class ModelType(Enum):
    """Supported model types for trading platform"""
    SENTIMENT_CLASSIFIER = "sentiment_classifier"
    PRICE_PREDICTOR = "price_predictor"
    RISK_ASSESSOR = "risk_assessor"
    ANOMALY_DETECTOR = "anomaly_detector"
    PORTFOLIO_OPTIMIZER = "portfolio_optimizer"
    MARKET_REGIME_DETECTOR = "market_regime_detector"

class MLflowModelRegistry:
    """
    Comprehensive MLflow Model Registry for trading platform
    
    Features:
    - Model versioning with semantic versioning
    - Lineage tracking and data lineage
    - Automated model validation
    - Performance benchmarking
    - Model comparison and A/B testing
    """
    
    def __init__(self, tracking_uri: Optional[str] = None, registry_uri: Optional[str] = None):
        """Initialize MLflow Model Registry"""
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.registry_uri = registry_uri or os.getenv("MLFLOW_REGISTRY_URI", self.tracking_uri)
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        
        self.client = MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.registry_uri)
        
        # Model performance thresholds for automatic promotion
        self.performance_thresholds = {
            ModelType.SENTIMENT_CLASSIFIER: {"accuracy": 0.85, "f1_score": 0.80},
            ModelType.PRICE_PREDICTOR: {"mae": 0.05, "rmse": 0.08, "r2": 0.70},
            ModelType.RISK_ASSESSOR: {"precision": 0.85, "recall": 0.80, "auc": 0.90},
            ModelType.ANOMALY_DETECTOR: {"precision": 0.90, "recall": 0.75, "f1_score": 0.80},
            ModelType.PORTFOLIO_OPTIMIZER: {"sharpe_ratio": 1.2, "max_drawdown": 0.15},
            ModelType.MARKET_REGIME_DETECTOR: {"accuracy": 0.75, "macro_f1": 0.70}
        }
    
    def create_experiment(self, experiment_name: str, model_type: ModelType, 
                         description: str = None) -> str:
        """Create MLflow experiment for model tracking"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                return experiment.experiment_id
            
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                tags={
                    "model_type": model_type.value,
                    "created_at": datetime.utcnow().isoformat(),
                    "description": description or f"Experiment for {model_type.value} models"
                }
            )
            
            logger.info(f"Created experiment {experiment_name} with ID {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error creating experiment {experiment_name}: {e}")
            raise
    
    def register_model(self, model_name: str, model_type: ModelType, 
                      description: str = None) -> RegisteredModel:
        """Register a new model in the model registry"""
        try:
            # Check if model already exists
            try:
                model = self.client.get_registered_model(model_name)
                logger.info(f"Model {model_name} already exists")
                return model
            except RestException:
                pass
            
            model = self.client.create_registered_model(
                name=model_name,
                tags={
                    "model_type": model_type.value,
                    "created_at": datetime.utcnow().isoformat(),
                    "description": description or f"{model_type.value} model for trading platform"
                },
                description=description
            )
            
            logger.info(f"Registered new model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Error registering model {model_name}: {e}")
            raise
    
    def log_model_run(self, experiment_id: str, model_name: str, model_type: ModelType,
                     model, model_signature, input_example, 
                     metrics: Dict[str, float], params: Dict[str, Any],
                     artifacts: Dict[str, str] = None,
                     data_lineage: Dict[str, Any] = None,
                     tags: Dict[str, str] = None) -> str:
        """
        Log a model run with comprehensive tracking
        
        Returns:
            str: MLflow run ID
        """
        with mlflow.start_run(experiment_id=experiment_id) as run:
            try:
                # Log parameters
                for key, value in params.items():
                    mlflow.log_param(key, value)
                
                # Log metrics
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)
                
                # Log model based on type
                model_info = self._log_model_by_type(
                    model=model,
                    model_type=model_type,
                    signature=model_signature,
                    input_example=input_example
                )
                
                # Log data lineage information
                if data_lineage:
                    mlflow.log_dict(data_lineage, "data_lineage.json")
                
                # Log artifacts
                if artifacts:
                    for artifact_name, artifact_path in artifacts.items():
                        mlflow.log_artifact(artifact_path, artifact_name)
                
                # Set tags
                run_tags = {
                    "model_name": model_name,
                    "model_type": model_type.value,
                    "logged_at": datetime.utcnow().isoformat(),
                    "model_uri": model_info.model_uri
                }
                if tags:
                    run_tags.update(tags)
                
                mlflow.set_tags(run_tags)
                
                # Generate model hash for versioning
                model_hash = self._generate_model_hash(params, metrics, model_signature)
                mlflow.log_param("model_hash", model_hash)
                
                logger.info(f"Logged model run {run.info.run_id} for {model_name}")
                return run.info.run_id
                
            except Exception as e:
                logger.error(f"Error logging model run for {model_name}: {e}")
                raise
    
    def _log_model_by_type(self, model, model_type: ModelType, signature, input_example):
        """Log model based on its type using appropriate MLflow flavor"""
        model_path = f"models/{model_type.value}"
        
        if hasattr(model, 'fit') and hasattr(model, 'predict'):
            # Scikit-learn compatible model
            return mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_path,
                signature=signature,
                input_example=input_example
            )
        elif hasattr(model, 'state_dict'):
            # PyTorch model
            return mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=model_path,
                signature=signature,
                input_example=input_example
            )
        elif hasattr(model, 'save'):
            # TensorFlow/Keras model
            return mlflow.tensorflow.log_model(
                tf_saved_model_dir=model,
                tf_meta_graph_tags=None,
                tf_signature_def_key=None,
                artifact_path=model_path,
                signature=signature,
                input_example=input_example
            )
        else:
            # Generic Python model
            return mlflow.pyfunc.log_model(
                artifact_path=model_path,
                python_model=model,
                signature=signature,
                input_example=input_example
            )
    
    def create_model_version(self, model_name: str, run_id: str, 
                           stage: ModelStage = ModelStage.NONE,
                           description: str = None) -> ModelVersion:
        """Create a new version of a registered model"""
        try:
            source = f"runs:/{run_id}/models"
            
            version = self.client.create_model_version(
                name=model_name,
                source=source,
                run_id=run_id,
                description=description
            )
            
            # Set initial stage
            if stage != ModelStage.NONE:
                self.transition_model_stage(
                    model_name=model_name,
                    version=version.version,
                    stage=stage
                )
            
            logger.info(f"Created model version {version.version} for {model_name}")
            return version
            
        except Exception as e:
            logger.error(f"Error creating model version for {model_name}: {e}")
            raise
    
    def transition_model_stage(self, model_name: str, version: str, 
                             stage: ModelStage, archive_existing: bool = True) -> ModelVersion:
        """Transition model version to a new stage"""
        try:
            version = self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage.value,
                archive_existing_versions=archive_existing
            )
            
            logger.info(f"Transitioned {model_name} v{version.version} to {stage.value}")
            return version
            
        except Exception as e:
            logger.error(f"Error transitioning {model_name} to {stage.value}: {e}")
            raise
    
    def get_model_versions(self, model_name: str, 
                          stages: List[ModelStage] = None) -> List[ModelVersion]:
        """Get model versions by stage"""
        try:
            if stages:
                stage_list = [stage.value for stage in stages]
                versions = self.client.get_latest_versions(
                    name=model_name,
                    stages=stage_list
                )
            else:
                model = self.client.get_registered_model(model_name)
                versions = model.latest_versions
            
            return versions
            
        except Exception as e:
            logger.error(f"Error getting model versions for {model_name}: {e}")
            raise
    
    def get_production_model(self, model_name: str) -> Optional[ModelVersion]:
        """Get the current production model version"""
        try:
            versions = self.get_model_versions(model_name, [ModelStage.PRODUCTION])
            return versions[0] if versions else None
            
        except Exception as e:
            logger.error(f"Error getting production model {model_name}: {e}")
            return None
    
    def compare_model_versions(self, model_name: str, version1: str, 
                             version2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        try:
            v1 = self.client.get_model_version(model_name, version1)
            v2 = self.client.get_model_version(model_name, version2)
            
            # Get run metrics for comparison
            run1 = self.client.get_run(v1.run_id)
            run2 = self.client.get_run(v2.run_id)
            
            comparison = {
                "model_name": model_name,
                "version_1": {
                    "version": version1,
                    "stage": v1.current_stage,
                    "metrics": run1.data.metrics,
                    "params": run1.data.params,
                    "created_at": v1.creation_timestamp
                },
                "version_2": {
                    "version": version2,
                    "stage": v2.current_stage,
                    "metrics": run2.data.metrics,
                    "params": run2.data.params,
                    "created_at": v2.creation_timestamp
                },
                "metric_differences": {}
            }
            
            # Calculate metric differences
            for metric, value1 in run1.data.metrics.items():
                if metric in run2.data.metrics:
                    value2 = run2.data.metrics[metric]
                    comparison["metric_differences"][metric] = {
                        "difference": value2 - value1,
                        "percentage_change": ((value2 - value1) / value1 * 100) if value1 != 0 else float('inf')
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing model versions: {e}")
            raise
    
    def validate_model_performance(self, model_name: str, version: str, 
                                 model_type: ModelType, metrics: Dict[str, float]) -> bool:
        """Validate if model meets performance thresholds"""
        try:
            thresholds = self.performance_thresholds.get(model_type, {})
            
            for metric_name, threshold_value in thresholds.items():
                if metric_name in metrics:
                    actual_value = metrics[metric_name]
                    
                    # For error metrics (lower is better)
                    if metric_name.lower() in ['mae', 'rmse', 'mse', 'max_drawdown']:
                        if actual_value > threshold_value:
                            logger.warning(f"Model {model_name} v{version} failed {metric_name} threshold: "
                                         f"{actual_value} > {threshold_value}")
                            return False
                    else:
                        # For performance metrics (higher is better)
                        if actual_value < threshold_value:
                            logger.warning(f"Model {model_name} v{version} failed {metric_name} threshold: "
                                         f"{actual_value} < {threshold_value}")
                            return False
            
            logger.info(f"Model {model_name} v{version} passed all performance thresholds")
            return True
            
        except Exception as e:
            logger.error(f"Error validating model performance: {e}")
            return False
    
    def auto_promote_model(self, model_name: str, version: str, 
                          model_type: ModelType, metrics: Dict[str, float]) -> bool:
        """Automatically promote model based on performance validation"""
        try:
            # Validate performance
            if not self.validate_model_performance(model_name, version, model_type, metrics):
                logger.info(f"Model {model_name} v{version} does not meet promotion criteria")
                return False
            
            # Get current production model
            current_prod = self.get_production_model(model_name)
            
            if current_prod:
                # Compare with current production model
                comparison = self.compare_model_versions(
                    model_name, current_prod.version, version
                )
                
                # Check if new model is significantly better
                improvement_threshold = 0.02  # 2% improvement required
                key_metrics = ["accuracy", "f1_score", "r2", "sharpe_ratio"]
                
                improved = False
                for metric in key_metrics:
                    if metric in comparison["metric_differences"]:
                        perc_change = comparison["metric_differences"][metric]["percentage_change"]
                        if perc_change > improvement_threshold * 100:
                            improved = True
                            break
                
                if not improved:
                    logger.info(f"Model {model_name} v{version} does not show significant improvement")
                    return False
            
            # Promote to production
            self.transition_model_stage(
                model_name=model_name,
                version=version,
                stage=ModelStage.PRODUCTION
            )
            
            logger.info(f"Auto-promoted model {model_name} v{version} to production")
            return True
            
        except Exception as e:
            logger.error(f"Error auto-promoting model: {e}")
            return False
    
    def _generate_model_hash(self, params: Dict[str, Any], metrics: Dict[str, float], 
                           signature) -> str:
        """Generate unique hash for model version identification"""
        hash_input = {
            "params": params,
            "metrics": {k: round(v, 6) for k, v in metrics.items()},  # Round for consistency
            "signature": str(signature) if signature else None,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d")  # Date only for daily versions
        }
        
        hash_string = json.dumps(hash_input, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()[:12]
    
    def delete_model_version(self, model_name: str, version: str) -> None:
        """Delete a model version (use with caution)"""
        try:
            self.client.delete_model_version(model_name, version)
            logger.info(f"Deleted model version {model_name} v{version}")
            
        except Exception as e:
            logger.error(f"Error deleting model version: {e}")
            raise
    
    def archive_old_versions(self, model_name: str, keep_versions: int = 5) -> None:
        """Archive old model versions, keeping only the specified number of recent versions"""
        try:
            all_versions = self.client.search_model_versions(f"name='{model_name}'")
            
            # Sort by version number (descending)
            sorted_versions = sorted(all_versions, key=lambda x: int(x.version), reverse=True)
            
            # Archive older versions
            for version in sorted_versions[keep_versions:]:
                if version.current_stage not in [ModelStage.PRODUCTION.value, ModelStage.STAGING.value]:
                    self.transition_model_stage(
                        model_name=model_name,
                        version=version.version,
                        stage=ModelStage.ARCHIVED,
                        archive_existing=False
                    )
                    
            logger.info(f"Archived old versions for {model_name}, keeping {keep_versions} recent versions")
            
        except Exception as e:
            logger.error(f"Error archiving old versions: {e}")
            raise
    
    def get_model_lineage(self, model_name: str, version: str) -> Dict[str, Any]:
        """Get complete lineage information for a model version"""
        try:
            model_version = self.client.get_model_version(model_name, version)
            run = self.client.get_run(model_version.run_id)
            
            lineage = {
                "model_name": model_name,
                "version": version,
                "run_id": model_version.run_id,
                "experiment_id": run.info.experiment_id,
                "created_at": model_version.creation_timestamp,
                "current_stage": model_version.current_stage,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
                "artifacts": [artifact.path for artifact in self.client.list_artifacts(run.info.run_id)]
            }
            
            # Get data lineage if available
            try:
                data_lineage = self.client.download_artifacts(
                    run.info.run_id, "data_lineage.json"
                )
                with open(data_lineage, 'r') as f:
                    lineage["data_lineage"] = json.load(f)
            except:
                lineage["data_lineage"] = None
            
            return lineage
            
        except Exception as e:
            logger.error(f"Error getting model lineage: {e}")
            raise
    
    def search_models(self, query: str = None, model_type: ModelType = None) -> List[RegisteredModel]:
        """Search for models in the registry"""
        try:
            if model_type:
                filter_string = f"tags.model_type='{model_type.value}'"
                if query:
                    filter_string += f" and name ILIKE '%{query}%'"
            elif query:
                filter_string = f"name ILIKE '%{query}%'"
            else:
                filter_string = ""
            
            models = self.client.search_registered_models(filter_string=filter_string)
            return models
            
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            raise