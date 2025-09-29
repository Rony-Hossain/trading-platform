#!/usr/bin/env python3
"""
MLflow Tracking Integration

Comprehensive MLflow integration for experiment tracking, model versioning,
and lifecycle management. Replaces temporary CSV/JSON logging with production-grade
experiment management and model registry.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# MLflow imports with fallback
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    import mlflow.tensorflow
    from mlflow.tracking import MlflowClient
    from mlflow.entities import ViewType
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for MLflow experiment."""
    experiment_name: str
    run_name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    description: Optional[str] = None


@dataclass
class ModelMetrics:
    """Model performance metrics for logging."""
    # ML Metrics
    mse: float
    mae: float
    rmse: float
    r2: float
    cv_mean: float
    cv_std: float
    
    # Financial Metrics
    sharpe_ratio: float
    hit_rate: float
    max_drawdown: float
    total_return: float
    volatility: float
    information_ratio: float
    calmar_ratio: float
    
    # Performance Metrics
    training_time: float
    prediction_time: float
    
    # Model Info
    model_type: str
    feature_count: int
    data_points: int


@dataclass
class ExperimentResult:
    """Results from MLflow experiment."""
    experiment_id: str
    run_id: str
    artifact_uri: str
    status: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    tags: Dict[str, str]
    model_uri: Optional[str] = None


class MLflowTracker:
    """
    MLflow integration for experiment tracking and model management.
    
    Provides comprehensive experiment logging, model versioning, and lifecycle
    management for financial forecasting models.
    """
    
    def __init__(self, 
                 tracking_uri: str = None,
                 experiment_name: str = "financial_forecasting",
                 enable_autolog: bool = True):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI (None for local)
            experiment_name: Default experiment name
            enable_autolog: Enable automatic logging for supported frameworks
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available. Install with: pip install mlflow")
        
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        self.default_experiment = experiment_name
        self.client = None
        self.current_run = None
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Enable autologging if requested
        if enable_autolog:
            self._enable_autologging()
        
        logger.info(f"MLflow tracker initialized with URI: {self.tracking_uri}")
    
    def _enable_autologging(self):
        """Enable automatic logging for supported ML frameworks."""
        try:
            # Enable autologging for various frameworks
            mlflow.sklearn.autolog(
                log_input_examples=False,
                log_model_signatures=True,
                log_models=True,
                disable=False,
                exclusive=False,
                disable_for_unsupported_versions=True,
                silent=True
            )
            
            # Enable for other frameworks if available
            try:
                mlflow.tensorflow.autolog()
            except:
                pass
                
            try:
                mlflow.pytorch.autolog()
            except:
                pass
                
            logger.info("MLflow autologging enabled")
            
        except Exception as e:
            logger.warning(f"Could not enable MLflow autologging: {e}")
    
    def get_or_create_experiment(self, experiment_name: str) -> str:
        """Get or create MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    tags={
                        "mlflow.note.content": f"Financial forecasting experiment: {experiment_name}",
                        "created_by": "trading_platform",
                        "version": "1.0"
                    }
                )
                logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
                return experiment_id
            else:
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
                return experiment.experiment_id
                
        except Exception as e:
            logger.error(f"Error creating/getting experiment {experiment_name}: {e}")
            raise
    
    async def start_run(self, 
                       experiment_config: ExperimentConfig,
                       nested: bool = False) -> str:
        """
        Start MLflow run with configuration.
        
        Args:
            experiment_config: Experiment configuration
            nested: Whether this is a nested run
            
        Returns:
            Run ID
        """
        try:
            # Get or create experiment
            experiment_id = self.get_or_create_experiment(experiment_config.experiment_name)
            
            # Start run
            run = mlflow.start_run(
                experiment_id=experiment_id,
                run_name=experiment_config.run_name,
                nested=nested,
                tags=experiment_config.tags or {}
            )
            
            self.current_run = run
            
            # Log description if provided
            if experiment_config.description:
                mlflow.set_tag("mlflow.note.content", experiment_config.description)
            
            # Log system info
            mlflow.set_tag("mlflow.source.type", "LOCAL")
            mlflow.set_tag("platform", "trading_platform")
            mlflow.set_tag("timestamp", datetime.now().isoformat())
            
            logger.info(f"Started MLflow run: {run.info.run_id}")
            return run.info.run_id
            
        except Exception as e:
            logger.error(f"Error starting MLflow run: {e}")
            raise
    
    def log_parameters(self, parameters: Dict[str, Any]):
        """Log parameters to current run."""
        try:
            # Convert complex objects to strings
            processed_params = {}
            for key, value in parameters.items():
                if isinstance(value, (dict, list)):
                    processed_params[key] = json.dumps(value)
                elif isinstance(value, (np.ndarray, pd.Series, pd.DataFrame)):
                    processed_params[key] = str(type(value).__name__)
                else:
                    processed_params[key] = str(value)
            
            mlflow.log_params(processed_params)
            logger.debug(f"Logged {len(processed_params)} parameters")
            
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")
    
    def log_metrics(self, metrics: Union[Dict[str, float], ModelMetrics]):
        """Log metrics to current run."""
        try:
            if isinstance(metrics, ModelMetrics):
                metrics_dict = asdict(metrics)
                # Remove non-numeric fields
                metrics_dict = {k: v for k, v in metrics_dict.items() 
                              if isinstance(v, (int, float)) and not isinstance(v, bool)}
            else:
                metrics_dict = metrics
            
            # Filter out NaN and inf values
            processed_metrics = {}
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                    processed_metrics[key] = float(value)
            
            mlflow.log_metrics(processed_metrics)
            logger.debug(f"Logged {len(processed_metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def log_artifacts(self, 
                     artifacts: Dict[str, Any],
                     artifact_path: str = "artifacts"):
        """Log artifacts to current run."""
        try:
            temp_dir = Path("temp_artifacts")
            temp_dir.mkdir(exist_ok=True)
            
            for name, artifact in artifacts.items():
                file_path = temp_dir / f"{name}.json"
                
                if isinstance(artifact, (dict, list)):
                    with open(file_path, 'w') as f:
                        json.dump(artifact, f, indent=2, default=str)
                elif isinstance(artifact, pd.DataFrame):
                    csv_path = temp_dir / f"{name}.csv"
                    artifact.to_csv(csv_path, index=False)
                    file_path = csv_path
                elif isinstance(artifact, np.ndarray):
                    np.save(temp_dir / f"{name}.npy", artifact)
                    file_path = temp_dir / f"{name}.npy"
                else:
                    with open(file_path, 'w') as f:
                        f.write(str(artifact))
                
                mlflow.log_artifact(str(file_path), artifact_path)
            
            # Cleanup temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            logger.debug(f"Logged {len(artifacts)} artifacts")
            
        except Exception as e:
            logger.error(f"Error logging artifacts: {e}")
    
    def log_model(self, 
                  model,
                  model_name: str,
                  signature=None,
                  input_example=None,
                  conda_env=None,
                  registered_model_name: str = None):
        """Log model to current run."""
        try:
            # Determine model type and log appropriately
            model_type = type(model).__name__
            
            if hasattr(model, 'fit') and hasattr(model, 'predict'):
                # Scikit-learn compatible model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=model_name,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env,
                    registered_model_name=registered_model_name
                )
            else:
                # Fallback to pickle
                mlflow.log_artifact(
                    local_path=self._save_model_temp(model, model_name),
                    artifact_path=f"models/{model_name}"
                )
            
            # Log model metadata
            mlflow.set_tag("model.type", model_type)
            mlflow.set_tag("model.name", model_name)
            
            logger.info(f"Logged model: {model_name} (type: {model_type})")
            
        except Exception as e:
            logger.error(f"Error logging model {model_name}: {e}")
    
    def _save_model_temp(self, model, model_name: str) -> str:
        """Save model to temporary file for logging."""
        temp_dir = Path("temp_models")
        temp_dir.mkdir(exist_ok=True)
        
        model_path = temp_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return str(model_path)
    
    def end_run(self, status: str = "FINISHED"):
        """End current MLflow run."""
        try:
            mlflow.end_run(status=status)
            self.current_run = None
            logger.info(f"Ended MLflow run with status: {status}")
            
        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")
    
    async def log_complete_experiment(self,
                                    experiment_config: ExperimentConfig,
                                    model,
                                    parameters: Dict[str, Any],
                                    metrics: Union[Dict[str, float], ModelMetrics],
                                    artifacts: Dict[str, Any] = None,
                                    model_name: str = "model") -> ExperimentResult:
        """
        Log complete experiment with model, parameters, metrics, and artifacts.
        
        Args:
            experiment_config: Experiment configuration
            model: Trained model to log
            parameters: Model parameters and hyperparameters
            metrics: Performance metrics
            artifacts: Additional artifacts to log
            model_name: Name for the logged model
            
        Returns:
            Experiment result with run information
        """
        try:
            # Start run
            run_id = await self.start_run(experiment_config)
            
            # Log parameters
            self.log_parameters(parameters)
            
            # Log metrics
            self.log_metrics(metrics)
            
            # Log artifacts
            if artifacts:
                self.log_artifacts(artifacts)
            
            # Log model
            self.log_model(model, model_name)
            
            # Get run info
            run = mlflow.get_run(run_id)
            
            # End run
            self.end_run("FINISHED")
            
            return ExperimentResult(
                experiment_id=run.info.experiment_id,
                run_id=run_id,
                artifact_uri=run.info.artifact_uri,
                status="FINISHED",
                metrics=run.data.metrics,
                parameters=run.data.params,
                tags=run.data.tags,
                model_uri=f"runs:/{run_id}/{model_name}"
            )
            
        except Exception as e:
            logger.error(f"Error logging complete experiment: {e}")
            self.end_run("FAILED")
            raise
    
    def search_runs(self, 
                   experiment_name: str,
                   filter_string: str = "",
                   order_by: List[str] = None,
                   max_results: int = 100) -> List[Dict[str, Any]]:
        """Search runs in experiment."""
        try:
            experiment_id = self.get_or_create_experiment(experiment_name)
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment_id],
                filter_string=filter_string,
                order_by=order_by or ["start_time DESC"],
                max_results=max_results
            )
            
            return runs.to_dict('records') if not runs.empty else []
            
        except Exception as e:
            logger.error(f"Error searching runs: {e}")
            return []
    
    def get_best_run(self, 
                    experiment_name: str,
                    metric_name: str,
                    ascending: bool = False) -> Optional[Dict[str, Any]]:
        """Get best run based on metric."""
        try:
            order_direction = "ASC" if ascending else "DESC"
            filter_string = f"metrics.{metric_name} IS NOT NULL"
            
            runs = self.search_runs(
                experiment_name=experiment_name,
                filter_string=filter_string,
                order_by=[f"metrics.{metric_name} {order_direction}"],
                max_results=1
            )
            
            return runs[0] if runs else None
            
        except Exception as e:
            logger.error(f"Error getting best run: {e}")
            return None
    
    def load_model(self, model_uri: str):
        """Load model from MLflow."""
        try:
            return mlflow.sklearn.load_model(model_uri)
        except:
            # Fallback to generic loader
            return mlflow.pyfunc.load_model(model_uri)
    
    def register_model(self, 
                      model_uri: str,
                      name: str,
                      description: str = None,
                      tags: Dict[str, str] = None) -> str:
        """Register model in MLflow model registry."""
        try:
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=name,
                tags=tags
            )
            
            if description:
                client = MlflowClient()
                client.update_model_version(
                    name=name,
                    version=model_version.version,
                    description=description
                )
            
            logger.info(f"Registered model {name} version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise
    
    def transition_model_stage(self, 
                              name: str,
                              version: str,
                              stage: str,
                              archive_existing_versions: bool = False):
        """Transition model to different stage."""
        try:
            client = MlflowClient()
            client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions
            )
            
            logger.info(f"Transitioned model {name} v{version} to {stage}")
            
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            raise
    
    def get_experiment_summary(self, experiment_name: str) -> Dict[str, Any]:
        """Get experiment summary statistics."""
        try:
            runs = self.search_runs(experiment_name, max_results=1000)
            
            if not runs:
                return {"error": "No runs found"}
            
            # Calculate summary statistics
            total_runs = len(runs)
            successful_runs = len([r for r in runs if r.get('status') == 'FINISHED'])
            
            # Get metric statistics
            metrics_summary = {}
            if runs:
                numeric_cols = [col for col in runs[0].keys() if col.startswith('metrics.')]
                for col in numeric_cols:
                    metric_name = col.replace('metrics.', '')
                    values = [r[col] for r in runs if r.get(col) is not None]
                    if values:
                        metrics_summary[metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'count': len(values)
                        }
            
            return {
                'experiment_name': experiment_name,
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'success_rate': successful_runs / total_runs if total_runs > 0 else 0,
                'metrics_summary': metrics_summary,
                'last_run_time': runs[0].get('start_time') if runs else None
            }
            
        except Exception as e:
            logger.error(f"Error getting experiment summary: {e}")
            return {"error": str(e)}


# Integration with existing model evaluation
class MLflowModelEvaluator:
    """MLflow integration for model evaluation framework."""
    
    def __init__(self, tracker: MLflowTracker):
        self.tracker = tracker
    
    async def evaluate_with_tracking(self,
                                   model,
                                   model_name: str,
                                   X_train, y_train,
                                   X_test, y_test,
                                   parameters: Dict[str, Any],
                                   symbol: str = "unknown") -> ExperimentResult:
        """Evaluate model with MLflow tracking."""
        
        # Create experiment configuration
        experiment_config = ExperimentConfig(
            experiment_name=f"model_evaluation_{symbol}",
            run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={
                "model_type": model_name,
                "symbol": symbol,
                "framework": "sklearn",
                "evaluation_type": "backtesting"
            },
            description=f"Model evaluation for {model_name} on {symbol}"
        )
        
        # Train model and calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import time
        
        # Training
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Prediction
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = ModelMetrics(
            mse=mse,
            mae=mae,
            rmse=np.sqrt(mse),
            r2=r2,
            cv_mean=r2,  # Simplified
            cv_std=0.0,
            sharpe_ratio=0.0,  # Would need price data
            hit_rate=0.0,
            max_drawdown=0.0,
            total_return=0.0,
            volatility=0.0,
            information_ratio=0.0,
            calmar_ratio=0.0,
            training_time=training_time,
            prediction_time=prediction_time,
            model_type=model_name,
            feature_count=X_train.shape[1],
            data_points=len(X_train)
        )
        
        # Prepare artifacts
        artifacts = {
            "feature_importance": dict(zip(
                [f"feature_{i}" for i in range(X_train.shape[1])],
                getattr(model, 'feature_importances_', [0] * X_train.shape[1])
            )) if hasattr(model, 'feature_importances_') else {},
            "predictions": y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
            "model_params": parameters
        }
        
        # Log complete experiment
        return await self.tracker.log_complete_experiment(
            experiment_config=experiment_config,
            model=model,
            parameters=parameters,
            metrics=metrics,
            artifacts=artifacts,
            model_name=model_name
        )

    async def get_tracking_status(self) -> Dict[str, Any]:
        """Get tracking service status and configuration."""
        try:
            if not MLFLOW_AVAILABLE:
                return {
                    "status": "unavailable",
                    "reason": "MLflow not installed",
                    "mlflow_version": None
                }
            
            # Check if tracking URI is accessible
            try:
                client = self._get_client()
                experiments = client.search_experiments()
                experiment_count = len(experiments)
            except Exception:
                experiment_count = 0
            
            return {
                "status": "active",
                "mlflow_version": mlflow.__version__,
                "tracking_uri": self.tracking_uri,
                "experiment_count": experiment_count,
                "client_active": True,
                "default_experiment": self.default_experiment
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "mlflow_version": mlflow.__version__ if MLFLOW_AVAILABLE else None
            }

    async def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        try:
            client = self._get_client()
            experiments = client.search_experiments()
            
            return [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "artifact_location": exp.artifact_location,
                    "creation_time": exp.creation_time,
                    "last_update_time": exp.last_update_time
                }
                for exp in experiments
            ]
        except Exception as e:
            logger.error(f"Error listing experiments: {e}")
            return []

    async def list_runs(self, experiment_name: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List runs for an experiment."""
        try:
            client = self._get_client()
            
            if experiment_name:
                experiment = client.get_experiment_by_name(experiment_name)
                if not experiment:
                    return []
                experiment_ids = [experiment.experiment_id]
            else:
                experiments = client.search_experiments()
                experiment_ids = [exp.experiment_id for exp in experiments]
            
            runs = client.search_runs(experiment_ids=experiment_ids, max_results=limit)
            
            return [
                {
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": dict(run.data.metrics),
                    "params": dict(run.data.params),
                    "tags": dict(run.data.tags)
                }
                for run in runs
            ]
        except Exception as e:
            logger.error(f"Error listing runs: {e}")
            return []

    async def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific run."""
        try:
            client = self._get_client()
            run = client.get_run(run_id)
            
            return {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "lifecycle_stage": run.info.lifecycle_stage,
                "artifact_uri": run.info.artifact_uri,
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
                "tags": dict(run.data.tags)
            }
        except Exception as e:
            logger.error(f"Error getting run details: {e}")
            return {"error": str(e)}

    async def list_registered_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        try:
            client = self._get_client()
            models = client.search_registered_models()
            
            return [
                {
                    "name": model.name,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "description": model.description,
                    "latest_version": model.latest_versions[0].version if model.latest_versions else None,
                    "stage": model.latest_versions[0].current_stage if model.latest_versions else None
                }
                for model in models
            ]
        except Exception as e:
            logger.error(f"Error listing registered models: {e}")
            return []

    async def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a registered model."""
        try:
            client = self._get_client()
            model = client.get_registered_model(model_name)
            
            versions = [
                {
                    "version": version.version,
                    "stage": version.current_stage,
                    "creation_timestamp": version.creation_timestamp,
                    "last_updated_timestamp": version.last_updated_timestamp,
                    "description": version.description,
                    "source": version.source,
                    "run_id": version.run_id
                }
                for version in model.latest_versions
            ]
            
            return {
                "name": model.name,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "description": model.description,
                "versions": versions
            }
        except Exception as e:
            logger.error(f"Error getting model details: {e}")
            return {"error": str(e)}

    async def transition_model_stage(self, model_name: str, version: str, 
                                   stage: str, description: str = None) -> Dict[str, Any]:
        """Transition a model version to a new stage."""
        try:
            client = self._get_client()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=False
            )
            
            return {
                "success": True,
                "model_name": model_name,
                "version": version,
                "new_stage": stage
            }
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            return {"error": str(e)}

    async def search_runs(self, experiment_names: List[str] = None, 
                         filter_string: str = None, order_by: List[str] = None,
                         max_results: int = 100) -> List[Dict[str, Any]]:
        """Search runs with flexible filtering."""
        try:
            client = self._get_client()
            
            if experiment_names:
                experiment_ids = []
                for name in experiment_names:
                    try:
                        exp = client.get_experiment_by_name(name)
                        if exp:
                            experiment_ids.append(exp.experiment_id)
                    except:
                        continue
            else:
                experiments = client.search_experiments()
                experiment_ids = [exp.experiment_id for exp in experiments]
            
            runs = client.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                order_by=order_by,
                max_results=max_results
            )
            
            return [
                {
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "experiment_name": client.get_experiment(run.info.experiment_id).name,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": dict(run.data.metrics),
                    "params": dict(run.data.params),
                    "tags": dict(run.data.tags)
                }
                for run in runs
            ]
        except Exception as e:
            logger.error(f"Error searching runs: {e}")
            return []


# Demo function
async def run_mlflow_demo():
    """Demo of MLflow tracking integration."""
    print("MLflow Tracking Integration Demo")
    print("=" * 50)
    
    if not MLFLOW_AVAILABLE:
        print("MLflow not available. Please install with: pip install mlflow")
        return
    
    # Initialize tracker
    tracker = MLflowTracker(
        tracking_uri="file:./demo_mlruns",
        experiment_name="demo_financial_models"
    )
    
    # Generate synthetic data
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Create and evaluate model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    parameters = {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42,
        "dataset_size": len(X_train),
        "feature_count": X.shape[1]
    }
    
    # Use MLflow evaluator
    evaluator = MLflowModelEvaluator(tracker)
    
    print("Running model evaluation with MLflow tracking...")
    result = await evaluator.evaluate_with_tracking(
        model=model,
        model_name="RandomForest",
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        parameters=parameters,
        symbol="DEMO"
    )
    
    print(f"Experiment completed!")
    print(f"Run ID: {result.run_id}")
    print(f"Model URI: {result.model_uri}")
    print(f"Artifact URI: {result.artifact_uri}")
    
    # Search for runs
    print("\nSearching recent runs...")
    runs = tracker.search_runs("demo_financial_models", max_results=5)
    print(f"Found {len(runs)} runs")
    
    for run in runs[:3]:
        print(f"  Run: {run.get('run_id', 'unknown')[:8]} - R²: {run.get('metrics.r2', 0):.4f}")
    
    # Get experiment summary
    summary = tracker.get_experiment_summary("demo_financial_models")
    print(f"\nExperiment Summary:")
    print(f"  Total runs: {summary.get('total_runs', 0)}")
    print(f"  Success rate: {summary.get('success_rate', 0):.1%}")
    
    print("\nMLflow demo completed!")
    print(f"View results: mlflow ui --backend-store-uri {tracker.tracking_uri}")


if __name__ == "__main__":
    asyncio.run(run_mlflow_demo())