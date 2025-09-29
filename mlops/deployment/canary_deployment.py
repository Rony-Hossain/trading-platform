"""
Canary Deployment System for ML Models
Implements shadow/canary testing with statistical validation
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import statistics
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
import uuid
import time
import mlflow
from mlflow.tracking import MlflowClient

from ..model_registry.mlflow_registry import MLflowModelRegistry, ModelStage, ModelType

logger = logging.getLogger(__name__)

class DeploymentStrategy(Enum):
    """Deployment strategies for canary testing"""
    SHADOW = "shadow"          # New model runs in parallel but doesn't serve traffic
    CANARY = "canary"          # New model serves small percentage of traffic
    BLUE_GREEN = "blue_green"  # Switch all traffic at once after validation
    A_B_TEST = "a_b_test"      # Split traffic for direct comparison

class ModelStatus(Enum):
    """Status of deployed models"""
    DEPLOYING = "deploying"
    RUNNING = "running"
    TESTING = "testing"
    VALIDATED = "validated"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ARCHIVED = "archived"

@dataclass
class PredictionMetrics:
    """Metrics for model prediction performance"""
    model_name: str
    model_version: str
    prediction_time: float
    confidence_score: float
    prediction_value: Any
    request_id: str
    timestamp: datetime
    latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    
@dataclass
class StatisticalTestResult:
    """Result of statistical significance testing"""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str

@dataclass
class CanaryConfig:
    """Configuration for canary deployment"""
    model_name: str
    candidate_version: str
    production_version: str
    traffic_percentage: float = 5.0  # Percentage of traffic for canary
    min_samples: int = 1000
    max_duration_hours: int = 24
    significance_level: float = 0.05
    effect_size_threshold: float = 0.1
    success_rate_threshold: float = 0.95
    latency_threshold_ms: float = 100
    error_rate_threshold: float = 0.01
    rollback_on_failure: bool = True
    auto_promote_on_success: bool = False

class CanaryDeploymentManager:
    """
    Manages canary deployments with statistical validation
    
    Features:
    - Shadow testing (parallel execution without serving)
    - Canary testing (small traffic percentage)
    - A/B testing with statistical significance
    - Automated rollback on failure
    - Performance monitoring and alerting
    """
    
    def __init__(self, model_registry: MLflowModelRegistry):
        self.model_registry = model_registry
        self.active_deployments: Dict[str, CanaryConfig] = {}
        self.deployment_metrics: Dict[str, List[PredictionMetrics]] = {}
        self.deployment_history: List[Dict[str, Any]] = []
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Load models cache
        self.loaded_models: Dict[str, Any] = {}
        
        # Statistical test configurations
        self.statistical_tests = {
            "accuracy": self._test_accuracy_difference,
            "latency": self._test_latency_difference,
            "error_rate": self._test_error_rate_difference,
            "prediction_distribution": self._test_prediction_distribution
        }
    
    async def start_canary_deployment(self, config: CanaryConfig) -> str:
        """Start a canary deployment"""
        try:
            deployment_id = str(uuid.uuid4())
            
            # Validate configuration
            await self._validate_canary_config(config)
            
            # Load both models
            await self._load_model_versions(config)
            
            # Initialize metrics tracking
            self.deployment_metrics[deployment_id] = []
            self.active_deployments[deployment_id] = config
            
            # Start monitoring task
            asyncio.create_task(self._monitor_canary_deployment(deployment_id))
            
            logger.info(f"Started canary deployment {deployment_id} for {config.model_name} "
                       f"v{config.candidate_version} vs v{config.production_version}")
            
            return deployment_id
            
        except Exception as e:
            logger.error(f"Error starting canary deployment: {e}")
            raise
    
    async def _validate_canary_config(self, config: CanaryConfig) -> None:
        """Validate canary deployment configuration"""
        # Check if models exist
        candidate = self.model_registry.client.get_model_version(
            config.model_name, config.candidate_version
        )
        production = self.model_registry.client.get_model_version(
            config.model_name, config.production_version
        )
        
        if not candidate or not production:
            raise ValueError("Model versions not found in registry")
        
        # Validate parameters
        if not 0 < config.traffic_percentage <= 100:
            raise ValueError("Traffic percentage must be between 0 and 100")
        
        if config.min_samples < 100:
            raise ValueError("Minimum samples should be at least 100")
    
    async def _load_model_versions(self, config: CanaryConfig) -> None:
        """Load both candidate and production model versions"""
        try:
            # Load candidate model
            candidate_uri = f"models:/{config.model_name}/{config.candidate_version}"
            candidate_model = mlflow.pyfunc.load_model(candidate_uri)
            self.loaded_models[f"{config.model_name}_candidate"] = candidate_model
            
            # Load production model
            production_uri = f"models:/{config.model_name}/{config.production_version}"
            production_model = mlflow.pyfunc.load_model(production_uri)
            self.loaded_models[f"{config.model_name}_production"] = production_model
            
            logger.info(f"Loaded models for canary deployment: {config.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model versions: {e}")
            raise
    
    async def make_prediction(self, deployment_id: str, input_data: Any, 
                            strategy: DeploymentStrategy = DeploymentStrategy.CANARY) -> Dict[str, Any]:
        """
        Make prediction using canary deployment strategy
        
        Returns:
            Dict containing prediction and metadata
        """
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        config = self.active_deployments[deployment_id]
        request_id = str(uuid.uuid4())
        
        try:
            if strategy == DeploymentStrategy.SHADOW:
                # Run both models, return production result
                result = await self._shadow_prediction(config, input_data, request_id)
            elif strategy == DeploymentStrategy.CANARY:
                # Route based on traffic percentage
                result = await self._canary_prediction(config, input_data, request_id)
            elif strategy == DeploymentStrategy.A_B_TEST:
                # Random assignment for A/B testing
                result = await self._ab_test_prediction(config, input_data, request_id)
            else:
                raise ValueError(f"Unsupported deployment strategy: {strategy}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction for deployment {deployment_id}: {e}")
            raise
    
    async def _shadow_prediction(self, config: CanaryConfig, input_data: Any, 
                               request_id: str) -> Dict[str, Any]:
        """Make shadow prediction (both models run, production serves)"""
        start_time = time.time()
        
        # Run both models concurrently
        tasks = [
            self._run_model_prediction(f"{config.model_name}_production", input_data, request_id),
            self._run_model_prediction(f"{config.model_name}_candidate", input_data, request_id)
        ]
        
        production_result, candidate_result = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log metrics for both models
        if not isinstance(production_result, Exception):
            await self._log_prediction_metrics(
                config.model_name, config.production_version, production_result, 
                request_id, time.time() - start_time, "production"
            )
        
        if not isinstance(candidate_result, Exception):
            await self._log_prediction_metrics(
                config.model_name, config.candidate_version, candidate_result, 
                request_id, time.time() - start_time, "candidate"
            )
        
        # Return production result
        if isinstance(production_result, Exception):
            raise production_result
        
        return {
            "prediction": production_result["prediction"],
            "confidence": production_result.get("confidence", 1.0),
            "model_version": config.production_version,
            "strategy": "shadow",
            "request_id": request_id,
            "shadow_prediction": candidate_result.get("prediction") if not isinstance(candidate_result, Exception) else None
        }
    
    async def _canary_prediction(self, config: CanaryConfig, input_data: Any, 
                               request_id: str) -> Dict[str, Any]:
        """Make canary prediction (route based on traffic percentage)"""
        # Determine which model to use based on traffic percentage
        use_candidate = np.random.random() < (config.traffic_percentage / 100.0)
        
        if use_candidate:
            model_key = f"{config.model_name}_candidate"
            version = config.candidate_version
            model_type = "candidate"
        else:
            model_key = f"{config.model_name}_production"
            version = config.production_version
            model_type = "production"
        
        start_time = time.time()
        result = await self._run_model_prediction(model_key, input_data, request_id)
        
        await self._log_prediction_metrics(
            config.model_name, version, result, request_id, 
            time.time() - start_time, model_type
        )
        
        return {
            "prediction": result["prediction"],
            "confidence": result.get("confidence", 1.0),
            "model_version": version,
            "strategy": "canary",
            "request_id": request_id,
            "model_type": model_type
        }
    
    async def _ab_test_prediction(self, config: CanaryConfig, input_data: Any, 
                                request_id: str) -> Dict[str, Any]:
        """Make A/B test prediction (50/50 split)"""
        use_candidate = np.random.random() < 0.5
        
        if use_candidate:
            model_key = f"{config.model_name}_candidate"
            version = config.candidate_version
            model_type = "candidate"
        else:
            model_key = f"{config.model_name}_production"
            version = config.production_version
            model_type = "production"
        
        start_time = time.time()
        result = await self._run_model_prediction(model_key, input_data, request_id)
        
        await self._log_prediction_metrics(
            config.model_name, version, result, request_id, 
            time.time() - start_time, model_type
        )
        
        return {
            "prediction": result["prediction"],
            "confidence": result.get("confidence", 1.0),
            "model_version": version,
            "strategy": "a_b_test",
            "request_id": request_id,
            "model_type": model_type
        }
    
    async def _run_model_prediction(self, model_key: str, input_data: Any, 
                                  request_id: str) -> Dict[str, Any]:
        """Run prediction on a specific model"""
        if model_key not in self.loaded_models:
            raise ValueError(f"Model {model_key} not loaded")
        
        model = self.loaded_models[model_key]
        
        # Run prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        try:
            prediction = await loop.run_in_executor(
                self.executor, 
                lambda: model.predict(input_data)
            )
            
            # Extract confidence if available
            confidence = 1.0
            if hasattr(model, 'predict_proba'):
                proba = await loop.run_in_executor(
                    self.executor,
                    lambda: model.predict_proba(input_data)
                )
                confidence = float(np.max(proba))
            
            return {
                "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                "confidence": confidence,
                "request_id": request_id
            }
            
        except Exception as e:
            logger.error(f"Error running model prediction: {e}")
            raise
    
    async def _log_prediction_metrics(self, model_name: str, version: str, 
                                    result: Dict[str, Any], request_id: str,
                                    latency: float, model_type: str) -> None:
        """Log prediction metrics for analysis"""
        metrics = PredictionMetrics(
            model_name=model_name,
            model_version=version,
            prediction_time=time.time(),
            confidence_score=result.get("confidence", 1.0),
            prediction_value=result["prediction"],
            request_id=request_id,
            timestamp=datetime.utcnow(),
            latency_ms=latency * 1000,
            memory_usage_mb=0.0,  # Would integrate with system monitoring
            cpu_usage_percent=0.0
        )
        
        # Store metrics for statistical analysis
        for deployment_id, config in self.active_deployments.items():
            if config.model_name == model_name:
                self.deployment_metrics[deployment_id].append(metrics)
                break
    
    async def _monitor_canary_deployment(self, deployment_id: str) -> None:
        """Monitor canary deployment and trigger actions based on results"""
        config = self.active_deployments[deployment_id]
        start_time = datetime.utcnow()
        
        logger.info(f"Starting monitoring for deployment {deployment_id}")
        
        while deployment_id in self.active_deployments:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Check if we have enough samples
                metrics = self.deployment_metrics[deployment_id]
                if len(metrics) < config.min_samples:
                    continue
                
                # Check maximum duration
                if datetime.utcnow() - start_time > timedelta(hours=config.max_duration_hours):
                    logger.warning(f"Deployment {deployment_id} exceeded maximum duration, stopping")
                    await self._stop_deployment(deployment_id, "timeout")
                    break
                
                # Run statistical analysis
                analysis_result = await self._analyze_deployment_performance(deployment_id)
                
                if analysis_result["should_rollback"]:
                    logger.warning(f"Deployment {deployment_id} failed validation, rolling back")
                    await self._rollback_deployment(deployment_id)
                    break
                elif analysis_result["should_promote"]:
                    logger.info(f"Deployment {deployment_id} passed validation, promoting")
                    await self._promote_deployment(deployment_id)
                    break
                    
            except Exception as e:
                logger.error(f"Error monitoring deployment {deployment_id}: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _analyze_deployment_performance(self, deployment_id: str) -> Dict[str, Any]:
        """Analyze deployment performance using statistical tests"""
        config = self.active_deployments[deployment_id]
        metrics = self.deployment_metrics[deployment_id]
        
        # Separate metrics by model type
        candidate_metrics = [m for m in metrics if m.model_version == config.candidate_version]
        production_metrics = [m for m in metrics if m.model_version == config.production_version]
        
        if len(candidate_metrics) < 100 or len(production_metrics) < 100:
            return {"should_rollback": False, "should_promote": False, "reason": "insufficient_data"}
        
        analysis = {
            "should_rollback": False,
            "should_promote": False,
            "reason": "",
            "test_results": {},
            "summary_stats": {}
        }
        
        try:
            # Test latency performance
            candidate_latencies = [m.latency_ms for m in candidate_metrics]
            production_latencies = [m.latency_ms for m in production_metrics]
            
            latency_test = await self._test_latency_difference(
                candidate_latencies, production_latencies, config.significance_level
            )
            analysis["test_results"]["latency"] = asdict(latency_test)
            
            # Check if candidate is significantly slower
            if latency_test.is_significant and latency_test.effect_size > config.effect_size_threshold:
                if statistics.mean(candidate_latencies) > statistics.mean(production_latencies):
                    analysis["should_rollback"] = True
                    analysis["reason"] = "significantly_slower_latency"
                    return analysis
            
            # Test error rates (if we have error tracking)
            candidate_errors = len([m for m in candidate_metrics if m.confidence_score < 0.5])
            production_errors = len([m for m in production_metrics if m.confidence_score < 0.5])
            
            candidate_error_rate = candidate_errors / len(candidate_metrics)
            production_error_rate = production_errors / len(production_metrics)
            
            if candidate_error_rate > config.error_rate_threshold:
                analysis["should_rollback"] = True
                analysis["reason"] = "high_error_rate"
                return analysis
            
            # Test prediction distributions (for regression/classification)
            pred_distribution_test = await self._test_prediction_distribution(
                [m.prediction_value for m in candidate_metrics],
                [m.prediction_value for m in production_metrics],
                config.significance_level
            )
            analysis["test_results"]["prediction_distribution"] = asdict(pred_distribution_test)
            
            # Summary statistics
            analysis["summary_stats"] = {
                "candidate": {
                    "sample_count": len(candidate_metrics),
                    "avg_latency_ms": statistics.mean(candidate_latencies),
                    "avg_confidence": statistics.mean([m.confidence_score for m in candidate_metrics]),
                    "error_rate": candidate_error_rate
                },
                "production": {
                    "sample_count": len(production_metrics),
                    "avg_latency_ms": statistics.mean(production_latencies),
                    "avg_confidence": statistics.mean([m.confidence_score for m in production_metrics]),
                    "error_rate": production_error_rate
                }
            }
            
            # Decide on promotion
            if config.auto_promote_on_success:
                # All tests passed, no significant degradation
                significant_degradations = [
                    test for test in analysis["test_results"].values() 
                    if test["is_significant"] and test["effect_size"] > config.effect_size_threshold
                ]
                
                if not significant_degradations and candidate_error_rate <= production_error_rate:
                    analysis["should_promote"] = True
                    analysis["reason"] = "validation_passed"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing deployment performance: {e}")
            analysis["should_rollback"] = True
            analysis["reason"] = f"analysis_error: {str(e)}"
            return analysis
    
    async def _test_latency_difference(self, candidate_latencies: List[float], 
                                     production_latencies: List[float],
                                     significance_level: float) -> StatisticalTestResult:
        """Test if there's a significant difference in latency"""
        try:
            statistic, p_value = stats.mannwhitneyu(
                candidate_latencies, production_latencies, alternative='two-sided'
            )
            
            is_significant = p_value < significance_level
            
            # Calculate effect size (Cohen's d)
            candidate_mean = statistics.mean(candidate_latencies)
            production_mean = statistics.mean(production_latencies)
            pooled_std = np.sqrt((np.var(candidate_latencies) + np.var(production_latencies)) / 2)
            effect_size = abs(candidate_mean - production_mean) / pooled_std if pooled_std > 0 else 0
            
            # Confidence interval for difference in means
            diff = candidate_mean - production_mean
            se_diff = np.sqrt(np.var(candidate_latencies)/len(candidate_latencies) + 
                            np.var(production_latencies)/len(production_latencies))
            ci_lower = diff - 1.96 * se_diff
            ci_upper = diff + 1.96 * se_diff
            
            interpretation = ""
            if is_significant:
                if candidate_mean > production_mean:
                    interpretation = f"Candidate model is significantly slower by {diff:.2f}ms on average"
                else:
                    interpretation = f"Candidate model is significantly faster by {abs(diff):.2f}ms on average"
            else:
                interpretation = "No significant difference in latency between models"
            
            return StatisticalTestResult(
                test_name="latency_comparison",
                statistic=statistic,
                p_value=p_value,
                is_significant=is_significant,
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.error(f"Error testing latency difference: {e}")
            raise
    
    async def _test_prediction_distribution(self, candidate_predictions: List[Any],
                                          production_predictions: List[Any],
                                          significance_level: float) -> StatisticalTestResult:
        """Test if prediction distributions are significantly different"""
        try:
            # Convert predictions to numeric if possible
            def to_numeric(predictions):
                try:
                    return [float(p) if isinstance(p, (int, float)) else hash(str(p)) for p in predictions]
                except:
                    return [hash(str(p)) for p in predictions]
            
            candidate_numeric = to_numeric(candidate_predictions)
            production_numeric = to_numeric(production_predictions)
            
            # Use Kolmogorov-Smirnov test for distribution comparison
            statistic, p_value = stats.ks_2samp(candidate_numeric, production_numeric)
            
            is_significant = p_value < significance_level
            effect_size = statistic  # KS statistic is itself an effect size measure
            
            interpretation = ""
            if is_significant:
                interpretation = f"Prediction distributions are significantly different (KS statistic: {statistic:.3f})"
            else:
                interpretation = "Prediction distributions are not significantly different"
            
            return StatisticalTestResult(
                test_name="prediction_distribution",
                statistic=statistic,
                p_value=p_value,
                is_significant=is_significant,
                effect_size=effect_size,
                confidence_interval=(0, statistic),  # KS confidence interval
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.error(f"Error testing prediction distribution: {e}")
            raise
    
    async def _rollback_deployment(self, deployment_id: str) -> None:
        """Rollback canary deployment"""
        try:
            config = self.active_deployments[deployment_id]
            
            # Log rollback event
            self.deployment_history.append({
                "deployment_id": deployment_id,
                "model_name": config.model_name,
                "candidate_version": config.candidate_version,
                "production_version": config.production_version,
                "action": "rollback",
                "timestamp": datetime.utcnow().isoformat(),
                "reason": "failed_validation"
            })
            
            # Clean up
            await self._stop_deployment(deployment_id, "rollback")
            
            logger.info(f"Rolled back deployment {deployment_id}")
            
        except Exception as e:
            logger.error(f"Error rolling back deployment {deployment_id}: {e}")
            raise
    
    async def _promote_deployment(self, deployment_id: str) -> None:
        """Promote candidate model to production"""
        try:
            config = self.active_deployments[deployment_id]
            
            # Promote in model registry
            self.model_registry.transition_model_stage(
                model_name=config.model_name,
                version=config.candidate_version,
                stage=ModelStage.PRODUCTION
            )
            
            # Log promotion event
            self.deployment_history.append({
                "deployment_id": deployment_id,
                "model_name": config.model_name,
                "candidate_version": config.candidate_version,
                "production_version": config.production_version,
                "action": "promote",
                "timestamp": datetime.utcnow().isoformat(),
                "reason": "validation_passed"
            })
            
            # Clean up
            await self._stop_deployment(deployment_id, "promote")
            
            logger.info(f"Promoted deployment {deployment_id} to production")
            
        except Exception as e:
            logger.error(f"Error promoting deployment {deployment_id}: {e}")
            raise
    
    async def _stop_deployment(self, deployment_id: str, reason: str) -> None:
        """Stop canary deployment and clean up resources"""
        try:
            if deployment_id in self.active_deployments:
                config = self.active_deployments[deployment_id]
                
                # Unload models
                candidate_key = f"{config.model_name}_candidate"
                production_key = f"{config.model_name}_production"
                
                self.loaded_models.pop(candidate_key, None)
                self.loaded_models.pop(production_key, None)
                
                # Clean up tracking
                del self.active_deployments[deployment_id]
                
                # Keep metrics for historical analysis
                logger.info(f"Stopped deployment {deployment_id} (reason: {reason})")
            
        except Exception as e:
            logger.error(f"Error stopping deployment {deployment_id}: {e}")
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a deployment"""
        if deployment_id not in self.active_deployments:
            return None
        
        config = self.active_deployments[deployment_id]
        metrics = self.deployment_metrics.get(deployment_id, [])
        
        return {
            "deployment_id": deployment_id,
            "model_name": config.model_name,
            "candidate_version": config.candidate_version,
            "production_version": config.production_version,
            "traffic_percentage": config.traffic_percentage,
            "total_requests": len(metrics),
            "candidate_requests": len([m for m in metrics if m.model_version == config.candidate_version]),
            "production_requests": len([m for m in metrics if m.model_version == config.production_version]),
            "status": "active"
        }
    
    def get_all_deployments(self) -> List[Dict[str, Any]]:
        """Get status of all active deployments"""
        return [
            self.get_deployment_status(deployment_id) 
            for deployment_id in self.active_deployments.keys()
        ]
    
    def get_deployment_history(self, model_name: str = None) -> List[Dict[str, Any]]:
        """Get deployment history"""
        if model_name:
            return [
                event for event in self.deployment_history 
                if event["model_name"] == model_name
            ]
        return self.deployment_history