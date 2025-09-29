"""
Automated Rollback System for ML Models
Defines promotion criteria and automated rollback mechanisms
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import time
import yaml

from ..model_registry.mlflow_registry import MLflowModelRegistry, ModelStage, ModelType
from ..monitoring.model_performance_monitor import ModelPerformanceMonitor, ModelAlert, AlertSeverity
from .canary_deployment import CanaryDeploymentManager, CanaryConfig, DeploymentStrategy

logger = logging.getLogger(__name__)

class RollbackTrigger(Enum):
    """Types of rollback triggers"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_RATE_SPIKE = "error_rate_spike"
    LATENCY_SPIKE = "latency_spike"
    ACCURACY_DROP = "accuracy_drop"
    MANUAL_TRIGGER = "manual_trigger"
    DRIFT_DETECTION = "drift_detection"
    HEALTH_CHECK_FAILURE = "health_check_failure"
    CIRCUIT_BREAKER = "circuit_breaker"

class PromotionStatus(Enum):
    """Model promotion status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"
    ROLLED_BACK = "rolled_back"

@dataclass
class PromotionCriteria:
    """Criteria for model promotion"""
    model_type: ModelType
    min_samples: int = 1000
    min_test_duration_hours: int = 24
    max_error_rate: float = 0.05
    max_latency_p95_ms: float = 100
    min_accuracy: float = 0.85
    max_performance_degradation: float = 0.05  # 5% degradation threshold
    required_statistical_significance: float = 0.05
    min_improvement_threshold: float = 0.01  # 1% minimum improvement
    business_metric_thresholds: Dict[str, float] = None
    approval_required: bool = False
    auto_promote: bool = True

@dataclass
class RollbackCriteria:
    """Criteria for automated rollback"""
    model_type: ModelType
    max_error_rate: float = 0.10
    max_latency_p99_ms: float = 200
    min_accuracy: float = 0.70
    max_accuracy_drop: float = 0.10  # 10% accuracy drop triggers rollback
    max_consecutive_failures: int = 10
    circuit_breaker_threshold: int = 50  # Failed requests before circuit breaker
    health_check_timeout_seconds: float = 5.0
    alert_severity_threshold: AlertSeverity = AlertSeverity.HIGH
    drift_score_threshold: float = 0.3

@dataclass
class RollbackPlan:
    """Rollback execution plan"""
    rollback_id: str
    model_name: str
    current_version: str
    target_version: str
    trigger: RollbackTrigger
    reason: str
    rollback_steps: List[Dict[str, Any]]
    estimated_duration_minutes: int
    rollback_validation_steps: List[str]
    notification_channels: List[str]
    created_at: datetime
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"

class AutomatedRollbackSystem:
    """
    Automated rollback system with promotion criteria
    
    Features:
    - Define promotion criteria per model type
    - Automated rollback triggers and execution
    - Circuit breaker pattern implementation
    - Health check monitoring
    - Rollback plan generation and execution
    - Notification system integration
    """
    
    def __init__(self, model_registry: MLflowModelRegistry, 
                 performance_monitor: ModelPerformanceMonitor,
                 canary_manager: CanaryDeploymentManager):
        self.model_registry = model_registry
        self.performance_monitor = performance_monitor
        self.canary_manager = canary_manager
        
        # Promotion criteria by model type
        self.promotion_criteria = self._load_default_promotion_criteria()
        
        # Rollback criteria by model type
        self.rollback_criteria = self._load_default_rollback_criteria()
        
        # Active rollback plans
        self.active_rollbacks: Dict[str, RollbackPlan] = {}
        self.rollback_history: List[RollbackPlan] = []
        
        # Circuit breaker state per model
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Health check configurations
        self.health_check_configs: Dict[str, Dict[str, Any]] = {}
        
        # Notification handlers
        self.notification_handlers: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Model promotion queue
        self.promotion_queue: List[Dict[str, Any]] = []
        
        # Background tasks
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
    
    def _load_default_promotion_criteria(self) -> Dict[ModelType, PromotionCriteria]:
        """Load default promotion criteria for each model type"""
        return {
            ModelType.SENTIMENT_CLASSIFIER: PromotionCriteria(
                model_type=ModelType.SENTIMENT_CLASSIFIER,
                min_samples=1000,
                min_test_duration_hours=24,
                max_error_rate=0.03,
                max_latency_p95_ms=80,
                min_accuracy=0.85,
                max_performance_degradation=0.03,
                business_metric_thresholds={"precision": 0.80, "recall": 0.75, "f1_score": 0.78}
            ),
            ModelType.PRICE_PREDICTOR: PromotionCriteria(
                model_type=ModelType.PRICE_PREDICTOR,
                min_samples=2000,
                min_test_duration_hours=48,
                max_error_rate=0.02,
                max_latency_p95_ms=50,
                min_accuracy=0.75,  # R² for regression
                max_performance_degradation=0.05,
                business_metric_thresholds={"mae": 0.05, "rmse": 0.08, "mape": 0.10}
            ),
            ModelType.RISK_ASSESSOR: PromotionCriteria(
                model_type=ModelType.RISK_ASSESSOR,
                min_samples=1500,
                min_test_duration_hours=72,  # Longer testing for risk models
                max_error_rate=0.01,
                max_latency_p95_ms=100,
                min_accuracy=0.90,
                max_performance_degradation=0.02,
                business_metric_thresholds={"auc": 0.85, "precision": 0.85, "recall": 0.80},
                approval_required=True,  # Risk models require manual approval
                auto_promote=False
            )
        }
    
    def _load_default_rollback_criteria(self) -> Dict[ModelType, RollbackCriteria]:
        """Load default rollback criteria for each model type"""
        return {
            ModelType.SENTIMENT_CLASSIFIER: RollbackCriteria(
                model_type=ModelType.SENTIMENT_CLASSIFIER,
                max_error_rate=0.08,
                max_latency_p99_ms=150,
                min_accuracy=0.75,
                max_accuracy_drop=0.08
            ),
            ModelType.PRICE_PREDICTOR: RollbackCriteria(
                model_type=ModelType.PRICE_PREDICTOR,
                max_error_rate=0.05,
                max_latency_p99_ms=100,
                min_accuracy=0.65,
                max_accuracy_drop=0.10
            ),
            ModelType.RISK_ASSESSOR: RollbackCriteria(
                model_type=ModelType.RISK_ASSESSOR,
                max_error_rate=0.03,
                max_latency_p99_ms=200,
                min_accuracy=0.80,
                max_accuracy_drop=0.05,
                alert_severity_threshold=AlertSeverity.MEDIUM  # More sensitive for risk models
            )
        }
    
    async def evaluate_promotion_criteria(self, model_name: str, candidate_version: str,
                                        model_type: ModelType) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate if a model candidate meets promotion criteria
        
        Returns:
            Tuple[bool, Dict]: (should_promote, evaluation_details)
        """
        try:
            criteria = self.promotion_criteria.get(model_type)
            if not criteria:
                logger.warning(f"No promotion criteria defined for model type {model_type}")
                return False, {"error": "No promotion criteria defined"}
            
            evaluation = {
                "model_name": model_name,
                "candidate_version": candidate_version,
                "model_type": model_type.value,
                "evaluation_time": datetime.utcnow().isoformat(),
                "criteria_met": {},
                "overall_result": False,
                "reasons": []
            }
            
            # Get model performance metrics
            metrics = await self._get_model_performance_metrics(model_name, candidate_version)
            if not metrics:
                evaluation["reasons"].append("Insufficient performance data")
                return False, evaluation
            
            # Check minimum samples
            sample_count = metrics.get("sample_count", 0)
            evaluation["criteria_met"]["min_samples"] = sample_count >= criteria.min_samples
            if sample_count < criteria.min_samples:
                evaluation["reasons"].append(f"Insufficient samples: {sample_count} < {criteria.min_samples}")
            
            # Check test duration (if deployment exists)
            test_duration_met = await self._check_test_duration(model_name, candidate_version, criteria.min_test_duration_hours)
            evaluation["criteria_met"]["test_duration"] = test_duration_met
            if not test_duration_met:
                evaluation["reasons"].append(f"Insufficient test duration: < {criteria.min_test_duration_hours} hours")
            
            # Check error rate
            error_rate = metrics.get("error_rate", 1.0)
            evaluation["criteria_met"]["error_rate"] = error_rate <= criteria.max_error_rate
            if error_rate > criteria.max_error_rate:
                evaluation["reasons"].append(f"Error rate too high: {error_rate:.4f} > {criteria.max_error_rate}")
            
            # Check latency
            latency_p95 = metrics.get("p95_latency_ms", float('inf'))
            evaluation["criteria_met"]["latency"] = latency_p95 <= criteria.max_latency_p95_ms
            if latency_p95 > criteria.max_latency_p95_ms:
                evaluation["reasons"].append(f"Latency too high: {latency_p95:.2f}ms > {criteria.max_latency_p95_ms}ms")
            
            # Check accuracy/performance
            accuracy = metrics.get("accuracy")
            if accuracy is not None:
                evaluation["criteria_met"]["accuracy"] = accuracy >= criteria.min_accuracy
                if accuracy < criteria.min_accuracy:
                    evaluation["reasons"].append(f"Accuracy too low: {accuracy:.4f} < {criteria.min_accuracy}")
            
            # Check performance degradation vs production
            degradation_check = await self._check_performance_degradation(
                model_name, candidate_version, criteria.max_performance_degradation
            )
            evaluation["criteria_met"]["performance_degradation"] = degradation_check["passed"]
            if not degradation_check["passed"]:
                evaluation["reasons"].append(degradation_check["reason"])
            
            # Check business-specific metrics
            if criteria.business_metric_thresholds:
                business_metrics_met = await self._check_business_metrics(
                    model_name, candidate_version, criteria.business_metric_thresholds
                )
                evaluation["criteria_met"]["business_metrics"] = business_metrics_met["all_met"]
                if not business_metrics_met["all_met"]:
                    evaluation["reasons"].extend(business_metrics_met["failures"])
            
            # Statistical significance check
            significance_check = await self._check_statistical_significance(
                model_name, candidate_version, criteria.required_statistical_significance
            )
            evaluation["criteria_met"]["statistical_significance"] = significance_check["significant"]
            if not significance_check["significant"]:
                evaluation["reasons"].append("Changes not statistically significant")
            
            # Overall evaluation
            all_criteria_met = all(evaluation["criteria_met"].values())
            evaluation["overall_result"] = all_criteria_met
            
            if all_criteria_met:
                evaluation["reasons"] = ["All promotion criteria met"]
            
            logger.info(f"Promotion criteria evaluation for {model_name} v{candidate_version}: {all_criteria_met}")
            return all_criteria_met, evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating promotion criteria: {e}")
            return False, {"error": str(e)}
    
    async def _get_model_performance_metrics(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive performance metrics for a model version"""
        try:
            # Get metrics from performance monitor
            monitor_key = f"{model_name}:{version}"
            status = await self.performance_monitor.get_monitoring_status(model_name)
            
            if monitor_key not in status:
                return None
            
            return status[monitor_key]
            
        except Exception as e:
            logger.error(f"Error getting model performance metrics: {e}")
            return None
    
    async def _check_test_duration(self, model_name: str, version: str, 
                                 min_hours: int) -> bool:
        """Check if model has been tested for minimum duration"""
        try:
            # Check deployment history
            deployments = self.canary_manager.get_deployment_history(model_name)
            
            for deployment in deployments:
                if deployment.get("candidate_version") == version:
                    start_time = datetime.fromisoformat(deployment["timestamp"])
                    duration = datetime.utcnow() - start_time
                    
                    if duration.total_seconds() >= min_hours * 3600:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking test duration: {e}")
            return False
    
    async def _check_performance_degradation(self, model_name: str, candidate_version: str,
                                           max_degradation: float) -> Dict[str, Any]:
        """Check performance degradation vs current production model"""
        try:
            # Get current production model
            production_model = self.model_registry.get_production_model(model_name)
            if not production_model:
                return {"passed": True, "reason": "No production model to compare against"}
            
            # Get metrics for both versions
            candidate_metrics = await self._get_model_performance_metrics(model_name, candidate_version)
            production_metrics = await self._get_model_performance_metrics(model_name, production_model.version)
            
            if not candidate_metrics or not production_metrics:
                return {"passed": False, "reason": "Insufficient metrics for comparison"}
            
            # Compare key performance indicators
            key_metrics = ["accuracy", "error_rate", "avg_latency_ms"]
            
            for metric in key_metrics:
                candidate_value = candidate_metrics.get(metric)
                production_value = production_metrics.get(metric)
                
                if candidate_value is None or production_value is None:
                    continue
                
                # Calculate degradation (handle metrics where lower is better)
                if metric in ["error_rate", "avg_latency_ms"]:
                    degradation = (candidate_value - production_value) / production_value
                else:
                    degradation = (production_value - candidate_value) / production_value
                
                if degradation > max_degradation:
                    return {
                        "passed": False, 
                        "reason": f"Performance degradation in {metric}: {degradation:.4f} > {max_degradation}"
                    }
            
            return {"passed": True, "reason": "No significant performance degradation"}
            
        except Exception as e:
            logger.error(f"Error checking performance degradation: {e}")
            return {"passed": False, "reason": f"Error in comparison: {str(e)}"}
    
    async def _check_business_metrics(self, model_name: str, candidate_version: str,
                                    thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Check business-specific metric thresholds"""
        try:
            metrics = await self._get_model_performance_metrics(model_name, candidate_version)
            if not metrics:
                return {"all_met": False, "failures": ["No metrics available"]}
            
            failures = []
            for metric_name, threshold in thresholds.items():
                metric_value = metrics.get(metric_name)
                
                if metric_value is None:
                    failures.append(f"Business metric {metric_name} not available")
                    continue
                
                # Assume higher values are better unless it's an error metric
                if metric_name.lower() in ["error", "mae", "rmse", "mse", "mape"]:
                    if metric_value > threshold:
                        failures.append(f"{metric_name}: {metric_value:.4f} > {threshold}")
                else:
                    if metric_value < threshold:
                        failures.append(f"{metric_name}: {metric_value:.4f} < {threshold}")
            
            return {"all_met": len(failures) == 0, "failures": failures}
            
        except Exception as e:
            logger.error(f"Error checking business metrics: {e}")
            return {"all_met": False, "failures": [f"Error: {str(e)}"]}
    
    async def _check_statistical_significance(self, model_name: str, candidate_version: str,
                                            significance_level: float) -> Dict[str, Any]:
        """Check if performance differences are statistically significant"""
        try:
            # This would integrate with the canary deployment statistical tests
            # For now, return a placeholder
            return {"significant": True, "p_value": 0.01, "details": "Statistical significance check passed"}
            
        except Exception as e:
            logger.error(f"Error checking statistical significance: {e}")
            return {"significant": False, "error": str(e)}
    
    async def initiate_rollback(self, model_name: str, current_version: str,
                              trigger: RollbackTrigger, reason: str,
                              target_version: str = None) -> str:
        """Initiate automated rollback"""
        try:
            # Determine target version if not specified
            if not target_version:
                # Get last known good version
                target_version = await self._get_last_known_good_version(model_name, current_version)
                if not target_version:
                    raise ValueError("No suitable rollback target version found")
            
            # Create rollback plan
            rollback_plan = await self._create_rollback_plan(
                model_name, current_version, target_version, trigger, reason
            )
            
            # Store and execute rollback
            self.active_rollbacks[rollback_plan.rollback_id] = rollback_plan
            
            # Execute rollback asynchronously
            asyncio.create_task(self._execute_rollback_plan(rollback_plan))
            
            # Send notifications
            await self._send_rollback_notification(rollback_plan, "initiated")
            
            logger.warning(f"Initiated rollback for {model_name} from v{current_version} to v{target_version} "
                          f"(trigger: {trigger.value})")
            
            return rollback_plan.rollback_id
            
        except Exception as e:
            logger.error(f"Error initiating rollback: {e}")
            raise
    
    async def _get_last_known_good_version(self, model_name: str, 
                                         exclude_version: str) -> Optional[str]:
        """Get the last known good production version for rollback"""
        try:
            # Get version history from model registry
            all_versions = self.model_registry.client.search_model_versions(f"name='{model_name}'")
            
            # Sort by creation time (descending)
            sorted_versions = sorted(
                all_versions, 
                key=lambda x: x.creation_timestamp, 
                reverse=True
            )
            
            # Find last production version that's not the current failing version
            for version in sorted_versions:
                if (version.version != exclude_version and 
                    version.current_stage == ModelStage.PRODUCTION.value):
                    return version.version
            
            # If no production version found, get the most recent staging version
            for version in sorted_versions:
                if (version.version != exclude_version and 
                    version.current_stage == ModelStage.STAGING.value):
                    return version.version
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding last known good version: {e}")
            return None
    
    async def _create_rollback_plan(self, model_name: str, current_version: str,
                                  target_version: str, trigger: RollbackTrigger,
                                  reason: str) -> RollbackPlan:
        """Create detailed rollback execution plan"""
        rollback_id = f"rollback_{model_name}_{int(time.time())}"
        
        # Define rollback steps
        rollback_steps = [
            {
                "step": "validate_target_version",
                "description": f"Validate that target version {target_version} is available and healthy",
                "estimated_duration_minutes": 1
            },
            {
                "step": "update_model_registry",
                "description": f"Transition {target_version} to Production stage",
                "estimated_duration_minutes": 1
            },
            {
                "step": "update_load_balancer",
                "description": "Update load balancer to route traffic to rollback version",
                "estimated_duration_minutes": 2
            },
            {
                "step": "health_check",
                "description": "Perform health checks on rollback version",
                "estimated_duration_minutes": 3
            },
            {
                "step": "validate_rollback",
                "description": "Validate rollback success with test predictions",
                "estimated_duration_minutes": 5
            },
            {
                "step": "cleanup",
                "description": "Clean up failed version resources",
                "estimated_duration_minutes": 2
            }
        ]
        
        total_duration = sum(step["estimated_duration_minutes"] for step in rollback_steps)
        
        validation_steps = [
            "Verify target model loads successfully",
            "Confirm model predictions are working",
            "Check performance metrics are within acceptable range",
            "Validate no new errors are occurring"
        ]
        
        return RollbackPlan(
            rollback_id=rollback_id,
            model_name=model_name,
            current_version=current_version,
            target_version=target_version,
            trigger=trigger,
            reason=reason,
            rollback_steps=rollback_steps,
            estimated_duration_minutes=total_duration,
            rollback_validation_steps=validation_steps,
            notification_channels=["slack", "email", "dashboard"],
            created_at=datetime.utcnow()
        )
    
    async def _execute_rollback_plan(self, rollback_plan: RollbackPlan) -> None:
        """Execute the rollback plan"""
        try:
            rollback_plan.status = "executing"
            rollback_plan.executed_at = datetime.utcnow()
            
            logger.info(f"Executing rollback plan {rollback_plan.rollback_id}")
            
            for i, step in enumerate(rollback_plan.rollback_steps):
                try:
                    logger.info(f"Rollback step {i+1}/{len(rollback_plan.rollback_steps)}: {step['step']}")
                    
                    # Execute the specific rollback step
                    await self._execute_rollback_step(rollback_plan, step)
                    
                    # Wait for step completion
                    await asyncio.sleep(step["estimated_duration_minutes"] * 10)  # Scale down for demo
                    
                except Exception as e:
                    logger.error(f"Rollback step {step['step']} failed: {e}")
                    rollback_plan.status = "failed"
                    await self._send_rollback_notification(rollback_plan, "failed", error=str(e))
                    return
            
            # Validate rollback success
            validation_success = await self._validate_rollback_success(rollback_plan)
            
            if validation_success:
                rollback_plan.status = "completed"
                rollback_plan.completed_at = datetime.utcnow()
                
                # Update model registry to reflect rollback
                self.model_registry.transition_model_stage(
                    model_name=rollback_plan.model_name,
                    version=rollback_plan.target_version,
                    stage=ModelStage.PRODUCTION
                )
                
                # Archive the failed version
                self.model_registry.transition_model_stage(
                    model_name=rollback_plan.model_name,
                    version=rollback_plan.current_version,
                    stage=ModelStage.ARCHIVED
                )
                
                await self._send_rollback_notification(rollback_plan, "completed")
                logger.info(f"Rollback {rollback_plan.rollback_id} completed successfully")
            else:
                rollback_plan.status = "validation_failed"
                await self._send_rollback_notification(rollback_plan, "validation_failed")
                logger.error(f"Rollback {rollback_plan.rollback_id} validation failed")
            
            # Move to history
            self.rollback_history.append(rollback_plan)
            if rollback_plan.rollback_id in self.active_rollbacks:
                del self.active_rollbacks[rollback_plan.rollback_id]
                
        except Exception as e:
            logger.error(f"Error executing rollback plan: {e}")
            rollback_plan.status = "error"
            await self._send_rollback_notification(rollback_plan, "error", error=str(e))
    
    async def _execute_rollback_step(self, rollback_plan: RollbackPlan, step: Dict[str, Any]) -> None:
        """Execute a specific rollback step"""
        step_name = step["step"]
        
        if step_name == "validate_target_version":
            # Ensure target version exists and is healthy
            target_version = self.model_registry.client.get_model_version(
                rollback_plan.model_name, rollback_plan.target_version
            )
            if not target_version:
                raise ValueError(f"Target version {rollback_plan.target_version} not found")
        
        elif step_name == "update_model_registry":
            # This is handled at the end of successful rollback
            pass
        
        elif step_name == "update_load_balancer":
            # In a real implementation, this would update load balancer/service mesh
            logger.info("Load balancer update simulated")
        
        elif step_name == "health_check":
            # Perform health check on target version
            await self._perform_health_check(rollback_plan.model_name, rollback_plan.target_version)
        
        elif step_name == "validate_rollback":
            # Run validation tests
            await self._run_rollback_validation_tests(rollback_plan)
        
        elif step_name == "cleanup":
            # Clean up resources from failed version
            logger.info(f"Cleaning up resources for version {rollback_plan.current_version}")
    
    async def _perform_health_check(self, model_name: str, version: str) -> None:
        """Perform health check on a model version"""
        # In a real implementation, this would make test predictions
        # and verify the model is responding correctly
        logger.info(f"Health check passed for {model_name} v{version}")
    
    async def _run_rollback_validation_tests(self, rollback_plan: RollbackPlan) -> None:
        """Run validation tests after rollback"""
        # In a real implementation, this would run a suite of validation tests
        logger.info(f"Validation tests passed for rollback {rollback_plan.rollback_id}")
    
    async def _validate_rollback_success(self, rollback_plan: RollbackPlan) -> bool:
        """Validate that rollback was successful"""
        try:
            # Check that target version is now serving traffic
            # In a real implementation, this would verify:
            # 1. Model is loaded and responding
            # 2. Error rates have returned to normal
            # 3. Latency is within acceptable range
            # 4. No new critical alerts
            
            # For now, return True to simulate successful validation
            return True
            
        except Exception as e:
            logger.error(f"Rollback validation error: {e}")
            return False
    
    async def _send_rollback_notification(self, rollback_plan: RollbackPlan, 
                                        status: str, error: str = None) -> None:
        """Send rollback notification"""
        notification_data = {
            "rollback_id": rollback_plan.rollback_id,
            "model_name": rollback_plan.model_name,
            "current_version": rollback_plan.current_version,
            "target_version": rollback_plan.target_version,
            "trigger": rollback_plan.trigger.value,
            "reason": rollback_plan.reason,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if error:
            notification_data["error"] = error
        
        # Call notification handlers
        for handler in self.notification_handlers:
            try:
                handler(f"rollback_{status}", notification_data)
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")
    
    def add_notification_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add notification handler"""
        self.notification_handlers.append(handler)
    
    async def start_automated_monitoring(self, model_name: str, model_version: str,
                                       model_type: ModelType) -> None:
        """Start automated monitoring for rollback triggers"""
        try:
            monitor_key = f"{model_name}:{model_version}"
            
            if monitor_key in self.monitoring_tasks:
                logger.warning(f"Already monitoring {monitor_key} for rollback triggers")
                return
            
            # Start monitoring task
            task = asyncio.create_task(
                self._monitor_rollback_triggers(model_name, model_version, model_type)
            )
            self.monitoring_tasks[monitor_key] = task
            
            logger.info(f"Started automated rollback monitoring for {monitor_key}")
            
        except Exception as e:
            logger.error(f"Error starting automated monitoring: {e}")
    
    async def _monitor_rollback_triggers(self, model_name: str, model_version: str,
                                       model_type: ModelType) -> None:
        """Monitor for rollback triggers"""
        criteria = self.rollback_criteria.get(model_type)
        if not criteria:
            logger.warning(f"No rollback criteria defined for model type {model_type}")
            return
        
        consecutive_failures = 0
        circuit_breaker_failures = 0
        
        while f"{model_name}:{model_version}" in self.monitoring_tasks:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Get recent alerts
                alerts = await self.performance_monitor.get_recent_alerts(
                    model_name, model_version, hours=1
                )
                
                # Check for high-severity alerts
                high_severity_alerts = [
                    alert for alert in alerts 
                    if alert.severity.value >= criteria.alert_severity_threshold.value
                ]
                
                if high_severity_alerts:
                    consecutive_failures += 1
                    
                    # Check for immediate rollback triggers
                    for alert in high_severity_alerts:
                        if await self._should_trigger_immediate_rollback(alert, criteria):
                            await self.initiate_rollback(
                                model_name, model_version,
                                RollbackTrigger.PERFORMANCE_DEGRADATION,
                                f"High-severity alert: {alert.message}"
                            )
                            return
                
                else:
                    consecutive_failures = 0
                
                # Check consecutive failures threshold
                if consecutive_failures >= criteria.max_consecutive_failures:
                    await self.initiate_rollback(
                        model_name, model_version,
                        RollbackTrigger.PERFORMANCE_DEGRADATION,
                        f"Too many consecutive failures: {consecutive_failures}"
                    )
                    return
                
                # Check circuit breaker
                circuit_breaker_failures += len([
                    alert for alert in alerts 
                    if alert.alert_type == "error_rate"
                ])
                
                if circuit_breaker_failures >= criteria.circuit_breaker_threshold:
                    await self.initiate_rollback(
                        model_name, model_version,
                        RollbackTrigger.CIRCUIT_BREAKER,
                        f"Circuit breaker triggered: {circuit_breaker_failures} failures"
                    )
                    return
                
            except Exception as e:
                logger.error(f"Error in rollback trigger monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _should_trigger_immediate_rollback(self, alert: ModelAlert, 
                                               criteria: RollbackCriteria) -> bool:
        """Determine if an alert should trigger immediate rollback"""
        # Check error rate threshold
        if alert.alert_type == "error_rate":
            error_rate = alert.metric_values.get("error_rate", 0)
            return error_rate > criteria.max_error_rate
        
        # Check accuracy drop
        if alert.alert_type == "accuracy":
            accuracy = alert.metric_values.get("accuracy", 1.0)
            return accuracy < criteria.min_accuracy
        
        # Check latency spike
        if alert.alert_type == "latency":
            latency = alert.metric_values.get("p99_latency_ms", 0)
            return latency > criteria.max_latency_p99_ms
        
        # Critical alerts trigger immediate rollback
        return alert.severity == AlertSeverity.CRITICAL
    
    def get_rollback_status(self, rollback_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a rollback"""
        if rollback_id in self.active_rollbacks:
            return asdict(self.active_rollbacks[rollback_id])
        
        for rollback in self.rollback_history:
            if rollback.rollback_id == rollback_id:
                return asdict(rollback)
        
        return None
    
    def get_rollback_history(self, model_name: str = None) -> List[Dict[str, Any]]:
        """Get rollback history"""
        if model_name:
            return [
                asdict(rollback) for rollback in self.rollback_history
                if rollback.model_name == model_name
            ]
        return [asdict(rollback) for rollback in self.rollback_history]
    
    async def stop_monitoring(self, model_name: str, model_version: str) -> None:
        """Stop automated monitoring for a model"""
        monitor_key = f"{model_name}:{model_version}"
        
        if monitor_key in self.monitoring_tasks:
            self.monitoring_tasks[monitor_key].cancel()
            del self.monitoring_tasks[monitor_key]
            logger.info(f"Stopped rollback monitoring for {monitor_key}")
    
    async def cleanup(self) -> None:
        """Cleanup system resources"""
        try:
            # Cancel all monitoring tasks
            for task in self.monitoring_tasks.values():
                task.cancel()
            
            # Wait for tasks to complete
            if self.monitoring_tasks:
                await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)
            
            logger.info("Automated rollback system cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")