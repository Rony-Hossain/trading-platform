"""
Automated Retraining Workflow Orchestrator
Manages end-to-end automated model retraining workflows
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import aioredis
import aiohttp
import mlflow
from mlflow.tracking import MlflowClient

from ..model_registry.mlflow_registry import MLflowModelRegistry, ModelStage, ModelType
from ..deployment.canary_deployment import CanaryDeploymentManager, CanaryConfig, DeploymentStrategy
from ..monitoring.drift_monitoring_service import DriftMonitoringService, RetrainingTrigger

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Retraining workflow statuses"""
    PENDING = "pending"
    RUNNING = "running"
    DATA_COLLECTION = "data_collection"
    TRAINING = "training"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class WorkflowPriority(Enum):
    """Workflow execution priorities"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    SCHEDULED = 5

@dataclass
class RetrainingWorkflow:
    """Retraining workflow definition"""
    workflow_id: str
    model_name: str
    current_version: str
    trigger_reason: str
    priority: WorkflowPriority
    status: WorkflowStatus
    created_timestamp: datetime
    started_timestamp: Optional[datetime] = None
    completed_timestamp: Optional[datetime] = None
    
    # Configuration
    data_config: Dict[str, Any] = None
    training_config: Dict[str, Any] = None
    validation_config: Dict[str, Any] = None
    deployment_config: Dict[str, Any] = None
    
    # Progress tracking
    progress_percentage: float = 0.0
    current_step: str = ""
    steps_completed: List[str] = None
    error_message: str = ""
    
    # Results
    new_model_version: str = ""
    performance_metrics: Dict[str, float] = None
    validation_results: Dict[str, Any] = None

@dataclass
class WorkflowStep:
    """Individual workflow step definition"""
    step_name: str
    step_function: Callable
    timeout_seconds: int = 3600  # 1 hour default
    retry_count: int = 3
    required_inputs: List[str] = None
    outputs: List[str] = None

class RetrainingOrchestrator:
    """
    Automated retraining workflow orchestrator
    
    Features:
    - Priority-based workflow queue management
    - End-to-end retraining pipeline automation
    - Data collection and validation
    - Model training with hyperparameter optimization
    - Automated model validation and testing
    - Canary deployment integration
    - Progress tracking and monitoring
    - Error handling and recovery
    """
    
    def __init__(self,
                 model_registry: MLflowModelRegistry,
                 canary_manager: CanaryDeploymentManager,
                 drift_monitor: DriftMonitoringService,
                 redis_url: str = "redis://localhost:6379"):
        
        self.model_registry = model_registry
        self.canary_manager = canary_manager
        self.drift_monitor = drift_monitor
        self.redis_url = redis_url
        self.redis_client = None
        
        # Workflow queue and tracking
        self.active_workflows: Dict[str, RetrainingWorkflow] = {}
        self.workflow_queue: List[str] = []  # Workflow IDs sorted by priority
        
        # Configuration
        self.max_concurrent_workflows = 2
        self.workflow_timeout = timedelta(hours=6)
        
        # Define workflow steps
        self.workflow_steps = [
            WorkflowStep("data_collection", self._collect_training_data, 1800),
            WorkflowStep("data_validation", self._validate_training_data, 600),
            WorkflowStep("feature_engineering", self._engineer_features, 1800),
            WorkflowStep("model_training", self._train_model, 3600),
            WorkflowStep("model_validation", self._validate_model, 1200),
            WorkflowStep("model_registration", self._register_new_model, 300),
            WorkflowStep("canary_deployment", self._deploy_canary, 1800),
            WorkflowStep("performance_validation", self._validate_canary_performance, 3600),
            WorkflowStep("model_promotion", self._promote_model, 300),
            WorkflowStep("cleanup", self._cleanup_workflow, 300)
        ]
        
    async def initialize(self):
        """Initialize orchestrator"""
        self.redis_client = await aioredis.from_url(self.redis_url)
        await self._load_active_workflows()
        logger.info("Retraining orchestrator initialized")
    
    async def close(self):
        """Close orchestrator"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def submit_retraining_workflow(self, trigger: RetrainingTrigger) -> str:
        """
        Submit a new retraining workflow based on trigger
        """
        try:
            workflow_id = str(uuid.uuid4())
            
            # Create workflow configuration
            workflow = RetrainingWorkflow(
                workflow_id=workflow_id,
                model_name=trigger.model_name,
                current_version=trigger.model_version,
                trigger_reason=trigger.trigger_type,
                priority=WorkflowPriority(trigger.priority),
                status=WorkflowStatus.PENDING,
                created_timestamp=datetime.utcnow(),
                data_config=self._get_data_config(trigger),
                training_config=self._get_training_config(trigger),
                validation_config=self._get_validation_config(trigger),
                deployment_config=self._get_deployment_config(trigger),
                steps_completed=[],
                performance_metrics={},
                validation_results={}
            )
            
            # Add to workflow tracking
            self.active_workflows[workflow_id] = workflow
            await self._persist_workflow(workflow)
            
            # Add to priority queue
            await self._add_to_queue(workflow_id)
            
            logger.info(f"Retraining workflow {workflow_id} submitted for {trigger.model_name}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Error submitting retraining workflow: {str(e)}")
            raise
    
    async def start_workflow_processor(self):
        """
        Start the workflow processor (runs continuously)
        """
        logger.info("Starting retraining workflow processor")
        
        while True:
            try:
                # Process workflows from queue
                await self._process_workflow_queue()
                
                # Check for timed out workflows
                await self._check_workflow_timeouts()
                
                # Clean up completed workflows
                await self._cleanup_completed_workflows()
                
                # Wait before next iteration
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in workflow processor: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[RetrainingWorkflow]:
        """Get workflow status"""
        return self.active_workflows.get(workflow_id)
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow"""
        try:
            if workflow_id in self.active_workflows:
                workflow = self.active_workflows[workflow_id]
                workflow.status = WorkflowStatus.CANCELLED
                workflow.completed_timestamp = datetime.utcnow()
                
                await self._persist_workflow(workflow)
                logger.info(f"Workflow {workflow_id} cancelled")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cancelling workflow {workflow_id}: {str(e)}")
            return False
    
    # Private methods for workflow execution
    async def _process_workflow_queue(self):
        """Process workflows from the priority queue"""
        # Count running workflows
        running_count = sum(1 for w in self.active_workflows.values() 
                           if w.status == WorkflowStatus.RUNNING)
        
        if running_count >= self.max_concurrent_workflows:
            return
        
        # Get next workflow from queue
        if not self.workflow_queue:
            return
        
        workflow_id = self.workflow_queue.pop(0)
        workflow = self.active_workflows.get(workflow_id)
        
        if not workflow or workflow.status != WorkflowStatus.PENDING:
            return
        
        # Start workflow execution
        asyncio.create_task(self._execute_workflow(workflow))
    
    async def _execute_workflow(self, workflow: RetrainingWorkflow):
        """Execute a complete retraining workflow"""
        try:
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_timestamp = datetime.utcnow()
            workflow.current_step = "Starting workflow"
            await self._persist_workflow(workflow)
            
            logger.info(f"Starting workflow {workflow.workflow_id} for {workflow.model_name}")
            
            # Execute each workflow step
            for i, step in enumerate(self.workflow_steps):
                try:
                    workflow.current_step = step.step_name
                    workflow.progress_percentage = (i / len(self.workflow_steps)) * 100
                    await self._persist_workflow(workflow)
                    
                    logger.info(f"Executing step: {step.step_name} for workflow {workflow.workflow_id}")
                    
                    # Execute step with timeout and retry
                    success = await self._execute_step_with_retry(workflow, step)
                    
                    if not success:
                        raise Exception(f"Step {step.step_name} failed after retries")
                    
                    workflow.steps_completed.append(step.step_name)
                    await self._persist_workflow(workflow)
                    
                except Exception as e:
                    workflow.status = WorkflowStatus.FAILED
                    workflow.error_message = f"Failed at step {step.step_name}: {str(e)}"
                    workflow.completed_timestamp = datetime.utcnow()
                    await self._persist_workflow(workflow)
                    
                    logger.error(f"Workflow {workflow.workflow_id} failed: {workflow.error_message}")
                    return
            
            # Workflow completed successfully
            workflow.status = WorkflowStatus.COMPLETED
            workflow.progress_percentage = 100.0
            workflow.current_step = "Completed"
            workflow.completed_timestamp = datetime.utcnow()
            await self._persist_workflow(workflow)
            
            logger.info(f"Workflow {workflow.workflow_id} completed successfully")
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.error_message = str(e)
            workflow.completed_timestamp = datetime.utcnow()
            await self._persist_workflow(workflow)
            
            logger.error(f"Workflow {workflow.workflow_id} failed with error: {str(e)}")
    
    async def _execute_step_with_retry(self, workflow: RetrainingWorkflow, step: WorkflowStep) -> bool:
        """Execute a workflow step with retry logic"""
        for attempt in range(step.retry_count):
            try:
                # Execute step with timeout
                result = await asyncio.wait_for(
                    step.step_function(workflow), 
                    timeout=step.timeout_seconds
                )
                
                if result:
                    return True
                else:
                    logger.warning(f"Step {step.step_name} returned False, attempt {attempt + 1}")
                    
            except asyncio.TimeoutError:
                logger.error(f"Step {step.step_name} timed out, attempt {attempt + 1}")
                
            except Exception as e:
                logger.error(f"Step {step.step_name} failed with error: {str(e)}, attempt {attempt + 1}")
            
            # Wait before retry
            if attempt < step.retry_count - 1:
                await asyncio.sleep(60)  # Wait 1 minute before retry
        
        return False
    
    # Workflow step implementations
    async def _collect_training_data(self, workflow: RetrainingWorkflow) -> bool:
        """Collect training data for retraining"""
        try:
            logger.info(f"Collecting training data for {workflow.model_name}")
            
            # This would implement data collection logic
            # For now, simulate data collection
            await asyncio.sleep(5)  # Simulate data collection time
            
            # Store data collection results
            workflow.validation_results["data_collection"] = {
                "samples_collected": 10000,
                "date_range": "2024-01-01 to 2024-09-29",
                "data_quality_score": 0.95
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Data collection failed: {str(e)}")
            return False
    
    async def _validate_training_data(self, workflow: RetrainingWorkflow) -> bool:
        """Validate training data quality"""
        try:
            logger.info(f"Validating training data for {workflow.model_name}")
            
            # Simulate data validation
            await asyncio.sleep(2)
            
            workflow.validation_results["data_validation"] = {
                "validation_passed": True,
                "quality_checks": ["completeness", "consistency", "validity"],
                "quality_score": 0.93
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return False
    
    async def _engineer_features(self, workflow: RetrainingWorkflow) -> bool:
        """Perform feature engineering"""
        try:
            logger.info(f"Engineering features for {workflow.model_name}")
            
            # Simulate feature engineering
            await asyncio.sleep(5)
            
            workflow.validation_results["feature_engineering"] = {
                "features_created": 25,
                "features_selected": 20,
                "feature_importance_available": True
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            return False
    
    async def _train_model(self, workflow: RetrainingWorkflow) -> bool:
        """Train the new model"""
        try:
            logger.info(f"Training model for {workflow.model_name}")
            
            # Simulate model training
            await asyncio.sleep(10)
            
            # Generate new model version
            current_version = workflow.current_version
            version_parts = current_version.split('.')
            new_patch = int(version_parts[2]) + 1 if len(version_parts) > 2 else 1
            workflow.new_model_version = f"{version_parts[0]}.{version_parts[1]}.{new_patch}"
            
            workflow.performance_metrics = {
                "mae": 0.025,
                "rmse": 0.045,
                "r2_score": 0.78,
                "training_time": 600
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return False
    
    async def _validate_model(self, workflow: RetrainingWorkflow) -> bool:
        """Validate the trained model"""
        try:
            logger.info(f"Validating model for {workflow.model_name}")
            
            # Simulate model validation
            await asyncio.sleep(3)
            
            workflow.validation_results["model_validation"] = {
                "validation_passed": True,
                "backtesting_results": {
                    "sharpe_ratio": 1.45,
                    "max_drawdown": -0.12,
                    "hit_rate": 0.62
                },
                "performance_vs_baseline": {
                    "improvement": 0.08,
                    "significance": "p < 0.01"
                }
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            return False
    
    async def _register_new_model(self, workflow: RetrainingWorkflow) -> bool:
        """Register the new model in MLflow"""
        try:
            logger.info(f"Registering new model version for {workflow.model_name}")
            
            # This would register model in MLflow
            # For now, simulate registration
            await asyncio.sleep(2)
            
            return True
            
        except Exception as e:
            logger.error(f"Model registration failed: {str(e)}")
            return False
    
    async def _deploy_canary(self, workflow: RetrainingWorkflow) -> bool:
        """Deploy model as canary"""
        try:
            logger.info(f"Deploying canary for {workflow.model_name} version {workflow.new_model_version}")
            
            # Simulate canary deployment
            await asyncio.sleep(5)
            
            workflow.validation_results["canary_deployment"] = {
                "deployment_successful": True,
                "traffic_percentage": 5.0,
                "deployment_timestamp": datetime.utcnow().isoformat()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {str(e)}")
            return False
    
    async def _validate_canary_performance(self, workflow: RetrainingWorkflow) -> bool:
        """Validate canary performance"""
        try:
            logger.info(f"Validating canary performance for {workflow.model_name}")
            
            # Simulate performance validation
            await asyncio.sleep(15)  # Longer validation period
            
            workflow.validation_results["canary_validation"] = {
                "validation_passed": True,
                "performance_metrics": {
                    "latency_p95": 45,
                    "error_rate": 0.001,
                    "accuracy": 0.89
                },
                "statistical_significance": "p < 0.05"
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Canary validation failed: {str(e)}")
            return False
    
    async def _promote_model(self, workflow: RetrainingWorkflow) -> bool:
        """Promote canary to production"""
        try:
            logger.info(f"Promoting model {workflow.model_name} version {workflow.new_model_version}")
            
            # Simulate model promotion
            await asyncio.sleep(3)
            
            return True
            
        except Exception as e:
            logger.error(f"Model promotion failed: {str(e)}")
            return False
    
    async def _cleanup_workflow(self, workflow: RetrainingWorkflow) -> bool:
        """Cleanup workflow resources"""
        try:
            logger.info(f"Cleaning up workflow {workflow.workflow_id}")
            
            # Simulate cleanup
            await asyncio.sleep(1)
            
            return True
            
        except Exception as e:
            logger.error(f"Workflow cleanup failed: {str(e)}")
            return False
    
    # Helper methods
    async def _add_to_queue(self, workflow_id: str):
        """Add workflow to priority queue"""
        workflow = self.active_workflows[workflow_id]
        
        # Insert based on priority
        inserted = False
        for i, existing_id in enumerate(self.workflow_queue):
            existing_workflow = self.active_workflows.get(existing_id)
            if existing_workflow and workflow.priority.value < existing_workflow.priority.value:
                self.workflow_queue.insert(i, workflow_id)
                inserted = True
                break
        
        if not inserted:
            self.workflow_queue.append(workflow_id)
    
    async def _persist_workflow(self, workflow: RetrainingWorkflow):
        """Persist workflow state to Redis"""
        await self.redis_client.setex(
            f"workflow:{workflow.workflow_id}",
            86400,  # 24 hours
            json.dumps(asdict(workflow), default=str)
        )
    
    async def _load_active_workflows(self):
        """Load active workflows from Redis"""
        try:
            keys = await self.redis_client.keys("workflow:*")
            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    workflow_data = json.loads(data)
                    workflow_id = workflow_data["workflow_id"]
                    
                    # Recreate workflow object
                    workflow = RetrainingWorkflow(**workflow_data)
                    self.active_workflows[workflow_id] = workflow
                    
                    # Add to queue if pending
                    if workflow.status == WorkflowStatus.PENDING:
                        await self._add_to_queue(workflow_id)
                        
        except Exception as e:
            logger.error(f"Error loading active workflows: {str(e)}")
    
    async def _check_workflow_timeouts(self):
        """Check for and handle workflow timeouts"""
        current_time = datetime.utcnow()
        
        for workflow in list(self.active_workflows.values()):
            if (workflow.status == WorkflowStatus.RUNNING and 
                workflow.started_timestamp and
                current_time - workflow.started_timestamp > self.workflow_timeout):
                
                workflow.status = WorkflowStatus.FAILED
                workflow.error_message = "Workflow timed out"
                workflow.completed_timestamp = current_time
                await self._persist_workflow(workflow)
                
                logger.warning(f"Workflow {workflow.workflow_id} timed out")
    
    async def _cleanup_completed_workflows(self):
        """Clean up old completed workflows"""
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        
        workflows_to_remove = []
        for workflow_id, workflow in self.active_workflows.items():
            if (workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED] and
                workflow.completed_timestamp and
                workflow.completed_timestamp < cutoff_time):
                
                workflows_to_remove.append(workflow_id)
        
        for workflow_id in workflows_to_remove:
            del self.active_workflows[workflow_id]
            await self.redis_client.delete(f"workflow:{workflow_id}")
    
    def _get_data_config(self, trigger: RetrainingTrigger) -> Dict[str, Any]:
        """Get data configuration for retraining"""
        return {
            "lookback_days": 90,
            "min_samples": 1000,
            "data_sources": ["market_data", "sentiment", "fundamentals"],
            "validation_split": 0.2
        }
    
    def _get_training_config(self, trigger: RetrainingTrigger) -> Dict[str, Any]:
        """Get training configuration"""
        return {
            "model_type": "lgb_regressor",
            "hyperparameter_tuning": True,
            "cross_validation_folds": 5,
            "early_stopping": True,
            "max_training_time": 3600
        }
    
    def _get_validation_config(self, trigger: RetrainingTrigger) -> Dict[str, Any]:
        """Get validation configuration"""
        return {
            "backtesting_period": 30,
            "performance_thresholds": {
                "min_sharpe_ratio": 0.8,
                "max_drawdown": -0.15
            },
            "statistical_significance": 0.05
        }
    
    def _get_deployment_config(self, trigger: RetrainingTrigger) -> Dict[str, Any]:
        """Get deployment configuration"""
        return {
            "canary_traffic_percentage": 5.0,
            "validation_duration_hours": 24,
            "auto_promote": True,
            "rollback_on_failure": True
        }