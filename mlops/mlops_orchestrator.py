"""
MLOps Orchestrator for Trading Platform
Coordinates model registry, deployment, monitoring, and rollback systems
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pathlib import Path

from .model_registry.mlflow_registry import MLflowModelRegistry, ModelStage, ModelType
from .deployment.canary_deployment import CanaryDeploymentManager, CanaryConfig, DeploymentStrategy
from .monitoring.model_performance_monitor import ModelPerformanceMonitor, PerformanceThreshold
from .deployment.rollback_system import AutomatedRollbackSystem, PromotionCriteria, RollbackCriteria

logger = logging.getLogger(__name__)

# Pydantic models for API
class ModelRegistrationRequest(BaseModel):
    model_name: str
    model_type: str
    description: str = None
    model_artifact_path: str
    model_signature: Dict[str, Any] = None
    input_example: Dict[str, Any] = None
    metrics: Dict[str, float]
    params: Dict[str, Any]
    tags: Dict[str, str] = None

class CanaryDeploymentRequest(BaseModel):
    model_name: str
    candidate_version: str
    production_version: str = None
    traffic_percentage: float = 5.0
    min_samples: int = 1000
    max_duration_hours: int = 24
    auto_promote_on_success: bool = False
    rollback_on_failure: bool = True

class ModelPromotionRequest(BaseModel):
    model_name: str
    version: str
    target_stage: str
    approval_required: bool = False

class MonitoringConfigRequest(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    custom_thresholds: List[Dict[str, Any]] = None

class MLOpsOrchestrator:
    """
    Central orchestrator for MLOps pipeline
    
    Features:
    - Model lifecycle management
    - Automated deployment pipeline
    - Performance monitoring coordination
    - Rollback automation
    - API endpoints for MLOps operations
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_default_config()
        
        # Initialize components
        self.model_registry = MLflowModelRegistry(
            tracking_uri=self.config.get("mlflow_tracking_uri"),
            registry_uri=self.config.get("mlflow_registry_uri")
        )
        
        self.canary_manager = CanaryDeploymentManager(self.model_registry)
        self.performance_monitor = ModelPerformanceMonitor(
            self.model_registry,
            redis_url=self.config.get("redis_url", "redis://localhost:6379")
        )
        self.rollback_system = AutomatedRollbackSystem(
            self.model_registry, 
            self.performance_monitor, 
            self.canary_manager
        )
        
        # FastAPI app for MLOps API
        self.app = self._create_app()
        
        # System state
        self.active_deployments: Dict[str, Dict[str, Any]] = {}
        self.system_health = {
            "status": "initializing",
            "components": {
                "model_registry": "unknown",
                "canary_manager": "unknown",
                "performance_monitor": "unknown",
                "rollback_system": "unknown"
            },
            "last_updated": datetime.utcnow()
        }
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            "mlflow_registry_uri": os.getenv("MLFLOW_REGISTRY_URI", "http://localhost:5000"),
            "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
            "api_host": "0.0.0.0",
            "api_port": 8090,
            "log_level": "INFO",
            "enable_cors": True,
            "health_check_interval_seconds": 300
        }
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application with MLOps endpoints"""
        app = FastAPI(
            title="Trading Platform MLOps API",
            description="MLOps orchestration API for model deployment and monitoring",
            version="1.0.0"
        )
        
        if self.config.get("enable_cors", True):
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Health check endpoint
        @app.get("/health")
        async def health_check():
            return self.system_health
        
        # Model registry endpoints
        @app.post("/models/register")
        async def register_model(request: ModelRegistrationRequest):
            try:
                model_type = ModelType(request.model_type.lower())
                
                # Register model
                registered_model = self.model_registry.register_model(
                    request.model_name, model_type, request.description
                )
                
                # Create experiment if doesn't exist
                experiment_id = self.model_registry.create_experiment(
                    f"{request.model_name}_experiment", model_type
                )
                
                # Log model run (simplified - in practice would load actual model)
                run_id = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model_registry.log_model_run(
                        experiment_id=experiment_id,
                        model_name=request.model_name,
                        model_type=model_type,
                        model=None,  # Would be actual model object
                        model_signature=request.model_signature,
                        input_example=request.input_example,
                        metrics=request.metrics,
                        params=request.params,
                        tags=request.tags
                    )
                )
                
                # Create model version
                model_version = self.model_registry.create_model_version(
                    request.model_name, run_id, ModelStage.STAGING
                )
                
                return {
                    "status": "success",
                    "model_name": request.model_name,
                    "model_version": model_version.version,
                    "run_id": run_id,
                    "stage": model_version.current_stage
                }
                
            except Exception as e:
                logger.error(f"Error registering model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/models/{model_name}/versions")
        async def get_model_versions(model_name: str, stages: str = None):
            try:
                if stages:
                    stage_list = [ModelStage(s.strip()) for s in stages.split(",")]
                    versions = self.model_registry.get_model_versions(model_name, stage_list)
                else:
                    versions = self.model_registry.get_model_versions(model_name)
                
                return {
                    "model_name": model_name,
                    "versions": [
                        {
                            "version": v.version,
                            "stage": v.current_stage,
                            "creation_timestamp": v.creation_timestamp,
                            "run_id": v.run_id
                        }
                        for v in versions
                    ]
                }
                
            except Exception as e:
                logger.error(f"Error getting model versions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/models/{model_name}/promote")
        async def promote_model(model_name: str, request: ModelPromotionRequest):
            try:
                target_stage = ModelStage(request.target_stage.upper())
                
                # Evaluate promotion criteria if promoting to production
                if target_stage == ModelStage.PRODUCTION:
                    # Get model type from registry
                    model_version = self.model_registry.client.get_model_version(
                        model_name, request.version
                    )
                    run = self.model_registry.client.get_run(model_version.run_id)
                    model_type_str = run.data.tags.get("model_type", "sentiment_classifier")
                    model_type = ModelType(model_type_str)
                    
                    # Evaluate promotion criteria
                    should_promote, evaluation = await self.rollback_system.evaluate_promotion_criteria(
                        model_name, request.version, model_type
                    )
                    
                    if not should_promote and not request.approval_required:
                        return {
                            "status": "promotion_criteria_not_met",
                            "evaluation": evaluation
                        }
                
                # Perform promotion
                promoted_version = self.model_registry.transition_model_stage(
                    model_name, request.version, target_stage
                )
                
                # Start monitoring if promoted to production
                if target_stage == ModelStage.PRODUCTION:
                    await self._start_production_monitoring(model_name, request.version, model_type)
                
                return {
                    "status": "success",
                    "model_name": model_name,
                    "version": request.version,
                    "new_stage": promoted_version.current_stage
                }
                
            except Exception as e:
                logger.error(f"Error promoting model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Canary deployment endpoints
        @app.post("/deployments/canary")
        async def start_canary_deployment(request: CanaryDeploymentRequest):
            try:
                # Get production version if not specified
                production_version = request.production_version
                if not production_version:
                    prod_model = self.model_registry.get_production_model(request.model_name)
                    if prod_model:
                        production_version = prod_model.version
                    else:
                        raise ValueError("No production version found and none specified")
                
                # Create canary configuration
                config = CanaryConfig(
                    model_name=request.model_name,
                    candidate_version=request.candidate_version,
                    production_version=production_version,
                    traffic_percentage=request.traffic_percentage,
                    min_samples=request.min_samples,
                    max_duration_hours=request.max_duration_hours,
                    auto_promote_on_success=request.auto_promote_on_success,
                    rollback_on_failure=request.rollback_on_failure
                )
                
                # Start canary deployment
                deployment_id = await self.canary_manager.start_canary_deployment(config)
                
                # Track deployment
                self.active_deployments[deployment_id] = {
                    "deployment_id": deployment_id,
                    "model_name": request.model_name,
                    "candidate_version": request.candidate_version,
                    "production_version": production_version,
                    "status": "active",
                    "started_at": datetime.utcnow().isoformat()
                }
                
                return {
                    "status": "success",
                    "deployment_id": deployment_id,
                    "config": asdict(config)
                }
                
            except Exception as e:
                logger.error(f"Error starting canary deployment: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/deployments")
        async def get_all_deployments():
            try:
                # Get active deployments from canary manager
                canary_deployments = self.canary_manager.get_all_deployments()
                
                # Get rollback information
                rollback_history = self.rollback_system.get_rollback_history()
                
                return {
                    "active_deployments": canary_deployments,
                    "recent_rollbacks": rollback_history[-10:],  # Last 10 rollbacks
                    "total_active": len(canary_deployments)
                }
                
            except Exception as e:
                logger.error(f"Error getting deployments: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/deployments/{deployment_id}")
        async def get_deployment_status(deployment_id: str):
            try:
                status = self.canary_manager.get_deployment_status(deployment_id)
                if not status:
                    raise HTTPException(status_code=404, detail="Deployment not found")
                
                return status
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting deployment status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Monitoring endpoints
        @app.post("/monitoring/start")
        async def start_monitoring(request: MonitoringConfigRequest):
            try:
                model_type = ModelType(request.model_type.lower())
                
                # Convert custom thresholds if provided
                custom_thresholds = None
                if request.custom_thresholds:
                    custom_thresholds = [
                        PerformanceThreshold(**threshold) 
                        for threshold in request.custom_thresholds
                    ]
                
                # Start monitoring
                monitor_key = await self.performance_monitor.start_monitoring_model(
                    request.model_name, request.model_version, model_type, custom_thresholds
                )
                
                # Start rollback monitoring
                await self.rollback_system.start_automated_monitoring(
                    request.model_name, request.model_version, model_type
                )
                
                return {
                    "status": "success",
                    "monitor_key": monitor_key,
                    "message": "Monitoring started successfully"
                }
                
            except Exception as e:
                logger.error(f"Error starting monitoring: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/monitoring/status")
        async def get_monitoring_status(model_name: str = None):
            try:
                status = await self.performance_monitor.get_monitoring_status(model_name)
                return {
                    "monitoring_status": status,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting monitoring status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/monitoring/{model_name}/{model_version}/alerts")
        async def get_model_alerts(model_name: str, model_version: str, hours: int = 24):
            try:
                alerts = await self.performance_monitor.get_recent_alerts(
                    model_name, model_version, hours
                )
                
                return {
                    "model_name": model_name,
                    "model_version": model_version,
                    "alerts": [asdict(alert) for alert in alerts],
                    "alert_count": len(alerts)
                }
                
            except Exception as e:
                logger.error(f"Error getting model alerts: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Rollback endpoints
        @app.post("/rollback/{model_name}/{model_version}")
        async def initiate_rollback(model_name: str, model_version: str, 
                                  reason: str = "Manual rollback", 
                                  target_version: str = None):
            try:
                rollback_id = await self.rollback_system.initiate_rollback(
                    model_name, model_version, 
                    trigger=self.rollback_system.RollbackTrigger.MANUAL_TRIGGER,
                    reason=reason,
                    target_version=target_version
                )
                
                return {
                    "status": "success",
                    "rollback_id": rollback_id,
                    "message": "Rollback initiated"
                }
                
            except Exception as e:
                logger.error(f"Error initiating rollback: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/rollback/{rollback_id}")
        async def get_rollback_status(rollback_id: str):
            try:
                status = self.rollback_system.get_rollback_status(rollback_id)
                if not status:
                    raise HTTPException(status_code=404, detail="Rollback not found")
                
                return status
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting rollback status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/rollback/history")
        async def get_rollback_history(model_name: str = None):
            try:
                history = self.rollback_system.get_rollback_history(model_name)
                return {
                    "rollback_history": history,
                    "total_rollbacks": len(history)
                }
                
            except Exception as e:
                logger.error(f"Error getting rollback history: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Prediction endpoint for canary testing
        @app.post("/predict/{deployment_id}")
        async def make_prediction(deployment_id: str, input_data: Dict[str, Any]):
            try:
                result = await self.canary_manager.make_prediction(
                    deployment_id, input_data, DeploymentStrategy.CANARY
                )
                return result
                
            except Exception as e:
                logger.error(f"Error making prediction: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # System management endpoints
        @app.post("/system/initialize")
        async def initialize_system():
            try:
                await self.initialize()
                return {
                    "status": "success",
                    "message": "MLOps system initialized successfully"
                }
                
            except Exception as e:
                logger.error(f"Error initializing system: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/system/shutdown")
        async def shutdown_system():
            try:
                await self.cleanup()
                return {
                    "status": "success",
                    "message": "MLOps system shutdown initiated"
                }
                
            except Exception as e:
                logger.error(f"Error shutting down system: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        return app
    
    async def initialize(self) -> None:
        """Initialize the MLOps orchestrator and all components"""
        try:
            logger.info("Initializing MLOps orchestrator...")
            
            self.system_health["status"] = "initializing"
            
            # Initialize performance monitor
            await self.performance_monitor.initialize()
            self.system_health["components"]["performance_monitor"] = "healthy"
            logger.info("Performance monitor initialized")
            
            # Test model registry connection
            try:
                models = self.model_registry.search_models()
                self.system_health["components"]["model_registry"] = "healthy"
                logger.info(f"Model registry connected ({len(models)} models found)")
            except Exception as e:
                self.system_health["components"]["model_registry"] = f"error: {str(e)}"
                logger.warning(f"Model registry connection issue: {e}")
            
            # Initialize other components
            self.system_health["components"]["canary_manager"] = "healthy"
            self.system_health["components"]["rollback_system"] = "healthy"
            
            # Start background health monitoring
            health_task = asyncio.create_task(self._health_monitoring_loop())
            self.background_tasks.append(health_task)
            
            # Setup notification handlers
            self._setup_notification_handlers()
            
            self.system_health["status"] = "healthy"
            self.system_health["last_updated"] = datetime.utcnow()
            
            logger.info("MLOps orchestrator initialized successfully")
            
        except Exception as e:
            self.system_health["status"] = "error"
            self.system_health["error"] = str(e)
            logger.error(f"Error initializing MLOps orchestrator: {e}")
            raise
    
    async def _start_production_monitoring(self, model_name: str, model_version: str, 
                                         model_type: ModelType) -> None:
        """Start comprehensive monitoring for production model"""
        try:
            # Start performance monitoring
            await self.performance_monitor.start_monitoring_model(
                model_name, model_version, model_type
            )
            
            # Start automated rollback monitoring
            await self.rollback_system.start_automated_monitoring(
                model_name, model_version, model_type
            )
            
            logger.info(f"Started production monitoring for {model_name} v{model_version}")
            
        except Exception as e:
            logger.error(f"Error starting production monitoring: {e}")
    
    def _setup_notification_handlers(self) -> None:
        """Setup notification handlers for alerts and rollbacks"""
        def log_notification(event_type: str, data: Dict[str, Any]) -> None:
            logger.info(f"Notification: {event_type} - {json.dumps(data, indent=2)}")
        
        # Add notification handlers
        self.performance_monitor.add_alert_handler(
            lambda alert: log_notification("performance_alert", asdict(alert))
        )
        
        self.rollback_system.add_notification_handler(log_notification)
    
    async def _health_monitoring_loop(self) -> None:
        """Background task for system health monitoring"""
        while True:
            try:
                await asyncio.sleep(self.config.get("health_check_interval_seconds", 300))
                
                # Update component health
                await self._update_system_health()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_system_health(self) -> None:
        """Update system health status"""
        try:
            # Check model registry
            try:
                self.model_registry.search_models()
                self.system_health["components"]["model_registry"] = "healthy"
            except Exception as e:
                self.system_health["components"]["model_registry"] = f"error: {str(e)[:100]}"
            
            # Check performance monitor
            try:
                await self.performance_monitor.get_monitoring_status()
                self.system_health["components"]["performance_monitor"] = "healthy"
            except Exception as e:
                self.system_health["components"]["performance_monitor"] = f"error: {str(e)[:100]}"
            
            # Overall system status
            component_statuses = list(self.system_health["components"].values())
            if all("healthy" in status for status in component_statuses):
                self.system_health["status"] = "healthy"
            elif any("error" in status for status in component_statuses):
                self.system_health["status"] = "degraded"
            else:
                self.system_health["status"] = "unknown"
            
            self.system_health["last_updated"] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating system health: {e}")
    
    def start_server(self, host: str = None, port: int = None) -> None:
        """Start the MLOps API server"""
        host = host or self.config.get("api_host", "0.0.0.0")
        port = port or self.config.get("api_port", 8090)
        
        # Configure logging
        log_level = self.config.get("log_level", "INFO")
        logging.basicConfig(level=getattr(logging, log_level))
        
        logger.info(f"Starting MLOps orchestrator server on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level=log_level.lower()
        )
    
    async def run_async_server(self, host: str = None, port: int = None) -> None:
        """Run the MLOps API server asynchronously"""
        host = host or self.config.get("api_host", "0.0.0.0")
        port = port or self.config.get("api_port", 8090)
        
        # Initialize system first
        await self.initialize()
        
        # Configure uvicorn server
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level=self.config.get("log_level", "INFO").lower()
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    async def cleanup(self) -> None:
        """Cleanup all system resources"""
        try:
            logger.info("Cleaning up MLOps orchestrator...")
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Cleanup components
            await self.performance_monitor.cleanup()
            await self.rollback_system.cleanup()
            
            logger.info("MLOps orchestrator cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Convenience function to create and run orchestrator
async def main():
    """Main entry point for MLOps orchestrator"""
    orchestrator = MLOpsOrchestrator()
    
    try:
        await orchestrator.run_async_server()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())