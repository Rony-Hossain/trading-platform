"""
Drift Monitoring API Endpoints for MLOps Orchestrator
Additional endpoints for enhanced model drift and decay monitoring
"""

from fastapi import HTTPException, BackgroundTasks
import logging
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

from ..monitoring.drift_monitoring_service import DriftMonitoringService, MonitoringMetric
from ..workflows.retraining_orchestrator import RetrainingOrchestrator

logger = logging.getLogger(__name__)

def add_drift_monitoring_endpoints(app, orchestrator):
    """Add drift monitoring endpoints to the FastAPI app"""
    
    @app.post("/drift/psi-analysis")
    async def calculate_psi_score(request: "DriftMonitoringRequest"):
        """Calculate Population Stability Index (PSI) for feature drift detection"""
        try:
            expected_data = np.array(request.expected_data)
            actual_data = np.array(request.actual_data)
            
            psi_result = await orchestrator.drift_monitor.calculate_psi(
                expected_data, 
                actual_data, 
                request.feature_name
            )
            
            return {
                "status": "success",
                "psi_result": {
                    "feature_name": psi_result.feature_name,
                    "psi_score": psi_result.psi_score,
                    "is_unstable": psi_result.is_unstable,
                    "stability_threshold": psi_result.stability_threshold,
                    "recommendations": psi_result.recommendations
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/drift/performance-decay")
    async def track_performance_decay(request: "PerformanceDecayRequest"):
        """Track model performance decay compared to baseline"""
        try:
            from ..model_registry.mlflow_registry import ModelType
            
            # Convert string model type to enum
            model_type = ModelType(request.model_type)
            
            decay_results = await orchestrator.drift_monitor.track_performance_decay(
                request.model_name,
                request.model_version,
                model_type,
                request.current_predictions,
                request.actual_values
            )
            
            results_data = []
            for result in decay_results:
                results_data.append({
                    "metric": result.metric.value,
                    "baseline_value": result.baseline_value,
                    "current_value": result.current_value,
                    "decay_percentage": result.decay_percentage,
                    "is_degraded": result.is_degraded,
                    "degradation_threshold": result.degradation_threshold,
                    "recommendation": result.recommendation
                })
            
            return {
                "status": "success",
                "model_name": request.model_name,
                "model_version": request.model_version,
                "performance_decay_results": results_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error tracking performance decay: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/drift/retraining-triggers")
    async def check_retraining_triggers(model_name: str, model_version: str, model_type: str):
        """Check if automated retraining should be triggered"""
        try:
            from ..model_registry.mlflow_registry import ModelType
            
            model_type_enum = ModelType(model_type)
            
            triggers = await orchestrator.drift_monitor.check_retraining_triggers(
                model_name,
                model_version,
                model_type_enum
            )
            
            trigger_data = []
            for trigger in triggers:
                trigger_data.append({
                    "trigger_id": trigger.trigger_id,
                    "trigger_type": trigger.trigger_type,
                    "is_triggered": trigger.is_triggered,
                    "priority": trigger.priority,
                    "trigger_conditions": trigger.trigger_conditions,
                    "trigger_timestamp": trigger.trigger_timestamp.isoformat()
                })
            
            return {
                "status": "success",
                "model_name": model_name,
                "model_version": model_version,
                "retraining_triggers": trigger_data,
                "triggers_count": len(triggers),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking retraining triggers: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/retraining/submit-workflow")
    async def submit_retraining_workflow(request: "RetrainingTriggerRequest", background_tasks: BackgroundTasks):
        """Submit an automated retraining workflow"""
        try:
            from ..monitoring.drift_monitoring_service import RetrainingTrigger
            
            # Create retraining trigger
            trigger = RetrainingTrigger(
                trigger_id=f"manual_{request.model_name}_{int(datetime.utcnow().timestamp())}",
                model_name=request.model_name,
                model_version=request.model_version,
                trigger_type=request.trigger_type,
                trigger_conditions=request.trigger_conditions,
                is_triggered=True,
                trigger_timestamp=datetime.utcnow(),
                priority=request.priority,
                retraining_config={}
            )
            
            # Submit workflow
            workflow_id = await orchestrator.retraining_orchestrator.submit_retraining_workflow(trigger)
            
            # Execute workflow if auto_execute is enabled
            if request.auto_execute:
                background_tasks.add_task(
                    orchestrator.retraining_orchestrator.start_workflow_processor
                )
            
            return {
                "status": "success",
                "workflow_id": workflow_id,
                "trigger_id": trigger.trigger_id,
                "message": "Retraining workflow submitted successfully",
                "auto_execute": request.auto_execute,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error submitting retraining workflow: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/retraining/workflows/{workflow_id}/status")
    async def get_workflow_status(workflow_id: str):
        """Get status of a retraining workflow"""
        try:
            workflow = await orchestrator.retraining_orchestrator.get_workflow_status(workflow_id)
            
            if not workflow:
                raise HTTPException(status_code=404, detail="Workflow not found")
            
            return {
                "status": "success",
                "workflow": {
                    "workflow_id": workflow.workflow_id,
                    "model_name": workflow.model_name,
                    "current_version": workflow.current_version,
                    "new_model_version": workflow.new_model_version,
                    "status": workflow.status.value,
                    "progress_percentage": workflow.progress_percentage,
                    "current_step": workflow.current_step,
                    "steps_completed": workflow.steps_completed,
                    "created_timestamp": workflow.created_timestamp.isoformat() if workflow.created_timestamp else None,
                    "started_timestamp": workflow.started_timestamp.isoformat() if workflow.started_timestamp else None,
                    "completed_timestamp": workflow.completed_timestamp.isoformat() if workflow.completed_timestamp else None,
                    "error_message": workflow.error_message,
                    "performance_metrics": workflow.performance_metrics,
                    "validation_results": workflow.validation_results
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/retraining/workflows/{workflow_id}/cancel")
    async def cancel_workflow(workflow_id: str):
        """Cancel a running retraining workflow"""
        try:
            success = await orchestrator.retraining_orchestrator.cancel_workflow(workflow_id)
            
            return {
                "status": "success" if success else "failed",
                "workflow_id": workflow_id,
                "cancelled": success,
                "message": "Workflow cancelled successfully" if success else "Failed to cancel workflow",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error cancelling workflow: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/drift/monitoring-dashboard/{model_name}/{model_version}")
    async def get_monitoring_dashboard(model_name: str, model_version: str):
        """Get comprehensive monitoring dashboard data"""
        try:
            dashboard_data = await orchestrator.drift_monitor.get_monitoring_dashboard_data(
                model_name, model_version
            )
            
            return {
                "status": "success",
                "dashboard_data": dashboard_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/drift/health-check")
    async def drift_monitoring_health():
        """Health check for drift monitoring services"""
        try:
            health_status = {
                "drift_monitor": "healthy",
                "retraining_orchestrator": "healthy",
                "psi_calculation": "available",
                "performance_decay_tracking": "available",
                "automated_retraining": "available",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check Redis connection
            if hasattr(orchestrator.drift_monitor, 'redis_client') and orchestrator.drift_monitor.redis_client:
                try:
                    await orchestrator.drift_monitor.redis_client.ping()
                    health_status["redis_connection"] = "healthy"
                except:
                    health_status["redis_connection"] = "unhealthy"
            
            return {
                "status": "success",
                "health_status": health_status
            }
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            raise HTTPException(status_code=500, detail=str(e))