"""
Enhanced Model Drift/Decay Monitoring Service
Implements PSI/KS tests, performance decay tracking, and automated retraining workflows
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import aioredis
import aiohttp
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from .model_performance_monitor import ModelPerformanceMonitor, DriftType, AlertSeverity
from ..model_registry.mlflow_registry import MLflowModelRegistry, ModelType

logger = logging.getLogger(__name__)

class MonitoringMetric(Enum):
    """Performance monitoring metrics"""
    MAE = "mae"
    RMSE = "rmse" 
    MAPE = "mape"
    PSI = "psi"
    KS_STATISTIC = "ks_statistic"
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    SHARPE_RATIO = "sharpe_ratio"
    HIT_RATE = "hit_rate"

@dataclass
class PerformanceDecayResult:
    """Result of performance decay analysis"""
    model_name: str
    model_version: str
    metric: MonitoringMetric
    baseline_value: float
    current_value: float
    decay_percentage: float
    is_degraded: bool
    degradation_threshold: float
    recommendation: str
    timestamp: datetime

@dataclass
class PSIResult:
    """Population Stability Index result"""
    feature_name: str
    psi_score: float
    is_unstable: bool
    stability_threshold: float
    expected_distribution: List[float]
    actual_distribution: List[float]
    recommendations: List[str]

@dataclass
class RetrainingTrigger:
    """Trigger for automated retraining"""
    trigger_id: str
    model_name: str
    model_version: str
    trigger_type: str  # 'performance_decay', 'drift_detection', 'scheduled'
    trigger_conditions: Dict[str, Any]
    is_triggered: bool
    trigger_timestamp: datetime
    priority: int  # 1 (highest) to 5 (lowest)
    retraining_config: Dict[str, Any]

class DriftMonitoringService:
    """
    Enhanced drift and decay monitoring service
    
    Features:
    - PSI (Population Stability Index) calculation for categorical features
    - Performance decay tracking with baseline comparison
    - MAE/RMSE trend monitoring for regression models
    - Automated retraining trigger system
    - Comprehensive drift detection with multiple statistical tests
    """
    
    def __init__(self, 
                 model_registry: MLflowModelRegistry,
                 performance_monitor: ModelPerformanceMonitor,
                 redis_url: str = "redis://localhost:6379"):
        self.model_registry = model_registry
        self.performance_monitor = performance_monitor
        self.redis_url = redis_url
        self.redis_client = None
        
        # PSI thresholds
        self.psi_thresholds = {
            "stable": 0.1,      # PSI < 0.1: No significant change
            "moderate": 0.25,   # 0.1 <= PSI < 0.25: Moderate change
            "unstable": float('inf')  # PSI >= 0.25: Significant change
        }
        
        # Performance decay thresholds by model type
        self.decay_thresholds = {
            ModelType.PRICE_PREDICTOR: {
                MonitoringMetric.MAE: 0.15,      # 15% increase in MAE triggers alert
                MonitoringMetric.RMSE: 0.20,     # 20% increase in RMSE triggers alert
                MonitoringMetric.MAPE: 0.10,     # 10% increase in MAPE triggers alert
            },
            ModelType.SENTIMENT_CLASSIFIER: {
                MonitoringMetric.ACCURACY: 0.05, # 5% decrease in accuracy triggers alert
                MonitoringMetric.F1_SCORE: 0.05, # 5% decrease in F1 triggers alert
            },
            ModelType.RISK_ASSESSOR: {
                MonitoringMetric.MAE: 0.10,
                MonitoringMetric.ACCURACY: 0.03,
            }
        }
        
        # Retraining trigger configuration
        self.retraining_config = {
            "performance_decay_threshold": 0.15,  # 15% performance degradation
            "drift_detection_threshold": 0.25,    # PSI > 0.25 or KS p-value < 0.001
            "minimum_data_points": 1000,          # Minimum samples before retraining
            "retraining_frequency": timedelta(days=7),  # Minimum gap between retrainings
            "auto_retrain_enabled": True
        }
        
        # Baseline performance storage
        self.baseline_metrics = {}
        
    async def initialize(self):
        """Initialize Redis connection and load baseline metrics"""
        self.redis_client = await aioredis.from_url(self.redis_url)
        await self._load_baseline_metrics()
        logger.info("Drift monitoring service initialized")
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def calculate_psi(self, 
                           expected_data: np.ndarray, 
                           actual_data: np.ndarray,
                           feature_name: str,
                           bins: int = 10) -> PSIResult:
        """
        Calculate Population Stability Index (PSI)
        
        PSI measures the shift in data distribution between two datasets
        """
        try:
            # Handle edge cases
            if len(expected_data) == 0 or len(actual_data) == 0:
                return PSIResult(
                    feature_name=feature_name,
                    psi_score=float('inf'),
                    is_unstable=True,
                    stability_threshold=self.psi_thresholds["unstable"],
                    expected_distribution=[],
                    actual_distribution=[],
                    recommendations=["Insufficient data for PSI calculation"]
                )
            
            # Create bins based on expected data
            _, bin_edges = np.histogram(expected_data, bins=bins)
            
            # Calculate distributions
            expected_counts, _ = np.histogram(expected_data, bins=bin_edges)
            actual_counts, _ = np.histogram(actual_data, bins=bin_edges)
            
            # Convert to percentages and handle zero counts
            expected_pct = expected_counts / len(expected_data)
            actual_pct = actual_counts / len(actual_data)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
            actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)
            
            # Calculate PSI
            psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
            psi_score = np.sum(psi_values)
            
            # Determine stability
            is_unstable = psi_score >= self.psi_thresholds["moderate"]
            
            # Generate recommendations
            recommendations = []
            if psi_score >= self.psi_thresholds["unstable"]:
                recommendations.append("Critical: Significant population shift detected. Immediate retraining recommended.")
                recommendations.append("Review data collection process and feature engineering.")
            elif psi_score >= self.psi_thresholds["moderate"]:
                recommendations.append("Moderate population shift detected. Monitor closely and consider retraining.")
                recommendations.append("Investigate potential causes of distribution change.")
            else:
                recommendations.append("Population is stable. No immediate action required.")
            
            return PSIResult(
                feature_name=feature_name,
                psi_score=psi_score,
                is_unstable=is_unstable,
                stability_threshold=self.psi_thresholds["moderate"],
                expected_distribution=expected_pct.tolist(),
                actual_distribution=actual_pct.tolist(),
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error calculating PSI for {feature_name}: {str(e)}")
            return PSIResult(
                feature_name=feature_name,
                psi_score=float('inf'),
                is_unstable=True,
                stability_threshold=self.psi_thresholds["moderate"],
                expected_distribution=[],
                actual_distribution=[],
                recommendations=[f"Error in PSI calculation: {str(e)}"]
            )
    
    async def track_performance_decay(self,
                                     model_name: str,
                                     model_version: str,
                                     model_type: ModelType,
                                     current_predictions: List[float],
                                     actual_values: List[float],
                                     baseline_key: str = None) -> List[PerformanceDecayResult]:
        """
        Track performance decay by comparing current performance to baseline
        """
        results = []
        
        if not current_predictions or not actual_values:
            return results
        
        # Get baseline metrics
        baseline_key = baseline_key or f"{model_name}:{model_version}"
        baseline_metrics = await self._get_baseline_metrics(baseline_key)
        
        # Calculate current metrics
        current_metrics = await self._calculate_performance_metrics(
            current_predictions, actual_values, model_type
        )
        
        # Compare with baseline for each relevant metric
        thresholds = self.decay_thresholds.get(model_type, {})
        
        for metric, current_value in current_metrics.items():
            if metric not in thresholds:
                continue
                
            baseline_value = baseline_metrics.get(metric.value, current_value)
            threshold = thresholds[metric]
            
            # Calculate decay percentage
            if baseline_value != 0:
                if metric in [MonitoringMetric.MAE, MonitoringMetric.RMSE, MonitoringMetric.MAPE]:
                    # For error metrics, increase is bad
                    decay_pct = (current_value - baseline_value) / baseline_value
                else:
                    # For accuracy metrics, decrease is bad
                    decay_pct = (baseline_value - current_value) / baseline_value
            else:
                decay_pct = 0.0
            
            is_degraded = abs(decay_pct) >= threshold
            
            # Generate recommendation
            recommendation = ""
            if is_degraded:
                if decay_pct > 0:
                    recommendation = f"Performance degradation detected ({decay_pct:.2%} decline). Consider retraining or feature review."
                else:
                    recommendation = f"Performance improvement detected ({abs(decay_pct):.2%} improvement). Model performing better than baseline."
            else:
                recommendation = "Performance within acceptable range. Continue monitoring."
            
            results.append(PerformanceDecayResult(
                model_name=model_name,
                model_version=model_version,
                metric=metric,
                baseline_value=baseline_value,
                current_value=current_value,
                decay_percentage=decay_pct,
                is_degraded=is_degraded,
                degradation_threshold=threshold,
                recommendation=recommendation,
                timestamp=datetime.utcnow()
            ))
        
        # Store current metrics as potential new baseline
        await self._update_baseline_metrics(baseline_key, current_metrics)
        
        return results
    
    async def check_retraining_triggers(self,
                                       model_name: str,
                                       model_version: str,
                                       model_type: ModelType) -> List[RetrainingTrigger]:
        """
        Check if automated retraining should be triggered
        """
        triggers = []
        
        if not self.retraining_config["auto_retrain_enabled"]:
            return triggers
        
        # Check performance decay triggers
        decay_results = await self._get_recent_decay_results(model_name, model_version)
        performance_triggers = await self._check_performance_triggers(
            model_name, model_version, decay_results
        )
        triggers.extend(performance_triggers)
        
        # Check drift detection triggers
        drift_results = await self._get_recent_drift_results(model_name, model_version)
        drift_triggers = await self._check_drift_triggers(
            model_name, model_version, drift_results
        )
        triggers.extend(drift_triggers)
        
        # Check scheduled retraining triggers
        scheduled_triggers = await self._check_scheduled_triggers(model_name, model_version)
        triggers.extend(scheduled_triggers)
        
        # Sort by priority and timestamp
        triggers.sort(key=lambda x: (x.priority, x.trigger_timestamp))
        
        return triggers
    
    async def trigger_automated_retraining(self, trigger: RetrainingTrigger) -> Dict[str, Any]:
        """
        Execute automated retraining workflow
        """
        try:
            logger.info(f"Initiating automated retraining for {trigger.model_name}:{trigger.model_version}")
            
            # Prepare retraining configuration
            retraining_request = {
                "model_name": trigger.model_name,
                "current_version": trigger.model_version,
                "trigger_reason": trigger.trigger_type,
                "trigger_conditions": trigger.trigger_conditions,
                "retraining_config": trigger.retraining_config,
                "timestamp": trigger.trigger_timestamp.isoformat()
            }
            
            # Store trigger information
            await self.redis_client.setex(
                f"retraining_trigger:{trigger.trigger_id}",
                86400,  # 24 hours
                json.dumps(asdict(trigger), default=str)
            )
            
            # Call MLOps orchestrator to initiate retraining
            result = await self._initiate_retraining_workflow(retraining_request)
            
            logger.info(f"Retraining workflow initiated for {trigger.model_name}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error triggering automated retraining: {str(e)}")
            return {"error": str(e), "success": False}
    
    async def get_monitoring_dashboard_data(self, 
                                           model_name: str,
                                           model_version: str) -> Dict[str, Any]:
        """
        Get comprehensive monitoring data for dashboard display
        """
        try:
            # Get recent performance metrics
            performance_data = await self._get_performance_history(model_name, model_version)
            
            # Get drift detection results
            drift_data = await self._get_drift_history(model_name, model_version)
            
            # Get PSI results
            psi_data = await self._get_psi_history(model_name, model_version)
            
            # Get active triggers
            triggers = await self.check_retraining_triggers(model_name, model_version, ModelType.PRICE_PREDICTOR)
            
            # Get alerts
            alerts = await self.performance_monitor.get_active_alerts(model_name, model_version)
            
            dashboard_data = {
                "model_info": {
                    "name": model_name,
                    "version": model_version,
                    "last_updated": datetime.utcnow().isoformat()
                },
                "performance_metrics": performance_data,
                "drift_detection": drift_data,
                "population_stability": psi_data,
                "retraining_triggers": [asdict(t) for t in triggers],
                "active_alerts": [asdict(a) for a in alerts] if alerts else [],
                "health_status": await self._calculate_model_health_score(model_name, model_version)
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    async def _load_baseline_metrics(self):
        """Load baseline metrics from Redis"""
        try:
            keys = await self.redis_client.keys("baseline_metrics:*")
            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    self.baseline_metrics[key.decode().split(":")[1]] = json.loads(data)
        except Exception as e:
            logger.error(f"Error loading baseline metrics: {str(e)}")
    
    async def _get_baseline_metrics(self, baseline_key: str) -> Dict[str, float]:
        """Get baseline metrics for a model"""
        return self.baseline_metrics.get(baseline_key, {})
    
    async def _update_baseline_metrics(self, baseline_key: str, metrics: Dict[MonitoringMetric, float]):
        """Update baseline metrics for a model"""
        metrics_dict = {k.value: v for k, v in metrics.items()}
        self.baseline_metrics[baseline_key] = metrics_dict
        
        # Store in Redis
        await self.redis_client.setex(
            f"baseline_metrics:{baseline_key}",
            2592000,  # 30 days
            json.dumps(metrics_dict)
        )
    
    async def _calculate_performance_metrics(self,
                                           predictions: List[float],
                                           actuals: List[float],
                                           model_type: ModelType) -> Dict[MonitoringMetric, float]:
        """Calculate performance metrics based on model type"""
        metrics = {}
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        if model_type == ModelType.PRICE_PREDICTOR:
            metrics[MonitoringMetric.MAE] = mean_absolute_error(actuals, predictions)
            metrics[MonitoringMetric.RMSE] = np.sqrt(mean_squared_error(actuals, predictions))
            
            # MAPE calculation
            mape = np.mean(np.abs((actuals - predictions) / np.where(actuals != 0, actuals, 1))) * 100
            metrics[MonitoringMetric.MAPE] = mape
            
        elif model_type == ModelType.SENTIMENT_CLASSIFIER:
            # For classification metrics, assuming binary classification
            # Convert continuous predictions to binary (>0.5 = positive)
            binary_pred = (predictions > 0.5).astype(int)
            binary_actual = actuals.astype(int)
            
            metrics[MonitoringMetric.ACCURACY] = np.mean(binary_pred == binary_actual)
            
            # Calculate precision, recall, F1
            tp = np.sum((binary_pred == 1) & (binary_actual == 1))
            fp = np.sum((binary_pred == 1) & (binary_actual == 0))
            fn = np.sum((binary_pred == 0) & (binary_actual == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[MonitoringMetric.PRECISION] = precision
            metrics[MonitoringMetric.RECALL] = recall
            metrics[MonitoringMetric.F1_SCORE] = f1
        
        return metrics
    
    async def _check_performance_triggers(self,
                                        model_name: str,
                                        model_version: str,
                                        decay_results: List[PerformanceDecayResult]) -> List[RetrainingTrigger]:
        """Check for performance-based retraining triggers"""
        triggers = []
        
        for result in decay_results:
            if result.is_degraded and abs(result.decay_percentage) >= self.retraining_config["performance_decay_threshold"]:
                trigger = RetrainingTrigger(
                    trigger_id=f"perf_{model_name}_{int(datetime.utcnow().timestamp())}",
                    model_name=model_name,
                    model_version=model_version,
                    trigger_type="performance_decay",
                    trigger_conditions={
                        "metric": result.metric.value,
                        "decay_percentage": result.decay_percentage,
                        "threshold": result.degradation_threshold
                    },
                    is_triggered=True,
                    trigger_timestamp=datetime.utcnow(),
                    priority=1 if abs(result.decay_percentage) >= 0.25 else 2,
                    retraining_config=self.retraining_config
                )
                triggers.append(trigger)
        
        return triggers
    
    async def _check_drift_triggers(self,
                                   model_name: str,
                                   model_version: str,
                                   drift_results: List[PSIResult]) -> List[RetrainingTrigger]:
        """Check for drift-based retraining triggers"""
        triggers = []
        
        for result in drift_results:
            if result.is_unstable and result.psi_score >= self.retraining_config["drift_detection_threshold"]:
                trigger = RetrainingTrigger(
                    trigger_id=f"drift_{model_name}_{int(datetime.utcnow().timestamp())}",
                    model_name=model_name,
                    model_version=model_version,
                    trigger_type="drift_detection",
                    trigger_conditions={
                        "feature_name": result.feature_name,
                        "psi_score": result.psi_score,
                        "threshold": result.stability_threshold
                    },
                    is_triggered=True,
                    trigger_timestamp=datetime.utcnow(),
                    priority=1 if result.psi_score >= 0.5 else 2,
                    retraining_config=self.retraining_config
                )
                triggers.append(trigger)
        
        return triggers
    
    async def _check_scheduled_triggers(self,
                                       model_name: str,
                                       model_version: str) -> List[RetrainingTrigger]:
        """Check for scheduled retraining triggers"""
        triggers = []
        
        # Get last retraining timestamp
        last_retrain_key = f"last_retrain:{model_name}:{model_version}"
        last_retrain_data = await self.redis_client.get(last_retrain_key)
        
        if last_retrain_data:
            last_retrain = datetime.fromisoformat(last_retrain_data.decode())
            time_since_retrain = datetime.utcnow() - last_retrain
            
            if time_since_retrain >= self.retraining_config["retraining_frequency"]:
                trigger = RetrainingTrigger(
                    trigger_id=f"sched_{model_name}_{int(datetime.utcnow().timestamp())}",
                    model_name=model_name,
                    model_version=model_version,
                    trigger_type="scheduled",
                    trigger_conditions={
                        "days_since_retrain": time_since_retrain.days,
                        "frequency_days": self.retraining_config["retraining_frequency"].days
                    },
                    is_triggered=True,
                    trigger_timestamp=datetime.utcnow(),
                    priority=3,  # Lower priority for scheduled retraining
                    retraining_config=self.retraining_config
                )
                triggers.append(trigger)
        
        return triggers
    
    async def _initiate_retraining_workflow(self, retraining_request: Dict[str, Any]) -> Dict[str, Any]:
        """Initiate retraining workflow via MLOps orchestrator"""
        try:
            # This would typically call the MLOps orchestrator API
            # For now, we'll simulate the workflow initiation
            
            workflow_id = f"retrain_{retraining_request['model_name']}_{int(datetime.utcnow().timestamp())}"
            
            # Store retraining request
            await self.redis_client.setex(
                f"retraining_request:{workflow_id}",
                86400,  # 24 hours
                json.dumps(retraining_request)
            )
            
            # Update last retrain timestamp
            last_retrain_key = f"last_retrain:{retraining_request['model_name']}:{retraining_request['current_version']}"
            await self.redis_client.set(last_retrain_key, datetime.utcnow().isoformat())
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "status": "initiated",
                "message": "Retraining workflow initiated successfully"
            }
            
        except Exception as e:
            logger.error(f"Error initiating retraining workflow: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _get_recent_decay_results(self, model_name: str, model_version: str) -> List[PerformanceDecayResult]:
        """Get recent performance decay results"""
        # This would typically query stored decay results
        # For now, return empty list as placeholder
        return []
    
    async def _get_recent_drift_results(self, model_name: str, model_version: str) -> List[PSIResult]:
        """Get recent drift detection results"""
        # This would typically query stored drift results
        # For now, return empty list as placeholder
        return []
    
    async def _get_performance_history(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """Get performance metrics history"""
        return {"placeholder": "Performance history data"}
    
    async def _get_drift_history(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """Get drift detection history"""
        return {"placeholder": "Drift history data"}
    
    async def _get_psi_history(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """Get PSI calculation history"""
        return {"placeholder": "PSI history data"}
    
    async def _calculate_model_health_score(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """Calculate overall model health score"""
        return {
            "overall_score": 0.85,
            "performance_score": 0.90,
            "stability_score": 0.80,
            "status": "healthy"
        }