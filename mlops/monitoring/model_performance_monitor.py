"""
Model Performance Monitoring System
Real-time monitoring with automated alerting and drift detection
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
import aioredis
import aiohttp
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

from ..model_registry.mlflow_registry import MLflowModelRegistry, ModelType
from ..deployment.canary_deployment import PredictionMetrics

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DriftType(Enum):
    """Types of model drift"""
    DATA_DRIFT = "data_drift"          # Input data distribution changes
    CONCEPT_DRIFT = "concept_drift"    # Relationship between input and output changes
    PREDICTION_DRIFT = "prediction_drift"  # Model predictions change significantly
    PERFORMANCE_DRIFT = "performance_drift"  # Model performance degrades

@dataclass
class PerformanceThreshold:
    """Performance monitoring thresholds"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison_operator: str  # 'gt', 'lt', 'gte', 'lte'
    window_size: int = 1000  # Number of samples to consider

@dataclass
class ModelAlert:
    """Model performance alert"""
    alert_id: str
    model_name: str
    model_version: str
    alert_type: str
    severity: AlertSeverity
    message: str
    metric_values: Dict[str, float]
    threshold_values: Dict[str, float]
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis"""
    drift_type: DriftType
    is_drift_detected: bool
    drift_score: float
    p_value: float
    reference_period: Tuple[datetime, datetime]
    current_period: Tuple[datetime, datetime]
    affected_features: List[str]
    recommendation: str

class ModelPerformanceMonitor:
    """
    Comprehensive model performance monitoring system
    
    Features:
    - Real-time performance tracking
    - Data drift detection using statistical tests
    - Concept drift detection
    - Automated alerting system
    - Performance degradation detection
    - A/B test result monitoring
    """
    
    def __init__(self, model_registry: MLflowModelRegistry, redis_url: str = "redis://localhost:6379"):
        self.model_registry = model_registry
        self.redis_url = redis_url
        self.redis_client = None
        
        # Performance thresholds by model type
        self.default_thresholds = {
            ModelType.SENTIMENT_CLASSIFIER: [
                PerformanceThreshold("accuracy", 0.80, 0.70, "lt"),
                PerformanceThreshold("f1_score", 0.75, 0.65, "lt"),
                PerformanceThreshold("latency_ms", 100, 200, "gt"),
                PerformanceThreshold("error_rate", 0.05, 0.10, "gt")
            ],
            ModelType.PRICE_PREDICTOR: [
                PerformanceThreshold("mae", 0.10, 0.20, "gt"),
                PerformanceThreshold("rmse", 0.15, 0.30, "gt"),
                PerformanceThreshold("r2", 0.60, 0.40, "lt"),
                PerformanceThreshold("latency_ms", 50, 100, "gt")
            ],
            ModelType.RISK_ASSESSOR: [
                PerformanceThreshold("precision", 0.80, 0.70, "lt"),
                PerformanceThreshold("recall", 0.75, 0.65, "lt"),
                PerformanceThreshold("auc", 0.85, 0.75, "lt"),
                PerformanceThreshold("latency_ms", 75, 150, "gt")
            ]
        }
        
        # Active monitoring tasks
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.alert_handlers: List[Callable[[ModelAlert], None]] = []
        
        # Data storage for drift detection
        self.prediction_history: Dict[str, List[Dict]] = {}
        self.performance_history: Dict[str, List[Dict]] = {}
        
        # Drift detection parameters
        self.drift_detection_config = {
            "min_samples": 1000,
            "reference_window_days": 7,
            "current_window_days": 1,
            "significance_level": 0.05,
            "drift_threshold": 0.1
        }
    
    async def initialize(self) -> None:
        """Initialize monitoring system"""
        try:
            self.redis_client = await aioredis.from_url(self.redis_url)
            logger.info("Model performance monitor initialized")
            
        except Exception as e:
            logger.error(f"Error initializing performance monitor: {e}")
            raise
    
    async def start_monitoring_model(self, model_name: str, model_version: str, 
                                   model_type: ModelType, 
                                   custom_thresholds: List[PerformanceThreshold] = None) -> str:
        """Start monitoring a specific model version"""
        try:
            monitor_key = f"{model_name}:{model_version}"
            
            if monitor_key in self.monitoring_tasks:
                logger.warning(f"Already monitoring {monitor_key}")
                return monitor_key
            
            # Use custom thresholds or defaults
            thresholds = custom_thresholds or self.default_thresholds.get(model_type, [])
            
            # Store monitoring configuration
            monitor_config = {
                "model_name": model_name,
                "model_version": model_version,
                "model_type": model_type.value,
                "thresholds": [asdict(t) for t in thresholds],
                "started_at": datetime.utcnow().isoformat(),
                "status": "active"
            }
            
            await self.redis_client.hset(
                f"monitor_config:{monitor_key}",
                mapping={k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                        for k, v in monitor_config.items()}
            )
            
            # Start monitoring task
            task = asyncio.create_task(self._monitor_model_performance(monitor_key, thresholds))
            self.monitoring_tasks[monitor_key] = task
            
            logger.info(f"Started monitoring model {monitor_key}")
            return monitor_key
            
        except Exception as e:
            logger.error(f"Error starting model monitoring: {e}")
            raise
    
    async def _monitor_model_performance(self, monitor_key: str, 
                                       thresholds: List[PerformanceThreshold]) -> None:
        """Main monitoring loop for a model"""
        model_name, model_version = monitor_key.split(":", 1)
        
        logger.info(f"Monitoring task started for {monitor_key}")
        
        while monitor_key in self.monitoring_tasks:
            try:
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
                # Get recent performance metrics
                metrics = await self._get_recent_metrics(model_name, model_version)
                
                if not metrics:
                    continue
                
                # Check performance thresholds
                await self._check_performance_thresholds(monitor_key, metrics, thresholds)
                
                # Run drift detection every hour
                current_time = datetime.utcnow()
                if current_time.minute == 0:  # Top of the hour
                    await self._detect_model_drift(monitor_key)
                
                # Update monitoring status
                await self._update_monitoring_status(monitor_key, metrics)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop for {monitor_key}: {e}")
                await asyncio.sleep(60)
    
    async def _get_recent_metrics(self, model_name: str, model_version: str) -> Optional[Dict[str, Any]]:
        """Get recent performance metrics for a model"""
        try:
            # Get metrics from Redis (would be populated by prediction service)
            metrics_key = f"model_metrics:{model_name}:{model_version}"
            raw_metrics = await self.redis_client.lrange(metrics_key, 0, 999)  # Last 1000 predictions
            
            if not raw_metrics:
                return None
            
            # Parse metrics
            metrics_list = []
            for raw_metric in raw_metrics:
                try:
                    metric = json.loads(raw_metric)
                    metrics_list.append(metric)
                except json.JSONDecodeError:
                    continue
            
            if not metrics_list:
                return None
            
            # Calculate aggregate metrics
            latencies = [m.get('latency_ms', 0) for m in metrics_list]
            confidences = [m.get('confidence', 1.0) for m in metrics_list]
            errors = [m for m in metrics_list if m.get('error', False)]
            
            # Get ground truth for accuracy calculation (if available)
            accuracy = await self._calculate_accuracy_if_available(model_name, model_version, metrics_list)
            
            aggregated_metrics = {
                "sample_count": len(metrics_list),
                "avg_latency_ms": np.mean(latencies) if latencies else 0,
                "max_latency_ms": np.max(latencies) if latencies else 0,
                "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
                "avg_confidence": np.mean(confidences) if confidences else 0,
                "min_confidence": np.min(confidences) if confidences else 0,
                "error_rate": len(errors) / len(metrics_list) if metrics_list else 0,
                "throughput_per_minute": len(metrics_list),  # Assuming 1-minute window
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if accuracy is not None:
                aggregated_metrics["accuracy"] = accuracy
            
            return aggregated_metrics
            
        except Exception as e:
            logger.error(f"Error getting recent metrics: {e}")
            return None
    
    async def _calculate_accuracy_if_available(self, model_name: str, model_version: str,
                                             metrics_list: List[Dict]) -> Optional[float]:
        """Calculate accuracy if ground truth labels are available"""
        try:
            # Check if we have ground truth data
            ground_truth_key = f"ground_truth:{model_name}"
            
            predictions_with_truth = []
            for metric in metrics_list:
                request_id = metric.get('request_id')
                if request_id:
                    truth = await self.redis_client.hget(ground_truth_key, request_id)
                    if truth:
                        predictions_with_truth.append({
                            'prediction': metric.get('prediction'),
                            'truth': json.loads(truth)
                        })
            
            if len(predictions_with_truth) < 10:  # Need minimum samples
                return None
            
            predictions = [p['prediction'] for p in predictions_with_truth]
            truths = [p['truth'] for p in predictions_with_truth]
            
            # Calculate accuracy based on model type
            if isinstance(predictions[0], (list, np.ndarray)):
                # Multi-class or regression
                if len(set(truths)) <= 10:  # Classification
                    return accuracy_score(truths, predictions)
                else:  # Regression - use R²
                    correlation = np.corrcoef(predictions, truths)[0, 1]
                    return correlation ** 2 if not np.isnan(correlation) else 0.0
            else:
                # Binary or simple classification
                return accuracy_score(truths, predictions)
                
        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            return None
    
    async def _check_performance_thresholds(self, monitor_key: str, 
                                          metrics: Dict[str, Any],
                                          thresholds: List[PerformanceThreshold]) -> None:
        """Check if metrics exceed performance thresholds"""
        try:
            model_name, model_version = monitor_key.split(":", 1)
            
            for threshold in thresholds:
                metric_value = metrics.get(threshold.metric_name)
                
                if metric_value is None:
                    continue
                
                # Check threshold violation
                violation_severity = self._check_threshold_violation(metric_value, threshold)
                
                if violation_severity:
                    alert = ModelAlert(
                        alert_id=f"perf_{monitor_key}_{threshold.metric_name}_{int(datetime.utcnow().timestamp())}",
                        model_name=model_name,
                        model_version=model_version,
                        alert_type="performance_threshold",
                        severity=violation_severity,
                        message=f"Performance threshold violated: {threshold.metric_name} = {metric_value:.4f}",
                        metric_values={threshold.metric_name: metric_value},
                        threshold_values={
                            "warning": threshold.warning_threshold,
                            "critical": threshold.critical_threshold
                        },
                        timestamp=datetime.utcnow()
                    )
                    
                    await self._handle_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error checking performance thresholds: {e}")
    
    def _check_threshold_violation(self, value: float, threshold: PerformanceThreshold) -> Optional[AlertSeverity]:
        """Check if a value violates threshold and return severity"""
        def compare(val, thresh, op):
            if op == "gt":
                return val > thresh
            elif op == "lt":
                return val < thresh
            elif op == "gte":
                return val >= thresh
            elif op == "lte":
                return val <= thresh
            return False
        
        if compare(value, threshold.critical_threshold, threshold.comparison_operator):
            return AlertSeverity.CRITICAL
        elif compare(value, threshold.warning_threshold, threshold.comparison_operator):
            return AlertSeverity.MEDIUM
        
        return None
    
    async def _detect_model_drift(self, monitor_key: str) -> None:
        """Detect various types of model drift"""
        try:
            model_name, model_version = monitor_key.split(":", 1)
            
            # Get historical data
            current_data = await self._get_recent_prediction_data(model_name, model_version, days=1)
            reference_data = await self._get_historical_prediction_data(
                model_name, model_version, 
                days_back=8, days_duration=7  # 7-day reference period, 1 week ago
            )
            
            if not current_data or not reference_data:
                logger.debug(f"Insufficient data for drift detection: {monitor_key}")
                return
            
            if len(current_data) < self.drift_detection_config["min_samples"] or \
               len(reference_data) < self.drift_detection_config["min_samples"]:
                logger.debug(f"Insufficient samples for drift detection: {monitor_key}")
                return
            
            # Detect different types of drift
            drift_results = []
            
            # Prediction drift detection
            prediction_drift = await self._detect_prediction_drift(current_data, reference_data)
            if prediction_drift:
                drift_results.append(prediction_drift)
            
            # Data drift detection (if input features are available)
            data_drift = await self._detect_data_drift(current_data, reference_data)
            if data_drift:
                drift_results.append(data_drift)
            
            # Performance drift detection
            perf_drift = await self._detect_performance_drift(model_name, model_version)
            if perf_drift:
                drift_results.append(perf_drift)
            
            # Handle detected drifts
            for drift_result in drift_results:
                if drift_result.is_drift_detected:
                    await self._handle_drift_detection(monitor_key, drift_result)
                    
        except Exception as e:
            logger.error(f"Error detecting model drift for {monitor_key}: {e}")
    
    async def _detect_prediction_drift(self, current_data: List[Dict], 
                                     reference_data: List[Dict]) -> Optional[DriftDetectionResult]:
        """Detect drift in model predictions using statistical tests"""
        try:
            current_predictions = [d.get('prediction', 0) for d in current_data]
            reference_predictions = [d.get('prediction', 0) for d in reference_data]
            
            # Handle different prediction types
            if isinstance(current_predictions[0], (list, np.ndarray)):
                # Multi-dimensional predictions - use first dimension or flatten
                current_predictions = [p[0] if isinstance(p, (list, np.ndarray)) else p for p in current_predictions]
                reference_predictions = [p[0] if isinstance(p, (list, np.ndarray)) else p for p in reference_predictions]
            
            # Convert to numeric
            current_predictions = [float(p) for p in current_predictions if p is not None]
            reference_predictions = [float(p) for p in reference_predictions if p is not None]
            
            if not current_predictions or not reference_predictions:
                return None
            
            # Use Kolmogorov-Smirnov test for distribution comparison
            ks_statistic, p_value = stats.ks_2samp(current_predictions, reference_predictions)
            
            is_drift_detected = (
                p_value < self.drift_detection_config["significance_level"] and
                ks_statistic > self.drift_detection_config["drift_threshold"]
            )
            
            current_period = (
                datetime.utcnow() - timedelta(days=1),
                datetime.utcnow()
            )
            reference_period = (
                datetime.utcnow() - timedelta(days=8),
                datetime.utcnow() - timedelta(days=1)
            )
            
            recommendation = ""
            if is_drift_detected:
                if np.mean(current_predictions) > np.mean(reference_predictions):
                    recommendation = "Model predictions have shifted upward significantly. Consider retraining."
                else:
                    recommendation = "Model predictions have shifted downward significantly. Consider retraining."
            
            return DriftDetectionResult(
                drift_type=DriftType.PREDICTION_DRIFT,
                is_drift_detected=is_drift_detected,
                drift_score=ks_statistic,
                p_value=p_value,
                reference_period=reference_period,
                current_period=current_period,
                affected_features=["predictions"],
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error detecting prediction drift: {e}")
            return None
    
    async def _detect_data_drift(self, current_data: List[Dict], 
                               reference_data: List[Dict]) -> Optional[DriftDetectionResult]:
        """Detect drift in input data distribution"""
        try:
            # Extract input features if available
            current_features = []
            reference_features = []
            
            for d in current_data:
                if 'input_features' in d:
                    current_features.append(d['input_features'])
                    
            for d in reference_data:
                if 'input_features' in d:
                    reference_features.append(d['input_features'])
            
            if not current_features or not reference_features:
                return None  # No input features available for drift detection
            
            # Convert to DataFrames for easier handling
            current_df = pd.DataFrame(current_features)
            reference_df = pd.DataFrame(reference_features)
            
            affected_features = []
            drift_scores = []
            
            # Check drift for each feature
            for column in current_df.columns:
                if current_df[column].dtype in ['int64', 'float64']:
                    # Numerical feature
                    current_values = current_df[column].dropna()
                    reference_values = reference_df[column].dropna()
                    
                    if len(current_values) > 10 and len(reference_values) > 10:
                        ks_stat, p_val = stats.ks_2samp(current_values, reference_values)
                        
                        if p_val < self.drift_detection_config["significance_level"]:
                            affected_features.append(column)
                            drift_scores.append(ks_stat)
                else:
                    # Categorical feature - use chi-square test
                    current_counts = current_df[column].value_counts()
                    reference_counts = reference_df[column].value_counts()
                    
                    # Align categories
                    all_categories = set(current_counts.index) | set(reference_counts.index)
                    current_aligned = [current_counts.get(cat, 0) for cat in all_categories]
                    reference_aligned = [reference_counts.get(cat, 0) for cat in all_categories]
                    
                    if sum(current_aligned) > 10 and sum(reference_aligned) > 10:
                        chi2_stat, p_val = stats.chisquare(current_aligned, reference_aligned)
                        
                        if p_val < self.drift_detection_config["significance_level"]:
                            affected_features.append(column)
                            drift_scores.append(chi2_stat / max(current_aligned + reference_aligned))
            
            is_drift_detected = len(affected_features) > 0
            overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0
            
            recommendation = ""
            if is_drift_detected:
                recommendation = f"Data drift detected in {len(affected_features)} features: {', '.join(affected_features[:3])}{'...' if len(affected_features) > 3 else ''}. Consider data validation and model retraining."
            
            return DriftDetectionResult(
                drift_type=DriftType.DATA_DRIFT,
                is_drift_detected=is_drift_detected,
                drift_score=overall_drift_score,
                p_value=np.min([0.05] + drift_scores) if drift_scores else 1.0,  # Conservative p-value
                reference_period=(datetime.utcnow() - timedelta(days=8), datetime.utcnow() - timedelta(days=1)),
                current_period=(datetime.utcnow() - timedelta(days=1), datetime.utcnow()),
                affected_features=affected_features,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error detecting data drift: {e}")
            return None
    
    async def _detect_performance_drift(self, model_name: str, 
                                      model_version: str) -> Optional[DriftDetectionResult]:
        """Detect drift in model performance over time"""
        try:
            # Get performance metrics over time
            current_perf = await self._get_performance_metrics(model_name, model_version, days=1)
            reference_perf = await self._get_performance_metrics(model_name, model_version, days_back=8, days_duration=7)
            
            if not current_perf or not reference_perf:
                return None
            
            # Compare key performance metrics
            perf_metrics = ['accuracy', 'latency_ms', 'error_rate', 'confidence']
            affected_metrics = []
            drift_detected = False
            
            for metric in perf_metrics:
                current_values = [p.get(metric) for p in current_perf if p.get(metric) is not None]
                reference_values = [p.get(metric) for p in reference_perf if p.get(metric) is not None]
                
                if len(current_values) < 10 or len(reference_values) < 10:
                    continue
                
                # Use t-test for performance metrics
                t_stat, p_value = stats.ttest_ind(current_values, reference_values)
                
                if p_value < self.drift_detection_config["significance_level"]:
                    effect_size = abs(np.mean(current_values) - np.mean(reference_values)) / np.sqrt(
                        (np.var(current_values) + np.var(reference_values)) / 2
                    )
                    
                    if effect_size > self.drift_detection_config["drift_threshold"]:
                        affected_metrics.append(metric)
                        drift_detected = True
            
            recommendation = ""
            if drift_detected:
                if 'accuracy' in affected_metrics:
                    recommendation = "Model accuracy has degraded significantly. Immediate attention required."
                elif 'latency_ms' in affected_metrics:
                    recommendation = "Model latency has increased significantly. Check infrastructure and model complexity."
                else:
                    recommendation = f"Performance drift detected in: {', '.join(affected_metrics)}. Monitor closely."
            
            return DriftDetectionResult(
                drift_type=DriftType.PERFORMANCE_DRIFT,
                is_drift_detected=drift_detected,
                drift_score=len(affected_metrics) / len(perf_metrics),  # Proportion of affected metrics
                p_value=0.01 if drift_detected else 0.5,
                reference_period=(datetime.utcnow() - timedelta(days=8), datetime.utcnow() - timedelta(days=1)),
                current_period=(datetime.utcnow() - timedelta(days=1), datetime.utcnow()),
                affected_features=affected_metrics,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error detecting performance drift: {e}")
            return None
    
    async def _get_recent_prediction_data(self, model_name: str, model_version: str, 
                                        days: int) -> List[Dict]:
        """Get recent prediction data for drift analysis"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            # Get data from Redis
            key = f"predictions:{model_name}:{model_version}"
            raw_data = await self.redis_client.lrange(key, 0, -1)
            
            predictions = []
            for raw_item in raw_data:
                try:
                    item = json.loads(raw_item)
                    item_time = datetime.fromisoformat(item.get('timestamp', start_time.isoformat()))
                    
                    if start_time <= item_time <= end_time:
                        predictions.append(item)
                except (json.JSONDecodeError, ValueError):
                    continue
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting recent prediction data: {e}")
            return []
    
    async def _get_historical_prediction_data(self, model_name: str, model_version: str,
                                           days_back: int, days_duration: int) -> List[Dict]:
        """Get historical prediction data for comparison"""
        try:
            end_time = datetime.utcnow() - timedelta(days=days_back)
            start_time = end_time - timedelta(days=days_duration)
            
            # Get data from Redis or database
            key = f"predictions:{model_name}:{model_version}"
            raw_data = await self.redis_client.lrange(key, 0, -1)
            
            predictions = []
            for raw_item in raw_data:
                try:
                    item = json.loads(raw_item)
                    item_time = datetime.fromisoformat(item.get('timestamp', start_time.isoformat()))
                    
                    if start_time <= item_time <= end_time:
                        predictions.append(item)
                except (json.JSONDecodeError, ValueError):
                    continue
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting historical prediction data: {e}")
            return []
    
    async def _get_performance_metrics(self, model_name: str, model_version: str,
                                     days: int = 1, days_back: int = 0,
                                     days_duration: int = None) -> List[Dict]:
        """Get performance metrics for a time period"""
        try:
            if days_duration is None:
                days_duration = days
            
            end_time = datetime.utcnow() - timedelta(days=days_back)
            start_time = end_time - timedelta(days=days_duration)
            
            key = f"performance:{model_name}:{model_version}"
            raw_data = await self.redis_client.lrange(key, 0, -1)
            
            metrics = []
            for raw_item in raw_data:
                try:
                    item = json.loads(raw_item)
                    item_time = datetime.fromisoformat(item.get('timestamp', start_time.isoformat()))
                    
                    if start_time <= item_time <= end_time:
                        metrics.append(item)
                except (json.JSONDecodeError, ValueError):
                    continue
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return []
    
    async def _handle_drift_detection(self, monitor_key: str, 
                                    drift_result: DriftDetectionResult) -> None:
        """Handle detected model drift"""
        try:
            model_name, model_version = monitor_key.split(":", 1)
            
            severity = AlertSeverity.HIGH if drift_result.drift_type == DriftType.PERFORMANCE_DRIFT else AlertSeverity.MEDIUM
            
            alert = ModelAlert(
                alert_id=f"drift_{monitor_key}_{drift_result.drift_type.value}_{int(datetime.utcnow().timestamp())}",
                model_name=model_name,
                model_version=model_version,
                alert_type=f"drift_detection_{drift_result.drift_type.value}",
                severity=severity,
                message=f"{drift_result.drift_type.value.replace('_', ' ').title()} detected. {drift_result.recommendation}",
                metric_values={
                    "drift_score": drift_result.drift_score,
                    "p_value": drift_result.p_value,
                    "affected_features_count": len(drift_result.affected_features)
                },
                threshold_values={
                    "significance_level": self.drift_detection_config["significance_level"],
                    "drift_threshold": self.drift_detection_config["drift_threshold"]
                },
                timestamp=datetime.utcnow()
            )
            
            await self._handle_alert(alert)
            
            # Store drift detection result
            drift_key = f"drift_results:{monitor_key}"
            await self.redis_client.lpush(drift_key, json.dumps(asdict(drift_result), default=str))
            await self.redis_client.ltrim(drift_key, 0, 99)  # Keep last 100 results
            
        except Exception as e:
            logger.error(f"Error handling drift detection: {e}")
    
    async def _handle_alert(self, alert: ModelAlert) -> None:
        """Handle model performance alert"""
        try:
            # Store alert
            alert_key = f"alerts:{alert.model_name}:{alert.model_version}"
            await self.redis_client.lpush(alert_key, json.dumps(asdict(alert), default=str))
            await self.redis_client.ltrim(alert_key, 0, 999)  # Keep last 1000 alerts
            
            # Call registered alert handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")
            
            logger.warning(f"Alert generated: {alert.alert_type} for {alert.model_name} v{alert.model_version} - {alert.message}")
            
        except Exception as e:
            logger.error(f"Error handling alert: {e}")
    
    async def _update_monitoring_status(self, monitor_key: str, metrics: Dict[str, Any]) -> None:
        """Update monitoring status"""
        try:
            status_key = f"monitor_status:{monitor_key}"
            status = {
                "last_check": datetime.utcnow().isoformat(),
                "sample_count": metrics["sample_count"],
                "avg_latency_ms": metrics["avg_latency_ms"],
                "error_rate": metrics["error_rate"],
                "status": "healthy" if metrics["error_rate"] < 0.05 else "degraded"
            }
            
            await self.redis_client.hset(status_key, mapping=status)
            
        except Exception as e:
            logger.error(f"Error updating monitoring status: {e}")
    
    def add_alert_handler(self, handler: Callable[[ModelAlert], None]) -> None:
        """Add custom alert handler"""
        self.alert_handlers.append(handler)
    
    async def stop_monitoring_model(self, model_name: str, model_version: str) -> None:
        """Stop monitoring a model"""
        try:
            monitor_key = f"{model_name}:{model_version}"
            
            if monitor_key in self.monitoring_tasks:
                task = self.monitoring_tasks[monitor_key]
                task.cancel()
                del self.monitoring_tasks[monitor_key]
                
                # Update configuration
                config_key = f"monitor_config:{monitor_key}"
                await self.redis_client.hset(config_key, "status", "stopped")
                
                logger.info(f"Stopped monitoring {monitor_key}")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    async def get_monitoring_status(self, model_name: str = None) -> Dict[str, Any]:
        """Get current monitoring status"""
        try:
            if model_name:
                # Get status for specific model
                pattern = f"monitor_status:{model_name}:*"
            else:
                # Get status for all models
                pattern = "monitor_status:*"
            
            keys = await self.redis_client.keys(pattern)
            status = {}
            
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                model_key = key_str.replace("monitor_status:", "")
                status[model_key] = await self.redis_client.hgetall(key)
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting monitoring status: {e}")
            return {}
    
    async def get_recent_alerts(self, model_name: str, model_version: str,
                              hours: int = 24) -> List[ModelAlert]:
        """Get recent alerts for a model"""
        try:
            alert_key = f"alerts:{model_name}:{model_version}"
            raw_alerts = await self.redis_client.lrange(alert_key, 0, -1)
            
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_alerts = []
            
            for raw_alert in raw_alerts:
                try:
                    alert_data = json.loads(raw_alert)
                    alert_time = datetime.fromisoformat(alert_data['timestamp'])
                    
                    if alert_time >= cutoff_time:
                        # Convert back to ModelAlert object
                        alert_data['timestamp'] = alert_time
                        alert_data['severity'] = AlertSeverity(alert_data['severity'])
                        recent_alerts.append(ModelAlert(**alert_data))
                        
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
            
            return recent_alerts
            
        except Exception as e:
            logger.error(f"Error getting recent alerts: {e}")
            return []
    
    async def cleanup(self) -> None:
        """Cleanup monitoring resources"""
        try:
            # Cancel all monitoring tasks
            for task in self.monitoring_tasks.values():
                task.cancel()
            
            # Wait for tasks to complete
            if self.monitoring_tasks:
                await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Model performance monitor cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")