"""
Model Drift Monitoring System

Implements Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests
for detecting model degradation due to data distribution changes.

Key Features:
- PSI for feature distribution monitoring
- KS test for data distribution skew detection
- Automated alerting when drift exceeds thresholds
- Integration with model retraining workflows
- Production model health monitoring
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import json
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer
import warnings

logger = logging.getLogger(__name__)

class DriftSeverity(Enum):
    """Drift severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DriftStatus(Enum):
    """Overall drift status"""
    STABLE = "stable"
    MINOR_DRIFT = "minor_drift"
    SIGNIFICANT_DRIFT = "significant_drift"
    CRITICAL_DRIFT = "critical_drift"

@dataclass
class DriftThresholds:
    """Configurable thresholds for drift detection"""
    psi_low: float = 0.1      # Low drift threshold for PSI
    psi_medium: float = 0.2   # Medium drift threshold for PSI
    psi_high: float = 0.25    # High drift threshold for PSI
    ks_low: float = 0.1       # Low drift threshold for KS statistic
    ks_medium: float = 0.2    # Medium drift threshold for KS statistic
    ks_high: float = 0.3      # High drift threshold for KS statistic
    ks_p_value: float = 0.05  # P-value threshold for KS test significance
    min_samples: int = 100    # Minimum samples required for reliable drift detection

@dataclass
class FeatureDriftResult:
    """Results of drift analysis for a single feature"""
    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_p_value: float
    drift_severity: DriftSeverity
    is_significant: bool
    baseline_stats: Dict[str, float]
    current_stats: Dict[str, float]
    bin_comparison: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_name': self.feature_name,
            'psi_score': self.psi_score,
            'ks_statistic': self.ks_statistic,
            'ks_p_value': self.ks_p_value,
            'drift_severity': self.drift_severity.value,
            'is_significant': self.is_significant,
            'baseline_stats': self.baseline_stats,
            'current_stats': self.current_stats,
            'bin_comparison': self.bin_comparison
        }

@dataclass
class ModelDriftReport:
    """Comprehensive model drift analysis report"""
    model_name: str
    model_version: str
    symbol: str
    analysis_timestamp: datetime
    baseline_period: str
    current_period: str
    overall_drift_status: DriftStatus
    total_features: int
    drifted_features: int
    feature_results: List[FeatureDriftResult]
    summary_statistics: Dict[str, float]
    recommendations: List[str]
    alert_triggered: bool
    next_analysis_due: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'symbol': self.symbol,
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'baseline_period': self.baseline_period,
            'current_period': self.current_period,
            'overall_drift_status': self.overall_drift_status.value,
            'total_features': self.total_features,
            'drifted_features': self.drifted_features,
            'feature_results': [f.to_dict() for f in self.feature_results],
            'summary_statistics': self.summary_statistics,
            'recommendations': self.recommendations,
            'alert_triggered': self.alert_triggered,
            'next_analysis_due': self.next_analysis_due.isoformat()
        }

class PopulationStabilityIndex:
    """Calculate Population Stability Index (PSI) for feature drift detection"""
    
    def __init__(self, n_bins: int = 10, bin_strategy: str = 'quantile'):
        self.n_bins = n_bins
        self.bin_strategy = bin_strategy
        self.discretizer = None
        
    def fit(self, baseline_data: np.ndarray) -> 'PopulationStabilityIndex':
        """Fit the binning strategy on baseline data"""
        baseline_data = baseline_data.reshape(-1, 1)
        
        if self.bin_strategy == 'quantile':
            self.discretizer = KBinsDiscretizer(
                n_bins=self.n_bins, 
                encode='ordinal', 
                strategy='quantile',
                subsample=None
            )
        else:  # uniform
            self.discretizer = KBinsDiscretizer(
                n_bins=self.n_bins, 
                encode='ordinal', 
                strategy='uniform'
            )
        
        # Handle edge cases
        if len(np.unique(baseline_data)) < self.n_bins:
            self.n_bins = len(np.unique(baseline_data))
            self.discretizer.n_bins = self.n_bins
            
        self.discretizer.fit(baseline_data)
        return self
    
    def calculate_psi(self, baseline_data: np.ndarray, current_data: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Calculate PSI between baseline and current data distributions"""
        
        if self.discretizer is None:
            self.fit(baseline_data)
        
        # Transform data to bins
        baseline_data = baseline_data.reshape(-1, 1)
        current_data = current_data.reshape(-1, 1)
        
        baseline_binned = self.discretizer.transform(baseline_data).flatten()
        current_binned = self.discretizer.transform(current_data).flatten()
        
        # Calculate distributions
        baseline_dist = np.bincount(baseline_binned.astype(int), minlength=self.n_bins) / len(baseline_binned)
        current_dist = np.bincount(current_binned.astype(int), minlength=self.n_bins) / len(current_binned)
        
        # Avoid division by zero
        baseline_dist = np.where(baseline_dist == 0, 1e-7, baseline_dist)
        current_dist = np.where(current_dist == 0, 1e-7, current_dist)
        
        # Calculate PSI
        psi = np.sum((current_dist - baseline_dist) * np.log(current_dist / baseline_dist))
        
        # Detailed bin comparison
        bin_comparison = {
            'bin_edges': self.discretizer.bin_edges_[0].tolist() if hasattr(self.discretizer, 'bin_edges_') else None,
            'baseline_distribution': baseline_dist.tolist(),
            'current_distribution': current_dist.tolist(),
            'bin_contributions': ((current_dist - baseline_dist) * np.log(current_dist / baseline_dist)).tolist()
        }
        
        return psi, bin_comparison

class ModelDriftMonitor:
    """Complete model drift monitoring system"""
    
    def __init__(self, thresholds: Optional[DriftThresholds] = None):
        self.thresholds = thresholds or DriftThresholds()
        self.psi_calculator = PopulationStabilityIndex()
        self.baseline_data_cache = {}  # Cache baseline data for models
        
    async def store_baseline_data(self, model_name: str, symbol: str, features: pd.DataFrame) -> bool:
        """Store baseline feature distributions for a model"""
        try:
            cache_key = f"{model_name}_{symbol}"
            
            # Calculate baseline statistics for each feature
            baseline_stats = {}
            for column in features.columns:
                if pd.api.types.is_numeric_dtype(features[column]):
                    baseline_stats[column] = {
                        'mean': float(features[column].mean()),
                        'std': float(features[column].std()),
                        'min': float(features[column].min()),
                        'max': float(features[column].max()),
                        'median': float(features[column].median()),
                        'q25': float(features[column].quantile(0.25)),
                        'q75': float(features[column].quantile(0.75)),
                        'skewness': float(features[column].skew()),
                        'kurtosis': float(features[column].kurtosis()),
                        'data': features[column].dropna().values.tolist()
                    }
            
            self.baseline_data_cache[cache_key] = {
                'model_name': model_name,
                'symbol': symbol,
                'timestamp': datetime.now(),
                'features': baseline_stats,
                'sample_count': len(features)
            }
            
            logger.info(f"Stored baseline data for {model_name}_{symbol} with {len(features)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error storing baseline data for {model_name}_{symbol}: {e}")
            return False
    
    def _calculate_feature_drift(self, feature_name: str, baseline_data: np.ndarray, 
                                current_data: np.ndarray, baseline_stats: Dict[str, float]) -> FeatureDriftResult:
        """Calculate drift metrics for a single feature"""
        
        # Calculate current statistics
        current_stats = {
            'mean': float(np.mean(current_data)),
            'std': float(np.std(current_data)),
            'min': float(np.min(current_data)),
            'max': float(np.max(current_data)),
            'median': float(np.median(current_data)),
            'q25': float(np.percentile(current_data, 25)),
            'q75': float(np.percentile(current_data, 75)),
            'skewness': float(stats.skew(current_data)),
            'kurtosis': float(stats.kurtosis(current_data))
        }
        
        # Calculate PSI
        psi_score, bin_comparison = self.psi_calculator.calculate_psi(baseline_data, current_data)
        
        # Calculate KS test
        ks_statistic, ks_p_value = stats.ks_2samp(baseline_data, current_data)
        
        # Determine drift severity
        drift_severity = self._assess_drift_severity(psi_score, ks_statistic, ks_p_value)
        is_significant = ks_p_value < self.thresholds.ks_p_value
        
        return FeatureDriftResult(
            feature_name=feature_name,
            psi_score=psi_score,
            ks_statistic=ks_statistic,
            ks_p_value=ks_p_value,
            drift_severity=drift_severity,
            is_significant=is_significant,
            baseline_stats=baseline_stats,
            current_stats=current_stats,
            bin_comparison=bin_comparison
        )
    
    def _assess_drift_severity(self, psi_score: float, ks_statistic: float, ks_p_value: float) -> DriftSeverity:
        """Assess overall drift severity based on PSI and KS metrics"""
        
        # PSI-based assessment
        if psi_score >= self.thresholds.psi_high:
            psi_severity = DriftSeverity.HIGH
        elif psi_score >= self.thresholds.psi_medium:
            psi_severity = DriftSeverity.MEDIUM
        elif psi_score >= self.thresholds.psi_low:
            psi_severity = DriftSeverity.LOW
        else:
            psi_severity = DriftSeverity.LOW
        
        # KS-based assessment
        if ks_statistic >= self.thresholds.ks_high and ks_p_value < self.thresholds.ks_p_value:
            ks_severity = DriftSeverity.HIGH
        elif ks_statistic >= self.thresholds.ks_medium and ks_p_value < self.thresholds.ks_p_value:
            ks_severity = DriftSeverity.MEDIUM
        elif ks_statistic >= self.thresholds.ks_low and ks_p_value < self.thresholds.ks_p_value:
            ks_severity = DriftSeverity.LOW
        else:
            ks_severity = DriftSeverity.LOW
        
        # Take the maximum severity
        severities = [psi_severity, ks_severity]
        severity_order = [DriftSeverity.LOW, DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL]
        
        max_severity = max(severities, key=lambda x: severity_order.index(x))
        
        # Critical threshold: either metric extremely high
        if psi_score > 0.5 or (ks_statistic > 0.5 and ks_p_value < 0.001):
            return DriftSeverity.CRITICAL
        
        return max_severity
    
    async def analyze_model_drift(self, model_name: str, symbol: str, current_features: pd.DataFrame,
                                 model_version: str = "latest") -> ModelDriftReport:
        """Perform comprehensive drift analysis for a model"""
        
        try:
            cache_key = f"{model_name}_{symbol}"
            
            if cache_key not in self.baseline_data_cache:
                raise ValueError(f"No baseline data found for {model_name}_{symbol}. Store baseline data first.")
            
            baseline_cache = self.baseline_data_cache[cache_key]
            baseline_features = baseline_cache['features']
            
            # Analyze drift for each feature
            feature_results = []
            drifted_features = 0
            
            for feature_name in current_features.columns:
                if feature_name in baseline_features and pd.api.types.is_numeric_dtype(current_features[feature_name]):
                    
                    # Get baseline and current data
                    baseline_data = np.array(baseline_features[feature_name]['data'])
                    current_data = current_features[feature_name].dropna().values
                    
                    if len(current_data) < self.thresholds.min_samples:
                        logger.warning(f"Insufficient current data for {feature_name}: {len(current_data)} < {self.thresholds.min_samples}")
                        continue
                    
                    # Calculate drift for this feature
                    drift_result = self._calculate_feature_drift(
                        feature_name, baseline_data, current_data, baseline_features[feature_name]
                    )
                    
                    feature_results.append(drift_result)
                    
                    if drift_result.drift_severity in [DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                        drifted_features += 1
            
            # Calculate summary statistics
            if feature_results:
                psi_scores = [f.psi_score for f in feature_results]
                ks_statistics = [f.ks_statistic for f in feature_results]
                
                summary_statistics = {
                    'avg_psi': float(np.mean(psi_scores)),
                    'max_psi': float(np.max(psi_scores)),
                    'avg_ks': float(np.mean(ks_statistics)),
                    'max_ks': float(np.max(ks_statistics)),
                    'drift_percentage': float(drifted_features / len(feature_results)) if feature_results else 0.0,
                    'features_analyzed': len(feature_results)
                }
            else:
                summary_statistics = {
                    'avg_psi': 0.0,
                    'max_psi': 0.0,
                    'avg_ks': 0.0,
                    'max_ks': 0.0,
                    'drift_percentage': 0.0,
                    'features_analyzed': 0
                }
            
            # Determine overall drift status
            overall_drift_status = self._determine_overall_drift_status(feature_results, summary_statistics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(overall_drift_status, feature_results, summary_statistics)
            
            # Determine if alert should be triggered
            alert_triggered = overall_drift_status in [DriftStatus.SIGNIFICANT_DRIFT, DriftStatus.CRITICAL_DRIFT]
            
            # Next analysis due (daily for critical, weekly for significant, monthly for stable)
            if overall_drift_status == DriftStatus.CRITICAL_DRIFT:
                next_analysis = datetime.now() + timedelta(days=1)
            elif overall_drift_status == DriftStatus.SIGNIFICANT_DRIFT:
                next_analysis = datetime.now() + timedelta(days=7)
            else:
                next_analysis = datetime.now() + timedelta(days=30)
            
            report = ModelDriftReport(
                model_name=model_name,
                model_version=model_version,
                symbol=symbol,
                analysis_timestamp=datetime.now(),
                baseline_period=f"Since {baseline_cache['timestamp'].strftime('%Y-%m-%d')}",
                current_period=f"{len(current_features)} recent samples",
                overall_drift_status=overall_drift_status,
                total_features=len(feature_results),
                drifted_features=drifted_features,
                feature_results=feature_results,
                summary_statistics=summary_statistics,
                recommendations=recommendations,
                alert_triggered=alert_triggered,
                next_analysis_due=next_analysis
            )
            
            logger.info(f"Drift analysis completed for {model_name}_{symbol}: {overall_drift_status.value}")
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing model drift for {model_name}_{symbol}: {e}")
            raise
    
    def _determine_overall_drift_status(self, feature_results: List[FeatureDriftResult], 
                                       summary_stats: Dict[str, float]) -> DriftStatus:
        """Determine overall model drift status"""
        
        if not feature_results:
            return DriftStatus.STABLE
        
        critical_features = sum(1 for f in feature_results if f.drift_severity == DriftSeverity.CRITICAL)
        high_features = sum(1 for f in feature_results if f.drift_severity == DriftSeverity.HIGH)
        medium_features = sum(1 for f in feature_results if f.drift_severity == DriftSeverity.MEDIUM)
        
        total_features = len(feature_results)
        drift_percentage = summary_stats['drift_percentage']
        
        # Critical: Any critical features or >50% high drift
        if critical_features > 0 or (high_features / total_features) > 0.5:
            return DriftStatus.CRITICAL_DRIFT
        
        # Significant: >30% medium+ drift or >20% high drift
        if drift_percentage > 0.3 or (high_features / total_features) > 0.2:
            return DriftStatus.SIGNIFICANT_DRIFT
        
        # Minor: >10% medium+ drift
        if drift_percentage > 0.1:
            return DriftStatus.MINOR_DRIFT
        
        return DriftStatus.STABLE
    
    def _generate_recommendations(self, drift_status: DriftStatus, feature_results: List[FeatureDriftResult],
                                 summary_stats: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on drift analysis"""
        
        recommendations = []
        
        if drift_status == DriftStatus.CRITICAL_DRIFT:
            recommendations.extend([
                "🚨 CRITICAL: Immediate model retraining required",
                "🔄 Consider rolling back to previous model version",
                "📊 Investigate data quality and collection processes",
                "⚡ Increase monitoring frequency to daily"
            ])
        
        elif drift_status == DriftStatus.SIGNIFICANT_DRIFT:
            recommendations.extend([
                "⚠️ Schedule model retraining within 1-2 weeks",
                "🔍 Analyze most drifted features for root causes",
                "📈 Consider updating feature engineering pipeline",
                "🔄 Increase monitoring frequency to weekly"
            ])
        
        elif drift_status == DriftStatus.MINOR_DRIFT:
            recommendations.extend([
                "📋 Monitor closely and prepare for potential retraining",
                "🔍 Investigate specific drifted features",
                "📊 Continue current monitoring schedule"
            ])
        
        else:  # STABLE
            recommendations.extend([
                "✅ Model performance appears stable",
                "📊 Continue regular monitoring schedule",
                "🔄 Consider extending monitoring intervals"
            ])
        
        # Feature-specific recommendations
        if feature_results:
            high_drift_features = [f.feature_name for f in feature_results if f.drift_severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]]
            
            if high_drift_features:
                recommendations.append(f"🎯 High drift detected in: {', '.join(high_drift_features[:3])}")
        
        # Data quality recommendations
        if summary_stats['max_psi'] > 0.5:
            recommendations.append("📈 Extreme PSI detected - check for data collection changes")
        
        if summary_stats['max_ks'] > 0.5:
            recommendations.append("📊 Large distribution shift detected - validate data sources")
        
        return recommendations
    
    async def trigger_retraining_workflow(self, model_name: str, symbol: str, drift_report: ModelDriftReport) -> Dict[str, Any]:
        """Trigger model retraining workflow when drift exceeds thresholds"""
        
        try:
            logger.info(f"Triggering retraining workflow for {model_name}_{symbol}")
            
            # In a real implementation, this would:
            # 1. Queue a retraining job
            # 2. Notify the MLOps team
            # 3. Update model registry
            # 4. Schedule validation
            
            workflow_config = {
                'trigger_reason': 'model_drift_detected',
                'drift_status': drift_report.overall_drift_status.value,
                'model_name': model_name,
                'symbol': symbol,
                'drift_severity': drift_report.overall_drift_status.value,
                'drifted_features': drift_report.drifted_features,
                'total_features': drift_report.total_features,
                'summary_stats': drift_report.summary_statistics,
                'triggered_at': datetime.now().isoformat(),
                'priority': 'high' if drift_report.overall_drift_status == DriftStatus.CRITICAL_DRIFT else 'medium'
            }
            
            # Mock workflow response
            return {
                'workflow_id': f"retrain_{model_name}_{symbol}_{int(datetime.now().timestamp())}",
                'status': 'queued',
                'estimated_completion': (datetime.now() + timedelta(hours=2)).isoformat(),
                'config': workflow_config
            }
            
        except Exception as e:
            logger.error(f"Error triggering retraining workflow: {e}")
            return {'error': str(e), 'status': 'failed'}

    async def get_drift_summary(self, model_name: Optional[str] = None, 
                               symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of drift monitoring status across models"""
        
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_models_monitored': len(self.baseline_data_cache),
                'models': {},
                'overall_status': 'healthy',
                'alerts_count': 0
            }
            
            for cache_key, baseline_data in self.baseline_data_cache.items():
                model_info = {
                    'model_name': baseline_data['model_name'],
                    'symbol': baseline_data['symbol'],
                    'baseline_date': baseline_data['timestamp'].isoformat(),
                    'sample_count': baseline_data['sample_count'],
                    'features_count': len(baseline_data['features']),
                    'last_analysis': 'not_analyzed',
                    'status': 'monitoring'
                }
                
                summary['models'][cache_key] = model_info
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting drift summary: {e}")
            return {'error': str(e)}

    def configure_thresholds(self, new_thresholds: Dict[str, float]) -> bool:
        """Update drift detection thresholds"""
        
        try:
            for key, value in new_thresholds.items():
                if hasattr(self.thresholds, key):
                    setattr(self.thresholds, key, value)
                    logger.info(f"Updated threshold {key} to {value}")
                else:
                    logger.warning(f"Unknown threshold parameter: {key}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error configuring thresholds: {e}")
            return False