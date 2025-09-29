"""
Advanced data quality validation for sentiment data.
Implements Great Expectations-style validation, PSI monitoring, and distribution shift detection.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from sqlalchemy.orm import Session
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json

logger = logging.getLogger(__name__)

class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ValidationStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"

class DriftType(str, Enum):
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    DISTRIBUTION_SHIFT = "distribution_shift"
    QUALITY_DEGRADATION = "quality_degradation"

@dataclass
class ValidationRule:
    name: str
    description: str
    metric_name: str
    threshold: float
    operator: str  # 'gt', 'lt', 'gte', 'lte', 'eq', 'ne'
    severity: AlertSeverity
    enabled: bool = True

@dataclass
class ValidationResult:
    rule_name: str
    status: ValidationStatus
    observed_value: float
    threshold: float
    message: str
    severity: AlertSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PSIResult:
    feature_name: str
    psi_score: float
    interpretation: str
    bucket_details: List[Dict[str, float]]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DistributionShiftResult:
    feature_name: str
    shift_detected: bool
    drift_type: DriftType
    drift_score: float
    p_value: float
    test_statistic: float
    statistical_test: str
    severity: AlertSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    
@dataclass
class DataQualityReport:
    validation_timestamp: datetime
    symbol: str
    total_records: int
    validation_results: List[ValidationResult]
    psi_results: List[PSIResult]
    drift_results: List[DistributionShiftResult]
    overall_status: ValidationStatus
    quality_score: float  # 0-1 overall quality score
    recommendations: List[str]
    data_characteristics: Dict[str, Any]

class DataQualityValidator:
    """
    Comprehensive data quality validation for sentiment data.
    Implements schema validation, distribution monitoring, and drift detection.
    """
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        self.baseline_stats = {}  # Store baseline statistics for PSI calculation
        self.scaler = StandardScaler()
        
    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize default validation rules for sentiment data"""
        return [
            # Sentiment score validation
            ValidationRule(
                name="sentiment_score_range",
                description="Sentiment scores must be between -1 and 1",
                metric_name="sentiment_score_out_of_range_pct",
                threshold=0.05,  # Max 5% out of range
                operator="lt",
                severity=AlertSeverity.HIGH
            ),
            ValidationRule(
                name="sentiment_score_completeness",
                description="Sentiment scores must be present",
                metric_name="sentiment_score_null_pct",
                threshold=0.02,  # Max 2% null
                operator="lt",
                severity=AlertSeverity.MEDIUM
            ),
            # Confidence score validation
            ValidationRule(
                name="confidence_range",
                description="Confidence scores must be between 0 and 1",
                metric_name="confidence_out_of_range_pct",
                threshold=0.05,
                operator="lt",
                severity=AlertSeverity.HIGH
            ),
            ValidationRule(
                name="low_confidence_threshold",
                description="Too many low confidence predictions",
                metric_name="low_confidence_pct",
                threshold=0.30,  # Max 30% low confidence (< 0.3)
                operator="lt",
                severity=AlertSeverity.MEDIUM
            ),
            # Content quality validation
            ValidationRule(
                name="content_completeness",
                description="Content must be present",
                metric_name="content_null_pct",
                threshold=0.01,  # Max 1% null content
                operator="lt",
                severity=AlertSeverity.CRITICAL
            ),
            ValidationRule(
                name="content_length",
                description="Content should have reasonable length",
                metric_name="content_too_short_pct",
                threshold=0.10,  # Max 10% too short (< 10 chars)
                operator="lt",
                severity=AlertSeverity.LOW
            ),
            # Source diversity validation
            ValidationRule(
                name="source_diversity",
                description="Maintain diversity across sources",
                metric_name="single_source_dominance_pct",
                threshold=0.80,  # No single source > 80%
                operator="lt",
                severity=AlertSeverity.MEDIUM
            ),
            # Duplicate detection
            ValidationRule(
                name="duplicate_content",
                description="Detect excessive duplicate content",
                metric_name="duplicate_content_pct",
                threshold=0.15,  # Max 15% duplicates
                operator="lt",
                severity=AlertSeverity.HIGH
            ),
            # Temporal consistency
            ValidationRule(
                name="temporal_gaps",
                description="Detect unusual temporal gaps in data",
                metric_name="temporal_gap_hours",
                threshold=6.0,  # Max 6 hour gaps
                operator="lt",
                severity=AlertSeverity.MEDIUM
            )
        ]
    
    async def validate_data_quality(self, data: pd.DataFrame, symbol: str,
                                  baseline_data: Optional[pd.DataFrame] = None) -> DataQualityReport:
        """
        Comprehensive data quality validation for sentiment data.
        
        Args:
            data: Current sentiment data to validate
            symbol: Stock symbol being analyzed
            baseline_data: Historical baseline data for comparison
            
        Returns:
            Comprehensive data quality report
        """
        try:
            logger.info(f"Starting data quality validation for {symbol}")
            
            # Compute basic metrics
            metrics = self._compute_data_metrics(data)
            
            # Run validation rules
            validation_results = []
            for rule in self.validation_rules:
                if rule.enabled and rule.metric_name in metrics:
                    result = self._evaluate_rule(rule, metrics[rule.metric_name])
                    validation_results.append(result)
            
            # PSI analysis if baseline available
            psi_results = []
            if baseline_data is not None and len(baseline_data) > 0:
                psi_results = await self._compute_psi_analysis(data, baseline_data)
            
            # Distribution shift detection
            drift_results = []
            if baseline_data is not None and len(baseline_data) > 0:
                drift_results = await self._detect_distribution_shifts(data, baseline_data)
            
            # Compute overall quality score
            quality_score = self._compute_quality_score(validation_results, psi_results, drift_results)
            
            # Determine overall status
            overall_status = self._determine_overall_status(validation_results, drift_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(validation_results, psi_results, drift_results)
            
            # Data characteristics
            data_characteristics = self._compute_data_characteristics(data)
            
            return DataQualityReport(
                validation_timestamp=datetime.now(),
                symbol=symbol,
                total_records=len(data),
                validation_results=validation_results,
                psi_results=psi_results,
                drift_results=drift_results,
                overall_status=overall_status,
                quality_score=quality_score,
                recommendations=recommendations,
                data_characteristics=data_characteristics
            )
            
        except Exception as e:
            logger.error(f"Data quality validation failed for {symbol}: {e}")
            raise
    
    def _compute_data_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Compute various data quality metrics"""
        metrics = {}
        
        if len(data) == 0:
            return metrics
        
        try:
            # Sentiment score metrics
            if 'sentiment_score' in data.columns:
                sentiment_scores = data['sentiment_score']
                metrics['sentiment_score_null_pct'] = sentiment_scores.isnull().sum() / len(data)
                
                valid_scores = sentiment_scores.dropna()
                if len(valid_scores) > 0:
                    out_of_range = ((valid_scores < -1) | (valid_scores > 1)).sum()
                    metrics['sentiment_score_out_of_range_pct'] = out_of_range / len(valid_scores)
            
            # Confidence metrics
            if 'confidence' in data.columns:
                confidence_scores = data['confidence']
                metrics['confidence_null_pct'] = confidence_scores.isnull().sum() / len(data)
                
                valid_confidence = confidence_scores.dropna()
                if len(valid_confidence) > 0:
                    out_of_range = ((valid_confidence < 0) | (valid_confidence > 1)).sum()
                    metrics['confidence_out_of_range_pct'] = out_of_range / len(valid_confidence)
                    
                    low_confidence = (valid_confidence < 0.3).sum()
                    metrics['low_confidence_pct'] = low_confidence / len(valid_confidence)
            
            # Content metrics
            if 'content' in data.columns:
                content_data = data['content']
                metrics['content_null_pct'] = content_data.isnull().sum() / len(data)
                
                valid_content = content_data.dropna()
                if len(valid_content) > 0:
                    too_short = (valid_content.str.len() < 10).sum()
                    metrics['content_too_short_pct'] = too_short / len(valid_content)
                    
                    # Duplicate content detection
                    duplicates = valid_content.duplicated().sum()
                    metrics['duplicate_content_pct'] = duplicates / len(valid_content)
            
            # Source diversity
            if 'source' in data.columns:
                source_counts = data['source'].value_counts()
                if len(source_counts) > 0:
                    max_source_pct = source_counts.iloc[0] / len(data)
                    metrics['single_source_dominance_pct'] = max_source_pct
            
            # Temporal gaps
            if 'timestamp' in data.columns:
                timestamps = pd.to_datetime(data['timestamp']).sort_values()
                if len(timestamps) > 1:
                    time_diffs = timestamps.diff().dropna()
                    max_gap_hours = time_diffs.max().total_seconds() / 3600
                    metrics['temporal_gap_hours'] = max_gap_hours
            
        except Exception as e:
            logger.warning(f"Error computing metrics: {e}")
        
        return metrics
    
    def _evaluate_rule(self, rule: ValidationRule, observed_value: float) -> ValidationResult:
        """Evaluate a single validation rule"""
        operators = {
            'gt': lambda x, y: x > y,
            'lt': lambda x, y: x < y,
            'gte': lambda x, y: x >= y,
            'lte': lambda x, y: x <= y,
            'eq': lambda x, y: abs(x - y) < 1e-9,
            'ne': lambda x, y: abs(x - y) >= 1e-9
        }
        
        operator_func = operators.get(rule.operator)
        if not operator_func:
            raise ValueError(f"Unknown operator: {rule.operator}")
        
        passed = operator_func(observed_value, rule.threshold)
        
        if passed:
            status = ValidationStatus.PASS
            message = f"{rule.description}: PASSED ({observed_value:.4f} {rule.operator} {rule.threshold})"
        else:
            status = ValidationStatus.FAIL
            message = f"{rule.description}: FAILED ({observed_value:.4f} not {rule.operator} {rule.threshold})"
        
        return ValidationResult(
            rule_name=rule.name,
            status=status,
            observed_value=observed_value,
            threshold=rule.threshold,
            message=message,
            severity=rule.severity
        )
    
    async def _compute_psi_analysis(self, current_data: pd.DataFrame, 
                                  baseline_data: pd.DataFrame) -> List[PSIResult]:
        """Compute Population Stability Index for key features"""
        psi_results = []
        
        # Features to monitor for PSI
        features_to_monitor = ['sentiment_score', 'confidence']
        
        for feature in features_to_monitor:
            if feature in current_data.columns and feature in baseline_data.columns:
                try:
                    psi_result = self._calculate_psi(
                        baseline_data[feature].dropna(),
                        current_data[feature].dropna(),
                        feature
                    )
                    psi_results.append(psi_result)
                except Exception as e:
                    logger.warning(f"PSI calculation failed for {feature}: {e}")
        
        return psi_results
    
    def _calculate_psi(self, baseline: pd.Series, current: pd.Series, feature_name: str) -> PSIResult:
        """Calculate Population Stability Index"""
        # Create buckets (10 quantiles)
        try:
            # Use baseline to create bucket boundaries
            bucket_boundaries = np.percentile(baseline, np.linspace(0, 100, 11))
            bucket_boundaries[0] = -np.inf
            bucket_boundaries[-1] = np.inf
            
            # Calculate distributions
            baseline_dist = pd.cut(baseline, bucket_boundaries, include_lowest=True).value_counts(normalize=True).sort_index()
            current_dist = pd.cut(current, bucket_boundaries, include_lowest=True).value_counts(normalize=True).sort_index()
            
            # Ensure both distributions have same index
            baseline_dist = baseline_dist.reindex(current_dist.index, fill_value=0.001)  # Small value to avoid log(0)
            current_dist = current_dist.fillna(0.001)
            
            # Calculate PSI
            psi_values = (current_dist - baseline_dist) * np.log(current_dist / baseline_dist)
            psi_score = psi_values.sum()
            
            # Interpret PSI
            if psi_score < 0.1:
                interpretation = "No significant change"
            elif psi_score < 0.2:
                interpretation = "Minor change - monitor"
            else:
                interpretation = "Major change - investigate"
            
            # Create bucket details
            bucket_details = []
            for i, (bucket, baseline_pct, current_pct, psi_val) in enumerate(
                zip(baseline_dist.index, baseline_dist.values, current_dist.values, psi_values.values)
            ):
                bucket_details.append({
                    'bucket': str(bucket),
                    'baseline_pct': float(baseline_pct),
                    'current_pct': float(current_pct),
                    'psi_contribution': float(psi_val)
                })
            
            return PSIResult(
                feature_name=feature_name,
                psi_score=float(psi_score),
                interpretation=interpretation,
                bucket_details=bucket_details
            )
            
        except Exception as e:
            logger.error(f"PSI calculation error for {feature_name}: {e}")
            return PSIResult(
                feature_name=feature_name,
                psi_score=999.0,  # Error indicator
                interpretation="Calculation failed",
                bucket_details=[]
            )
    
    async def _detect_distribution_shifts(self, current_data: pd.DataFrame,
                                        baseline_data: pd.DataFrame) -> List[DistributionShiftResult]:
        """Detect distribution shifts using statistical tests"""
        drift_results = []
        
        # Features to monitor for drift
        numerical_features = ['sentiment_score', 'confidence']
        categorical_features = ['sentiment_label', 'source']
        
        # Test numerical features
        for feature in numerical_features:
            if feature in current_data.columns and feature in baseline_data.columns:
                try:
                    result = self._test_numerical_drift(
                        baseline_data[feature].dropna(),
                        current_data[feature].dropna(),
                        feature
                    )
                    drift_results.append(result)
                except Exception as e:
                    logger.warning(f"Numerical drift test failed for {feature}: {e}")
        
        # Test categorical features
        for feature in categorical_features:
            if feature in current_data.columns and feature in baseline_data.columns:
                try:
                    result = self._test_categorical_drift(
                        baseline_data[feature].dropna(),
                        current_data[feature].dropna(),
                        feature
                    )
                    drift_results.append(result)
                except Exception as e:
                    logger.warning(f"Categorical drift test failed for {feature}: {e}")
        
        return drift_results
    
    def _test_numerical_drift(self, baseline: pd.Series, current: pd.Series, 
                            feature_name: str) -> DistributionShiftResult:
        """Test for drift in numerical features using Kolmogorov-Smirnov test"""
        try:
            # Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(baseline, current)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((baseline.var() + current.var()) / 2)
            cohens_d = abs(baseline.mean() - current.mean()) / pooled_std if pooled_std > 0 else 0
            
            # Determine significance and severity
            alpha = 0.05
            shift_detected = p_value < alpha
            
            if not shift_detected:
                severity = AlertSeverity.LOW
            elif cohens_d < 0.2:
                severity = AlertSeverity.LOW
            elif cohens_d < 0.5:
                severity = AlertSeverity.MEDIUM
            elif cohens_d < 0.8:
                severity = AlertSeverity.HIGH
            else:
                severity = AlertSeverity.CRITICAL
            
            return DistributionShiftResult(
                feature_name=feature_name,
                shift_detected=shift_detected,
                drift_type=DriftType.DISTRIBUTION_SHIFT,
                drift_score=float(cohens_d),
                p_value=float(p_value),
                test_statistic=float(ks_statistic),
                statistical_test="Kolmogorov-Smirnov",
                severity=severity
            )
            
        except Exception as e:
            logger.error(f"Numerical drift test error for {feature_name}: {e}")
            return DistributionShiftResult(
                feature_name=feature_name,
                shift_detected=True,
                drift_type=DriftType.QUALITY_DEGRADATION,
                drift_score=999.0,
                p_value=0.0,
                test_statistic=999.0,
                statistical_test="Failed",
                severity=AlertSeverity.CRITICAL
            )
    
    def _test_categorical_drift(self, baseline: pd.Series, current: pd.Series,
                              feature_name: str) -> DistributionShiftResult:
        """Test for drift in categorical features using Chi-square test"""
        try:
            # Get value counts and align
            baseline_counts = baseline.value_counts()
            current_counts = current.value_counts()
            
            # Align categories
            all_categories = set(baseline_counts.index) | set(current_counts.index)
            baseline_aligned = baseline_counts.reindex(all_categories, fill_value=1)  # Small count to avoid issues
            current_aligned = current_counts.reindex(all_categories, fill_value=1)
            
            # Chi-square test
            contingency_table = np.array([baseline_aligned, current_aligned])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Cramér's V (effect size)
            n = contingency_table.sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            # Determine significance and severity
            alpha = 0.05
            shift_detected = p_value < alpha
            
            if not shift_detected:
                severity = AlertSeverity.LOW
            elif cramers_v < 0.1:
                severity = AlertSeverity.LOW
            elif cramers_v < 0.3:
                severity = AlertSeverity.MEDIUM
            elif cramers_v < 0.5:
                severity = AlertSeverity.HIGH
            else:
                severity = AlertSeverity.CRITICAL
            
            return DistributionShiftResult(
                feature_name=feature_name,
                shift_detected=shift_detected,
                drift_type=DriftType.DISTRIBUTION_SHIFT,
                drift_score=float(cramers_v),
                p_value=float(p_value),
                test_statistic=float(chi2),
                statistical_test="Chi-square",
                severity=severity
            )
            
        except Exception as e:
            logger.error(f"Categorical drift test error for {feature_name}: {e}")
            return DistributionShiftResult(
                feature_name=feature_name,
                shift_detected=True,
                drift_type=DriftType.QUALITY_DEGRADATION,
                drift_score=999.0,
                p_value=0.0,
                test_statistic=999.0,
                statistical_test="Failed",
                severity=AlertSeverity.CRITICAL
            )
    
    def _compute_quality_score(self, validation_results: List[ValidationResult],
                             psi_results: List[PSIResult],
                             drift_results: List[DistributionShiftResult]) -> float:
        """Compute overall quality score (0-1)"""
        try:
            total_score = 0.0
            total_weight = 0.0
            
            # Validation results (60% weight)
            validation_weight = 0.6
            validation_pass_rate = sum(1 for r in validation_results if r.status == ValidationStatus.PASS) / max(len(validation_results), 1)
            total_score += validation_pass_rate * validation_weight
            total_weight += validation_weight
            
            # PSI results (20% weight)
            if psi_results:
                psi_weight = 0.2
                good_psi_rate = sum(1 for r in psi_results if r.psi_score < 0.2) / len(psi_results)
                total_score += good_psi_rate * psi_weight
                total_weight += psi_weight
            
            # Drift results (20% weight)
            if drift_results:
                drift_weight = 0.2
                no_drift_rate = sum(1 for r in drift_results if not r.shift_detected) / len(drift_results)
                total_score += no_drift_rate * drift_weight
                total_weight += drift_weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Quality score calculation error: {e}")
            return 0.0
    
    def _determine_overall_status(self, validation_results: List[ValidationResult],
                                drift_results: List[DistributionShiftResult]) -> ValidationStatus:
        """Determine overall validation status"""
        # Check for critical failures
        critical_failures = [r for r in validation_results 
                           if r.status == ValidationStatus.FAIL and r.severity == AlertSeverity.CRITICAL]
        if critical_failures:
            return ValidationStatus.FAIL
        
        # Check for critical drift
        critical_drift = [r for r in drift_results 
                         if r.shift_detected and r.severity == AlertSeverity.CRITICAL]
        if critical_drift:
            return ValidationStatus.FAIL
        
        # Check for high severity issues
        high_severity_issues = [r for r in validation_results 
                              if r.status == ValidationStatus.FAIL and r.severity == AlertSeverity.HIGH]
        high_severity_drift = [r for r in drift_results 
                             if r.shift_detected and r.severity == AlertSeverity.HIGH]
        
        if high_severity_issues or high_severity_drift:
            return ValidationStatus.WARNING
        
        # Check for any failures
        any_failures = [r for r in validation_results if r.status == ValidationStatus.FAIL]
        any_drift = [r for r in drift_results if r.shift_detected]
        
        if any_failures or any_drift:
            return ValidationStatus.WARNING
        
        return ValidationStatus.PASS
    
    def _generate_recommendations(self, validation_results: List[ValidationResult],
                                psi_results: List[PSIResult],
                                drift_results: List[DistributionShiftResult]) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        # Validation-based recommendations
        failed_validations = [r for r in validation_results if r.status == ValidationStatus.FAIL]
        
        for result in failed_validations:
            if "sentiment_score" in result.rule_name:
                recommendations.append("Review sentiment analysis model - scores outside expected range")
            elif "confidence" in result.rule_name:
                recommendations.append("Investigate model confidence - consider retraining or feature engineering")
            elif "content" in result.rule_name:
                recommendations.append("Improve content quality validation at data ingestion")
            elif "source" in result.rule_name:
                recommendations.append("Diversify data sources to reduce bias")
            elif "duplicate" in result.rule_name:
                recommendations.append("Implement better duplicate detection and deduplication")
            elif "temporal" in result.rule_name:
                recommendations.append("Check data collection pipeline for interruptions")
        
        # PSI-based recommendations
        high_psi = [r for r in psi_results if r.psi_score >= 0.2]
        for result in high_psi:
            recommendations.append(f"Investigate {result.feature_name} distribution shift - may need model retraining")
        
        # Drift-based recommendations
        significant_drift = [r for r in drift_results if r.shift_detected and r.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]]
        for result in significant_drift:
            recommendations.append(f"Significant drift detected in {result.feature_name} - consider data pipeline review")
        
        # General recommendations if no specific issues
        if not recommendations:
            recommendations.append("Data quality looks good - continue monitoring")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _compute_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary characteristics of the data"""
        characteristics = {}
        
        try:
            characteristics['total_records'] = len(data)
            characteristics['date_range'] = {
                'start': data['timestamp'].min().isoformat() if 'timestamp' in data.columns else None,
                'end': data['timestamp'].max().isoformat() if 'timestamp' in data.columns else None
            }
            
            if 'sentiment_score' in data.columns:
                scores = data['sentiment_score'].dropna()
                characteristics['sentiment_stats'] = {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'min': float(scores.min()),
                    'max': float(scores.max()),
                    'null_count': int(data['sentiment_score'].isnull().sum())
                }
            
            if 'source' in data.columns:
                characteristics['source_distribution'] = data['source'].value_counts().to_dict()
            
            if 'sentiment_label' in data.columns:
                characteristics['label_distribution'] = data['sentiment_label'].value_counts().to_dict()
                
        except Exception as e:
            logger.warning(f"Error computing data characteristics: {e}")
        
        return characteristics