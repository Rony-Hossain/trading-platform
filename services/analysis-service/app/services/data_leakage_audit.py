#!/usr/bin/env python3
"""
Data Leakage Audit Framework

Comprehensive audit system to detect and prevent look-ahead bias in financial 
forecasting pipelines. Focuses on temporal integrity, especially for options 
and event-based factors.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import re
import warnings
from enum import Enum
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class LeakageType(Enum):
    """Types of data leakage detected."""
    TEMPORAL_LEAKAGE = "temporal_leakage"  # Future data used
    FEATURE_LEAKAGE = "feature_leakage"   # Target-derived features
    PREPROCESSING_LEAKAGE = "preprocessing_leakage"  # Future stats in normalization
    OPTIONS_LEAKAGE = "options_leakage"   # Options data without proper lags
    EVENT_LEAKAGE = "event_leakage"       # Event data without proper lags
    INTRADAY_LEAKAGE = "intraday_leakage" # Same-day future information


@dataclass
class LeakageViolation:
    """Single leakage violation detected."""
    violation_id: str
    leakage_type: LeakageType
    feature_name: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    evidence: Dict[str, Any]
    recommendation: str
    timestamp: datetime


@dataclass
class AuditResult:
    """Results from data pipeline audit."""
    audit_timestamp: datetime
    dataset_info: Dict[str, Any]
    total_features: int
    violations: List[LeakageViolation]
    passed_features: List[str]
    audit_summary: Dict[str, Any]
    recommendations: List[str]
    compliance_score: float


class DataLeakageAuditor:
    """
    Comprehensive data leakage detection for financial forecasting.
    
    Performs systematic auditing of feature engineering pipelines to detect
    temporal violations, especially in options and event-based factors.
    """
    
    def __init__(self):
        self.violations = []
        self.feature_patterns = {
            'options': [
                r'iv_.*', r'.*_iv', r'implied_vol.*', r'option.*', 
                r'.*_skew', r'skew_.*', r'gamma.*', r'delta.*', 
                r'theta.*', r'vega.*', r'rho.*'
            ],
            'events': [
                r'earnings.*', r'event_.*', r'announcement.*', 
                r'.*_event', r'catalyst.*', r'news_.*'
            ],
            'intraday': [
                r'.*_intraday', r'minute_.*', r'hour_.*', r'tick_.*',
                r'bid_ask.*', r'spread_.*', r'volume_profile.*'
            ],
            'future': [
                r'.*_future', r'next_.*', r'forward_.*', r't\+\d+.*',
                r'lead_.*', r'ahead_.*'
            ]
        }
        
        # Temporal patterns that indicate potential leakage
        self.temporal_indicators = [
            'next', 'future', 'forward', 'ahead', 'lead', 'tomorrow',
            't+1', 't+2', 'following', 'subsequent', 'post'
        ]
        
    def audit_temporal_integrity(self, 
                                data: pd.DataFrame, 
                                target_col: str = 'target',
                                forecast_date_col: str = None) -> List[LeakageViolation]:
        """
        Audit temporal integrity of features.
        
        Args:
            data: DataFrame with datetime index
            target_col: Target column name
            forecast_date_col: Column indicating forecast date (optional)
            
        Returns:
            List of temporal leakage violations
        """
        violations = []
        
        logger.info("Auditing temporal integrity")
        
        # Check if data has proper datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            violations.append(LeakageViolation(
                violation_id=f"temp_001",
                leakage_type=LeakageType.TEMPORAL_LEAKAGE,
                feature_name="index",
                severity="critical",
                description="Data does not have proper datetime index",
                evidence={"index_type": str(type(data.index))},
                recommendation="Convert index to datetime for proper temporal analysis",
                timestamp=datetime.now()
            ))
            return violations
        
        # Check for future data in features
        for col in data.columns:
            if col == target_col:
                continue
                
            # Check feature naming for temporal indicators
            col_lower = col.lower()
            for indicator in self.temporal_indicators:
                if indicator in col_lower:
                    violations.append(LeakageViolation(
                        violation_id=f"temp_{len(violations)+1:03d}",
                        leakage_type=LeakageType.TEMPORAL_LEAKAGE,
                        feature_name=col,
                        severity="high",
                        description=f"Feature name contains temporal indicator '{indicator}'",
                        evidence={"indicator": indicator, "feature_name": col},
                        recommendation=f"Verify feature uses only t-1 or earlier data",
                        timestamp=datetime.now()
                    ))
        
        # Check for same-timestamp correlation violations
        if target_col in data.columns:
            for col in data.columns:
                if col == target_col:
                    continue
                    
                try:
                    # Check correlation with target at same timestamp
                    same_time_corr = data[col].corr(data[target_col])
                    
                    # Check correlation with lagged target (this should be lower)
                    if len(data) > 1:
                        lagged_target = data[target_col].shift(1)
                        lagged_corr = data[col].corr(lagged_target)
                        
                        # If correlation with current target is much higher than lagged,
                        # it might indicate look-ahead bias
                        if abs(same_time_corr) > 0.8 and abs(same_time_corr) > abs(lagged_corr) * 2:
                            violations.append(LeakageViolation(
                                violation_id=f"temp_{len(violations)+1:03d}",
                                leakage_type=LeakageType.TEMPORAL_LEAKAGE,
                                feature_name=col,
                                severity="high",
                                description="High correlation with same-period target suggests potential leakage",
                                evidence={
                                    "same_time_correlation": same_time_corr,
                                    "lagged_correlation": lagged_corr,
                                    "correlation_ratio": abs(same_time_corr) / max(abs(lagged_corr), 0.01)
                                },
                                recommendation="Verify feature is calculated using only historical data",
                                timestamp=datetime.now()
                            ))
                            
                except Exception as e:
                    logger.warning(f"Could not calculate correlation for {col}: {e}")
        
        return violations
    
    def audit_options_features(self, 
                             data: pd.DataFrame,
                             options_features: List[str] = None) -> List[LeakageViolation]:
        """
        Audit options-related features for proper lag structure.
        
        Args:
            data: DataFrame with options features
            options_features: List of options feature names (auto-detected if None)
            
        Returns:
            List of options leakage violations
        """
        violations = []
        
        logger.info("Auditing options features")
        
        # Auto-detect options features if not provided
        if options_features is None:
            options_features = []
            for col in data.columns:
                for pattern in self.feature_patterns['options']:
                    if re.match(pattern, col.lower()):
                        options_features.append(col)
                        break
        
        logger.info(f"Found {len(options_features)} options features to audit")
        
        for feature in options_features:
            if feature not in data.columns:
                continue
                
            # Check for proper naming indicating historical data
            feature_lower = feature.lower()
            has_lag_indicator = any(indicator in feature_lower for indicator in [
                'lag', 'prev', 'yesterday', 't-1', 'historical', 'past'
            ])
            
            if not has_lag_indicator:
                violations.append(LeakageViolation(
                    violation_id=f"opt_{len(violations)+1:03d}",
                    leakage_type=LeakageType.OPTIONS_LEAKAGE,
                    feature_name=feature,
                    severity="medium",
                    description="Options feature lacks clear lag indicator in naming",
                    evidence={"feature_name": feature},
                    recommendation="Add lag indicator (e.g., '_lag1', '_prev') to clarify temporal relationship",
                    timestamp=datetime.now()
                ))
            
            # Check for suspicious patterns in options data
            try:
                if not data[feature].empty:
                    # Check for values that change too dramatically (might indicate real-time data)
                    daily_changes = data[feature].pct_change().abs()
                    extreme_changes = (daily_changes > 2.0).sum()  # >200% daily change
                    
                    if extreme_changes > len(data) * 0.05:  # More than 5% of days
                        violations.append(LeakageViolation(
                            violation_id=f"opt_{len(violations)+1:03d}",
                            leakage_type=LeakageType.OPTIONS_LEAKAGE,
                            feature_name=feature,
                            severity="medium",
                            description="Options feature shows excessive volatility suggesting real-time data",
                            evidence={
                                "extreme_change_days": int(extreme_changes),
                                "total_days": len(data),
                                "extreme_change_percentage": extreme_changes / len(data)
                            },
                            recommendation="Verify options data uses previous day's close values",
                            timestamp=datetime.now()
                        ))
                    
                    # Check for weekend/holiday data issues
                    if isinstance(data.index, pd.DatetimeIndex):
                        weekend_mask = data.index.weekday >= 5  # Saturday = 5, Sunday = 6
                        weekend_data = data.loc[weekend_mask, feature]
                        
                        if not weekend_data.empty and weekend_data.notna().sum() > 0:
                            violations.append(LeakageViolation(
                                violation_id=f"opt_{len(violations)+1:03d}",
                                leakage_type=LeakageType.OPTIONS_LEAKAGE,
                                feature_name=feature,
                                severity="low",
                                description="Options feature has data on weekends (markets closed)",
                                evidence={
                                    "weekend_data_points": int(weekend_data.notna().sum()),
                                    "sample_weekend_values": weekend_data.dropna().head(3).tolist()
                                },
                                recommendation="Ensure options data only includes market days",
                                timestamp=datetime.now()
                            ))
                            
            except Exception as e:
                logger.warning(f"Could not analyze options feature {feature}: {e}")
        
        return violations
    
    def audit_event_features(self, 
                           data: pd.DataFrame,
                           event_features: List[str] = None) -> List[LeakageViolation]:
        """
        Audit event-based features for proper temporal alignment.
        
        Args:
            data: DataFrame with event features
            event_features: List of event feature names (auto-detected if None)
            
        Returns:
            List of event leakage violations
        """
        violations = []
        
        logger.info("Auditing event features")
        
        # Auto-detect event features if not provided
        if event_features is None:
            event_features = []
            for col in data.columns:
                for pattern in self.feature_patterns['events']:
                    if re.match(pattern, col.lower()):
                        event_features.append(col)
                        break
        
        logger.info(f"Found {len(event_features)} event features to audit")
        
        for feature in event_features:
            if feature not in data.columns:
                continue
            
            feature_lower = feature.lower()
            
            # Check for future event indicators
            future_indicators = ['next', 'upcoming', 'future', 'scheduled', 'expected']
            has_future_indicator = any(indicator in feature_lower for indicator in future_indicators)
            
            if has_future_indicator:
                violations.append(LeakageViolation(
                    violation_id=f"evt_{len(violations)+1:03d}",
                    leakage_type=LeakageType.EVENT_LEAKAGE,
                    feature_name=feature,
                    severity="critical",
                    description="Event feature contains future indicator",
                    evidence={"indicators_found": [ind for ind in future_indicators if ind in feature_lower]},
                    recommendation="Use only historical events or ensure future events are properly lagged",
                    timestamp=datetime.now()
                ))
            
            # Check for event announcement vs event date confusion
            if 'announcement' in feature_lower or 'scheduled' in feature_lower:
                # This could be okay if it's announcement date, not event date
                violations.append(LeakageViolation(
                    violation_id=f"evt_{len(violations)+1:03d}",
                    leakage_type=LeakageType.EVENT_LEAKAGE,
                    feature_name=feature,
                    severity="low",
                    description="Event feature may need clarification on announcement vs event timing",
                    evidence={"feature_name": feature},
                    recommendation="Clarify if this uses announcement date (OK) or actual event date (potential leakage)",
                    timestamp=datetime.now()
                ))
            
            # Check for binary event patterns that might indicate post-event data
            try:
                if data[feature].dtype in ['bool', 'int64'] and data[feature].nunique() <= 10:
                    # Check if events cluster around target spikes
                    if 'target' in data.columns:
                        target_changes = data['target'].abs()
                        high_target_days = target_changes > target_changes.quantile(0.9)
                        
                        event_on_high_days = data.loc[high_target_days, feature].mean()
                        event_on_normal_days = data.loc[~high_target_days, feature].mean()
                        
                        if event_on_high_days > event_on_normal_days * 3:  # 3x more frequent
                            violations.append(LeakageViolation(
                                violation_id=f"evt_{len(violations)+1:03d}",
                                leakage_type=LeakageType.EVENT_LEAKAGE,
                                feature_name=feature,
                                severity="high",
                                description="Event feature strongly correlates with high target days",
                                evidence={
                                    "event_rate_high_days": event_on_high_days,
                                    "event_rate_normal_days": event_on_normal_days,
                                    "ratio": event_on_high_days / max(event_on_normal_days, 0.001)
                                },
                                recommendation="Verify events are not derived from same-day price movements",
                                timestamp=datetime.now()
                            ))
                            
            except Exception as e:
                logger.warning(f"Could not analyze event feature {feature}: {e}")
        
        return violations
    
    def audit_preprocessing_leakage(self, 
                                  data: pd.DataFrame,
                                  preprocessing_info: Dict[str, Any] = None) -> List[LeakageViolation]:
        """
        Audit preprocessing steps for look-ahead bias.
        
        Args:
            data: Processed DataFrame
            preprocessing_info: Information about preprocessing steps
            
        Returns:
            List of preprocessing leakage violations
        """
        violations = []
        
        logger.info("Auditing preprocessing for leakage")
        
        # Check for features that look like they used global statistics
        numeric_features = data.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_features:
            try:
                if data[feature].empty or data[feature].isna().all():
                    continue
                
                # Check if feature is perfectly normalized (mean~0, std~1)
                feature_mean = data[feature].mean()
                feature_std = data[feature].std()
                
                if abs(feature_mean) < 0.01 and abs(feature_std - 1.0) < 0.01:
                    violations.append(LeakageViolation(
                        violation_id=f"prep_{len(violations)+1:03d}",
                        leakage_type=LeakageType.PREPROCESSING_LEAKAGE,
                        feature_name=feature,
                        severity="medium",
                        description="Feature appears globally normalized (potential future data in stats)",
                        evidence={
                            "mean": feature_mean,
                            "std": feature_std
                        },
                        recommendation="Use rolling statistics or train-test split for normalization",
                        timestamp=datetime.now()
                    ))
                
                # Check for impossible values that might indicate calculation errors
                if feature_std == 0 and data[feature].nunique() > 1:
                    violations.append(LeakageViolation(
                        violation_id=f"prep_{len(violations)+1:03d}",
                        leakage_type=LeakageType.PREPROCESSING_LEAKAGE,
                        feature_name=feature,
                        severity="medium",
                        description="Feature has zero standard deviation but multiple values",
                        evidence={
                            "std": feature_std,
                            "unique_values": data[feature].nunique()
                        },
                        recommendation="Check feature calculation for errors",
                        timestamp=datetime.now()
                    ))
                
            except Exception as e:
                logger.warning(f"Could not analyze preprocessing for {feature}: {e}")
        
        return violations
    
    def audit_feature_engineering(self, 
                                data: pd.DataFrame,
                                target_col: str = 'target') -> List[LeakageViolation]:
        """
        Audit feature engineering for target-derived features.
        
        Args:
            data: DataFrame with engineered features
            target_col: Target column name
            
        Returns:
            List of feature leakage violations
        """
        violations = []
        
        logger.info("Auditing feature engineering")
        
        if target_col not in data.columns:
            return violations
        
        # Check for features that are too perfectly correlated with target
        for feature in data.columns:
            if feature == target_col:
                continue
                
            try:
                correlation = data[feature].corr(data[target_col])
                
                if abs(correlation) > 0.95:
                    violations.append(LeakageViolation(
                        violation_id=f"feat_{len(violations)+1:03d}",
                        leakage_type=LeakageType.FEATURE_LEAKAGE,
                        feature_name=feature,
                        severity="critical",
                        description="Feature has extremely high correlation with target",
                        evidence={"correlation": correlation},
                        recommendation="Verify feature is not derived from target variable",
                        timestamp=datetime.now()
                    ))
                
                # Check for features that might be transformations of the target
                if abs(correlation) > 0.8:
                    # Check if feature is a simple transformation of target
                    feature_clean = data[feature].dropna()
                    target_clean = data[target_col].dropna()
                    
                    if len(feature_clean) > 10 and len(target_clean) > 10:
                        # Check if feature = target * constant
                        if feature_clean.std() > 0 and target_clean.std() > 0:
                            ratio = (feature_clean / target_clean.reindex(feature_clean.index)).dropna()
                            if len(ratio) > 5 and ratio.std() < 0.01:  # Very consistent ratio
                                violations.append(LeakageViolation(
                                    violation_id=f"feat_{len(violations)+1:03d}",
                                    leakage_type=LeakageType.FEATURE_LEAKAGE,
                                    feature_name=feature,
                                    severity="high",
                                    description="Feature appears to be a linear transformation of target",
                                    evidence={
                                        "correlation": correlation,
                                        "ratio_std": ratio.std(),
                                        "ratio_mean": ratio.mean()
                                    },
                                    recommendation="Remove feature or verify it's not derived from target",
                                    timestamp=datetime.now()
                                ))
                
            except Exception as e:
                logger.warning(f"Could not analyze feature correlation for {feature}: {e}")
        
        return violations
    
    async def comprehensive_audit(self, 
                                data: pd.DataFrame,
                                target_col: str = 'target',
                                options_features: List[str] = None,
                                event_features: List[str] = None,
                                preprocessing_info: Dict[str, Any] = None) -> AuditResult:
        """
        Perform comprehensive data leakage audit.
        
        Args:
            data: DataFrame to audit
            target_col: Target column name
            options_features: List of options features (auto-detected if None)
            event_features: List of event features (auto-detected if None)
            preprocessing_info: Preprocessing information
            
        Returns:
            Comprehensive audit results
        """
        audit_start = datetime.now()
        logger.info(f"Starting comprehensive data leakage audit for dataset with {len(data)} rows, {len(data.columns)} columns")
        
        all_violations = []
        
        # 1. Temporal integrity audit
        temporal_violations = self.audit_temporal_integrity(data, target_col)
        all_violations.extend(temporal_violations)
        
        # 2. Options features audit
        options_violations = self.audit_options_features(data, options_features)
        all_violations.extend(options_violations)
        
        # 3. Event features audit
        event_violations = self.audit_event_features(data, event_features)
        all_violations.extend(event_violations)
        
        # 4. Preprocessing audit
        preprocessing_violations = self.audit_preprocessing_leakage(data, preprocessing_info)
        all_violations.extend(preprocessing_violations)
        
        # 5. Feature engineering audit
        feature_violations = self.audit_feature_engineering(data, target_col)
        all_violations.extend(feature_violations)
        
        # Calculate compliance score
        total_features = len(data.columns)
        violation_count = len(all_violations)
        compliance_score = max(0.0, (total_features - violation_count) / total_features)
        
        # Identify passed features
        violated_features = {v.feature_name for v in all_violations}
        passed_features = [col for col in data.columns if col not in violated_features]
        
        # Generate summary statistics
        violation_by_type = {}
        violation_by_severity = {}
        
        for violation in all_violations:
            # Count by type
            vtype = violation.leakage_type.value
            violation_by_type[vtype] = violation_by_type.get(vtype, 0) + 1
            
            # Count by severity
            severity = violation.severity
            violation_by_severity[severity] = violation_by_severity.get(severity, 0) + 1
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_violations, data)
        
        # Dataset info
        dataset_info = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'date_range': {
                'start': data.index.min().isoformat() if isinstance(data.index, pd.DatetimeIndex) else None,
                'end': data.index.max().isoformat() if isinstance(data.index, pd.DatetimeIndex) else None
            },
            'missing_data_percentage': (data.isna().sum().sum() / (len(data) * len(data.columns))) * 100,
            'numeric_features': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(data.select_dtypes(exclude=[np.number]).columns)
        }
        
        audit_summary = {
            'total_violations': len(all_violations),
            'violations_by_type': violation_by_type,
            'violations_by_severity': violation_by_severity,
            'compliance_score': compliance_score,
            'audit_duration_seconds': (datetime.now() - audit_start).total_seconds()
        }
        
        result = AuditResult(
            audit_timestamp=audit_start,
            dataset_info=dataset_info,
            total_features=total_features,
            violations=all_violations,
            passed_features=passed_features,
            audit_summary=audit_summary,
            recommendations=recommendations,
            compliance_score=compliance_score
        )
        
        logger.info(f"Audit completed: {len(all_violations)} violations found, compliance score: {compliance_score:.2%}")
        
        return result
    
    def _generate_recommendations(self, 
                                violations: List[LeakageViolation],
                                data: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on violations."""
        recommendations = []
        
        violation_types = {v.leakage_type for v in violations}
        severities = {v.severity for v in violations}
        
        if LeakageType.TEMPORAL_LEAKAGE in violation_types:
            recommendations.append(
                "Implement strict temporal validation: ensure all features use only t-1 or earlier data"
            )
        
        if LeakageType.OPTIONS_LEAKAGE in violation_types:
            recommendations.append(
                "Review options data pipeline: use previous day's closing IV and Greeks, never intraday"
            )
        
        if LeakageType.EVENT_LEAKAGE in violation_types:
            recommendations.append(
                "Audit event timing: use announcement dates, not event dates, with appropriate lags"
            )
        
        if LeakageType.PREPROCESSING_LEAKAGE in violation_types:
            recommendations.append(
                "Use time-aware preprocessing: rolling statistics or proper train/test splits for normalization"
            )
        
        if LeakageType.FEATURE_LEAKAGE in violation_types:
            recommendations.append(
                "Remove target-derived features: audit feature engineering for circular dependencies"
            )
        
        if 'critical' in severities:
            recommendations.append(
                "CRITICAL: Address critical violations immediately before model deployment"
            )
        
        if not violations:
            recommendations.append(
                "Excellent: No leakage violations detected. Data pipeline appears temporally sound."
            )
        
        return recommendations
    
    async def save_audit_artifacts(self, 
                                 result: AuditResult,
                                 output_dir: str = "artifacts/leakage_audit") -> Path:
        """Save audit results and artifacts."""
        
        timestamp = result.audit_timestamp.strftime("%Y%m%d_%H%M%S")
        artifacts_path = Path(output_dir) / f"leakage_audit_{timestamp}"
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        with open(artifacts_path / "audit_results.json", "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        # Save violations details
        violations_data = [asdict(v) for v in result.violations]
        with open(artifacts_path / "violations.json", "w") as f:
            json.dump(violations_data, f, indent=2, default=str)
        
        # Save compliance report
        compliance_report = {
            "audit_summary": result.audit_summary,
            "compliance_score": result.compliance_score,
            "total_features": result.total_features,
            "passed_features": len(result.passed_features),
            "failed_features": len(result.violations),
            "recommendations": result.recommendations
        }
        
        with open(artifacts_path / "compliance_report.json", "w") as f:
            json.dump(compliance_report, f, indent=2)
        
        # Create detailed CSV report
        if result.violations:
            violations_df = pd.DataFrame([
                {
                    'Feature': v.feature_name,
                    'Violation_Type': v.leakage_type.value,
                    'Severity': v.severity,
                    'Description': v.description,
                    'Recommendation': v.recommendation
                }
                for v in result.violations
            ])
            violations_df.to_csv(artifacts_path / "violations_report.csv", index=False)
        
        logger.info(f"Audit artifacts saved to {artifacts_path}")
        return artifacts_path


# Demo function for testing
async def run_leakage_audit_demo():
    """Demo of the data leakage audit framework."""
    print("Data Leakage Audit Framework Demo")
    print("=" * 60)
    
    # Generate synthetic financial dataset with intentional leakage
    np.random.seed(42)
    n_samples = 500
    
    # Create datetime index
    start_date = datetime(2022, 1, 1)
    date_index = pd.date_range(start=start_date, periods=n_samples, freq='D')
    
    # Create synthetic data with various leakage patterns
    data = pd.DataFrame(index=date_index)
    
    # Target: daily returns
    data['target'] = np.random.normal(0.001, 0.02, n_samples)
    
    # Good features (no leakage)
    data['price_lag1'] = np.random.normal(100, 10, n_samples)
    data['volume_lag1'] = np.random.lognormal(15, 0.5, n_samples)
    data['rsi_lag1'] = 50 + np.random.randn(n_samples) * 15
    
    # Features with leakage (intentional for demo)
    data['future_return'] = data['target'].shift(-1)  # Future data!
    data['next_earnings'] = np.random.binomial(1, 0.1, n_samples)  # Future events
    data['same_day_iv'] = np.random.normal(0.3, 0.1, n_samples)  # Same day options data
    data['target_derived'] = data['target'] * 2 + 0.001  # Target transformation
    data['normalized_feature'] = (np.random.randn(n_samples) - 0) / 1  # Global normalization
    
    # Options features with mixed compliance
    data['iv_lag1'] = np.random.normal(0.25, 0.05, n_samples)  # Good
    data['option_volume'] = np.random.lognormal(10, 1, n_samples)  # Questionable naming
    
    print(f"Generated dataset: {len(data)} samples, {len(data.columns)} features")
    print(f"Includes intentional leakage for demonstration")
    print()
    
    # Initialize auditor
    auditor = DataLeakageAuditor()
    
    # Run comprehensive audit
    print("Running comprehensive leakage audit...")
    audit_result = await auditor.comprehensive_audit(
        data=data,
        target_col='target'
    )
    
    # Display results
    print(f"Audit completed in {audit_result.audit_summary['audit_duration_seconds']:.2f} seconds")
    print(f"Compliance Score: {audit_result.compliance_score:.1%}")
    print()
    
    print("Violations Found:")
    print("-" * 40)
    
    for violation in audit_result.violations:
        severity_icon = {
            'critical': '🔴',
            'high': '🟠', 
            'medium': '🟡',
            'low': '🟢'
        }.get(violation.severity, '⚪')
        
        print(f"{severity_icon} {violation.severity.upper()}: {violation.feature_name}")
        print(f"   Type: {violation.leakage_type.value}")
        print(f"   Issue: {violation.description}")
        print(f"   Fix: {violation.recommendation}")
        print()
    
    print("Violation Summary:")
    for vtype, count in audit_result.audit_summary['violations_by_type'].items():
        print(f"  {vtype}: {count}")
    print()
    
    print("Recommendations:")
    for i, rec in enumerate(audit_result.recommendations, 1):
        print(f"  {i}. {rec}")
    print()
    
    print(f"Clean features ({len(audit_result.passed_features)}):")
    for feature in audit_result.passed_features:
        print(f"  ✅ {feature}")
    
    print()
    print("Leakage audit demo completed!")


if __name__ == "__main__":
    asyncio.run(run_leakage_audit_demo())