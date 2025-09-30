"""
Feature Contracts - Point-in-Time (PIT) Enforcement System

This module provides comprehensive PIT contract validation and enforcement
for all features used in the trading platform. Ensures data lineage integrity
and prevents look-ahead bias in model training and production.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import yaml
import json
import os
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Feature categorization"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental" 
    SENTIMENT = "sentiment"
    MACRO = "macro"
    OPTIONS = "options"
    DERIVED = "derived"
    EVENT = "event"

class PIIClassification(Enum):
    """Personal information classification"""
    NONE = "none"
    MASKED = "masked"
    ANONYMIZED = "anonymized"
    SENSITIVE = "sensitive"

class ValidationSeverity(Enum):
    """Contract violation severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class VendorSLA:
    """Data vendor service level agreement"""
    availability: float = 99.9  # Uptime percentage
    latency_minutes: int = 15   # Maximum delivery delay
    quality: float = 99.5       # Accuracy/completeness percentage

@dataclass
class RevisionPolicy:
    """Data revision and correction policy"""
    revision_frequency: str = "never"  # never, daily, weekly, monthly
    revision_window_days: int = 0      # How far back revisions occur
    notification_method: str = "none"  # none, email, webhook, alert

@dataclass
class ValidationRules:
    """Feature validation and monitoring rules"""
    valid_range: Optional[tuple] = None          # (min, max) values
    null_handling: str = "reject"               # reject, forward_fill, interpolate
    outlier_detection: str = "3_sigma"          # Method for anomaly detection
    monitoring_alerts: List[str] = field(default_factory=list)

@dataclass
class FeatureContract:
    """Complete feature contract specification"""
    # Identification
    feature_name: str
    feature_type: FeatureType
    data_source: str
    version: str
    
    # Point-in-Time Constraints  
    as_of_ts_rule: str           # When feature becomes available
    effective_ts_rule: str       # When underlying event occurred
    arrival_latency_minutes: int # Expected delay
    point_in_time_rule: str      # Specific PIT constraints
    
    # Data Quality & SLA
    vendor_sla: VendorSLA
    revision_policy: RevisionPolicy
    
    # Business Logic
    computation_logic: str
    dependencies: List[str] = field(default_factory=list)
    lookback_period_days: int = 30
    update_frequency: str = "daily"
    
    # Validation & Monitoring
    validation_rules: ValidationRules = field(default_factory=ValidationRules)
    
    # Compliance & Audit
    pii_classification: PIIClassification = PIIClassification.NONE
    regulatory_notes: str = ""
    audit_trail: str = "git_commits"
    retention_policy: str = "7_years"
    
    # Metadata
    created_by: str = ""
    created_at: Optional[datetime] = None
    approved_by: str = ""
    approved_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None

    def __post_init__(self):
        """Validate contract after initialization"""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_modified is None:
            self.last_modified = datetime.utcnow()

@dataclass
class ContractViolation:
    """Represents a contract validation violation"""
    feature_name: str
    violation_type: str
    severity: ValidationSeverity
    message: str
    actual_value: Any = None
    expected_value: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

class FeatureContractValidator:
    """Validates feature contracts and enforces PIT constraints"""
    
    def __init__(self, contracts_dir: str = "docs/feature-contracts"):
        self.contracts_dir = Path(contracts_dir)
        self.contracts: Dict[str, FeatureContract] = {}
        self.load_contracts()
        
    def load_contracts(self) -> None:
        """Load all feature contracts from directory"""
        if not self.contracts_dir.exists():
            logger.warning(f"Contracts directory not found: {self.contracts_dir}")
            return
            
        pattern = self.contracts_dir / "*.yml"
        contract_files = list(self.contracts_dir.glob("*.yml")) + list(self.contracts_dir.glob("*.yaml"))
        
        for contract_file in contract_files:
            try:
                with open(contract_file, 'r') as f:
                    contract_data = yaml.safe_load(f)
                    
                contract = self._parse_contract(contract_data)
                self.contracts[contract.feature_name] = contract
                logger.info(f"Loaded contract for feature: {contract.feature_name}")
                
            except Exception as e:
                logger.error(f"Failed to load contract {contract_file}: {e}")
    
    def _parse_contract(self, data: Dict) -> FeatureContract:
        """Parse contract data into FeatureContract object"""
        # Parse nested objects
        vendor_sla = VendorSLA(**data.get('vendor_sla', {}))
        revision_policy = RevisionPolicy(**data.get('revision_policy', {}))
        validation_rules = ValidationRules(**data.get('validation_rules', {}))
        
        # Convert string enums
        feature_type = FeatureType(data['feature_type'])
        pii_classification = PIIClassification(data.get('pii_classification', 'none'))
        
        return FeatureContract(
            feature_name=data['feature_name'],
            feature_type=feature_type,
            data_source=data['data_source'],
            version=data['version'],
            as_of_ts_rule=data['as_of_ts_rule'],
            effective_ts_rule=data['effective_ts_rule'],
            arrival_latency_minutes=data['arrival_latency_minutes'],
            point_in_time_rule=data['point_in_time_rule'],
            vendor_sla=vendor_sla,
            revision_policy=revision_policy,
            computation_logic=data['computation_logic'],
            dependencies=data.get('dependencies', []),
            lookback_period_days=data.get('lookback_period_days', 30),
            update_frequency=data.get('update_frequency', 'daily'),
            validation_rules=validation_rules,
            pii_classification=pii_classification,
            regulatory_notes=data.get('regulatory_notes', ''),
            audit_trail=data.get('audit_trail', 'git_commits'),
            retention_policy=data.get('retention_policy', '7_years'),
            created_by=data.get('created_by', ''),
            approved_by=data.get('approved_by', ''),
        )
    
    def validate_feature_usage(self, feature_name: str, usage_timestamp: datetime, 
                             data_timestamp: datetime) -> List[ContractViolation]:
        """Validate that feature usage complies with PIT contract"""
        violations = []
        
        if feature_name not in self.contracts:
            violations.append(ContractViolation(
                feature_name=feature_name,
                violation_type="missing_contract",
                severity=ValidationSeverity.CRITICAL,
                message=f"No contract found for feature: {feature_name}"
            ))
            return violations
            
        contract = self.contracts[feature_name]
        
        # Check arrival latency compliance
        min_delay = timedelta(minutes=contract.arrival_latency_minutes)
        actual_delay = usage_timestamp - data_timestamp
        
        if actual_delay < min_delay:
            violations.append(ContractViolation(
                feature_name=feature_name,
                violation_type="insufficient_latency",
                severity=ValidationSeverity.HIGH,
                message=f"Feature used too soon after data timestamp. "
                       f"Required delay: {min_delay}, Actual: {actual_delay}",
                expected_value=min_delay,
                actual_value=actual_delay
            ))
        
        # Check dependencies
        for dep in contract.dependencies:
            if dep not in self.contracts:
                violations.append(ContractViolation(
                    feature_name=feature_name,
                    violation_type="missing_dependency_contract",
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Dependency {dep} lacks contract"
                ))
        
        return violations
    
    def validate_feature_value(self, feature_name: str, value: Any) -> List[ContractViolation]:
        """Validate feature value against contract rules"""
        violations = []
        
        if feature_name not in self.contracts:
            return violations  # Already handled in usage validation
            
        contract = self.contracts[feature_name]
        rules = contract.validation_rules
        
        # Check valid range
        if rules.valid_range and value is not None:
            min_val, max_val = rules.valid_range
            if not (min_val <= value <= max_val):
                violations.append(ContractViolation(
                    feature_name=feature_name,
                    violation_type="value_out_of_range",
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Value {value} outside valid range [{min_val}, {max_val}]",
                    expected_value=rules.valid_range,
                    actual_value=value
                ))
        
        # Check null handling
        if value is None and rules.null_handling == "reject":
            violations.append(ContractViolation(
                feature_name=feature_name,
                violation_type="null_value_rejected",
                severity=ValidationSeverity.MEDIUM,
                message="Null value not allowed per contract rules"
            ))
            
        return violations
    
    def get_missing_contracts(self, feature_list: List[str]) -> List[str]:
        """Return list of features without contracts"""
        return [f for f in feature_list if f not in self.contracts]
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_contracts": len(self.contracts),
            "contracts_by_type": {},
            "contracts_by_source": {},
            "compliance_summary": {
                "features_with_contracts": len(self.contracts),
                "features_missing_contracts": 0,  # Set by caller
                "total_violations_last_24h": 0,   # Set by monitoring
            },
            "contracts": []
        }
        
        # Group by type and source
        for contract in self.contracts.values():
            type_key = contract.feature_type.value
            source_key = contract.data_source
            
            report["contracts_by_type"][type_key] = report["contracts_by_type"].get(type_key, 0) + 1
            report["contracts_by_source"][source_key] = report["contracts_by_source"].get(source_key, 0) + 1
            
            # Add contract summary
            report["contracts"].append({
                "feature_name": contract.feature_name,
                "feature_type": contract.feature_type.value,
                "data_source": contract.data_source,
                "version": contract.version,
                "arrival_latency_minutes": contract.arrival_latency_minutes,
                "update_frequency": contract.update_frequency,
                "pii_classification": contract.pii_classification.value,
                "created_by": contract.created_by,
                "approved_by": contract.approved_by,
                "last_modified": contract.last_modified.isoformat() if contract.last_modified else None
            })
        
        return report
    
    def enforce_pit_compliance(self, feature_name: str, as_of_ts: datetime) -> bool:
        """Enforce PIT compliance for feature access"""
        if feature_name not in self.contracts:
            logger.error(f"PIT enforcement failed: No contract for {feature_name}")
            return False
            
        contract = self.contracts[feature_name]
        
        # Parse PIT rule (simplified - extend based on your rules)
        if "market_open" in contract.point_in_time_rule.lower():
            # Example: Feature available at market open next day
            market_open = as_of_ts.replace(hour=9, minute=30, second=0, microsecond=0)
            if as_of_ts < market_open:
                logger.warning(f"PIT violation: {feature_name} not available until {market_open}")
                return False
                
        return True

def validate_contracts_directory(contracts_dir: str) -> Dict[str, Any]:
    """Validate all contracts in directory and return summary"""
    validator = FeatureContractValidator(contracts_dir)
    
    summary = {
        "total_contracts": len(validator.contracts),
        "validation_errors": [],
        "contracts_loaded": list(validator.contracts.keys()),
        "missing_required_fields": [],
        "invalid_enums": []
    }
    
    # Additional validation logic can be added here
    
    return summary

# Environment variable for enabling enforcement
PIT_CONTRACTS_ENFORCE = os.getenv("PIT_CONTRACTS_ENFORCE", "false").lower() == "true"