"""
Feature Contracts and PIT Enforcement System

This module implements comprehensive Point-in-Time (PIT) compliance validation 
for all features used in the trading platform to prevent data leakage.
"""

import json
import os
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

import jsonschema
from pydantic import BaseModel, Field, validator
from pydantic.types import StrictStr, StrictFloat, StrictInt


class ViolationType(Enum):
    """Types of PIT compliance violations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PITRuleType(Enum):
    """PIT rule enforcement levels"""
    STRICT = "strict"
    RELAXED = "relaxed"
    CUSTOM = "custom"


@dataclass
class PITViolation:
    """Represents a Point-in-Time compliance violation"""
    violation_type: ViolationType
    feature_id: str
    description: str
    timestamp: datetime
    remediation_required: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_type": self.violation_type.value,
            "feature_id": self.feature_id,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "remediation_required": self.remediation_required
        }


class VendorSLA(BaseModel):
    """Vendor Service Level Agreement specifications"""
    availability_pct: StrictFloat = Field(..., ge=0.0, le=100.0)
    max_latency_seconds: StrictInt = Field(..., ge=0)
    update_frequency: StrictStr
    backfill_policy: StrictStr
    quality_guarantee: StrictStr
    
    @validator('availability_pct')
    def validate_availability(cls, v):
        if not 0.0 <= v <= 100.0:
            raise ValueError('Availability must be between 0.0 and 100.0')
        return v


class RevisionPolicy(BaseModel):
    """Data revision and correction policy"""
    allows_revisions: bool
    revision_window_hours: StrictInt = Field(..., ge=0)
    revision_notification: StrictStr
    backfill_on_revision: bool
    version_tracking: bool


class FeatureContract(BaseModel):
    """Complete feature contract specification"""
    
    # Core identification
    feature_id: StrictStr = Field(..., pattern=r"^[a-z][a-z0-9_]*[a-z0-9]$")
    feature_name: StrictStr
    data_source: StrictStr
    feature_type: StrictStr
    granularity: StrictStr
    
    # Temporal constraints
    as_of_ts: StrictStr  # ISO 8601 timestamp
    effective_ts: StrictStr  # ISO 8601 timestamp  
    arrival_latency: StrictInt = Field(..., ge=0)  # seconds
    
    # PIT rules and quality
    point_in_time_rule: PITRuleType
    vendor_SLA: VendorSLA
    revision_policy: RevisionPolicy
    
    # Metadata
    contract_version: StrictStr = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    created_by: StrictStr
    approved_by: Optional[StrictStr] = None
    approval_date: Optional[StrictStr] = None
    
    # Optional fields
    currency: Optional[StrictStr] = None
    unit: Optional[StrictStr] = None
    additional_rules: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[StrictStr]] = None
    tags: Optional[List[StrictStr]] = None
    
    # System generated
    pit_audit_status: Optional[StrictStr] = None
    last_audit_ts: Optional[StrictStr] = None
    
    @validator('feature_type')
    def validate_feature_type(cls, v):
        allowed_types = ["price", "volume", "sentiment", "fundamental", "technical", "macro", "derived"]
        if v not in allowed_types:
            raise ValueError(f'Feature type must be one of: {allowed_types}')
        return v
    
    @validator('as_of_ts', 'effective_ts', 'approval_date', 'last_audit_ts')
    def validate_timestamps(cls, v):
        if v is None:
            return v
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError('Timestamp must be valid ISO 8601 format')
        return v
    
    def validate_temporal_consistency(self) -> List[str]:
        """Validate temporal consistency rules"""
        violations = []
        
        try:
            as_of_dt = datetime.fromisoformat(self.as_of_ts.replace('Z', '+00:00'))
            effective_dt = datetime.fromisoformat(self.effective_ts.replace('Z', '+00:00'))
            
            # Rule 1: as_of_ts <= effective_ts
            if as_of_dt > effective_dt:
                violations.append("as_of_ts must be <= effective_ts")
            
            # Rule 2: effective_ts >= as_of_ts + arrival_latency
            expected_effective_dt = as_of_dt.timestamp() + self.arrival_latency
            if effective_dt.timestamp() < expected_effective_dt:
                violations.append("effective_ts must be >= as_of_ts + arrival_latency")
                
        except Exception as e:
            violations.append(f"Timestamp parsing error: {str(e)}")
            
        return violations


class PitComplianceError(Exception):
    """Exception raised for PIT compliance violations"""
    def __init__(self, violations: List[PITViolation]):
        self.violations = violations
        super().__init__(f"PIT compliance violations: {[v.description for v in violations]}")


class FeatureContractValidator:
    """Main validator for feature contracts and PIT compliance"""
    
    def __init__(self, enforce_pit_compliance: bool = None):
        self.contracts: Dict[str, FeatureContract] = {}
        self.violations: List[PITViolation] = []
        
        # Check environment variable for enforcement
        env_enforce = os.getenv('PIT_CONTRACTS_ENFORCE', 'false').lower()
        self.enforce_pit_compliance = (
            enforce_pit_compliance if enforce_pit_compliance is not None 
            else env_enforce in ('true', '1', 'yes', 'on')
        )
        
        self._load_schema()
    
    def _load_schema(self):
        """Load JSON schema for contract validation"""
        self.contract_schema = {
            "type": "object",
            "required": [
                "feature_id", "feature_name", "data_source", "feature_type",
                "granularity", "as_of_ts", "effective_ts", "arrival_latency",
                "point_in_time_rule", "vendor_SLA", "revision_policy",
                "contract_version", "created_by"
            ],
            "properties": {
                "feature_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]*[a-z0-9]$"},
                "feature_name": {"type": "string"},
                "data_source": {"type": "string"},
                "feature_type": {"type": "string", "enum": [
                    "price", "volume", "sentiment", "fundamental", "technical", "macro", "derived"
                ]},
                "granularity": {"type": "string"},
                "as_of_ts": {"type": "string"},
                "effective_ts": {"type": "string"},
                "arrival_latency": {"type": "integer", "minimum": 0},
                "point_in_time_rule": {"type": "string", "enum": ["strict", "relaxed", "custom"]},
                "contract_version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
                "created_by": {"type": "string"}
            }
        }
    
    def register_contract(self, contract_data: Dict[str, Any]) -> bool:
        """Register a new feature contract"""
        try:
            # Validate against JSON schema
            jsonschema.validate(contract_data, self.contract_schema)
            
            # Create contract object with Pydantic validation
            contract = FeatureContract(**contract_data)
            
            # Validate temporal consistency
            temporal_violations = contract.validate_temporal_consistency()
            if temporal_violations:
                raise ValueError(f"Temporal consistency violations: {temporal_violations}")
            
            # Store contract
            self.contracts[contract.feature_id] = contract
            
            # Update audit status
            contract.pit_audit_status = "compliant"
            contract.last_audit_ts = datetime.now(timezone.utc).isoformat()
            
            return True
            
        except Exception as e:
            if self.enforce_pit_compliance:
                raise PitComplianceError([
                    PITViolation(
                        violation_type=ViolationType.CRITICAL,
                        feature_id=contract_data.get('feature_id', 'unknown'),
                        description=f"Contract registration failed: {str(e)}",
                        timestamp=datetime.now(timezone.utc)
                    )
                ])
            return False
    
    def get_contract(self, feature_id: str) -> Optional[FeatureContract]:
        """Retrieve contract for a feature"""
        return self.contracts.get(feature_id)
    
    def check_pit_compliance(self, feature_id: str, feature_data: Dict[str, Any]) -> List[PITViolation]:
        """Check PIT compliance for feature usage"""
        violations = []
        
        # Check if contract exists
        contract = self.get_contract(feature_id)
        if not contract:
            violations.append(PITViolation(
                violation_type=ViolationType.CRITICAL,
                feature_id=feature_id,
                description=f"No contract found for feature: {feature_id}",
                timestamp=datetime.now(timezone.utc)
            ))
            return violations
        
        # Extract timestamps from feature data
        as_of_timestamp = feature_data.get('as_of_timestamp')
        effective_timestamp = feature_data.get('effective_timestamp')
        prediction_timestamp = feature_data.get('prediction_timestamp')
        
        if not all([as_of_timestamp, effective_timestamp]):
            violations.append(PITViolation(
                violation_type=ViolationType.HIGH,
                feature_id=feature_id,
                description="Missing required timestamps in feature data",
                timestamp=datetime.now(timezone.utc)
            ))
            return violations
        
        try:
            # Parse timestamps
            as_of_dt = datetime.fromisoformat(as_of_timestamp.replace('Z', '+00:00'))
            effective_dt = datetime.fromisoformat(effective_timestamp.replace('Z', '+00:00'))
            
            if prediction_timestamp:
                pred_dt = datetime.fromisoformat(prediction_timestamp.replace('Z', '+00:00'))
                
                # Critical PIT rule: as_of_ts <= prediction_timestamp
                if as_of_dt > pred_dt:
                    violations.append(PITViolation(
                        violation_type=ViolationType.CRITICAL,
                        feature_id=feature_id,
                        description=f"Future data access: as_of_ts ({as_of_timestamp}) > prediction_ts ({prediction_timestamp})",
                        timestamp=datetime.now(timezone.utc)
                    ))
                
                # Check effective timestamp availability
                if effective_dt > pred_dt:
                    violations.append(PITViolation(
                        violation_type=ViolationType.HIGH,
                        feature_id=feature_id,
                        description=f"Feature not yet available: effective_ts ({effective_timestamp}) > prediction_ts ({prediction_timestamp})",
                        timestamp=datetime.now(timezone.utc)
                    ))
            
            # Validate arrival latency modeling
            expected_effective_ts = as_of_dt.timestamp() + contract.arrival_latency
            if effective_dt.timestamp() < expected_effective_ts:
                violations.append(PITViolation(
                    violation_type=ViolationType.MEDIUM,
                    feature_id=feature_id,
                    description=f"Arrival latency violation: feature available before expected time",
                    timestamp=datetime.now(timezone.utc)
                ))
                
        except Exception as e:
            violations.append(PITViolation(
                violation_type=ViolationType.HIGH,
                feature_id=feature_id,
                description=f"Timestamp validation error: {str(e)}",
                timestamp=datetime.now(timezone.utc)
            ))
        
        # Store violations
        self.violations.extend(violations)
        
        # Update contract audit status
        if violations:
            severity_levels = [v.violation_type for v in violations]
            if ViolationType.CRITICAL in severity_levels:
                contract.pit_audit_status = "violation"
            elif ViolationType.HIGH in severity_levels:
                contract.pit_audit_status = "warning"
        
        return violations
    
    def audit_feature_set(self, features: Dict[str, Any], 
                         prediction_timestamp: str = None) -> List[PITViolation]:
        """Audit a complete set of features for PIT compliance"""
        all_violations = []
        
        for feature_id, feature_data in features.items():
            if prediction_timestamp:
                feature_data['prediction_timestamp'] = prediction_timestamp
                
            violations = self.check_pit_compliance(feature_id, feature_data)
            all_violations.extend(violations)
        
        return all_violations
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        total_contracts = len(self.contracts)
        compliant_contracts = sum(1 for c in self.contracts.values() 
                                if c.pit_audit_status == "compliant")
        
        violation_counts = {}
        for violation_type in ViolationType:
            violation_counts[violation_type.value] = sum(
                1 for v in self.violations if v.violation_type == violation_type
            )
        
        return {
            "total_contracts": total_contracts,
            "compliant_contracts": compliant_contracts,
            "compliance_rate": compliant_contracts / total_contracts if total_contracts > 0 else 0.0,
            "total_violations": len(self.violations),
            "violation_breakdown": violation_counts,
            "enforcement_enabled": self.enforce_pit_compliance,
            "last_audit_time": datetime.now(timezone.utc).isoformat()
        }
    
    def load_contracts_from_directory(self, contracts_dir: str) -> int:
        """Load all contracts from a directory"""
        contracts_loaded = 0
        contracts_path = Path(contracts_dir)
        
        if not contracts_path.exists():
            return 0
            
        for contract_file in contracts_path.glob("*.json"):
            try:
                with open(contract_file, 'r') as f:
                    contract_data = json.load(f)
                    
                if self.register_contract(contract_data):
                    contracts_loaded += 1
                    
            except Exception as e:
                print(f"Failed to load contract from {contract_file}: {str(e)}")
                
        return contracts_loaded
    
    def save_contract(self, feature_id: str, contracts_dir: str) -> bool:
        """Save a contract to file"""
        contract = self.get_contract(feature_id)
        if not contract:
            return False
            
        contracts_path = Path(contracts_dir)
        contracts_path.mkdir(parents=True, exist_ok=True)
        
        contract_file = contracts_path / f"{feature_id}.json"
        
        try:
            with open(contract_file, 'w') as f:
                json.dump(contract.dict(), f, indent=2)
            return True
        except Exception:
            return False


def get_validator() -> FeatureContractValidator:
    """Get the global feature contract validator instance"""
    global _global_validator
    if '_global_validator' not in globals():
        _global_validator = FeatureContractValidator()
    return _global_validator


def enforce_pit_compliance() -> bool:
    """Check if PIT compliance enforcement is enabled"""
    return get_validator().enforce_pit_compliance

