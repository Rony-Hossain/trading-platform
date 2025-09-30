"""
Comprehensive tests for Feature Contract Validator and PIT Compliance

This test suite validates all aspects of the feature contract system including:
- Contract validation and registration
- PIT compliance checking
- Temporal consistency validation
- CI/CD integration
- Error handling and edge cases
"""

import pytest
import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add services to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'services', 'analysis-service', 'app'))

from core.feature_contracts import (
    FeatureContractValidator, 
    FeatureContract,
    VendorSLA,
    RevisionPolicy,
    ViolationType, 
    PITViolation,
    PitComplianceError,
    PITRuleType,
    get_validator,
    enforce_pit_compliance
)


class TestVendorSLA:
    """Test VendorSLA model validation"""
    
    def test_valid_vendor_sla(self):
        """Test creating a valid VendorSLA"""
        sla = VendorSLA(
            availability_pct=99.9,
            max_latency_seconds=300,
            update_frequency="1min",
            backfill_policy="24h_window",
            quality_guarantee="99.95%"
        )
        assert sla.availability_pct == 99.9
        assert sla.max_latency_seconds == 300
    
    def test_invalid_availability_percentage(self):
        """Test VendorSLA with invalid availability percentage"""
        with pytest.raises(ValueError, match="Availability must be between 0.0 and 100.0"):
            VendorSLA(
                availability_pct=101.0,
                max_latency_seconds=300,
                update_frequency="1min",
                backfill_policy="24h_window",
                quality_guarantee="99.95%"
            )
    
    def test_negative_latency(self):
        """Test VendorSLA with negative latency"""
        with pytest.raises(ValueError):
            VendorSLA(
                availability_pct=99.9,
                max_latency_seconds=-10,
                update_frequency="1min",
                backfill_policy="24h_window",
                quality_guarantee="99.95%"
            )


class TestRevisionPolicy:
    """Test RevisionPolicy model validation"""
    
    def test_valid_revision_policy(self):
        """Test creating a valid RevisionPolicy"""
        policy = RevisionPolicy(
            allows_revisions=True,
            revision_window_hours=24,
            revision_notification="immediate",
            backfill_on_revision=True,
            version_tracking=True
        )
        assert policy.allows_revisions is True
        assert policy.revision_window_hours == 24
    
    def test_negative_revision_window(self):
        """Test RevisionPolicy with negative revision window"""
        with pytest.raises(ValueError):
            RevisionPolicy(
                allows_revisions=True,
                revision_window_hours=-5,
                revision_notification="immediate",
                backfill_on_revision=True,
                version_tracking=True
            )


class TestFeatureContract:
    """Test FeatureContract model validation"""
    
    @pytest.fixture
    def valid_contract_data(self):
        """Fixture providing valid contract data"""
        return {
            "feature_id": "aapl_close_price_1min",
            "feature_name": "Apple Inc. 1-Minute Close Price",
            "data_source": "bloomberg_api",
            "feature_type": "price",
            "granularity": "1min",
            "currency": "USD",
            "unit": "dollars",
            "as_of_ts": "2024-01-15T09:30:00.000Z",
            "effective_ts": "2024-01-15T09:35:00.000Z",
            "arrival_latency": 300,
            "point_in_time_rule": "strict",
            "vendor_SLA": {
                "availability_pct": 99.9,
                "max_latency_seconds": 300,
                "update_frequency": "1min",
                "backfill_policy": "24h_window",
                "quality_guarantee": "99.95%"
            },
            "revision_policy": {
                "allows_revisions": False,
                "revision_window_hours": 0,
                "revision_notification": "none",
                "backfill_on_revision": False,
                "version_tracking": True
            },
            "contract_version": "1.0.0",
            "created_by": "quant.team@company.com",
            "approved_by": "risk.manager@company.com",
            "approval_date": "2024-01-15T14:30:00.000Z",
            "tags": ["equity", "price", "real_time", "core", "aapl"],
            "dependencies": []
        }
    
    def test_valid_contract_creation(self, valid_contract_data):
        """Test creating a valid FeatureContract"""
        contract = FeatureContract(**valid_contract_data)
        assert contract.feature_id == "aapl_close_price_1min"
        assert contract.feature_type == "price"
        assert contract.point_in_time_rule == PITRuleType.STRICT
    
    def test_invalid_feature_id_pattern(self, valid_contract_data):
        """Test contract with invalid feature_id pattern"""
        valid_contract_data["feature_id"] = "Invalid-Feature-ID"
        with pytest.raises(ValueError):
            FeatureContract(**valid_contract_data)
    
    def test_invalid_feature_type(self, valid_contract_data):
        """Test contract with invalid feature type"""
        valid_contract_data["feature_type"] = "invalid_type"
        with pytest.raises(ValueError, match="Feature type must be one of"):
            FeatureContract(**valid_contract_data)
    
    def test_invalid_contract_version(self, valid_contract_data):
        """Test contract with invalid version format"""
        valid_contract_data["contract_version"] = "invalid_version"
        with pytest.raises(ValueError):
            FeatureContract(**valid_contract_data)
    
    def test_invalid_timestamp_format(self, valid_contract_data):
        """Test contract with invalid timestamp format"""
        valid_contract_data["as_of_ts"] = "invalid_timestamp"
        with pytest.raises(ValueError, match="Timestamp must be valid ISO 8601 format"):
            FeatureContract(**valid_contract_data)
    
    def test_temporal_consistency_validation(self, valid_contract_data):
        """Test temporal consistency validation"""
        contract = FeatureContract(**valid_contract_data)
        violations = contract.validate_temporal_consistency()
        assert len(violations) == 0  # Should be consistent
        
        # Test violation case
        valid_contract_data["effective_ts"] = "2024-01-15T09:25:00.000Z"  # Before as_of_ts
        contract = FeatureContract(**valid_contract_data)
        violations = contract.validate_temporal_consistency()
        assert len(violations) > 0
        assert "as_of_ts must be <= effective_ts" in violations[0]


class TestFeatureContractValidator:
    """Test FeatureContractValidator functionality"""
    
    @pytest.fixture
    def validator(self):
        """Create a validator for testing"""
        return FeatureContractValidator(enforce_pit_compliance=True)
    
    @pytest.fixture
    def valid_contract_data(self):
        """Fixture providing valid contract data"""
        return {
            "feature_id": "test_feature_1min",
            "feature_name": "Test Feature 1-Minute",
            "data_source": "test_provider",
            "feature_type": "price",
            "granularity": "1min",
            "as_of_ts": "2024-01-15T09:30:00.000Z",
            "effective_ts": "2024-01-15T09:35:00.000Z",
            "arrival_latency": 300,
            "point_in_time_rule": "strict",
            "vendor_SLA": {
                "availability_pct": 99.5,
                "max_latency_seconds": 300,
                "update_frequency": "1min",
                "backfill_policy": "24h_window",
                "quality_guarantee": "99.9%"
            },
            "revision_policy": {
                "allows_revisions": False,
                "revision_window_hours": 0,
                "revision_notification": "none",
                "backfill_on_revision": False,
                "version_tracking": True
            },
            "contract_version": "1.0.0",
            "created_by": "test@company.com"
        }
    
    def test_contract_registration_success(self, validator, valid_contract_data):
        """Test successful contract registration"""
        result = validator.register_contract(valid_contract_data)
        assert result is True
        
        contract = validator.get_contract("test_feature_1min")
        assert contract is not None
        assert contract.feature_id == "test_feature_1min"
        assert contract.pit_audit_status == "compliant"
    
    def test_contract_registration_failure_missing_fields(self, validator):
        """Test contract registration failure with missing fields"""
        incomplete_data = {
            "feature_id": "incomplete_feature",
            "feature_name": "Incomplete Feature"
            # Missing required fields
        }
        
        with pytest.raises(PitComplianceError):
            validator.register_contract(incomplete_data)
    
    def test_contract_registration_temporal_violation(self, validator, valid_contract_data):
        """Test contract registration with temporal consistency violation"""
        # Make effective_ts before as_of_ts
        valid_contract_data["effective_ts"] = "2024-01-15T09:25:00.000Z"
        
        with pytest.raises(PitComplianceError):
            validator.register_contract(valid_contract_data)
    
    def test_pit_compliance_check_no_contract(self, validator):
        """Test PIT compliance check for feature without contract"""
        feature_data = {
            "as_of_timestamp": "2024-01-15T09:30:00.000Z",
            "effective_timestamp": "2024-01-15T09:35:00.000Z",
            "value": 150.25
        }
        
        violations = validator.check_pit_compliance("nonexistent_feature", feature_data)
        assert len(violations) == 1
        assert violations[0].violation_type == ViolationType.CRITICAL
        assert "No contract found" in violations[0].description
    
    def test_pit_compliance_check_missing_timestamps(self, validator, valid_contract_data):
        """Test PIT compliance check with missing timestamps"""
        validator.register_contract(valid_contract_data)
        
        feature_data = {
            "value": 150.25
            # Missing required timestamps
        }
        
        violations = validator.check_pit_compliance("test_feature_1min", feature_data)
        assert len(violations) == 1
        assert violations[0].violation_type == ViolationType.HIGH
        assert "Missing required timestamps" in violations[0].description
    
    def test_pit_compliance_check_future_data_access(self, validator, valid_contract_data):
        """Test PIT compliance check with future data access violation"""
        validator.register_contract(valid_contract_data)
        
        feature_data = {
            "as_of_timestamp": "2024-01-15T09:30:00.000Z",
            "effective_timestamp": "2024-01-15T09:35:00.000Z",
            "prediction_timestamp": "2024-01-15T09:25:00.000Z",  # Before as_of_ts
            "value": 150.25
        }
        
        violations = validator.check_pit_compliance("test_feature_1min", feature_data)
        assert len(violations) >= 1
        critical_violations = [v for v in violations if v.violation_type == ViolationType.CRITICAL]
        assert len(critical_violations) == 1
        assert "Future data access" in critical_violations[0].description
    
    def test_pit_compliance_check_feature_not_available(self, validator, valid_contract_data):
        """Test PIT compliance check with feature not yet available"""
        validator.register_contract(valid_contract_data)
        
        feature_data = {
            "as_of_timestamp": "2024-01-15T09:30:00.000Z",
            "effective_timestamp": "2024-01-15T09:35:00.000Z",
            "prediction_timestamp": "2024-01-15T09:32:00.000Z",  # Before effective_ts
            "value": 150.25
        }
        
        violations = validator.check_pit_compliance("test_feature_1min", feature_data)
        assert len(violations) >= 1
        high_violations = [v for v in violations if v.violation_type == ViolationType.HIGH]
        assert len(high_violations) == 1
        assert "Feature not yet available" in high_violations[0].description
    
    def test_pit_compliance_check_valid(self, validator, valid_contract_data):
        """Test PIT compliance check with valid data"""
        validator.register_contract(valid_contract_data)
        
        feature_data = {
            "as_of_timestamp": "2024-01-15T09:30:00.000Z",
            "effective_timestamp": "2024-01-15T09:35:00.000Z",
            "prediction_timestamp": "2024-01-15T09:40:00.000Z",  # After effective_ts
            "value": 150.25
        }
        
        violations = validator.check_pit_compliance("test_feature_1min", feature_data)
        # Should have no critical or high violations
        critical_high_violations = [v for v in violations 
                                  if v.violation_type in [ViolationType.CRITICAL, ViolationType.HIGH]]
        assert len(critical_high_violations) == 0
    
    def test_audit_feature_set(self, validator, valid_contract_data):
        """Test auditing a complete feature set"""
        validator.register_contract(valid_contract_data)
        
        # Register second contract
        valid_contract_data["feature_id"] = "test_feature_2min"
        valid_contract_data["granularity"] = "2min"
        validator.register_contract(valid_contract_data)
        
        features = {
            "test_feature_1min": {
                "as_of_timestamp": "2024-01-15T09:30:00.000Z",
                "effective_timestamp": "2024-01-15T09:35:00.000Z",
                "value": 150.25
            },
            "test_feature_2min": {
                "as_of_timestamp": "2024-01-15T09:30:00.000Z",
                "effective_timestamp": "2024-01-15T09:35:00.000Z",
                "value": 75.50
            }
        }
        
        violations = validator.audit_feature_set(
            features, 
            prediction_timestamp="2024-01-15T09:40:00.000Z"
        )
        
        # Should have no critical violations
        critical_violations = [v for v in violations if v.violation_type == ViolationType.CRITICAL]
        assert len(critical_violations) == 0
    
    def test_compliance_report_generation(self, validator, valid_contract_data):
        """Test compliance report generation"""
        validator.register_contract(valid_contract_data)
        
        report = validator.get_compliance_report()
        
        assert report["total_contracts"] == 1
        assert report["compliant_contracts"] == 1
        assert report["compliance_rate"] == 1.0
        assert report["enforcement_enabled"] is True
        assert "last_audit_time" in report
    
    def test_contract_directory_loading(self, validator):
        """Test loading contracts from directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test contract files
            contract1 = {
                "feature_id": "dir_test_feature1",
                "feature_name": "Directory Test Feature 1",
                "data_source": "test_provider",
                "feature_type": "price",
                "granularity": "1min",
                "as_of_ts": "2024-01-15T09:30:00.000Z",
                "effective_ts": "2024-01-15T09:35:00.000Z",
                "arrival_latency": 300,
                "point_in_time_rule": "strict",
                "vendor_SLA": {
                    "availability_pct": 99.5,
                    "max_latency_seconds": 300,
                    "update_frequency": "1min",
                    "backfill_policy": "24h_window",
                    "quality_guarantee": "99.9%"
                },
                "revision_policy": {
                    "allows_revisions": False,
                    "revision_window_hours": 0,
                    "revision_notification": "none",
                    "backfill_on_revision": False,
                    "version_tracking": True
                },
                "contract_version": "1.0.0",
                "created_by": "test@company.com"
            }
            
            contract_file = Path(temp_dir) / "test_contract1.json"
            with open(contract_file, 'w') as f:
                json.dump(contract1, f)
            
            # Load contracts
            loaded_count = validator.load_contracts_from_directory(temp_dir)
            assert loaded_count == 1
            assert validator.get_contract("dir_test_feature1") is not None
    
    def test_contract_saving(self, validator, valid_contract_data):
        """Test saving contract to file"""
        validator.register_contract(valid_contract_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            success = validator.save_contract("test_feature_1min", temp_dir)
            assert success is True
            
            contract_file = Path(temp_dir) / "test_feature_1min.json"
            assert contract_file.exists()
            
            # Verify saved content
            with open(contract_file, 'r') as f:
                saved_data = json.load(f)
            assert saved_data["feature_id"] == "test_feature_1min"


class TestPITComplianceIntegration:
    """Test PIT compliance integration scenarios"""
    
    def test_environment_variable_enforcement(self):
        """Test PIT enforcement based on environment variables"""
        # Test with environment variable set to true
        with patch.dict(os.environ, {'PIT_CONTRACTS_ENFORCE': 'true'}):
            validator = FeatureContractValidator()
            assert validator.enforce_pit_compliance is True
        
        # Test with environment variable set to false
        with patch.dict(os.environ, {'PIT_CONTRACTS_ENFORCE': 'false'}):
            validator = FeatureContractValidator()
            assert validator.enforce_pit_compliance is False
    
    def test_global_validator_instance(self):
        """Test global validator instance functionality"""
        validator1 = get_validator()
        validator2 = get_validator()
        
        # Should return the same instance
        assert validator1 is validator2
    
    def test_enforce_pit_compliance_function(self):
        """Test enforce_pit_compliance utility function"""
        with patch.dict(os.environ, {'PIT_CONTRACTS_ENFORCE': 'true'}):
            assert enforce_pit_compliance() is True
        
        with patch.dict(os.environ, {'PIT_CONTRACTS_ENFORCE': 'false'}):
            # Create new validator to pick up env change
            from core.feature_contracts import _global_validator
            if '_global_validator' in globals():
                del globals()['_global_validator']
            assert enforce_pit_compliance() is False


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_contract_with_malformed_json_schema(self):
        """Test contract validation with malformed data"""
        validator = FeatureContractValidator(enforce_pit_compliance=True)
        
        malformed_data = {
            "feature_id": "malformed_feature",
            "vendor_SLA": "not_an_object"  # Should be object
        }
        
        with pytest.raises(PitComplianceError):
            validator.register_contract(malformed_data)
    
    def test_timestamp_parsing_errors(self):
        """Test handling of timestamp parsing errors"""
        validator = FeatureContractValidator(enforce_pit_compliance=True)
        
        # Register a valid contract first
        valid_contract = {
            "feature_id": "timestamp_test_feature",
            "feature_name": "Timestamp Test Feature",
            "data_source": "test_provider",
            "feature_type": "price",
            "granularity": "1min",
            "as_of_ts": "2024-01-15T09:30:00.000Z",
            "effective_ts": "2024-01-15T09:35:00.000Z",
            "arrival_latency": 300,
            "point_in_time_rule": "strict",
            "vendor_SLA": {
                "availability_pct": 99.5,
                "max_latency_seconds": 300,
                "update_frequency": "1min",
                "backfill_policy": "24h_window",
                "quality_guarantee": "99.9%"
            },
            "revision_policy": {
                "allows_revisions": False,
                "revision_window_hours": 0,
                "revision_notification": "none",
                "backfill_on_revision": False,
                "version_tracking": True
            },
            "contract_version": "1.0.0",
            "created_by": "test@company.com"
        }
        
        validator.register_contract(valid_contract)
        
        # Test with malformed timestamps
        feature_data = {
            "as_of_timestamp": "invalid_timestamp",
            "effective_timestamp": "2024-01-15T09:35:00.000Z",
            "prediction_timestamp": "2024-01-15T09:40:00.000Z",
            "value": 150.25
        }
        
        violations = validator.check_pit_compliance("timestamp_test_feature", feature_data)
        assert len(violations) >= 1
        timestamp_error_violations = [v for v in violations if "Timestamp validation error" in v.description]
        assert len(timestamp_error_violations) >= 1
    
    def test_nonexistent_contract_file_loading(self):
        """Test loading contracts from nonexistent directory"""
        validator = FeatureContractValidator()
        
        loaded_count = validator.load_contracts_from_directory("/nonexistent/directory")
        assert loaded_count == 0
    
    def test_contract_saving_failure(self):
        """Test contract saving failure scenarios"""
        validator = FeatureContractValidator()
        
        # Try to save nonexistent contract
        success = validator.save_contract("nonexistent_feature", "/tmp")
        assert success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

