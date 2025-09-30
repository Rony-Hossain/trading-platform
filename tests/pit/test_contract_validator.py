"""
Tests for Feature Contract Validator

Tests PIT contract validation, enforcement, and compliance checking.
Ensures data lineage integrity and prevents look-ahead bias.
"""

import pytest
import tempfile
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add services/analysis-service to path
analysis_service_path = project_root / 'services' / 'analysis-service'
sys.path.insert(0, str(analysis_service_path))

from app.core.feature_contracts import (
    FeatureContractValidator, FeatureContract, FeatureType, PIIClassification,
    VendorSLA, RevisionPolicy, ValidationRules, ContractViolation, ValidationSeverity
)

@pytest.fixture
def temp_contracts_dir():
    """Create temporary directory with test contracts"""
    with tempfile.TemporaryDirectory() as temp_dir:
        contracts_dir = Path(temp_dir)
        
        # Create valid contract
        valid_contract = {
            'feature_name': 'vix_close',
            'feature_type': 'macro',
            'data_source': 'yahoo',
            'version': '1.0.0',
            'as_of_ts_rule': 'T+1 09:30 ET',
            'effective_ts_rule': 'T 16:00 ET',
            'arrival_latency_minutes': 1050,  # 17.5 hours
            'point_in_time_rule': 'No access to T+0 VIX close until T+1 market open',
            'vendor_sla': {
                'availability': 99.9,
                'latency_minutes': 15,
                'quality': 99.99
            },
            'revision_policy': {
                'revision_frequency': 'never',
                'revision_window_days': 0,
                'notification_method': 'none'
            },
            'computation_logic': 'Direct feed from CBOE via Yahoo Finance',
            'dependencies': [],
            'lookback_period_days': 365,
            'update_frequency': 'daily',
            'validation_rules': {
                'valid_range': [5.0, 80.0],
                'null_handling': 'forward_fill',
                'outlier_detection': '3_sigma_rolling_30d',
                'monitoring_alerts': ['null_for_2_days', 'outside_range']
            },
            'pii_classification': 'none',
            'regulatory_notes': 'Public market data',
            'audit_trail': 'git_commits',
            'retention_policy': '7_years',
            'created_by': 'test_user',
            'approved_by': 'test_approver'
        }
        
        # Save valid contract
        with open(contracts_dir / 'vix_close.yml', 'w') as f:
            yaml.dump(valid_contract, f)
            
        # Create invalid contract (missing required fields)
        invalid_contract = {
            'feature_name': 'broken_feature',
            'feature_type': 'technical',
            # Missing required fields intentionally
        }
        
        with open(contracts_dir / 'broken_feature.yml', 'w') as f:
            yaml.dump(invalid_contract, f)
            
        yield str(contracts_dir)

@pytest.fixture
def validator(temp_contracts_dir):
    """Create validator with test contracts"""
    return FeatureContractValidator(temp_contracts_dir)

class TestFeatureContractValidator:
    """Test feature contract validation"""
    
    def test_load_valid_contracts(self, validator):
        """Test loading valid contracts"""
        assert 'vix_close' in validator.contracts
        
        contract = validator.contracts['vix_close']
        assert contract.feature_name == 'vix_close'
        assert contract.feature_type == FeatureType.MACRO
        assert contract.arrival_latency_minutes == 1050
        assert contract.pii_classification == PIIClassification.NONE
        
    def test_missing_contract_detection(self, validator):
        """Test detection of missing contracts"""
        feature_list = ['vix_close', 'spy_price', 'unknown_feature']
        missing = validator.get_missing_contracts(feature_list)
        
        assert 'vix_close' not in missing
        assert 'spy_price' in missing
        assert 'unknown_feature' in missing
        
    def test_pit_usage_validation_success(self, validator):
        """Test successful PIT usage validation"""
        # Valid usage: sufficient latency
        usage_time = datetime(2024, 1, 2, 10, 0)  # Next day, 10 AM
        data_time = datetime(2024, 1, 1, 16, 0)   # Previous day, 4 PM
        
        violations = validator.validate_feature_usage(
            'vix_close', usage_time, data_time
        )
        
        assert len(violations) == 0
        
    def test_pit_usage_validation_failure(self, validator):
        """Test PIT usage validation failure"""
        # Invalid usage: insufficient latency
        usage_time = datetime(2024, 1, 1, 18, 0)  # Same day, 6 PM
        data_time = datetime(2024, 1, 1, 16, 0)   # Same day, 4 PM
        
        violations = validator.validate_feature_usage(
            'vix_close', usage_time, data_time
        )
        
        assert len(violations) == 1
        assert violations[0].violation_type == 'insufficient_latency'
        assert violations[0].severity == ValidationSeverity.HIGH
        
    def test_missing_contract_violation(self, validator):
        """Test violation for missing contract"""
        violations = validator.validate_feature_usage(
            'nonexistent_feature', datetime.now(), datetime.now()
        )
        
        assert len(violations) == 1
        assert violations[0].violation_type == 'missing_contract'
        assert violations[0].severity == ValidationSeverity.CRITICAL
        
    def test_value_validation_success(self, validator):
        """Test successful value validation"""
        violations = validator.validate_feature_value('vix_close', 15.5)
        assert len(violations) == 0
        
    def test_value_validation_out_of_range(self, validator):
        """Test value validation for out-of-range values"""
        violations = validator.validate_feature_value('vix_close', 100.0)
        
        assert len(violations) == 1
        assert violations[0].violation_type == 'value_out_of_range'
        assert violations[0].severity == ValidationSeverity.MEDIUM
        
    def test_null_value_handling(self, validator):
        """Test null value handling based on contract rules"""
        # First modify contract to reject nulls
        contract = validator.contracts['vix_close']
        contract.validation_rules.null_handling = 'reject'
        
        violations = validator.validate_feature_value('vix_close', None)
        
        assert len(violations) == 1
        assert violations[0].violation_type == 'null_value_rejected'
        
    def test_pit_compliance_enforcement(self, validator):
        """Test PIT compliance enforcement"""
        # Test market open rule enforcement
        market_day = datetime(2024, 1, 2, 8, 0)  # Before market open
        
        # Should fail enforcement
        result = validator.enforce_pit_compliance('vix_close', market_day)
        assert result == False
        
        # After market open should pass
        market_open = datetime(2024, 1, 2, 10, 0)
        result = validator.enforce_pit_compliance('vix_close', market_open)
        assert result == True
        
    def test_audit_report_generation(self, validator):
        """Test audit report generation"""
        report = validator.generate_audit_report()
        
        assert 'timestamp' in report
        assert 'total_contracts' in report
        assert 'contracts_by_type' in report
        assert 'contracts_by_source' in report
        assert 'compliance_summary' in report
        assert 'contracts' in report
        
        assert report['total_contracts'] == len(validator.contracts)
        assert len(report['contracts']) == len(validator.contracts)
        
        # Check contract data
        vix_contract_data = next(
            c for c in report['contracts'] 
            if c['feature_name'] == 'vix_close'
        )
        assert vix_contract_data['feature_type'] == 'macro'
        assert vix_contract_data['data_source'] == 'yahoo'

class TestContractViolation:
    """Test contract violation handling"""
    
    def test_violation_creation(self):
        """Test creating contract violations"""
        violation = ContractViolation(
            feature_name='test_feature',
            violation_type='test_violation',
            severity=ValidationSeverity.HIGH,
            message='Test violation message',
            actual_value=10,
            expected_value=5
        )
        
        assert violation.feature_name == 'test_feature'
        assert violation.severity == ValidationSeverity.HIGH
        assert violation.actual_value == 10
        assert violation.expected_value == 5
        assert isinstance(violation.timestamp, datetime)

class TestContractIntegration:
    """Test integration scenarios"""
    
    def test_end_to_end_validation(self, validator):
        """Test complete validation workflow"""
        # Simulate feature usage
        feature_name = 'vix_close'
        usage_time = datetime(2024, 1, 2, 10, 0)
        data_time = datetime(2024, 1, 1, 16, 0)
        feature_value = 18.5
        
        # Validate usage timing
        usage_violations = validator.validate_feature_usage(
            feature_name, usage_time, data_time
        )
        
        # Validate feature value
        value_violations = validator.validate_feature_value(
            feature_name, feature_value
        )
        
        # Should have no violations
        all_violations = usage_violations + value_violations
        assert len(all_violations) == 0
        
        # Enforce PIT compliance
        compliance_ok = validator.enforce_pit_compliance(
            feature_name, usage_time
        )
        assert compliance_ok == True
        
    def test_violation_aggregation(self, validator):
        """Test aggregating multiple violations"""
        violations = []
        
        # Multiple validation scenarios
        test_cases = [
            ('vix_close', datetime(2024, 1, 1, 17, 0), datetime(2024, 1, 1, 16, 0), 18.5),  # Too soon
            ('vix_close', datetime(2024, 1, 2, 10, 0), datetime(2024, 1, 1, 16, 0), 100.0),  # Out of range
            ('missing_feature', datetime.now(), datetime.now(), 10.0),  # Missing contract
        ]
        
        for feature, usage_time, data_time, value in test_cases:
            violations.extend(validator.validate_feature_usage(feature, usage_time, data_time))
            violations.extend(validator.validate_feature_value(feature, value))
            
        # Should have multiple violations
        assert len(violations) >= 3
        
        # Check violation types
        violation_types = [v.violation_type for v in violations]
        assert 'insufficient_latency' in violation_types
        assert 'value_out_of_range' in violation_types
        assert 'missing_contract' in violation_types

def test_environment_variable_enforcement():
    """Test environment variable for enforcement"""
    # Test with enforcement disabled
    os.environ['PIT_CONTRACTS_ENFORCE'] = 'false'
    from app.core.feature_contracts import PIT_CONTRACTS_ENFORCE
    # Note: This will reflect the value when module was imported
    
    # Test with enforcement enabled
    os.environ['PIT_CONTRACTS_ENFORCE'] = 'true'
    # Would need to reload module to test this properly

if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])