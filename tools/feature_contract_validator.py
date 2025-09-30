#!/usr/bin/env python3
"""
Feature Contract Validator

Validates feature contract YAML files against comprehensive schemas and business rules.
Used in CI/CD pipeline to ensure feature governance compliance.

Usage:
    python feature_contract_validator.py validate <contract_file>
    python feature_contract_validator.py validate-all <contracts_dir>
    python feature_contract_validator.py generate-report <contracts_dir>
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from jsonschema import Draft7Validator, ValidationError
from jsonschema.exceptions import SchemaError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureContractValidator:
    """Comprehensive feature contract validation system."""
    
    # Feature type schemas
    SCHEMAS = {
        "technical": {
            "type": "object",
            "required": [
                "feature_name", "feature_type", "data_source", "version",
                "as_of_ts_rule", "effective_ts_rule", "arrival_latency_minutes",
                "point_in_time_rule", "vendor_sla", "revision_policy",
                "computation_logic", "dependencies", "lookback_period_days",
                "update_frequency", "validation_rules", "pii_classification",
                "regulatory_notes", "audit_trail", "retention_policy",
                "created_by", "created_at", "approved_by", "approved_at"
            ],
            "properties": {
                "feature_name": {
                    "type": "string",
                    "pattern": "^[a-z][a-z0-9_]*[a-z0-9]$",
                    "minLength": 3,
                    "maxLength": 50
                },
                "feature_type": {"const": "technical"},
                "data_source": {"type": "string", "minLength": 1},
                "version": {
                    "type": "string",
                    "pattern": "^\\d+\\.\\d+\\.\\d+$"
                },
                "as_of_ts_rule": {
                    "enum": ["immediate", "market_close", "next_day", "custom"]
                },
                "effective_ts_rule": {
                    "enum": ["trade_timestamp", "bar_close", "custom"]
                },
                "arrival_latency_minutes": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1440
                },
                "vendor_sla": {
                    "type": "object",
                    "required": ["availability", "latency_minutes", "quality"],
                    "properties": {
                        "availability": {"type": "number", "minimum": 90.0, "maximum": 100.0},
                        "latency_minutes": {"type": "number", "minimum": 0},
                        "quality": {"type": "number", "minimum": 90.0, "maximum": 100.0}
                    }
                },
                "revision_policy": {
                    "type": "object",
                    "required": ["revision_frequency", "revision_window_days", "notification_method"],
                    "properties": {
                        "revision_frequency": {"enum": ["never", "daily", "weekly", "monthly", "quarterly"]},
                        "revision_window_days": {"type": "integer", "minimum": 0},
                        "notification_method": {"enum": ["none", "email", "webhook", "alert"]}
                    }
                },
                "validation_rules": {
                    "type": "object",
                    "required": ["null_handling", "outlier_detection", "monitoring_alerts"],
                    "properties": {
                        "valid_range": {
                            "type": ["array", "null"],
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "null_handling": {"enum": ["reject", "forward_fill", "interpolate"]},
                        "outlier_detection": {"enum": ["3_sigma", "iqr", "percentile", "custom"]},
                        "monitoring_alerts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1
                        }
                    }
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1
                },
                "lookback_period_days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3650
                },
                "update_frequency": {
                    "enum": ["real_time", "1min", "5min", "15min", "30min", "1h", "4h", "daily", "weekly"]
                },
                "pii_classification": {"enum": ["none", "masked", "anonymized", "sensitive"]},
                "audit_trail": {"enum": ["git_commits", "database_logs", "manual"]},
                "retention_policy": {
                    "enum": ["1_year", "2_years", "3_years", "5_years", "7_years", "10_years", "permanent"]
                },
                "created_at": {"type": "string", "format": "date-time"},
                "approved_at": {"type": "string", "format": "date-time"}
            }
        },
        
        "fundamental": {
            "type": "object",
            "required": [
                "feature_name", "feature_type", "data_source", "version",
                "as_of_ts_rule", "effective_ts_rule", "arrival_latency_minutes",
                "point_in_time_rule", "vendor_sla", "revision_policy",
                "computation_logic", "dependencies", "lookback_period_days",
                "update_frequency", "validation_rules", "pii_classification",
                "regulatory_notes", "audit_trail", "retention_policy",
                "created_by", "created_at", "approved_by", "approved_at"
            ],
            "properties": {
                "feature_name": {
                    "type": "string",
                    "pattern": "^[a-z][a-z0-9_]*[a-z0-9]$",
                    "minLength": 3,
                    "maxLength": 50
                },
                "feature_type": {"const": "fundamental"},
                "as_of_ts_rule": {
                    "enum": ["market_close_plus_1d", "next_day", "filing_date", "custom"]
                },
                "effective_ts_rule": {
                    "enum": ["report_period_end", "filing_date", "announcement_date"]
                },
                "arrival_latency_minutes": {
                    "type": "number",
                    "minimum": 60,  # Fundamental data has longer latency
                    "maximum": 10080  # 1 week max
                },
                "revision_policy": {
                    "type": "object",
                    "properties": {
                        "revision_frequency": {"enum": ["never", "daily", "weekly", "monthly", "quarterly"]},
                        "revision_window_days": {"type": "integer", "minimum": 0, "maximum": 365}
                    }
                },
                "lookback_period_days": {
                    "type": "integer",
                    "minimum": 90,   # Fundamental needs longer lookback
                    "maximum": 3650
                },
                "update_frequency": {
                    "enum": ["daily", "weekly", "monthly", "quarterly", "annual"]
                }
            }
        },
        
        "sentiment": {
            "type": "object",
            "required": [
                "feature_name", "feature_type", "data_source", "version",
                "as_of_ts_rule", "effective_ts_rule", "arrival_latency_minutes",
                "point_in_time_rule", "vendor_sla", "revision_policy",
                "computation_logic", "dependencies", "lookback_period_days",
                "update_frequency", "validation_rules", "pii_classification",
                "regulatory_notes", "audit_trail", "retention_policy",
                "created_by", "created_at", "approved_by", "approved_at"
            ],
            "properties": {
                "feature_name": {
                    "type": "string",
                    "pattern": "^[a-z][a-z0-9_]*[a-z0-9]$"
                },
                "feature_type": {"const": "sentiment"},
                "as_of_ts_rule": {
                    "enum": ["event_timestamp", "collection_time", "processing_time"]
                },
                "effective_ts_rule": {
                    "enum": ["event_timestamp", "publication_time", "collection_time"]
                },
                "pii_classification": {"enum": ["masked", "anonymized", "sensitive"]},  # Sentiment requires PII handling
                "validation_rules": {
                    "type": "object",
                    "properties": {
                        "valid_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "minimum": -1.0,
                            "maximum": 1.0
                        }
                    }
                }
            }
        },
        
        "macro": {
            "type": "object",
            "required": [
                "feature_name", "feature_type", "data_source", "version",
                "as_of_ts_rule", "effective_ts_rule", "arrival_latency_minutes",
                "point_in_time_rule", "vendor_sla", "revision_policy",
                "computation_logic", "dependencies", "lookback_period_days",
                "update_frequency", "validation_rules", "pii_classification",
                "regulatory_notes", "audit_trail", "retention_policy",
                "created_by", "created_at", "approved_by", "approved_at"
            ],
            "properties": {
                "feature_name": {
                    "type": "string",
                    "pattern": "^[a-z][a-z0-9_]*[a-z0-9]$"
                },
                "feature_type": {"const": "macro"},
                "as_of_ts_rule": {
                    "enum": ["announcement_time", "release_time", "market_close"]
                },
                "effective_ts_rule": {
                    "enum": ["effective_date", "announcement_time", "publication_time"]
                },
                "vendor_sla": {
                    "type": "object",
                    "properties": {
                        "availability": {"type": "number", "minimum": 99.0},  # Higher standard for macro
                        "quality": {"type": "number", "minimum": 99.0}  # Very high quality expected
                    }
                },
                "retention_policy": {"enum": ["10_years", "permanent"]},  # Longer retention for macro
                "lookback_period_days": {
                    "type": "integer",
                    "minimum": 365,  # Macro needs historical context
                    "maximum": 7300  # 20 years max
                }
            }
        },
        
        "options": {
            "type": "object",
            "required": [
                "feature_name", "feature_type", "data_source", "version",
                "as_of_ts_rule", "effective_ts_rule", "arrival_latency_minutes",
                "point_in_time_rule", "vendor_sla", "revision_policy",
                "computation_logic", "dependencies", "lookback_period_days",
                "update_frequency", "validation_rules", "pii_classification",
                "regulatory_notes", "audit_trail", "retention_policy",
                "created_by", "created_at", "approved_by", "approved_at"
            ],
            "properties": {
                "feature_name": {
                    "type": "string",
                    "pattern": "^[a-z][a-z0-9_]*[a-z0-9]$"
                },
                "feature_type": {"const": "options"},
                "as_of_ts_rule": {
                    "enum": ["quote_timestamp", "trade_timestamp", "market_close"]
                },
                "effective_ts_rule": {
                    "enum": ["quote_timestamp", "expiration_date", "settlement_time"]
                },
                "arrival_latency_minutes": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 5  # Options data should be very timely
                },
                "lookback_period_days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30  # Options have short lookback
                },
                "update_frequency": {
                    "enum": ["real_time", "1min", "5min", "15min", "1h"]  # High frequency only
                }
            }
        },
        
        "event": {
            "type": "object",
            "required": [
                "feature_name", "feature_type", "data_source", "version",
                "as_of_ts_rule", "effective_ts_rule", "arrival_latency_minutes",
                "point_in_time_rule", "vendor_sla", "revision_policy",
                "computation_logic", "dependencies", "lookback_period_days",
                "update_frequency", "validation_rules", "pii_classification",
                "regulatory_notes", "audit_trail", "retention_policy",
                "created_by", "created_at", "approved_by", "approved_at"
            ],
            "properties": {
                "feature_name": {
                    "type": "string",
                    "pattern": "^[a-z][a-z0-9_]*[a-z0-9]$"
                },
                "feature_type": {"const": "event"},
                "as_of_ts_rule": {
                    "enum": ["announcement_time", "market_close", "filing_time"]
                },
                "effective_ts_rule": {
                    "enum": ["announcement_time", "market_open_next", "immediate"]
                },
                "update_frequency": {"const": "event_driven"},  # Events only update when they occur
                "retention_policy": {"enum": ["7_years", "10_years", "permanent"]},  # Longer retention for events
                "lookback_period_days": {
                    "type": "integer",
                    "minimum": 365,  # Events need historical context
                    "maximum": 3650
                }
            }
        }
    }
    
    def __init__(self):
        """Initialize the validator."""
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.feature_registry: Set[str] = set()
        
    def validate_yaml_syntax(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Validate YAML syntax and load the contract."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                contract = yaml.safe_load(f)
            return contract
        except yaml.YAMLError as e:
            self.errors.append(f"YAML syntax error in {file_path}: {e}")
            return None
        except FileNotFoundError:
            self.errors.append(f"File not found: {file_path}")
            return None
        except Exception as e:
            self.errors.append(f"Error reading {file_path}: {e}")
            return None
    
    def validate_schema(self, contract: Dict[str, Any], file_path: str) -> bool:
        """Validate contract against feature type schema."""
        if not contract:
            return False
            
        feature_type = contract.get('feature_type')
        if not feature_type:
            self.errors.append(f"{file_path}: Missing required field 'feature_type'")
            return False
            
        if feature_type not in self.SCHEMAS:
            self.errors.append(f"{file_path}: Unsupported feature_type '{feature_type}'")
            return False
            
        schema = self.SCHEMAS[feature_type]
        validator = Draft7Validator(schema)
        
        validation_errors = []
        for error in validator.iter_errors(contract):
            validation_errors.append(f"  {error.json_path}: {error.message}")
            
        if validation_errors:
            self.errors.append(f"{file_path}: Schema validation errors:")
            self.errors.extend(validation_errors)
            return False
            
        return True
    
    def validate_business_rules(self, contract: Dict[str, Any], file_path: str) -> bool:
        """Validate business logic and consistency rules."""
        valid = True
        
        # Check feature name uniqueness
        feature_name = contract.get('feature_name', '')
        if feature_name in self.feature_registry:
            self.errors.append(f"{file_path}: Duplicate feature_name '{feature_name}'")
            valid = False
        else:
            self.feature_registry.add(feature_name)
        
        # Validate timestamp consistency
        created_at = contract.get('created_at')
        approved_at = contract.get('approved_at')
        
        if created_at and approved_at:
            try:
                created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                approved_dt = datetime.fromisoformat(approved_at.replace('Z', '+00:00'))
                
                if approved_dt < created_dt:
                    self.errors.append(f"{file_path}: approved_at cannot be before created_at")
                    valid = False
                    
            except ValueError as e:
                self.errors.append(f"{file_path}: Invalid timestamp format: {e}")
                valid = False
        
        # Validate SLA consistency
        vendor_sla = contract.get('vendor_sla', {})
        arrival_latency = contract.get('arrival_latency_minutes', 0)
        sla_latency = vendor_sla.get('latency_minutes', 0)
        
        if sla_latency < arrival_latency:
            self.warnings.append(
                f"{file_path}: SLA latency ({sla_latency}min) is less than "
                f"expected arrival latency ({arrival_latency}min)"
            )
        
        # Validate dependencies exist (if we have a registry)
        dependencies = contract.get('dependencies', [])
        for dep in dependencies:
            if not isinstance(dep, str) or not dep.strip():
                self.errors.append(f"{file_path}: Invalid dependency format: {dep}")
                valid = False
        
        # Feature-type specific validations
        feature_type = contract.get('feature_type')
        
        if feature_type == 'sentiment':
            # Validate sentiment range
            valid_range = contract.get('validation_rules', {}).get('valid_range')
            if valid_range and len(valid_range) == 2:
                if valid_range[0] < -1.0 or valid_range[1] > 1.0:
                    self.warnings.append(
                        f"{file_path}: Sentiment range {valid_range} extends beyond [-1, 1]"
                    )
        
        elif feature_type == 'options':
            # Validate options-specific rules
            lookback = contract.get('lookback_period_days', 0)
            if lookback > 30:
                self.warnings.append(
                    f"{file_path}: Options features typically don't need lookback > 30 days"
                )
        
        elif feature_type == 'fundamental':
            # Validate fundamental-specific rules
            update_freq = contract.get('update_frequency')
            if update_freq in ['real_time', '1min', '5min']:
                self.warnings.append(
                    f"{file_path}: Fundamental data rarely updates at {update_freq} frequency"
                )
        
        return valid
    
    def validate_contract_file(self, file_path: str) -> bool:
        """Validate a single contract file."""
        logger.info(f"Validating {file_path}")
        
        # Reset per-file state
        file_errors_start = len(self.errors)
        file_warnings_start = len(self.warnings)
        
        # Load and validate YAML
        contract = self.validate_yaml_syntax(file_path)
        if not contract:
            return False
        
        # Validate schema
        if not self.validate_schema(contract, file_path):
            return False
        
        # Validate business rules
        if not self.validate_business_rules(contract, file_path):
            return False
        
        # Check for any new errors/warnings
        new_errors = len(self.errors) - file_errors_start
        new_warnings = len(self.warnings) - file_warnings_start
        
        if new_errors == 0:
            logger.info(f"✅ {file_path} - Valid ({new_warnings} warnings)")
            return True
        else:
            logger.error(f"❌ {file_path} - {new_errors} errors, {new_warnings} warnings")
            return False
    
    def validate_all_contracts(self, contracts_dir: str) -> Tuple[int, int]:
        """Validate all contract files in a directory."""
        contracts_path = Path(contracts_dir)
        
        if not contracts_path.exists():
            logger.error(f"Directory not found: {contracts_dir}")
            return 0, 0
        
        # Find all YAML files
        yaml_files = list(contracts_path.rglob("*.yml")) + list(contracts_path.rglob("*.yaml"))
        
        # Exclude template files
        contract_files = [f for f in yaml_files if 'template' not in f.name.lower()]
        
        if not contract_files:
            logger.warning(f"No contract files found in {contracts_dir}")
            return 0, 0
        
        logger.info(f"Found {len(contract_files)} contract files to validate")
        
        valid_count = 0
        total_count = len(contract_files)
        
        for file_path in contract_files:
            if self.validate_contract_file(str(file_path)):
                valid_count += 1
        
        return valid_count, total_count
    
    def generate_validation_report(self, contracts_dir: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        valid_count, total_count = self.validate_all_contracts(contracts_dir)
        
        report = {
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "contracts_directory": contracts_dir,
            "summary": {
                "total_contracts": total_count,
                "valid_contracts": valid_count,
                "invalid_contracts": total_count - valid_count,
                "success_rate": (valid_count / total_count * 100) if total_count > 0 else 0
            },
            "errors": self.errors,
            "warnings": self.warnings,
            "feature_registry": sorted(list(self.feature_registry))
        }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Validation report saved to {output_file}")
        
        return report
    
    def print_summary(self):
        """Print validation summary to console."""
        if self.errors:
            logger.error(f"\n❌ VALIDATION FAILED: {len(self.errors)} errors found")
            for error in self.errors:
                logger.error(f"  {error}")
        
        if self.warnings:
            logger.warning(f"\n⚠️  {len(self.warnings)} warnings found")
            for warning in self.warnings:
                logger.warning(f"  {warning}")
        
        if not self.errors and not self.warnings:
            logger.info("\n✅ All contracts are valid!")


def create_ci_script():
    """Create CI integration script."""
    ci_script_content = '''#!/bin/bash
#
# Feature Contract Validation CI Script
# Run this in your CI/CD pipeline to validate feature contracts
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONTRACTS_DIR="$PROJECT_ROOT/docs/feature-contracts"
VALIDATOR_SCRIPT="$SCRIPT_DIR/feature_contract_validator.py"

echo "🔍 Validating feature contracts..."
echo "Contracts directory: $CONTRACTS_DIR"

# Create reports directory
mkdir -p "$PROJECT_ROOT/reports"

# Run validation
python "$VALIDATOR_SCRIPT" generate-report "$CONTRACTS_DIR" \
    --output "$PROJECT_ROOT/reports/feature_contract_validation.json"

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ Feature contract validation passed"
    exit 0
else
    echo "❌ Feature contract validation failed"
    echo "See validation report for details"
    exit 1
fi
'''
    
    ci_script_path = "E:/rony-data/trading-platform/tools/validate_contracts_ci.sh"
    with open(ci_script_path, 'w', encoding='utf-8') as f:
        f.write(ci_script_content)
    
    # Make executable on Unix systems
    try:
        os.chmod(ci_script_path, 0o755)
    except:
        pass  # Windows doesn't support chmod
    
    logger.info(f"CI script created: {ci_script_path}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Feature Contract Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python feature_contract_validator.py validate contract.yml
  python feature_contract_validator.py validate-all docs/feature-contracts/
  python feature_contract_validator.py generate-report docs/feature-contracts/ --output report.json
  python feature_contract_validator.py create-ci-script
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate single file
    validate_parser = subparsers.add_parser('validate', help='Validate a single contract file')
    validate_parser.add_argument('file', help='Path to contract file')
    
    # Validate all files
    validate_all_parser = subparsers.add_parser('validate-all', help='Validate all contracts in directory')
    validate_all_parser.add_argument('directory', help='Path to contracts directory')
    
    # Generate report
    report_parser = subparsers.add_parser('generate-report', help='Generate validation report')
    report_parser.add_argument('directory', help='Path to contracts directory')
    report_parser.add_argument('--output', '-o', help='Output file for report (JSON)')
    
    # Create CI script
    ci_parser = subparsers.add_parser('create-ci-script', help='Create CI integration script')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    validator = FeatureContractValidator()
    
    try:
        if args.command == 'validate':
            success = validator.validate_contract_file(args.file)
            validator.print_summary()
            return 0 if success else 1
            
        elif args.command == 'validate-all':
            valid_count, total_count = validator.validate_all_contracts(args.directory)
            validator.print_summary()
            
            if valid_count == total_count:
                logger.info(f"✅ All {total_count} contracts are valid")
                return 0
            else:
                logger.error(f"❌ {total_count - valid_count} of {total_count} contracts failed validation")
                return 1
                
        elif args.command == 'generate-report':
            report = validator.generate_validation_report(args.directory, args.output)
            validator.print_summary()
            
            success_rate = report['summary']['success_rate']
            if success_rate == 100:
                logger.info(f"✅ Validation report generated - 100% success rate")
                return 0
            else:
                logger.error(f"❌ Validation report generated - {success_rate:.1f}% success rate")
                return 1
                
        elif args.command == 'create-ci-script':
            create_ci_script()
            logger.info("✅ CI script created successfully")
            return 0
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())