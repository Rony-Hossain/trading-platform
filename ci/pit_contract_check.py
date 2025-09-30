#!/usr/bin/env python3
"""
PIT Contract Check for CI/CD Pipeline

This script validates feature contracts and ensures PIT compliance for all
features used in the trading platform. It can be run as a pre-commit hook
or in GitHub Actions to enforce contract requirements.

Usage:
    python ci/pit_contract_check.py --validate-all
    python ci/pit_contract_check.py --check-features path/to/features.py
    python ci/pit_contract_check.py --pre-commit
"""

import argparse
import sys
import os
import json
import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass

# Add services to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services', 'analysis-service', 'app'))

from core.feature_contracts import (
    FeatureContractValidator, 
    ViolationType, 
    PITViolation,
    PitComplianceError
)


@dataclass
class CheckResult:
    """Result of contract check"""
    success: bool
    violations: List[PITViolation]
    missing_contracts: List[str]
    errors: List[str]
    
    @property
    def exit_code(self) -> int:
        """Return appropriate exit code for CI/CD"""
        if not self.success:
            return 1
        return 0


class FeatureExtractor:
    """Extract feature usage from Python code"""
    
    def __init__(self):
        self.feature_patterns = [
            # Direct feature access patterns
            r"features\[[\'\"]([a-z_][a-z0-9_]*)[\'\"]",  # features['feature_name']
            r"get_feature\([\'\"]([a-z_][a-z0-9_]*)[\'\"]",  # get_feature('feature_name')
            r"feature_id\s*=\s*[\'\"]([a-z_][a-z0-9_]*)[\'\"]",  # feature_id = 'feature_name'
            
            # Data pipeline patterns
            r"\.load\([\'\"]([a-z_][a-z0-9_]*)[\'\"]",  # dataloader.load('feature_name')
            r"feature_name\s*:\s*[\'\"]([a-z_][a-z0-9_]*)[\'\"]",  # feature_name: 'feature_name'
            
            # ML pipeline patterns
            r"X\[[\'\"]([a-z_][a-z0-9_]*)[\'\"]",  # X['feature_name']
            r"\.select\([\'\"]([a-z_][a-z0-9_]*)[\'\"]",  # df.select('feature_name')
        ]
    
    def extract_from_file(self, file_path: Path) -> Set[str]:
        """Extract feature IDs from a Python file"""
        features = set()
        
        if not file_path.exists() or file_path.suffix != '.py':
            return features
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Use regex patterns
            for pattern in self.feature_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                features.update(matches)
                
            # Also try AST parsing for more complex cases
            features.update(self._extract_from_ast(content))
            
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")
            
        return features
    
    def _extract_from_ast(self, content: str) -> Set[str]:
        """Extract features using AST parsing"""
        features = set()
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Look for string literals that look like feature names
                if isinstance(node, ast.Str):
                    value = node.s
                    if isinstance(value, str) and self._looks_like_feature_id(value):
                        features.add(value)
                        
                # Look for dictionary access with string keys
                elif isinstance(node, ast.Subscript):
                    if isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Str):
                        value = node.slice.value.s
                        if isinstance(value, str) and self._looks_like_feature_id(value):
                            features.add(value)
                            
        except Exception:
            pass  # AST parsing failed, regex should catch most cases
            
        return features
    
    def _looks_like_feature_id(self, value: str) -> bool:
        """Check if a string looks like a feature ID"""
        # Feature IDs should match our contract pattern
        pattern = r"^[a-z][a-z0-9_]*[a-z0-9]$"
        return re.match(pattern, value) is not None and len(value) > 2
    
    def extract_from_directory(self, directory: Path, 
                              exclude_patterns: List[str] = None) -> Dict[str, Set[str]]:
        """Extract features from all Python files in directory"""
        exclude_patterns = exclude_patterns or ['test_*', '*_test.py', '__pycache__/*']
        
        file_features = {}
        
        for py_file in directory.rglob("*.py"):
            # Skip excluded files
            skip = False
            for pattern in exclude_patterns:
                if py_file.match(pattern):
                    skip = True
                    break
            
            if skip:
                continue
                
            features = self.extract_from_file(py_file)
            if features:
                file_features[str(py_file)] = features
                
        return file_features


class PITContractChecker:
    """Main PIT contract checker for CI/CD"""
    
    def __init__(self, enforce_contracts: bool = True):
        self.validator = FeatureContractValidator(enforce_pit_compliance=enforce_contracts)
        self.extractor = FeatureExtractor()
        self.contracts_dir = Path("docs/feature-contracts/contracts")
        
    def load_contracts(self) -> int:
        """Load all available feature contracts"""
        if not self.contracts_dir.exists():
            print(f"Warning: Contracts directory {self.contracts_dir} does not exist")
            return 0
            
        return self.validator.load_contracts_from_directory(str(self.contracts_dir))
    
    def validate_all_contracts(self) -> CheckResult:
        """Validate all contracts in the contracts directory"""
        violations = []
        errors = []
        
        contracts_loaded = self.load_contracts()
        print(f"Loaded {contracts_loaded} feature contracts")
        
        if contracts_loaded == 0:
            errors.append("No feature contracts found to validate")
            return CheckResult(
                success=False,
                violations=[],
                missing_contracts=[],
                errors=errors
            )
        
        # Validate each contract
        for feature_id, contract in self.validator.contracts.items():
            try:
                # Check temporal consistency
                temporal_violations = contract.validate_temporal_consistency()
                if temporal_violations:
                    for violation_desc in temporal_violations:
                        violations.append(PITViolation(
                            violation_type=ViolationType.HIGH,
                            feature_id=feature_id,
                            description=f"Temporal consistency: {violation_desc}",
                            timestamp=None
                        ))
                        
            except Exception as e:
                errors.append(f"Contract validation failed for {feature_id}: {str(e)}")
        
        success = len(violations) == 0 and len(errors) == 0
        
        return CheckResult(
            success=success,
            violations=violations,
            missing_contracts=[],
            errors=errors
        )
    
    def check_feature_usage(self, target_paths: List[str]) -> CheckResult:
        """Check that all used features have contracts"""
        violations = []
        errors = []
        missing_contracts = []
        
        # Load existing contracts
        contracts_loaded = self.load_contracts()
        print(f"Loaded {contracts_loaded} feature contracts")
        
        # Extract features from target paths
        all_features = set()
        
        for target_path in target_paths:
            path = Path(target_path)
            
            if path.is_file():
                features = self.extractor.extract_from_file(path)
                all_features.update(features)
                print(f"Found {len(features)} features in {path}")
                
            elif path.is_dir():
                file_features = self.extractor.extract_from_directory(path)
                for file_path, features in file_features.items():
                    all_features.update(features)
                    print(f"Found {len(features)} features in {file_path}")
                    
            else:
                errors.append(f"Path not found: {target_path}")
        
        print(f"Total unique features found: {len(all_features)}")
        
        # Check contracts for all features
        for feature_id in all_features:
            contract = self.validator.get_contract(feature_id)
            
            if not contract:
                missing_contracts.append(feature_id)
                violations.append(PITViolation(
                    violation_type=ViolationType.CRITICAL,
                    feature_id=feature_id,
                    description=f"No contract found for feature: {feature_id}",
                    timestamp=None
                ))
        
        success = len(violations) == 0 and len(errors) == 0
        
        return CheckResult(
            success=success,
            violations=violations,
            missing_contracts=missing_contracts,
            errors=errors
        )
    
    def pre_commit_check(self) -> CheckResult:
        """Run pre-commit checks (git staged files only)"""
        # Get staged files
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACMR'],
                capture_output=True, text=True, check=True
            )
            staged_files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
            
        except Exception as e:
            return CheckResult(
                success=False,
                violations=[],
                missing_contracts=[],
                errors=[f"Failed to get staged files: {str(e)}"]
            )
        
        # Filter Python files
        python_files = [f for f in staged_files if f.endswith('.py')]
        
        if not python_files:
            return CheckResult(success=True, violations=[], missing_contracts=[], errors=[])
        
        print(f"Checking {len(python_files)} staged Python files for PIT compliance")
        
        return self.check_feature_usage(python_files)
    
    def print_results(self, result: CheckResult):
        """Print check results in a readable format"""
        if result.success:
            print("[SUCCESS] All PIT contract checks passed!")
            return
            
        print("[FAILED] PIT contract checks failed:")
        print()
        
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  * {error}")
            print()
        
        if result.missing_contracts:
            print("Missing Contracts:")
            for feature_id in result.missing_contracts:
                print(f"  * {feature_id}")
            print()
            print("Create contracts using the template at docs/feature-contracts/CONTRACT_TEMPLATE.md")
            print()
        
        if result.violations:
            print("Violations:")
            for violation in result.violations:
                print(f"  * [{violation.violation_type.value.upper()}] {violation.feature_id}: {violation.description}")
            print()
        
        print("Fix these issues before proceeding with deployment.")


def main():
    parser = argparse.ArgumentParser(description="PIT Contract Checker for CI/CD")
    
    parser.add_argument(
        '--validate-all', 
        action='store_true',
        help='Validate all contracts in contracts directory'
    )
    
    parser.add_argument(
        '--check-features',
        nargs='+',
        help='Check feature usage in specified files/directories'
    )
    
    parser.add_argument(
        '--pre-commit',
        action='store_true', 
        help='Run pre-commit checks on staged files'
    )
    
    parser.add_argument(
        '--contracts-dir',
        default='docs/feature-contracts/contracts',
        help='Directory containing feature contracts'
    )
    
    parser.add_argument(
        '--no-enforce',
        action='store_true',
        help='Disable strict enforcement (warnings only)'
    )
    
    parser.add_argument(
        '--json-output',
        action='store_true',
        help='Output results in JSON format'
    )
    
    args = parser.parse_args()
    
    # Set enforcement level
    enforce_contracts = not args.no_enforce
    if 'PIT_CONTRACTS_ENFORCE' in os.environ:
        enforce_contracts = os.environ['PIT_CONTRACTS_ENFORCE'].lower() in ('true', '1', 'yes')
    
    # Create checker
    checker = PITContractChecker(enforce_contracts=enforce_contracts)
    checker.contracts_dir = Path(args.contracts_dir)
    
    # Run appropriate check
    if args.validate_all:
        result = checker.validate_all_contracts()
        
    elif args.check_features:
        result = checker.check_feature_usage(args.check_features)
        
    elif args.pre_commit:
        result = checker.pre_commit_check()
        
    else:
        # Default: check both contract validation and feature usage
        print("Running full PIT contract validation...")
        
        # First validate contracts
        contract_result = checker.validate_all_contracts()
        
        # Then check feature usage in services
        services_dir = ['services/']
        usage_result = checker.check_feature_usage(services_dir)
        
        # Combine results
        result = CheckResult(
            success=contract_result.success and usage_result.success,
            violations=contract_result.violations + usage_result.violations,
            missing_contracts=usage_result.missing_contracts,
            errors=contract_result.errors + usage_result.errors
        )
    
    # Output results
    if args.json_output:
        output = {
            "success": result.success,
            "violations": [v.to_dict() for v in result.violations],
            "missing_contracts": result.missing_contracts,
            "errors": result.errors
        }
        print(json.dumps(output, indent=2))
    else:
        checker.print_results(result)
    
    # Exit with appropriate code
    sys.exit(result.exit_code)


if __name__ == "__main__":
    main()

