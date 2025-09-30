#!/usr/bin/env python3
"""
PIT Contract Check - CI/CD Pipeline Script

This script enforces Point-in-Time feature contract compliance in CI/CD.
Blocks deployment if features are used without proper contracts.

Usage:
  python ci/pit_contract_check.py
  python ci/pit_contract_check.py --contracts-dir docs/feature-contracts
  python ci/pit_contract_check.py --strict --audit-only

Exit codes:
  0 - All contracts valid, compliance passed
  1 - Missing contracts or violations found
  2 - Script error or misconfiguration
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set
import re

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    # Add services/analysis-service to path
    analysis_service_path = Path(__file__).parent.parent / 'services' / 'analysis-service'
    sys.path.insert(0, str(analysis_service_path))
    
    from app.core.feature_contracts import (
        FeatureContractValidator, validate_contracts_directory
    )
except ImportError as e:
    print(f"Failed to import feature contracts module: {e}")
    print("Ensure you're running from project root and dependencies are installed")
    sys.exit(2)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PITComplianceChecker:
    """Comprehensive PIT contract compliance checker for CI/CD"""
    
    def __init__(self, contracts_dir: str = "docs/feature-contracts", strict: bool = False):
        self.contracts_dir = Path(contracts_dir)
        self.strict = strict
        self.validator = FeatureContractValidator(str(contracts_dir))
        self.violations: List[Dict] = []
        self.warnings: List[Dict] = []
        
    def scan_codebase_for_features(self) -> Set[str]:
        """Scan codebase for feature usage patterns"""
        feature_names = set()
        
        # Patterns to identify feature usage
        patterns = [
            r"feature[_\s]*[\"']([a-zA-Z_][a-zA-Z0-9_]*)[\"']",  # feature_name references
            r"[\"']([a-z_]+(?:_(?:iv|volume|price|ratio|delta|signal|score))?)[\"']",  # typical feature names
            r"\.get_feature\([\"']([^\"']+)[\"']\)",  # API calls
            r"features\[[\"']([^\"']+)[\"']\]",  # feature dictionary access
        ]
        
        # Directories to scan
        scan_dirs = [
            "services/analysis-service/app",
            "services/strategy-service/app", 
            "services/market-data-service/app",
            "services/fundamentals-service/app",
            "services/sentiment-service/app"
        ]
        
        for scan_dir in scan_dirs:
            scan_path = project_root / scan_dir
            if scan_path.exists():
                feature_names.update(self._scan_directory(scan_path, patterns))
                
        # Filter out obvious non-features
        filtered_features = set()
        for name in feature_names:
            if self._is_likely_feature(name):
                filtered_features.add(name)
                
        return filtered_features
    
    def _scan_directory(self, directory: Path, patterns: List[str]) -> Set[str]:
        """Scan directory for feature name patterns"""
        features = set()
        
        for py_file in directory.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    features.update(matches)
                    
            except Exception as e:
                logger.warning(f"Could not scan {py_file}: {e}")
                
        return features
    
    def _is_likely_feature(self, name: str) -> bool:
        """Filter out non-feature strings"""
        # Skip short names, common words, and obvious non-features
        skip_patterns = [
            r'^[a-z]{1,2}$',  # Very short names
            r'^\d+$',         # Pure numbers
            r'^(id|key|name|type|value|data|info|test|debug)$',  # Common variables
            r'^(get|set|post|put|delete|create|update)$',        # HTTP methods
            r'^(true|false|null|none|error|success)$',           # Common values
        ]
        
        for pattern in skip_patterns:
            if re.match(pattern, name, re.IGNORECASE):
                return False
                
        # Include if it looks like a feature name
        feature_indicators = [
            '_price', '_volume', '_iv', '_delta', '_signal', '_score',
            '_ratio', '_spread', '_volatility', '_momentum', '_trend',
            'vix_', 'spy_', 'qqq_', 'sentiment_', 'macro_', 'technical_'
        ]
        
        return any(indicator in name.lower() for indicator in feature_indicators)
    
    def check_missing_contracts(self) -> bool:
        """Check for features without contracts"""
        logger.info("Scanning codebase for feature usage...")
        used_features = self.scan_codebase_for_features()
        
        logger.info(f"Found {len(used_features)} potential features in codebase")
        logger.info(f"Loaded {len(self.validator.contracts)} feature contracts")
        
        missing_contracts = self.validator.get_missing_contracts(list(used_features))
        
        if missing_contracts:
            self.violations.append({
                "type": "missing_contracts",
                "severity": "critical",
                "count": len(missing_contracts),
                "features": missing_contracts[:10],  # Limit output
                "message": f"{len(missing_contracts)} features found without contracts"
            })
            
            logger.error(f"Missing contracts for {len(missing_contracts)} features:")
            for feature in missing_contracts[:20]:  # Show first 20
                logger.error(f"  - {feature}")
                
            return False
            
        logger.info("✓ All discovered features have contracts")
        return True
    
    def validate_contract_completeness(self) -> bool:
        """Validate that all contracts are complete and valid"""
        logger.info("Validating contract completeness...")
        
        all_valid = True
        required_fields = [
            'feature_name', 'feature_type', 'data_source', 'version',
            'as_of_ts_rule', 'effective_ts_rule', 'arrival_latency_minutes'
        ]
        
        for feature_name, contract in self.validator.contracts.items():
            # Check required fields
            missing_fields = []
            for field in required_fields:
                if not hasattr(contract, field) or getattr(contract, field) is None:
                    missing_fields.append(field)
                    
            if missing_fields:
                self.violations.append({
                    "type": "incomplete_contract",
                    "severity": "high",
                    "feature": feature_name,
                    "missing_fields": missing_fields,
                    "message": f"Contract for {feature_name} missing required fields"
                })
                all_valid = False
                
            # Validate latency is reasonable (not negative, not excessive)
            if hasattr(contract, 'arrival_latency_minutes'):
                latency = contract.arrival_latency_minutes
                if latency < 0:
                    self.violations.append({
                        "type": "invalid_latency",
                        "severity": "medium",
                        "feature": feature_name,
                        "value": latency,
                        "message": f"Negative latency not allowed: {latency}"
                    })
                    all_valid = False
                elif latency > 10080:  # > 1 week
                    self.warnings.append({
                        "type": "high_latency",
                        "severity": "low",
                        "feature": feature_name,
                        "value": latency,
                        "message": f"Unusually high latency: {latency} minutes"
                    })
        
        if all_valid:
            logger.info("✓ All contracts are complete and valid")
        else:
            logger.error(f"Found {len([v for v in self.violations if v['type'] in ['incomplete_contract', 'invalid_latency']])} contract validation errors")
            
        return all_valid
    
    def check_dependency_contracts(self) -> bool:
        """Ensure all feature dependencies have contracts"""
        logger.info("Checking feature dependency contracts...")
        
        all_deps_valid = True
        for feature_name, contract in self.validator.contracts.items():
            for dep in contract.dependencies:
                if dep not in self.validator.contracts:
                    self.violations.append({
                        "type": "missing_dependency_contract",
                        "severity": "medium",
                        "feature": feature_name,
                        "dependency": dep,
                        "message": f"Dependency {dep} of {feature_name} lacks contract"
                    })
                    all_deps_valid = False
                    
        if all_deps_valid:
            logger.info("✓ All feature dependencies have contracts")
        else:
            dep_violations = [v for v in self.violations if v['type'] == 'missing_dependency_contract']
            logger.error(f"Found {len(dep_violations)} missing dependency contracts")
            
        return all_deps_valid
    
    def generate_compliance_report(self) -> Dict:
        """Generate comprehensive compliance report"""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "compliance_status": "PASS" if not self.violations else "FAIL",
            "summary": {
                "total_contracts": len(self.validator.contracts),
                "violations": len(self.violations),
                "warnings": len(self.warnings),
                "critical_violations": len([v for v in self.violations if v['severity'] == 'critical']),
                "high_violations": len([v for v in self.violations if v['severity'] == 'high']),
            },
            "violations": self.violations,
            "warnings": self.warnings,
            "contracts": self.validator.generate_audit_report()
        }
        
        return report
    
    def run_compliance_check(self) -> bool:
        """Run complete compliance check"""
        logger.info("Starting PIT contract compliance check...")
        
        # Check if contracts directory exists
        if not self.contracts_dir.exists():
            logger.error(f"Contracts directory not found: {self.contracts_dir}")
            self.violations.append({
                "type": "missing_contracts_directory",
                "severity": "critical",
                "message": f"Contracts directory not found: {self.contracts_dir}"
            })
            return False
        
        # Run all checks
        checks = [
            ("Missing Contracts", self.check_missing_contracts),
            ("Contract Completeness", self.validate_contract_completeness),
            ("Dependency Contracts", self.check_dependency_contracts),
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            logger.info(f"Running check: {check_name}")
            try:
                if not check_func():
                    all_passed = False
            except Exception as e:
                logger.error(f"Check {check_name} failed with error: {e}")
                self.violations.append({
                    "type": "check_error",
                    "severity": "critical",
                    "check": check_name,
                    "error": str(e),
                    "message": f"Error running check: {check_name}"
                })
                all_passed = False
        
        # Generate report
        report = self.generate_compliance_report()
        
        # Save report
        report_file = project_root / "artifacts" / "audits" / f"pit_contracts_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Compliance report saved to: {report_file}")
        
        # Log summary
        if all_passed:
            logger.info("✅ PIT contract compliance check PASSED")
        else:
            logger.error("❌ PIT contract compliance check FAILED")
            logger.error(f"Found {len(self.violations)} violations:")
            for violation in self.violations:
                logger.error(f"  - {violation['severity'].upper()}: {violation['message']}")
        
        return all_passed

def main():
    """Main entry point for CI script"""
    parser = argparse.ArgumentParser(
        description="PIT Contract Compliance Checker for CI/CD"
    )
    parser.add_argument(
        "--contracts-dir", 
        default="docs/feature-contracts",
        help="Directory containing feature contracts"
    )
    parser.add_argument(
        "--strict", 
        action="store_true",
        help="Enable strict mode (warnings become errors)"
    )
    parser.add_argument(
        "--audit-only", 
        action="store_true",
        help="Generate audit report without failing on violations"
    )
    
    args = parser.parse_args()
    
    # Set environment variable if not set
    if not os.getenv("PIT_CONTRACTS_ENFORCE"):
        os.environ["PIT_CONTRACTS_ENFORCE"] = "true"
    
    # Run compliance check
    checker = PITComplianceChecker(
        contracts_dir=args.contracts_dir,
        strict=args.strict
    )
    
    compliance_passed = checker.run_compliance_check()
    
    # Handle strict mode (warnings become errors)
    if args.strict and checker.warnings:
        logger.error(f"Strict mode: {len(checker.warnings)} warnings treated as errors")
        compliance_passed = False
    
    # Exit with appropriate code
    if args.audit_only:
        logger.info("Audit-only mode: exiting with success regardless of violations")
        sys.exit(0)
    elif compliance_passed:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()