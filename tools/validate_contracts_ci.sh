#!/bin/bash
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
python "$VALIDATOR_SCRIPT" generate-report "$CONTRACTS_DIR"     --output "$PROJECT_ROOT/reports/feature_contract_validation.json"

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ Feature contract validation passed"
    exit 0
else
    echo "❌ Feature contract validation failed"
    echo "See validation report for details"
    exit 1
fi
