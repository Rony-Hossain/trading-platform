#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8002}"
TARGET="${BASE_URL%/}/openapi.json"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts}"

mkdir -p "${ARTIFACT_DIR}"

echo "Running Schemathesis against ${TARGET}"
schemathesis run "${TARGET}" \
  --checks all \
  --workers=2 \
  --junit-xml "${ARTIFACT_DIR}/schemathesis.xml"
