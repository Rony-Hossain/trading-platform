#!/bin/bash
# Pre-Deployment Verification Script for Market Data Service
# Automates the production readiness checklist

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_WARNED=0

echo "=========================================="
echo "Market Data Service - Pre-Deployment Check"
echo "=========================================="
echo ""

# Helper functions
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((CHECKS_PASSED++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((CHECKS_FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((CHECKS_WARNED++))
}

# Configuration
SERVICE_URL="${SERVICE_URL:-http://localhost:8000}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-market_data}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"

echo "Configuration:"
echo "  Service URL: $SERVICE_URL"
echo "  PostgreSQL: $POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB"
echo "  Redis: $REDIS_HOST:$REDIS_PORT"
echo "  Prometheus: $PROMETHEUS_URL"
echo ""

# ==========================================
# 1. Database Checks
# ==========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Database Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check PostgreSQL connectivity
if pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" > /dev/null 2>&1; then
    check_pass "PostgreSQL is accessible"
else
    check_fail "PostgreSQL is NOT accessible"
fi

# Check if database exists
if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -lqt | cut -d \| -f 1 | grep -qw "$POSTGRES_DB"; then
    check_pass "Database '$POSTGRES_DB' exists"
else
    check_fail "Database '$POSTGRES_DB' does NOT exist"
fi

# Check if TimescaleDB extension is enabled
if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "SELECT COUNT(*) FROM pg_extension WHERE extname='timescaledb';" | grep -q "1"; then
    check_pass "TimescaleDB extension enabled"
else
    check_warn "TimescaleDB extension NOT enabled (standard PostgreSQL mode)"
fi

# Check required tables exist
REQUIRED_TABLES=("candles_intraday" "quotes_l1" "symbol_universe" "ingestion_cursor" "backfill_jobs")
for table in "${REQUIRED_TABLES[@]}"; do
    if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='$table';" | grep -q "1"; then
        check_pass "Table '$table' exists"
    else
        check_fail "Table '$table' does NOT exist"
    fi
done

# Check if symbol universe is populated
SYMBOL_COUNT=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "SELECT COUNT(*) FROM symbol_universe WHERE active=true;" 2>/dev/null || echo "0")
if [ "$SYMBOL_COUNT" -ge 100 ]; then
    check_pass "Symbol universe populated ($SYMBOL_COUNT active symbols)"
elif [ "$SYMBOL_COUNT" -gt 0 ]; then
    check_warn "Symbol universe has only $SYMBOL_COUNT symbols (minimum 100 recommended)"
else
    check_fail "Symbol universe is EMPTY (minimum 100 symbols required)"
fi

echo ""

# ==========================================
# 2. Redis Checks
# ==========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Redis Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Redis connectivity
if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping > /dev/null 2>&1; then
    check_pass "Redis is accessible"
else
    check_fail "Redis is NOT accessible"
fi

# Check Redis memory
REDIS_MEMORY=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" INFO memory | grep "used_memory_human" | cut -d: -f2 | tr -d '\r' || echo "unknown")
echo "  Redis memory usage: $REDIS_MEMORY"

echo ""

# ==========================================
# 3. Service Health Checks
# ==========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. Service Health Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check service health endpoint
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL/health" || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    check_pass "Service health check passing (HTTP 200)"
else
    check_fail "Service health check FAILED (HTTP $HTTP_CODE)"
fi

# Check metrics endpoint
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL/metrics" || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    check_pass "Metrics endpoint accessible (HTTP 200)"
else
    check_fail "Metrics endpoint NOT accessible (HTTP $HTTP_CODE)"
fi

# Check SLO metrics are being reported
if curl -s "$SERVICE_URL/metrics" | grep -q "slo_gap_violation_rate"; then
    check_pass "SLO metrics are being reported"
else
    check_warn "SLO metrics NOT found (SLO monitor may not be started yet)"
fi

# Check provider metrics
if curl -s "$SERVICE_URL/metrics" | grep -q "provider_selected_total"; then
    check_pass "Provider metrics are being reported"
else
    check_fail "Provider metrics NOT found"
fi

echo ""

# ==========================================
# 4. Configuration Checks
# ==========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. Configuration Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check environment variables
REQUIRED_VARS=("DATABASE_URL" "POLICY_BARS_1M")
for var in "${REQUIRED_VARS[@]}"; do
    if [ -n "${!var}" ]; then
        check_pass "Environment variable $var is set"
    else
        check_warn "Environment variable $var is NOT set"
    fi
done

# Check config reload endpoint
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$SERVICE_URL/ops/config/reload" || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    check_pass "Config hot-reload endpoint working"
elif [ "$HTTP_CODE" = "404" ]; then
    check_warn "Config hot-reload endpoint not implemented (manual patch required)"
else
    check_fail "Config hot-reload endpoint FAILED (HTTP $HTTP_CODE)"
fi

echo ""

# ==========================================
# 5. Monitoring Stack Checks
# ==========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. Monitoring Stack Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Prometheus is accessible
if curl -s "$PROMETHEUS_URL/-/healthy" > /dev/null 2>&1; then
    check_pass "Prometheus is accessible"

    # Check if Prometheus is scraping our service
    if curl -s "$PROMETHEUS_URL/api/v1/targets" | grep -q "market-data"; then
        check_pass "Prometheus is scraping market-data-service"
    else
        check_warn "Prometheus is NOT scraping market-data-service yet"
    fi
else
    check_warn "Prometheus is NOT accessible (check PROMETHEUS_URL)"
fi

# Check if using Kubernetes
if command -v kubectl &> /dev/null; then
    echo "  Kubernetes detected, checking resources..."

    # Check ServiceMonitors
    SM_COUNT=$(kubectl get servicemonitor -A | grep -c "market-data" || echo "0")
    if [ "$SM_COUNT" -ge 1 ]; then
        check_pass "ServiceMonitor(s) deployed ($SM_COUNT found)"
    else
        check_warn "No ServiceMonitors found for market-data"
    fi

    # Check PrometheusRules
    PR_COUNT=$(kubectl get prometheusrule -A | grep -c "market-data" || echo "0")
    if [ "$PR_COUNT" -ge 1 ]; then
        check_pass "PrometheusRule(s) deployed ($PR_COUNT found)"
    else
        check_warn "No PrometheusRules found for market-data"
    fi

    # Check Blackbox Exporter
    if kubectl get pods -A | grep -q "blackbox-exporter"; then
        check_pass "Blackbox Exporter deployed"
    else
        check_warn "Blackbox Exporter NOT deployed"
    fi
else
    check_warn "kubectl not found (skipping Kubernetes checks)"
fi

echo ""

# ==========================================
# 6. DLQ Admin Checks
# ==========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. DLQ Admin Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check DLQ endpoints
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL/ops/dlq/stats" || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    check_pass "DLQ admin endpoints accessible"

    # Check failed job count
    FAILED_COUNT=$(curl -s "$SERVICE_URL/ops/dlq/stats" | grep -o '"total_failed":[0-9]*' | cut -d: -f2 || echo "unknown")
    if [ "$FAILED_COUNT" != "unknown" ]; then
        echo "  Failed jobs in DLQ: $FAILED_COUNT"
    fi
elif [ "$HTTP_CODE" = "404" ]; then
    check_warn "DLQ admin endpoints not implemented (manual patch required)"
else
    check_fail "DLQ admin endpoints FAILED (HTTP $HTTP_CODE)"
fi

echo ""

# ==========================================
# 7. Provider Checks
# ==========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7. Provider Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check provider health history endpoint
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL/providers/polygon/health-history" || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    check_pass "Provider health history tracking enabled"
elif [ "$HTTP_CODE" = "404" ]; then
    check_warn "Provider health history endpoint not implemented (manual patch required)"
else
    check_warn "Provider health history check inconclusive (HTTP $HTTP_CODE)"
fi

# Check circuit breaker metrics
if curl -s "$SERVICE_URL/metrics" | grep -q "circuit_state"; then
    check_pass "Circuit breaker metrics found"
else
    check_fail "Circuit breaker metrics NOT found"
fi

echo ""

# ==========================================
# 8. Test Suite Checks
# ==========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "8. Test Suite Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if pytest is available
if command -v pytest &> /dev/null; then
    check_pass "pytest is installed"

    # Run tests if requested
    if [ "${RUN_TESTS}" = "true" ]; then
        echo "  Running tests..."
        if pytest tests/ -v --tb=short > /tmp/test-results.txt 2>&1; then
            check_pass "All tests PASSED"
        else
            check_fail "Some tests FAILED (see /tmp/test-results.txt)"
        fi
    else
        echo "  Skipping test run (set RUN_TESTS=true to run)"
    fi
else
    check_warn "pytest not installed (tests cannot be run)"
fi

echo ""

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "Pre-Deployment Check Summary"
echo "=========================================="
echo -e "${GREEN}Passed:${NC}  $CHECKS_PASSED"
echo -e "${YELLOW}Warnings:${NC} $CHECKS_WARNED"
echo -e "${RED}Failed:${NC}  $CHECKS_FAILED"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    if [ $CHECKS_WARNED -eq 0 ]; then
        echo -e "${GREEN}✓ ALL CHECKS PASSED - READY FOR DEPLOYMENT${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠ CHECKS PASSED WITH WARNINGS - REVIEW BEFORE DEPLOYMENT${NC}"
        exit 0
    fi
else
    echo -e "${RED}✗ CHECKS FAILED - NOT READY FOR DEPLOYMENT${NC}"
    echo ""
    echo "Fix the failed checks before deploying to production."
    exit 1
fi
