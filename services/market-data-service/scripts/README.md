# Deployment Scripts

This directory contains automation scripts for deploying and verifying the Market Data Service.

---

## 📋 Pre-Deployment Check Script

Automates the production readiness checklist from [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md).

### Usage

**Bash version (Linux/Mac):**
```bash
chmod +x scripts/pre-deployment-check.sh
./scripts/pre-deployment-check.sh
```

**Python version (Cross-platform):**
```bash
python scripts/pre_deployment_check.py
```

### Configuration

Set environment variables to customize checks:

```bash
# Service configuration
export SERVICE_URL=http://localhost:8000

# Database configuration
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=market_data
export POSTGRES_USER=postgres

# Redis configuration
export REDIS_HOST=localhost
export REDIS_PORT=6379

# Prometheus configuration
export PROMETHEUS_URL=http://localhost:9090

# Run tests as part of checks
export RUN_TESTS=true

# Run the script
python scripts/pre_deployment_check.py
```

### Checks Performed

The script validates **18 production readiness criteria**:

#### 1. Database Checks (6 checks)
- ✓ PostgreSQL connectivity
- ✓ Database exists
- ✓ TimescaleDB extension enabled
- ✓ Required tables exist (candles_intraday, quotes_l1, etc.)
- ✓ Symbol universe populated (≥100 symbols)

#### 2. Redis Checks (1 check)
- ✓ Redis connectivity and health

#### 3. Service Health Checks (4 checks)
- ✓ `/health` endpoint passing
- ✓ `/metrics` endpoint accessible
- ✓ SLO metrics being reported
- ✓ Provider metrics being reported

#### 4. Configuration Checks (2 checks)
- ✓ Required environment variables set
- ✓ Config hot-reload endpoint working

#### 5. Monitoring Stack Checks (4 checks)
- ✓ Prometheus accessible
- ✓ Prometheus scraping service
- ✓ ServiceMonitors deployed (Kubernetes)
- ✓ PrometheusRules deployed (Kubernetes)
- ✓ Blackbox Exporter deployed (Kubernetes)

#### 6. DLQ Admin Checks (1 check)
- ✓ DLQ admin endpoints accessible
- ✓ Failed job count reporting

#### 7. Provider Checks (2 checks)
- ✓ Provider health history tracking
- ✓ Circuit breaker metrics

#### 8. Test Suite Checks (1 check)
- ✓ pytest installed
- ✓ All tests passing (optional with `RUN_TESTS=true`)

### Output Example

```
==================================================
Market Data Service - Pre-Deployment Check
==================================================

Configuration:
  Service URL: http://localhost:8000
  PostgreSQL: localhost:5432/market_data
  Redis: localhost:6379
  Prometheus: http://localhost:9090

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Database Checks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ PostgreSQL accessible
✓ Database 'market_data' exists
✓ TimescaleDB extension enabled
✓ Table 'candles_intraday' exists
✓ Table 'quotes_l1' exists
✓ Table 'symbol_universe' exists
✓ Table 'ingestion_cursor' exists
✓ Table 'backfill_jobs' exists
✓ Symbol universe populated - 8432 active symbols

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. Redis Checks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Redis accessible
  Redis memory usage: 15.23M

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. Service Health Checks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Service health check passing - HTTP 200
✓ Metrics endpoint accessible - HTTP 200
✓ SLO metrics are being reported
✓ Provider metrics are being reported

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. Configuration Checks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Environment variable DATABASE_URL set
✓ Environment variable POLICY_BARS_1M set
✓ Config hot-reload endpoint working

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. Monitoring Stack Checks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Prometheus accessible
✓ Prometheus scraping market-data-service
  Kubernetes detected, checking resources...
✓ ServiceMonitor(s) deployed - 2 found
✓ PrometheusRule(s) deployed - 1 found
✓ Blackbox Exporter deployed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. DLQ Admin Checks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ DLQ admin endpoints accessible
  Failed jobs in DLQ: 0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. Provider Checks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Provider health history tracking enabled
✓ Circuit breaker metrics found

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. Test Suite Checks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ pytest installed
  Skipping test run (set RUN_TESTS=true to run)

==================================================
Pre-Deployment Check Summary
==================================================
Passed:  25
Warnings: 0
Failed:  0

✓ ALL CHECKS PASSED - READY FOR DEPLOYMENT
```

### Exit Codes

- `0` - All checks passed (ready for deployment)
- `1` - Some checks failed (not ready)

### Integration with CI/CD

**GitHub Actions:**
```yaml
name: Pre-Deployment Check

on:
  push:
    branches: [main]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt

      - name: Start services
        run: docker-compose up -d

      - name: Run pre-deployment checks
        env:
          SERVICE_URL: http://localhost:8000
          POSTGRES_HOST: localhost
          REDIS_HOST: localhost
          RUN_TESTS: true
        run: python scripts/pre_deployment_check.py
```

**GitLab CI:**
```yaml
pre-deployment-check:
  stage: verify
  image: python:3.11
  services:
    - postgres:15
    - redis:7
  variables:
    SERVICE_URL: "http://localhost:8000"
    POSTGRES_HOST: "postgres"
    REDIS_HOST: "redis"
    RUN_TESTS: "true"
  script:
    - pip install -r requirements.txt -r requirements-test.txt
    - python scripts/pre_deployment_check.py
```

**Jenkins:**
```groovy
stage('Pre-Deployment Check') {
    steps {
        sh '''
            export SERVICE_URL=http://localhost:8000
            export RUN_TESTS=true
            python scripts/pre_deployment_check.py
        '''
    }
}
```

---

## 📊 Manual Checklist

If you prefer a manual checklist, use this:

### Pre-Production Verification

```
□ Database Migration Applied
  □ TimescaleDB migration run
  □ All 8 tables created
  □ Hypertables configured
  □ Compression policies active
  □ Retention policies active

□ Infrastructure Ready
  □ PostgreSQL cluster healthy
  □ Redis cluster healthy
  □ Network connectivity verified
  □ Load balancer configured

□ Service Configuration
  □ Environment variables set
  □ Provider API keys configured
  □ Database connection strings correct
  □ Redis connection string correct

□ Data Populated
  □ Symbol universe loaded (≥100 symbols)
  □ Tier assignments correct (T0/T1/T2)
  □ Provider capabilities registered

□ Monitoring Stack
  □ Prometheus scraping metrics
  □ ServiceMonitors deployed
  □ PrometheusRules deployed
  □ Blackbox Exporter probing endpoints
  □ Grafana dashboards imported
  □ Alert routing configured (PagerDuty/Slack)

□ Service Health
  □ /health endpoint returning 200
  □ /metrics endpoint returning 200
  □ All 19 metrics being reported
  □ SLO monitor running (check slo_gap_violation_rate)

□ Functional Testing
  □ Provider failover tested
  □ Circuit breaker opens/closes correctly
  □ Gap detection working
  □ Backfill queue processing
  □ WebSocket clients can connect
  □ Config hot-reload working

□ Operational Readiness
  □ DLQ admin endpoints accessible
  □ Runbook URLs updated in alerts
  □ Dashboard URLs updated in alerts
  □ On-call rotation configured
  □ Escalation policies defined

□ Final Verification
  □ All unit tests passing
  □ Integration tests passing
  □ Load test completed
  □ Rollback plan documented
  □ Deployment runbook reviewed
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue: PostgreSQL connection refused**
```bash
# Check if PostgreSQL is running
systemctl status postgresql

# Check connection string
echo $DATABASE_URL

# Test connection
psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT 1"
```

**Issue: Redis connection refused**
```bash
# Check if Redis is running
systemctl status redis

# Test connection
redis-cli -h $REDIS_HOST -p $REDIS_PORT ping
```

**Issue: Service health check fails**
```bash
# Check if service is running
curl -v http://localhost:8000/health

# Check logs
tail -f logs/market-data-service.log

# Check process
ps aux | grep uvicorn
```

**Issue: Prometheus not scraping**
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq

# Check ServiceMonitor labels
kubectl get servicemonitor -n monitoring -o yaml | grep -A5 labels

# Check service port name
kubectl get svc market-data-service -o yaml | grep -A10 ports
```

**Issue: Tests fail**
```bash
# Run tests with verbose output
pytest tests/ -v -s

# Run specific test
pytest tests/test_slo_monitor.py -v

# Check test dependencies
pip list | grep -E "(pytest|pydantic)"
```

---

## 📚 Additional Resources

- [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) - Full deployment documentation
- [INTEGRATION_GUIDE.md](../INTEGRATION_GUIDE.md) - Step-by-step integration
- [MANUAL_PATCHES.md](../MANUAL_PATCHES.md) - Manual integration patches
- [INDEX.md](../INDEX.md) - Master documentation index

---

## 🎯 Quick Commands

```bash
# Full check with tests
RUN_TESTS=true python scripts/pre_deployment_check.py

# Check specific service
SERVICE_URL=http://production.example.com:8000 python scripts/pre_deployment_check.py

# Check with custom Prometheus
PROMETHEUS_URL=http://prometheus.monitoring.svc:9090 python scripts/pre_deployment_check.py

# Check remote database
POSTGRES_HOST=db.prod.example.com \
POSTGRES_USER=market_data_user \
POSTGRES_DB=market_data_prod \
python scripts/pre_deployment_check.py
```

---

**The pre-deployment check script automates 90% of the production readiness verification!** ✅
