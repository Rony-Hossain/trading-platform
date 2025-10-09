# Market Data Service - Production Deployment Guide

## Overview

This guide covers deploying the Market Data Service to production with full monitoring, alerting, and TimescaleDB optimization.

## Prerequisites

- Kubernetes cluster (1.19+)
- Helm 3.x
- PostgreSQL 14+ with TimescaleDB 2.x
- Redis 6+ cluster
- Prometheus Operator installed
- Grafana installed

## Deployment Steps

### 1. Database Setup

#### Option A: Standard PostgreSQL

```bash
# Run the standard migration
psql -U trading_user -d trading_db -f db/migrations/20251008_market_data_core.sql
```

#### Option B: TimescaleDB (Recommended for Production)

```bash
# Ensure TimescaleDB is installed
psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# Run the TimescaleDB-optimized migration
psql -U trading_user -d trading_db -f db/migrations/20251008_timescale_market_data.sql
```

**Benefits of TimescaleDB:**
- Automatic partitioning by time
- Compression (14-day policy saves ~90% storage)
- Continuous aggregates (auto-generated 5m/1h bars)
- Retention policies (365 days for 1m bars)
- 10-100x faster time-series queries

#### Verify Database Setup

```sql
-- Check hypertables
SELECT * FROM timescaledb_information.hypertables;

-- Check compression
SELECT * FROM timescaledb_information.compression_settings;

-- Check continuous aggregates
SELECT * FROM timescaledb_information.continuous_aggregates;

-- Verify tables
\dt
```

### 2. Deploy Blackbox Exporter

```bash
# Add Prometheus community Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Blackbox Exporter
helm install blackbox-exporter prometheus-community/prometheus-blackbox-exporter \
  -f deploy/blackbox-exporter/values.yaml \
  -n monitoring \
  --create-namespace

# Verify deployment
kubectl get pods -n monitoring -l app.kubernetes.io/name=prometheus-blackbox-exporter
```

### 3. Configure Prometheus

#### For Prometheus Operator

```bash
# Apply ServiceMonitors
kubectl apply -f deploy/monitoring/servicemonitor-market-data.yaml
kubectl apply -f deploy/monitoring/servicemonitor-blackbox.yaml

# Apply alert rules
kubectl apply -f deploy/monitoring/alerts-slo.yaml

# Verify ServiceMonitors
kubectl get servicemonitors -n monitoring
```

#### For Standard Prometheus

```bash
# Merge the scrape configs into your prometheus.yml
cat deploy/monitoring/prometheus-config.yaml >> /path/to/prometheus.yml

# Reload Prometheus
kubectl rollout restart deployment prometheus -n monitoring
```

### 4. Deploy Market Data Service

#### Create ConfigMap

```bash
kubectl create configmap market-data-config -n trading \
  --from-literal=DATABASE_URL='postgresql://trading_user:trading_pass@postgres:5432/trading_db' \
  --from-literal=RLC_REDIS_URL='redis://redis-cluster:6379/0' \
  --from-literal=USE_RLC='false' \
  --from-literal=LOCAL_SWEEP_ENABLED='true'
```

#### Create Secrets

```bash
kubectl create secret generic market-data-secrets -n trading \
  --from-literal=FINNHUB_API_KEY='your_finnhub_key' \
  --from-literal=ALPHA_VANTAGE_API_KEY='your_alphavantage_key'
```

#### Deploy the Application

```bash
# Option 1: Using Docker Compose (Development)
docker-compose up -d

# Option 2: Using Kubernetes (Production)
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
```

### 5. Populate Symbol Universe

```sql
-- Example: Load top US stocks
INSERT INTO symbol_universe (symbol, exchange, asset_type, adv_21d, mkt_cap, tier, active) VALUES
  ('AAPL', 'NASDAQ', 'equity', 50000000, 3000000000000, 'T0', true),
  ('GOOGL', 'NASDAQ', 'equity', 25000000, 1800000000000, 'T0', true),
  ('MSFT', 'NASDAQ', 'equity', 30000000, 2800000000000, 'T0', true),
  ('AMZN', 'NASDAQ', 'equity', 40000000, 1500000000000, 'T0', true),
  ('TSLA', 'NASDAQ', 'equity', 100000000, 800000000000, 'T0', true),
  ('SPY', 'NYSE', 'etf', 80000000, 500000000000, 'T0', true),
  ('QQQ', 'NASDAQ', 'etf', 50000000, 200000000000, 'T0', true)
ON CONFLICT (symbol) DO UPDATE SET
  adv_21d = EXCLUDED.adv_21d,
  mkt_cap = EXCLUDED.mkt_cap,
  tier = EXCLUDED.tier;

-- Record universe version
INSERT INTO universe_versions (version_tag, source_meta)
VALUES ('2025w41-manual-top100', '{"source": "manual", "count": 100}'::jsonb);
```

### 6. Verify Deployment

#### Check Service Health

```bash
# Port-forward for local access
kubectl port-forward -n trading svc/market-data-service 8001:8001

# Health check
curl http://localhost:8001/health

# Provider status
curl http://localhost:8001/stats/providers | jq '.providers[] | {provider: .provider, health: .health_score}'

# Cadence configuration
curl http://localhost:8001/stats/cadence

# Metrics
curl http://localhost:8001/metrics | head -50
```

#### Check Prometheus Targets

```bash
# Access Prometheus UI
kubectl port-forward -n monitoring svc/prometheus 9090:9090

# Open browser to http://localhost:9090/targets
# Verify these targets are UP:
# - market-data-service
# - blackbox-http-market-data-health
# - blackbox-http-market-data-metrics
# - blackbox-ws-market-data
```

#### Test Data Ingestion

```bash
# Fetch stock price
curl http://localhost:8001/stocks/AAPL/price

# Check database
psql -U trading_user -d trading_db -c "SELECT COUNT(*) FROM candles_intraday;"

# Check ingestion lag
curl http://localhost:8001/metrics | grep ingestion_lag_seconds
```

### 7. Configure Grafana Dashboards

#### Import Pre-built Dashboards

1. **Provider Health Dashboard**
   - Circuit breaker states
   - Error rates by provider
   - Latency p95/p99
   - Health score history

2. **Ingestion Quality Dashboard**
   - Gap violation rate
   - Ingestion lag
   - Jobs processed rate
   - Backfill queue depth

3. **Performance Dashboard**
   - DB write latency
   - Throughput (bars/sec)
   - Memory usage
   - CPU usage

4. **WebSocket Dashboard**
   - Active clients
   - Publish latency
   - Dropped messages
   - Replay buffer usage

#### Key Metrics Queries

```promql
# Provider health score
provider_selected_total

# Gap rate (should be < 0.5%)
rate(gaps_found_total{interval="1m"}[5m])

# Ingestion lag
ingestion_lag_seconds{interval="1m"}

# Backfill queue depth
backfill_queue_depth

# WS publish latency p99
histogram_quantile(0.99, sum by (le) (rate(ws_publish_latency_ms_bucket[5m])))

# DB write latency p95
histogram_quantile(0.95, sum by (le) (rate(write_batch_ms_bucket[5m])))
```

### 8. Test Scenarios

#### Test 1: Provider Failover

```bash
# Set invalid Finnhub key
kubectl set env deployment/market-data-service -n trading FINNHUB_API_KEY=invalid

# Reload config
curl -X POST http://localhost:8001/ops/reload

# Check provider status
curl http://localhost:8001/stats/providers

# Expected: Finnhub health_score drops, traffic routes to Yahoo/AlphaVantage

# Restore
kubectl set env deployment/market-data-service -n trading FINNHUB_API_KEY=<valid_key>
curl -X POST http://localhost:8001/ops/reload
```

#### Test 2: Gap Detection & Backfill

```bash
# Delete a time window
psql -U trading_user -d trading_db -c "
DELETE FROM candles_intraday
WHERE symbol = 'AAPL'
  AND interval = '1m'
  AND ts BETWEEN '2025-10-08 14:00:00' AND '2025-10-08 14:10:00';
"

# Trigger manual backfill
curl -X POST http://localhost:8001/ops/backfill \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "interval": "1m",
    "start": "2025-10-08T14:00:00Z",
    "end": "2025-10-08T14:10:00Z",
    "priority": "T0"
  }'

# Watch backfill queue
watch -n 1 "curl -s http://localhost:8001/metrics | grep backfill_queue_depth"

# Check gaps metric
curl http://localhost:8001/metrics | grep gaps_found_total

# Verify data restored
psql -U trading_user -d trading_db -c "
SELECT COUNT(*) FROM candles_intraday
WHERE symbol = 'AAPL'
  AND interval = '1m'
  AND ts BETWEEN '2025-10-08 14:00:00' AND '2025-10-08 14:10:00';
"
```

#### Test 3: WebSocket Streaming

```bash
# Install wscat
npm install -g wscat

# Connect to WebSocket
wscat -c ws://localhost:8001/ws/AAPL

# Expected: Receive initial snapshot + real-time updates

# Disconnect and reconnect
# Expected: Receive replay buffer (last 120 bars)
```

#### Test 4: SLO Monitoring

```bash
# Check SLO violation rate
curl http://localhost:8001/metrics | grep slo_gap_violation_rate

# Expected for healthy system:
# slo_gap_violation_rate{tier="T0",interval="1m"} < 0.005
```

### 9. Production Checklist

- [ ] Database migration applied (TimescaleDB preferred)
- [ ] Redis cluster configured and accessible
- [ ] Blackbox Exporter deployed
- [ ] ServiceMonitors created (Prometheus Operator)
- [ ] Alert rules applied
- [ ] Grafana dashboards imported
- [ ] Symbol universe populated (minimum 100 symbols)
- [ ] Provider API keys configured
- [ ] Service health check passing
- [ ] All Prometheus targets UP
- [ ] WebSocket connectivity tested
- [ ] Provider failover tested
- [ ] Gap detection tested
- [ ] Backfill queue draining
- [ ] SLO metrics reporting
- [ ] Alerts firing correctly (test with invalid config)
- [ ] Runbook URLs updated in alerts
- [ ] Dashboard URLs updated in alerts

### 10. Monitoring & Alerting

#### Alert Severity Levels

- **Critical**: Service down, health endpoint failing
- **Warning**: SLO violations, provider errors, high latency

#### Alert Response Times

- **Critical**: Page on-call immediately
- **Warning**: Review within 30 minutes during business hours

#### Key SLOs

| SLO | Target | Metric | Alert |
|-----|--------|--------|-------|
| Continuity | T0 gap ≤ 2× interval | `slo_gap_violation_rate{T0}` | < 0.5% |
| Latency | WS p99 < 250ms | `ws_publish_latency_ms_bucket` | p99 < 250ms |
| Backfill | Drain < 10× outage | `backfill_oldest_age_seconds` | T0 < 15m |
| Availability | 99.9% uptime | `up{job="market-data-service"}` | 2min downtime |

### 11. Troubleshooting

#### No data ingesting

```bash
# Check collector status
kubectl logs -n trading -l app=market-data-service --tail=100 | grep "Data collector"

# Check job queue
# If USE_RLC=false, local sweeper should emit jobs every 30s
```

#### High backfill queue

```bash
# Check queue depth per tier
curl http://localhost:8001/metrics | grep backfill_queue_depth

# List failed backfills
curl http://localhost:8001/ops/dlq/backfills

# Requeue failed job
curl -X POST http://localhost:8001/ops/dlq/backfills/requeue/12345
```

#### Provider circuit breaker stuck OPEN

```bash
# Check provider health
curl http://localhost:8001/stats/providers | jq '.providers[] | select(.state=="OPEN")'

# Reload config (may help if transient)
curl -X POST http://localhost:8001/ops/reload

# If API key issue, update secret and reload
kubectl edit secret market-data-secrets -n trading
curl -X POST http://localhost:8001/ops/reload
```

#### Database slow writes

```bash
# Check write latency
curl http://localhost:8001/metrics | grep write_batch_ms_bucket

# If using TimescaleDB, check compression settings
psql -c "SELECT * FROM timescaledb_information.chunks ORDER BY total_bytes DESC LIMIT 10;"

# Consider increasing batch size
kubectl set env deployment/market-data-service -n trading LIVE_BATCH_SIZE=1000
```

### 12. Scaling Considerations

#### Horizontal Scaling

- Service is stateless and can scale horizontally
- Multiple pods share Redis job queues
- Database connection pooling handles multiple writers

```bash
# Scale to 3 replicas
kubectl scale deployment market-data-service -n trading --replicas=3
```

#### Vertical Scaling

- Increase batch sizes for higher throughput
- Increase database pool size for more concurrent operations
- Add more worker concurrency for backfills

```bash
# Example: High-throughput configuration
kubectl set env deployment/market-data-service -n trading \
  LIVE_BATCH_SIZE=2000 \
  BACKFILL_BATCH_SIZE=10000 \
  BACKFILL_MAX_CONCURRENCY_T0=8
```

### 13. Backup & Recovery

#### Database Backups

```bash
# Daily backup (automated via CronJob recommended)
pg_dump -U trading_user trading_db | gzip > backup_$(date +%Y%m%d).sql.gz

# Restore
gunzip < backup_20251008.sql.gz | psql -U trading_user trading_db
```

#### Configuration Backups

```bash
# Export current config
kubectl get configmap market-data-config -n trading -o yaml > config-backup.yaml
kubectl get secret market-data-secrets -n trading -o yaml > secrets-backup.yaml
```

### 14. Maintenance Windows

```bash
# Graceful shutdown
kubectl scale deployment market-data-service -n trading --replicas=0

# Perform maintenance
# ...

# Restart
kubectl scale deployment market-data-service -n trading --replicas=3
```

## Support & Documentation

- **Architecture**: See [IMPLEMENTATION.md](IMPLEMENTATION.md)
- **Quick Start**: See [QUICKSTART.md](QUICKSTART.md)
- **Enhancements**: See [ENHANCEMENTS.md](ENHANCEMENTS.md)
- **Summary**: See [COMPLETE_SUMMARY.md](COMPLETE_SUMMARY.md)

## Contact

For issues or questions:
- File issue in repository
- Contact platform team
- Check Grafana dashboards for real-time status

---

**Production Status**: ✅ Ready
**Last Updated**: 2025-10-08
