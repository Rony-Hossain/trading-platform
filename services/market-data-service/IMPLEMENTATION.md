# Market Data Service - Master Algorithm v1.0 Implementation

## Overview

This implementation follows the "Master Algorithm v1.0" plan for a provider-independent, gap-safe, low-latency market data service with health-aware routing, gap detection, and automatic backfill.

## Architecture

### Key Components

1. **Provider Registry** ([app/providers/registry.py](app/providers/registry.py))
   - Health-score-based provider ranking
   - Circuit breaker integration
   - Deterministic routing with policy hints
   - Prometheus metrics for latency, errors, and circuit state

2. **Data Collector** ([app/services/data_collector.py](app/services/data_collector.py))
   - RLC job consumer (for production rate-limit coordination)
   - Local sweeper (development fallback mode)
   - Gap detection and backfill orchestration
   - Rate-limited, tier-aware backfill workers

3. **Database Service** ([app/services/database.py](app/services/database.py))
   - PIT-safe candle storage with `as_of`, `provider`, `status`
   - Universe and tiering management
   - Ingestion cursors for continuity tracking
   - Backfill job queue

4. **Configuration** ([app/core/config.py](app/core/config.py))
   - Hot-reloadable settings
   - Provider policies (routing preferences)
   - Tier cadences and backfill limits
   - Feature flags (USE_RLC, ALLOW_SOURCE_OVERRIDE, etc.)

5. **Observability** ([app/observability/metrics.py](app/observability/metrics.py))
   - Prometheus metrics for all components
   - SLO-aligned dashboards (gap rate, backfill age, latency)

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ (TimescaleDB optional but recommended)
- Redis 6+ (required for RLC mode and backfills)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install redis for RLC/backfill support
pip install redis
```

### Database Migration

Run the SQL migration to create all required tables:

```bash
# Using psql
psql -U trading_user -d trading_db -f db/migrations/20251008_market_data_core.sql

# Or using Docker
docker exec -i postgres-container psql -U trading_user -d trading_db < db/migrations/20251008_market_data_core.sql
```

#### What the migration creates:

- `candles_intraday` - PIT-safe OHLCV bars with provider tracking
- `quotes_l1` - Level-1 quote data
- `symbol_universe` - Symbol tiering and metadata
- `universe_versions` - Universe snapshots for auditing
- `ingestion_cursor` - Per-symbol/interval/provider cursors for gap detection
- `backfill_jobs` - Backfill work queue
- `macro_factors` - Macro data storage (if not exists)
- `options_metrics` - Options metrics storage (if not exists)

#### Optional: TimescaleDB Enhancements

If you have TimescaleDB installed, uncomment the TimescaleDB sections in the migration to enable:
- Hypertable partitioning on `ts`
- Compression policies (compress data older than 14 days)
- Retention policies (keep 365 days of 1m bars)
- Continuous aggregates (auto-generated 5m/15m bars)

### Configuration

Create a `.env` file or set environment variables:

```bash
# Database
DATABASE_URL=postgresql://trading_user:trading_pass@localhost:5432/trading_db

# Providers
FINNHUB_API_KEY=your_finnhub_key_here
ALPHA_VANTAGE_API_KEY=your_alphavantage_key_here

# Feature Flags
USE_RLC=false                    # Set to true when RLC service is available
LOCAL_SWEEP_ENABLED=true          # Local job generator (dev mode)
ALLOW_SOURCE_OVERRIDE=true        # Honor provider hints from jobs

# Routing Policies (comma-separated, priority order)
POLICY_BARS_1M=finnhub,alphavantage,yfinance
POLICY_EOD=yfinance,alphavantage,finnhub
POLICY_QUOTES_L1=finnhub,yfinance
POLICY_OPTIONS_CHAIN=yfinance,finnhub

# Tier Configuration
TIER_MAXS__T0=1200                # Max symbols for T0 (most liquid)
TIER_MAXS__T1=3000                # Max symbols for T1
# Rest fall into T2

# Cadence (seconds for quotes, minutes for bars)
CADENCE_T0__bars_1m=60            # T0: fetch 1m bars every 60s
CADENCE_T0__quotes_l1=5           # T0: fetch quotes every 5s
CADENCE_T1__bars_1m=300           # T1: fetch 1m bars every 5min
CADENCE_T1__quotes_l1=15          # T1: fetch quotes every 15s
CADENCE_T2__eod=1                 # T2: EOD only

# Backfill Controls
BACKFILL_CHUNK_MINUTES=1440       # 1 day chunks for 1m bars
BACKFILL_MAX_CONCURRENCY_T0=4
BACKFILL_MAX_CONCURRENCY_T1=2
BACKFILL_MAX_CONCURRENCY_T2=1
BACKFILL_DISPATCH_RATE_PER_SEC=2.0

# Redis (for RLC jobs and backfills)
RLC_REDIS_URL=redis://localhost:6379/0
RLC_REDIS_JOBS_KEY=market:jobs
RLC_REDIS_BACKFILL_KEYS__T0=market:backfills:T0
RLC_REDIS_BACKFILL_KEYS__T1=market:backfills:T1
RLC_REDIS_BACKFILL_KEYS__T2=market:backfills:T2

# Storage
LIVE_BATCH_SIZE=500               # Batch size for live writes
BACKFILL_BATCH_SIZE=5000          # Batch size for backfills
```

## Running the Service

### Development Mode (Local Sweep)

```bash
# Start the service
python -m app.main

# Or with uvicorn
uvicorn app.main:app --reload --port 8001
```

In this mode:
- `USE_RLC=false` and `LOCAL_SWEEP_ENABLED=true`
- The collector automatically sweeps symbols by tier and generates jobs
- No external RLC service needed

### Production Mode (RLC Integration)

```bash
# Ensure Redis is running
docker-compose up -d redis

# Set RLC mode
export USE_RLC=true
export LOCAL_SWEEP_ENABLED=false

# Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 4
```

In this mode:
- Service consumes jobs from `market:jobs` Redis list
- RLC service (external) handles rate limiting and provider budgets
- Backfills still managed locally with rate limiting

## API Endpoints

### Core Endpoints

- `GET /` - Service info and endpoint list
- `GET /health` - Health check with provider status
- `GET /metrics` - Prometheus metrics

### Provider & Policy

- `GET /stats/providers` - Provider health scores, latency, circuit state
- `GET /stats/cadence` - Tier configuration and cadence policy
- `POST /ops/reload` - Hot-reload configuration
- `POST /ops/validate` - Validate configuration without applying

### Cursor & Backfill Management

- `GET /ops/cursor/{symbol}/{interval}/{source}` - Read ingestion cursor
- `POST /ops/backfill` - Manually trigger backfill

Example backfill request:
```json
{
  "symbol": "AAPL",
  "interval": "1m",
  "start": "2025-10-01T09:30:00Z",
  "end": "2025-10-01T16:00:00Z",
  "priority": "T0"
}
```

### Market Data

- `GET /stocks/{symbol}/price` - Current price
- `GET /stocks/{symbol}/history?period=1mo` - Historical data
- `GET /stocks/{symbol}/intraday?interval=1m` - Intraday bars
- `POST /stocks/batch` - Batch quotes
- `WS /ws/{symbol}` - Real-time WebSocket

### Options

- `GET /options/{symbol}/chain` - Full options chain
- `GET /options/{symbol}/metrics` - ATM IV, skew, implied move
- `GET /options/{symbol}/strategies` - Advanced strategies
- `WS /ws/options/{symbol}` - Real-time options updates

### Macro

- `GET /factors/macro` - Macro factor snapshot
- `GET /factors/macro/{factor_key}/history` - Historical macro data
- `POST /admin/macro/refresh` - Force refresh

## Monitoring & Observability

### Prometheus Metrics

Available at `GET /metrics`:

**Provider Health:**
- `provider_selected_total{capability, provider}` - Provider selections
- `provider_errors_total{provider, endpoint, code}` - Error counts
- `provider_latency_ms_bucket{provider, endpoint}` - Latency histogram
- `circuit_state{provider}` - Circuit breaker state (0=open, 0.5=half_open, 1=closed)

**Ingestion & Backfill:**
- `jobs_processed_total{type, provider}` - Jobs completed
- `gaps_found_total{interval}` - Gaps detected
- `backfill_jobs_enqueued_total{tier}` - Backfills created
- `backfill_jobs_completed_total{tier, status}` - Backfills done/failed
- `backfill_queue_depth` - Current backfill queue size
- `backfill_oldest_age_seconds` - Oldest queued backfill age
- `write_batch_ms_bucket` - DB write latency
- `ingestion_lag_seconds{interval}` - Lag behind current time

**WebSocket:**
- `ws_clients_gauge` - Active WebSocket clients
- `ws_publish_latency_ms_bucket` - WS publish latency
- `ws_dropped_messages_total` - Messages dropped due to backpressure

### Grafana Dashboard Queries

**Provider Health Panel:**
```promql
# Circuit state
circuit_state{provider}

# Error rate (excluding rate limits)
rate(provider_errors_total{code!~"429"}[5m])

# P95 latency
histogram_quantile(0.95, sum by (le, provider) (rate(provider_latency_ms_bucket[5m])))
```

**Ingestion Lag:**
```promql
ingestion_lag_seconds{interval="1m"}
```

**Gap Rate:**
```promql
rate(gaps_found_total{interval="1m"}[5m])
```

**Backfill Health:**
```promql
backfill_queue_depth
backfill_oldest_age_seconds
rate(backfill_jobs_completed_total{status="failed"}[5m])
```

## Testing

### Acceptance Tests

#### 1. Provider Failover

```bash
# Check initial provider ranking
curl http://localhost:8001/stats/providers | jq '.providers[] | {provider: .provider, health: .health_score}'

# Kill a provider (e.g., set invalid API key)
export FINNHUB_API_KEY=invalid
curl -X POST http://localhost:8001/ops/reload

# Fetch data - should automatically use next provider
curl http://localhost:8001/stocks/AAPL/price

# Verify traffic moved
curl http://localhost:8001/stats/providers | jq '.providers[] | {provider: .provider, state: .state}'
```

#### 2. Gap Detection & Backfill

```bash
# Trigger a manual backfill
curl -X POST http://localhost:8001/ops/backfill \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "interval": "1m",
    "start": "2025-10-08T09:30:00Z",
    "end": "2025-10-08T16:00:00Z",
    "priority": "T0"
  }'

# Check backfill queue depth
curl http://localhost:8001/metrics | grep backfill_queue_depth

# Wait for backfill completion
watch -n 1 "curl -s http://localhost:8001/metrics | grep backfill_queue_depth"

# Verify cursor advanced
curl "http://localhost:8001/ops/cursor/AAPL/1m/yfinance"
```

#### 3. WebSocket Streaming

```bash
# Install wscat if needed: npm install -g wscat

# Connect and receive updates
wscat -c ws://localhost:8001/ws/AAPL

# For options:
wscat -c ws://localhost:8001/ws/options/AAPL
```

#### 4. Load Test

```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:8001/stocks/AAPL/price

# Check metrics
curl http://localhost:8001/metrics | grep provider_latency
```

### Unit Tests

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=app --cov-report=html tests/
```

## SLO Verification

### Continuity SLO
**Target:** T0 p99 gap ≤ 2× interval during RTH

```promql
# Query gaps in last hour
histogram_quantile(0.99,
  sum by (le) (rate(gaps_found_total{interval="1m"}[1h]))
) <= 120  # 2 minutes for 1m interval
```

### Latency SLO
**Target:** T0 WS publish p99 <250ms

```promql
histogram_quantile(0.99,
  sum by (le) (rate(ws_publish_latency_ms_bucket[5m]))
) < 250
```

### Coverage SLO
**Target:** ≥8k active US symbols

```sql
SELECT COUNT(*) FROM symbol_universe WHERE active = true;
```

### Backfill Drain SLO
**Target:** Backlog drains in <10× outage

Monitor `backfill_oldest_age_seconds` - should not exceed 10× the gap duration.

## Troubleshooting

### No providers available
- Check `GET /stats/providers` - ensure at least one provider has `health_score > 0.55`
- Verify API keys are set correctly
- Check circuit breaker state

### Gaps not filling
- Verify backfill worker is running: `curl /metrics | grep jobs_processed`
- Check backfill queue: `curl /metrics | grep backfill_queue_depth`
- Look for provider errors: `curl /stats/providers | jq '.providers[] | .error_ewma'`

### High ingestion lag
- Check write batch latency: `curl /metrics | grep write_batch`
- Verify database connection pool size
- Consider increasing `LIVE_BATCH_SIZE`

### Redis connection errors
- Ensure Redis is running: `redis-cli ping`
- Verify `RLC_REDIS_URL` is correct
- Check Redis logs for connection issues

## Rollout Checklist

- [x] M1: Provider policy + health-aware routing ✅
- [x] M2: Universe/tiering + RLC jobs (with local fallback) ✅
- [x] M3: Cursors, gap engine, backfill queues, PIT-safe storage ✅
- [x] M4: Metrics, dashboards, observability ✅
- [ ] M5: WS snapshot-then-stream, backpressure, replay (partially implemented)

## Next Steps

1. **Populate Universe**: Load symbols into `symbol_universe` table with tiers
2. **Configure Grafana**: Import dashboards using the Prometheus queries above
3. **RLC Integration**: Connect to external Rate-Limit Coordinator if available
4. **Corporate Actions**: Add split/dividend handling (Phase B.1 enhancement)
5. **Market Calendar**: Integrate exchange calendars for exact RTH boundaries

## Architecture Decisions

### Why Health Score = 0.45×breaker + 0.25×latency + 0.20×error + 0.10×completeness?

- **Circuit breaker (45%)**: Dominant factor - an OPEN breaker should kill routing immediately
- **Latency (25%)**: Users notice slow providers; penalize high p95
- **Error rate (20%)**: Errors are costly but may be transient
- **Completeness (10%)**: Missing data is bad but detectable via gaps

### Why separate live vs backfill queues?

- **Isolation**: Backfill storms shouldn't starve live ingestion
- **Rate limits**: Different concurrency/rate limits per tier
- **Observability**: Separate metrics for live vs backfill success

### Why Redis for jobs instead of DB?

- **Performance**: BRPOP is faster than polling SELECT FOR UPDATE
- **Horizontal scaling**: Multiple workers can consume from same queue
- **RLC integration**: External RLC service can push directly to Redis

## License

Proprietary - Internal Trading Platform

## Support

For issues or questions, contact the platform team or file an issue in the repository.
