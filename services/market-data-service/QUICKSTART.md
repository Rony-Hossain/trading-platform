# Quick Start Guide

Get the Market Data Service running in 5 minutes.

## Step 1: Run the Database Migration

```bash
# Navigate to the service directory
cd e:\rony-data\trading-platform\services\market-data-service

# Run the migration (update credentials as needed)
psql -U trading_user -d trading_db -f db/migrations/20251008_market_data_core.sql
```

Expected output:
```
CREATE TYPE
CREATE TYPE
CREATE TABLE
CREATE INDEX
...
CREATE TRIGGER
```

## Step 2: Set Environment Variables

Create a `.env` file:

```bash
# Quick start configuration
DATABASE_URL=postgresql://trading_user:trading_pass@localhost:5432/trading_db
FINNHUB_API_KEY=your_key_here
USE_RLC=false
LOCAL_SWEEP_ENABLED=true
```

## Step 3: Start the Service

```bash
# Install dependencies if not already done
pip install -r requirements.txt

# Start the service
uvicorn app.main:app --reload --port 8001
```

Expected output:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Database service initialized
INFO:     Finnhub provider registered
INFO:     Yahoo Finance provider registered
INFO:     Data collector started (USE_RLC=False, LOCAL_SWEEP=True)
INFO:     Background tasks started (including data collector)
INFO:     Application startup complete.
```

## Step 4: Verify It's Working

### Check Health
```bash
curl http://localhost:8001/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "market-data-service",
  "providers_status": {
    "finnhub": true,
    "yfinance": true
  }
}
```

### Check Provider Health Scores
```bash
curl http://localhost:8001/stats/providers | jq '.providers[] | {provider: .provider, health: .health_score, state: .state}'
```

Expected:
```json
{
  "provider": "finnhub",
  "health": 1.0,
  "state": "CLOSED"
}
{
  "provider": "yfinance",
  "health": 1.0,
  "state": "CLOSED"
}
```

### Fetch Stock Data
```bash
curl http://localhost:8001/stocks/AAPL/price
```

### Check Prometheus Metrics
```bash
curl http://localhost:8001/metrics | head -20
```

## Step 5: Optional - Populate Symbol Universe

The service will work without this, but for tier-based sweeping:

```sql
-- Example: Insert some symbols with tiers
INSERT INTO symbol_universe (symbol, exchange, asset_type, adv_21d, mkt_cap, tier, active) VALUES
  ('AAPL', 'NASDAQ', 'equity', 50000000, 3000000000000, 'T0', true),
  ('GOOGL', 'NASDAQ', 'equity', 25000000, 1800000000000, 'T0', true),
  ('TSLA', 'NASDAQ', 'equity', 100000000, 800000000000, 'T0', true),
  ('SPY', 'NYSE', 'etf', 80000000, 500000000000, 'T0', true),
  ('NVDA', 'NASDAQ', 'equity', 40000000, 2000000000000, 'T0', true)
ON CONFLICT (symbol) DO NOTHING;
```

Now the local sweeper will automatically fetch data for these symbols.

## Next Steps

- Read [IMPLEMENTATION.md](IMPLEMENTATION.md) for full details
- Set up Grafana dashboards using the Prometheus metrics
- Configure Redis for backfill queue support
- Add more symbols to the universe table

## Troubleshooting

**"No providers available"**
- Check your FINNHUB_API_KEY is valid
- Ensure yfinance is installed: `pip install yfinance`

**Database connection errors**
- Verify DATABASE_URL is correct
- Check PostgreSQL is running: `pg_isready`

**Import errors**
- Install all dependencies: `pip install -r requirements.txt`
- You may need: `pip install redis` for RLC mode

## Common Commands

```bash
# View logs
tail -f logs/market-data.log

# Hot-reload configuration
curl -X POST http://localhost:8001/ops/reload

# Trigger manual backfill
curl -X POST http://localhost:8001/ops/backfill \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "interval": "1m", "start": "2025-10-08T09:30:00Z", "end": "2025-10-08T16:00:00Z", "priority": "T0"}'

# Check backfill status
curl http://localhost:8001/metrics | grep backfill

# View ingestion cursor
curl http://localhost:8001/ops/cursor/AAPL/1m/yfinance
```

## Development Tips

1. Use `--reload` flag during development for auto-restart on code changes
2. Check `/stats/providers` regularly to monitor health scores
3. Monitor `/metrics` to see what's happening under the hood
4. Use `/ops/validate` to test config changes before reloading

Happy trading! 🚀
