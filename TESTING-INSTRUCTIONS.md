# Trading Platform – Testing Instructions

This guide helps you start the stack and validate core services. Use Docker Desktop on Windows.

## Quick Start

1) Start Docker Desktop and open a terminal in the repo root.

2) Start services (first build may take 5–10 minutes):

```
docker compose up -d --build
docker compose ps
```

3) Run the automated test script:

```
python test-all-services.py
```

The script checks health endpoints, basic API routes, DB connectivity (Postgres/Redis), and writes `test-results.json`.

## What Gets Tested

- Market Data API (8002): health, quote, candles, intraday
- Analysis API (8003): health, analyze, forecast
- Sentiment Service (8005): health, stats, posts, summary
- Fundamentals/Earnings (8006): health, calendar, upcoming, monitor, trends
- Databases: PostgreSQL/TimescaleDB connectivity and extensions, Redis ping

Note: Frontend runs outside compose (see README). If not running, frontend checks may show unreachable in the report.

## Manual Checks (optional)

Database checks (containers names may vary):

```
docker exec -it trading-platform-postgres-1 psql -U trading_user -d trading_db -c "SELECT version();"
docker exec -it trading-platform-postgres-1 psql -U trading_user -d trading_db -c "SELECT extname FROM pg_extension WHERE extname='timescaledb';"
docker exec -it trading-platform-redis-1 redis-cli ping
```

Service health:

```
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8005/health
curl http://localhost:8006/health
```

## Troubleshooting

- Rebuild and restart:

```
docker compose down
docker compose up -d --build
```

- View logs:

```
docker compose logs -f [service-name]
```

- Common tips:
  - Ensure `.env` is present (copy from `.env.example`).
  - First builds can take several minutes (especially analysis).
  - If Postgres is slow to initialize, re-run tests after `docker compose ps` shows healthy.

