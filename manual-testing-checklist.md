# Manual Testing Checklist

Run this after `docker compose up -d --build`.

## Core Services Health

- Market Data (8002):
  - http://localhost:8002/health
  - http://localhost:8002/
  - http://localhost:8002/stocks/AAPL/quote
  - http://localhost:8002/stocks/AAPL/candles?period=1d

- Analysis (8003):
  - http://localhost:8003/health
  - http://localhost:8003/
  - http://localhost:8003/analyze/AAPL

- Sentiment (8005):
  - http://localhost:8005/health
  - http://localhost:8005/
  - http://localhost:8005/stats
  - http://localhost:8005/posts/AAPL

- Fundamentals/Earnings (8006):
  - http://localhost:8006/health
  - http://localhost:8006/
  - http://localhost:8006/earnings/upcoming
  - http://localhost:8006/earnings/AAPL/monitor

## Database Connectivity

PostgreSQL/TimescaleDB (5432):

```
docker exec -it trading-platform-postgres-1 psql -U trading_user -d trading_db -c "SELECT version();"
docker exec -it trading-platform-postgres-1 psql -U trading_user -d trading_db -c "SELECT table_name FROM timescaledb_information.hypertables;"
```

Redis (6379):

```
docker exec -it trading-platform-redis-1 redis-cli ping
```

## Troubleshooting

```
docker compose logs market-data-api
docker compose logs sentiment-service
docker compose logs fundamentals-service
docker compose logs analysis-api
docker compose logs postgres
docker compose logs redis
```

