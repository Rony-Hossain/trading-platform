# Trading Platform

A microservices trading platform with FastAPI services, PostgreSQL (TimescaleDB), Redis, and a Next.js frontend.

## What’s Implemented

- Database: PostgreSQL 16 + TimescaleDB with migrations and seed data
- Services: Market Data API, Analysis API, Sentiment Service, Fundamentals/Earnings Service
- Docker Compose: Postgres, Redis, and core services with healthchecks
- Frontend: Next.js app under `trading-frontend` (run locally for dev)

## Repo Structure (high level)

- `migrations/` – SQL schema and seed data
- `services/` – Python FastAPI microservices
- `trading-frontend/` – Next.js app (Nx workspace)
- `docker-compose.yml` – Local stack for DB + core APIs
- `TESTING-INSTRUCTIONS.md` – How to start and test services
- `SETUP.md` – End-to-end setup guide

## Quick Start (Development)

1) Create environment file

```
cp .env.example .env
```

2) Start databases (Docker Desktop required)

```
docker compose up -d postgres redis
docker compose ps
```

3) Bring up core services

```
docker compose up -d --build market-data-api analysis-api sentiment-service fundamentals-service
```

4) Verify endpoints

- Market Data: http://localhost:8002/health
- Analysis: http://localhost:8003/health
- Sentiment: http://localhost:8005/health
- Fundamentals: http://localhost:8006/health

5) Frontend (run locally)

```
cd trading-frontend
npm install
npm run dev
```

Visit http://localhost:3000

## Notes

- First image builds (especially Analysis) may take time.
- See `TESTING-INSTRUCTIONS.md` for the full automated test script and manual checklist.
- For production deployment, review `docker-compose.prod.yml` and scripts in `scripts/`.


