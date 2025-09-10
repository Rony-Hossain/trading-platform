# Commands Cheat Sheet (Frontend, Backend, Docker)

This file lists the common commands to run the frontend app and supporting services locally. Adjust paths if your environment differs.

## Prerequisites

- Node.js 20+ and npm 10+
- Docker Desktop (or Docker Engine + Compose)
- Optional: Python 3.11+ if your backend services are Python-based; or Node.js if they are Node-based

## Environment Variables

Frontend expects these variables (defaults exist but it’s best to set them):

- `NEXT_PUBLIC_MARKET_DATA_API` (default `http://localhost:8002`)
- `NEXT_PUBLIC_ANALYSIS_API` (default `http://localhost:8003`)

PowerShell (Windows):

```
$env:NEXT_PUBLIC_MARKET_DATA_API = "http://localhost:8002"
$env:NEXT_PUBLIC_ANALYSIS_API = "http://localhost:8003"
```

bash/zsh (macOS/Linux):

```
export NEXT_PUBLIC_MARKET_DATA_API=http://localhost:8002
export NEXT_PUBLIC_ANALYSIS_API=http://localhost:8003
```

## Start Databases (Docker Compose)

From repo root (`e:\rony-data\trading-platform`):

```
docker compose up -d postgres redis
```

Useful checks:

```
docker compose ps
docker compose logs -f postgres
docker compose logs -f redis
```

Stop services:

```
docker compose down
```

## Run Backend Services (choose the stack that applies)

The frontend expects two services: Market Data API on 8002 and Analysis API on 8003. Use either Python/FastAPI or Node/NestJS examples below depending on your implementation.

- Python (FastAPI/Uvicorn) example:

```
# Market Data API
cd services/market-data
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
# bash/zsh
# source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8002 --reload
```

```
# Analysis API
cd services/analysis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8003 --reload
```

- Node.js (NestJS/Express) example:

```
# Market Data API
cd services/market-data
npm install
npm run start:dev   # or: npm run dev
```

```
# Analysis API
cd services/analysis
npm install
npm run start:dev   # or: npm run dev
```

Optional Dockerized APIs (if you add Dockerfiles as suggested):

```
docker compose up -d market-data-api analysis-api
```

## Run Frontend (Nx + Next.js)

From repo root or the `trading-frontend` folder:

```
cd trading-frontend
npm install
# ensure env vars are set (see Environment Variables section)
npm run dev          # runs: nx serve trading-web
```

Build and start (production):

```
cd trading-frontend
npm run build        # nx build trading-web
npm run start        # nx serve trading-web --configuration=production
```

Lint and tests:

```
cd trading-frontend
npm run lint         # nx lint trading-web
npm run test         # nx test trading-web
```

## Database Utilities

Connect to Postgres (psql):

```
psql "postgres://trading_user:trading_pass@localhost:5432/trading_db"
```

Redis CLI ping:

```
docker exec -it $(docker ps -qf name=redis) redis-cli ping
```

## Troubleshooting

- Ports busy: change host ports in `docker-compose.yml` or stop conflicting services.
- CORS/API URL: verify `NEXT_PUBLIC_*` env vars match your backend hosts/ports.
- Docker perms on Windows: run terminal as Administrator if needed, or use WSL2.
- Node version: ensure Node 20+ to match Next.js 15 requirements.

