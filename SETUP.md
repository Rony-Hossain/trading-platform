# Trading Platform – Complete Setup Guide

A concise guide to set up and run the platform with frontend, backend APIs, and database infrastructure.

## Architecture Overview

- Development on Windows 11 (Docker Desktop, Node.js, Python)
- Backend: FastAPI microservices + PostgreSQL + Redis
- Frontend: Next.js (TypeScript, Nx workspace)
- Orchestration: Docker Compose (local), Docker Swarm (optional/prod)

## Prerequisites (Windows 11)

Install via winget (or manually):

```
winget install OpenJS.NodeJS
winget install Docker.DockerDesktop
winget install Git.Git
winget install Python.Python.3.11
```

Optional:

```
winget install PostgreSQL.PostgreSQL
winget install Microsoft.VisualStudioCode
```

## Quick Start (Local Development)

1) Clone and prepare

```
git clone <your-repo-url> trading-platform
cd trading-platform
cp .env.example .env
```

2) Start databases

```
docker compose up -d postgres redis
docker compose ps
```

3) Apply schema (auto-runs from migrations volume). To verify:

```
psql "postgresql://trading_user:trading_pass@localhost:5432/trading_db" -f scripts/test-db-connection.sql
```

4) Start backend services (first build may take several minutes):

```
docker compose up -d --build market-data-api analysis-api sentiment-service fundamentals-service
```

5) Run the frontend (local dev):

```
cd trading-frontend
npm install
npm run dev
```

Visit http://localhost:3000

## Helpful VS Code Tasks (optional)

Create `.vscode/tasks.json` with common tasks (start DB, services, frontend) if desired.

## Production Notes (optional)

- Review `docker-compose.prod.yml` and scripts under `scripts/` for Swarm-based deploys.
- Monitoring via `monitoring/` (Prometheus/Grafana) can be enabled using the provided compose files.

