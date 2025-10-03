# Signal Service

Beginner-friendly trading orchestration layer that transforms complex quantitative models into plain-English trading recommendations.

## Features

- **Backend-Driven:** All decision logic on server; frontend is thin rendering layer
- **Graceful Degradation:** Service works with partial upstream failures
- **Safety-First:** Server-enforced guardrails for beginners
- **Audit Trail:** Every decision stored immutably for 30 days
- **Observable:** Structured logs, Prometheus metrics, SLO tracking

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env

# Run service
python -m app.main
```

### Docker

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f signal-service
```

## API Endpoints

### Core Endpoints

- `GET /api/v1/plan` - Get today's trading plan
- `GET /api/v1/alerts` - List active alerts
- `POST /api/v1/alerts/arm` - Configure alerts
- `POST /api/v1/buy` - Execute buy action (idempotent)
- `POST /api/v1/sell` - Execute sell action (idempotent)
- `GET /api/v1/positions` - Get simplified portfolio view
- `GET /api/v1/explain/{term}` - Get term explanation

### System Endpoints

- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /docs` - OpenAPI documentation

## Architecture

```
Signal Service
├── Translators (complex → plain English)
├── Aggregators (orchestrate upstreams)
├── Guardrails (volatility brake, caps)
├── Observability (metrics, logs, SLOs)
└── Storage (Redis: cache, idempotency, decisions)
```

## Configuration

Configuration via environment variables or `.env` file:

```bash
# Service
SERVICE_NAME=signal-service
VERSION=1.0.0
DEBUG=false

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Upstream Services
INFERENCE_SERVICE_URL=http://localhost:8001
FORECAST_SERVICE_URL=http://localhost:8002
SENTIMENT_SERVICE_URL=http://localhost:8003
```

See `app/config.py` for all configuration options.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/contract_tests/test_plan_contract.py
```

## Load Testing

```bash
# Run load test
locust -f scripts/load_test.py --host=http://localhost:8000
```

## Monitoring

- **Metrics:** http://localhost:8000/metrics
- **Logs:** Structured JSON logs to stdout
- **Health:** http://localhost:8000/health

## Operations

### Hot-Reload Policies

```bash
# Edit policies
vim config/policies.yaml

# Reload without restart
kill -HUP $(cat /var/run/signal-service.pid)
```

### Check SLOs

```bash
python scripts/check_slos.py
```

## Documentation

- [Implementation Plan](../../documentation/signal-service-implementation-plan.md)
- [API Documentation](http://localhost:8000/docs)
- [Runbook](./RUNBOOK.md)

## License

Proprietary
