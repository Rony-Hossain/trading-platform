# Market Data Test Service

Black-box test harness for the Market Data API exposed at `http://localhost:8002`. The suite combines functional, contract, WebSocket, and performance coverage with VizTracer profiling for critical scenarios.

## Quick Start

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
3. Copy `.env.example` to `.env` and adjust values as needed.
4. Ensure the Market Data service is running locally on port 8002 (Docker Compose or native).

## Commands

- `make test` – full pytest integration suite (`tests/market_data`, data quality checks).
- `make bdd` – run pytest-bdd scenarios (`tests/bdd`).
- `make contract` – Schemathesis fuzzing against `${BASE_URL}/openapi.json`.
- `make perf` – k6 smoke tests for price REST and WebSocket workloads (requires k6 CLI).
- `make trace` – execute only `@profile`-tagged tests with VizTracer enabled.
- `make lint` – pre-commit formatting and linting hooks.
- `make ci` – convenience target for local CI parity (`test`, `bdd`, `contract`).

## Environment Variables

Load via `.env` or CI secrets:

- `BASE_URL` (`http://localhost:8002`)
- `WS_URL` (`ws://localhost:8002`)
- `FINNHUB_API_KEY` (optional; exercises primary provider)
- `REDIS_URL` (`redis://localhost:6379/0`)
- `LIVE_TESTS=1` to force tests even if `/health` fails.
- `ENABLE_PROVIDER_MOCKS=1` to run fallback simulations with mocks.

## VizTracer & Profiling

Tests or scenarios tagged with `@profile` automatically capture VizTracer traces under `artifacts/viztraces/*.json`. Use `vizviewer` or `viztracer --open` to inspect traces post-run.

## Contract Fuzzing

`scripts/run-schemathesis.sh` wraps `schemathesis run ${BASE_URL}/openapi.json --checks all`, emitting `artifacts/schemathesis.xml` for CI visibility. Override `BASE_URL` at runtime to target alternate environments.

## Performance Smoke

`tests/perf/k6-price-smoke.js` and `tests/perf/k6-ws-smoke.js` provide k6 workloads aligned with the test plan’s latency and capacity goals. Install k6 separately (https://k6.io/docs/getting-started/installation/) before running `make perf`.

## Test Plan Alignment

Success criteria excerpt (Market Data Service — Test Plan v1.0):

- **Functional** – all REST/WS endpoints respond correctly with valid inputs; precise errors for bad requests.
- **Reliability** – provider 429/5xx/timeout gracefully fall back to secondary provider; cache TTL honored.
- **Performance** – REST p95 < 250 ms locally; WebSocket first tick < 2 s, supports 100+ subscribers.
- **Data quality** – prices > 0, volumes ≥ 0, OHLC candles monotonic and timezone-consistent.
- **Contract** – OpenAPI contract passes Schemathesis fuzzing with negative checks.

The tests, BDD scenarios, mocks, and perf scripts directly map to these acceptance criteria.

## Reporting

Pytest emits JUnit XML and HTML under `artifacts/`. Schemathesis and VizTracer outputs are also stored in `artifacts/` for CI collection. Review CI workflow (`.github/workflows/ci.yml`) for artifact upload behaviour.
