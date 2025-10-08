# Market Data Test Service — TODO Plan

## 1. Repository Bootstrap
- [ ] Scaffold `market-data-test-service/` repo with baseline files (README, pyproject, requirements, pytest.ini, Makefile, `.gitignore`, `.env.example`).
- [ ] Add CI workflow under `.github/workflows/ci.yml` covering pytest, BDD, Schemathesis, artifact upload.
- [ ] Configure pre-commit hooks (Black, isort, Ruff, whitespace fixers).

## 2. Core Test Infrastructure
- [ ] Implement `tests/conftest.py` with:
  - [ ] Environment loading via `.env`.
  - [ ] Async HTTP client fixture targeting `${BASE_URL}`.
  - [ ] Service availability probe with `LIVE_TESTS` override.
  - [ ] VizTracer hook for `@profile` markers writing to `artifacts/viztraces/`.
  - [ ] Respx mock fixture and freeze-time helper.
- [ ] Populate data fixtures (`tracked_symbols`, expectations JSON).
- [ ] Ensure shared artifacts directory creation per test session.

## 3. REST Integration Tests (`tests/market_data/`)
- [ ] `test_health.py`: `/health` expectancy and skip-on-down logic.
- [ ] `test_price.py`: happy paths, invalid symbol (400), unknown symbol (404).
- [ ] `test_history.py`: 1y daily history integrity checks, invalid interval, latest candle timestamp.
- [ ] `test_profile.py`: company metadata presence & schema sanity.
- [ ] `test_coverage.py`: booleans for coverage matrix.
- [ ] `test_batch.py`: mixed symbol batch handling (success/error), invalid payload validation.
- [ ] `test_options_chain.py`: expiries, strikes, greeks numeric.
- [ ] `test_options_suggestions.py`: descending score order, numeric validation.
- [ ] `test_admin_stats.py`: counter presence, non-negative metrics.
- [ ] `test_fallback_and_cache.py`: cache header detection, respx/httpx mock fallback, TTL simulation.

## 4. WebSocket Validation
- [ ] Async WebSocket test verifying first tick ≤ 2s, JSON schema.
- [ ] Multi-subscriber or load-oriented test (optional; pair with perf scripts).
- [ ] Negative test for invalid symbol closure.

## 5. BDD Scenarios (`tests/bdd/`)
- [ ] Write features: price, options, ws (already sketched).
- [ ] Implement step definitions sharing fixtures with pytest.
- [ ] Tag scenarios with `@profile` to trigger VizTracer.
- [ ] Provide reusable HTTP/WS helpers and context storage.

## 6. Contract & Fuzz Testing
- [ ] Author `scripts/run-schemathesis.sh` to invoke `schemathesis run ${BASE_URL}/openapi.json --checks all`.
- [ ] Wire Makefile target `contract` and ensure artifacts saved to `artifacts/schemathesis.xml`.
- [ ] Add CI job step running script with retries/logging.

## 7. Performance Smoke (`tests/perf/`)
- [ ] `k6-price-smoke.js`: 50 RPS / 1 min, thresholds p95 < 250ms.
- [ ] `k6-ws-smoke.js`: 100 concurrent subscribers, first tick validation.
- [ ] Document prerequisites (install k6) and environment variables.

## 8. Data Quality Checks
- [ ] Define Great Expectations-like `expectations_history.json`.
- [ ] Optional: integrate into tests to assert schema (or document manual usage).

## 9. Reporting & Artifacts
- [ ] Configure pytest to emit JUnit + HTML into `artifacts/`.
- [ ] Capture VizTracer traces from `@profile` tests.
- [ ] Ensure CI uploads artifacts on failure.

## 10. Documentation
- [ ] README quick start (venv, install, env vars, commands).
- [ ] Link to official test plan excerpt & acceptance criteria alignment.
- [ ] Provide instructions for VizTracer, Schemathesis, k6 usage.
- [ ] Document toggles (`LIVE_TESTS`, `ENABLE_PROVIDER_MOCKS`).

## 11. Stretch / Nice-to-Have
- [ ] Add nightly scheduled job for live smoke (optional).
- [ ] Integrate Great Expectations runtime validation into pytest markers.
- [ ] Provide Dockerfile/compose snippet to run tests in container.
- [ ] Publish coverage badge or summary in CI logs.

