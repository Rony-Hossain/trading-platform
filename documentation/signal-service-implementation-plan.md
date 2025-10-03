# Signal Service Implementation Plan

## Executive Summary

**Objective:** Build a production-grade orchestration service that transforms complex quantitative models into a beginner-friendly "Today's Plan" with plain-English explanations, safety guardrails, and expert progressive disclosure.

**Timeline:** 8 weeks to production-ready MVP
**Team Size:** 2-3 engineers
**Success Criteria:** p95 latency < 150ms, 99.9% availability, beginner safety guardrails enforced, audit trail for all decisions

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (React + TanStack Query)           │
│                     Bundle: ~80KB gzipped                       │
└──────────────────────┬──────────────────────────────────────────┘
                       │ REST API (JSON)
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SIGNAL SERVICE (NEW)                         │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐ │
│  │ Translators  │ Aggregators  │ Guardrails   │ Observability│ │
│  │ (beginner/   │ (orchestrate │ (volatility  │ (metrics/    │ │
│  │  expert)     │  upstreams)  │  brake, caps)│  logs/SLOs)  │ │
│  └──────────────┴──────────────┴──────────────┴──────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Decision Store (Redis) - Immutable audit trail             │ │
│  │ Idempotency Manager - Prevent double-execution             │ │
│  │ Policy Manager - Hot-reloadable rules (YAML)               │ │
│  │ SWR Cache - Stale-while-revalidate (30s TTL / 10s SWR)     │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Parallel fetch (timeouts enforced)
        ┌──────────────┼──────────────┬───────────────┐
        ▼              ▼              ▼               ▼
┌───────────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────┐
│ Inference     │ │ Forecast │ │ Sentiment │ │ Portfolio    │
│ Service       │ │ Service  │ │ Service   │ │ Service      │
│ (60ms budget) │ │ (80ms)   │ │ (80ms)    │ │ (100ms)      │
│ EXISTING      │ │ EXISTING │ │ EXISTING  │ │ EXISTING     │
└───────────────┘ └──────────┘ └───────────┘ └──────────────┘
```

### Key Design Principles

1. **Backend-Driven:** All decision logic on server; frontend is thin rendering layer
2. **Graceful Degradation:** Service works with partial upstream failures
3. **Safety-First:** Server-enforced guardrails for beginners (no client bypass)
4. **Audit Trail:** Every decision stored immutably for 30 days
5. **Observable:** Structured logs, Prometheus metrics, SLO tracking from day one

---

## Phase 1: Foundation (Week 1-2)

### Week 1: Core Infrastructure

#### Task 1.1: Project Scaffold
**Owner:** Backend Lead
**Duration:** 1 day

```bash
services/signal-service/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app
│   ├── config.py                  # Settings via Pydantic
│   ├── core/
│   │   ├── contracts.py           # Pydantic schemas (ALL response models)
│   │   ├── observability.py       # Structured logging + Prometheus
│   │   ├── decision_store.py      # Immutable audit trail (Redis)
│   │   ├── idempotency.py         # Deduplication for actions
│   │   ├── policy_manager.py      # Hot-reload YAML policies
│   │   ├── cache_manager.py       # Stale-while-revalidate
│   │   └── slo_tracker.py         # Error budget monitoring
│   ├── upstream/
│   │   ├── base_client.py         # Circuit breaker base class
│   │   ├── inference_client.py    # 60ms timeout
│   │   ├── forecast_client.py     # 80ms timeout
│   │   ├── sentiment_client.py    # 80ms timeout (optional)
│   │   └── portfolio_client.py    # 100ms timeout
│   ├── translators/
│   │   ├── beginner_translator.py # Complex → Plain English
│   │   ├── expert_translator.py   # Add indicators/options panels
│   │   ├── reason_builder.py      # reason_codes → human text
│   │   ├── reason_scorer.py       # Quality scoring (0-1)
│   │   └── driver_calculator.py   # Top 3 feature importances
│   ├── aggregators/
│   │   ├── plan_aggregator.py     # Orchestrate Today's Plan
│   │   ├── alert_aggregator.py    # Throttle + dedupe alerts
│   │   └── portfolio_aggregator.py# Simplified positions view
│   ├── guardrails/
│   │   ├── guardrail_engine.py    # Volatility brake, position limits
│   │   └── fitness_checker.py     # Liquidity, spread, size checks
│   └── api/
│       └── v1/
│           ├── plan.py            # GET /plan
│           ├── alerts.py          # GET /alerts, POST /alerts/arm
│           ├── actions.py         # POST /buy, POST /sell (idempotent)
│           ├── positions.py       # GET /positions
│           ├── explain.py         # GET /explain/{term}
│           └── internal.py        # Health, SLO status, admin endpoints
├── config/
│   └── policies.yaml              # Hot-reloadable rules
├── tests/
│   ├── fixtures/
│   │   ├── normal.json
│   │   ├── volatile_earnings.json
│   │   ├── degraded_sentiment.json
│   │   └── drift_red.json
│   ├── contract_tests/
│   │   ├── test_plan_contract.py
│   │   ├── test_alerts_contract.py
│   │   └── test_explain_contract.py
│   └── integration/
│       ├── test_plan_aggregator.py
│       └── test_upstream_degradation.py
├── scripts/
│   ├── check_slos.py              # CI + cron SLO validator
│   └── load_test.py               # Locust load test
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── RUNBOOK.md
└── README.md
```

**Acceptance:**
- [ ] Repo structure created
- [ ] FastAPI app runs on port 8000
- [ ] Health check endpoint responds: `GET /health`
- [ ] Prometheus `/metrics` endpoint exposes basic metrics

---

## Timeline Summary

| **Phase** | **Weeks** | **Deliverables** |
|-----------|-----------|------------------|
| **Phase 1: Foundation** | 1-2 | Core contracts, observability, policies, decision store, idempotency, SWR cache |
| **Phase 2: Upstream Integration** | 3-4 | Upstream clients (circuit breakers), translators, guardrails, fitness checks |
| **Phase 3: Aggregation & API** | 5-6 | Plan aggregator, alerts, actions, positions, explain endpoints |
| **Phase 4: Expert Mode & Testing** | 7-8 | Expert panels, drivers, contract tests, integration tests, SLO monitoring, runbook |
| **Phase 5: Deployment** | 9-11 | Staging deployment, frontend integration, E2E tests, production rollout (canary → 100%) |

**Total:** 11 weeks from kickoff to production (with 2-3 engineer team)

---

## Success Criteria (MVP Acceptance)

### Reliability
- [x] p95 latency < 150ms (measured over 1hr with normal load)
- [x] p99 latency < 350ms
- [x] Availability ≥ 99.9% (30-day window)
- [x] Graceful degradation: service works with 1 upstream down (served from cache)
- [x] SWR cache serves stale data < 20ms

### Safety
- [x] All beginner picks pass fitness checks (liquidity, spread, position size)
- [x] Daily cap enforced server-side (no client bypass)
- [x] Idempotent actions prevent double-execution
- [x] Volatility brake activates correctly (position size reduced)
- [x] Earnings window suppresses beginner picks (or reduces size for experts)
- [x] Conservative mode reduces risk by ≥ 70% when enabled

### Explainability
- [x] Every pick has `reason_codes` (machine-readable)
- [x] `reason` field is plain English (CEFR B1 readability)
- [x] Top 3 drivers available for expert mode
- [x] Decision snapshots stored for 30 days (audit trail)

### Observability
- [x] Structured logs with `request_id` for tracing
- [x] Prometheus metrics exported (`/metrics` endpoint)
- [x] SLO dashboard live in Grafana
- [x] Error budget tracking active
- [x] Alerts configured for SLO violations (Slack/PagerDuty)

---

**Document Version:** 1.0
**Last Updated:** 2025-10-02
**Next Review:** 2025-10-16 (after Week 2 milestone)
