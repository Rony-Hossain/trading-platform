# Signal Service Runbook

## Overview
Operational runbook for Signal Service - beginner-friendly trading orchestration layer.

**Service Owner:** Trading Platform Team
**On-Call Rotation:** PagerDuty schedule "signal-service"
**Escalation Path:** Backend Lead → Engineering Manager → VP Engineering

---

## Quick Reference

### Service Endpoints
- **Production:** https://api.trading-platform.com/signal
- **Staging:** https://staging-api.trading-platform.com/signal
- **Health:** `GET /health`
- **Metrics:** `GET /metrics`
- **SLO Status:** `GET /internal/slo/status`

### Key Dashboards
- **Grafana:** https://grafana.company.com/d/signal-service
- **Error Budget:** https://grafana.company.com/d/signal-service-slo
- **Upstream Dependencies:** https://grafana.company.com/d/signal-service-upstream

### Log Locations
- **Production:** CloudWatch Logs `/aws/ecs/signal-service/production`
- **Staging:** CloudWatch Logs `/aws/ecs/signal-service/staging`
- **Query:** Use request_id for distributed tracing

---

## Common Incidents

### 1. High Latency (p95 > 150ms)

**Symptoms:**
- Grafana alert: "Signal Service p95 Latency High"
- Users report slow "Today's Plan" loading
- Metrics show p95 > 150ms

**Investigation:**
```bash
# Check SLO status
curl http://localhost:8000/internal/slo/status | jq

# Check upstream latencies
curl http://localhost:8000/metrics | grep upstream_latency

# Check cache hit rate
curl http://localhost:8000/internal/stats | jq '.cache'
```

**Common Causes:**
1. **Upstream service degradation** → Check circuit breaker status
2. **Cache misses** → Verify Redis connectivity
3. **High traffic** → Check request rate

**Resolution:**
```bash
# 1. Check upstream circuit breakers
curl http://localhost:8000/internal/stats | jq

# 2. Verify Redis connection
redis-cli -h $REDIS_HOST ping

# 3. If inference service slow, enable conservative mode
kubectl scale deployment inference-service --replicas=5

# 4. Increase cache TTL temporarily (edit policies.yaml, reload)
curl -X POST http://localhost:8000/internal/policy/reload
```

---

### 2. Error Budget Exhausted

**Symptoms:**
- Grafana alert: "Signal Service Error Budget < 10%"
- SLO dashboard shows budget_remaining_pct < 10%
- Availability < 99.9%

**Investigation:**
```bash
# Check error budget
python scripts/check_slos.py --url http://localhost:8000

# Get detailed error budget
curl http://localhost:8000/internal/slo/error-budget?window_days=7 | jq

# Check recent errors
grep "level=error" /var/log/signal-service/app.log | tail -20
```

**Common Causes:**
1. **Upstream service failures** → Check degraded_fields in responses
2. **Database connectivity** → Check Redis health
3. **Bad deployment** → Check recent deploys

**Resolution:**
```bash
# 1. Enable graceful degradation (already automatic)
# Service continues with cached data

# 2. If critical upstream down, scale it up
kubectl scale deployment $UPSTREAM_SERVICE --replicas=3

# 3. If deployment issue, rollback
kubectl rollout undo deployment/signal-service

# 4. Coordinate with upstream team
# Post in #platform-incidents Slack channel
```

---

### 3. Circuit Breaker Open

**Symptoms:**
- Logs show: "circuit_breaker_open"
- Users see degraded_fields in /plan response
- Specific upstream service unavailable

**Investigation:**
```bash
# Check circuit breaker states
curl http://localhost:8000/internal/stats | jq '.circuit_breakers'

# Check upstream health
curl http://$INFERENCE_SERVICE_URL/health
curl http://$FORECAST_SERVICE_URL/health
curl http://$SENTIMENT_SERVICE_URL/health
curl http://$PORTFOLIO_SERVICE_URL/health
```

**Common Causes:**
1. **Upstream service down** → Check upstream health
2. **Network issues** → Check connectivity
3. **Timeout too aggressive** → Review timeout settings

**Resolution:**
```bash
# 1. Check upstream service status
kubectl get pods -l app=$UPSTREAM_SERVICE

# 2. If upstream healthy but circuit open, wait for recovery timeout (60s)
# Circuit will automatically transition to HALF_OPEN

# 3. If persistent issues, increase timeout in config
# Edit config/policies.yaml, reload policies
curl -X POST http://localhost:8000/internal/policy/reload

# 4. Manual circuit breaker reset (emergency only)
# Restart service to reset all circuit breakers
kubectl rollout restart deployment/signal-service
```

---

### 4. Guardrail Violations

**Symptoms:**
- Users report "can't execute trade"
- Logs show: "guardrail_check_failed"
- Actions endpoint returning 400 errors

**Investigation:**
```bash
# Check recent guardrail violations
grep "guardrail_check_failed" /var/log/signal-service/app.log | tail -10

# Check current policy
curl http://localhost:8000/internal/policy/current | jq '.policies.beginner_mode'
```

**Common Causes:**
1. **User hit daily trade cap** → Expected behavior
2. **Volatility brake triggered** → Market conditions
3. **Policy misconfiguration** → Review policies.yaml

**Resolution:**
```bash
# 1. Verify policy is correct
cat config/policies.yaml | grep -A 5 "beginner_mode"

# 2. If policy incorrect, fix and reload
vim config/policies.yaml
curl -X POST http://localhost:8000/internal/policy/reload

# 3. If legitimate guardrail, communicate to user
# This is working as designed for beginner safety
```

---

### 5. Idempotency Failures

**Symptoms:**
- Users report "duplicate action" errors
- Logs show: "action_duplicate_detected"
- Same Idempotency-Key used twice

**Investigation:**
```bash
# Check Redis for idempotency records
redis-cli -h $REDIS_HOST
> KEYS action:idem:*
> GET action:idem:<IDEMPOTENCY_KEY>

# Check action record
curl http://localhost:8000/internal/decision/<REQUEST_ID> | jq
```

**Common Causes:**
1. **Client retry without new key** → Expected behavior
2. **Redis key not expiring** → Check Redis TTL
3. **Clock skew** → Check server time

**Resolution:**
```bash
# 1. This is usually working as designed
# Idempotency prevents duplicate trades

# 2. If legitimate retry needed, client should use new key
# Communicate to frontend team

# 3. If Redis TTL issue, check TTL
redis-cli -h $REDIS_HOST TTL action:idem:<KEY>

# 4. Manual cleanup (emergency only)
redis-cli -h $REDIS_HOST DEL action:idem:<KEY>
```

---

## Deployment

### Pre-Deployment Checklist
- [ ] All tests passing in CI
- [ ] Staging deployment successful
- [ ] SLO status healthy in staging
- [ ] Load test passed (p95 < 150ms)
- [ ] Policy changes reviewed
- [ ] Rollback plan documented

### Deployment Process

```bash
# 1. Deploy to staging
kubectl apply -f k8s/staging/

# 2. Verify staging health
curl https://staging-api.trading-platform.com/signal/health
python scripts/check_slos.py --url https://staging-api.trading-platform.com/signal

# 3. Run load test
locust -f scripts/load_test.py --host=https://staging-api.trading-platform.com/signal

# 4. Deploy to production (canary)
kubectl apply -f k8s/production/canary/

# 5. Monitor for 10 minutes
# Check Grafana dashboard, error rates, latency

# 6. If healthy, promote to 100%
kubectl apply -f k8s/production/full/

# 7. Monitor for 1 hour
```

### Rollback

```bash
# Quick rollback
kubectl rollout undo deployment/signal-service

# Rollback to specific revision
kubectl rollout history deployment/signal-service
kubectl rollout undo deployment/signal-service --to-revision=5
```

---

## Configuration Management

### Hot-Reload Policies

```bash
# 1. Edit policies.yaml
vim config/policies.yaml

# 2. Validate YAML
python -c "import yaml; yaml.safe_load(open('config/policies.yaml'))"

# 3. Reload via API
curl -X POST http://localhost:8000/internal/policy/reload

# 4. Verify new policy
curl http://localhost:8000/internal/policy/current | jq '.policies.version'

# Alternative: Send SIGHUP signal
kill -HUP $(pgrep -f "signal-service")
```

### Environment Variables

Key environment variables:
- `REDIS_HOST` - Redis hostname
- `INFERENCE_SERVICE_URL` - Inference service URL
- `INFERENCE_TIMEOUT_MS` - Inference timeout (default: 60ms)
- `PLAN_CACHE_TTL` - Plan cache TTL (default: 30s)
- `DEBUG` - Enable debug mode (default: false)

Update via Kubernetes ConfigMap:
```bash
kubectl edit configmap signal-service-config
kubectl rollout restart deployment/signal-service
```

---

## Monitoring & Alerts

### Key Metrics

**Latency:**
- `plan_requests_total` - Total plan requests
- `upstream_latency_seconds` - Upstream service latency
- `slo_availability` - SLO availability gauge

**Circuit Breakers:**
- `circuit_breaker_state` - Circuit breaker state (0=closed, 1=open, 2=half_open)
- `circuit_breaker_failures` - Failure count

**Cache:**
- `cache_hit_rate` - SWR cache hit rate
- `cache_stale_served` - Stale responses served

### Alert Thresholds

| Alert | Threshold | Severity |
|-------|-----------|----------|
| p95 Latency High | > 150ms for 5 min | Warning |
| p99 Latency Critical | > 300ms for 5 min | Critical |
| Error Budget Low | < 25% remaining | Warning |
| Error Budget Critical | < 10% remaining | Critical |
| Circuit Breaker Open | Any upstream | Warning |
| Availability Low | < 99.9% over 1hr | Critical |

---

## Data Retention

- **Decision Snapshots:** 30 days (Redis)
- **Idempotency Records:** 5 minutes (10 minutes if successful)
- **Logs:** 90 days (CloudWatch)
- **Metrics:** 1 year (Prometheus)

---

## Contact Information

- **On-Call:** PagerDuty "signal-service" schedule
- **Slack:** #signal-service-alerts
- **Email:** platform-team@company.com
- **Documentation:** https://wiki.company.com/signal-service

---

## Troubleshooting Tools

### Useful Commands

```bash
# Check service status
kubectl get pods -l app=signal-service

# View logs
kubectl logs -f deployment/signal-service --tail=100

# Port forward for local testing
kubectl port-forward svc/signal-service 8000:8000

# Execute into pod
kubectl exec -it $(kubectl get pod -l app=signal-service -o name | head -1) -- /bin/bash

# Check Redis
redis-cli -h $REDIS_HOST ping
redis-cli -h $REDIS_HOST INFO stats

# Run SLO check
python scripts/check_slos.py --url http://localhost:8000 --alert
```

### Debug Mode

Enable debug logging:
```bash
# Temporary (until restart)
kubectl set env deployment/signal-service LOG_LEVEL=DEBUG

# Permanent
kubectl edit configmap signal-service-config
# Set LOG_LEVEL: DEBUG
kubectl rollout restart deployment/signal-service
```

---

## Appendix

### Related Services
- **Inference Service:** Model predictions
- **Forecast Service:** Technical indicators
- **Sentiment Service:** Market sentiment
- **Portfolio Service:** User positions

### Useful Links
- **Repository:** https://github.com/company/trading-platform
- **CI/CD:** https://jenkins.company.com/signal-service
- **SLO Dashboard:** https://grafana.company.com/d/signal-service-slo
- **Architecture Docs:** https://wiki.company.com/signal-service/architecture
