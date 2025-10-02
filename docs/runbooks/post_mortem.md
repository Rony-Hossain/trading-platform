# Incident Post-Mortem Template

**Incident ID**: [INC-YYYY-NNNN]
**Date of Incident**: [YYYY-MM-DD]
**Post-Mortem Author**: [Name]
**Post-Mortem Date**: [YYYY-MM-DD] (within 48h of incident)
**Incident Severity**: 🔴 Sev 1 / 🟡 Sev 2 / 🟢 Sev 3

---

## Executive Summary

**TL;DR**: [1-2 sentence summary of what happened and impact]

**Example**: Production momentum alpha model experienced 45-minute outage due to feature service timeout. $12K in missed trading opportunities. Root cause: database connection pool exhaustion. Mitigated with connection pool scaling and circuit breaker.

---

## 1. Incident Overview

### 1.1 What Happened?

[Clear, chronological description of the incident from start to resolution]

**Example**:
On 2025-10-01 at 10:15 AM ET, the momentum alpha model stopped generating predictions due to feature service timeouts. The feature service was unable to retrieve real-time market data from the PostgreSQL database. Investigation revealed the connection pool had been exhausted by long-running queries from a separate analytics job. The incident was resolved at 11:00 AM ET by scaling the connection pool and terminating the blocking queries.

### 1.2 Impact

**Business Impact**:
- Missed trading opportunities: $12,000 estimated (30 minutes of downtime during high-volume period)
- Model downtime: 45 minutes
- Affected strategies: Momentum Alpha v1.0 (20% of daily volume)

**Customer/User Impact**:
- Internal only (no external customer impact)
- Trading desk had to manual override for 45 minutes

**Systems Affected**:
- Momentum Alpha prediction service
- Feature Store API
- PostgreSQL market data database

### 1.3 Timeline

| Time (ET) | Event | Who |
|-----------|-------|-----|
| 10:15 AM | First alert: Feature service latency > 5s | PagerDuty |
| 10:17 AM | On-call engineer paged | J. Smith |
| 10:20 AM | Investigation began, identified database timeouts | J. Smith |
| 10:25 AM | Escalated to database team | J. Smith |
| 10:30 AM | Root cause identified: connection pool exhaustion | K. Lee (DB) |
| 10:35 AM | Temporary fix: killed blocking queries | K. Lee |
| 10:40 AM | Service recovering, predictions resuming | J. Smith |
| 10:45 AM | Connection pool scaled from 50 to 100 | K. Lee |
| 11:00 AM | Incident declared resolved | J. Smith |
| 11:30 AM | Post-incident review scheduled | Team |

---

## 2. Trigger & Detection

### 2.1 How Was the Incident Triggered?

**Trigger Type**:
- [ ] Alert breach (specify metric)
- [ ] VaR breach
- [ ] Model performance degradation
- [ ] Infrastructure failure
- [ ] Data quality issue
- [ ] Human error
- [ ] External dependency failure
- [ ] Other: [specify]

**Specific Trigger**: [Describe what initiated the incident]

**Example**: A long-running analytics query (analyzing 6 months of tick data) started at 10:00 AM and consumed 40 of 50 available database connections. Concurrent real-time feature requests from the model service exhausted the remaining 10 connections within 15 minutes, causing cascading timeouts.

### 2.2 How Was the Incident Detected?

**Detection Method**:
- [ ] Automated monitoring alert (Prometheus/Grafana)
- [ ] Manual observation (trading desk, engineer)
- [ ] Customer report
- [ ] Scheduled health check
- [ ] Post-trade analysis
- [ ] Other: [specify]

**Time to Detect** (TTD): [N minutes from trigger to detection]

**Alert Details**:
- Alert name: [e.g., "feature_service_latency_high"]
- Threshold breached: [e.g., "p95 latency > 5000ms"]
- Alert destination: [PagerDuty, Slack, Email]

---

## 3. Root Cause Analysis

### 3.1 Root Cause

**Primary Root Cause**: [Single sentence root cause]

**Example**: Database connection pool was sized too small (50 connections) to handle concurrent real-time and analytical workloads.

**Contributing Factors**:
1. **[Factor 1]**: [Description]
2. **[Factor 2]**: [Description]
3. **[Factor 3]**: [Description]

**Example**:
1. **Lack of workload isolation**: Real-time and analytical queries shared same connection pool
2. **Missing circuit breaker**: Feature service retried failed requests, amplifying load
3. **Insufficient monitoring**: No alerts on connection pool utilization

### 3.2 Why Did It Happen?

**5 Whys Analysis**:

1. **Why did the model stop generating predictions?**
   - Because the feature service timed out when fetching features from the database.

2. **Why did the feature service time out?**
   - Because the database connection pool was exhausted.

3. **Why was the connection pool exhausted?**
   - Because a long-running analytics query consumed 40 of 50 connections, leaving only 10 for real-time requests.

4. **Why did the analytics query consume so many connections?**
   - Because there was no resource isolation between real-time and batch workloads.

5. **Why was there no resource isolation?**
   - Because we didn't anticipate this usage pattern during initial architecture design.

### 3.3 What Worked?

**Things that went well**:
- ✅ [Positive aspect 1]
- ✅ [Positive aspect 2]
- ✅ [Positive aspect 3]

**Example**:
- ✅ Alert fired within 2 minutes of latency spike
- ✅ On-call engineer responded quickly (5 min from page to investigation)
- ✅ Database team identified root cause quickly (15 min)
- ✅ Temporary mitigation was effective (killed blocking queries)

### 3.4 What Didn't Work?

**Things that could have been better**:
- ❌ [Issue 1]
- ❌ [Issue 2]
- ❌ [Issue 3]

**Example**:
- ❌ No monitoring on database connection pool utilization
- ❌ No circuit breaker to prevent retry storms
- ❌ No resource isolation between workloads
- ❌ Alert escalation was slower than expected (2 min delay)

---

## 4. Resolution & Mitigation

### 4.1 How Was the Incident Resolved?

**Immediate Actions** (during incident):
1. [Action 1 with timestamp]
2. [Action 2 with timestamp]
3. [Action 3 with timestamp]

**Example**:
1. 10:35 AM: Killed blocking analytics query to free connections
2. 10:40 AM: Restarted feature service to clear retry queue
3. 10:45 AM: Scaled connection pool from 50 to 100 connections
4. 11:00 AM: Verified all systems operational

**Time to Mitigate** (TTM): [N minutes from detection to mitigation]

### 4.2 Temporary vs. Permanent Fixes

**Temporary Mitigations** (applied during incident):
- [Temp fix 1]: [Description and timeline]
- [Temp fix 2]: [Description and timeline]

**Permanent Fixes** (long-term solutions):
- [Permanent fix 1]: [Description, owner, timeline]
- [Permanent fix 2]: [Description, owner, timeline]

**Example**:

**Temporary**:
- Scaled connection pool to 100 (immediate)
- Killed blocking analytics query (immediate)

**Permanent**:
- Implement connection pool per workload type (J. Smith, by 2025-10-15)
- Add circuit breaker to feature service (K. Lee, by 2025-10-10)
- Add connection pool utilization monitoring (M. Patel, by 2025-10-08)

---

## 5. Action Items & Preventive Measures

### 5.1 Immediate Action Items (< 1 week)

| Action | Owner | Due Date | Status | Priority |
|--------|-------|----------|--------|----------|
| [Action 1] | [Name] | YYYY-MM-DD | ⏳ / ✅ / ❌ | 🔴 / 🟡 / 🟢 |
| [Action 2] | [Name] | YYYY-MM-DD | ⏳ / ✅ / ❌ | 🔴 / 🟡 / 🟢 |

**Example**:
| Action | Owner | Due Date | Status | Priority |
|--------|-------|----------|--------|----------|
| Add connection pool utilization alerts | M. Patel | 2025-10-08 | ⏳ | 🔴 |
| Implement circuit breaker in feature service | K. Lee | 2025-10-10 | ⏳ | 🔴 |
| Document connection pool sizing guidelines | J. Smith | 2025-10-07 | ⏳ | 🟡 |

### 5.2 Short-Term Actions (1-4 weeks)

| Action | Owner | Due Date | Status | Priority |
|--------|-------|----------|--------|----------|
| [Action 1] | [Name] | YYYY-MM-DD | ⏳ / ✅ / ❌ | 🔴 / 🟡 / 🟢 |
| [Action 2] | [Name] | YYYY-MM-DD | ⏳ / ✅ / ❌ | 🔴 / 🟡 / 🟢 |

**Example**:
| Action | Owner | Due Date | Status | Priority |
|--------|-------|----------|--------|----------|
| Separate connection pools for real-time vs batch | J. Smith | 2025-10-15 | ⏳ | 🔴 |
| Implement read replicas for analytics queries | K. Lee | 2025-10-25 | ⏳ | 🟡 |
| Create runbook for connection pool incidents | J. Smith | 2025-10-20 | ⏳ | 🟢 |

### 5.3 Long-Term Improvements (> 1 month)

| Action | Owner | Due Date | Status | Priority |
|--------|-------|----------|--------|----------|
| [Action 1] | [Name] | YYYY-MM-DD | ⏳ / ✅ / ❌ | 🔴 / 🟡 / 🟢 |
| [Action 2] | [Name] | YYYY-MM-DD | ⏳ / ✅ / ❌ | 🔴 / 🟡 / 🟢 |

**Example**:
| Action | Owner | Due Date | Status | Priority |
|--------|-------|----------|--------|----------|
| Migrate to connection pooling service (PgBouncer) | K. Lee | 2025-11-15 | ⏳ | 🟡 |
| Implement workload-based resource quotas | M. Patel | 2025-11-30 | ⏳ | 🟢 |
| Chaos engineering test for DB failures | J. Smith | 2025-12-01 | ⏳ | 🟢 |

### 5.4 Process Improvements

**Documentation**:
- [ ] Update runbook: [runbook name]
- [ ] Update architecture diagram
- [ ] Update monitoring playbook
- [ ] Add to incident knowledge base

**Training**:
- [ ] Share learnings with [team/organization]
- [ ] Update on-call training materials
- [ ] Conduct incident response drill

**Monitoring & Alerting**:
- [ ] Add new alerts: [list metrics]
- [ ] Adjust alert thresholds: [list changes]
- [ ] Create new dashboard: [dashboard name]

---

## 6. Lessons Learned

### 6.1 What We Learned

**Technical Lessons**:
1. [Lesson 1]
2. [Lesson 2]
3. [Lesson 3]

**Example**:
1. Connection pools must be sized for peak concurrent workloads, not average
2. Real-time and batch workloads should have isolated resource pools
3. Circuit breakers are critical for preventing cascading failures

**Process Lessons**:
1. [Lesson 1]
2. [Lesson 2]

**Example**:
1. Database connection pool metrics should be monitored with alerts
2. Incident response playbooks need regular testing and updates

### 6.2 What Would We Do Differently?

**Next time we would**:
- [Improvement 1]
- [Improvement 2]
- [Improvement 3]

**Example**:
- Implement chaos engineering tests for database connection exhaustion
- Add automatic circuit breakers to all external service calls
- Create dedicated read replicas for analytics workloads

---

## 7. Risk Assessment

### 7.1 Recurrence Risk

**Likelihood of Recurrence**: 🟢 Low / 🟡 Medium / 🔴 High

**Reasoning**: [Why is recurrence likely or unlikely?]

**Example**: 🟡 Medium - While we've scaled the connection pool and added monitoring, we haven't yet implemented full workload isolation. Risk remains until permanent fixes are deployed.

### 7.2 Blast Radius

**Potential Future Impact**:
- Systems at risk: [List]
- Financial exposure: [$ estimate]
- Mitigation: [Controls in place]

**Example**:
- Systems at risk: All models using feature service (4 production models)
- Financial exposure: $50K/hour during peak trading
- Mitigation: Circuit breakers deployed, connection pool scaled 2x

---

## 8. Financial Impact

### 8.1 Direct Costs

| Cost Category | Amount | Notes |
|---------------|--------|-------|
| Lost trading opportunities | $[amount] | [Calculation method] |
| Engineering time | $[amount] | [Hours × rate] |
| Infrastructure changes | $[amount] | [One-time or recurring] |
| **Total Direct Cost** | **$[total]** | |

### 8.2 Indirect Costs

- Reputation impact: [Assessment]
- Team morale: [Assessment]
- Customer trust: [Assessment]

---

## 9. Communication

### 9.1 Internal Communication

**During Incident**:
- [ ] Trading desk notified: [Time]
- [ ] Management notified: [Time]
- [ ] Engineering team notified: [Time]
- [ ] Slack #incidents updated: [Frequency]

**Post-Incident**:
- [ ] Post-mortem review meeting: [Date/Time]
- [ ] All-hands announcement: [Yes/No]
- [ ] Email summary sent: [Yes/No]

### 9.2 External Communication

**Customer Communication** (if applicable):
- [ ] Customer notification sent: [Yes/No]
- [ ] Public status page updated: [Yes/No]
- [ ] Regulatory reporting required: [Yes/No]

---

## 10. Appendix

### 10.1 Supporting Data

**Logs**:
- Error logs: [Link to logs or snippet]
- Alert history: [Link or screenshot]

**Metrics**:
- Graphs showing incident: [Links to Grafana dashboards]
- Performance before/during/after: [Data]

**Screenshots**:
- [Attach relevant screenshots]

### 10.2 Related Incidents

**Similar Past Incidents**:
- [INC-YYYY-NNNN]: [Brief description and link]
- [INC-YYYY-NNNN]: [Brief description and link]

**Pattern Analysis**: [Are there recurring themes?]

### 10.3 References

- **Jira Ticket**: [PROJ-1234]
- **Slack Thread**: [#incidents thread link]
- **Runbook Updated**: [Link]
- **Code Changes**: [GitHub PR links]

---

## 11. Review & Approval

### 11.1 Post-Mortem Review

**Review Meeting**:
- **Date/Time**: [YYYY-MM-DD HH:MM]
- **Attendees**: [Names and roles]
- **Duration**: [N minutes]

**Discussion Notes**:
[Key points discussed during review meeting]

### 11.2 Sign-Off

**Post-Mortem Completed By**: _________________________ Date: __________

**Reviewed By**:
- Engineering Lead: _________________________ Date: __________
- Head of Trading: _________________________ Date: __________
- CTO: _________________________ Date: __________

**Action Item Tracking**: [Jira epic link for follow-up tasks]

---

## 12. Follow-Up

### 12.1 Action Item Status (30-day review)

**Review Date**: [YYYY-MM-DD]

| Original Action | Status | Notes |
|----------------|--------|-------|
| [Action 1] | ✅ / ⏳ / ❌ | [Update] |
| [Action 2] | ✅ / ✅ / ❌ | [Update] |

**Outstanding Items**: [List any incomplete actions with new timeline]

### 12.2 Effectiveness Assessment

**Did our fixes work?**
- No recurrence: ✅ / ❌
- Metrics improved: ✅ / ❌
- Team confidence: ✅ / ❌

**Further Actions Needed**: [Yes/No with details]

---

**Document Version**: 1.0
**Last Updated**: [YYYY-MM-DD]
**Status**: ✅ Complete / ⏳ Draft / 🔄 In Review

---

## Post-Mortem Best Practices

**Remember**:
- ✅ **Blameless**: Focus on systems and processes, not individuals
- ✅ **Actionable**: Every insight should lead to a concrete action
- ✅ **Timely**: Complete within 48 hours of incident resolution
- ✅ **Comprehensive**: Include all relevant details, timelines, and impacts
- ✅ **Shared**: Distribute learnings across the organization
- ✅ **Tracked**: All action items must have owners and due dates
- ✅ **Reviewed**: Follow up in 30 days to verify actions completed

**This is a learning opportunity, not a blame session.**
