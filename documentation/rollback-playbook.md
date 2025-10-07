# Frontend Rollback Playbook

## Canary & Experiment Stop Rules
- Alert tone experiment (`alert-tone`) monitors `follow_through_rate` with a 30% floor across 50 samples.
- Driver detail depth experiment (`driver-detail-depth`) monitors driver helpfulness with a 40% floor across 30 samples.
- When a rule fails, the experiment engine clears the canary assignment and reverts users to control on next render (see `lib/experiments/experiment-engine.ts`).

## Rollback Procedure
1. Open `/admin/site-config` and flip the feature flag override off or drag rollout slider to `0%`.
2. Confirm rollback event emits via telemetry (`launch_review_scheduled` + KPI snapshot).
3. Ensure navigation badge no longer highlights experimental tone for affected users.
4. Announce rollback in release notes update and log timestamp in incident tracker.

## Post-Rollback Checklist
- Capture latest KPI snapshot from `LaunchKpiMonitor` (median alert P&L, helpfulness %, loss-cap saves).
- Reset experiment assignments with `experimentEngine.clearAssignment(<id>)` if a clean restart is needed.
- Schedule follow-up via `scheduleLaunchReview` (automated for week-one review).
