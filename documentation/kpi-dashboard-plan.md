# Launch KPI Ownership & Dashboards

| KPI | Owner | Dashboard | Notes |
| --- | --- | --- | --- |
| Beginner activation % | Growth PM | `dashboards/activation.json` | Derived from mode-switch events via `launchKpiTracker`. |
| Alert follow-through % | Trading Lead | `dashboards/alerts.json` | Canary vs control split, stop rule triggers rollback. |
| Median alert P&L | Quant Lead | `dashboards/pnl.json` | Tracks expected vs realized P&L samples. |
| Alert helpfulness % | UX Research | `dashboards/feedback.json` | Aggregates feedback from alerts + diagnostics. |
| Loss-cap saves | Risk Officer | `dashboards/risk.json` | Populated from PlanList daily-cap observer. |

Dashboards are provisioned in the analytics workspace; see `components/analytics/LaunchKpiMonitor.tsx` for the telemetry feed emitted once per minute.
