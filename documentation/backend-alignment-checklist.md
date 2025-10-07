# Backend Alignment Checklist (Phase 9)
- **Contracts synced**: `tests/contracts/plan.contract.test.ts` and `tests/contracts/alert.contract.test.ts` cover the data shapes expected by the frontend. Update when backend introduces new fields (news arrays, analytics blocks, etc.).
- **Change control**: raise a ticket whenever plan picks add/remove keys or alerts modify safety analytics. Link contract tests in the ticket for visibility.
- **Drift/backpressure**: realtime limits (`lib/performance/budgets.ts`, `lib/api.ts`) require backend adherence to `max_points_per_chart` and `throttle_threshold_ms`. Notify backend if those budgets need adjustments per tenant.
- **Privacy**: telemetry now sanitises payloads (`lib/privacy/sanitize.ts`). Backend services ingesting telemetry should expect redacted email/account values.
