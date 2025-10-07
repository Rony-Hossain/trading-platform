# Privacy & PII Handling
- **Telemetry**: `trackEvent` sanitises payloads via `lib/privacy/sanitize.ts`, redacting email/account identifiers before any outbound send.
- **UI notices**: Alerts and news surfaces remind users of retention windows and provider ToS (`components/alerts/AlertTriggerItem.tsx`, `components/news/NewsQuickList.tsx`).
- **Data retention**: Feedback data is transient and surfaced as “stored for 30 days” in copy; backend should align retention schedules.
- **Consent gating**: User preferences persist locally (`useUserStore`) and device sync audits (`useDeviceSyncAudit`) keep session restore compliant—do not emit personal data without explicit consent flags.
