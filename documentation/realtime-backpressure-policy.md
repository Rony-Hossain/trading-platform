### Real-time Backpressure Policy

- **Scope**: Applies to `RealTimeData` (lib/api.ts) and real-time consumers (`StockChart`, `RealTimeAlerts`).
- **Update Cadence**: Beginner streams throttle to 1 update/sec; expert streams up to 4 updates/sec (`REALTIME_BUDGETS`).
- **Drop Strategy**: Excess ticks collapse to the latest value (`drop-oldest`) with telemetry (`realtime_throttled`).
- **Point Caps**: Chart buffers trim to 500 points (`StockChart.MAX_POINTS`) to avoid rendering blowups.
- **Tenant Controls**: Mode-aware subscriptions pass `{ mode }`, letting build/site config cap expert throttles when modules disabled.
