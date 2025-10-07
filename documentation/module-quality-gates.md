# Module Quality Gates

| Module | Definition of Done | Tests | Accessibility | i18n | Telemetry |
| --- | --- | --- | --- | --- | --- |
| Today (@modules/today) | Plan cards, stale banner, news list | Contract tests + unit coverage | Keyboard/SR verified | CopyService keys prefixed `plan.*` | PLAN events + launch KPI |
| Alerts (@modules/alerts) | Drawer, actions, feedback | Alert contract test | Drawer focus trap + shortcuts | CopyService `alerts.*` | ALERTS taxonomy + experiments |
| Diagnostics (@modules/diagnostics) | Chip + trends + driver feedback | Driver feedback coverage | Buttons accessible | CopyService `feedback.*` | ML_INSIGHTS events |
| Admin site-config | Flag overrides, module table | React testing (planned) | High contrast pass | CopyService `nav.admin` | Telemetry (launch review) |

Semantic versioning: bump minor for feature work, patch for fixes. Update module README with migration notes and cross-link docs.
