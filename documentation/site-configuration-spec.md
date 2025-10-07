# Site Configuration Specification

The site configuration is loaded via `lib/config/site-config.ts` and may be sourced from env or remote JSON. A canonical JSON structure:

```json
{
  "defaultMode": "beginner",
  "features": {
    "paperTradingEnabled": true,
    "expertModeEnabled": true,
    "newsIntegrationEnabled": true,
    "advancedChartsEnabled": false,
    "optionsTradingEnabled": false
  },
  "modules": {
    "today": true,
    "portfolio": true,
    "alerts": true,
    "journal": false,
    "settings": true,
    "learn": true,
    "mlInsights": true
  },
  "theme": {
    "primary": "#2070F3",
    "secondary": "#63A0F6"
  }
}
```

Runtime overrides are stored in localStorage by `FeatureFlagProvider`; the admin panel (`/admin/site-config`) manipulates overrides safely without mutating the base config. Hot reload support: call `resetSiteConfig()` from `lib/config/site-config.ts` after updating the remote JSON to force re-fetch.
