## Phase 8 Hardening Release Notes

### Security Enhancements
- Content Security Policy applied via `next.config.js` (default/self origins, strict framing, hardened transport headers).
- Subresource Integrity manifest generated for static assets with runtime enforcement on `/favicon.ico`.
- Build profiles published (`config/build-profiles.json`) allowing per-tenant module stripping with Webpack aliasing + runtime guards.
- Global status banners warn when offline or rate limited, providing retry guidance.

### Dependency License Audit
| Package | Version | License | Notes |
| --- | --- | --- | --- |
| next | 15.5.2 | MIT | Core framework |
| react / react-dom | 19.1.0 | MIT | UI runtime |
| @mui/material | 7.3.2 | MIT | UI component suite |
| lucide-react | 0.543.0 | ISC | Icon set |
| @tanstack/react-query | 5.87.1 | MIT | Data fetching/cache |
| axios | 1.11.0 | MIT | HTTP client |
| recharts | 3.2.0 | MIT | Charting library |

No new licenses introduced in Phase 8; all third-party packages remain permissive (MIT/ISC).
