# Modularization Charter
Last updated: 2025-10-03

## Vision

Build a composable, multi-tenant frontend where features can be selectively enabled, deployed, and scaled based on tenant requirements, user segments, and business needs.

## Core Principles

### 1. Bounded Contexts
Each module represents a distinct business capability with clear boundaries.

### 2. Loose Coupling
Modules communicate through well-defined contracts, not direct dependencies.

### 3. High Cohesion
Related functionality lives together within a module.

### 4. Progressive Enhancement
Beginner modules work standalone; expert modules extend base functionality.

### 5. Tenant Flexibility
Site configuration controls which modules are active for each tenant.

## Module Boundaries

### Core Modules (Always Active)

| Module | Responsibility | Exports |
|--------|---------------|---------|
| `@core/auth` | Authentication, session management | `useAuth`, `ProtectedRoute`, `login/logout` |
| `@core/api` | API client, error handling | `apiClient`, `useQuery`, `useMutation` |
| `@core/ui` | Base components, design system | `Button`, `Input`, `Modal`, `Card` |
| `@core/copy` | CopyService, i18n | `getCopy`, `useTranslation` |
| `@core/telemetry` | Event tracking, observability | `trackEvent`, `logger` |
| `@core/compliance` | Guardrails, feature flags | `getGuardrails`, `FeatureGate` |

### Feature Modules (Beginner)

| Module | Responsibility | Dependencies | Entry Points |
|--------|---------------|--------------|--------------|
| `@modules/today` | Today's plan view | `@core/api`, `@core/ui` | `/today` route |
| `@modules/portfolio` | Portfolio view, position details | `@core/api`, `@core/ui` | `/portfolio` route |
| `@modules/alerts` | Alert notifications, management | `@core/api`, `@core/ui` | `/alerts` route |
| `@modules/journal` | Trade journal, notes | `@core/api`, `@core/ui` | `/journal` route |
| `@modules/settings` | User settings, preferences | `@core/api`, `@core/ui` | `/settings` route |

### Feature Modules (Expert - Lazy Loaded)

| Module | Responsibility | Dependencies | Entry Points |
|--------|---------------|--------------|--------------|
| `@modules/indicators` | Technical indicators, chart overlays | `@core/ui`, `@modules/today` | Expert panel in `/today` |
| `@modules/options` | Options chain, strategies | `@core/api`, `@core/ui` | Expert panel in `/today` |
| `@modules/diagnostics` | Model diagnostics, drivers | `@core/api`, `@core/ui` | Expert panel in `/today` |
| `@modules/rules` | Custom alert rules, conditions | `@modules/alerts` | `/rules` route |
| `@modules/ml-insights` | ML drivers, regime hinting | `@core/api`, `@modules/today` | Expert panel in `/today` |
| `@modules/learn` | Educational content, lessons | `@core/ui` | `/learn` route |
| `@modules/explore` | Symbol search, discovery | `@core/api`, `@core/ui` | `/explore` route |

### Shared Utilities

| Module | Responsibility | Exports |
|--------|---------------|---------|
| `@shared/types` | TypeScript contracts | All interface definitions |
| `@shared/utils` | Common utilities | `formatCurrency`, `formatDate`, etc. |
| `@shared/hooks` | Reusable React hooks | `useMediaQuery`, `useDebounce` |
| `@shared/constants` | App constants | `ROUTES`, `API_ENDPOINTS` |

## Module Structure

### Standard Module Layout

```
@modules/today/
├── index.ts                 # Public API exports
├── routes/
│   └── TodayPage.tsx        # Route component
├── components/
│   ├── PlanCard.tsx         # Feature components
│   ├── PlanList.tsx
│   └── ExplainPopover.tsx
├── hooks/
│   ├── usePlanQuery.ts      # Data fetching hooks
│   └── useExplainEntry.ts
├── stores/
│   └── todayStore.ts        # UI state (Zustand)
├── types/
│   └── index.ts             # Module-specific types
├── utils/
│   └── planHelpers.ts       # Module utilities
├── config/
│   └── module.config.ts     # Module metadata
└── README.md                # Module documentation
```

### Module Configuration

Each module defines metadata for registration:

```typescript
// @modules/today/config/module.config.ts
import { ModuleConfig } from '@core/module-registry'

export const todayModuleConfig: ModuleConfig = {
  id: 'today',
  name: "Today's Plan",
  version: '1.0.0',
  mode: 'beginner', // 'beginner' | 'expert' | 'both'

  routes: [
    {
      path: '/today',
      component: () => import('../routes/TodayPage'),
      guard: 'authenticated',
    },
  ],

  permissions: ['plan:read'],

  dependencies: ['@core/api', '@core/ui'],

  telemetryNamespace: 'plan',

  featureFlags: ['enable_plan_view'],

  // Site configuration keys
  siteConfigKeys: [
    'plan.refresh_interval_ms',
    'plan.max_picks',
    'plan.show_confidence',
  ],
}
```

## Extension Points

### 1. Route Extensions

Modules can register routes dynamically:

```typescript
// @core/module-registry/router.ts
export function registerModuleRoutes(modules: ModuleConfig[]) {
  return modules.flatMap((module) =>
    module.routes.map((route) => ({
      ...route,
      element: (
        <FeatureGate feature={module.featureFlags[0]}>
          <Suspense fallback={<RouteLoader />}>
            <route.component />
          </Suspense>
        </FeatureGate>
      ),
    }))
  )
}
```

### 2. Component Slots

Modules can contribute components to extension slots:

```typescript
// @modules/indicators/extensions.ts
export const indicatorExtensions = {
  // Add indicator panel to Today view
  'today.expert-panel': {
    component: () => import('./components/IndicatorPanel'),
    priority: 1,
    condition: (mode) => mode === 'expert',
  },

  // Add indicator selector to settings
  'settings.expert-section': {
    component: () => import('./components/IndicatorSettings'),
    priority: 2,
  },
}
```

Usage in host module:

```tsx
// @modules/today/routes/TodayPage.tsx
import { ExtensionSlot } from '@core/module-registry'

export function TodayPage() {
  const { mode } = useUserSettings()

  return (
    <div>
      <PlanList />

      {/* Expert modules can inject here */}
      <ExtensionSlot
        name="today.expert-panel"
        context={{ mode }}
        fallback={null}
      />
    </div>
  )
}
```

### 3. Event Bus

Modules communicate via event bus:

```typescript
// @core/module-registry/event-bus.ts
import { EventEmitter } from 'events'

export const moduleEventBus = new EventEmitter()

// Module publishes event
moduleEventBus.emit('trade:executed', {
  symbol: 'AAPL',
  action: 'BUY',
  shares: 10,
})

// Other modules subscribe
moduleEventBus.on('trade:executed', (event) => {
  // @modules/portfolio updates positions
  // @modules/journal logs trade
  // @modules/today refreshes plan
})
```

## Site Configuration

### Configuration Schema

```typescript
// @core/site-config/schema.ts
export interface SiteConfig {
  // Tenant identification
  tenant_id: string
  tenant_name: string

  // Module enablement
  modules: {
    enabled: string[] // Module IDs
    disabled: string[] // Explicitly disabled
  }

  // Feature flags (per-tenant overrides)
  features: Record<string, boolean>

  // Theme overrides
  theme: {
    primary_color?: string
    logo_url?: string
    custom_css_url?: string
  }

  // Module-specific configuration
  plan: {
    refresh_interval_ms: number
    max_picks: number
    show_confidence: boolean
  }

  alerts: {
    delivery_channels: ('in_app' | 'push' | 'email' | 'sms')[]
    default_cooldown_sec: number
  }

  compliance: {
    region: Region
    jurisdiction: Jurisdiction
  }

  // Data provider keys (secure, backend-only)
  providers: {
    news_api_key?: string
    market_data_provider?: string
  }
}
```

### Configuration Loading

```typescript
// @core/site-config/loader.ts
export async function loadSiteConfig(): Promise<SiteConfig> {
  // Option 1: Static JSON (bundled at build time)
  if (process.env.CONFIG_MODE === 'static') {
    return require(`./configs/${process.env.TENANT_ID}.json`)
  }

  // Option 2: Remote JSON (fetched at runtime)
  if (process.env.CONFIG_MODE === 'remote') {
    const response = await fetch(
      `${process.env.CONFIG_API}/tenants/${process.env.TENANT_ID}/config`
    )
    return response.json()
  }

  // Option 3: Environment variables
  return {
    tenant_id: process.env.TENANT_ID!,
    tenant_name: process.env.TENANT_NAME!,
    modules: {
      enabled: process.env.ENABLED_MODULES?.split(',') || [],
      disabled: process.env.DISABLED_MODULES?.split(',') || [],
    },
    // ... map env vars to config
  }
}
```

### Hot Reload (Advanced)

```typescript
// @core/site-config/hot-reload.ts
export function watchConfigChanges() {
  const ws = new WebSocket(`${process.env.CONFIG_API}/ws/config`)

  ws.onmessage = (event) => {
    const newConfig = JSON.parse(event.data)

    // Validate config
    if (validateConfig(newConfig)) {
      // Update global config
      updateSiteConfig(newConfig)

      // Reload affected modules
      reloadModules(getChangedModules(oldConfig, newConfig))
    }
  }
}
```

## Module Lifecycle

### Registration

```typescript
// @core/module-registry/registry.ts
export class ModuleRegistry {
  private modules = new Map<string, ModuleConfig>()

  register(config: ModuleConfig) {
    // Validate dependencies
    for (const dep of config.dependencies) {
      if (!this.modules.has(dep) && !dep.startsWith('@core/')) {
        throw new Error(`Missing dependency: ${dep} for module ${config.id}`)
      }
    }

    // Check permissions
    if (!hasPermissions(config.permissions)) {
      console.warn(`Insufficient permissions for module ${config.id}`)
      return
    }

    // Register module
    this.modules.set(config.id, config)

    // Register routes
    registerModuleRoutes([config])

    // Register extensions
    registerExtensions(config.id, config.extensions)

    console.info(`Module ${config.id} registered successfully`)
  }

  unregister(moduleId: string) {
    const config = this.modules.get(moduleId)
    if (!config) return

    // Unregister routes
    unregisterModuleRoutes(config.routes)

    // Unregister extensions
    unregisterExtensions(moduleId)

    // Remove from registry
    this.modules.delete(moduleId)

    console.info(`Module ${moduleId} unregistered`)
  }

  isEnabled(moduleId: string): boolean {
    const siteConfig = getSiteConfig()
    return (
      siteConfig.modules.enabled.includes(moduleId) &&
      !siteConfig.modules.disabled.includes(moduleId)
    )
  }
}

export const moduleRegistry = new ModuleRegistry()
```

### Bootstrap

```typescript
// app/bootstrap.ts
import { moduleRegistry } from '@core/module-registry'
import { todayModuleConfig } from '@modules/today/config/module.config'
import { portfolioModuleConfig } from '@modules/portfolio/config/module.config'
// ... import all module configs

export async function bootstrapApp() {
  // Load site configuration
  const siteConfig = await loadSiteConfig()
  setSiteConfig(siteConfig)

  // Register core modules (always)
  moduleRegistry.register(authModuleConfig)
  moduleRegistry.register(apiModuleConfig)

  // Register feature modules (if enabled)
  const allModules = [
    todayModuleConfig,
    portfolioModuleConfig,
    alertsModuleConfig,
    journalModuleConfig,
    settingsModuleConfig,
    indicatorsModuleConfig, // expert
    optionsModuleConfig, // expert
    diagnosticsModuleConfig, // expert
    rulesModuleConfig, // expert
    mlInsightsModuleConfig, // expert
    learnModuleConfig,
    exploreModuleConfig,
  ]

  for (const config of allModules) {
    if (moduleRegistry.isEnabled(config.id)) {
      moduleRegistry.register(config)
    }
  }

  // Initialize observability
  initSentry()
  initPostHog()
  initWebVitals()
}
```

## Dependency Injection

### Service Container

```typescript
// @core/di/container.ts
export class ServiceContainer {
  private services = new Map<string, any>()

  register<T>(key: string, factory: () => T) {
    this.services.set(key, { factory, instance: null })
  }

  get<T>(key: string): T {
    const service = this.services.get(key)
    if (!service) {
      throw new Error(`Service ${key} not found`)
    }

    // Singleton pattern
    if (!service.instance) {
      service.instance = service.factory()
    }

    return service.instance
  }
}

export const container = new ServiceContainer()

// Register services
container.register('apiClient', () => new ApiClient())
container.register('logger', () => new Logger())
container.register('telemetry', () => new TelemetryService())
```

### Hook-based Injection

```typescript
// @core/di/hooks.ts
export function useService<T>(key: string): T {
  return container.get<T>(key)
}

// Usage in components
export function TodayPage() {
  const apiClient = useService<ApiClient>('apiClient')
  const logger = useService<Logger>('logger')

  // ...
}
```

## Multi-Tenant Packaging

### Build-Time Tenant Profiles

```typescript
// next.config.js
const tenantConfig = require(`./tenants/${process.env.TENANT_ID}/config.json`)

module.exports = {
  // Tree-shake disabled modules
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      // Replace disabled modules with empty exports
      ...(tenantConfig.modules.disabled.reduce((acc, moduleId) => {
        acc[`@modules/${moduleId}`] = path.resolve(__dirname, 'lib/empty-module.ts')
        return acc
      }, {})),
    }
    return config
  },

  // Custom theme
  sassOptions: {
    additionalData: `@import "${tenantConfig.theme.custom_scss}";`,
  },
}
```

### Runtime Module Loading

```typescript
// Conditionally import expert modules
const { mode } = useUserSettings()

const IndicatorPanel = mode === 'expert'
  ? lazy(() => import('@modules/indicators/components/IndicatorPanel'))
  : null

return (
  <div>
    {IndicatorPanel && (
      <Suspense fallback={<PanelSkeleton />}>
        <IndicatorPanel />
      </Suspense>
    )}
  </div>
)
```

## Quality Gates

### Definition of Done (per module)

- [ ] Module config defined with all metadata
- [ ] Public API documented in module README
- [ ] Unit tests for all exported functions (>80% coverage)
- [ ] Integration tests for critical paths
- [ ] Accessibility audit passed (axe, Lighthouse)
- [ ] i18n strings externalized
- [ ] Telemetry events instrumented
- [ ] Error boundaries in place
- [ ] Performance budget met
- [ ] Security review completed (if handling sensitive data)

### Versioning Strategy

- **Semantic versioning**: MAJOR.MINOR.PATCH
- **Breaking changes**: Increment MAJOR version
- **New features**: Increment MINOR version
- **Bug fixes**: Increment PATCH version

### Upgrade Notes

Each module version includes upgrade guide:

```markdown
## @modules/today v2.0.0

### Breaking Changes
- Removed `usePlan` hook (use `usePlanQuery` instead)
- Changed `PlanCard` props interface (added required `mode` prop)

### Migration Guide
1. Replace `usePlan()` with `usePlanQuery()`
2. Add `mode` prop to all `<PlanCard>` instances

### New Features
- Added `useExplainEntry` hook for glossary integration
- New `ExplainPopover` component
```

## Ownership

| Module | Owner Team | Slack Channel |
|--------|-----------|---------------|
| `@core/*` | Platform Team | `#team-platform` |
| `@modules/today` | Trading Team | `#team-trading` |
| `@modules/portfolio` | Trading Team | `#team-trading` |
| `@modules/alerts` | Notifications Team | `#team-notifications` |
| `@modules/indicators` | Quant Team | `#team-quant` |
| `@modules/ml-insights` | ML Team | `#team-ml` |
| `@modules/learn` | Education Team | `#team-education` |

## Rollout Plan

1. **Phase 0**: Establish module registry and core modules
2. **Phase 1**: Migrate beginner modules (`@modules/today`, `@modules/portfolio`, `@modules/alerts`)
3. **Phase 2**: Implement site configuration and feature flags
4. **Phase 3**: Add expert modules with lazy loading
5. **Phase 4**: Enable multi-tenant packaging
6. **Phase 5**: Add hot reload for configuration changes
