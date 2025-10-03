# Frontend Observability Stack
Last updated: 2025-10-03

## Overview

Complete observability solution for monitoring frontend health, performance, user behavior, and errors.

## Stack Components

### 1. Error Tracking & Crash Reporting

**Recommended**: Sentry

**Why Sentry**:
- Real-time error tracking with full stack traces
- Release tracking and deployment notifications
- Source map support for unminified errors
- Performance monitoring integration
- User feedback collection
- Breadcrumbs for error context

**Configuration**:
```typescript
// lib/sentry/init.ts
import * as Sentry from '@sentry/nextjs'

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  environment: process.env.NODE_ENV,

  // Performance monitoring
  tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,

  // Session replay (privacy-safe)
  replaysSessionSampleRate: 0.1,
  replaysOnErrorSampleRate: 1.0,

  // Privacy controls
  beforeSend(event, hint) {
    // Redact PII
    if (event.request) {
      delete event.request.cookies
      delete event.request.headers
    }

    // Filter sensitive data from breadcrumbs
    if (event.breadcrumbs) {
      event.breadcrumbs = event.breadcrumbs.map(breadcrumb => ({
        ...breadcrumb,
        data: redactSensitiveData(breadcrumb.data)
      }))
    }

    return event
  },

  // Ignore known errors
  ignoreErrors: [
    'ResizeObserver loop limit exceeded',
    'Non-Error promise rejection captured',
  ],
})
```

**Error Boundaries**:
```tsx
import { ErrorBoundary } from '@sentry/nextjs'

<ErrorBoundary fallback={<ErrorFallback />} showDialog>
  <App />
</ErrorBoundary>
```

### 2. Real User Monitoring (RUM)

**Recommended**: Vercel Analytics + Web Vitals

**Why This Stack**:
- Native Next.js integration
- Zero config for Core Web Vitals
- Edge network performance insights
- Privacy-compliant (no cookies)

**Implementation**:
```typescript
// app/layout.tsx
import { Analytics } from '@vercel/analytics/react'
import { SpeedInsights } from '@vercel/speed-insights/next'

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        {children}
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  )
}
```

**Web Vitals Tracking**:
```typescript
// lib/vitals.ts
import { onCLS, onFID, onFCP, onLCP, onTTFB, onINP } from 'web-vitals'
import { trackPerformance } from './telemetry/taxonomy'

export function initWebVitals() {
  onLCP((metric) => trackPerformance('lcp', metric.value, window.location.pathname))
  onFID((metric) => trackPerformance('fid', metric.value, window.location.pathname))
  onCLS((metric) => trackPerformance('cls', metric.value, window.location.pathname))
  onFCP((metric) => trackPerformance('fcp', metric.value, window.location.pathname))
  onTTFB((metric) => trackPerformance('ttfb', metric.value, window.location.pathname))
  onINP((metric) => trackPerformance('inp', metric.value, window.location.pathname))
}
```

### 3. Product Analytics

**Recommended**: PostHog (self-hosted or cloud)

**Why PostHog**:
- Product analytics + session replay in one tool
- Feature flags integration
- Privacy-first (GDPR/CCPA compliant)
- Self-hosting option for sensitive data
- Funnel and retention analysis

**Configuration**:
```typescript
// lib/posthog/client.ts
import posthog from 'posthog-js'

export function initPostHog() {
  if (typeof window !== 'undefined') {
    posthog.init(process.env.NEXT_PUBLIC_POSTHOG_KEY!, {
      api_host: process.env.NEXT_PUBLIC_POSTHOG_HOST || 'https://app.posthog.com',

      // Privacy settings
      persistence: 'localStorage',
      autocapture: false, // Manual event tracking only
      capture_pageview: false, // Manual pageview tracking

      // Session replay (opt-in only)
      disable_session_recording: !userConsent.sessionRecording,

      // Mask sensitive elements
      mask_all_text: false,
      mask_all_element_attributes: false,
      session_recording: {
        maskTextSelector: '[data-sensitive]',
        maskInputSelector: 'input[type="password"], input[type="email"]',
      },
    })
  }
}
```

**Event Tracking Integration**:
```typescript
// lib/telemetry/posthog-adapter.ts
import posthog from 'posthog-js'
import { trackEvent, TelemetryEvent } from './taxonomy'

// Adapter to send telemetry events to PostHog
export function trackToPostHog(event: TelemetryEvent) {
  posthog.capture(`${event.category}_${event.action}`, {
    ...event,
    timestamp: new Date().toISOString(),
  })
}

// Override trackEvent to send to PostHog
const originalTrackEvent = trackEvent
export { originalTrackEvent as trackEvent }

// Wrap to send to multiple destinations
export function track(event: TelemetryEvent) {
  originalTrackEvent(event)
  trackToPostHog(event)
}
```

### 4. Application Performance Monitoring (APM)

**Recommended**: Built-in Next.js + Sentry Performance

**Route-level Performance**:
```typescript
// middleware.ts
import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

export function middleware(request: NextRequest) {
  const startTime = Date.now()
  const response = NextResponse.next()

  // Track server-side timing
  response.headers.set('X-Response-Time', `${Date.now() - startTime}ms`)

  return response
}
```

**Client-side Performance Marks**:
```typescript
// utils/performance.ts
export function markPerformance(name: string) {
  if (typeof window !== 'undefined' && window.performance) {
    performance.mark(name)
  }
}

export function measurePerformance(name: string, startMark: string, endMark: string) {
  if (typeof window !== 'undefined' && window.performance) {
    performance.measure(name, startMark, endMark)

    const measure = performance.getEntriesByName(name)[0]
    if (measure) {
      trackPerformance(name, measure.duration, window.location.pathname)
    }
  }
}

// Usage
markPerformance('plan-fetch-start')
// ... fetch plan data
markPerformance('plan-fetch-end')
measurePerformance('plan-fetch-duration', 'plan-fetch-start', 'plan-fetch-end')
```

### 5. Log Aggregation

**Recommended**: Custom Client Logger → Server API → Centralized Logging

**Client Logger**:
```typescript
// lib/logger/client.ts
type LogLevel = 'debug' | 'info' | 'warn' | 'error'

interface LogEntry {
  level: LogLevel
  message: string
  context?: Record<string, unknown>
  timestamp: string
  sessionId: string
  userId?: string
  route: string
}

class ClientLogger {
  private buffer: LogEntry[] = []
  private readonly BUFFER_SIZE = 50
  private readonly FLUSH_INTERVAL = 30000 // 30 seconds

  constructor() {
    if (typeof window !== 'undefined') {
      setInterval(() => this.flush(), this.FLUSH_INTERVAL)
      window.addEventListener('beforeunload', () => this.flush())
    }
  }

  private log(level: LogLevel, message: string, context?: Record<string, unknown>) {
    const entry: LogEntry = {
      level,
      message,
      context: this.sanitizeContext(context),
      timestamp: new Date().toISOString(),
      sessionId: this.getSessionId(),
      userId: this.getUserId(),
      route: window.location.pathname,
    }

    this.buffer.push(entry)

    // Console output in development
    if (process.env.NODE_ENV === 'development') {
      console[level](message, context)
    }

    // Flush if buffer is full or it's an error
    if (this.buffer.length >= this.BUFFER_SIZE || level === 'error') {
      this.flush()
    }
  }

  private async flush() {
    if (this.buffer.length === 0) return

    const logs = [...this.buffer]
    this.buffer = []

    try {
      await fetch('/api/logs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ logs }),
        keepalive: true, // Ensure logs are sent even during page unload
      })
    } catch (error) {
      // Restore logs to buffer if send fails
      this.buffer = [...logs, ...this.buffer]
    }
  }

  private sanitizeContext(context?: Record<string, unknown>): Record<string, unknown> | undefined {
    if (!context) return undefined

    // Remove sensitive fields
    const sanitized = { ...context }
    const sensitiveKeys = ['password', 'token', 'apiKey', 'secret']

    for (const key of sensitiveKeys) {
      if (key in sanitized) {
        sanitized[key] = '[REDACTED]'
      }
    }

    return sanitized
  }

  debug(message: string, context?: Record<string, unknown>) {
    this.log('debug', message, context)
  }

  info(message: string, context?: Record<string, unknown>) {
    this.log('info', message, context)
  }

  warn(message: string, context?: Record<string, unknown>) {
    this.log('warn', message, context)
  }

  error(message: string, context?: Record<string, unknown>) {
    this.log('error', message, context)
  }
}

export const logger = new ClientLogger()
```

**Server-side Log Handler**:
```typescript
// app/api/logs/route.ts
import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  const { logs } = await request.json()

  // Forward to centralized logging (e.g., Datadog, CloudWatch, etc.)
  await forwardToLoggingService(logs)

  return NextResponse.json({ success: true })
}
```

## Privacy & Compliance

### Data Sampling

```typescript
// Sample rates by environment
const SAMPLE_RATES = {
  development: {
    errors: 1.0,
    performance: 1.0,
    analytics: 1.0,
    logs: 1.0,
  },
  staging: {
    errors: 1.0,
    performance: 0.5,
    analytics: 1.0,
    logs: 0.5,
  },
  production: {
    errors: 1.0, // Always capture errors
    performance: 0.1, // 10% of sessions
    analytics: 1.0, // All events
    logs: 0.05, // 5% of logs (except errors)
  },
}

function shouldSample(type: keyof typeof SAMPLE_RATES.production): boolean {
  const env = process.env.NODE_ENV as keyof typeof SAMPLE_RATES
  const rate = SAMPLE_RATES[env]?.[type] ?? 0
  return Math.random() < rate
}
```

### PII Redaction

```typescript
// utils/privacy.ts
export function redactPII(data: any): any {
  const piiPatterns = {
    email: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g,
    ssn: /\b\d{3}-\d{2}-\d{4}\b/g,
    creditCard: /\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b/g,
    phone: /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/g,
  }

  let sanitized = JSON.stringify(data)

  for (const [type, pattern] of Object.entries(piiPatterns)) {
    sanitized = sanitized.replace(pattern, `[REDACTED_${type.toUpperCase()}]`)
  }

  return JSON.parse(sanitized)
}
```

### User Consent

```typescript
// lib/consent/manager.ts
interface ConsentPreferences {
  necessary: boolean // Always true
  analytics: boolean
  performance: boolean
  sessionRecording: boolean
}

export class ConsentManager {
  getPreferences(): ConsentPreferences {
    const stored = localStorage.getItem('consent-preferences')
    return stored ? JSON.parse(stored) : {
      necessary: true,
      analytics: false,
      performance: false,
      sessionRecording: false,
    }
  }

  updatePreferences(preferences: ConsentPreferences) {
    localStorage.setItem('consent-preferences', JSON.stringify(preferences))
    this.applyPreferences(preferences)
  }

  private applyPreferences(prefs: ConsentPreferences) {
    // Enable/disable PostHog
    if (!prefs.analytics) {
      posthog.opt_out_capturing()
    } else {
      posthog.opt_in_capturing()
    }

    // Enable/disable session recording
    if (!prefs.sessionRecording) {
      posthog.set_config({ disable_session_recording: true })
    }

    // Enable/disable performance monitoring
    if (!prefs.performance) {
      // Disable Vercel Analytics
    }
  }
}
```

## Alerting & Notifications

### Error Rate Alerts

Configure alerts in Sentry:
- **P0**: Error rate > 5% (immediate Slack/PagerDuty)
- **P1**: Error rate > 1% (30min delay)
- **P2**: New error type (daily digest)

### Performance Degradation

Configure alerts in Vercel/PostHog:
- **P1**: LCP > 4s on /today or /portfolio (critical paths)
- **P2**: INP > 500ms
- **P3**: Bundle size increased by > 20%

### Custom Metrics

```typescript
// Send custom metrics
import { trackEvent, TelemetryCategory } from './telemetry/taxonomy'

// Track business metrics
trackEvent({
  category: TelemetryCategory.PLAN,
  action: 'plan_viewed',
  mode: 'beginner',
  picks_count: 5,
  daily_cap_status: 'ok',
  load_time_ms: 1200,
  has_degraded_fields: false,
})

// Create alert rules in PostHog/Sentry based on event properties
```

## Dashboards

### Core Metrics Dashboard

**Metrics to track**:
1. **Availability**: Uptime percentage (target: 99.9%)
2. **Performance**: P95 LCP, INP, TTFB per route
3. **Error Rate**: Errors per 1000 sessions
4. **User Flow**: Conversion rate through critical paths
5. **Bundle Size**: JS/CSS size over time

### User Behavior Dashboard

**Metrics**:
1. Mode distribution (beginner vs expert)
2. Feature adoption (alerts enabled, journal usage)
3. Session duration and depth
4. Alert CTR and helpfulness ratings
5. Median P&L per alert followed

### Real-time Monitoring

**Live metrics**:
1. Active users by route
2. Error count (last 5 minutes)
3. Slow requests (> 3s)
4. WebSocket connection status
5. Rate limit hits

## Testing Observability

```typescript
// test/observability.test.ts
describe('Observability', () => {
  it('should track page views', () => {
    const trackSpy = jest.spyOn(posthog, 'capture')

    render(<TodayPage />)

    expect(trackSpy).toHaveBeenCalledWith('page_viewed', expect.objectContaining({
      page: '/today',
    }))
  })

  it('should log errors to Sentry', () => {
    const sentrySpy = jest.spyOn(Sentry, 'captureException')

    // Trigger error
    fireEvent.click(screen.getByRole('button', { name: /error/ }))

    expect(sentrySpy).toHaveBeenCalled()
  })

  it('should redact PII from logs', () => {
    const data = { email: 'user@example.com', message: 'Hello' }
    const redacted = redactPII(data)

    expect(redacted.email).toBe('[REDACTED_EMAIL]')
    expect(redacted.message).toBe('Hello')
  })
})
```

## Rollout Plan

1. **Phase 0**: Set up Sentry for error tracking
2. **Phase 1**: Add Vercel Analytics for Web Vitals
3. **Phase 2**: Integrate PostHog for product analytics
4. **Phase 3**: Deploy client logger with server-side forwarding
5. **Phase 4**: Create dashboards and configure alerts
6. **Phase 5**: Fine-tune sampling rates based on costs and signal quality
