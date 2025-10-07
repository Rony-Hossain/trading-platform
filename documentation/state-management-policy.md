# State Management & Caching Policy
Last updated: 2025-10-03

## Overview

The frontend uses a hybrid state management approach:
- **TanStack Query (React Query)** - Server state, caching, and data fetching
- **Zustand** - Local UI state (modals, filters, user preferences)

This document defines caching policies, staleness thresholds, and state management patterns.

## Technology Stack

### TanStack Query v5
- Primary tool for server state management
- Automatic caching, refetching, and invalidation
- Background updates and optimistic updates
- DevTools for debugging

### Zustand
- Lightweight state management for UI-only state
- No boilerplate, minimal re-renders
- Persist middleware for localStorage sync
- TypeScript-first

## Caching Strategy

### Query Keys Structure

All query keys follow a hierarchical pattern:

```typescript
// Pattern: [entity, ...params]
['plan', { mode: 'beginner' }]
['positions']
['alerts', { armed: true }]
['position', symbol]
['explain', term]
['news', { symbols: ['AAPL'], lookback_hours: 24 }]
```

### Stale Time by Endpoint

Different data types have different freshness requirements:

| Endpoint | Stale Time | Cache Time | Refetch Interval | Rationale |
|----------|------------|------------|------------------|-----------|
| `/plan` | 5 min | 10 min | On focus | Plans update infrequently, 5min is acceptable |
| `/positions` | 30 sec | 5 min | 1 min | Portfolio needs near-realtime data |
| `/alerts` | 0 sec | 5 min | WebSocket | Alerts are critical, always fresh |
| `/explain/*` | 24 hours | 7 days | Never | Glossary terms rarely change |
| `/settings` | 1 hour | 24 hours | On mutation | Settings change infrequently |
| `/news` | 5 min | 15 min | On focus | News updates regularly but not instantly |
| `/journal` | 15 min | 1 hour | On focus | Journal is historical, less critical |

### Default Query Configuration

```typescript
// lib/query/config.ts
export const queryConfig = {
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      refetchOnWindowFocus: true,
      refetchOnReconnect: true,
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    },
  },
}
```

## Realtime Data Strategy

### WebSocket Streams

For critical realtime data (alerts, price updates):

1. **Initial load** - Fetch via REST for fast initial paint
2. **Stream connection** - Establish WebSocket after initial render
3. **Merge strategy** - Merge stream updates into React Query cache
4. **Fallback** - If WebSocket fails, fall back to polling

```typescript
// Example: Alerts stream integration
useEffect(() => {
  const ws = new WebSocket('wss://api/alerts/stream')

  ws.onmessage = (event) => {
    const alert = JSON.parse(event.data)

    // Merge into React Query cache
    queryClient.setQueryData(['alerts'], (old) => ({
      ...old,
      alerts: [alert, ...(old?.alerts || [])],
    }))
  }

  return () => ws.close()
}, [])
```

### Throttling & Backpressure

Apply throttling for high-frequency updates:

```typescript
// Beginner mode: 1 update/sec max
const throttledUpdate = throttle(updateChart, 1000)

// Expert mode: 4 updates/sec max
const throttledUpdate = throttle(updateChart, 250)
```

**Drop strategy**: When buffer exceeds max points, drop oldest data first.

## Optimistic Updates

For better UX, apply optimistic updates for mutations:

```typescript
const { mutate } = useMutation({
  mutationFn: updateSetting,
  onMutate: async (newSetting) => {
    // Cancel in-flight queries
    await queryClient.cancelQueries(['settings'])

    // Snapshot current value
    const previous = queryClient.getQueryData(['settings'])

    // Optimistically update
    queryClient.setQueryData(['settings'], (old) => ({
      ...old,
      ...newSetting,
    }))

    return { previous }
  },
  onError: (err, variables, context) => {
    // Rollback on error
    queryClient.setQueryData(['settings'], context.previous)
  },
  onSettled: () => {
    // Refetch to ensure consistency
    queryClient.invalidateQueries(['settings'])
  },
})
```

## Invalidation Rules

### Manual Invalidation

Invalidate queries when related data changes:

```typescript
// After executing a trade
queryClient.invalidateQueries(['positions'])
queryClient.invalidateQueries(['plan']) // Plan may update based on new position

// After changing settings
queryClient.invalidateQueries(['settings'])
queryClient.invalidateQueries(['plan']) // Plan respects user settings
```

### Automatic Invalidation

Use mutation side effects to auto-invalidate:

```typescript
const { mutate: executeTrade } = useMutation({
  mutationFn: postTrade,
  onSuccess: () => {
    queryClient.invalidateQueries(['positions'])
    queryClient.invalidateQueries(['plan'])
    queryClient.invalidateQueries(['journal']) // Auto-logged to journal
  },
})
```

## UI State Management (Zustand)

Use Zustand for ephemeral UI state:

```typescript
// stores/uiStore.ts
interface UiState {
  // Modals
  isAlertDrawerOpen: boolean
  isSettingsModalOpen: boolean

  // Filters
  journalFilters: { tags: string[]; dateRange: [Date, Date] }

  // View preferences
  chartLayout: 'single' | 'multi'

  // Actions
  openAlertDrawer: () => void
  closeAlertDrawer: () => void
  setJournalFilters: (filters: UiState['journalFilters']) => void
}

export const useUiStore = create<UiState>((set) => ({
  isAlertDrawerOpen: false,
  isSettingsModalOpen: false,
  journalFilters: { tags: [], dateRange: [new Date(), new Date()] },
  chartLayout: 'single',

  openAlertDrawer: () => set({ isAlertDrawerOpen: true }),
  closeAlertDrawer: () => set({ isAlertDrawerOpen: false }),
  setJournalFilters: (filters) => set({ journalFilters: filters }),
}))
```

### Persisted State

Persist critical UI preferences:

```typescript
import { persist } from 'zustand/middleware'

export const useUserPreferencesStore = create(
  persist<UserPreferences>(
    (set) => ({
      mode: 'beginner',
      theme: 'light',
      chartPreferences: { indicators: ['rsi', 'macd'] },

      setMode: (mode) => set({ mode }),
      setTheme: (theme) => set({ theme }),
    }),
    {
      name: 'user-preferences', // localStorage key
      version: 1,
    }
  )
)
```

## Error Handling

### Retry Logic

Configure retry based on error type:

```typescript
const { data, error } = useQuery({
  queryKey: ['plan'],
  queryFn: fetchPlan,
  retry: (failureCount, error) => {
    // Don't retry on client errors
    if (error.response?.status === 400) return false

    // Retry network errors up to 3 times
    if (error.code === 'NETWORK_ERROR') return failureCount < 3

    // Default retry
    return failureCount < 2
  },
})
```

### Error Boundaries

Wrap query errors in ErrorBoundary for graceful degradation:

```typescript
<ErrorBoundary fallback={<ErrorState />}>
  <Suspense fallback={<LoadingState />}>
    <PlanView />
  </Suspense>
</ErrorBoundary>
```

## Background Sync

Sync data in background when user is inactive:

```typescript
// Refetch plan every 5 minutes when window is focused
const { data } = useQuery({
  queryKey: ['plan'],
  queryFn: fetchPlan,
  refetchInterval: 5 * 60 * 1000,
  refetchIntervalInBackground: false, // Only when tab is active
})
```

## Offline Support

Detect offline state and adjust behavior:

```typescript
const { data } = useQuery({
  queryKey: ['plan'],
  queryFn: fetchPlan,
  networkMode: 'offlineFirst', // Serve from cache when offline
  placeholderData: keepPreviousData, // Show stale data while refetching
})
```

## Performance Considerations

### Prevent Over-fetching

Use `enabled` option to conditionally fetch:

```typescript
const { data: expertData } = useQuery({
  queryKey: ['expert-panels'],
  queryFn: fetchExpertPanels,
  enabled: mode === 'expert', // Only fetch in expert mode
})
```

### Prefetching

Prefetch likely-needed data:

```typescript
const prefetchPosition = (symbol: string) => {
  queryClient.prefetchQuery({
    queryKey: ['position', symbol],
    queryFn: () => fetchPosition(symbol),
    staleTime: 30 * 1000, // 30 seconds
  })
}

// On hover, prefetch position details
<PositionRow onMouseEnter={() => prefetchPosition(symbol)} />
```

### Deduplication

React Query automatically deduplicates simultaneous requests with the same key. No additional work needed.

## Testing Strategy

### Mock Query Client

```typescript
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  })

// In tests
const queryClient = createTestQueryClient()
render(
  <QueryClientProvider client={queryClient}>
    <App />
  </QueryClientProvider>
)
```

### Mock Data

Use MSW (Mock Service Worker) for API mocking:

```typescript
import { rest } from 'msw'
import { setupServer } from 'msw/node'

const server = setupServer(
  rest.get('/api/plan', (req, res, ctx) => {
    return res(ctx.json({ picks: [] }))
  })
)

beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())
```

## State Matrix Reference

| Surface | Loading | Empty | Error | Slow | Realtime |
| --- | --- | --- | --- | --- | --- |
| Today (PlanList) | Spinner + CopyService loading text | CopyService empty text | MUI Alert with retry | Stale banner triggered at >5m | `REALTIME_BUDGETS` throttle + Launch KPI tracking |
| Alerts Drawer | Skeleton placeholders | Friendly empty copy | Alert with fallback CTA | Rate-limit banner (`GlobalStatusBanners`) | Canary tone experiment + feedback loops |
| Diagnostics Chip | Pulse shimmer | Hidden when no diagnostics | Drift warning panel | Stability trend indicator | Driver feedback experiments |
| Admin Control Panel | Inline loading message | N/A | Planned snackbar | N/A | Flag overrides persist immediately |

## Migration Path

1. **Phase 0**: Set up React Query client and Zustand stores
2. **Phase 1**: Migrate critical paths (plan, alerts) to React Query
3. **Phase 2**: Add WebSocket integration for realtime data
4. **Phase 3**: Implement optimistic updates for mutations
5. **Phase 4**: Add prefetching and background sync
6. **Phase 5**: Fine-tune cache policies based on production metrics
