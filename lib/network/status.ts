export interface RateLimitRecord {
  source: string
  retryAt?: Date
  limit?: number
  remaining?: number
  message?: string
}

export interface NetworkStatusState {
  online: boolean
  rateLimits: RateLimitRecord[]
  lastUpdated: number
}

type Listener = () => void

let state: NetworkStatusState = {
  online: typeof navigator === 'undefined' ? true : navigator.onLine,
  rateLimits: [],
  lastUpdated: Date.now(),
}

const listeners = new Set<Listener>()
let listenersInitialized = false

function emit() {
  state = { ...state, rateLimits: [...state.rateLimits], lastUpdated: Date.now() }
  listeners.forEach((listener) => listener())
}

export function subscribeNetworkStatus(listener: Listener): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

export function getNetworkStatusSnapshot(): NetworkStatusState {
  return state
}

export function ensureNetworkListeners() {
  if (listenersInitialized || typeof window === 'undefined') return
  listenersInitialized = true

  const updateFromBrowser = () => updateOnlineStatus(navigator.onLine)
  window.addEventListener('online', updateFromBrowser)
  window.addEventListener('offline', updateFromBrowser)
  updateFromBrowser()
}

export function updateOnlineStatus(online: boolean) {
  if (state.online === online) return
  state = {
    ...state,
    online,
  }
  emit()
}

export function notifyRateLimitHit(record: RateLimitRecord) {
  const existingIndex = state.rateLimits.findIndex((entry) => entry.source === record.source)
  const normalized: RateLimitRecord = {
    ...record,
    retryAt: record.retryAt,
  }

  if (existingIndex >= 0) {
    state.rateLimits[existingIndex] = {
      ...state.rateLimits[existingIndex],
      ...normalized,
    }
  } else {
    state.rateLimits.push(normalized)
  }

  emit()
}

export function clearRateLimit(source: string) {
  const nextLimits = state.rateLimits.filter((entry) => entry.source !== source)
  if (nextLimits.length === state.rateLimits.length) return
  state = {
    ...state,
    rateLimits: nextLimits,
  }
  emit()
}
