import { currentLocale } from './translate'

function ensureDate(input: Date | string | number): Date | null {
  if (input instanceof Date) return isNaN(input.getTime()) ? null : input
  const date = new Date(input)
  return isNaN(date.getTime()) ? null : date
}

export function formatNumber(value: number, options?: Intl.NumberFormatOptions): string {
  if (isNaN(value)) return '--'
  return new Intl.NumberFormat(currentLocale(), options).format(value)
}

export function formatCurrency(
  value: number,
  currency: string = 'USD',
  options?: Intl.NumberFormatOptions
): string {
  if (isNaN(value)) return '--'
  return new Intl.NumberFormat(currentLocale(), {
    style: 'currency',
    currency,
    currencyDisplay: 'symbol',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
    ...options,
  }).format(value)
}

export function formatPercent(value: number, options?: Intl.NumberFormatOptions): string {
  if (isNaN(value)) return '--'
  return new Intl.NumberFormat(currentLocale(), {
    style: 'percent',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
    ...options,
  }).format(value)
}

export function formatDate(input: Date | string | number, options?: Intl.DateTimeFormatOptions): string {
  const date = ensureDate(input)
  if (!date) return '--'
  return new Intl.DateTimeFormat(currentLocale(), {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    ...options,
  }).format(date)
}

export function formatTime(input: Date | string | number, options?: Intl.DateTimeFormatOptions): string {
  const date = ensureDate(input)
  if (!date) return '--'
  return new Intl.DateTimeFormat(currentLocale(), {
    hour: '2-digit',
    minute: '2-digit',
    second: undefined,
    hour12: false,
    ...options,
  }).format(date)
}

export function formatDateTime(input: Date | string | number, options?: Intl.DateTimeFormatOptions): string {
  const date = ensureDate(input)
  if (!date) return '--'
  return new Intl.DateTimeFormat(currentLocale(), {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    ...options,
  }).format(date)
}

const RELATIVE_TIME_UNITS: Array<['year' | 'month' | 'week' | 'day' | 'hour' | 'minute' | 'second', number]> = [
  ['year', 60 * 60 * 24 * 365],
  ['month', 60 * 60 * 24 * 30],
  ['week', 60 * 60 * 24 * 7],
  ['day', 60 * 60 * 24],
  ['hour', 60 * 60],
  ['minute', 60],
  ['second', 1],
]

export function formatRelativeTime(from: Date | string | number, to: Date | string | number = Date.now()): string {
  const fromDate = ensureDate(from)
  const toDate = ensureDate(to)
  if (!fromDate || !toDate) return '--'

  const diffSeconds = Math.round((fromDate.getTime() - toDate.getTime()) / 1000)
  for (const [unit, unitSeconds] of RELATIVE_TIME_UNITS) {
    if (Math.abs(diffSeconds) >= unitSeconds || unit === 'second') {
      const value = Math.round(diffSeconds / unitSeconds)
      return new Intl.RelativeTimeFormat(currentLocale(), { numeric: 'auto' }).format(value, unit)
    }
  }
  return '--'
}
