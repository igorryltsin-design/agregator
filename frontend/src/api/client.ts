/**
 * Typed HTTP client for the Agregator REST API.
 *
 * Provides a centralized `apiFetch` wrapper around `fetch()` that handles:
 * - JSON serialization / deserialization
 * - 401 → redirect to login
 * - Error normalization
 * - Abort controller propagation
 * - Retry logic for transient failures
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  status: number
  body: unknown

  constructor(message: string, status: number, body?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

export interface ApiFetchOptions extends Omit<RequestInit, 'body'> {
  /** JSON body — will be stringified automatically. */
  json?: unknown
  /** Raw body (for FormData uploads etc.). */
  body?: BodyInit
  /** If true, returns the raw `Response` instead of parsed JSON. */
  raw?: boolean
  /** Retry transient errors (5xx) up to this many times. Default: 0. */
  retries?: number
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

const TRANSIENT_CODES = new Set([502, 503, 504])

async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function isTransient(status: number): boolean {
  return TRANSIENT_CODES.has(status)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Core fetch wrapper. All API calls go through here.
 *
 * @example
 * const data = await apiFetch<{ files: FileItem[] }>('/api/files')
 * const result = await apiFetch<{ ok: boolean }>('/api/ai-search', {
 *   method: 'POST',
 *   json: { query: 'тест' },
 * })
 */
export async function apiFetch<T = unknown>(
  url: string,
  opts: ApiFetchOptions = {},
): Promise<T> {
  const { json, raw, retries = 0, ...init } = opts

  const headers = new Headers(init.headers)

  if (json !== undefined && !init.body) {
    headers.set('Content-Type', 'application/json')
    init.body = JSON.stringify(json)
  }

  // Always expect JSON unless explicitly overridden
  if (!headers.has('Accept')) {
    headers.set('Accept', 'application/json')
  }

  init.headers = headers

  let lastError: Error | null = null

  for (let attempt = 0; attempt <= retries; attempt++) {
    if (attempt > 0) {
      await sleep(Math.min(1000 * 2 ** (attempt - 1), 8000))
    }

    let response: Response

    try {
      response = await fetch(url, init)
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err))
      if (attempt < retries) continue
      throw lastError
    }

    // Redirect to login on 401
    if (response.status === 401) {
      const path = window.location.pathname
      if (!path.includes('/login')) {
        window.location.href = '/app/login'
      }
      throw new ApiError('Не авторизовано', 401)
    }

    // Retry transient errors
    if (isTransient(response.status) && attempt < retries) {
      continue
    }

    if (raw) {
      return response as unknown as T
    }

    let body: unknown
    const contentType = response.headers.get('Content-Type') || ''
    if (contentType.includes('json')) {
      body = await response.json()
    } else {
      body = await response.text()
    }

    if (!response.ok) {
      const message =
        typeof body === 'object' && body !== null && 'error' in body
          ? String((body as Record<string, unknown>).error)
          : `HTTP ${response.status}`
      throw new ApiError(message, response.status, body)
    }

    return body as T
  }

  throw lastError ?? new Error('Request failed')
}
