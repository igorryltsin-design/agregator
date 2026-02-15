/**
 * React Error Boundaries for graceful error handling.
 *
 * - `ErrorBoundary`  — generic, reusable error boundary with a fallback UI.
 * - `PageErrorBoundary` — wraps entire page sections (Catalogue, DocChat, OSINT).
 * - `AppErrorBoundary`  — top-level boundary wrapping the entire app.
 */

import React, { Component, type ErrorInfo, type ReactNode } from 'react'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ErrorBoundaryProps {
  children: ReactNode
  /** Custom fallback UI.  Receives `error` and a `reset` callback. */
  fallback?: (props: { error: Error; reset: () => void }) => ReactNode
  /** Callback fired when an error is caught (e.g. for logging). */
  onError?: (error: Error, info: ErrorInfo) => void
  /** Optional label shown in the default fallback (e.g. "Catalogue"). */
  label?: string
}

interface ErrorBoundaryState {
  error: Error | null
}

// ---------------------------------------------------------------------------
// Generic ErrorBoundary
// ---------------------------------------------------------------------------

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(`[ErrorBoundary${this.props.label ? `: ${this.props.label}` : ''}]`, error, info)
    this.props.onError?.(error, info)
  }

  reset = (): void => {
    this.setState({ error: null })
  }

  render(): ReactNode {
    const { error } = this.state
    if (error) {
      if (this.props.fallback) {
        return this.props.fallback({ error, reset: this.reset })
      }
      return <DefaultFallback error={error} reset={this.reset} label={this.props.label} />
    }
    return this.props.children
  }
}

// ---------------------------------------------------------------------------
// Default fallback UI
// ---------------------------------------------------------------------------

function DefaultFallback({
  error,
  reset,
  label,
}: {
  error: Error
  reset: () => void
  label?: string
}) {
  return (
    <div
      className="error-boundary-glass"
      style={{
        padding: '2rem',
        margin: '1rem',
        borderRadius: 'var(--card-radius-sm, 14px)',
        border: '1px solid var(--danger, #ef4444)',
        background: 'var(--card-bg, var(--surface, #1e1e2e))',
        color: 'var(--text, #c9d1d9)',
      }}
    >
      <h2 style={{ margin: '0 0 0.5rem', fontSize: '1.25rem', color: 'var(--danger, #ef4444)' }}>
        {label ? `Ошибка в разделе «${label}»` : 'Произошла непредвиденная ошибка'}
      </h2>
      <p style={{ margin: '0 0 1rem', opacity: 0.8 }}>
        {error.message || 'Неизвестная ошибка'}
      </p>
      <details style={{ marginBottom: '1rem', fontSize: '0.85rem', opacity: 0.6 }}>
        <summary>Подробности</summary>
        <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', marginTop: '0.5rem' }}>
          {error.stack}
        </pre>
      </details>
      <button
        onClick={reset}
        style={{
          padding: '0.5rem 1.25rem',
          borderRadius: '8px',
          border: 'none',
          background: 'var(--primary, #3b82f6)',
          color: '#fff',
          cursor: 'pointer',
          fontSize: '0.9rem',
        }}
      >
        Попробовать снова
      </button>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page-level boundary
// ---------------------------------------------------------------------------

export function PageErrorBoundary({
  children,
  label,
}: {
  children: ReactNode
  label: string
}) {
  return (
    <ErrorBoundary label={label}>
      {children}
    </ErrorBoundary>
  )
}

// ---------------------------------------------------------------------------
// App-level boundary
// ---------------------------------------------------------------------------

function AppFallback({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <div
      className="app-error-boundary-glass"
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--bg, #0d1117)',
        color: 'var(--text, #c9d1d9)',
        fontFamily:
          "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif",
      }}
    >
      <div style={{ maxWidth: '480px', padding: '2rem', textAlign: 'center' }}>
        <h1 style={{ fontSize: '1.75rem', marginBottom: '1rem' }}>
          Что-то пошло не так
        </h1>
        <p style={{ opacity: 0.7, marginBottom: '1.5rem' }}>
          Произошла критическая ошибка приложения. Попробуйте перезагрузить страницу.
        </p>
        <p style={{ fontSize: '0.85rem', opacity: 0.5, marginBottom: '1.5rem' }}>
          {error.message}
        </p>
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center' }}>
          <button
            onClick={() => window.location.reload()}
            style={{
              padding: '0.6rem 1.5rem',
              borderRadius: '8px',
              border: 'none',
              background: 'var(--primary, #3b82f6)',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.95rem',
            }}
          >
            Перезагрузить
          </button>
          <button
            onClick={reset}
            style={{
              padding: '0.6rem 1.5rem',
              borderRadius: '8px',
              border: '1px solid var(--border, #30363d)',
              background: 'transparent',
              color: 'var(--text, #c9d1d9)',
              cursor: 'pointer',
              fontSize: '0.95rem',
            }}
          >
            Попробовать снова
          </button>
        </div>
      </div>
    </div>
  )
}

export function AppErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary fallback={({ error, reset }) => <AppFallback error={error} reset={reset} />}>
      {children}
    </ErrorBoundary>
  )
}
