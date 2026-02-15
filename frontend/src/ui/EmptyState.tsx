/**
 * Reusable empty state component for lists, search results, collections, etc.
 *
 * Provides a consistent, visually appealing placeholder when content is empty.
 */

import type { ReactNode } from 'react'

interface EmptyStateProps {
  /** Primary heading. */
  title: string
  /** Optional description text below the heading. */
  description?: string
  /** Optional icon or illustration. Defaults to a document icon. */
  icon?: ReactNode
  /** Optional action button/link. */
  action?: ReactNode
}

const defaultIcon = (
  <svg
    width="64"
    height="64"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
    strokeLinecap="round"
    strokeLinejoin="round"
    style={{ opacity: 0.35 }}
  >
    <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
    <line x1="10" y1="9" x2="8" y2="9" />
  </svg>
)

export function EmptyState({ title, description, icon, action }: EmptyStateProps) {
  return (
    <div
      className="empty-state"
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '3rem 1.5rem',
        textAlign: 'center',
        minHeight: '200px',
      }}
    >
      <div style={{ marginBottom: '1rem' }}>{icon || defaultIcon}</div>
      <h3
        style={{
          margin: '0 0 0.5rem',
          fontSize: '1.15rem',
          fontWeight: 600,
          color: 'var(--text, #c9d1d9)',
        }}
      >
        {title}
      </h3>
      {description && (
        <p
          style={{
            margin: '0 0 1rem',
            fontSize: '0.9rem',
            color: 'var(--muted, #8b949e)',
            maxWidth: '360px',
          }}
        >
          {description}
        </p>
      )}
      {action && <div style={{ marginTop: '0.5rem' }}>{action}</div>}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Pre-built variants
// ---------------------------------------------------------------------------

export function EmptySearch({ query }: { query?: string }) {
  return (
    <EmptyState
      title={query ? `Ничего не найдено по запросу «${query}»` : 'Нет результатов'}
      description="Попробуйте изменить запрос или ослабить фильтры."
      icon={
        <svg
          width="64"
          height="64"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          style={{ opacity: 0.35 }}
        >
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
      }
    />
  )
}

export function EmptyCollection() {
  return (
    <EmptyState
      title="Коллекция пуста"
      description="Добавьте файлы через импорт или сканирование каталога."
    />
  )
}

export function EmptyChat() {
  return (
    <EmptyState
      title="Нет активного чата"
      description="Выберите документы и начните диалог."
      icon={
        <svg
          width="64"
          height="64"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          style={{ opacity: 0.35 }}
        >
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
      }
    />
  )
}
