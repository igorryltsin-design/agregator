import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useAuth } from '../ui/Auth'

type AiMetricRow = {
  id: number
  query_hash: string
  user_id?: number | null
  user_username?: string | null
  user_full_name?: string | null
  total_ms?: number | null
  keywords_ms?: number | null
  candidate_ms?: number | null
  deep_ms?: number | null
  llm_answer_ms?: number | null
  llm_snippet_ms?: number | null
  created_at?: string | null
  meta?: Record<string, unknown> | string | null
}

type SummaryResponse = Record<string, number>

const METRIC_LABELS: Array<{ key: keyof AiMetricRow; label: string }> = [
  { key: 'total_ms', label: 'Всего' },
  { key: 'keywords_ms', label: 'Ключевые слова' },
  { key: 'candidate_ms', label: 'Кандидаты' },
  { key: 'deep_ms', label: 'Глубокий поиск' },
  { key: 'llm_answer_ms', label: 'Ответ LLM' },
]

const formatMs = (value?: number | null) => {
  if (value === null || value === undefined) return '—'
  if (value >= 1000) return `${(value / 1000).toFixed(1)} с`
  return `${value} мс`
}

const formatDateTime = (value?: string | null) => {
  if (!value) return '—'
  try {
    const date = new Date(value)
    return `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`
  } catch {
    return value
  }
}

export default function AdminAiMetricsPage() {
  const { user } = useAuth()
  const isAdmin = user?.role === 'admin'
  const [metrics, setMetrics] = useState<AiMetricRow[]>([])
  const [summary, setSummary] = useState<SummaryResponse>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [updatedAt, setUpdatedAt] = useState<number>(0)
  const [clearing, setClearing] = useState(false)

  const loadMetrics = useCallback(async (limit = 100) => {
    if (!isAdmin) return
    setLoading(true)
    setError('')
    try {
      const resp = await fetch(`/api/admin/ai-search/metrics?limit=${Math.max(1, Math.min(limit, 500))}`)
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }
      const data = await resp.json()
      const items = Array.isArray(data?.items) ? data.items.map((row: any) => {
        if (row.meta && typeof row.meta === 'string') {
          try { row.meta = JSON.parse(row.meta) }
          catch { row.meta = { raw: row.meta } }
        }
        return row
      }) : []
      setMetrics(items)
      setSummary(data?.summary || {})
      setUpdatedAt(Date.now())
    } catch (err: any) {
      console.error('[AI metrics] fetch error', err)
      setError(err?.message || 'Не удалось загрузить метрики')
    } finally {
      setLoading(false)
    }
  }, [isAdmin])

  useEffect(() => {
    if (isAdmin) {
      loadMetrics()
    }
  }, [isAdmin, loadMetrics])

  const summaryCards = useMemo(() => {
    return METRIC_LABELS.filter(entry => summary[entry.key as string] !== undefined)
      .map(entry => ({
        label: entry.label,
        value: summary[entry.key as string],
        key: entry.key,
      }))
  }, [summary])

  const clearMetrics = useCallback(async () => {
    setClearing(true)
    setError('')
    try {
      const resp = await fetch('/api/admin/ai-search/metrics', { method: 'DELETE' })
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }
      await loadMetrics()
    } catch (err: any) {
      console.error('[AI metrics] clear error', err)
      setError(err?.message || 'Не удалось очистить метрики')
    } finally {
      setClearing(false)
    }
  }, [loadMetrics])

  if (!isAdmin) {
    return (
      <div className="alert alert-warning" role="alert">
        Доступ к метрикам доступен только администраторам.
      </div>
    )
  }

  return (
    <div className="card p-3">
      <div className="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-3">
        <div>
          <div className="fw-semibold">Метрики AI-поиска</div>
          <div className="text-muted" style={{ fontSize: 13 }}>Последние измерения длительностей по этапам конвейера.</div>
        </div>
        <div className="d-flex align-items-center gap-2">
          {updatedAt ? <span className="text-muted" style={{ fontSize: 12 }}>Обновлено {new Date(updatedAt).toLocaleTimeString()}</span> : null}
          <button className="btn btn-sm btn-outline-secondary" onClick={() => loadMetrics()} disabled={loading}>Обновить</button>
          <button className="btn btn-sm btn-outline-danger" onClick={clearMetrics} disabled={clearing || loading}>Очистить</button>
        </div>
      </div>
      {error && <div className="alert alert-danger" role="alert">{error}</div>}
      <div className="d-flex flex-wrap gap-3 mb-3">
        {loading && summaryCards.length === 0 && <div className="text-muted">Загрузка…</div>}
        {summaryCards.map(card => (
          <div key={card.key as string} className="border rounded-3 px-3 py-2" style={{ minWidth: 150 }}>
            <div className="text-muted" style={{ fontSize: 12 }}>{card.label}</div>
            <div className="fw-semibold" style={{ fontSize: 18 }}>{formatMs(card.value)}</div>
          </div>
        ))}
            {!loading && summaryCards.length === 0 && !error && (
              <div className="text-muted">Нет агрегированных данных</div>
            )}
      </div>
      <div className="table-responsive">
        <table className="table table-sm align-middle">
          <thead>
            <tr>
              <th>Hash</th>
              <th>Запрос</th>
              <th>Ответ</th>
              <th>Исключено</th>
              <th>Всего</th>
              <th>Ключи</th>
              <th>Кандидаты</th>
              <th>Глубокий</th>
              <th>Ответ</th>
              <th>Пользователь</th>
              <th>Время</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr><td colSpan={11} className="text-muted">Загрузка…</td></tr>
            )}
            {!loading && metrics.length === 0 && !error && (
              <tr><td colSpan={11} className="text-muted">Пока нет сохранённых метрик</td></tr>
            )}
            {!loading && metrics.map(row => {
              const hash = row.query_hash || ''
              const meta = (row.meta && typeof row.meta === 'object') ? row.meta as Record<string, unknown> : {}
              const queryPreview = (meta?.query_preview as string) || '—'
              const answerPreview = (meta?.answer_preview as string) || '—'
              const filteredKw = Array.isArray(meta?.filtered_keywords) ? (meta.filtered_keywords as string[]).join(', ') : '—'
              return (
                <tr key={row.id}>
                  <td><code>{hash ? hash.slice(0, 12) : '—'}</code></td>
                  <td style={{ maxWidth: 220 }}>{queryPreview}</td>
                  <td style={{ maxWidth: 220 }}>{answerPreview}</td>
                  <td style={{ maxWidth: 160 }}>{filteredKw}</td>
                  <td>{formatMs(row.total_ms)}</td>
                  <td>{formatMs(row.keywords_ms)}</td>
                  <td>{formatMs(row.candidate_ms)}</td>
                  <td>{formatMs(row.deep_ms)}</td>
                  <td>{formatMs(row.llm_answer_ms)}</td>
                  <td>{renderUser(row)}</td>
                  <td>{formatDateTime(row.created_at)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
  const renderUser = (row: AiMetricRow) => {
    if (row.user_full_name) return row.user_full_name
    if (row.user_username) return `@${row.user_username}`
    if (row.user_id != null) return `ID ${row.user_id}`
    return '—'
  }
