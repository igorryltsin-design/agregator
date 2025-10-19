import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useAuth } from '../ui/Auth'
import LoadingState from '../ui/LoadingState'

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

type TaskSummary = {
  id: number
  name: string
  status?: string | null
  progress?: number | null
  payload?: string | null
  payload_json?: Record<string, unknown> | null
  created_at?: string | null
  started_at?: string | null
  finished_at?: string | null
  error?: string | null
}

type FeedbackModelEntry = {
  file_id: number
  title: string
  author?: string | null
  year?: string | null
  collection_id?: number | null
  collection_name?: string | null
  weight: number
  positive: number
  negative: number
  clicks: number
  updated_at?: string | null
}

type FeedbackStatus = {
  scheduler_enabled: boolean
  interval_hours: number
  cutoff_days: number
  thread_started: boolean
  total_weighted: number
  last_updated_at?: string | null
  active_task?: TaskSummary | null
  last_task?: TaskSummary | null
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

const formatWeight = (value?: number | null) => {
  if (value === null || value === undefined) return '0.000'
  const num = Number(value)
  if (!Number.isFinite(num)) return '0.000'
  return num.toFixed(3)
}

const formatProgress = (value?: number | null) => {
  if (value === null || value === undefined) return '0%'
  const pct = Math.max(0, Math.min(100, Math.round(Number(value || 0) * 100)))
  return `${pct}%`
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
  const [training, setTraining] = useState(false)
  const [trainMessage, setTrainMessage] = useState<string>('')
  const [feedbackLoading, setFeedbackLoading] = useState(false)
  const [feedbackError, setFeedbackError] = useState('')
  const [feedbackModel, setFeedbackModel] = useState<{ total: number; positive: FeedbackModelEntry[]; negative: FeedbackModelEntry[] } | null>(null)
  const [feedbackStatus, setFeedbackStatus] = useState<FeedbackStatus | null>(null)
  const [statusError, setStatusError] = useState('')
  const [feedbackVisible, setFeedbackVisible] = useState(false)
  const feedbackSectionRef = useRef<HTMLDivElement | null>(null)

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

const loadFeedbackModel = useCallback(async (limit = 30) => {
    if (!isAdmin) return
    setFeedbackLoading(true)
    setFeedbackError('')
    try {
      const resp = await fetch(`/api/admin/ai-search/feedback/model?limit=${Math.max(1, Math.min(limit, 200))}`)
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }
      const data = await resp.json()
      const normalize = (items: any[]): FeedbackModelEntry[] => (Array.isArray(items) ? items : []).map((item: any) => ({
        file_id: Number(item?.file_id || 0),
        title: item?.title || `Документ ${item?.file_id ?? ''}`,
        author: item?.author ?? null,
        year: item?.year ?? null,
        collection_id: item?.collection_id ?? null,
        collection_name: item?.collection_name ?? null,
        weight: Number(item?.weight || 0),
        positive: Number(item?.positive || 0),
        negative: Number(item?.negative || 0),
        clicks: Number(item?.clicks || 0),
        updated_at: item?.updated_at ?? null,
      }))
      setFeedbackModel({
        total: Number(data?.total || 0),
        positive: normalize(data?.positive),
        negative: normalize(data?.negative),
      })
    } catch (err: any) {
      console.error('[AI feedback] fetch model error', err)
      setFeedbackError(err?.message || 'Не удалось загрузить веса фидбэка')
    } finally {
      setFeedbackLoading(false)
    }
  }, [isAdmin])

  const loadFeedbackStatus = useCallback(async () => {
    if (!isAdmin) return
    setStatusError('')
    try {
      const resp = await fetch('/api/admin/ai-search/feedback/status')
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }
      const data = await resp.json()
      if (data?.ok === false) {
        throw new Error(data?.error || 'status error')
      }
      setFeedbackStatus({
        scheduler_enabled: Boolean(data?.scheduler_enabled),
        interval_hours: Number(data?.interval_hours || 0),
        cutoff_days: Number(data?.cutoff_days || 0),
        thread_started: Boolean(data?.thread_started),
        total_weighted: Number(data?.total_weighted || 0),
        last_updated_at: data?.last_updated_at || null,
        active_task: data?.active_task || null,
        last_task: data?.last_task || null,
      })
    } catch (err: any) {
      console.error('[AI feedback] status error', err)
      setStatusError(err?.message || 'Не удалось загрузить статус планировщика')
    }
  }, [isAdmin])

  const runFeedbackTraining = useCallback(async (cutoffDays = 90) => {
    if (!isAdmin) return
    setTraining(true)
    setTrainMessage('')
    setFeedbackError('')
    try {
      const resp = await fetch('/api/admin/ai-search/feedback/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cutoff_days: cutoffDays }),
      })
      const data = await resp.json().catch(() => ({}))
      if (!resp.ok || data?.ok === false) {
        throw new Error(data?.error || `HTTP ${resp.status}`)
      }
      const queued = data?.queued !== false
      const taskId = data?.task_id
      if (queued) {
        setTrainMessage(taskId ? `Задача #${taskId} поставлена в очередь.` : 'Задача обучения поставлена в очередь.')
      } else {
        setTrainMessage(taskId ? `Обучение уже выполняется (задача #${taskId}).` : 'Обучение уже выполняется.')
      }
      await loadFeedbackModel()
      await loadFeedbackStatus()
    } catch (err: any) {
      console.error('[AI feedback] train error', err)
      setTrainMessage(`Ошибка: ${err?.message || err}`)
    } finally {
      setTraining(false)
    }
  }, [isAdmin, loadFeedbackModel, loadFeedbackStatus])

  const exportMetrics = useCallback(() => {
    const payload = {
      generated_at: new Date().toISOString(),
      summary,
      metrics,
      feedback: feedbackModel,
      status: feedbackStatus,
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ai-metrics-${new Date().toISOString().slice(0, 10)}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [metrics, summary, feedbackModel, feedbackStatus])

  const refreshFeedback = useCallback(async () => {
    setFeedbackVisible(true)
    await loadFeedbackModel()
    await loadFeedbackStatus()
    window.requestAnimationFrame(() => {
      feedbackSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    })
  }, [loadFeedbackModel, loadFeedbackStatus])

  useEffect(() => {
    if (isAdmin) {
      loadMetrics()
      loadFeedbackModel()
      loadFeedbackStatus()
    }
  }, [isAdmin, loadMetrics, loadFeedbackModel, loadFeedbackStatus])

  const summaryCards = useMemo(() => {
    return METRIC_LABELS.filter(entry => summary[entry.key as string] !== undefined)
      .map(entry => ({
        label: entry.label,
        value: summary[entry.key as string],
        key: entry.key,
      }))
  }, [summary])

  const schedulerBadge = useMemo(() => {
    if (!feedbackStatus) return null
    const running = Boolean(feedbackStatus.active_task)
    const className = running
      ? 'badge bg-warning text-dark'
      : feedbackStatus.scheduler_enabled
        ? 'badge bg-success'
        : 'badge bg-secondary'
    const label = running
      ? 'Обучение выполняется'
      : feedbackStatus.scheduler_enabled
        ? 'Планировщик включён'
        : 'Планировщик выключен'
    return <span className={className}>{label}</span>
  }, [feedbackStatus])

  const lastWeightsText = useMemo(() => {
    if (!feedbackStatus?.last_updated_at) return null
    return formatDateTime(feedbackStatus.last_updated_at)
  }, [feedbackStatus])

  const lastTaskStats = useMemo(() => {
    if (!feedbackStatus?.last_task?.payload_json) return null
    const payload = feedbackStatus.last_task.payload_json as Record<string, unknown>
    if (!payload || typeof payload !== 'object') return null
    const stats = (payload as any).stats
    return stats && typeof stats === 'object' ? stats as Record<string, unknown> : null
  }, [feedbackStatus])

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
    <div className="card p-3" aria-busy={loading}>
      <div className="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-3">
        <div>
          <div className="fw-semibold">Метрики AI-поиска</div>
          <div className="text-muted" style={{ fontSize: 13 }}>Последние измерения длительностей по этапам конвейера.</div>
          {feedbackStatus && lastWeightsText && (
            <div className="text-muted" style={{ fontSize: 12 }}>Последнее обновление весов: {lastWeightsText}</div>
          )}
        </div>
        <div className="d-flex align-items-center gap-2">
          {schedulerBadge}
          {updatedAt ? <span className="text-muted" style={{ fontSize: 12 }}>Обновлено {new Date(updatedAt).toLocaleTimeString()}</span> : null}
          <button className="btn btn-sm btn-outline-secondary" onClick={() => loadMetrics()} disabled={loading}>Обновить</button>
          <button className="btn btn-sm btn-outline-primary" onClick={() => runFeedbackTraining()} disabled={training}>{training ? 'Обучение…' : 'Обучить фидбэк'}</button>
          <button className="btn btn-sm btn-outline-secondary" onClick={refreshFeedback} disabled={feedbackLoading}>Веса фидбэка</button>
          <button className="btn btn-sm btn-outline-secondary" onClick={exportMetrics}>Экспорт JSON</button>
          <button className="btn btn-sm btn-outline-danger" onClick={clearMetrics} disabled={clearing || loading}>Очистить</button>
        </div>
      </div>
      {error && <div className="alert alert-danger" role="alert">{error}</div>}
      {statusError && <div className="alert alert-warning" role="alert">{statusError}</div>}
      {trainMessage && (
        <div className={`alert alert-${trainMessage.toLowerCase().includes('ошибка') ? 'danger' : 'info'}`} role="alert">{trainMessage}</div>
      )}
      {feedbackError && <div className="alert alert-danger" role="alert">{feedbackError}</div>}
      {feedbackStatus?.active_task && (
        <div className="alert alert-info" role="alert">
          Обучение запущено (задача #{feedbackStatus.active_task.id}, статус {feedbackStatus.active_task.status || '—'}, прогресс {formatProgress(feedbackStatus.active_task.progress)}).
        </div>
      )}
      <div className="d-flex flex-wrap gap-3 mb-3">
        {loading && summaryCards.length === 0 && (
          <LoadingState
            title="Загружаем сводку"
            description="Собираем последние измерения по этапам поиска"
            lines={4}
            variant="inline"
            className="w-100"
          />
        )}
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
      {feedbackVisible && (
        <div ref={feedbackSectionRef} className="mb-3">
          {feedbackLoading && <div className="text-muted mb-3">Загрузка весов фидбэка…</div>}
          {feedbackModel && feedbackModel.total === 0 && !feedbackLoading && (
            <div className="text-muted mb-3">Фидбэк ещё не собран.</div>
          )}
          {feedbackStatus && feedbackStatus.total_weighted > 0 && (
            <div className="text-muted mb-3" style={{ fontSize: 12 }}>
              Всего документов с весами: {feedbackStatus.total_weighted}.
              {lastTaskStats && (
                <>
                  {' '}Последний прогон: файлов {((lastTaskStats as any).files) ?? '—'},
                  обновлено {((lastTaskStats as any).updated) ?? '—'}, завершено {feedbackStatus.last_task?.finished_at ? formatDateTime(feedbackStatus.last_task.finished_at) : '—'}.
                </>
              )}
            </div>
          )}
          {feedbackModel && (feedbackModel.positive.length > 0 || feedbackModel.negative.length > 0) && (
            <div className="row g-3 mb-3">
              <div className="col-12 col-xl-6">
                <div className="card p-3 h-100">
                  <div className="fw-semibold mb-2">Лучшие документы по фидбэку</div>
                  <ol className="mb-0 ps-3">
                    {feedbackModel.positive.map(entry => (
                      <li key={`pos-${entry.file_id}`} className="mb-2">
                        <div className="d-flex justify-content-between align-items-center" style={{ fontSize: 14 }}>
                          <span>{entry.title}</span>
                          <span className="text-success">{formatWeight(entry.weight)}</span>
                        </div>
                        <div className="text-secondary" style={{ fontSize: 12 }}>
                          👍 {entry.positive} · 👎 {entry.negative} · кликов {entry.clicks}
                        </div>
                        {entry.updated_at && (
                          <div className="text-muted" style={{ fontSize: 11 }}>Обновлено {formatDateTime(entry.updated_at)}</div>
                        )}
                      </li>
                    ))}
                    {feedbackModel.positive.length === 0 && <li className="text-muted">Нет положительных оценок</li>}
                  </ol>
                </div>
              </div>
              <div className="col-12 col-xl-6">
                <div className="card p-3 h-100">
                  <div className="fw-semibold mb-2">Документы с отрицательным весом</div>
                  <ol className="mb-0 ps-3">
                    {feedbackModel.negative.map(entry => (
                      <li key={`neg-${entry.file_id}`} className="mb-2">
                        <div className="d-flex justify-content-between align-items-center" style={{ fontSize: 14 }}>
                          <span>{entry.title}</span>
                          <span className="text-danger">{formatWeight(entry.weight)}</span>
                        </div>
                        <div className="text-secondary" style={{ fontSize: 12 }}>
                          👍 {entry.positive} · 👎 {entry.negative} · кликов {entry.clicks}
                        </div>
                        {entry.updated_at && (
                          <div className="text-muted" style={{ fontSize: 11 }}>Обновлено {formatDateTime(entry.updated_at)}</div>
                        )}
                      </li>
                    ))}
                    {feedbackModel.negative.length === 0 && <li className="text-muted">Нет отрицательных оценок</li>}
                  </ol>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
      <div className="table-responsive">
        <table className="table table-sm align-middle">
          <thead>
            <tr>
              <th>Хэш</th>
              <th>Запрос</th>
              <th>Ответ</th>
              <th>Отсеяно</th>
              <th>Всего</th>
              <th>Ключевые слова</th>
              <th>Кандидаты</th>
              <th>Глубокий этап</th>
              <th>Ответ LLM</th>
              <th>Пользователь</th>
              <th>Время</th>
            </tr>
          </thead>
          <tbody aria-busy={loading}>
            {loading && (
              <tr>
                <td colSpan={11}>
                  <LoadingState variant="inline" lines={5} />
                </td>
              </tr>
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
