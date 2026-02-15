import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../ui/Auth'
import { useToasts } from '../ui/Toasts'
import { taskStatusRu } from '../utils/locale'

type ServiceInfo = {
  enabled?: boolean
  running?: boolean
  interval_hours?: number | null
  last_trigger_at?: string | null
  next_run_in_seconds?: number | null
  total_weighted?: number | null
}

type CacheInfo = {
  enabled?: boolean
  items?: number
  max_items?: number
  ttl_seconds?: number
}

type RuntimeLlmEndpoint = {
  id?: number | null
  name?: string | null
  model?: string | null
  provider?: string | null
  weight?: number | null
  purposes?: string[]
  tokenizer_hint?: string | null
}

type FeedbackSchedulerStatus = {
  ok: boolean
  scheduler_enabled: boolean
  interval_hours?: number | null
  cutoff_days?: number | null
  thread_started?: boolean
  total_weighted?: number | null
  last_updated_at?: string | null
  active_task?: Record<string, any> | null
  last_task?: Record<string, any> | null
  last_trigger_at?: string | null
  next_run_in_seconds?: number | null
  updated?: boolean
  manual_task?: { task_id: number; queued: boolean } | null
}

type OsintSourceProgress = {
  status?: string
  label?: string
  type?: string
  updated_at?: string | null
  created_at?: string | null
  error?: string | null
  fallback?: boolean
  fallback_url?: string | null
  final_url?: string | null
  links_collected?: number | null
  results_on_page?: number | null
  results_parsed?: number | null
  pages_estimated?: number | null
  pages_processed?: number | null
  retry_available?: boolean
}

type OsintScheduleInfo = {
  id?: number | null
  label?: string | null
  active?: boolean
  running?: boolean
  interval_minutes?: number | null
  next_run_at?: string | null
  last_run_at?: string | null
  notify?: boolean
  notify_channel?: string | null
}

type OsintJobSummary = {
  id?: number | null
  query?: string | null
  status?: string | null
  error?: string | null
  created_at?: string | null
  started_at?: string | null
  completed_at?: string | null
  sources_total?: number | null
  sources_completed?: number | null
  percent_complete?: number | null
  progress?: Record<string, OsintSourceProgress>
  schedule?: OsintScheduleInfo | null
}

type OsintStatus = {
  queue?: {
    name?: string | null
    workers?: number | null
    max_workers?: number | null
    queued?: number | null
    started?: boolean
    shutdown?: boolean
  } | null
  active?: OsintJobSummary[]
  recent?: OsintJobSummary[]
}

type RagJobSummary = {
  id?: number | null
  status?: string | null
  progress?: number | null
  created_at?: string | null
  started_at?: string | null
  finished_at?: string | null
  error?: string | null
  collection_id?: number | null
  collection_name?: string | null
  total_files?: number | null
}

type LlmSummary = {
  cache?: CacheInfo | null
  endpoints?: {
    total?: number | null
    per_provider?: Record<string, number>
  } | null
  latency_ms?: {
    total_avg_ms?: number | null
    answer_avg_ms?: number | null
    snippet_avg_ms?: number | null
  } | null
}

type UserLoginSummary = {
  user_id?: number | null
  username?: string | null
  full_name?: string | null
  at?: string | null
}

type UserActivitySummary = {
  total?: number | null
  active_24h?: number | null
  recent?: UserLoginSummary[]
}

type IntegrationBrowser = {
  headless?: boolean
  proxy?: string | null
  viewport?: number[]
  context_timeout_ms?: number | null
  navigation_timeout_ms?: number | null
}

type IntegrationHttp = {
  timeout?: number | null
  connect_timeout?: number | null
  retries?: number | null
  backoff_factor?: number | null
}

type IntegrationOsint = {
  cache_enabled?: boolean
  cache_ttl_seconds?: number | null
  retry_user_agents?: number | null
  retry_proxies?: number | null
  wait_after_load_ms?: number | null
  navigation_timeout_ms?: number | null
  user_agent_override?: string | null
  queue?: Record<string, any> | null
}

type IntegrationsSummary = {
  browser?: IntegrationBrowser | null
  http?: IntegrationHttp | null
  osint?: IntegrationOsint | null
}

type StatusPayload = {
  ok: boolean
  status: string
  generated_at?: string | null
  warnings?: string[]
  errors?: Record<string, string>
  app?: {
    created_at?: string | null
    started_at?: string | null
    uptime_seconds?: number | null
    uptime_human?: string | null
    environment?: string | null
    debug?: boolean
    version?: string | null
  }
  system?: {
    host?: string | null
    platform?: string | null
    python_version?: string | null
    pid?: number | null
    timezone?: string[] | null
    load_average?: number[] | null
  }
  queue?: {
    status?: string | null
    name?: string | null
    workers?: number | null
    max_workers?: number | null
    queued?: number | null
    started?: boolean
    shutdown?: boolean
  }
  database?: {
    status?: string | null
    path?: string | null
    size_bytes?: number | null
    size_pretty?: string | null
    counts?: Record<string, number | null>
    user_roles?: Record<string, number>
    rag_status?: {
      counts?: Record<string, number>
      last_indexed_at?: string | null
    }
  }
  tasks?: {
    counts?: Record<string, number>
    oldest_active?: Record<string, any> | null
    recent?: Array<Record<string, any>>
  }
  ai_search?: {
    window_size?: number
    latency_avg_ms?: number | null
    last_measurement?: {
      created_at?: string | null
      total_ms?: number | null
      keywords_ms?: number | null
      candidate_ms?: number | null
      deep_ms?: number | null
      llm_answer_ms?: number | null
      llm_snippet_ms?: number | null
      meta?: Record<string, any> | null
    } | null
  }
  scan?: {
    running?: boolean
    stage?: string | null
    updated_at?: string | null
    processed?: number | null
    total?: number | null
    added?: number | null
    updated?: number | null
    removed?: number | null
    current?: string | null
    eta_seconds?: number | null
    eta_human?: string | null
    percent?: number | null
    scope?: {
      type?: string | null
      label?: string | null
      count?: number | null
    } | null
    task_id?: number | null
    last_log?: {
      at?: string | null
      level?: string | null
      message?: string | null
    } | null
    error?: string | null
  }
  runtime?: {
    scan_root?: string | null
    import_subdir?: string | null
    default_use_llm?: boolean
    default_prune?: boolean
    keywords_to_tags_enabled?: boolean
    transcribe_enabled?: boolean
    images_vision_enabled?: boolean
    ai_query_variants_max?: number | null
    lm_default_provider?: string | null
    lm_model?: string | null
    lm_tokenizer?: string | null
    lm_max_input_chars?: number | null
    lm_max_output_tokens?: number | null
    rag_embedding_backend?: string | null
    rag_embedding_model?: string | null
    rag_embedding_dim?: number | null
    rag_rerank_backend?: string | null
    rag_rerank_model?: string | null
    rag_rerank_batch_size?: number | null
    rag_rerank_max_length?: number | null
    rag_rerank_max_chars?: number | null
    ai_rerank_llm?: boolean
    llm_endpoints?: RuntimeLlmEndpoint[]
    disk_usage?: {
      total_bytes?: number | null
      used_bytes?: number | null
      free_bytes?: number | null
      total_pretty?: string | null
      used_pretty?: string | null
      free_pretty?: string | null
    }
  }
  services?: Record<string, ServiceInfo>
  cache?: Record<string, CacheInfo>
  feedback?: FeedbackSchedulerStatus
  rag?: {
    ready?: number | null
    pending?: number | null
    jobs?: RagJobSummary[]
  }
  osint?: OsintStatus | null
  llm?: LlmSummary | null
  users?: UserActivitySummary | null
  integrations?: IntegrationsSummary | null
}

const statusBadge: Record<string, string> = {
  ok: 'bg-success',
  degraded: 'bg-warning text-dark',
  warning: 'bg-warning text-dark',
  error: 'bg-danger',
}

const serviceBadge: Record<string, string> = {
  running: 'bg-success',
  idle: 'bg-secondary',
  warning: 'bg-warning text-dark',
}

const osintStatusBadge: Record<string, string> = {
  running: 'bg-primary',
  queued: 'bg-secondary',
  pending: 'bg-secondary',
  completed: 'bg-success',
  error: 'bg-danger',
  blocked: 'bg-warning text-dark',
  cancelled: 'bg-warning text-dark',
}

const osintStatusLabel: Record<string, string> = {
  running: 'Выполняется',
  queued: 'В очереди',
  pending: 'Подготовка',
  completed: 'Готово',
  error: 'Ошибка',
  blocked: 'Блокировано',
  cancelled: 'Отменено',
}

const osintFinalStatuses = new Set(['completed', 'error', 'blocked', 'cancelled'])

const formatDateTime = (value?: string | null): string => {
  if (!value) return '—'
  try {
    return new Date(value).toLocaleString()
  } catch {
    return value
  }
}

const formatNumber = (value?: number | null): string => {
  if (value === null || value === undefined) return '—'
  if (!Number.isFinite(value)) return '—'
  return value.toLocaleString()
}

const formatBool = (value: boolean | undefined, { truthy = 'Вкл', falsy = 'Выкл' } = {}): string =>
  value ? truthy : falsy

const formatMs = (value?: number | null): string => {
  if (value === null || value === undefined) return '—'
  if (!Number.isFinite(value)) return '—'
  return `${Math.round(value).toLocaleString()} мс`
}

const formatPercent = (value?: number | null): string => {
  if (value === null || value === undefined) return '—'
  if (!Number.isFinite(value)) return '—'
  return `${value.toFixed(1)}%`
}

const formatSeconds = (value?: number | null): string => {
  if (value === null || value === undefined) return '—'
  if (!Number.isFinite(value)) return '—'
  const seconds = Math.max(0, Math.round(value))
  if (seconds < 60) return `${seconds} с`
  if (seconds < 3600) {
    const mins = Math.round(seconds / 60)
    return `${mins} мин`
  }
  if (seconds < 86400) {
    const hours = seconds / 3600
    return hours < 10 ? `${hours.toFixed(1)} ч` : `${Math.round(hours)} ч`
  }
  const days = seconds / 86400
  return `${days.toFixed(1)} д`
}

const formatTimeAgo = (value?: string | null): string => {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  const diffMs = Date.now() - date.getTime()
  if (diffMs < 0) return value
  const diffSec = Math.floor(diffMs / 1000)
  if (diffSec < 60) return 'только что'
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)} мин назад`
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)} ч назад`
  return `${Math.floor(diffSec / 86400)} дн назад`
}

const formatOsintStatus = (value?: string | null): string => {
  if (!value) return '—'
  const key = value.toLowerCase()
  return osintStatusLabel[key] || value
}

const formatTaskStatus = (value?: string | null): string => {
  if (!value) return '—'
  return taskStatusRu(value)
}

const getOsintJobCounters = (job?: OsintJobSummary | null): { completed: number; total: number } => {
  const progressEntries = Object.values(job?.progress || {})
  const totalRaw = job?.sources_total ?? progressEntries.length
  const total = Number.isFinite(totalRaw as number) ? Number(totalRaw) || 0 : progressEntries.length
  const completedRaw = job?.sources_completed
  let completed = 0
  if (typeof completedRaw === 'number' && Number.isFinite(completedRaw)) {
    completed = completedRaw
  } else {
    completed = progressEntries.filter(entry => osintFinalStatuses.has(String(entry?.status || '').toLowerCase())).length
  }
  return { completed, total: total || progressEntries.length }
}

const getOsintJobPercent = (job?: OsintJobSummary | null): number => {
  if (!job) return 0
  if (typeof job.percent_complete === 'number' && Number.isFinite(job.percent_complete)) {
    return Math.min(100, Math.max(0, job.percent_complete))
  }
  const { completed, total } = getOsintJobCounters(job)
  if (!total) return 0
  return Math.min(100, Math.max(0, (completed / total) * 100))
}

const getOsintBadgeClass = (status?: string | null): string => {
  if (!status) return 'bg-secondary'
  return osintStatusBadge[status.toLowerCase()] || 'bg-secondary'
}

const formatViewport = (value?: number[] | null): string => {
  if (!value || value.length < 2) return '—'
  return `${value[0]}×${value[1]}`
}

export default function AdminServiceStatusPage() {
  const { user } = useAuth()
  const isAdmin = user?.role === 'admin'
  const navigate = useNavigate()
  const toasts = useToasts()
  const [data, setData] = useState<StatusPayload | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [schedulerPending, setSchedulerPending] = useState(false)
  const [actionKey, setActionKey] = useState('')

  const loadStatus = useCallback(async ({ silent = false } = {}) => {
    if (!isAdmin) return
    if (!silent) {
      setLoading(true)
      setError('')
    }
    try {
      const resp = await fetch('/api/admin/status/overview')
      if (!resp.ok) {
        const detail = await resp.text().catch(() => '')
        throw new Error(detail || `HTTP ${resp.status}`)
      }
      const payload: StatusPayload = await resp.json()
      setData(payload)
      setError('')
    } catch (err: any) {
      setError(err?.message || 'Не удалось загрузить состояние сервиса')
    } finally {
      if (!silent) {
        setLoading(false)
      }
    }
  }, [isAdmin])

  useEffect(() => { loadStatus() }, [loadStatus])

  useEffect(() => {
    if (!autoRefresh) return undefined
    const timer = window.setInterval(() => loadStatus({ silent: true }), 20000)
    return () => window.clearInterval(timer)
  }, [autoRefresh, loadStatus])

  const feedback = data?.feedback
  const services = data?.services ?? {}
  const caches = data?.cache ?? {}
  const runtimeInfo = data?.runtime
  const runtimeLlmEndpointsRaw = runtimeInfo?.llm_endpoints
  const llmEndpoints = Array.isArray(runtimeLlmEndpointsRaw) ? runtimeLlmEndpointsRaw.slice(0, 6) : []
  const llmEndpointsHidden = Array.isArray(runtimeLlmEndpointsRaw)
    ? Math.max(0, runtimeLlmEndpointsRaw.length - llmEndpoints.length)
    : 0
  const tokenLimitsLabel = `Вход: ${formatNumber(runtimeInfo?.lm_max_input_chars)} симв · Выход: ${formatNumber(runtimeInfo?.lm_max_output_tokens)} ток`
  const embeddingMetaParts: string[] = []
  if (runtimeInfo?.rag_embedding_backend) embeddingMetaParts.push(runtimeInfo.rag_embedding_backend)
  if (typeof runtimeInfo?.rag_embedding_dim === 'number' && Number.isFinite(runtimeInfo.rag_embedding_dim) && runtimeInfo.rag_embedding_dim > 0) {
    embeddingMetaParts.push(`${formatNumber(runtimeInfo.rag_embedding_dim)} ед.`)
  }
  const embeddingMeta = embeddingMetaParts.join(' · ')
  const rerankMetaParts: string[] = []
  if (runtimeInfo?.rag_rerank_backend) rerankMetaParts.push(runtimeInfo.rag_rerank_backend)
  if (typeof runtimeInfo?.rag_rerank_batch_size === 'number' && Number.isFinite(runtimeInfo.rag_rerank_batch_size) && runtimeInfo.rag_rerank_batch_size > 0) {
    rerankMetaParts.push(`батч ${formatNumber(runtimeInfo.rag_rerank_batch_size)}`)
  }
  if (typeof runtimeInfo?.rag_rerank_max_length === 'number' && Number.isFinite(runtimeInfo.rag_rerank_max_length) && runtimeInfo.rag_rerank_max_length > 0) {
    rerankMetaParts.push(`макс ${formatNumber(runtimeInfo.rag_rerank_max_length)} ток`)
  }
  const rerankMeta = rerankMetaParts.join(' · ')
  const providerLabel = runtimeInfo?.lm_default_provider ? runtimeInfo.lm_default_provider.replace(/_/g, ' ') : ''
  const providerDisplay = providerLabel ? providerLabel.charAt(0).toUpperCase() + providerLabel.slice(1) : ''
  const schedulerEnabled = !!feedback?.scheduler_enabled
  const nextRunLabel = formatSeconds(feedback?.next_run_in_seconds)
  const lastTriggerLabel = formatTimeAgo(feedback?.last_trigger_at)

  const handleToggleScheduler = useCallback(async () => {
    if (!isAdmin) return
    const currentlyEnabled = !!feedback?.scheduler_enabled
    const interval = feedback?.interval_hours && feedback.interval_hours > 0 ? feedback.interval_hours : 6
    setSchedulerPending(true)
    try {
      const body: Record<string, any> = currentlyEnabled ? { enabled: false } : { enabled: true, interval_hours: interval }
      const resp = await fetch('/api/admin/ai-search/feedback/scheduler', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const result = await resp.json().catch(() => ({}))
      if (!resp.ok || result?.ok === false) {
        throw new Error(result?.error || `HTTP ${resp.status}`)
      }
      toasts.push(currentlyEnabled ? 'Автотренер отключён' : 'Автотренер включён', 'success')
      await loadStatus({ silent: true })
    } catch (err: any) {
      const message = err?.message || 'Не удалось обновить настройки тренера'
      setError(message)
      toasts.push(message, 'error')
    } finally {
      setSchedulerPending(false)
    }
  }, [feedback, isAdmin, loadStatus, toasts])

  const handleRunFeedbackNow = useCallback(async () => {
    if (!isAdmin) return
    setActionKey('train')
    try {
      const resp = await fetch('/api/admin/ai-search/feedback/scheduler', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_now: true }),
      })
      const result = await resp.json().catch(() => ({}))
      if (!resp.ok || result?.ok === false) {
        throw new Error(result?.error || `HTTP ${resp.status}`)
      }
      const queued = result?.manual_task ? !!result.manual_task.queued : true
      toasts.push(queued ? 'Задача обучения поставлена в очередь' : 'Тренировка уже выполняется', queued ? 'success' : 'info')
      await loadStatus({ silent: true })
    } catch (err: any) {
      const message = err?.message || 'Не удалось запустить обучение'
      setError(message)
      toasts.push(message, 'error')
    } finally {
      setActionKey('')
    }
  }, [isAdmin, loadStatus, toasts])

  const handleClearCache = useCallback(async (target: 'llm' | 'search') => {
    if (!isAdmin) return
    const endpoint = target === 'llm' ? '/api/admin/cache/llm' : '/api/admin/cache/search'
    setActionKey(target)
    try {
      const resp = await fetch(endpoint, { method: 'DELETE' })
      const result = await resp.json().catch(() => ({}))
      if (!resp.ok || result?.ok === false) {
        throw new Error(result?.error || `HTTP ${resp.status}`)
      }
      toasts.push(target === 'llm' ? 'Кэш LLM очищен' : 'Кэш поиска очищен', 'success')
      await loadStatus({ silent: true })
    } catch (err: any) {
      const message = err?.message || 'Не удалось очистить кэш'
      setError(message)
      toasts.push(message, 'error')
    } finally {
      setActionKey('')
    }
  }, [isAdmin, loadStatus, toasts])


  const activeBadgeClass = useMemo(() => {
    const key = (data?.status || '').toLowerCase()
    return statusBadge[key] || 'bg-secondary'
  }, [data?.status])

  if (!isAdmin) {
    return <div className="card p-3">Недостаточно прав для просмотра раздела.</div>
  }

  const hasData = !!data
  const showSkeleton = !hasData && loading

  const queueStatus = data?.queue?.status || 'unknown'
  const queueBadgeClass = statusBadge[queueStatus] || 'bg-secondary'

  const warnings = data?.warnings || []
  const errors = data?.errors || {}
  const errorEntries = Object.entries(errors)
  const osintInfo = data?.osint
  const osintQueue = osintInfo?.queue
  const osintActive = (osintInfo?.active ?? []).filter(
    (job): job is OsintJobSummary => Boolean(job && typeof job === 'object')
  )
  const osintRecent = (osintInfo?.recent ?? []).filter(
    (job): job is OsintJobSummary => Boolean(job && typeof job === 'object')
  )
  const osintRecentLimited = osintRecent.slice(0, 5)
  const ragJobs = (data?.rag?.jobs ?? []).filter(
    (job): job is RagJobSummary => Boolean(job && typeof job === 'object')
  )
  const llmSummary = data?.llm
  const userActivity = data?.users
  const integrations = data?.integrations

  return (
    <div className="d-grid gap-3 admin-page-glass" aria-busy={loading}>
      <div className="card p-3" aria-busy={loading}>
        <div className="d-flex flex-column flex-lg-row justify-content-between gap-3">
          <div>
            <div className="d-flex align-items-center gap-2">
              <h1 className="h4 mb-0">Состояние сервиса</h1>
              <span className={`badge ${activeBadgeClass}`} aria-live="polite">
                {data?.status === 'ok'
                  ? 'Норма'
                  : data?.status === 'degraded'
                    ? 'Есть предупреждения'
                    : data?.status === 'error'
                      ? 'Ошибка'
                      : (data?.status || 'Нет данных')}
              </span>
            </div>
            <div className="text-body-secondary small mt-1">
              Последнее обновление: {formatDateTime(data?.generated_at || data?.app?.created_at)}
              {data?.app?.uptime_human ? ` · Аптайм: ${data.app.uptime_human}` : ''}
            </div>
            <div className="text-body-secondary small">
              Версия: 2.12.3 · 10.02.2026
            </div>
            {data?.app?.environment && (
              <div className="text-body-secondary small">
                Среда: {data.app.environment}{data.app.debug ? ' (debug)' : ''}
              </div>
            )}
            {feedback && (
              <div className="text-body-secondary small mt-2">
                Автотренер: <span className={schedulerEnabled ? 'text-success' : 'text-warning'}>{schedulerEnabled ? 'включён' : 'выключен'}</span>
                {feedback.interval_hours ? ` · интервал: ${feedback.interval_hours} ч` : ''}
                {feedback.next_run_in_seconds !== undefined && feedback.next_run_in_seconds !== null ? ` · следующий запуск через ${nextRunLabel}` : ''}
                {feedback.last_trigger_at ? ` · последний запуск ${lastTriggerLabel}` : ''}
                {typeof feedback.total_weighted === 'number' ? ` · документов с весами: ${formatNumber(feedback.total_weighted)}` : ''}
              </div>
            )}
          </div>
          <div className="d-flex flex-column flex-sm-row gap-2 align-items-sm-center align-items-start">
            <label className="form-check form-switch m-0">
              <input
                type="checkbox"
                className="form-check-input"
                checked={autoRefresh}
                onChange={event => setAutoRefresh(event.target.checked)}
              />
              <span className="form-check-label small">Автообновление (20 с)</span>
            </label>
            <button
              type="button"
              className="btn btn-outline-secondary btn-sm"
              onClick={() => loadStatus()}
              disabled={loading}
            >
              {loading ? 'Обновление…' : 'Обновить'}
            </button>
          </div>
        </div>

        <div className="d-flex flex-wrap gap-2 mt-3">
          <button
            type="button"
            className={`btn btn-sm ${schedulerEnabled ? 'btn-outline-danger' : 'btn-outline-success'}`}
            onClick={handleToggleScheduler}
            disabled={schedulerPending || loading}
          >
            {schedulerPending ? 'Сохраняем…' : schedulerEnabled ? 'Выключить автотренер' : 'Включить автотренер'}
          </button>
          <button
            type="button"
            className="btn btn-sm btn-outline-primary"
            onClick={handleRunFeedbackNow}
            disabled={actionKey === 'train' || loading}
          >
            {actionKey === 'train' ? 'Запуск…' : 'Обучить фидбэк сейчас'}
          </button>
          <button
            type="button"
            className="btn btn-sm btn-outline-secondary"
            onClick={() => handleClearCache('llm')}
            disabled={actionKey === 'llm' || loading}
          >
            {actionKey === 'llm' ? 'Очищаем…' : 'Очистить кэш LLM'}
          </button>
          <button
            type="button"
            className="btn btn-sm btn-outline-secondary"
            onClick={() => handleClearCache('search')}
            disabled={actionKey === 'search' || loading}
          >
            {actionKey === 'search' ? 'Очищаем…' : 'Очистить кэш поиска'}
          </button>
          <button
            type="button"
            className="btn btn-sm btn-outline-secondary"
            onClick={() => navigate('/admin/ai-metrics')}
          >
            AI метрики
          </button>
          <button
            type="button"
            className="btn btn-sm btn-outline-secondary"
            onClick={() => navigate('/admin/tasks')}
          >
            Задачи
          </button>
          <button
            type="button"
            className="btn btn-sm btn-outline-secondary"
            onClick={() => navigate('/admin/llm')}
          >
            LLM эндпоинты
          </button>
          <button
            type="button"
            className="btn btn-sm btn-outline-secondary"
            onClick={() => navigate('/settings')}
          >
            Настройки
          </button>
          <button
            type="button"
            className="btn btn-sm btn-outline-secondary"
            onClick={() => navigate('/admin/logs')}
          >
            Логи
          </button>
          <button
            type="button"
            className="btn btn-sm btn-outline-secondary"
            onClick={() => navigate('/ingest')}
          >
            Импорт
          </button>
        </div>

        {error && (
          <div className="alert alert-danger mt-3 mb-0" role="alert">
            {error}
          </div>
        )}
        {warnings.length > 0 && (
          <div className="alert alert-warning mt-3 mb-0" role="alert">
            <div className="fw-semibold mb-2">Предупреждения ({warnings.length}):</div>
            <ul className="mb-0 ps-3">
              {warnings.map((item, idx) => (
                <li key={idx}>{item}</li>
              ))}
            </ul>
          </div>
        )}
        {errorEntries.length > 0 && (
          <details className="mt-3">
            <summary className="text-body-secondary small">Технические ошибки ({errorEntries.length})</summary>
            <ul className="mb-0 ps-3 small">
              {errorEntries.map(([key, value]) => (
                <li key={key}>
                  <strong>{key}</strong>: {value}
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>

      {showSkeleton && (
        <div className="card p-4 text-center text-body-secondary" role="status">
          Загрузка состояния…
        </div>
      )}

      {hasData && (
        <>
          <div className="row g-3">
            <div className="col-12 col-xl-6">
              <div className="card h-100 p-3">
                <div className="d-flex justify-content-between align-items-start">
                  <h2 className="h5 mb-3">Система</h2>
                  {data?.app?.version && <span className="badge bg-secondary">Версия: {data.app.version}</span>}
                </div>
                <dl className="row mb-0">
                  <dt className="col-sm-4">Хост</dt>
                  <dd className="col-sm-8">{data?.system?.host || '—'}</dd>
                  <dt className="col-sm-4">Платформа</dt>
                  <dd className="col-sm-8">
                    {data?.system?.platform || '—'}
                    {data?.system?.python_version ? ` · Python ${data.system.python_version}` : ''}
                  </dd>
                  <dt className="col-sm-4">PID</dt>
                  <dd className="col-sm-8">{formatNumber(data?.system?.pid)}</dd>
                  <dt className="col-sm-4">Часовой пояс</dt>
                  <dd className="col-sm-8">
                    {Array.isArray(data?.system?.timezone) ? data?.system?.timezone?.join(', ') : '—'}
                  </dd>
                  <dt className="col-sm-4">Load Avg</dt>
                  <dd className="col-sm-8">
                    {Array.isArray(data?.system?.load_average) && data.system.load_average?.length
                      ? data.system.load_average.map((value, idx) => (
                        <span key={idx} className="me-2">{Number(value).toFixed(2)}</span>
                      ))
                      : '—'}
                  </dd>
                  <dt className="col-sm-4">Очередь задач</dt>
                  <dd className="col-sm-8">
                    <span className={`badge ${queueBadgeClass} me-2`}>
                      {queueStatus === 'ok' ? 'Активна' : queueStatus === 'warning' ? 'Требует внимания' : queueStatus}
                    </span>
                    Потоков: {formatNumber(data?.queue?.workers)} / {formatNumber(data?.queue?.max_workers)} · В очереди: {formatNumber(data?.queue?.queued)}
                  </dd>
                </dl>
              </div>
            </div>
            <div className="col-12 col-xl-6">
              <div className="card h-100 p-3">
                <div className="d-flex justify-content-between align-items-start">
                  <h2 className="h5 mb-3">Настройки и диск</h2>
                  {data?.runtime?.disk_usage?.total_pretty && (
                    <span className="badge bg-secondary">
                      Диск: {data.runtime.disk_usage.total_pretty}
                    </span>
                  )}
                </div>
                <dl className="row mb-0">
                  <dt className="col-sm-5">Scan root</dt>
                  <dd className="col-sm-7 text-truncate" title={data?.runtime?.scan_root || undefined}>
                    {data?.runtime?.scan_root || '—'}
                  </dd>
                  <dt className="col-sm-5">Каталог импорта</dt>
                  <dd className="col-sm-7">{data?.runtime?.import_subdir || '—'}</dd>
                  <dt className="col-sm-5">Основная модель</dt>
                  <dd className="col-sm-7">
                    <div>{runtimeInfo?.lm_model || '—'}</div>
                    {(providerDisplay || runtimeInfo?.lm_default_provider) && (
                      <div className="text-body-secondary small">
                        Провайдер: {providerDisplay || runtimeInfo?.lm_default_provider}
                      </div>
                    )}
                  </dd>
                  <dt className="col-sm-5">Токенизатор</dt>
                  <dd className="col-sm-7">
                    <div>{runtimeInfo?.lm_tokenizer || '—'}</div>
                    <div className="text-body-secondary small">{tokenLimitsLabel}</div>
                  </dd>
                  {llmEndpoints.length > 0 && (
                    <>
                      <dt className="col-sm-5">Доп. модели</dt>
                      <dd className="col-sm-7">
                        <div className="d-grid gap-2">
                          {llmEndpoints.map((endpoint, idx) => {
                            const metaParts: string[] = []
                            if (endpoint.provider) metaParts.push(endpoint.provider)
                            if (Array.isArray(endpoint.purposes) && endpoint.purposes.length) {
                              metaParts.push(endpoint.purposes.join(', '))
                            }
                            if (endpoint.tokenizer_hint) metaParts.push(endpoint.tokenizer_hint)
                            return (
                              <div key={endpoint.id ?? `${endpoint.name || endpoint.model || 'llm'}-${idx}`}>
                                <div className="fw-semibold small">{endpoint.name || endpoint.model || '—'}</div>
                                {endpoint.model && endpoint.name && endpoint.name !== endpoint.model && (
                                  <div className="text-body-secondary small">{endpoint.model}</div>
                                )}
                                <div className="text-body-secondary small">{metaParts.join(' · ') || '—'}</div>
                              </div>
                            )
                          })}
                          {llmEndpointsHidden > 0 && (
                            <div className="text-body-secondary small">
                              Ещё {formatNumber(llmEndpointsHidden)}…
                            </div>
                          )}
                        </div>
                      </dd>
                    </>
                  )}
                  <dt className="col-sm-5">Эмбеддер</dt>
                  <dd className="col-sm-7">
                    <div>{runtimeInfo?.rag_embedding_model || '—'}</div>
                    <div className="text-body-secondary small">{embeddingMeta || '—'}</div>
                  </dd>
                  <dt className="col-sm-5">Реранкер</dt>
                  <dd className="col-sm-7">
                    <div>{runtimeInfo?.rag_rerank_model || '—'}</div>
                    <div className="text-body-secondary small">{rerankMeta || '—'}</div>
                  </dd>
                  <dt className="col-sm-5">AI rerank</dt>
                  <dd className="col-sm-7">{formatBool(runtimeInfo?.ai_rerank_llm)}</dd>
                  <dt className="col-sm-5">AI запросов</dt>
                  <dd className="col-sm-7">{formatNumber(data?.runtime?.ai_query_variants_max)}</dd>
                  <dt className="col-sm-5">Распознавание</dt>
                  <dd className="col-sm-7">
                    {formatBool(data?.runtime?.transcribe_enabled)} · Vision {formatBool(data?.runtime?.images_vision_enabled)}
                  </dd>
                  <dt className="col-sm-5">Поиск</dt>
                  <dd className="col-sm-7">
                    Использовать LLM: {formatBool(data?.runtime?.default_use_llm)}
                    {` · Обрезка: ${formatBool(data?.runtime?.default_prune)}`}
                  </dd>
                  <dt className="col-sm-5">Автогенерация тегов</dt>
                  <dd className="col-sm-7">{formatBool(data?.runtime?.keywords_to_tags_enabled, { truthy: 'Да', falsy: 'Нет' })}</dd>
                  {data?.runtime?.disk_usage && (
                    <>
                      <dt className="col-sm-5">Свободно</dt>
                      <dd className="col-sm-7">{data.runtime.disk_usage.free_pretty || '—'}</dd>
                      <dt className="col-sm-5">Использовано</dt>
                      <dd className="col-sm-7">{data.runtime.disk_usage.used_pretty || '—'}</dd>
                    </>
                  )}
                </dl>
              </div>
            </div>
          </div>

          <div className="row g-3">
            <div className="col-12 col-lg-6">
              <div className="card h-100 p-3">
                <div className="d-flex justify-content-between align-items-center mb-3">
                  <h2 className="h5 mb-0">База данных</h2>
                  <span className={`badge ${(statusBadge[(data?.database?.status || '').toLowerCase()] || 'bg-secondary')}`}>
                    {data?.database?.status === 'ok' ? 'OK' : data?.database?.status || 'Нет данных'}
                  </span>
                </div>
                <div className="mb-2 small text-body-secondary">
                  {data?.database?.path || 'Файл не определён'}
                </div>
                <div className="row g-2">
                  <div className="col-6">
                    <div className="border rounded-3 p-3 text-center">
                      <div className="text-body-secondary small text-uppercase mb-1">Размер</div>
                      <div className="fw-semibold">{data?.database?.size_pretty || '—'}</div>
                    </div>
                  </div>
                  <div className="col-6">
                    <div className="border rounded-3 p-3 text-center">
                      <div className="text-body-secondary small text-uppercase mb-1">RAG готово</div>
                      <div className="fw-semibold">{formatNumber(data?.rag?.ready)}</div>
                    </div>
                  </div>
                </div>
                <div className="mt-3">
                  <div className="fw-semibold mb-2">Объекты</div>
                  <table className="table table-sm align-middle mb-2">
                    <tbody>
                      {Object.entries(data?.database?.counts || {}).map(([key, value]) => (
                        <tr key={key}>
                          <td className="text-body-secondary text-capitalize">{key}</td>
                          <td className="text-end">{formatNumber(value)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {data?.database?.user_roles && (
                    <>
                      <div className="fw-semibold mb-2">Роли пользователей</div>
                      <div className="d-flex flex-wrap gap-2">
                        {Object.entries(data.database.user_roles).map(([role, count]) => (
                          <span key={role} className="badge bg-secondary">
                            {role}: {formatNumber(count)}
                          </span>
                        ))}
                      </div>
                    </>
                  )}
                  {data?.database?.rag_status?.last_indexed_at && (
                    <div className="text-body-secondary small mt-2">
                      Последняя индексация RAG: {formatDateTime(data.database.rag_status.last_indexed_at)}
                    </div>
                  )}
                </div>
              </div>
            </div>
            <div className="col-12 col-lg-6">
              <div className="card h-100 p-3">
                <h2 className="h5 mb-3">AI поиск</h2>
                <dl className="row mb-0">
                  <dt className="col-sm-6">Окно метрик</dt>
                  <dd className="col-sm-6">{formatNumber(data?.ai_search?.window_size)}</dd>
                  <dt className="col-sm-6">Средняя задержка</dt>
                  <dd className="col-sm-6">{formatMs(data?.ai_search?.latency_avg_ms)}</dd>
                  <dt className="col-sm-6">Последний замер</dt>
                  <dd className="col-sm-6">{formatDateTime(data?.ai_search?.last_measurement?.created_at)}</dd>
                  <dt className="col-sm-6">Общее время</dt>
                  <dd className="col-sm-6">{formatMs(data?.ai_search?.last_measurement?.total_ms)}</dd>
                  <dt className="col-sm-6">LLM ответ</dt>
                  <dd className="col-sm-6">{formatMs(data?.ai_search?.last_measurement?.llm_answer_ms)}</dd>
                  <dt className="col-sm-6">Глубокий поиск</dt>
                  <dd className="col-sm-6">{formatMs(data?.ai_search?.last_measurement?.deep_ms)}</dd>
                </dl>
              </div>
            </div>
          </div>

          <div className="row g-3">
            <div className="col-12 col-xl-6">
              <div className="card h-100 p-3">
                <div className="d-flex justify-content-between align-items-center mb-3">
                  <h2 className="h5 mb-0">Сканирование</h2>
                  <span className={`badge ${data?.scan?.running ? 'bg-primary' : 'bg-secondary'}`}>
                    {data?.scan?.running ? 'В процессе' : 'Ожидание'}
                  </span>
                </div>
                <dl className="row mb-0">
                  <dt className="col-sm-5">Стадия</dt>
                  <dd className="col-sm-7">{data?.scan?.stage || '—'}</dd>
                  <dt className="col-sm-5">Прогресс</dt>
                  <dd className="col-sm-7">
                    {formatNumber(data?.scan?.processed)} / {formatNumber(data?.scan?.total)} ({formatPercent(data?.scan?.percent)})
                  </dd>
                  <dt className="col-sm-5">ETA</dt>
                  <dd className="col-sm-7">{data?.scan?.eta_human || '—'}</dd>
                  <dt className="col-sm-5">Текущий файл</dt>
                  <dd className="col-sm-7">
                    <span className="text-truncate d-inline-block" style={{ maxWidth: '100%' }} title={data?.scan?.current || undefined}>
                      {data?.scan?.current || '—'}
                    </span>
                  </dd>
                  {data?.scan?.scope?.label && (
                    <>
                      <dt className="col-sm-5">Область</dt>
                      <dd className="col-sm-7">
                        {data.scan.scope.label}
                        {data.scan.scope.count ? ` · ${formatNumber(data.scan.scope.count)}` : ''}
                      </dd>
                    </>
                  )}
                  <dt className="col-sm-5">Обновлено</dt>
                  <dd className="col-sm-7">{formatDateTime(data?.scan?.updated_at)}</dd>
                </dl>
                {data?.scan?.last_log && (
                  <div className="mt-3 border-top pt-3 small text-body-secondary">
                    <div className="fw-semibold">Последняя запись журнала</div>
                    <div>{formatDateTime(data.scan.last_log.at)} · {data.scan.last_log.level}</div>
                    <div>{data.scan.last_log.message}</div>
                  </div>
                )}
                {data?.scan?.error && (
                  <div className="alert alert-danger mt-3 mb-0" role="alert">
                    Ошибка: {data.scan.error}
                  </div>
                )}
              </div>
            </div>
            <div className="col-12 col-xl-6">
              <div className="card h-100 p-3">
                <h2 className="h5 mb-3">Фоновые сервисы</h2>
                <div className="d-grid gap-2">
                  {Object.entries(data?.services || {}).map(([key, value]) => {
                    const enabled = value?.enabled ?? false
                    let badgeKey: keyof typeof serviceBadge = enabled
                      ? (value?.running ? 'running' : 'warning')
                      : 'idle'
                    if (!enabled) badgeKey = 'idle'
                    const badgeClass = serviceBadge[badgeKey] || 'bg-secondary'
                    return (
                      <div key={key} className="border rounded-3 p-3 d-flex justify-content-between align-items-center">
                        <div>
                          <div className="fw-semibold text-capitalize">{key.replace(/_/g, ' ')}</div>
                          <div className="text-body-secondary small">
                            {enabled ? 'Включен' : 'Отключен'}
                            {value?.interval_hours ? ` · Интервал: ${value.interval_hours} ч` : ''}
                          </div>
                        </div>
                        <span className={`badge ${badgeClass}`}>
                          {enabled ? (value?.running ? 'Работает' : 'Ожидание') : 'Отключено'}
                        </span>
                      </div>
                    )
                  })}
                </div>
                <div className="mt-3 border-top pt-3">
                  <h3 className="h6 mb-2">Кэши</h3>
                  <div className="d-grid gap-2">
                    {Object.entries(caches).map(([key, cache]) => {
                      const label = key === 'llm' ? 'Кэш LLM' : key === 'search' ? 'Кэш поиска' : key
                      return (
                        <div key={key} className="border rounded-3 p-3 d-flex justify-content-between align-items-center">
                          <div>
                            <div className="fw-semibold">{label}</div>
                            <div className="text-body-secondary small">
                              {cache?.enabled ? 'Включен' : 'Отключен'}
                              {cache?.ttl_seconds ? ` · TTL: ${formatSeconds(cache.ttl_seconds)}` : ''}
                            </div>
                          </div>
                          <div className="text-body-secondary small">
                            {formatNumber(cache?.items)} / {formatNumber(cache?.max_items)}
                          </div>
                        </div>
                      )
                    })}
                    {Object.keys(caches).length === 0 && (
                      <div className="text-body-secondary small">Нет активных кэшей</div>
                    )}
                  </div>
                </div>
                <div className="mt-3 border-top pt-3">
                  <h3 className="h6 mb-2">Задачи</h3>
                  <div className="d-flex flex-wrap gap-2 mb-3">
                    {Object.entries(data?.tasks?.counts || {}).map(([status, count]) => (
                      <span key={status} className="badge bg-secondary">
                        {status}: {formatNumber(count)}
                      </span>
                    ))}
                  </div>
                  <div className="table-responsive">
                    <table className="table table-sm align-middle mb-0">
                      <thead>
                        <tr>
                          <th>ID</th>
                          <th>Имя</th>
                          <th>Статус</th>
                          <th>Создана</th>
                          <th>Прогресс</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(data?.tasks?.recent || []).map(task => (
                          <tr key={task.id}>
                            <td>{task.id}</td>
                            <td>{task.name}</td>
                            <td>{task.status}</td>
                            <td>{formatDateTime(task.created_at)}</td>
                            <td>{`${Math.round((task.progress ?? 0) * 100)}%`}</td>
                          </tr>
                        ))}
                        {(data?.tasks?.recent || []).length === 0 && (
                          <tr>
                            <td colSpan={5} className="text-center text-body-secondary">Нет записей</td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                  {data?.tasks?.oldest_active && (
                    <div className="text-body-secondary small mt-2">
                      Долгоживущая задача: {data.tasks.oldest_active.name} · {data.tasks.oldest_active.status} ·{' '}
                      {formatDateTime(data.tasks.oldest_active.created_at)}
                    </div>
                  )}
                </div>
                <div className="mt-3 border-top pt-3">
                  <div className="d-flex justify-content-between align-items-center mb-2">
                    <h3 className="h6 mb-0">OSINT поиск</h3>
                    {osintQueue && (
                      <span className="text-body-secondary small">
                        Потоков: {formatNumber(osintQueue.workers)} / {formatNumber(osintQueue.max_workers)} · В очереди:{' '}
                        {formatNumber(osintQueue.queued)}
                      </span>
                    )}
                  </div>
                  {osintActive.length === 0 && (
                    <div className="text-body-secondary small">Активных OSINT-задач нет</div>
                  )}
                  <div className="d-grid gap-2">
                    {osintActive.map((job, idx) => {
                      const key = job.id ?? `${job.query || 'job'}-${idx}`
                      const { completed, total } = getOsintJobCounters(job)
                      const percent = getOsintJobPercent(job)
                      const sources = Object.entries(job.progress || {})
                      return (
                        <div key={key} className="border rounded-3 p-3">
                          <div className="d-flex justify-content-between align-items-start gap-2">
                            <div className="flex-grow-1">
                              <div className="fw-semibold text-truncate" title={job.query || undefined}>
                                {job.query || `OSINT #${job.id ?? idx + 1}`}
                              </div>
                              <div className="text-body-secondary small">
                                ID: {formatNumber(job.id)} · Источники: {formatNumber(completed)} / {formatNumber(total)}
                              </div>
                            </div>
                            <span className={`badge ${getOsintBadgeClass(job.status)}`}>{formatOsintStatus(job.status)}</span>
                          </div>
                          <div className="progress mt-2" style={{ height: '6px' }}>
                            <div
                              className="progress-bar"
                              role="progressbar"
                              style={{ width: `${percent}%` }}
                              aria-valuenow={percent}
                              aria-valuemin={0}
                              aria-valuemax={100}
                            />
                          </div>
                          <div className="text-body-secondary small mt-2">
                            Создана {formatDateTime(job.created_at)}
                            {job.started_at ? ` · Старт: ${formatDateTime(job.started_at)}` : ''}
                            {job.schedule?.next_run_at ? ` · След. запуск: ${formatDateTime(job.schedule.next_run_at)}` : ''}
                          </div>
                          <div className="d-grid gap-1 mt-2">
                            {sources.map(([sourceId, info]) => (
                              <div key={sourceId} className="d-flex justify-content-between align-items-start gap-2">
                                <div className="flex-grow-1">
                                  <div className="fw-semibold small text-truncate" title={info?.label || sourceId}>
                                    {info?.label || sourceId}
                                  </div>
                                  <div className="text-body-secondary small">
                                    {info?.updated_at ? `Обновлено ${formatTimeAgo(info.updated_at)}` : 'Ожидает обработки'}
                                    {typeof info?.links_collected === 'number'
                                      ? ` · Ссылок: ${formatNumber(info.links_collected)}`
                                      : ''}
                                    {info?.fallback ? ' · Fallback' : ''}
                                  </div>
                                  {info?.error && <div className="text-danger small">Ошибка: {info.error}</div>}
                                </div>
                                <span className={`badge ${getOsintBadgeClass(info?.status)}`}>
                                  {formatOsintStatus(info?.status)}
                                </span>
                              </div>
                            ))}
                            {sources.length === 0 && (
                              <div className="text-body-secondary small">Нет данных об источниках</div>
                            )}
                          </div>
                          {job.error && (
                            <div className="alert alert-danger mt-2 mb-0 py-1 px-2 small">Ошибка: {job.error}</div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                  <div className="mt-3">
                    <div className="fw-semibold small mb-2">Недавние задачи</div>
                    <div className="table-responsive">
                      <table className="table table-sm align-middle mb-0">
                        <thead>
                          <tr>
                            <th>ID</th>
                            <th>Запрос</th>
                            <th>Статус</th>
                            <th>Источники</th>
                            <th>Прогресс</th>
                            <th>Создана</th>
                          </tr>
                        </thead>
                        <tbody>
                          {osintRecentLimited.map((job, idx) => {
                            const rowKey = job.id ?? `${job.query || 'recent'}-${idx}`
                            const { completed, total } = getOsintJobCounters(job)
                            const percent = getOsintJobPercent(job)
                            return (
                              <tr key={rowKey}>
                                <td>{formatNumber(job.id)}</td>
                                <td className="text-truncate" title={job.query || undefined}>{job.query || '—'}</td>
                                <td>
                                  <span className={`badge ${getOsintBadgeClass(job.status)}`}>
                                    {formatOsintStatus(job.status)}
                                  </span>
                                </td>
                                <td>
                                  {formatNumber(completed)} / {formatNumber(total)}
                                </td>
                                <td>{formatPercent(percent)}</td>
                                <td>{formatDateTime(job.created_at)}</td>
                              </tr>
                            )
                          })}
                          {osintRecentLimited.length === 0 && (
                            <tr>
                              <td colSpan={6} className="text-center text-body-secondary small">Нет записей OSINT</td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="row g-3">
            <div className="col-12 col-xl-6">
              <div className="card h-100 p-3">
                <h2 className="h5 mb-3">RAG индексатор</h2>
                <div className="d-flex flex-wrap gap-3">
                  <div className="border rounded-3 p-3 text-center flex-grow-1">
                    <div className="text-body-secondary small text-uppercase mb-1">Готово</div>
                    <div className="fw-semibold fs-5">{formatNumber(data?.rag?.ready)}</div>
                  </div>
                  <div className="border rounded-3 p-3 text-center flex-grow-1">
                    <div className="text-body-secondary small text-uppercase mb-1">В очереди</div>
                    <div className="fw-semibold fs-5">{formatNumber(data?.rag?.pending)}</div>
                  </div>
                </div>
                <div className="mt-3 border-top pt-3">
                  <div className="fw-semibold mb-2">Активные задачи</div>
                  {ragJobs.length === 0 && (
                    <div className="text-body-secondary small">Нет запущенных задач построения RAG</div>
                  )}
                  <div className="d-grid gap-2">
                    {ragJobs.map(job => {
                      const percent = Math.round(Math.max(0, Math.min(100, (job.progress ?? 0) * 100)))
                      const badge = statusBadge[(job.status || '').toLowerCase()] || 'bg-secondary'
                      return (
                        <div key={job.id ?? `${job.collection_id}-${job.created_at}`} className="border rounded-3 p-3">
                          <div className="d-flex justify-content-between align-items-start gap-2">
                            <div>
                              <div className="fw-semibold">
                                {job.collection_name || `Коллекция #${formatNumber(job.collection_id)}`}
                              </div>
                              <div className="text-body-secondary small">
                                Задача #{formatNumber(job.id)} · Файлов: {formatNumber(job.total_files)}
                              </div>
                            </div>
                            <span className={`badge ${badge}`}>{formatTaskStatus(job.status)}</span>
                          </div>
                          <div className="progress mt-2" style={{ height: '6px' }}>
                            <div
                              className="progress-bar"
                              role="progressbar"
                              style={{ width: `${percent}%` }}
                              aria-valuenow={percent}
                              aria-valuemin={0}
                              aria-valuemax={100}
                            />
                          </div>
                          <div className="text-body-secondary small mt-2">
                            {job.started_at ? `Старт: ${formatDateTime(job.started_at)}` : `Создана: ${formatDateTime(job.created_at)}`}
                          </div>
                          {job.error && <div className="text-danger small mt-1">Ошибка: {job.error}</div>}
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>
            </div>
            <div className="col-12 col-xl-6">
              <div className="card h-100 p-3">
                <h2 className="h5 mb-3">LLM</h2>
                <div className="row g-2">
                  <div className="col-6">
                    <div className="border rounded-3 p-3 text-center">
                      <div className="text-body-secondary small text-uppercase mb-1">Кэш</div>
                      <div className="fw-semibold">
                        {formatNumber(llmSummary?.cache?.items)} / {formatNumber(llmSummary?.cache?.max_items)}
                      </div>
                      <div className="text-body-secondary small">TTL: {formatSeconds(llmSummary?.cache?.ttl_seconds)}</div>
                    </div>
                  </div>
                  <div className="col-6">
                    <div className="border rounded-3 p-3 text-center">
                      <div className="text-body-secondary small text-uppercase mb-1">Эндпоинтов</div>
                      <div className="fw-semibold">{formatNumber(llmSummary?.endpoints?.total)}</div>
                    </div>
                  </div>
                </div>
                <div className="mt-3">
                  <div className="fw-semibold mb-2">Провайдеры</div>
                  <div className="d-flex flex-wrap gap-2">
                    {Object.entries(llmSummary?.endpoints?.per_provider || {}).map(([provider, count]) => (
                      <span key={provider} className="badge bg-secondary">
                        {provider}: {formatNumber(count)}
                      </span>
                    ))}
                    {Object.keys(llmSummary?.endpoints?.per_provider || {}).length === 0 && (
                      <span className="text-body-secondary small">Нет данных по провайдерам</span>
                    )}
                  </div>
                </div>
                <div className="mt-3 border-top pt-3">
                  <div className="fw-semibold mb-2">Задержки</div>
                  <dl className="row mb-0 small">
                    <dt className="col-sm-6">Полный цикл</dt>
                    <dd className="col-sm-6">{formatMs(llmSummary?.latency_ms?.total_avg_ms)}</dd>
                    <dt className="col-sm-6">Ответ LLM</dt>
                    <dd className="col-sm-6">{formatMs(llmSummary?.latency_ms?.answer_avg_ms)}</dd>
                    <dt className="col-sm-6">LLM сниппеты</dt>
                    <dd className="col-sm-6">{formatMs(llmSummary?.latency_ms?.snippet_avg_ms)}</dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>

          <div className="row g-3">
            <div className="col-12 col-lg-6">
              <div className="card h-100 p-3">
                <h2 className="h5 mb-3">Активность пользователей</h2>
                <div className="d-flex flex-wrap gap-3">
                  <div className="border rounded-3 p-3 text-center flex-grow-1">
                    <div className="text-body-secondary small text-uppercase mb-1">Всего</div>
                    <div className="fw-semibold fs-5">{formatNumber(userActivity?.total)}</div>
                  </div>
                  <div className="border rounded-3 p-3 text-center flex-grow-1">
                    <div className="text-body-secondary small text-uppercase mb-1">За 24 часа</div>
                    <div className="fw-semibold fs-5">{formatNumber(userActivity?.active_24h)}</div>
                  </div>
                </div>
                <div className="mt-3 border-top pt-3">
                  <div className="fw-semibold mb-2">Недавние входы</div>
                  <div className="table-responsive">
                    <table className="table table-sm align-middle mb-0">
                      <thead>
                        <tr>
                          <th>Пользователь</th>
                          <th>Время</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(userActivity?.recent || []).map(entry => (
                          <tr key={`${entry.user_id}-${entry.at}`}>
                            <td>
                              <div className="fw-semibold">{entry.username || 'Неизвестно'}</div>
                              <div className="text-body-secondary small">{entry.full_name || '—'}</div>
                            </td>
                            <td>{formatDateTime(entry.at)}</td>
                          </tr>
                        ))}
                        {(userActivity?.recent || []).length === 0 && (
                          <tr>
                            <td colSpan={2} className="text-center text-body-secondary small">Нет данных о входах</td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
            <div className="col-12 col-lg-6">
              <div className="card h-100 p-3">
                <h2 className="h5 mb-3">Интеграции</h2>
                <div className="d-grid gap-2">
                  <div className="border rounded-3 p-3">
                    <div className="fw-semibold mb-1">Браузер</div>
                    <div className="text-body-secondary small">
                      Headless: {formatBool(integrations?.browser?.headless, { truthy: 'Да', falsy: 'Нет' })} · Прокси:{' '}
                      {integrations?.browser?.proxy || '—'}
                      <br />
                      Окно: {formatViewport(integrations?.browser?.viewport)} · Навигация:{' '}
                      {formatMs(integrations?.browser?.navigation_timeout_ms)}
                    </div>
                  </div>
                  <div className="border rounded-3 p-3">
                    <div className="fw-semibold mb-1">HTTP клиент</div>
                    <div className="text-body-secondary small">
                      Таймаут: {formatSeconds(integrations?.http?.timeout)} · Подкл.:{' '}
                      {formatSeconds(integrations?.http?.connect_timeout)} · Повторы:{' '}
                      {formatNumber(integrations?.http?.retries)}
                      <br />
                      Backoff: {integrations?.http?.backoff_factor ?? '—'}
                    </div>
                  </div>
                  <div className="border rounded-3 p-3">
                    <div className="fw-semibold mb-1">OSINT</div>
                    <div className="text-body-secondary small">
                      Кэш: {formatBool(integrations?.osint?.cache_enabled, { truthy: 'Вкл', falsy: 'Выкл' })} · TTL:{' '}
                      {formatSeconds(integrations?.osint?.cache_ttl_seconds)}
                      <br />
                      Ротация UA: {formatNumber(integrations?.osint?.retry_user_agents)} · Прокси:{' '}
                      {formatNumber(integrations?.osint?.retry_proxies)}
                      <br />
                      Навигация: {formatMs(integrations?.osint?.navigation_timeout_ms)} · Очередь:{' '}
                      {formatNumber(integrations?.osint?.queue?.queued)}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
