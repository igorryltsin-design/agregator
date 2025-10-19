import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useAuth } from '../ui/Auth'

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
    disk_usage?: {
      total_bytes?: number | null
      used_bytes?: number | null
      free_bytes?: number | null
      total_pretty?: string | null
      used_pretty?: string | null
      free_pretty?: string | null
    }
  }
  services?: Record<string, {
    enabled?: boolean
    running?: boolean
    interval_hours?: number | null
  }>
  rag?: {
    ready?: number | null
    pending?: number | null
  }
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

export default function AdminServiceStatusPage() {
  const { user } = useAuth()
  const isAdmin = user?.role === 'admin'
  const [data, setData] = useState<StatusPayload | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [autoRefresh, setAutoRefresh] = useState(true)

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

  return (
    <div className="d-grid gap-3" aria-busy={loading}>
      <div className="card p-3">
        <div className="d-flex flex-column flex-lg-row gap-3 justify-content-between align-items-start align-items-lg-center">
          <div>
            <div className="d-flex align-items-center gap-2">
              <h1 className="h4 mb-0">Состояние сервиса</h1>
              <span className={`badge ${activeBadgeClass}`} aria-live="polite">
                {data?.status === 'ok' ? 'Норма' : data?.status === 'degraded' ? 'Есть предупреждения' : data?.status === 'error' ? 'Ошибка' : (data?.status || 'Нет данных')}
              </span>
            </div>
            <div className="text-body-secondary small mt-1">
              Последнее обновление: {formatDateTime(data?.generated_at || data?.app?.created_at)}
              {data?.app?.uptime_human ? ` · Аптайм: ${data.app.uptime_human}` : ''}
            </div>
            {data?.app?.environment && (
              <div className="text-body-secondary small">
                Среда: {data.app.environment}{data.app.debug ? ' (debug)' : ''}
              </div>
            )}
          </div>
          <div className="d-flex flex-column flex-sm-row gap-2 align-items-stretch">
            <label className="form-check form-switch m-0 align-self-center">
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
              className="btn btn-outline-secondary"
              onClick={() => loadStatus()}
              disabled={loading}
            >
              {loading ? 'Обновление…' : 'Обновить'}
            </button>
          </div>
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
                  <dt className="col-sm-5">LLM провайдер</dt>
                  <dd className="col-sm-7">{data?.runtime?.lm_default_provider || '—'}</dd>
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
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
