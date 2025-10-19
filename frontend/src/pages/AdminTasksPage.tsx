import React, { useCallback, useEffect, useState } from 'react'
import { useAuth } from '../ui/Auth'
import { useToasts } from '../ui/Toasts'
import { taskStatusRu } from '../utils/locale'
import LoadingState from '../ui/LoadingState'

type Task = {
  id: number
  name: string
  status: string
  progress: number
  payload?: string | null
  payload_json?: any
  created_at?: string | null
  started_at?: string | null
  finished_at?: string | null
  error?: string | null
}

const statusBadge: Record<string, string> = {
  running: 'bg-success',
  queued: 'bg-secondary',
  pending: 'bg-secondary',
  cancelling: 'bg-warning',
  cancelled: 'bg-warning',
  completed: 'bg-primary',
  error: 'bg-danger',
}

const FINAL_STATUSES = new Set(['completed', 'error', 'cancelled'])

function formatDate(value?: string | null): string {
  if (!value) return '—'
  try {
    return new Date(value).toLocaleString()
  } catch {
    return value
  }
}

export default function AdminTasksPage() {
  const { user } = useAuth()
  const toasts = useToasts()
  const [tasks, setTasks] = useState<Task[]>([])
  const [loading, setLoading] = useState(false)
  const [expandedTaskId, setExpandedTaskId] = useState<number | null>(null)
  const [deletingTaskId, setDeletingTaskId] = useState<number | null>(null)
  const [clearingCompleted, setClearingCompleted] = useState(false)
  const isAdmin = user?.role === 'admin'

  const load = useCallback(async () => {
    if (!isAdmin) return
    setLoading(true)
    try {
      const r = await fetch('/api/admin/tasks')
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok) {
        const list: Task[] = Array.isArray(data.tasks)
          ? data.tasks.map((task: any) => ({
              ...task,
              payload_json: task.payload_json ?? undefined
            }))
          : []
        setTasks(list)
        setExpandedTaskId(prev => (prev !== null && !list.some(t => t.id === prev) ? null : prev))
      } else {
        toasts.push(data?.error || 'Не удалось получить список задач', 'error')
      }
    } catch {
      toasts.push('Ошибка соединения при получении задач', 'error')
    } finally {
      setLoading(false)
    }
  }, [isAdmin, toasts])

  useEffect(() => { load() }, [load])

  useEffect(() => {
    const timer = window.setInterval(load, 5000)
    return () => window.clearInterval(timer)
  }, [load])

  const cancel = async (task: Task) => {
    try {
      let ok = false
      if (task.name === 'scan') {
        const r = await fetch('/scan/cancel', { method: 'POST' })
        ok = r.ok
        if (!ok) {
          const txt = await r.text().catch(() => '')
          toasts.push(txt || 'Не удалось остановить сканирование', 'error')
          return
        }
      } else {
        const r = await fetch(`/api/admin/tasks/${task.id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status: 'cancel' })
        })
        const data = await r.json().catch(() => ({}))
        ok = r.ok && data?.ok
        if (!ok) {
          toasts.push(data?.error || 'Не удалось остановить задачу', 'error')
          return
        }
      }
      if (ok) {
        toasts.push('Запрошена остановка задачи', 'success')
        load()
      }
    } catch {
      toasts.push('Ошибка соединения при остановке задачи', 'error')
    }
  }

  const remove = async (task: Task) => {
    if (!FINAL_STATUSES.has(task.status)) {
      toasts.push('Удалять можно только завершённые, отменённые или с ошибкой задачи', 'info')
      return
    }
    if (!confirm('Удалить запись о задаче? Действие необратимо.')) return
    setDeletingTaskId(task.id)
    try {
      const r = await fetch(`/api/admin/tasks/${task.id}`, { method: 'DELETE' })
      if (r.ok) {
        setTasks(prev => prev.filter(t => t.id !== task.id))
        toasts.push('Задача удалена из журнала', 'success')
      } else {
        const data = await r.json().catch(() => ({}))
        toasts.push(data?.error || 'Не удалось удалить задачу', 'error')
      }
    } catch {
      toasts.push('Ошибка соединения при удалении задачи', 'error')
    } finally {
      setDeletingTaskId(prev => (prev === task.id ? null : prev))
    }
  }

  const clearCompleted = async () => {
    if (!tasks.some(t => t.status === 'completed')) {
      toasts.push('Нет задач со статусом "выполнено"', 'info')
      return
    }
    if (!confirm('Удалить все завершённые задачи из списка?')) return
    setClearingCompleted(true)
    try {
      const r = await fetch('/api/admin/tasks?status=completed', { method: 'DELETE' })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok) {
        const removed = Number(data.deleted || 0)
        setTasks(prev => prev.filter(t => t.status !== 'completed'))
        toasts.push(`Удалено завершённых задач: ${removed}`, removed ? 'success' : 'info')
      } else {
        toasts.push(data?.error || 'Не удалось очистить список', 'error')
      }
    } catch {
      toasts.push('Ошибка соединения при очистке списка', 'error')
    } finally {
      setClearingCompleted(false)
    }
  }

  const toggleDetails = (id: number) => {
    setExpandedTaskId(prev => (prev === id ? null : id))
  }

  const formatPayload = (task: Task) => {
    if (task.payload_json) {
      return JSON.stringify(task.payload_json, null, 2)
    }
    if (!task.payload) return '—'
    try {
      const parsed = JSON.parse(task.payload)
      return JSON.stringify(parsed, null, 2)
    } catch {
      return task.payload
    }
  }

  if (!isAdmin) {
    return <div className="card p-3">Недостаточно прав.</div>
  }

  return (
    <div className="d-grid gap-3">
      <div className="card p-3" aria-busy={loading}>
        <div className="d-flex justify-content-between align-items-center mb-3">
          <div className="fw-semibold fs-5">Очередь задач</div>
          <div className="d-flex gap-2">
            <button
              className="btn btn-outline-danger btn-sm"
              onClick={clearCompleted}
              disabled={clearingCompleted || loading}
            >
              {clearingCompleted ? 'Очистка…' : 'Очистить завершённые'}
            </button>
            <button className="btn btn-outline-secondary btn-sm" onClick={load} disabled={loading}>{loading ? 'Обновление…' : 'Обновить'}</button>
          </div>
        </div>
        <div className="table-responsive">
          <table className="table table-sm align-middle">
            <thead>
              <tr>
                <th>ID</th>
                <th>Имя</th>
                <th>Статус</th>
                <th>Прогресс</th>
                <th>Создана</th>
                <th>Старт</th>
                <th>Финиш</th>
                <th>Действия</th>
              </tr>
            </thead>
            <tbody aria-busy={loading}>
              {loading && tasks.length === 0 && (
                <tr>
                  <td colSpan={8}>
                    <LoadingState variant="inline" lines={4} />
                  </td>
                </tr>
              )}
              {tasks.map(task => {
                const expanded = expandedTaskId === task.id
                const pct = Number.isFinite(task.progress) ? Math.min(100, Math.max(0, Math.round(task.progress * 100))) : 0
                const isFinal = FINAL_STATUSES.has(task.status)
                const preview = task.payload_json?.result?.preview || task.payload_json?.initial_preview
                const payloadError: string | undefined = task.error || task.payload_json?.error || undefined
                return (
                  <React.Fragment key={task.id}>
                    <tr>
                      <td>{task.id}</td>
                      <td>{task.name}</td>
                      <td>
                        <span className={`badge ${statusBadge[task.status] || 'bg-secondary'}`}>{taskStatusRu(task.status)}</span>
                        {payloadError && <div className="text-danger" style={{ fontSize: 12 }}>{payloadError}</div>}
                      </td>
                      <td style={{ minWidth: 160 }}>
                        <div className="progress" style={{ height: 6 }}>
                          <div className="progress-bar" role="progressbar" style={{ width: `${pct}%` }} aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}></div>
                        </div>
                        <div className="text-muted" style={{ fontSize: 12 }}>{pct}%</div>
                      </td>
                      <td>{formatDate(task.created_at)}</td>
                      <td>{formatDate(task.started_at)}</td>
                      <td>{formatDate(task.finished_at)}</td>
                      <td>
                        <div className="btn-group btn-group-sm">
                          <button className="btn btn-outline-secondary" onClick={() => toggleDetails(task.id)}>
                            {expanded ? 'Скрыть' : 'Подробнее'}
                          </button>
                          <button
                            className="btn btn-outline-danger"
                            onClick={() => cancel(task)}
                            disabled={isFinal || deletingTaskId === task.id}
                          >
                            Остановить
                          </button>
                          <button
                            className="btn btn-danger"
                            onClick={() => remove(task)}
                            disabled={!isFinal || deletingTaskId === task.id}
                          >
                            {deletingTaskId === task.id ? 'Удаление…' : 'Удалить'}
                          </button>
                        </div>
                      </td>
                    </tr>
                    {expanded && (
                      <tr>
                        <td colSpan={8}>
                          <div className="p-3 bg-body-secondary rounded">
                            <div className="fw-semibold mb-2">Подробности задачи</div>
                            <div className="row g-3">
                              <div className="col-md-4">
                                <div className="text-muted" style={{ fontSize: 12 }}>Создана</div>
                                <div>{formatDate(task.created_at)}</div>
                              </div>
                              <div className="col-md-4">
                                <div className="text-muted" style={{ fontSize: 12 }}>Старт</div>
                                <div>{formatDate(task.started_at)}</div>
                              </div>
                              <div className="col-md-4">
                                <div className="text-muted" style={{ fontSize: 12 }}>Завершена</div>
                                <div>{formatDate(task.finished_at)}</div>
                              </div>
                            </div>
                            const preview = task.payload_json?.result?.preview || task.payload_json?.initial_preview
                            {payloadError && (
                              <div className="mt-3 text-danger">
                                <div className="text-muted" style={{ fontSize: 12 }}>Ошибка</div>
                                <div>{payloadError}</div>
                              </div>
                            )}
                            {!payloadError && preview && (
                              <div className="mt-3">
                                <div className="text-muted" style={{ fontSize: 12 }}>Превью результата</div>
                                <div className="small text-muted">
                                  {preview.filename && <div>Файл: {preview.filename}</div>}
                                  {preview.collection_name && <div>Коллекция: {preview.collection_name}</div>}
                                  {preview.material_type && <div>Тип: {preview.material_type}</div>}
                                  {preview.language && <div>Язык: {preview.language}</div>}
                                  {preview.title && <div>Заголовок: {preview.title}</div>}
                                </div>
                              </div>
                            )}
                            <div className="mt-3">
                              <div className="text-muted" style={{ fontSize: 12 }}>Данные задачи</div>
                              <pre className="bg-dark-subtle p-2 rounded" style={{ maxHeight: 240, overflow: 'auto', fontSize: 12 }}>{formatPayload(task)}</pre>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                )
              })}
              {tasks.length === 0 && (
                <tr><td colSpan={8} className="text-center text-muted py-3">Задачи не найдены</td></tr>
              )}
            </tbody>
          </table>
        </div>
        {tasks.some(t => t.payload || t.payload_json) && (
          <details className="mt-3">
            <summary className="fw-semibold">Сырые данные</summary>
            <pre className="bg-body-secondary p-2 rounded" style={{ maxHeight: 240, overflow: 'auto' }}>{JSON.stringify(tasks, null, 2)}</pre>
          </details>
        )}
      </div>
    </div>
  )
}
