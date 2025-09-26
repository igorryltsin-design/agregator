import React, { useCallback, useEffect, useState } from 'react'
import { useAuth } from '../ui/Auth'
import { useToasts } from '../ui/Toasts'

type ActionLog = {
  id: number
  user_id: number | null
  username?: string | null
  full_name?: string | null
  action: string
  entity?: string | null
  entity_id?: number | null
  detail?: string | null
  created_at?: string | null
}

type SystemLogFile = {
  name: string
  size?: number
  modified_at?: string
  rotated?: boolean
}

export default function AdminLogsPage() {
  const { user } = useAuth()
  const toasts = useToasts()
  const isAdmin = user?.role === 'admin'
  const [logs, setLogs] = useState<ActionLog[]>([])
  const [loading, setLoading] = useState(false)
  const [userFilter, setUserFilter] = useState('')
  const [actionFilter, setActionFilter] = useState('')
  const [deleteBefore, setDeleteBefore] = useState('')
  const [deleting, setDeleting] = useState(false)
  const [systemLogFiles, setSystemLogFiles] = useState<SystemLogFile[]>([])
  const [systemLogName, setSystemLogName] = useState('agregator.log')
  const [systemLogLimit, setSystemLogLimit] = useState(200)
  const [systemLogLines, setSystemLogLines] = useState<string[]>([])
  const [systemLogFile, setSystemLogFile] = useState<SystemLogFile | null>(null)
  const [systemLogLoading, setSystemLogLoading] = useState(false)
  const [systemLogAction, setSystemLogAction] = useState<'clear' | 'rotate' | null>(null)

  const load = useCallback(async () => {
    if (!isAdmin) return
    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (userFilter.trim()) params.set('user_id', userFilter.trim())
      if (actionFilter.trim()) params.set('action', actionFilter.trim())
      const r = await fetch(`/api/admin/actions?${params.toString()}`)
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok) {
        setLogs(Array.isArray(data.actions) ? data.actions : [])
      } else {
        toasts.push(data?.error || 'Не удалось получить журнал действий', 'error')
      }
    } catch {
      toasts.push('Ошибка соединения при получении журнала', 'error')
    } finally {
      setLoading(false)
    }
  }, [isAdmin, userFilter, actionFilter, toasts])

  const handleDelete = useCallback(async () => {
    if (!isAdmin || !deleteBefore.trim()) {
      return
    }
    setDeleting(true)
    try {
      const body = { before: deleteBefore.trim() }
      const r = await fetch('/api/admin/actions', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok) {
        toasts.push(`Удалено записей: ${data.deleted ?? 0}`, 'success')
        setDeleteBefore('')
        await load()
      } else {
        toasts.push(data?.error || 'Не удалось удалить записи', 'error')
      }
    } catch {
      toasts.push('Ошибка соединения при удалении', 'error')
    } finally {
      setDeleting(false)
    }
  }, [deleteBefore, isAdmin, load, toasts])

  useEffect(() => { load() }, [load])

  const loadSystemLog = useCallback(async (name: string, limit: number) => {
    if (!isAdmin) return
    setSystemLogLoading(true)
    try {
      const params = new URLSearchParams()
      if (name) params.set('name', name)
      params.set('limit', String(limit))
      const r = await fetch(`/api/admin/system-logs?${params.toString()}`)
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok) {
        const files = (Array.isArray(data.available) ? data.available : []) as SystemLogFile[]
        setSystemLogFiles(files)
        if (typeof data.limit === 'number' && data.limit !== limit) {
          setSystemLogLimit(data.limit)
        }
        let resolvedName = typeof data?.file?.name === 'string' && data.file.name ? data.file.name : name
        if (!files.some(f => f.name === resolvedName) && files.length) {
          resolvedName = files[0].name
        }
        if (resolvedName !== name) {
          setSystemLogName(resolvedName)
        }
        const selectedMeta = data?.file ? data.file as SystemLogFile : files.find(f => f.name === resolvedName) || null
        setSystemLogFile(selectedMeta)
        setSystemLogLines(Array.isArray(data.lines) ? data.lines : [])
      } else {
        toasts.push(data?.error || 'Не удалось получить системный лог', 'error')
      }
    } catch {
      toasts.push('Ошибка соединения при чтении системного лога', 'error')
    } finally {
      setSystemLogLoading(false)
    }
  }, [isAdmin, toasts])

  useEffect(() => {
    if (!isAdmin) return
    loadSystemLog(systemLogName, systemLogLimit)
  }, [isAdmin, loadSystemLog, systemLogLimit, systemLogName])

  const refreshSystemLog = useCallback(() => {
    loadSystemLog(systemLogName, systemLogLimit)
  }, [loadSystemLog, systemLogLimit, systemLogName])

  const clearSystemLog = useCallback(async () => {
    if (!isAdmin) return
    if (!confirm('Очистить текущий лог-файл? Записи будут удалены.')) return
    setSystemLogAction('clear')
    try {
      const r = await fetch(`/api/admin/system-logs?name=${encodeURIComponent(systemLogName)}`, { method: 'DELETE' })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok) {
        const files = Array.isArray(data.files) ? data.files : []
        setSystemLogFiles(files as SystemLogFile[])
        setSystemLogLines([])
        toasts.push('Лог очищен', 'success')
      } else {
        toasts.push(data?.error || 'Не удалось очистить лог', 'error')
      }
    } catch {
      toasts.push('Ошибка соединения при очистке лога', 'error')
    } finally {
      setSystemLogAction(null)
      loadSystemLog(systemLogName, systemLogLimit)
    }
  }, [isAdmin, loadSystemLog, systemLogLimit, systemLogName, toasts])

  const rotateSystemLog = useCallback(async () => {
    if (!isAdmin) return
    setSystemLogAction('rotate')
    try {
      const r = await fetch('/api/admin/system-logs/rotate', { method: 'POST' })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok) {
        const files = Array.isArray(data.files) ? data.files : []
        setSystemLogFiles(files as SystemLogFile[])
        toasts.push('Создан новый лог-файл', 'success')
        loadSystemLog(systemLogName, systemLogLimit)
      } else {
        toasts.push(data?.error || 'Не удалось выполнить ротацию', 'error')
      }
    } catch {
      toasts.push('Ошибка соединения при ротации логов', 'error')
    } finally {
      setSystemLogAction(null)
    }
  }, [isAdmin, loadSystemLog, systemLogLimit, systemLogName, toasts])

  const downloadSystemLog = useCallback(() => {
    const url = `/api/admin/system-logs/download?name=${encodeURIComponent(systemLogName)}`
    window.open(url, '_blank')
  }, [systemLogName])

  const formatBytes = useCallback((value?: number) => {
    if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
    if (value < 1024) return `${value} Б`
    const kb = value / 1024
    if (kb < 1024) return `${kb.toFixed(1)} КБ`
    const mb = kb / 1024
    if (mb < 1024) return `${mb.toFixed(1)} МБ`
    const gb = mb / 1024
    return `${gb.toFixed(2)} ГБ`
  }, [])

  const formatLogTime = useCallback((value?: string | null) => {
    if (!value) return '—'
    try {
      return new Date(value).toLocaleString()
    } catch {
      return value
    }
  }, [])

  const systemLogOptions = systemLogFiles.length ? systemLogFiles : [{ name: systemLogName }]

  useEffect(() => {
    const match = systemLogFiles.find(f => f.name === systemLogName)
    if (match) {
      setSystemLogFile(match)
    }
  }, [systemLogFiles, systemLogName])

  const flattenDetail = useCallback((value: any, prefix = '', acc: string[] = []) => {
    if (value === null || value === undefined) {
      const label = prefix || 'значение'
      acc.push(`${label}: нет данных`)
      return acc
    }
    if (typeof value === 'object') {
      if (Array.isArray(value)) {
        if (!value.length) {
          const label = prefix || 'значение'
          acc.push(`${label}: []`)
        } else {
          value.forEach((item, idx) => {
            const nextPrefix = prefix ? `${prefix}[${idx}]` : `[${idx}]`
            flattenDetail(item, nextPrefix, acc)
          })
        }
      } else {
        const entries = Object.entries(value)
        if (!entries.length) {
          const label = prefix || 'значение'
          acc.push(`${label}: {}`)
        } else {
          entries.forEach(([k, v]) => {
            const nextPrefix = prefix ? `${prefix}.${k}` : k
            flattenDetail(v, nextPrefix, acc)
          })
        }
      }
      return acc
    }
    const label = prefix || 'значение'
    acc.push(`${label}: ${String(value)}`)
    return acc
  }, [])

  const formatActionName = useCallback((action: string | null | undefined) => {
    if (!action) return '—'
    const dictionary: Record<string, string> = {
      user: 'пользователь',
      create: 'создание',
      update: 'обновление',
      delete: 'удаление',
      login: 'вход',
      logout: 'выход',
      file: 'файл',
      upload: 'загрузка',
      collection: 'коллекция',
      assign: 'назначение',
      revoke: 'отмена',
      ai: 'ИИ',
      search: 'поиск',
      settings: 'настройки',
      role: 'роль',
    }
    const parts = action.replace(/[:]/g, '_').split('_').filter(Boolean)
    if (!parts.length) return action
    const translated = parts.map(part => dictionary[part.toLowerCase()] || part)
    const label = translated.join(' ')
    return label.charAt(0).toUpperCase() + label.slice(1)
  }, [])

  const formatDetail = useCallback((detail: string | null) => {
    if (!detail) return '—'
    try {
      const parsed = JSON.parse(detail)
      const lines = flattenDetail(parsed)
      return lines.join('\n') || '—'
    } catch {
      return detail.replace(/[{}]/g, '')
    }
  }, [flattenDetail])

  if (!isAdmin) return <div className="card p-3">Недостаточно прав.</div>

  return (
    <div className="d-grid gap-3">
      <div className="card p-3">
        <div className="d-flex flex-wrap align-items-center gap-2 mb-3">
          <div className="fw-semibold fs-5 me-auto">Системный лог</div>
          <button className="btn btn-outline-secondary btn-sm" onClick={refreshSystemLog} disabled={systemLogLoading}>{systemLogLoading ? 'Загрузка…' : 'Обновить'}</button>
          <button className="btn btn-outline-secondary btn-sm" onClick={rotateSystemLog} disabled={systemLogAction === 'rotate' || systemLogLoading}>
            {systemLogAction === 'rotate' ? 'Ротация…' : 'Ротация'}
          </button>
          <button className="btn btn-outline-danger btn-sm" onClick={clearSystemLog} disabled={systemLogAction === 'clear' || systemLogLoading}>
            {systemLogAction === 'clear' ? 'Очистка…' : 'Очистить'}
          </button>
          <button className="btn btn-outline-primary btn-sm" onClick={downloadSystemLog} disabled={systemLogLoading || !systemLogFile}>
            Скачать
          </button>
        </div>
        <div className="row g-3 align-items-end mb-3">
          <div className="col-md-5">
            <label className="form-label">Файл</label>
            <select
              className="form-select form-select-sm"
              value={systemLogName}
              onChange={e => setSystemLogName(e.target.value)}
            >
              {systemLogOptions.map(file => {
                const label = file.rotated ? `${file.name} (архив)` : file.name
                return <option key={file.name} value={file.name}>{label}</option>
              })}
            </select>
          </div>
          <div className="col-md-2">
            <label className="form-label">Строк</label>
            <select
              className="form-select form-select-sm"
              value={systemLogLimit}
              onChange={e => setSystemLogLimit(Number(e.target.value) || 200)}
            >
              {[100, 200, 500, 1000].map(option => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
          <div className="col-md-5">
            <div className="form-text">
              Размер: {formatBytes(systemLogFile?.size)} · Обновлено: {formatLogTime(systemLogFile?.modified_at)}
            </div>
          </div>
        </div>
        <div className="border rounded bg-body-secondary" style={{ maxHeight: 320, overflow: 'auto' }}>
          {systemLogLoading ? (
            <div className="text-center text-muted py-4" style={{ fontSize: 13 }}>Загрузка…</div>
          ) : systemLogLines.length ? (
            <pre className="m-0 p-2" style={{ fontSize: 12, whiteSpace: 'pre-wrap' }}>{systemLogLines.join('\n')}</pre>
          ) : (
            <div className="text-center text-muted py-4" style={{ fontSize: 13 }}>Лог пуст</div>
          )}
        </div>
      </div>

      <div className="card p-3">
        <div className="d-flex flex-wrap gap-2 align-items-end mb-3">
          <div className="flex-grow-1" style={{ maxWidth: 200 }}>
            <label className="form-label">Пользователь ID</label>
            <input className="form-control form-control-sm" value={userFilter} onChange={e=>setUserFilter(e.target.value)} placeholder="пример: 2" />
          </div>
          <div className="flex-grow-1" style={{ maxWidth: 220 }}>
          <label className="form-label">Действие</label>
          <input className="form-control form-control-sm" value={actionFilter} onChange={e=>setActionFilter(e.target.value)} placeholder="например user_update" />
        </div>
        <div className="d-flex align-items-end gap-2">
          <button className="btn btn-outline-secondary" onClick={load} disabled={loading}>{loading ? 'Загрузка…' : 'Обновить'}</button>
          <button className="btn btn-outline-secondary" onClick={()=>{ setUserFilter(''); setActionFilter(''); }} disabled={loading}>Сбросить</button>
        </div>
        <div className="d-flex align-items-end gap-2" style={{ maxWidth: 280 }}>
          <div className="flex-grow-1">
            <label className="form-label">Удалить записи до даты</label>
            <input
              type="date"
              className="form-control form-control-sm"
              value={deleteBefore}
              onChange={e => setDeleteBefore(e.target.value)}
            />
          </div>
          <button
            className="btn btn-outline-danger"
            onClick={handleDelete}
            disabled={deleting || !deleteBefore.trim()}
          >
            {deleting ? 'Удаление…' : 'Удалить'}
          </button>
        </div>
      </div>
      <div className="table-responsive">
        <table className="table table-sm align-middle">
          <thead>
            <tr>
              <th>ID</th>
              <th>Пользователь</th>
              <th>Действие</th>
              <th>Объект</th>
              <th>Детали</th>
              <th>Время</th>
            </tr>
          </thead>
          <tbody>
            {logs.map(log => (
              <tr key={log.id}>
                <td>{log.id}</td>
                <td>
                  {log.username ? (
                    <span>{log.username}{log.full_name ? ` (${log.full_name})` : ''}</span>
                  ) : (log.user_id ?? '—')}
                </td>
                <td><span className="badge bg-secondary" title={log.action}>{formatActionName(log.action)}</span></td>
                <td>{log.entity ? `${log.entity}${log.entity_id ? `#${log.entity_id}` : ''}` : '—'}</td>
                <td>
                  <pre style={{ fontSize: 12, margin: 0, whiteSpace: 'pre-wrap' }}>{formatDetail(log.detail)}</pre>
                </td>
                <td>{log.created_at ? new Date(log.created_at).toLocaleString() : '—'}</td>
              </tr>
            ))}
            {logs.length === 0 && (
              <tr><td colSpan={6} className="text-center text-muted py-3">Нет записей</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
    </div>
  )
}
