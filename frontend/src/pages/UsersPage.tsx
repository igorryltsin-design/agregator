import React, { useEffect, useState } from 'react'
import { useAuth } from '../ui/Auth'
import { useToasts } from '../ui/Toasts'

type RoleId = 'admin' | 'editor' | 'viewer'

type UserRow = {
  id: number
  username: string
  role: RoleId
  full_name?: string | null
  created_at?: string | null
}

type UserForm = { username: string; full_name: string; password: string; role: RoleId }

type RoleOption = { id: RoleId; label: string }

export default function UsersPage(){
  const { user } = useAuth()
  const isAdmin = user?.role === 'admin'
  const toasts = useToasts()
  const [rows, setRows] = useState<UserRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [form, setForm] = useState<UserForm>({ username: '', full_name: '', password: '', role: 'editor' })
  const [saving, setSaving] = useState(false)
  const [roleOptions, setRoleOptions] = useState<RoleOption[]>([
    { id: 'admin', label: 'Администратор' },
    { id: 'editor', label: 'Редактор' },
    { id: 'viewer', label: 'Наблюдатель' },
  ])

  const load = async () => {
    if (!isAdmin) return
    setLoading(true)
    setError('')
    try {
      const r = await fetch('/api/users')
      if (r.status === 403) {
        setError('Недостаточно прав для просмотра пользователей')
        setRows([])
        return
      }
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok && Array.isArray(data.users)) {
        setRows(data.users)
      } else {
        setError(data?.error || 'Не удалось получить список пользователей')
        setRows([])
      }
    } catch {
      setError('Ошибка соединения с сервером')
      setRows([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [isAdmin])
  useEffect(() => {
    if (!roleOptions.some(opt => opt.id === form.role)) {
      setForm(prev => ({ ...prev, role: roleOptions[0]?.id ?? 'editor' }))
    }
  }, [roleOptions, form.role])
  useEffect(() => {
    if (!isAdmin) return
    (async () => {
      try {
        const r = await fetch('/api/users/roles')
        if (!r.ok) return
        const data = await r.json().catch(() => ({}))
        if (Array.isArray(data?.roles)) {
          const opts: RoleOption[] = data.roles
            .map((item: any) => ({ id: item.id as RoleId, label: item.label || item.id }))
            .filter(opt => opt.id)
          if (opts.length) setRoleOptions(opts)
        }
      } catch {
        /* Заглушка: ничего не делаем */
      }
    })()
  }, [isAdmin])

  if (!isAdmin) {
    return <div className="card p-3">Недостаточно прав.</div>
  }

  const createUser = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!form.username.trim() || !form.password.trim() || !form.full_name.trim()) {
      setError('Укажите логин, ФИО и пароль')
      return
    }
    if (form.full_name.trim().length < 5) {
      setError('ФИО должно содержать минимум 5 символов')
      return
    }
    setSaving(true)
    setError('')
    try {
      const r = await fetch('/api/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: form.username.trim(), full_name: form.full_name.trim(), password: form.password, role: form.role }),
      })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok && data.user) {
        setRows(prev => [...prev, data.user])
        setForm({ username: '', full_name: '', password: '', role: roleOptions[0]?.id ?? 'editor' })
        toasts.push('Пользователь создан', 'success')
      } else {
        setError(data?.error || 'Не удалось создать пользователя')
      }
    } catch {
      setError('Ошибка соединения при создании пользователя')
    } finally {
      setSaving(false)
    }
  }

  const changeRole = async (id: number, role: RoleId) => {
    try {
      const r = await fetch(`/api/users/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role }),
      })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok && data.user) {
        setRows(prev => prev.map(u => u.id === id ? data.user : u))
        toasts.push('Роль обновлена', 'success')
      } else {
        toasts.push(data?.error || 'Не удалось изменить роль', 'error')
      }
    } catch {
      toasts.push('Ошибка соединения при изменении роли', 'error')
    }
  }

  const resetPassword = async (id: number, username: string) => {
    const pw = prompt(`Введите новый пароль для ${username}`)
    if (!pw) return
    if (pw.length < 6) {
      alert('Пароль должен содержать минимум 6 символов')
      return
    }
    try {
      const r = await fetch(`/api/users/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: pw }),
      })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok) {
        toasts.push('Пароль обновлён', 'success')
      } else {
        toasts.push(data?.error || 'Не удалось обновить пароль', 'error')
      }
    } catch {
      toasts.push('Ошибка соединения при обновлении пароля', 'error')
    }
  }

  const removeUser = async (id: number, username: string) => {
    if (!confirm(`Удалить пользователя ${username}?`)) return
    try {
      const r = await fetch(`/api/users/${id}`, { method: 'DELETE' })
      if (r.ok) {
        setRows(prev => prev.filter(u => u.id !== id))
        toasts.push('Пользователь удалён', 'success')
      } else {
        const data = await r.json().catch(() => ({}))
        toasts.push(data?.error || 'Не удалось удалить пользователя', 'error')
      }
    } catch {
      toasts.push('Ошибка соединения при удалении пользователя', 'error')
    }
  }

  return (
    <div className="d-grid gap-3">
      <div className="card p-3">
        <div className="fw-semibold mb-2">Создать пользователя</div>
        <form className="row g-3 align-items-end" onSubmit={createUser}>
          <div className="col-12 col-lg-3">
            <label className="form-label">Логин</label>
            <input className="form-control" value={form.username} onChange={e=>setForm({...form, username:e.target.value})} autoComplete="off" />
          </div>
          <div className="col-12 col-lg-4">
            <label className="form-label">ФИО</label>
            <input className="form-control" value={form.full_name} onChange={e=>setForm({...form, full_name:e.target.value})} autoComplete="off" />
          </div>
          <div className="col-12 col-lg-3">
            <label className="form-label">Пароль</label>
            <input className="form-control" type="password" value={form.password} onChange={e=>setForm({...form, password:e.target.value})} autoComplete="new-password" />
          </div>
          <div className="col-6 col-lg-2">
            <label className="form-label">Роль</label>
            <select className="form-select" value={form.role} onChange={e=>setForm({...form, role: e.target.value as RoleId})}>
              {roleOptions.map(opt => (
                <option key={opt.id} value={opt.id}>{opt.label}</option>
              ))}
            </select>
          </div>
          <div className="col-6 col-lg-auto d-grid">
            <button className="btn btn-primary" type="submit" disabled={saving}>{saving ? 'Создание…' : 'Создать'}</button>
          </div>
          {error && <div className="col-12"><div className="alert alert-danger py-2 m-0">{error}</div></div>}
        </form>
      </div>
      <div className="card p-3">
        <div className="fw-semibold mb-2">Список пользователей</div>
        {loading ? (
          <div>Загрузка…</div>
        ) : (
          <div className="table-responsive">
            <table className="table table-sm align-middle">
              <thead>
                <tr>
                  <th>Логин</th>
                  <th>ФИО</th>
                  <th>Роль</th>
                  <th>Создан</th>
                  <th style={{ width: 220 }}>Действия</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(u => (
                  <tr key={u.id}>
                    <td>{u.username}</td>
                    <td>{u.full_name || '—'}</td>
                    <td>
                      <select
                        className="form-select form-select-sm"
                        value={u.role}
                        onChange={e=>changeRole(u.id, e.target.value as RoleId)}
                        disabled={u.id === user.id}
                      >
                        {roleOptions.map(opt => (
                          <option key={opt.id} value={opt.id}>{opt.label}</option>
                        ))}
                      </select>
                    </td>
                    <td>{u.created_at ? new Date(u.created_at).toLocaleString() : '—'}</td>
                    <td>
                      <div className="btn-group btn-group-sm">
                        <button className="btn btn-outline-secondary" onClick={()=>resetPassword(u.id, u.username)}>Пароль</button>
                        <button className="btn btn-outline-danger" onClick={()=>removeUser(u.id, u.username)} disabled={u.id === user.id}>Удалить</button>
                      </div>
                    </td>
                  </tr>
                ))}
                {rows.length === 0 && (
                  <tr><td colSpan={4} className="text-center text-muted py-3">Пользователи не найдены</td></tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
