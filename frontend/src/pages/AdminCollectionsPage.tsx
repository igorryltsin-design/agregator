import React, { useEffect, useMemo, useState } from 'react'
import { useAuth } from '../ui/Auth'
import { useToasts } from '../ui/Toasts'

type Member = {
  user_id: number
  username?: string | null
  role: string
}

type CollectionRow = {
  id: number
  name: string
  slug?: string | null
  owner_id?: number | null
  owner_username?: string | null
  is_private?: boolean
  count?: number
  members?: Member[]
  searchable?: boolean
  graphable?: boolean
}

type UserSuggestion = {
  id: number
  username: string
  full_name?: string | null
}

const ROLE_LABELS: Record<string, string> = {
  viewer: 'Читатель',
  editor: 'Редактор',
  owner: 'Владелец',
}

export default function AdminCollectionsPage() {
  const { user } = useAuth()
  const toasts = useToasts()
  const isAdmin = user?.role === 'admin'
  const [collections, setCollections] = useState<CollectionRow[]>([])
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [members, setMembers] = useState<Member[]>([])
  const [loading, setLoading] = useState(false)
  const [rescanLoading, setRescanLoading] = useState(false)
  const [renameLoading, setRenameLoading] = useState(false)
  const [deletingCollectionId, setDeletingCollectionId] = useState<number | null>(null)
  const [clearLoading, setClearLoading] = useState(false)
  const [exportLoading, setExportLoading] = useState(false)
  const [ragLoading, setRagLoading] = useState(false)
  const [ragDeleteLoading, setRagDeleteLoading] = useState(false)
  const [memberForm, setMemberForm] = useState<{ username: string; role: string; userId: number | null }>({ username: '', role: 'viewer', userId: null })
  const [memberOptions, setMemberOptions] = useState<UserSuggestion[]>([])
  const [memberLookupLoading, setMemberLookupLoading] = useState(false)
  const [docChatCollectionId, setDocChatCollectionId] = useState<number | null>(null)
  const [collectionsSaving, setCollectionsSaving] = useState(false)
  const [newCollectionName, setNewCollectionName] = useState('')

  const loadCollections = async () => {
    setLoading(true)
    try {
      if (isAdmin) {
        const r = await fetch('/api/admin/collections')
        const data = await r.json().catch(() => ({}))
        if (r.ok && data?.ok) {
          setCollections(Array.isArray(data.collections) ? data.collections : [])
        } else {
          toasts.push(data?.error || 'Не удалось получить коллекции', 'error')
        }
      } else {
        const r = await fetch('/api/collections')
        const data = await r.json().catch(() => [])
        if (Array.isArray(data)) {
          setCollections(data)
        } else {
          toasts.push('Не удалось получить коллекции', 'error')
        }
      }
    } catch {
      toasts.push('Ошибка соединения при загрузке коллекций', 'error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadCollections() }, [isAdmin])

  useEffect(() => {
    if (!selectedId && collections.length) {
      setSelectedId(collections[0].id)
    }
  }, [collections, selectedId])

  useEffect(() => {
    if (!selectedId) {
      setMembers([])
      return
    }
    const loadMembers = async () => {
      try {
        const r = await fetch(`/api/collections/${selectedId}/members`)
        const data = await r.json().catch(() => ({}))
        if (r.ok && data?.ok) {
          setMembers(Array.isArray(data.members) ? data.members : [])
        } else {
          setMembers([])
          if (data?.error) toasts.push(data.error, 'error')
        }
      } catch {
        setMembers([])
      }
    }
    loadMembers()
  }, [selectedId, toasts])

  useEffect(() => {
    setMemberForm(prev => ({ username: '', role: prev.role, userId: null }))
    setMemberOptions([])
    setMemberLookupLoading(false)
  }, [selectedId])

  const assignedMemberIds = useMemo(() => new Set(members.map(m => m.user_id)), [members])

  useEffect(() => {
    setMemberOptions(prev => prev.filter(opt => !assignedMemberIds.has(opt.id)))
  }, [assignedMemberIds])

  useEffect(() => {
    if (!isAdmin) return
    const query = memberForm.username.trim()
    if (!query) {
      setMemberOptions([])
      setMemberLookupLoading(false)
      return
    }
    const controller = new AbortController()
    const handle = window.setTimeout(async () => {
      setMemberLookupLoading(true)
      try {
        const r = await fetch(`/api/admin/users/search?q=${encodeURIComponent(query)}&limit=8`, { signal: controller.signal })
        const data = await r.json().catch(() => ({}))
        if (!controller.signal.aborted && r.ok && data?.ok && Array.isArray(data.users)) {
          const suggestions: UserSuggestion[] = data.users
            .filter((u: any) => !assignedMemberIds.has(u.id))
            .map((u: any) => ({ id: u.id, username: u.username, full_name: u.full_name }))
          setMemberOptions(suggestions)
        } else if (!controller.signal.aborted) {
          setMemberOptions([])
        }
      } catch (err) {
        if (!(err instanceof DOMException && err.name === 'AbortError')) {
          if (!controller.signal.aborted) {
            setMemberOptions([])
          }
        }
      } finally {
        if (!controller.signal.aborted) {
          setMemberLookupLoading(false)
        }
      }
    }, 250)
    return () => {
      controller.abort()
      window.clearTimeout(handle)
    }
  }, [assignedMemberIds, isAdmin, memberForm.username])

  const updateCollectionFlag = (collectionId: number, field: 'searchable' | 'graphable', value: boolean) => {
    setCollections(prev => prev.map(col => (col.id === collectionId ? { ...col, [field]: value } : col)))
  }

  const saveCollectionSettings = async () => {
    setCollectionsSaving(true)
    try {
      const fd = new FormData()
      collections.forEach(col => {
        if (col.searchable) fd.set(`search_${col.id}`, 'on')
        if (col.graphable) fd.set(`graph_${col.id}`, 'on')
      })
      const trimmed = newCollectionName.trim()
      if (trimmed) {
        fd.set('new_name', trimmed)
      }
      const r = await fetch('/settings/collections', { method: 'POST', body: fd })
      const data = await r.json().catch(() => ({}))
      if (!r.ok || data?.status !== 'ok') {
        throw new Error(data?.error || 'Не удалось сохранить коллекции')
      }
      toasts.push('Коллекции обновлены', 'success')
      setNewCollectionName('')
      await loadCollections()
    } catch (error: any) {
      toasts.push(error?.message || 'Ошибка сохранения коллекций', 'error')
    } finally {
      setCollectionsSaving(false)
    }
  }

  const prepareCollectionForChat = async (collectionId: number) => {
    setDocChatCollectionId(collectionId)
    try {
      const resp = await fetch(`/api/doc-chat/collections/${collectionId}/prepare`, { method: 'POST' })
      const data = await resp.json().catch(() => ({}))
      if (!resp.ok || !data?.ok) {
        throw new Error(data?.error || 'Не удалось запустить обработку коллекции для чата')
      }
      const pending = typeof data.pending === 'number' ? data.pending : null
      if (pending === 0) {
        toasts.push('Все документы коллекции уже готовы для чата.', 'info')
      } else if (pending && pending > 0) {
        toasts.push(`Запущена подготовка ${pending} документов для чата.`, 'success')
      } else {
        toasts.push('Задача подготовки документов для чата создана.', 'success')
      }
    } catch (error: any) {
      toasts.push(error?.message || 'Не удалось подготовить коллекцию для чата', 'error')
    } finally {
      setDocChatCollectionId(null)
    }
  }

  const deleteCollectionById = async (collectionId: number) => {
    const target = collections.find(c => c.id === collectionId)
    if (!target) return
    if (!confirm(`Удалить коллекцию «${target.name}» и все её файлы?`)) return
    setDeletingCollectionId(collectionId)
    try {
      const r = await fetch(`/api/collections/${collectionId}`, { method: 'DELETE' })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok) {
        toasts.push('Коллекция удалена', 'success')
        setCollections(prev => prev.filter(c => c.id !== collectionId))
        if (selectedId === collectionId) {
          setSelectedId(null)
          setMembers([])
        }
      } else {
        throw new Error(data?.error || 'Не удалось удалить коллекцию')
      }
    } catch (error: any) {
      toasts.push(error?.message || 'Ошибка при удалении коллекции', 'error')
    } finally {
      setDeletingCollectionId(null)
      await loadCollections()
    }
  }

  const selectedCollection = useMemo(() => collections.find(c => c.id === selectedId) || null, [collections, selectedId])

  const rescanCollection = async () => {
    if (!selectedId) return
    if (!confirm('Пересканировать выбранную коллекцию?')) return
    setRescanLoading(true)
    try {
      const fd = new FormData()
      const r = await fetch(`/scan/collection/${selectedId}`, { method: 'POST', body: fd })
      const data = await r.json().catch(() => ({}))
      if (r.status === 409 || data?.status === 'busy') {
        toasts.push('Сканирование уже запущено', 'error')
      } else if (r.status === 404 || data?.status === 'not_found') {
        toasts.push('Коллекция не найдена', 'error')
      } else if (r.ok && data?.status === 'empty') {
        const msg = data?.missing ? `В коллекции нет файлов для сканирования (отсутствует ${data.missing}).` : 'В коллекции нет файлов для сканирования.'
        toasts.push(msg, 'info')
      } else if (r.ok && data?.status === 'started') {
        const msg = data?.missing ? `Сканирование запущено (${data.files} файлов, пропущено ${data.missing}).` : `Сканирование запущено (${data.files} файлов).`
        toasts.push(msg, 'success')
        try { window.dispatchEvent(new Event('scan-open')) } catch {}
      } else {
        toasts.push('Не удалось запустить сканирование коллекции', 'error')
      }
    } catch {
      toasts.push('Ошибка при запуске сканирования коллекции', 'error')
    } finally {
      setRescanLoading(false)
    }
  }

  const rebuildRag = async () => {
    if (!selectedId) return
    if (!confirm('Запустить переиндексацию RAG для выбранной коллекции?')) return
    setRagLoading(true)
    try {
      const r = await fetch(`/api/collections/${selectedId}/rag/reindex`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok) {
        const files = typeof data.files === 'number' ? data.files : null
        const message = files !== null ? `Задача RAG запущена (${files} файлов).` : 'Задача RAG запущена.'
        toasts.push(message, 'success')
      } else if (data?.error_code === 'collection_empty') {
        toasts.push('В коллекции нет файлов для RAG.', 'info')
      } else {
        toasts.push(data?.error || 'Не удалось запустить RAG индексацию', 'error')
      }
    } catch {
      toasts.push('Ошибка при запуске RAG индексации', 'error')
    } finally {
      setRagLoading(false)
    }
  }

  const deleteRagIndex = async () => {
    if (!selectedId) return
    if (!confirm('Удалить RAG индекс для всех файлов выбранной коллекции?')) return
    setRagDeleteLoading(true)
    try {
      const r = await fetch(`/api/collections/${selectedId}/rag/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok) {
        const message = typeof data.message === 'string' ? data.message : 'RAG индекс удалён.'
        toasts.push(message, data?.removed_documents ? 'success' : 'info')
      } else {
        toasts.push(data?.error || 'Не удалось удалить RAG индекс', 'error')
      }
    } catch {
      toasts.push('Ошибка при удалении RAG индекса', 'error')
    } finally {
      setRagDeleteLoading(false)
    }
  }

  const renameAll = async () => {
    if (!selectedId) return
    if (!confirm('Переименовать все файлы в выбранной коллекции?')) return
    setRenameLoading(true)
    try {
      const r = await fetch(`/api/collections/${selectedId}/rename-all`, { method: 'POST' })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok) {
        const renamed = Number(data.renamed || 0)
        const errors = Array.isArray(data.errors) ? data.errors.length : 0
        toasts.push(`Переименовано файлов: ${renamed}${errors ? `, ошибок: ${errors}` : ''}`, 'success')
        loadCollections()
      } else {
        toasts.push(data?.error || 'Не удалось переименовать файлы', 'error')
      }
    } catch {
      toasts.push('Ошибка при переименовании файлов', 'error')
    } finally {
      setRenameLoading(false)
    }
  }

  const clearCollection = async () => {
    if (!selectedId) return
    if (!confirm('Удалить все файлы из коллекции, не удаляя её саму?')) return
    setClearLoading(true)
    try {
      const r = await fetch(`/api/collections/${selectedId}/clear`, { method: 'POST' })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok) {
        const removed = Number(data.removed_files || 0)
        toasts.push(`Из коллекции удалено файлов: ${removed}`, removed ? 'success' : 'info')
        setCollections(prev => prev.map(col => col.id === selectedId ? { ...col, count: 0 } : col))
      } else {
        toasts.push(data?.error || 'Не удалось очистить коллекцию', 'error')
      }
    } catch {
      toasts.push('Ошибка при очистке коллекции', 'error')
    } finally {
      setClearLoading(false)
      loadCollections()
    }
  }

  const deleteCollection = async () => {
    if (!selectedId) return
    await deleteCollectionById(selectedId)
  }

  const exportExcel = async () => {
    if (!selectedId) return
    setExportLoading(true)
    try {
      const r = await fetch(`/api/collections/${selectedId}/export/excel`)
      if (!r.ok) {
        let message = 'Не удалось экспортировать коллекцию'
        try {
          const data = await r.json()
          if (data?.error) message = data.error
        } catch {}
        toasts.push(message, 'error')
        return
      }
      const blob = await r.blob()
      let filename = `collection-${selectedId}.xlsx`
      const disposition = r.headers.get('Content-Disposition') || r.headers.get('content-disposition')
      if (disposition) {
        const match = disposition.match(/filename\s*=\s*"?([^";]+)"?/i)
        if (match && match[1]) {
          try {
            filename = decodeURIComponent(match[1])
          } catch {
            filename = match[1]
          }
        }
      }
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      window.setTimeout(() => {
        window.URL.revokeObjectURL(url)
        link.remove()
      }, 0)
      toasts.push('Файл экспорта загружен', 'success')
    } catch {
      toasts.push('Ошибка сети при экспорте коллекции', 'error')
    } finally {
      setExportLoading(false)
    }
  }

  const addMember = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!selectedId) return
    const payload: Record<string, unknown> = { role: memberForm.role }
    const login = memberForm.username.trim()
    if (memberForm.userId !== null) {
      payload.user_id = memberForm.userId
    } else if (login) {
      payload.username = login
    } else {
      toasts.push('Укажите логин пользователя', 'error')
      return
    }
    try {
      const r = await fetch(`/api/collections/${selectedId}/members`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok) {
        setMemberForm(prev => ({ username: '', role: prev.role, userId: null }))
        setMemberOptions([])
        setMembers(Array.isArray(data.members) ? data.members : [])
        toasts.push('Участник добавлен', 'success')
      } else {
        toasts.push(data?.error || 'Не удалось добавить участника', 'error')
      }
    } catch {
      toasts.push('Ошибка соединения при добавлении участника', 'error')
    }
  }

  const updateRole = async (userId: number, role: string) => {
    if (!selectedId) return
    try {
      const r = await fetch(`/api/collections/${selectedId}/members/${userId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role }),
      })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok) {
        toasts.push('Роль обновлена', 'success')
        setMembers(prev => prev.map(m => (m.user_id === userId ? { ...m, role } : m)))
      } else {
        toasts.push(data?.error || 'Не удалось обновить роль', 'error')
      }
    } catch {
      toasts.push('Ошибка соединения при обновлении роли', 'error')
    }
  }

  const removeMember = async (userId: number) => {
    if (!selectedId) return
    if (!confirm('Удалить участника из коллекции?')) return
    try {
      const r = await fetch(`/api/collections/${selectedId}/members/${userId}`, { method: 'DELETE' })
      if (r.ok) {
        toasts.push('Участник удалён', 'success')
        setMembers(prev => prev.filter(m => m.user_id !== userId))
      } else {
        const data = await r.json().catch(() => ({}))
        toasts.push(data?.error || 'Не удалось удалить участника', 'error')
      }
    } catch {
      toasts.push('Ошибка соединения при удалении участника', 'error')
    }
  }

  if (!collections.length && !loading) {
    return (
      <div className="card p-3">
        <div className="fw-semibold mb-2">Коллекции</div>
        <div className="text-muted">Нет доступных коллекций.</div>
      </div>
    )
  }

  return (
    <div className="row g-3">
      <div className="col-lg-5">
        <div className="card p-3 h-100">
          <div className="d-flex justify-content-between align-items-center mb-2">
            <div className="fw-semibold">Коллекции</div>
            <button className="btn btn-outline-secondary btn-sm" onClick={loadCollections} disabled={loading}>{loading ? '…' : 'Обновить'}</button>
          </div>
          <div className="list-group" style={{ maxHeight: 420, overflow: 'auto' }}>
            {collections.map(col => (
              <button
                key={col.id}
                type="button"
                className={`list-group-item list-group-item-action ${selectedId === col.id ? 'active' : ''}`}
                onClick={() => setSelectedId(col.id)}
              >
                <div className="d-flex justify-content-between align-items-center">
                  <div>
                    <div className="fw-semibold">{col.name}</div>
                    <div style={{ fontSize: 12 }} className="text-muted">
                      {col.owner_username ? `Владелец: ${col.owner_username}` : 'Владелец не задан'} · {col.is_private ? 'Приватная' : 'Общая'} · {col.count ?? 0} файлов
                    </div>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="col-lg-7">
        <div className="card p-3 h-100">
          <div className="fw-semibold mb-2">Участники</div>
          {selectedCollection ? (
            <>
              <div className="d-flex justify-content-between align-items-center mb-3 flex-wrap gap-2">
                <div>
                  <div className="fw-semibold">{selectedCollection.name}</div>
                  <div className="text-muted" style={{ fontSize: 12 }}>
                    {selectedCollection.owner_username ? `Владелец: ${selectedCollection.owner_username}` : 'Владелец не задан'} · {selectedCollection.is_private ? 'Приватная' : 'Общая'}
                  </div>
                </div>
                <div className="d-flex align-items-center gap-2 flex-wrap">
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    onClick={exportExcel}
                    disabled={exportLoading}
                  >
                    {exportLoading ? '...' : 'Экспорт в Excel'}
                  </button>
                  {isAdmin && (
                    <div className="btn-group btn-group-sm">
                  <button
                    type="button"
                    className="btn btn-outline-secondary"
                    onClick={rebuildRag}
                    disabled={ragLoading}
                  >
                    {ragLoading ? '...' : 'RAG индекс'}
                  </button>
                  <button
                    type="button"
                    className="btn btn-outline-danger"
                    onClick={deleteRagIndex}
                    disabled={ragDeleteLoading}
                  >
                    {ragDeleteLoading ? '...' : 'Удалить RAG'}
                  </button>
                  <button
                    type="button"
                    className="btn btn-outline-secondary"
                    onClick={rescanCollection}
                    disabled={rescanLoading}
                      >
                        {rescanLoading ? '...' : 'Пересканировать'}
                      </button>
                      <button
                        type="button"
                        className="btn btn-outline-secondary"
                        onClick={renameAll}
                        disabled={renameLoading}
                      >
                        {renameLoading ? '...' : 'Переименовать все'}
                      </button>
                      <button
                        type="button"
                        className="btn btn-outline-warning"
                        onClick={clearCollection}
                        disabled={clearLoading}
                      >
                        {clearLoading ? '...' : 'Очистить'}
                      </button>
                      <button
                        type="button"
                        className="btn btn-outline-danger"
                        onClick={deleteCollection}
                        disabled={deletingCollectionId === selectedId}
                      >
                        {deletingCollectionId === selectedId ? '...' : 'Удалить'}
                      </button>
                    </div>
                  )}
                </div>
              </div>
              <form className="row g-2 align-items-end mb-3" onSubmit={addMember}>
                <div className="col-md-5 position-relative">
                  <label className="form-label">Логин пользователя</label>
                  <input
                    className="form-control"
                    value={memberForm.username}
                    onChange={e => setMemberForm({ ...memberForm, username: e.target.value, userId: null })}
                    placeholder="Например, ivanov"
                    autoComplete="off"
                  />
                  {(memberLookupLoading || memberOptions.length > 0) && (
                    <div className="border rounded-3 mt-1" style={{ position: 'absolute', insetInlineStart: 0, top: '100%', width: '100%', background: 'var(--surface)', zIndex: 2000, boxShadow: 'var(--card-shadow)' }}>
                      {memberLookupLoading && <div className="px-3 py-2 text-muted" style={{ fontSize: 13 }}>Поиск…</div>}
                      {!memberLookupLoading && memberOptions.map((opt, idx) => (
                        <button
                          key={opt.id}
                          type="button"
                          className="btn w-100 text-start"
                          style={{ border: 'none', borderBottom: idx === memberOptions.length - 1 ? 'none' : '1px solid var(--border)', borderRadius: 0 }}
                          onMouseDown={() => {
                            setMemberForm({ username: opt.username, role: memberForm.role, userId: opt.id })
                            setMemberOptions([])
                          }}
                        >
                          <div className="fw-semibold" style={{ fontSize: 13 }}>{opt.full_name || '—'}</div>
                          <div className="text-muted" style={{ fontSize: 12 }}>@{opt.username}</div>
                        </button>
                      ))}
                    </div>
                  )}
                  {memberForm.userId !== null && <div className="form-text">ID: {memberForm.userId}</div>}
                </div>
                <div className="col-md-4">
                  <label className="form-label">Роль</label>
                  <select className="form-select" value={memberForm.role} onChange={e=>setMemberForm({...memberForm, role:e.target.value})}>
                    <option value="viewer">{ROLE_LABELS.viewer}</option>
                    <option value="editor">{ROLE_LABELS.editor}</option>
                    <option value="owner" disabled={!isAdmin}>{ROLE_LABELS.owner}</option>
                  </select>
                </div>
                <div className="col-md-3">
                  <button className="btn btn-primary w-100" type="submit">Добавить</button>
                </div>
              </form>
              <div className="table-responsive" style={{ maxHeight: 300, overflow: 'auto' }}>
                <table className="table table-sm">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Имя</th>
                      <th>Роль</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {members.map(m => (
                      <tr key={m.user_id}>
                        <td>{m.user_id}</td>
                        <td>{m.username ?? '—'}</td>
                        <td>
                          <select className="form-select form-select-sm" value={m.role} onChange={e=>updateRole(m.user_id, e.target.value)}>
                            <option value="viewer">{ROLE_LABELS.viewer}</option>
                            <option value="editor">{ROLE_LABELS.editor}</option>
                            <option value="owner" disabled={!isAdmin}>{ROLE_LABELS.owner}</option>
                          </select>
                        </td>
                        <td>
                          <button className="btn btn-outline-danger btn-sm" onClick={()=>removeMember(m.user_id)}>Удалить</button>
                        </td>
                      </tr>
                    ))}
                    {members.length === 0 && (
                      <tr><td colSpan={4} className="text-center text-muted py-3">Участники не найдены</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div className="text-muted">Выберите коллекцию, чтобы просмотреть участников.</div>
          )}
      </div>
    </div>
    <div className="col-12">
      <div className="card p-3">
        <div className="fw-semibold mb-2">Настройки коллекций</div>
        <div className="table-responsive">
          <table className="table table-sm align-middle">
            <thead>
              <tr>
                <th>Название</th>
                <th>Файлов</th>
                <th>Поиск</th>
                <th>Граф</th>
                <th className="text-end">Действия</th>
              </tr>
            </thead>
            <tbody>
              {collections.map(col => {
                const searchChecked = col.searchable !== false
                const graphChecked = col.graphable !== false
                return (
                  <tr key={col.id}>
                    <td>
                      <div className="fw-semibold">{col.name}</div>
                      <div className="text-muted" style={{ fontSize: 12 }}>{col.slug ? `slug: ${col.slug}` : 'slug не задан'}</div>
                    </td>
                    <td className="text-secondary">{col.count ?? 0}</td>
                    <td>
                      <input
                        className="form-check-input"
                        type="checkbox"
                        checked={searchChecked}
                        onChange={event => updateCollectionFlag(col.id, 'searchable', event.target.checked)}
                      />
                    </td>
                    <td>
                      <input
                        className="form-check-input"
                        type="checkbox"
                        checked={graphChecked}
                        onChange={event => updateCollectionFlag(col.id, 'graphable', event.target.checked)}
                      />
                    </td>
                    <td>
                      <div className="d-flex gap-2 justify-content-end">
                        <button
                          className="btn btn-sm btn-outline-primary"
                          onClick={() => prepareCollectionForChat(col.id)}
                          disabled={docChatCollectionId === col.id}
                        >
                          {docChatCollectionId === col.id ? '…' : 'Обработать для чата'}
                        </button>
                        <button
                          className="btn btn-sm btn-outline-danger"
                          onClick={() => deleteCollectionById(col.id)}
                          disabled={deletingCollectionId === col.id}
                        >
                          {deletingCollectionId === col.id ? '...' : 'Удалить'}
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
              {collections.length === 0 && (
                <tr>
                  <td colSpan={5} className="text-center text-muted py-3">Коллекции не найдены</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="d-flex flex-wrap align-items-center gap-2 mt-2">
          <input
            className="form-control"
            placeholder="Новая коллекция"
            value={newCollectionName}
            onChange={event => setNewCollectionName(event.target.value)}
            style={{ maxWidth: 320 }}
          />
          <button
            className="btn btn-outline-primary"
            onClick={saveCollectionSettings}
            disabled={collectionsSaving}
          >
            {collectionsSaving ? 'Сохранение…' : 'Сохранить'}
          </button>
        </div>
      </div>
    </div>
  </div>
)
}
