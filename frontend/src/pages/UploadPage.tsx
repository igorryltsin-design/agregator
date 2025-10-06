import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useAuth } from '../ui/Auth'

type Collection = { id: number; name: string }

type ImportJob = {
  id: number
  status: string
  progress: number
  created_at?: string | null
  started_at?: string | null
  finished_at?: string | null
  error?: string | null
  payload?: any
  payload_json?: any
}

const STATUS_LABELS: Record<string, string> = {
  queued: 'В очереди',
  pending: 'Ожидание',
  running: 'Выполняется',
  completed: 'Готово',
  error: 'Ошибка',
  cancelling: 'Отмена',
  cancelled: 'Отменено',
}

const FINAL_STATUSES = new Set(['completed', 'error', 'cancelled'])

export default function UploadPage(){
  const { user } = useAuth()
  const isAdmin = user?.role === 'admin'
  const [file, setFile] = useState<File|null>(null)
  const [busy, setBusy] = useState(false)
  const [collections, setCollections] = useState<Collection[]>([])
  const [collectionId, setCollectionId] = useState<string>('')
  const [newCollection, setNewCollection] = useState<string>('')
  const [isPrivate, setIsPrivate] = useState(!isAdmin)
  const [jobs, setJobs] = useState<ImportJob[]>([])
  const [loadingJobs, setLoadingJobs] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const loadCollections = useCallback(async () => {
    try{
      const r=await fetch('/upload')
      if (!r.ok) return
      const j=await r.json()
      const cols: Collection[] = Array.isArray(j.collections) ? j.collections : []
      setCollections(cols)
      if(cols[0]) setCollectionId(String(cols[0].id))
    }catch{}
  }, [])

  const loadJobs = useCallback(async () => {
    if (!user) return
    setLoadingJobs(true)
    try {
      const r = await fetch('/api/import/jobs')
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok && Array.isArray(data.tasks)) {
        const mapped: ImportJob[] = data.tasks.map((task: any) => ({
          ...task,
          payload: task.payload_json ?? task.payload,
          payload_json: task.payload_json ?? undefined
        }))
        setJobs(mapped)
      }
    } catch {
      /* ignore */
    } finally {
      setLoadingJobs(false)
    }
  }, [user])

  useEffect(()=>{ loadCollections() }, [loadCollections])
  useEffect(()=>{ loadJobs(); const t=window.setInterval(loadJobs, 4000); return ()=>window.clearInterval(t) }, [loadJobs])

  const submit = async () => {
    if(!file) return
    setBusy(true)
    try{
      const fd = new FormData()
      fd.append('file', file)
      if (collectionId) fd.append('collection_id', collectionId)
      if (newCollection.trim()) {
        fd.append('new_collection', newCollection.trim())
        if (isPrivate || !isAdmin) fd.append('private','on')
      }
      const r = await fetch('/api/import/jobs', { method:'POST', body: fd })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok) {
        setJobs(prev => [{
          id: data.task_id,
          status: 'queued',
          progress: 0,
          payload: { initial_preview: data.initial_preview },
          payload_json: { initial_preview: data.initial_preview }
        }, ...prev.filter(job => job.id !== data.task_id)])
        setFile(null)
        setNewCollection('')
        if (fileInputRef.current) fileInputRef.current.value = ''
        loadJobs()
      } else {
        alert(data?.error || 'Не удалось поставить задачу импорта')
      }
    }finally{ setBusy(false) }
  }

  const renderPreview = (payload: any) => {
    if (!payload) return null
    const preview = payload.result?.preview || payload.initial_preview || payload.preview
    if (!preview) return null
    const lines: string[] = []
    if (preview.filename) lines.push(`Файл: ${preview.filename}`)
    if (preview.collection_name) lines.push(`Коллекция: ${preview.collection_name}`)
    if (preview.material_type) lines.push(`Тип: ${preview.material_type}`)
    if (preview.language) lines.push(`Язык: ${preview.language}`)
    if (preview.title) lines.push(`Заголовок: ${preview.title}`)
    if (preview.author) lines.push(`Автор: ${preview.author}`)
    if (preview.text_excerpt) lines.push(`Фрагмент: ${preview.text_excerpt}`)
    const metadata = preview.metadata
    const fileId = payload.result?.file_id || payload.file_id
    const relPath = payload.result?.rel_path || payload.rel_path
    return (
      <div className="small text-muted">
        {lines.map((line, idx) => <div key={idx}>{line}</div>)}
        {fileId && (
          <div className="mt-1 d-flex gap-2 align-items-center">
            <a className="btn btn-outline-primary btn-sm" href={`/catalogue?highlight=${encodeURIComponent(String(fileId))}`} target="_blank" rel="noreferrer">Открыть каталог</a>
            {relPath && <span>{relPath}</span>}
          </div>
        )}
        {metadata && (
          <details className="mt-1">
            <summary>Метаданные LLM</summary>
            <pre className="small bg-light rounded p-2 mt-1">{JSON.stringify(metadata, null, 2)}</pre>
          </details>
        )}
      </div>
    )
  }

  const sortedJobs = useMemo(() => {
    return [...jobs].sort((a, b) => {
      const da = a.created_at ? new Date(a.created_at).getTime() : 0
      const db = b.created_at ? new Date(b.created_at).getTime() : 0
      return db - da
    })
  }, [jobs])

  return (
    <div className="d-grid gap-3">
      <div className="card p-3">
        <div className="fw-semibold mb-2">Загрузка файла</div>
        <input className="form-control" type="file" ref={fileInputRef} onChange={e=>setFile(e.target.files?.[0] || null)} />
        <div className="row g-2 mt-2">
          <div className="col-md-6">
            <label className="form-label">Коллекция</label>
            <select className="form-select" value={collectionId} onChange={e=>setCollectionId(e.target.value)}>
              {collections.map(c=> <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
          </div>
          <div className="col-md-6">
            <label className="form-label">Новая коллекция</label>
            <input className="form-control" placeholder="или новая…" value={newCollection} onChange={e=>setNewCollection(e.target.value)} />
          </div>
        </div>
        {newCollection.trim().length > 0 && (
          <div className="form-check form-switch mt-2">
            <input className="form-check-input" type="checkbox" id="priv-upload" checked={isPrivate || !isAdmin} disabled={!isAdmin} onChange={e=>setIsPrivate(e.target.checked)} />
            <label className="form-check-label" htmlFor="priv-upload">Приватная коллекция</label>
          </div>
        )}
        <div className="mt-3 d-flex gap-2">
          <button className="btn btn-primary" onClick={submit} disabled={!file||busy}>{busy?'Загрузка…':'Отправить в обработку'}</button>
          {!isAdmin && <span className="text-muted" style={{fontSize:12}}>Файл будет обработан фоном; следите за статусом ниже.</span>}
        </div>
      </div>

      <div className="card p-3">
        <div className="d-flex justify-content-between align-items-center mb-2">
          <div className="fw-semibold">Очередь импорта</div>
          <button className="btn btn-outline-secondary btn-sm" onClick={loadJobs} disabled={loadingJobs}>{loadingJobs?'Обновление…':'Обновить'}</button>
        </div>
        {sortedJobs.length === 0 ? (
          <div className="text-muted">Задач пока нет</div>
        ) : (
          <div className="d-grid gap-3">
            {sortedJobs.map(job => {
              const payload = job.payload ?? job.payload_json ?? {}
              const statusLabel = STATUS_LABELS[job.status] || job.status
              const progressPercent = Math.round((job.progress || 0) * 100)
              const error = job.error || payload?.error
              return (
                <div key={job.id} className="border rounded p-3">
                  <div className="d-flex justify-content-between flex-wrap gap-2">
                    <div>
                      <div className="fw-semibold">#{job.id} — {payload?.filename || payload?.result?.preview?.filename || 'импорт файла'}</div>
                      <div className="small text-muted">Статус: {statusLabel}</div>
                      {job.created_at && <div className="small text-muted">Создана: {new Date(job.created_at).toLocaleString()}</div>}
                      {job.finished_at && <div className="small text-muted">Завершена: {new Date(job.finished_at).toLocaleString()}</div>}
                    </div>
                    <div style={{minWidth:200}}>
                      <div className="progress" style={{height:6}}>
                        <div className={`progress-bar${FINAL_STATUSES.has(job.status) && job.status !== 'completed' ? ' bg-danger' : ''}`} style={{width:`${Math.min(100, Math.max(0, progressPercent))}%`}} />
                      </div>
                      <div className="small text-muted mt-1">{progressPercent}%</div>
                    </div>
                  </div>
                  {error && <div className="alert alert-danger py-2 px-3 mt-2 mb-0">{error}</div>}
                  {!error && renderPreview(payload)}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
