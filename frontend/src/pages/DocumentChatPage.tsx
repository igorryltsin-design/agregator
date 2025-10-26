import React, { FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import ProgressPanel from '../ui/ProgressPanel'
import { useToasts } from '../ui/Toasts'

type DocumentInfo = {
  id: number
  title: string
  author?: string | null
  year?: string | null
  collection?: string | null
  material_type?: string | null
  has_rag?: boolean
}

type DocTextSource = {
  label: string
  preview?: string
  page?: number | null
  section_path?: string | null
  score?: number
}

type DocImageSource = {
  label: string
  description?: string
  page?: number | null
  keywords?: string[]
  url?: string | null
  score?: number
}

type DocChatMessage = {
  role: 'user' | 'assistant'
  content: string
  sources?: {
    texts?: DocTextSource[]
    images?: DocImageSource[]
  }
}

type DocChatSession = {
  id: string
  status: 'queued' | 'processing' | 'ready' | 'error'
  percent?: number
  progress?: string[]
  error?: string
  file_meta?: {
    id?: number
    title?: string
    author?: string
    year?: string
    collection_name?: string
    filename?: string
  }
  data?: {
    chunk_count?: number
    language?: string
    image_count?: number
    images?: DocImageSource[]
  }
  history?: DocChatMessage[]
}

const POLL_INTERVAL_MS = 2000

const DocumentChatPage: React.FC = () => {
  const toasts = useToasts()
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [loadingDocs, setLoadingDocs] = useState<boolean>(false)
  const [docError, setDocError] = useState<string | null>(null)
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [selectedId, setSelectedId] = useState<number | ''>('')
  const [session, setSession] = useState<DocChatSession | null>(null)
  const [preparing, setPreparing] = useState<boolean>(false)
  const [question, setQuestion] = useState<string>('')
  const [sending, setSending] = useState<boolean>(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const chatEndRef = useRef<HTMLDivElement | null>(null)
  const prevStatusRef = useRef<string | null>(null)

  useEffect(() => {
    void fetchDocuments()
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    if (!session?.history) return
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [session?.history])

  useEffect(() => {
    if (!session) return
    if (session.status === 'ready' || session.status === 'error') {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    } else if (!pollRef.current) {
      pollRef.current = setInterval(() => {
        void fetchStatus(session.id)
      }, POLL_INTERVAL_MS)
    }
    const status = session.status
    if (status && status !== prevStatusRef.current) {
      if (status === 'ready') {
        toasts.push('Документ готов к диалогу.', 'success')
      } else if (status === 'error' && session.error) {
        toasts.push(session.error, 'error')
      }
      prevStatusRef.current = status
    }
  }, [session, toasts])

  const filteredDocuments = useMemo(() => {
    if (!searchTerm.trim()) {
      return documents
    }
    const term = searchTerm.trim().toLowerCase()
    return documents.filter(doc => {
      return [
        doc.title,
        doc.author || '',
        doc.collection || '',
        doc.year || '',
        doc.material_type || '',
      ].some(value => value.toLowerCase().includes(term))
    })
  }, [documents, searchTerm])

  const fetchDocuments = async () => {
    setLoadingDocs(true)
    setDocError(null)
    try {
      const resp = await fetch('/api/doc-chat/documents')
      if (!resp.ok) {
        throw new Error()
      }
      const data = await resp.json()
      setDocuments(Array.isArray(data?.items) ? data.items : [])
    } catch (error) {
      setDocError('Не удалось загрузить список документов.')
      toasts.push('Не удалось загрузить список документов', 'error')
    } finally {
      setLoadingDocs(false)
    }
  }

  const handlePrepare = async () => {
    if (!selectedId) {
      toasts.push('Выберите документ для подготовки', 'error')
      return
    }
    setPreparing(true)
    try {
      const resp = await fetch('/api/doc-chat/prepare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_id: selectedId }),
      })
      const data = await resp.json()
      if (!resp.ok || !data?.ok) {
        throw new Error(data?.error || 'Не удалось подготовить документ')
      }
      setSession(data.session || null)
      setQuestion('')
      toasts.push('Подготовка документа запущена', 'info')
    } catch (error: any) {
      toasts.push(error?.message || 'Не удалось подготовить документ', 'error')
    } finally {
      setPreparing(false)
    }
  }

  const fetchStatus = async (sessionId: string) => {
    try {
      const resp = await fetch(`/api/doc-chat/status/${sessionId}`)
      const data = await resp.json()
      if (!resp.ok || !data?.ok) {
        return
      }
      setSession(data.session || null)
    } catch {
      // игнорируем временные ошибки
    }
  }

  const handleAsk = async (event: FormEvent) => {
    event.preventDefault()
    if (!session || session.status !== 'ready') {
      return
    }
    const trimmed = question.trim()
    if (!trimmed) {
      toasts.push('Введите вопрос', 'error')
      return
    }
    setSending(true)
    try {
      const resp = await fetch('/api/doc-chat/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: session.id, question: trimmed }),
      })
      const data = await resp.json()
      if (!resp.ok || !data?.ok) {
        throw new Error(data?.error || 'Не удалось получить ответ')
      }
      setSession(data.session || null)
      setQuestion('')
    } catch (error: any) {
      toasts.push(error?.message || 'Не удалось получить ответ', 'error')
    } finally {
      setSending(false)
    }
  }

  const currentImages = session?.data?.images || []
  const history = session?.history || []
  const sessionStatus = session?.status
  const isProcessing = sessionStatus === 'queued' || sessionStatus === 'processing'

  return (
    <div className="container-xxl d-grid gap-3">
      <div className="card border-0 shadow-sm doc-chat-intro">
        <div className="card-body d-grid gap-2">
          <div>
            <h1 className="h4 mb-1 d-flex align-items-center gap-2">
              <span className="badge bg-primary-subtle text-primary-emphasis rounded-pill">
                Режим «Документ-чат»
              </span>
              <span className="text-muted fw-normal" style={{ fontSize: '0.95rem' }}>
                Анализ текста и изображений выбранного файла
              </span>
            </h1>
            <p className="text-muted mb-0" style={{ maxWidth: 720 }}>
              Выберите документ из каталога. Agregator построит индекс по тексту и визуальным материалам,
              а затем помощник будет отвечать только с опорой на найденные фрагменты и описания изображений.
            </p>
          </div>
          <ol className="doc-chat-steps">
            <li><strong>Выбор документа.</strong> Найдите файл по названию, автору или коллекции и нажмите «Подготовить».</li>
            <li><strong>Анализ и прогресс.</strong> Следите за этапами обработки и загрузкой изображений в панели прогресса.</li>
            <li><strong>Диалог по содержанию.</strong> После готовности задавайте вопросы — помощник цитирует текст и изображения из документа.</li>
          </ol>
        </div>
      </div>
      <div className="d-flex align-items-start justify-content-between flex-wrap gap-2">
        <div>
          <h2 className="h3 mb-1">Чат по документу</h2>
          <p className="text-muted mb-0">Выберите документ, дождитесь анализа и задавайте вопросы с учётом текста и изображений.</p>
        </div>
        <button
          className="btn btn-outline-secondary"
          type="button"
          onClick={() => { void fetchDocuments() }}
          disabled={loadingDocs}
        >
          {loadingDocs ? 'Обновляю…' : 'Обновить список'}
        </button>
      </div>

      <div className="card">
        <div className="card-body">
          <div className="row g-3 align-items-end">
            <div className="col-12 col-md-5">
              <label className="form-label" htmlFor="docChatSearch">Поиск по документам</label>
              <input
                id="docChatSearch"
                className="form-control"
                placeholder="Название, автор, коллекция…"
                value={searchTerm}
                onChange={event => setSearchTerm(event.target.value)}
              />
            </div>
            <div className="col-12 col-md-5">
              <label className="form-label" htmlFor="docChatSelect">Доступные документы</label>
              <select
                id="docChatSelect"
                className="form-select"
                value={selectedId === '' ? '' : String(selectedId)}
                onChange={event => {
                  const value = event.target.value
                  setSelectedId(value ? Number(value) : '')
                }}
                size={Math.min(filteredDocuments.length, 8) || undefined}
              >
                <option value="">— Выберите документ —</option>
                {filteredDocuments.map(doc => (
                  <option key={doc.id} value={doc.id}>
                    {doc.title}{doc.author ? ` · ${doc.author}` : ''}{doc.year ? ` · ${doc.year}` : ''}{doc.collection ? ` · ${doc.collection}` : ''}
                  </option>
                ))}
              </select>
            </div>
            <div className="col-12 col-md-2 d-grid">
              <button
                className="btn btn-primary"
                type="button"
                onClick={handlePrepare}
                disabled={!selectedId || preparing}
              >
                {preparing ? 'Подготовка…' : 'Подготовить'}
              </button>
            </div>
          </div>
          {docError && (
            <div className="alert alert-warning mt-3 mb-0" role="alert">
              {docError}
            </div>
          )}
          {!docError && !loadingDocs && documents.length === 0 && (
            <div className="alert alert-info mt-3 mb-0" role="alert">
              Документы недоступны. Проверьте права доступа или обновите список.
            </div>
          )}
        </div>
      </div>

      {session && (
        <div className="d-grid gap-3">
          {isProcessing && (
            <ProgressPanel
              title={session.status === 'queued' ? 'В очереди на обработку…' : 'Анализ документа…'}
              caption="Статус подготовки включает извлечение текста, генерацию эмбеддингов и разметку изображений."
              progress={typeof session.percent === 'number' ? { percent: Math.round(session.percent), label: `${Math.round(session.percent)}%` } : undefined}
              bullets={(session.progress || []).map((line, idx) => ({ id: `${idx}-${line}`, text: line }))}
            />
          )}
          {session.status === 'error' && (
            <div className="alert alert-danger mb-0" role="alert">
              {session.error || 'Не удалось подготовить документ.'}
            </div>
          )}
          {session.status === 'ready' && (
            <div className="row g-3">
              <div className="col-12 col-xl-7">
                <div className="card h-100">
                  <div className="card-header">
                    Диалог
                  </div>
                  <div className="card-body d-flex flex-column" style={{ maxHeight: '75vh' }}>
                    <div className="flex-grow-1 overflow-auto d-grid gap-3 pe-1">
                      {history.length === 0 && (
                        <div className="text-muted">
                          Задайте первый вопрос, чтобы начать диалог по документу.
                        </div>
                      )}
                      {history.map((msg, idx) => (
                        <div
                          key={idx}
                          className={`border rounded-3 p-3 ${msg.role === 'assistant' ? 'bg-light' : ''}`}
                        >
                          <div className="d-flex justify-content-between align-items-center">
                            <div className="small text-uppercase fw-semibold text-muted">
                              {msg.role === 'assistant' ? 'Помощник' : 'Вы'}
                            </div>
                          </div>
                          <div className="mt-2" style={{ whiteSpace: 'pre-wrap' }}>
                            {msg.content}
                          </div>
                          {msg.sources?.texts && msg.sources.texts.length > 0 && (
                            <div className="mt-3">
                              <div className="small text-muted mb-1">Текстовые фрагменты:</div>
                              <div className="d-grid gap-1">
                                {msg.sources.texts.map((src, srcIdx) => (
                                  <div key={srcIdx} className="small text-muted">
                                    <span className="badge bg-secondary me-2">{src.label}</span>
                                    {typeof src.page === 'number' && (
                                      <span className="me-2">стр. {src.page}</span>
                                    )}
                                    {src.section_path && (
                                      <span className="me-2">{src.section_path}</span>
                                    )}
                                    {typeof src.score === 'number' && (
                                      <span className="me-2">· score {src.score.toFixed(3)}</span>
                                    )}
                                    {src.preview && (
                                      <span className="d-block text-muted">{src.preview}</span>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                          {msg.sources?.images && msg.sources.images.length > 0 && (
                            <div className="mt-3">
                              <div className="small text-muted mb-1">Изображения:</div>
                              <div className="d-grid gap-1">
                                {msg.sources.images.map((img, imgIdx) => (
                                  <div key={imgIdx} className="small text-muted">
                                    <span className="badge bg-info text-dark me-2">{img.label}</span>
                                    {typeof img.page === 'number' && (
                                      <span className="me-2">стр. {img.page}</span>
                                    )}
                                    {typeof img.score === 'number' && (
                                      <span className="me-2">· score {img.score.toFixed(3)}</span>
                                    )}
                                    {img.description && (
                                      <span className="d-block">{img.description}</span>
                                    )}
                                    {img.keywords && img.keywords.length > 0 && (
                                      <span className="d-block text-muted">Ключевые слова: {img.keywords.join(', ')}</span>
                                    )}
                                    {img.url && (
                                      <a className="d-inline-block mt-1" href={img.url} target="_blank" rel="noopener noreferrer">
                                        Открыть изображение →
                                      </a>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                      <div ref={chatEndRef} />
                    </div>
                    <form className="mt-3 d-flex gap-2" onSubmit={handleAsk}>
                      <input
                        className="form-control"
                        placeholder="Задайте вопрос по документу…"
                        value={question}
                        onChange={event => setQuestion(event.target.value)}
                        disabled={sending}
                      />
                      <button
                        className="btn btn-primary"
                        type="submit"
                        disabled={sending || session.status !== 'ready'}
                      >
                        {sending ? 'Отправка…' : 'Спросить'}
                      </button>
                    </form>
                  </div>
                </div>
              </div>
              <div className="col-12 col-xl-5 d-grid gap-3">
                <div className="card">
                  <div className="card-header">
                    Информация о документе
                  </div>
                  <div className="card-body">
                    <dl className="row mb-0">
                      <dt className="col-sm-4">Название</dt>
                      <dd className="col-sm-8">{session.file_meta?.title || '—'}</dd>
                      <dt className="col-sm-4">Автор</dt>
                      <dd className="col-sm-8">{session.file_meta?.author || '—'}</dd>
                      <dt className="col-sm-4">Год</dt>
                      <dd className="col-sm-8">{session.file_meta?.year || '—'}</dd>
                      <dt className="col-sm-4">Коллекция</dt>
                      <dd className="col-sm-8">{session.file_meta?.collection_name || '—'}</dd>
                      <dt className="col-sm-4">Чанков</dt>
                      <dd className="col-sm-8">{session.data?.chunk_count ?? '—'}</dd>
                      <dt className="col-sm-4">Язык</dt>
                      <dd className="col-sm-8">{session.data?.language || '—'}</dd>
                    </dl>
                  </div>
                </div>
                <div className="card">
                  <div className="card-header">
                    Изображения ({currentImages.length})
                  </div>
                  <div className="card-body d-grid gap-3">
                    {currentImages.length === 0 && (
                      <div className="text-muted">
                        Изображения не были обнаружены или анализ отключён.
                      </div>
                    )}
                    {currentImages.map((img, idx) => (
                      <div key={idx} className="border rounded-3 p-3">
                        <div className="d-flex justify-content-between align-items-center mb-2">
                          <div className="fw-semibold">{img.label || `Изображение ${idx + 1}`}</div>
                          {typeof img.score === 'number' && (
                            <span className="badge bg-light text-dark">score {img.score.toFixed(3)}</span>
                          )}
                        </div>
                        {img.url && (
                          <a href={img.url} target="_blank" rel="noopener noreferrer" className="d-block mb-2">
                            <img src={img.url} alt={img.description || img.label || 'Изображение'} className="img-fluid rounded" />
                          </a>
                        )}
                        {typeof img.page === 'number' && (
                          <div className="small text-muted">Страница: {img.page}</div>
                        )}
                        {img.description && (
                          <div className="mt-2">{img.description}</div>
                        )}
                        {img.keywords && img.keywords.length > 0 && (
                          <div className="mt-2 small text-muted">Ключевые слова: {img.keywords.join(', ')}</div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default DocumentChatPage
