import React, { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import DOMPurify from 'dompurify'
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
  doc_chat_ready?: boolean
}

type DocTextSource = {
  label: string
  preview?: string
  page?: number | null
  section_path?: string | null
  score?: number
  highlights?: string[]
  matched_terms?: string[]
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
    rel_path?: string
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

const escapeHtml = (value: string) =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')

const formatInlineMarkdown = (input: string) => {
  let output = escapeHtml(input)
  output = output.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  output = output.replace(/__(.+?)__/g, '<strong>$1</strong>')
  output = output.replace(/`([^`]+?)`/g, '<code>$1</code>')
  output = output.replace(/_(.+?)_/g, '<em>$1</em>')
  return output
}

const markdownToHtml = (text: string): string => {
  const lines = text.replace(/\r\n?/g, '\n').split('\n')
  let html = ''
  let inList = false
  let inQuote = false

  const closeBlocks = () => {
    if (inList) {
      html += '</ul>'
      inList = false
    }
    if (inQuote) {
      html += '</blockquote>'
      inQuote = false
    }
  }

  lines.forEach(line => {
    const trimmed = line.trim()
    if (!trimmed) {
      closeBlocks()
      html += '<p></p>'
      return
    }
    if (/^[-*]\s+/.test(trimmed)) {
      if (inQuote) {
        html += '</blockquote>'
        inQuote = false
      }
      if (!inList) {
        html += '<ul>'
        inList = true
      }
      const item = formatInlineMarkdown(trimmed.replace(/^[-*]\s+/, ''))
      html += `<li>${item}</li>`
      return
    }
    if (/^>\s+/.test(trimmed)) {
      if (!inQuote) {
        closeBlocks()
        html += '<blockquote>'
        inQuote = true
      }
      const quote = formatInlineMarkdown(trimmed.replace(/^>\s+/, ''))
      html += `<p>${quote}</p>`
      return
    }
    closeBlocks()
    html += `<p>${formatInlineMarkdown(trimmed)}</p>`
  })

  closeBlocks()
  return html
}

const DocumentChatPage: React.FC = () => {
  const toasts = useToasts()
  const [searchParams, setSearchParams] = useSearchParams()
  const fileParam = useMemo(() => searchParams.get('file') || searchParams.get('fileId'), [searchParams])
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [loadingDocs, setLoadingDocs] = useState<boolean>(false)
  const [docError, setDocError] = useState<string | null>(null)
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [selectedId, setSelectedId] = useState<number | ''>('')
  const [session, setSession] = useState<DocChatSession | null>(null)
  const [preparing, setPreparing] = useState<boolean>(false)
  const [question, setQuestion] = useState<string>('')
  const [sending, setSending] = useState<boolean>(false)
  const [assistantThinking, setAssistantThinking] = useState<boolean>(false)
  const [clearing, setClearing] = useState<boolean>(false)
  const [sourcePanels, setSourcePanels] = useState<Record<number, boolean>>({})
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const chatEndRef = useRef<HTMLDivElement | null>(null)
  const prevStatusRef = useRef<string | null>(null)
  const questionInputRef = useRef<HTMLTextAreaElement | null>(null)
  const autoPrepareRef = useRef<Record<number, boolean>>({})

  const selectedDoc = useMemo(() => (selectedId ? documents.find(doc => doc.id === selectedId) || null : null), [documents, selectedId])

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
    if (!fileParam) return
    if (selectedId !== '') return
    const parsed = Number(fileParam)
    if (!Number.isFinite(parsed)) return
    if (!documents.some(doc => doc.id === parsed)) return
    setSelectedId(parsed)
  }, [documents, fileParam, selectedId])

  useEffect(() => {
    const currentParam = fileParam || ''
    if (selectedId === '') {
      if (currentParam) {
        const next = new URLSearchParams(searchParams)
        next.delete('file')
        next.delete('fileId')
        setSearchParams(next, { replace: true })
      }
      return
    }
    const target = String(selectedId)
    if (currentParam !== target) {
      const next = new URLSearchParams(searchParams)
      next.set('file', target)
      next.delete('fileId')
      setSearchParams(next, { replace: true })
    }
  }, [selectedId, fileParam, searchParams, setSearchParams])

  useEffect(() => {
    const targetId = selectedDoc?.id
    const currentSessionId = session?.file_meta?.id
    if (!targetId) {
      if (session) {
        setSession(null)
        setQuestion('')
      }
      return
    }
    if (currentSessionId && currentSessionId !== targetId) {
      setSession(null)
      setQuestion('')
      setSourcePanels({})
    }
  }, [selectedDoc?.id, session?.file_meta?.id])

  useEffect(() => {
    if (!session?.history) return
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [session?.history])

  useEffect(() => {
    if (session?.status === 'ready') {
      questionInputRef.current?.focus()
    }
  }, [session?.status])

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

  const quickPrompts = useMemo(() => {
    if (!session?.file_meta?.title) {
      return [
        { id: 'summary', label: 'Краткое резюме', prompt: 'Сделай краткое резюме выбранного документа.' },
        { id: 'terms', label: 'Термины', prompt: 'Какие ключевые термины и определения включает документ?' },
        { id: 'actions', label: 'Практика', prompt: 'Как можно применить рекомендации из документа на практике?' },
      ]
    }
    const title = session.file_meta.title
    return [
      { id: 'summary', label: 'Резюме', prompt: `Сделай краткое резюме документа «${title}».` },
      { id: 'context', label: 'Контекст', prompt: `Расскажи, какую проблему решает документ «${title}» и кого она касается.` },
      { id: 'actions', label: 'Что делать', prompt: `Какие действия рекомендует документ «${title}»?` },
    ]
  }, [session?.file_meta?.title])

  const renderMarkdown = useCallback(
    (content: string) =>
      DOMPurify.sanitize(markdownToHtml(content || ''), {
        ALLOWED_TAGS: ['p', 'ul', 'li', 'strong', 'em', 'code', 'blockquote', 'br'],
        ALLOWED_ATTR: [],
      }),
    [],
  )

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

  const handlePrepare = useCallback(async (options?: { silent?: boolean; force?: boolean }) => {
    if (!selectedId) {
      if (!options?.silent) {
        toasts.push('Выберите документ для подготовки', 'error')
      }
      return
    }
    const isForce = Boolean(options?.force)
    if (isForce) {
      autoPrepareRef.current[selectedId] = false
      setDocuments(prev => prev.map(doc => (doc.id === selectedId ? { ...doc, doc_chat_ready: false } : doc)))
      setSession(prev => (prev && prev.file_meta?.id === selectedId ? null : prev))
    }
    setPreparing(true)
    try {
      const resp = await fetch('/api/doc-chat/prepare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_id: selectedId, force: isForce }),
      })
      const data = await resp.json()
      if (!resp.ok || !data?.ok) {
        throw new Error(data?.error || 'Не удалось подготовить документ')
      }
      const sessionPayload: DocChatSession | null = data.session || null
      setSession(sessionPayload)
      setQuestion('')
      const readyId = sessionPayload?.file_meta?.id
      if (readyId && sessionPayload?.status === 'ready') {
        setDocuments(prev => prev.map(doc => (doc.id === readyId ? { ...doc, doc_chat_ready: true } : doc)))
      }
      if (readyId && selectedId === '') {
        setSelectedId(readyId)
      }
      if (!options?.silent) {
        if (sessionPayload?.status === 'ready') {
          if (data.cached) {
            toasts.push('Документ открыт из кэша — можно сразу общаться.', 'success')
          } else if (isForce) {
            toasts.push('Документ пересканирован и готов к чату.', 'success')
          } else {
            toasts.push('Документ подготовлен для чата.', 'success')
          }
        } else {
          toasts.push(isForce ? 'Пересканирование документа запущено' : 'Подготовка документа запущена', 'info')
        }
      }
    } catch (error: any) {
      if (!options?.silent) {
        toasts.push(error?.message || 'Не удалось подготовить документ', 'error')
      }
      return
    } finally {
      setPreparing(false)
    }
  }, [selectedId, setDocuments, setQuestion, setSelectedId, setSession, toasts])

  useEffect(() => {
    if (!selectedDoc) return
    const docId = selectedDoc.id
    if (!docId) return
    if (preparing) return
    if (session && session.file_meta?.id === docId && session.status !== 'error') {
      return
    }
    if (autoPrepareRef.current[docId]) {
      return
    }
    autoPrepareRef.current[docId] = true
    void handlePrepare({ silent: false }).catch(() => {})
  }, [selectedDoc, preparing, session, handlePrepare])

  const fetchStatus = async (sessionId: string) => {
    try {
      const resp = await fetch(`/api/doc-chat/status/${sessionId}`)
      const data = await resp.json()
      if (!resp.ok || !data?.ok) {
        return
      }
      const nextSession: DocChatSession | null = data.session || null
      setSession(nextSession)
      const readyId = nextSession?.file_meta?.id
      if (readyId && nextSession?.status === 'ready') {
        setDocuments(prev => prev.map(doc => (doc.id === readyId ? { ...doc, doc_chat_ready: true } : doc)))
      }
    } catch {
      // ignore transient errors
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
    setAssistantThinking(true)
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
      setAssistantThinking(false)
    }
  }

  const currentImages = session?.data?.images || []
  const history = session?.history || []
  const hasHistory = history.length > 0
  useEffect(() => {
    setSourcePanels({})
  }, [history.length])

  const toggleSources = useCallback((index: number) => {
    setSourcePanels(prev => ({ ...prev, [index]: !prev[index] }))
  }, [])

  const sessionStatus = session?.status
  const isProcessing = sessionStatus === 'queued' || sessionStatus === 'processing'
  const previewHref = session?.file_meta?.rel_path
    ? `/preview/${encodeURIComponent(session.file_meta.rel_path)}?embedded=1`
    : null

  const handleClearChat = useCallback(async () => {
    if (!session) {
      return
    }
    setClearing(true)
    try {
      const resp = await fetch('/api/doc-chat/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: session.id }),
      })
      const data = await resp.json()
      if (!resp.ok || !data?.ok) {
        throw new Error(data?.error || 'Не удалось очистить чат')
      }
      setSession(data.session || null)
      toasts.push('Диалог очищен', 'info')
    } catch (error: any) {
      toasts.push(error?.message || 'Не удалось очистить чат', 'error')
    } finally {
      setClearing(false)
    }
  }, [session, toasts])

  return (
    <div className="container-fluid d-grid gap-3 px-4">
      <div className="d-flex flex-wrap justify-content-between align-items-center gap-2">
        <h1 className="h4 mb-0">Документ-чат</h1>
        <div className="text-muted small">
          Подготовьте документ слева и ведите диалог по содержанию.
        </div>
      </div>

      <div className="row g-3 align-items-start">
        <div className="col-12 col-lg-4 col-xxl-3">
          <div className="card doc-chat-select-card h-100">
            <div className="card-header d-flex justify-content-between align-items-center">
              <div className="fw-semibold">Доступные документы</div>
              <button
                className="btn btn-outline-secondary btn-sm"
                type="button"
                onClick={() => { void fetchDocuments() }}
                disabled={loadingDocs}
              >
                {loadingDocs ? (
                  <>
                    <span className="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true" />
                    Обновляю…
                  </>
                ) : 'Обновить'}
              </button>
            </div>
            <div className="card-body d-grid gap-3">
              <div>
                <label className="form-label" htmlFor="docChatSearch">Поиск</label>
                <input
                  id="docChatSearch"
                  className="form-control"
                  placeholder="Название, автор, коллекция…"
                  value={searchTerm}
                  onChange={event => setSearchTerm(event.target.value)}
                />
              </div>
              <div>
                <label className="form-label" htmlFor="docChatSelect">Документ</label>
                <select
                  id="docChatSelect"
                  className="form-select doc-chat-select"
                  value={selectedId === '' ? '' : String(selectedId)}
                  onChange={event => {
                    const value = event.target.value
                    setSelectedId(value ? Number(value) : '')
                  }}
                  size={Math.min(filteredDocuments.length, 12) || undefined}
                >
                  <option value="">— Выберите документ —</option>
                  {filteredDocuments.map(doc => (
                    <option key={doc.id} value={doc.id}>
                      {doc.doc_chat_ready ? '✅ ' : ''}{doc.title}{doc.author ? ` · ${doc.author}` : ''}{doc.year ? ` · ${doc.year}` : ''}{doc.collection ? ` · ${doc.collection}` : ''}
                    </option>
                  ))}
                </select>
              </div>
              <div className="d-grid gap-2">
                <button
                  className="btn btn-primary"
                  type="button"
                  onClick={() => { void handlePrepare() }}
                  disabled={!selectedId || preparing}
                >
                  {preparing ? (
                    <>
                      <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true" />
                      Подготавливаю…
                    </>
                  ) : 'Подготовить'}
                </button>
                <button
                  className="btn btn-outline-secondary"
                  type="button"
                  onClick={() => { void handlePrepare({ force: true }) }}
                  disabled={!selectedId || preparing}
                >
                  {preparing ? 'Обработка…' : 'Пересканировать'}
                </button>
              </div>
              {docError && (
                <div className="alert alert-warning mb-0" role="alert">
                  {docError}
                </div>
              )}
              {!docError && !loadingDocs && documents.length === 0 && (
                <div className="alert alert-info mb-0" role="alert">
                  Документы недоступны. Проверьте права доступа или обновите список.
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="col-12 col-lg-8 col-xxl-6 d-grid gap-3">
          {session && isProcessing && (
            <ProgressPanel
              title={session.status === 'queued' ? 'В очереди на обработку…' : 'Анализ документа…'}
              caption="Статус подготовки включает извлечение текста, генерацию эмбеддингов и анализ изображений."
              progress={typeof session.percent === 'number' ? { percent: Math.round(session.percent), label: `${Math.round(session.percent)}%` } : undefined}
              bullets={(session.progress || []).map((line, idx) => ({ id: `${idx}-${line}`, text: line }))}
            />
          )}
          {session?.status === 'error' && (
            <div className="alert alert-danger mb-0" role="alert">
              {session.error || 'Не удалось подготовить документ.'}
            </div>
          )}
          <div className="card h-100 doc-chat-dialog-card">
            <div className="card-header d-flex flex-wrap align-items-center justify-content-between gap-2">
              <div>
                <div className="fw-semibold">Диалог</div>
                <div className="text-muted small">Помощник отвечает с учётом текста и изображений документа</div>
              </div>
              <button
                type="button"
                className="btn btn-outline-secondary btn-sm"
                onClick={handleClearChat}
                disabled={clearing || !hasHistory || !session || session.status !== 'ready'}
              >
                {clearing ? (
                  <>
                    <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true" />
                    Очищаю…
                  </>
                ) : 'Очистить чат'}
              </button>
            </div>
            <div className="card-body d-flex flex-column gap-3" style={{ maxHeight: '75vh' }}>
              <div className="flex-grow-1 overflow-auto d-grid gap-3 pe-1 doc-chat-message-stack">
                {(!session || history.length === 0) && !assistantThinking && (
                  <div className="doc-chat-empty text-muted">
                    Выберите документ слева, дождитесь подготовки и задайте первый вопрос.
                  </div>
                )}
                {history.map((msg, idx) => {
                  const isAssistant = msg.role === 'assistant'
                  const sources = msg.sources
                  const open = sourcePanels[idx] ?? false
                  const hasSources = Boolean(sources?.texts?.length || sources?.images?.length)
                  const bodyHtml = renderMarkdown(msg.content || '')
                  return (
                    <div
                      key={idx}
                      className={`doc-chat-message ${isAssistant ? 'doc-chat-message--assistant' : 'doc-chat-message--user'}`}
                    >
                      <div className="doc-chat-message__header">
                        <span className="doc-chat-message__author">
                          {isAssistant ? 'Помощник' : 'Вы'}
                        </span>
                        {isAssistant && hasSources && (
                          <button
                            type="button"
                            className="btn btn-outline-secondary btn-sm doc-chat-source-toggle"
                            onClick={() => toggleSources(idx)}
                          >
                            {open ? 'Скрыть источники' : `Источники (${(sources?.texts?.length || 0) + (sources?.images?.length || 0)})`}
                          </button>
                        )}
                      </div>
                      <div className="doc-chat-message__body" dangerouslySetInnerHTML={{ __html: bodyHtml }} />
                      {isAssistant && hasSources && open && (
                        <div className="doc-chat-source-list">
                          {sources?.texts?.map((src, srcIdx) => {
                            const metaParts: string[] = []
                            if (typeof src.page === 'number') metaParts.push(`Стр. ${src.page}`)
                            if (src.section_path) metaParts.push(src.section_path)
                            const highlightsHtml = renderMarkdown((src.highlights || []).join('\n'))
                            return (
                              <div key={`text-${idx}-${srcIdx}`} className="doc-chat-source-card">
                                <div className="doc-chat-source-card__title">
                                  <span className="badge bg-primary-subtle text-primary-emphasis">{src.label}</span>
                                  {typeof src.score === 'number' && (
                                    <span className="doc-chat-source-card__score">score {src.score.toFixed(3)}</span>
                                  )}
                                </div>
                                {src.matched_terms && src.matched_terms.length > 0 && (
                                  <div className="doc-chat-source-card__terms">
                                    Совпало: {src.matched_terms.join(', ')}
                                  </div>
                                )}
                                {src.highlights && src.highlights.length > 0 ? (
                                  <div dangerouslySetInnerHTML={{ __html: highlightsHtml }} className="doc-chat-source-card__highlights" />
                                ) : (
                                  <div className="doc-chat-source-card__preview">
                                    {src.preview || 'Фрагмент недоступен.'}
                                  </div>
                                )}
                                {metaParts.length > 0 && (
                                  <div className="doc-chat-source-card__meta">
                                    {metaParts.join(' · ')}
                                  </div>
                                )}
                              </div>
                            )
                          })}
                          {sources?.images?.map((img, imgIdx) => (
                            <div key={`image-${idx}-${imgIdx}`} className="doc-chat-source-card doc-chat-source-card--image">
                              <div className="doc-chat-source-card__title">
                                <span className="badge bg-info text-dark">{img.label}</span>
                                {typeof img.score === 'number' && (
                                  <span className="doc-chat-source-card__score">score {img.score.toFixed(3)}</span>
                                )}
                              </div>
                              {img.description && (
                                <div className="doc-chat-source-card__preview">
                                  {img.description}
                                </div>
                              )}
                              {img.url && (
                                <a
                                  href={img.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="doc-chat-source-card__link"
                                >
                                  Открыть изображение →
                                </a>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )
                })}
                {assistantThinking && (
                  <div className="doc-chat-message doc-chat-message--assistant doc-chat-message--thinking">
                    <div className="doc-chat-message__header">
                      <span className="doc-chat-message__author">Помощник</span>
                    </div>
                    <div className="doc-chat-message__body d-flex align-items-center gap-2">
                      <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true" />
                      Думаю над ответом…
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
              <div className="doc-chat-suggestions">
                {quickPrompts.map(item => (
                  <button
                    key={item.id}
                    type="button"
                    className="btn btn-outline-secondary btn-sm"
                    disabled={sending || session?.status !== 'ready'}
                    onClick={() => {
                      setQuestion(item.prompt)
                      questionInputRef.current?.focus()
                    }}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
              <form className="d-grid gap-2 doc-chat-input-form" onSubmit={handleAsk}>
                <textarea
                  ref={questionInputRef}
                  className="form-control doc-chat-input"
                  placeholder="Задайте вопрос по документу…"
                  value={question}
                  rows={3}
                  onChange={event => setQuestion(event.target.value)}
                  onKeyDown={event => {
                    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
                      event.preventDefault()
                      event.currentTarget.form?.requestSubmit()
                    }
                  }}
                  disabled={sending || !session || session.status !== 'ready'}
                />
                <div className="d-flex align-items-center justify-content-between gap-2 flex-wrap">
                  <span className="text-muted small">
                    Нажмите Ctrl+Enter, чтобы отправить быстрее.
                  </span>
                  <button
                    className="btn btn-primary"
                    type="submit"
                    disabled={sending || !session || session.status !== 'ready'}
                  >
                    {sending ? (
                      <>
                        <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true" />
                        Отправка…
                      </>
                    ) : 'Отправить'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>

        <div className="col-12 col-lg-12 col-xxl-3 d-grid gap-3">
          {session?.status === 'ready' ? (
            <>
              {previewHref && (
                <a
                  href={previewHref}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-outline-primary"
                >
                  Открыть документ в просмотре
                </a>
              )}
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
            </>
          ) : (
            <div className="card h-100">
              <div className="card-body d-flex align-items-center justify-content-center text-muted">
                Выберите документ слева, чтобы увидеть его данные.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default DocumentChatPage
