import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import DOMPurify from 'dompurify'
import { materialTypeRu, materialTypeSlug } from '../../utils/locale'
import { useToasts } from '../../ui/Toasts'
import type { CollectionOption, FacetData, FileItem, ProgressItem, Tag } from './types'

const AI_ANSWER_ALLOWED_TAGS = ['a', 'b', 'strong', 'em', 'i', 'code', 'pre', 'p', 'br', 'ul', 'ol', 'li', 'span', 'div']
const AI_ANSWER_ALLOWED_ATTRS = ['href', 'title', 'target', 'rel', 'class']

let dompurifyConfigured = false
const configureDomPurify = () => {
  if (dompurifyConfigured) return
  if (typeof window === 'undefined') return
  DOMPurify.addHook('afterSanitizeAttributes', node => {
    if (node.tagName === 'A') {
      const href = node.getAttribute('href') || ''
      if (/^javascript:/i.test(href)) {
        node.removeAttribute('href')
      }
      if (node.getAttribute('target') === '_blank') {
        node.setAttribute('rel', 'noopener noreferrer')
      }
    }
    node.removeAttribute('style')
  })
  dompurifyConfigured = true
}

function useDebounced<T>(value: T, delay = 400) {
  const [state, setState] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setState(value), delay)
    return () => clearTimeout(id)
  }, [value, delay])
  return state
}

export function useCatalogueState() {
  const [sp, setSp] = useSearchParams()
  const q = sp.get('q') || ''
  const type = sp.get('type') || ''
  const year_from = sp.get('year_from') || ''
  const year_to = sp.get('year_to') || ''
  const size_min = sp.get('size_min') || ''
  const size_max = sp.get('size_max') || ''
  const collectionId = sp.get('collection_id') || ''
  const page = Math.max(parseInt(sp.get('page') || '1'), 1)
  const commit = sp.get('commit') || ''
  const perPage = 50
  const offset = (page - 1) * perPage

  const dq = useDebounced(q, 350)
  const [aiMode, setAiMode] = useState(false)
  const [items, setItems] = useState<FileItem[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [aiLoading, setAiLoading] = useState(false)
  const [aiProgress, setAiProgress] = useState<string[]>([])
  const [aiTopK, setAiTopK] = useState<number>(3)
  const [aiDeepSearch, setAiDeepSearch] = useState<boolean>(true)
  const [aiUseTags, setAiUseTags] = useState<boolean>(true)
  const [aiUseText, setAiUseText] = useState<boolean>(true)
  const [aiMaxCandidates, setAiMaxCandidates] = useState<number>(15)
  const [aiChunkChars, setAiChunkChars] = useState<number>(5000)
  const [aiMaxChunks, setAiMaxChunks] = useState<number>(40)
  const [aiMaxSnippets, setAiMaxSnippets] = useState<number>(3)
  const [aiFullText, setAiFullText] = useState<boolean>(false)
  const [aiUseLlmSnippets, setAiUseLlmSnippets] = useState<boolean>(false)
  const [aiAllLanguages, setAiAllLanguages] = useState<boolean>(false)
  const [showAiSettings, setShowAiSettings] = useState<boolean>(false)
  const [facets, setFacets] = useState<FacetData | null>(null)
  const [facetsLoading, setFacetsLoading] = useState<boolean>(false)
  const [previewRel, setPreviewRel] = useState<string | null>(null)
  const [selectedTags, setSelectedTags] = useState<string[]>(() => sp.getAll('tag'))
  const [aiAnswer, setAiAnswer] = useState<string>('')
  const [aiKeywords, setAiKeywords] = useState<string[]>([])
  const [aiSources, setAiSources] = useState<any[]>([])
  const [aiFilteredKeywords, setAiFilteredKeywords] = useState<string[]>([])
  const [aiQueryHash, setAiQueryHash] = useState<string>('')
  const [feedbackStatus, setFeedbackStatus] = useState<Record<string, string>>({})
  const [speechSupported, setSpeechSupported] = useState<boolean>(() => typeof window !== 'undefined' && 'speechSynthesis' in window && typeof SpeechSynthesisUtterance !== 'undefined')
  const [autoSpeakAnswer, setAutoSpeakAnswer] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false
    try {
      return localStorage.getItem('catalogue.autoSpeakAnswer') === '1'
    } catch {
      return false
    }
  })
  const [speechState, setSpeechState] = useState<'idle' | 'speaking'>('idle')
  const [speechError, setSpeechError] = useState<string | null>(null)
  const sentinelRef = useRef<HTMLDivElement | null>(null)
  const speechUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null)
  const [localPage, setLocalPage] = useState(page)
  const [editItem, setEditItem] = useState<FileItem | null>(null)
  const [editForm, setEditForm] = useState<any>(null)
  const [collections, setCollections] = useState<CollectionOption[]>([])
  const toasts = useToasts()
  const logAction = useCallback((message: string, level: 'info' | 'error' | 'success' = 'info') => {
    const prefixed = `[catalogue] ${message}`
    if (level === 'error') {
      console.error(prefixed)
    } else if (level === 'success') {
      console.info(prefixed)
    } else {
      console.debug(prefixed)
    }
  }, [])

  const progressIconFor = useCallback((line: string): string => {
    const lower = line.toLowerCase()
    if (lower.startsWith('запрос')) return '📝'
    if (lower.includes('ключевые термины')) return '🔑'
    if (lower.includes('фильтр')) return '🧹'
    if (lower.includes('кандидат')) return '📄'
    if (lower.includes('глубок')) return '🔎'
    if (lower.includes('ранжирование')) return '🏁'
    if (lower.includes('llm ответ')) return '💬'
    if (lower.includes('llm сниппеты')) return '🧠'
    if (lower.includes('полнотекст')) return '📚'
    return '•'
  }, [])

  const progressItems = useMemo<ProgressItem[]>(() => aiProgress.map((line, idx) => ({
    id: `${idx}-${line}`,
    line,
    icon: progressIconFor(line),
  })), [aiProgress, progressIconFor])

  const aiAnswerPlain = useMemo(() => {
    if (!aiAnswer) return ''
    const raw = String(aiAnswer)
    if (typeof window === 'undefined') {
      return raw.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim()
    }
    const tmp = document.createElement('div')
    tmp.innerHTML = raw
    const text = tmp.textContent || tmp.innerText || ''
    return text.replace(/\s+/g, ' ').trim()
  }, [aiAnswer])

  const safeAiAnswer = useMemo(() => {
    if (!aiAnswer) return ''
    configureDomPurify()
    return DOMPurify.sanitize(String(aiAnswer), {
      ALLOWED_TAGS: AI_ANSWER_ALLOWED_TAGS,
      ALLOWED_ATTR: AI_ANSWER_ALLOWED_ATTRS,
      FORBID_TAGS: ['style', 'script'],
      FORBID_ATTR: ['style', 'onerror', 'onclick']
    })
  }, [aiAnswer])

  const sendFeedback = useCallback(async (payload: { action: string; file_id?: number; keyword?: string; score?: number }, onSuccess?: () => void) => {
    if (!aiQueryHash) return
    const ident = payload.file_id !== undefined && payload.file_id !== null ? `file-${payload.file_id}` : `kw-${payload.keyword}`
    const key = `${payload.action}:${ident}`
    setFeedbackStatus(prev => ({ ...prev, [key]: 'loading' }))
    try {
      const resp = await fetch('/api/ai-search/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...payload, query_hash: aiQueryHash }),
      })
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }
      setFeedbackStatus(prev => ({ ...prev, [key]: 'done' }))
      onSuccess?.()
    } catch (error) {
      console.error('feedback error', error)
      setFeedbackStatus(prev => ({ ...prev, [key]: 'error' }))
    }
  }, [aiQueryHash])

  const handleSourceFeedback = useCallback((fileId: number, action: 'relevant' | 'irrelevant') => {
    if (!aiQueryHash) return
    sendFeedback({ file_id: fileId, action })
  }, [aiQueryHash, sendFeedback])

  const handleKeywordFeedback = useCallback((keyword: string, action: 'relevant' | 'irrelevant') => {
    if (!aiQueryHash) return
    sendFeedback({ keyword, action })
  }, [aiQueryHash, sendFeedback])

  const handleKeywordRestore = useCallback((keyword: string) => {
    if (!aiQueryHash) return
    sendFeedback({ keyword, action: 'relevant' }, () => {
      setAiFilteredKeywords(prev => prev.filter(k => k !== keyword))
    })
  }, [aiQueryHash, sendFeedback])

  const stopSpeakingAnswer = useCallback(() => {
    if (!speechSupported) {
      setSpeechState('idle')
      setSpeechError(null)
      return
    }
    try {
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel()
      }
    } catch (error) {
      console.debug('speechSynthesis cancel failed', error)
    }
    speechUtteranceRef.current = null
    setSpeechState('idle')
    setSpeechError(null)
  }, [speechSupported])

  const speakAnswer = useCallback(() => {
    if (!speechSupported) return
    const text = aiAnswerPlain.trim()
    if (!text) return
    stopSpeakingAnswer()
    try {
      const utterance = new SpeechSynthesisUtterance(text)
      const cyrCount = (text.match(/[А-Яа-яЁё]/g) || []).length
      const latCount = (text.match(/[A-Za-z]/g) || []).length
      utterance.lang = cyrCount >= latCount ? 'ru-RU' : 'en-US'
      utterance.rate = 1
      utterance.onend = () => {
        speechUtteranceRef.current = null
        setSpeechState('idle')
      }
      utterance.onerror = () => {
        speechUtteranceRef.current = null
        setSpeechState('idle')
        setSpeechError('Не удалось озвучить ответ')
      }
      speechUtteranceRef.current = utterance
      setSpeechError(null)
      setSpeechState('speaking')
      window.speechSynthesis.speak(utterance)
    } catch (error) {
      speechUtteranceRef.current = null
      setSpeechState('idle')
      setSpeechError('Ошибка синтеза речи')
      console.error('speechSynthesis speak failed', error)
    }
  }, [aiAnswerPlain, speechSupported, stopSpeakingAnswer])

  const canSpeakAnswer = useMemo(() => speechSupported && aiAnswerPlain.length > 0, [speechSupported, aiAnswerPlain])

  const availableLanguages = useMemo(() => {
    const langFacet = facets?.tag_facets?.lang
    if (!langFacet || !Array.isArray(langFacet)) return []
    const seen = new Set<string>()
    const list: string[] = []
    for (const entry of langFacet) {
      if (!Array.isArray(entry) || entry.length === 0) continue
      const rawValue = entry[0]
      const code = String(rawValue || '').trim()
      if (!code) continue
      const key = code.toLowerCase()
      if (seen.has(key)) continue
      seen.add(key)
      list.push(code)
    }
    return list
  }, [facets])

  const resetAiState = useCallback(() => {
    setAiMode(false)
    setAiProgress([])
    setAiAnswer('')
    setAiKeywords([])
    setAiSources([])
    setAiFilteredKeywords([])
    setAiQueryHash('')
    setFeedbackStatus({})
    setAiAllLanguages(false)
    stopSpeakingAnswer()
  }, [stopSpeakingAnswer])

  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        localStorage.setItem('catalogue.autoSpeakAnswer', autoSpeakAnswer ? '1' : '0')
      } catch {
        // Игнорируем сбои доступа к localStorage
      }
    }
    if (!autoSpeakAnswer) {
      stopSpeakingAnswer()
    }
  }, [autoSpeakAnswer, stopSpeakingAnswer])

  useEffect(() => {
    if (!autoSpeakAnswer) return
    if (!speechSupported) return
    if (!aiAnswerPlain) return
    speakAnswer()
  }, [autoSpeakAnswer, aiAnswerPlain, speechSupported, speakAnswer])

  const openEdit = useCallback((file: FileItem) => {
    setEditItem(file)
    setEditForm({
      title: file.title || '',
      author: file.author || '',
      year: file.year || '',
      material_type: materialTypeRu(file.material_type, file.material_type || ''),
      filename: '',
      keywords: '',
      tagsText: (file.tags || []).map(t => `${t.key}=${t.value}`).join('\n')
    })
  }, [])

  const updateTagsInline = useCallback(async (file: FileItem, tags: Tag[]) => {
    try {
      const payload: any = { title: file.title, author: file.author, year: file.year, material_type: file.material_type, tags }
      const r = await fetch(`/api/files/${file.id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      if (r.ok) {
        const upd = await r.json()
        setItems(list => list.map(x => x.id === upd.id ? upd : x))
      }
    } catch (error) {
      console.error('updateTagsInline', error)
    }
  }, [])

  const saveEdit = useCallback(async () => {
    if (!editItem || !editForm) return
    const arr = (editForm.tagsText || '').split(/\n|;/).map((s: string) => s.trim()).filter(Boolean)
    const tags = arr.map((s: string) => {
      const i = s.indexOf('=')
      if (i === -1) return null
      return { key: s.slice(0, i).trim(), value: s.slice(i + 1).trim() }
    }).filter(Boolean) as Tag[]
    const normalizedMaterialType = materialTypeSlug(editForm.material_type)
    const payload: any = {
      title: editForm.title || null,
      author: editForm.author || null,
      year: editForm.year || null,
      material_type: normalizedMaterialType,
      keywords: editForm.keywords || null,
      tags
    }
    if ((editForm.filename || '').trim()) payload.filename = editForm.filename.trim()
    const r = await fetch(`/api/files/${editItem.id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
    if (r.ok) {
      const upd = await r.json()
      setItems(list => list.map(x => x.id === upd.id ? upd : x))
      setEditItem(null)
    } else {
      alert('Не удалось сохранить')
    }
  }, [editForm, editItem])

  const refreshFile = useCallback(async (file: FileItem) => {
    const url = `/api/files/${file.id}/refresh?use_llm=1&kws_audio=1&summarize=1`
    logAction(`REFRESH ${url}`)
    try {
      const response = await fetch(url, { method: 'POST' })
      const text = await response.text()
      if (response.ok) {
        let payload: any = null
        try { payload = JSON.parse(text) } catch {}
        if (payload && payload.file) {
          setItems(list => list.map(x => x.id === payload.file.id ? payload.file : x))
        }
        if (payload && Array.isArray(payload.log)) {
          payload.log.forEach((entry: any) => logAction(String(entry), 'info'))
        }
        toasts.push('Теги обновлены', 'success')
        logAction(`OK ${file.id} tags updated`, 'success')
      } else {
        toasts.push('Ошибка обновления тегов', 'error')
        logAction(`ERROR ${response.status} ${text.slice(0, 500)}`, 'error')
      }
    } catch (error: any) {
      toasts.push('Ошибка сети при обновлении', 'error')
      logAction(`NETWORK ${String(error)}`, 'error')
    }
  }, [logAction, toasts])

  const renameFile = useCallback(async (file: FileItem) => {
    logAction(`RENAME-SUGGEST /api/files/${file.id}/rename-suggest`)
    const suggestion = await fetch(`/api/files/${file.id}/rename-suggest`).then(r => r.json()).catch(() => null)
    const newName = suggestion && (suggestion.new_name || suggestion.suggested || suggestion.name)
    if (!newName) {
      alert('Не удалось получить предложение')
      return
    }
    const edited = prompt('Новое имя файла', String(newName))
    if (edited === null) return
    const finalName = edited.trim()
    if (!finalName) return
    logAction(`RENAME /api/files/${file.id}/rename -> ${finalName}`)
    try {
      const resp = await fetch(`/api/files/${file.id}/rename`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ base: finalName })
      })
      const text = await resp.text()
      if (resp.ok) {
        let payload: any = null
        try { payload = JSON.parse(text) } catch {}
        if (payload && payload.file) {
          setItems(list => list.map(x => x.id === payload.file.id ? payload.file : x))
          if (previewRel === file.rel_path && payload.file.rel_path) setPreviewRel(payload.file.rel_path)
        } else {
          const fresh = await fetch(`/api/files/${file.id}`).then(r => r.json()).catch(() => null)
          if (fresh) {
            setItems(list => list.map(x => x.id === fresh.id ? fresh : x))
            if (previewRel === file.rel_path && fresh.rel_path) setPreviewRel(fresh.rel_path)
          }
        }
        toasts.push('Файл переименован', 'success')
        logAction('OK rename', 'success')
      } else {
        toasts.push('Ошибка переименования', 'error')
        logAction(`ERROR ${resp.status} ${text.slice(0, 500)}`, 'error')
      }
    } catch (error: any) {
      toasts.push('Сбой сети при переименовании', 'error')
      logAction(`NETWORK ${String(error)}`, 'error')
    }
  }, [logAction, previewRel, toasts])

  const deleteFile = useCallback(async (file: FileItem) => {
    if (!confirm('Удалить запись?')) return
    try {
      const resp = await fetch(`/api/files/${file.id}`, { method: 'DELETE' })
      if (resp.status === 204) {
        setItems(list => list.filter(x => x.id !== file.id))
        setTotal(count => Math.max(0, count - 1))
        setAiSources(src => (Array.isArray(src) ? src.filter((s: any) => s?.file_id !== file.id) : src))
        setFeedbackStatus(prev => {
          const next = { ...prev }
          Object.keys(next).forEach(key => {
            if (key.startsWith(`relevant:file-${file.id}`) || key.startsWith(`irrelevant:file-${file.id}`)) {
              delete next[key]
            }
          })
          return next
        })
        if (previewRel === file.rel_path) setPreviewRel(null)
        toasts.push('Файл отправлен в удалённые', 'success')
      } else {
        const text = await resp.text().catch(() => '')
        console.error('delete failed', resp.status, text)
        toasts.push('Не удалось удалить файл', 'error')
      }
    } catch (error) {
      console.error('delete request failed', error)
      toasts.push('Сбой сети при удалении', 'error')
    }
  }, [previewRel, toasts])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setPreviewRel(null) }
    if (previewRel) window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [previewRel])

  const params = useMemo(() => {
    const p = new URLSearchParams()
    if (dq) p.set('q', dq)
    if (dq) p.set('smart', '1')
    if (type) p.set('type', type)
    if (year_from) p.set('year_from', year_from)
    if (year_to) p.set('year_to', year_to)
    if (size_min) p.set('size_min', size_min)
    if (size_max) p.set('size_max', size_max)
    if (collectionId) p.set('collection_id', collectionId)
    return p
  }, [dq, type, year_from, year_to, size_min, size_max, collectionId])

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch('/api/collections')
        if (!r.ok) return
        const data = await r.json().catch(() => [])
        if (Array.isArray(data)) {
          setCollections(data)
        }
      } catch (error) {
        console.error('load collections', error)
      }
    })()
  }, [])

  useEffect(() => {
    if (!aiMode) {
      setShowAiSettings(false)
      setAiProgress([])
    }
  }, [aiMode])

  useEffect(() => {
    if (typeof window === 'undefined') {
      setSpeechSupported(false)
      setAutoSpeakAnswer(false)
      return
    }
    const supported = 'speechSynthesis' in window && typeof SpeechSynthesisUtterance !== 'undefined'
    setSpeechSupported(supported)
    if (!supported) {
      setAutoSpeakAnswer(false)
    }
    return () => {
      if ('speechSynthesis' in window) {
        try {
          window.speechSynthesis.cancel()
        } catch (error) {
          console.debug('speechSynthesis cancel failed', error)
        }
      }
    }
  }, [])

  useEffect(() => {
    if (!speechSupported) {
      setSpeechError(null)
      return
    }
    if (!speechUtteranceRef.current) {
      setSpeechError(null)
      return
    }
    try {
      window.speechSynthesis.cancel()
    } catch (error) {
      console.debug('speechSynthesis cancel on answer change failed', error)
    }
    speechUtteranceRef.current = null
    setSpeechState('idle')
    setSpeechError(null)
  }, [aiAnswer, speechSupported])

  useEffect(() => {
    let cancelled = false
    let controller: AbortController | null = null

    const abortCurrent = () => {
      if (controller) {
        controller.abort()
        controller = null
      }
    }

    const hydrateItems = async (items: any[]) => {
      const ids: number[] = (items || []).map((x: any) => x?.file_id).filter(Boolean)
      if (!ids.length) {
        if (!cancelled) {
          setItems([])
          setTotal(0)
          setLocalPage(1)
        }
        return
      }
      const rows = await Promise.all(ids.map((id: number) => fetch(`/api/files/${id}`).then(r => r.json()).catch(() => null)))
      if (cancelled) return
      const clean = rows.filter(Boolean) as any[]
      setItems(clean as any)
      setTotal(clean.length)
      setLocalPage(1)
    }

    const runAiSearch = async (payload: any) => {
      abortCurrent()
      const localController = new AbortController()
      controller = localController
      try {
        const resp = await fetch('/api/ai-search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: localController.signal,
        })
        if (!resp.ok) {
          const text = await resp.text().catch(() => '')
          throw new Error(text || `HTTP ${resp.status}`)
        }
        const data = await resp.json()
        if (cancelled) return
        setAiQueryHash(String(data.query_hash || ''))
        const items = Array.isArray(data.items) ? data.items : []
        const progressLines = Array.isArray(data.progress) && data.progress.length ? data.progress : ['Поиск завершён']
        setAiProgress(progressLines)
        if (cancelled) return
        setAiSources(items)
        setAiAnswer(data.answer || '')
        setAiKeywords(Array.isArray(data.keywords) ? data.keywords : [])
        setAiFilteredKeywords(Array.isArray(data.filtered_keywords) ? data.filtered_keywords : [])
        await hydrateItems(items)
      } catch (error: any) {
        if (cancelled || (error && error.name === 'AbortError')) return
        throw error
      } finally {
        if (controller === localController) {
          controller = null
        }
      }
    }

    const load = async () => {
      setLoading(true)
      try {
        if (aiMode && dq && commit) {
          setFeedbackStatus({})
          setAiLoading(true)
          setAiProgress(['Формируем запрос…'])
          setAiAnswer('')
          setAiKeywords([])
          setAiSources([])
          setAiFilteredKeywords([])
          setAiQueryHash('')
          const payload: any = { query: dq, top_k: aiTopK, deep_search: aiDeepSearch }
          payload.sources = { tags: aiUseTags, text: aiUseText }
          payload.max_candidates = aiMaxCandidates
          payload.chunk_chars = aiChunkChars
          payload.max_chunks = aiMaxChunks
          payload.max_snippets = aiMaxSnippets
          payload.full_text = aiFullText
          payload.llm_snippets = aiUseLlmSnippets
          payload.all_languages = aiAllLanguages
          if (aiAllLanguages) {
            const langs = availableLanguages.filter(code => typeof code === 'string' && code.trim().length > 0)
            if (langs.length) payload.languages = langs
          }
          if (collectionId) payload.collection_id = Number(collectionId)
          if (type) payload.material_types = [type]
          if (year_from) payload.year_from = year_from
          if (year_to) payload.year_to = year_to
          if (selectedTags.length) payload.tag_filters = selectedTags
          try {
            await runAiSearch(payload)
          } catch (error: any) {
            if (cancelled) return
            const rawMessage = error instanceof Error ? error.message : String(error)
            const normalized = /<!doctype/i.test(rawMessage) ? 'HTTP 404' : rawMessage
            setAiProgress([`Ошибка поиска: ${normalized}`])
            setAiAnswer('')
            setAiKeywords([])
          }
        } else {
          if (aiMode) {
            if (!dq) {
              setAiProgress(['Введите запрос и нажмите Enter'])
            } else {
              setAiProgress(['Нажмите Enter для запуска поиска'])
            }
            setAiAnswer('')
            setAiKeywords([])
            setAiSources([])
            setAiFilteredKeywords([])
            setAiQueryHash('')
            setLocalPage(page)
            return
          }
          setAiAnswer('')
          setAiKeywords([])
          setAiSources([])
          setAiFilteredKeywords([])
          const p = new URLSearchParams(params)
          p.set('limit', String(perPage))
          p.set('offset', String(offset))
          selectedTags.forEach(t => p.append('tag', t))
          const res = await fetch(`/api/search_v2?${p.toString()}`)
          const data = await res.json()
          if (cancelled) return
          if (offset > 0) setItems(prev => [...prev, ...(data.items || [])])
          else setItems(data.items || [])
          setTotal(data.total || 0)
          setLocalPage(page)
        }
      } catch (error) {
        if (!cancelled) {
          console.error('Search error:', error)
          if (aiMode) {
            setAiProgress([`Ошибка поиска: ${error instanceof Error ? error.message : String(error)}`])
            setAiAnswer('')
            setAiKeywords([])
          }
        }
      } finally {
        if (!cancelled) {
          if (aiMode) setAiLoading(false)
          setLoading(false)
        }
      }
    }

    load()

    return () => {
      cancelled = true
      abortCurrent()
    }
  }, [params, offset, selectedTags, aiMode, commit, aiTopK, aiDeepSearch, aiUseTags, aiUseText, aiMaxCandidates, aiChunkChars, aiMaxChunks, aiMaxSnippets, aiFullText, aiUseLlmSnippets, aiAllLanguages, availableLanguages, dq, collectionId, type, year_from, year_to, page])

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      setFacetsLoading(true)
      try {
        const p = new URLSearchParams(params)
        p.delete('commit')
        p.set('context', 'search')
        selectedTags.forEach(t => p.append('tag', t))
        const res = await fetch(`/api/facets?${p.toString()}`)
        const data = await res.json().catch(() => null)
        if (cancelled) return
        if (res.ok && data) {
          const payload: FacetData = {
            types: Array.isArray(data.types) ? data.types : [],
            tag_facets: data.tag_facets && typeof data.tag_facets === 'object' ? data.tag_facets : {},
            include_types: data.include_types !== undefined ? Boolean(data.include_types) : true,
            allowed_keys: Array.isArray(data.allowed_keys) ? data.allowed_keys : null,
          }
          setFacets(payload)
        } else {
          setFacets(null)
        }
      } catch (error) {
        if (!cancelled) {
          console.error('facets error:', error)
          setFacets(null)
        }
      } finally {
        if (!cancelled) setFacetsLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [params, selectedTags])

  useEffect(() => {
    if (!aiMode && sentinelRef.current) {
      const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            setSp(prev => {
              const next = new URLSearchParams(prev)
              const nextPage = Math.min(9999, Math.max(1, page + 1))
              next.set('page', String(nextPage))
              return next
            })
          }
        })
      }, { threshold: 1 })
      observer.observe(sentinelRef.current)
      return () => observer.disconnect()
    }
  }, [aiMode, page, setSp, sentinelRef])

  const removeTag = useCallback((index: number) => {
    setSelectedTags(prev => prev.filter((_, i) => i !== index))
    setSp(prev => {
      const next = new URLSearchParams(prev)
      const tags = next.getAll('tag')
      next.delete('tag')
      tags.filter((_, i) => i !== index).forEach(t => next.append('tag', t))
      next.set('page', '1')
      return next
    })
  }, [setSp])

  const pages = useMemo(() => Math.max(1, Math.ceil(total / perPage)), [total])

  const canLoadMore = useMemo(() => !aiMode && localPage < pages, [aiMode, localPage, pages])

  const addTagFilter = useCallback((tag: string) => {
    setSelectedTags(prev => [...prev, tag])
    setSp(prev => {
      const next = new URLSearchParams(prev)
      next.append('tag', tag)
      next.set('page', '1')
      return next
    })
  }, [setSp])

  const resetFilters = useCallback(() => {
    const params = new URLSearchParams()
    params.set('page', '1')
    setSelectedTags([])
    setAiAnswer('')
    setAiKeywords([])
    setAiMode(false)
    setAiProgress([])
    setSp(params)
  }, [setSp])

  return {
    searchParams: { sp, setSp, q, type, year_from, year_to, size_min, size_max, collectionId, dq, params },
    pagination: { page, pages, perPage, offset, localPage, canLoadMore },
    selectors: { selectedTags, setSelectedTags, removeTag, addTagFilter },
    list: { items, total, loading, sentinelRef },
    ai: {
      aiMode, setAiMode, aiLoading, aiProgress, progressItems, aiAnswer, safeAiAnswer, aiAnswerPlain,
      aiKeywords, aiSources, aiFilteredKeywords, aiQueryHash, feedbackStatus,
      aiTopK, setAiTopK, aiDeepSearch, setAiDeepSearch, aiUseTags, setAiUseTags,
      aiUseText, setAiUseText, aiMaxCandidates, setAiMaxCandidates, aiChunkChars, setAiChunkChars,
      aiMaxChunks, setAiMaxChunks, aiMaxSnippets, setAiMaxSnippets, aiFullText, setAiFullText,
      aiUseLlmSnippets, setAiUseLlmSnippets, aiAllLanguages, setAiAllLanguages, showAiSettings, setShowAiSettings,
      aiLanguageOptions: availableLanguages,
      aiLoadingState: { speechSupported, autoSpeakAnswer, setAutoSpeakAnswer, speechState, speechError, setSpeechError },
      handlers: { handleSourceFeedback, handleKeywordFeedback, handleKeywordRestore, speakAnswer, stopSpeakingAnswer },
      canSpeakAnswer,
      resetAiState,
    },
    modals: { previewRel, setPreviewRel, editItem, setEditItem, editForm, setEditForm, saveEdit, openEdit },
    helpers: { toasts, logAction, updateTagsInline, refreshFile, renameFile, deleteFile },
    collections,
    facets,
    facetsLoading,
    state: { aiUseTags, aiUseText },
    config: {
      aiMaxCandidates, aiChunkChars, aiMaxChunks, aiMaxSnippets, aiFullText, aiUseLlmSnippets,
    },
  }
}
