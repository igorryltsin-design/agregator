import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { materialTypeRu, tagKeyRu, materialTypeOptions, materialTypeSlug } from '../utils/locale'
import { useToasts } from '../ui/Toasts'

type Tag = { key: string; value: string }
type FileItem = {
  id: number
  title: string | null
  author: string | null
  year: string | null
  material_type: string | null
  rel_path: string | null
  keywords?: string | null
  text_excerpt?: string | null
  tags: Tag[]
}

type CollectionOption = { id: number; name: string }

type FacetData = {
  types: [string | null, number][]
  tag_facets: Record<string, [string, number][]>
  include_types?: boolean
  allowed_keys?: string[] | null
}

function useDebounced<T>(v: T, delay = 400) {
  const [val, setVal] = useState(v)
  useEffect(() => { const id = setTimeout(() => setVal(v), delay); return () => clearTimeout(id) }, [v, delay])
  return val
}

export default function Catalogue() {
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
  const [showAiSettings, setShowAiSettings] = useState<boolean>(false)
  const [facets, setFacets] = useState<FacetData | null>(null)
  const [previewRel, setPreviewRel] = useState<string| null>(null)
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
  const sentinelRef = useRef<HTMLDivElement|null>(null)
  const speechUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null)
  const [localPage, setLocalPage] = useState(page)
  const [editItem, setEditItem] = useState<FileItem | null>(null)
  const [editForm, setEditForm] = useState<any>(null)
  const [collections, setCollections] = useState<CollectionOption[]>([])
  const toasts = useToasts()
  const logAction = useCallback((message: string, level: 'info'|'error'|'success' = 'info') => {
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

  const progressItems = useMemo(() => aiProgress.map((line, idx) => ({
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

  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        localStorage.setItem('catalogue.autoSpeakAnswer', autoSpeakAnswer ? '1' : '0')
      } catch {
        // ignore storage errors
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

  const openEdit = (f: FileItem) => {
    setEditItem(f)
    setEditForm({
      title: f.title || '', author: f.author || '', year: f.year || '', material_type: materialTypeRu(f.material_type, f.material_type || ''), filename: '', keywords: '',
      tagsText: (f.tags||[]).map(t=>`${t.key}=${t.value}`).join('\n')
    })
  }
  const updateTagsInline = async (file: FileItem, tags: Tag[]) => {
    try{
      const payload: any = { title: file.title, author: file.author, year: file.year, material_type: file.material_type, tags }
      const r = await fetch(`/api/files/${file.id}`, { method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) })
      if (r.ok){ const upd = await r.json(); setItems(list=>list.map(x=>x.id===upd.id? upd : x)) }
    }catch{}
  }
  const saveEdit = async () => {
    if (!editItem || !editForm) return
    const arr = (editForm.tagsText||'').split(/\n|;/).map((s:string)=>s.trim()).filter(Boolean)
    const tags = arr.map((s:string)=>{ const i=s.indexOf('='); if(i===-1) return null; return {key:s.slice(0,i).trim(), value:s.slice(i+1).trim()} }).filter(Boolean) as any
    const normalizedMaterialType = materialTypeSlug(editForm.material_type)
    const payload: any = { title: editForm.title||null, author: editForm.author||null, year: editForm.year||null, material_type: normalizedMaterialType, keywords: editForm.keywords||null, tags }
    if ((editForm.filename||'').trim()) payload.filename = editForm.filename.trim()
    const r = await fetch(`/api/files/${editItem.id}`, { method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) })
    if (r.ok) { const upd = await r.json(); setItems(list=>list.map(x=>x.id===upd.id? upd : x)); setEditItem(null) } else { alert('Не удалось сохранить') }
  }

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
      } catch {}
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
    let cancelled = false;
    let controller: AbortController | null = null;

    const abortCurrent = () => {
      if (controller) {
        controller.abort();
        controller = null;
      }
    };

    const hydrateItems = async (items: any[]) => {
      const ids: number[] = (items || []).map((x: any) => x?.file_id).filter(Boolean);
      if (!ids.length) {
        if (!cancelled) {
          setItems([]);
          setTotal(0);
          setLocalPage(1);
        }
        return;
      }
      const rows = await Promise.all(ids.map((id: number) =>
        fetch(`/api/files/${id}`).then(r => r.json()).catch(() => null)
      ));
      if (cancelled) return;
      const clean = rows.filter(Boolean) as any[];
      setItems(clean as any);
      setTotal(clean.length);
      setLocalPage(1);
    };

    const runAiSearch = async (payload: any) => {
      abortCurrent();
      const localController = new AbortController();
      controller = localController;
      try {
        const resp = await fetch('/api/ai-search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: localController.signal,
        });
        if (!resp.ok) {
          const text = await resp.text().catch(() => '');
          throw new Error(text || `HTTP ${resp.status}`);
        }
        const data = await resp.json();
        if (cancelled) return;
        setAiQueryHash(String(data.query_hash || ''));
        const items = Array.isArray(data.items) ? data.items : [];
        const progressLines = Array.isArray(data.progress) && data.progress.length ? data.progress : ['Поиск завершён'];
        setAiProgress(progressLines);
        if (cancelled) return;
        setAiSources(items);
        setAiAnswer(data.answer || '');
        setAiKeywords(Array.isArray(data.keywords) ? data.keywords : []);
        setAiFilteredKeywords(Array.isArray(data.filtered_keywords) ? data.filtered_keywords : []);
        await hydrateItems(items);
      } catch (error: any) {
        if (cancelled || (error && error.name === 'AbortError')) return;
        throw error;
      } finally {
        if (controller === localController) {
          controller = null;
        }
      }
    };
    const load = async () => {
      setLoading(true);
      try {
        if (aiMode && dq && commit) {
          setFeedbackStatus({})
          setAiLoading(true);
          setAiProgress(['Формируем запрос…']);
          setAiAnswer('');
          setAiKeywords([]);
          setAiSources([]);
          setAiFilteredKeywords([])
          setAiQueryHash('')
          const payload: any = { query: dq, top_k: aiTopK, deep_search: aiDeepSearch };
          payload.sources = { tags: aiUseTags, text: aiUseText };
          payload.max_candidates = aiMaxCandidates;
          payload.chunk_chars = aiChunkChars;
          payload.max_chunks = aiMaxChunks;
          payload.max_snippets = aiMaxSnippets;
          payload.full_text = aiFullText;
          payload.llm_snippets = aiUseLlmSnippets;
          if (collectionId) payload.collection_id = Number(collectionId);
          if (type) payload.material_types = [type];
          if (year_from) payload.year_from = year_from;
          if (year_to) payload.year_to = year_to;
          if (selectedTags.length) payload.tag_filters = selectedTags;
          try {
            await runAiSearch(payload);
          } catch (error: any) {
            if (cancelled) return;
            const rawMessage = error instanceof Error ? error.message : String(error);
            const normalized = /<!doctype/i.test(rawMessage) ? 'HTTP 404' : rawMessage;
            setAiProgress([`Ошибка поиска: ${normalized}`]);
            setAiAnswer('');
            setAiKeywords([]);
          }
        } else {
          if (aiMode) {
            if (!dq) {
              setAiProgress(['Введите запрос и нажмите Enter']);
            } else {
              setAiProgress(['Нажмите Enter для запуска поиска']);
            }
            setAiAnswer('');
            setAiKeywords([]);
            setAiSources([]);
            setAiFilteredKeywords([])
            setAiQueryHash('')
            setLocalPage(page);
            return;
          }
          setAiAnswer('');
          setAiKeywords([]);
          setAiSources([]);
          setAiFilteredKeywords([])
          const p = new URLSearchParams(params);
          p.set('limit', String(perPage));
          p.set('offset', String(offset));
          selectedTags.forEach(t => p.append('tag', t));
          const res = await fetch(`/api/search_v2?${p.toString()}`);
          const data = await res.json();
          if (cancelled) return;
          if (offset > 0) setItems(prev => [...prev, ...(data.items || [])]);
          else setItems(data.items || []);
          setTotal(data.total || 0);
          setLocalPage(page);
        }
      } catch (error) {
        if (!cancelled) {
          console.error('Search error:', error);
          if (aiMode) {
            setAiProgress([`Ошибка поиска: ${error instanceof Error ? error.message : String(error)}`]);
            setAiAnswer('');
            setAiKeywords([]);
          }
        }
      } finally {
        if (!cancelled) {
          if (aiMode) setAiLoading(false);
          setLoading(false);
        }
      }
    };

    load();

    return () => {
      cancelled = true;
      abortCurrent();
    };
  }, [params, offset, selectedTags, aiMode, commit, aiTopK, aiDeepSearch, aiUseTags, aiUseText, aiMaxCandidates, aiChunkChars, aiMaxChunks, aiMaxSnippets, aiFullText, aiUseLlmSnippets, dq, collectionId, type, year_from, year_to, page]);

  useEffect(() => {
    let cancelled = false
    const load = async () => {
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
      }
    }
    load()
    return () => { cancelled = true }
  }, [params, selectedTags])

  const pages = Math.max(Math.ceil(total / perPage), 1)
  const removeTag = (idx: number) => {
    const next = selectedTags.filter((_, i) => i !== idx)
    setSelectedTags(next)
    sp.delete('tag'); next.forEach(t => sp.append('tag', t)); sp.set('page','1'); setSp(sp)
  }

  const handleResetFacets = () => {
    if (!type && selectedTags.length === 0) return
    setSelectedTags([])
    sp.delete('tag')
    sp.delete('type')
    sp.set('page', '1')
    setSp(sp)
  }

  const handleCollectionChange = (value: string) => {
    if (value) sp.set('collection_id', value); else sp.delete('collection_id')
    sp.set('page', '1')
    setSp(sp)
  }

  // Infinite scroll observer
  useEffect(() => {
    if (aiMode) return
    const el = sentinelRef.current
    if (!el) return
    const io = new IntersectionObserver((entries) => {
      const e = entries[0]
      if (e.isIntersecting && !loading && localPage < pages) {
        sp.set('page', String(localPage + 1)); setSp(sp)
      }
    }, { rootMargin: '200px' })
    io.observe(el)
    return () => io.disconnect()
  }, [sentinelRef.current, loading, pages, localPage, aiMode])

  const hasFacetFilters = Boolean(type) || selectedTags.length > 0

  return (
    <>
    <div className="row g-3">
      <div className="col-12 col-lg-3">
        <div className="card p-3" style={{ position: 'sticky', top: 8, maxHeight: 'calc(100vh - 120px)', overflow: 'auto' }}>
          <div className="d-flex justify-content-between align-items-center mb-2">
            <div className="fw-semibold">Фасеты</div>
            <button className="btn btn-sm btn-outline-secondary" onClick={handleResetFacets} disabled={!hasFacetFilters}>
              Сбросить
            </button>
          </div>
          {!facets && <div className="text-secondary">Загрузка…</div>}
          {facets && (
            <div className="d-flex flex-column gap-3">
              {facets.include_types !== false && facets.types.length > 0 && (
                <div>
                  <div className="mb-2">Типы:</div>
                  <div className="d-flex flex-column gap-2">
                    {facets.types.map(([k, c], i) => {
                      const value = String(k || '')
                      const active = type === value
                      return (
                        <button
                          key={i}
                          className={`btn btn-sm ${active ? 'btn-secondary' : 'btn-outline-secondary'} d-flex justify-content-between`}
                          onClick={() => {
                            if (active) {
                              sp.delete('type')
                            } else if (k) {
                              sp.set('type', value)
                            }
                            sp.set('page', '1')
                            setSp(sp)
                          }}
                        >
                          <span>{materialTypeRu(k, 'Другое')}</span>
                          <span className="text-secondary">{c}</span>
                        </button>
                      )
                    })}
                  </div>
                </div>
              )}
              <div>
                <div className="mb-2">Теги:</div>
                {Object.keys(facets.tag_facets).length === 0 && (
                  <div className="text-secondary" style={{ fontSize: '0.9rem' }}>
                    Нет доступных фасетов. Настройте их в разделе администрирования.
                  </div>
                )}
                {Object.entries(facets.tag_facets).map(([key, values]) => (
                  <div key={key} className="mb-3">
                    <div className="text-secondary">{tagKeyRu(key)}</div>
                    <div className="d-flex flex-wrap gap-2 mt-1">
                      {values.slice(0, 12).map(([val, cnt], i) => {
                        const tagValue = `${key}=${val}`
                        const active = selectedTags.includes(tagValue)
                        return (
                          <button
                            key={i}
                            className={`btn btn-sm ${active ? 'btn-secondary' : 'btn-outline-secondary'}`}
                            onClick={() => {
                              if (active) {
                                const next = selectedTags.filter(x => x !== tagValue)
                                setSelectedTags(next)
                                sp.delete('tag'); next.forEach(x => sp.append('tag', x))
                              } else {
                                const next = [...selectedTags, tagValue]
                                setSelectedTags(next)
                                sp.delete('tag'); next.forEach(x => sp.append('tag', x))
                              }
                              sp.set('page', '1')
                              setSp(sp)
                            }}
                          >
                            {val} <span className="text-secondary">({cnt})</span>
                          </button>
                        )
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
      <div className="col-12 col-lg-9">
        <div className="d-flex flex-wrap gap-2 align-items-center mb-2">
          <div className="form-floating" style={{ minWidth: 220 }}>
            <select className="form-select" id="collectionSelect" value={collectionId} onChange={e=>handleCollectionChange(e.target.value)}>
              <option value="">Все доступные</option>
              {collections.map(col => (
                <option key={col.id} value={col.id}>{col.name}</option>
              ))}
            </select>
            <label htmlFor="collectionSelect">Коллекция</label>
          </div>
        </div>
        <div className="mb-2 d-flex align-items-center gap-2">
          <div className="form-check form-switch">
            <input className="form-check-input" type="checkbox" id="ai" checked={aiMode} onChange={e=>setAiMode(e.target.checked)} />
            <label className="form-check-label" htmlFor="ai">Поиск ИИ</label>
          </div>
          {aiMode && (
            <>
              <button className="btn btn-sm btn-outline-secondary" onClick={()=>setShowAiSettings(v=>!v)}>
                {showAiSettings ? 'Скрыть настройки' : 'Настройки ИИ'}
              </button>
              <span className="muted" style={{fontSize: '0.9rem'}}>
                Топ {aiTopK} · кандидатов {aiMaxCandidates} · {aiDeepSearch ? 'глубокий' : 'быстрый'}{aiFullText ? ' · полный текст' : ''}{aiUseLlmSnippets ? ' · сниппеты LLM' : ''}
              </span>
            </>
          )}
        </div>
        {aiMode && showAiSettings && (
          <div className="card p-3 mb-2 bg-light">
            <div className="row g-3">
              <div className="col-12 col-lg-3">
                <label className="form-label">Топ K (1–5)</label>
                <input
                  type="number"
                  className="form-control"
                  min={1}
                  max={5}
                  value={aiTopK}
                  onChange={e => {
                    const parsed = parseInt(e.target.value || '3', 10)
                    const next = Number.isNaN(parsed) ? 3 : parsed
                    setAiTopK(Math.max(1, Math.min(5, next)))
                  }}
                />
              </div>
              <div className="col-12 col-lg-3">
                <label className="form-label">Макс. кандидатов</label>
                <input
                  type="number"
                  className="form-control"
                  min={5}
                  max={400}
                  value={aiMaxCandidates}
                  onChange={e => {
                    const parsed = parseInt(e.target.value || '60', 10)
                    const next = Number.isNaN(parsed) ? 60 : parsed
                    setAiMaxCandidates(Math.max(5, Math.min(400, next)))
                  }}
                />
                <small className="text-muted">Ограничение кандидатов до углублённого анализа.</small>
              </div>
              <div className="col-12 col-lg-3">
                <label className="form-label">Сниппетов на файл</label>
                <input
                  type="number"
                  className="form-control"
                  min={1}
                  max={10}
                  value={aiMaxSnippets}
                  onChange={e => {
                    const parsed = parseInt(e.target.value || '3', 10)
                    const next = Number.isNaN(parsed) ? 3 : parsed
                    setAiMaxSnippets(Math.max(1, Math.min(10, next)))
                  }}
                />
              </div>
              <div className="col-12 col-lg-3">
                <label className="form-label">Размер чанка (симв.)</label>
                <input
                  type="number"
                  className="form-control"
                  min={500}
                  max={20000}
                  value={aiChunkChars}
                  onChange={e => {
                    const parsed = parseInt(e.target.value || '5000', 10)
                    const next = Number.isNaN(parsed) ? 5000 : parsed
                    setAiChunkChars(Math.max(500, Math.min(20000, next)))
                  }}
                />
              </div>
              <div className="col-12 col-lg-3">
                <label className="form-label">Макс. чанков</label>
                <input
                  type="number"
                  className="form-control"
                  min={1}
                  max={200}
                  value={aiMaxChunks}
                  onChange={e => {
                    const parsed = parseInt(e.target.value || '40', 10)
                    const next = Number.isNaN(parsed) ? 40 : parsed
                    setAiMaxChunks(Math.max(1, Math.min(200, next)))
                  }}
                />
              </div>
              <div className="col-12 col-lg-3 d-flex flex-column justify-content-center">
                <div className="form-check form-switch">
                  <input className="form-check-input" type="checkbox" id="aiDeep" checked={aiDeepSearch} onChange={e=>setAiDeepSearch(e.target.checked)} />
                  <label className="form-check-label" htmlFor="aiDeep">Глубокий поиск по документам</label>
                </div>
                <small className="text-muted">Читает топ файлы кусками, включая стенограммы и описания изображений.</small>
              </div>
              <div className="col-12 col-lg-3 d-flex flex-column justify-content-center">
                <div className="form-check form-switch">
                  <input className="form-check-input" type="checkbox" id="aiFullText" checked={aiFullText} onChange={e=>setAiFullText(e.target.checked)} />
                  <label className="form-check-label" htmlFor="aiFullText">Полнотекстовое сканирование</label>
                </div>
                <small className="text-muted">Читает сам файл вместо кэша. Медленнее, но точнее.</small>
              </div>
              <div className="col-12 col-lg-3 d-flex flex-column justify-content-center">
                <div className="form-check form-switch">
                  <input
                    className="form-check-input"
                    type="checkbox"
                    id="aiLlmSnippets"
                    checked={aiUseLlmSnippets}
                    onChange={e => setAiUseLlmSnippets(e.target.checked)}
                  />
                  <label className="form-check-label" htmlFor="aiLlmSnippets">LLM сниппеты</label>
                </div>
                <small className="text-muted">Формирует короткий пересказ для выдачи (медленнее, но информативнее).</small>
              </div>
              <div className="col-12 col-lg-3">
                <div className="form-check">
                  <input className="form-check-input" type="checkbox" id="aiTags" checked={aiUseTags} onChange={e=>setAiUseTags(e.target.checked)} />
                  <label className="form-check-label" htmlFor="aiTags">Учитывать теги</label>
                </div>
                <div className="form-check">
                  <input className="form-check-input" type="checkbox" id="aiText" checked={aiUseText} onChange={e=>setAiUseText(e.target.checked)} />
                  <label className="form-check-label" htmlFor="aiText">Учитывать метаданные</label>
                </div>
                <small className="text-muted">Выключите, если нужна чистая работа только по одному источнику.</small>
              </div>
              <div className="col-12 col-lg-3 d-flex flex-column justify-content-center">
                <div className="form-check form-switch">
                  <input
                    className="form-check-input"
                    type="checkbox"
                    id="aiAutoSpeak"
                    checked={autoSpeakAnswer}
                    disabled={!speechSupported}
                    onChange={e => setAutoSpeakAnswer(e.target.checked)}
                  />
                  <label className="form-check-label" htmlFor="aiAutoSpeak">Автоматически озвучивать ответ</label>
                </div>
                {speechSupported ? (
                  <small className="text-muted">Старт озвучки при появлении нового ответа.</small>
                ) : (
                  <small className="text-muted">Синтез речи недоступен в этом браузере.</small>
                )}
              </div>
            </div>
          </div>
        )}
        <div className="floating-search mb-2 d-flex align-items-center justify-content-between" role="toolbar" aria-label="Выбранные фильтры">
          <div className="d-flex flex-wrap align-items-center" style={{gap:8}}>
            {selectedTags.map((t, i) => {
              const [rawKey, ...rest] = t.split('=')
              const value = rest.join('=')
              const keyLabel = tagKeyRu(rawKey || t)
              const label = value ? `${keyLabel}: ${value}` : keyLabel
              return (
                <span key={i} className="tag" aria-label={`Фильтр ${label}`}>
                  {label} <button className="btn btn-sm btn-outline-secondary ms-1" onClick={()=>removeTag(i)} aria-label="Снять фильтр">×</button>
                </span>
              )
            })}
            <span className="muted ms-2">Найдено: {total}</span>
          </div>
          {pages > 1 && (
            <div className="btn-group">
              <button className="btn btn-sm btn-outline-secondary" disabled={page<=1} onClick={()=>{ sp.set('page', String(page-1)); setSp(sp) }}>«</button>
              {Array.from({length: Math.min(7, pages)}).map((_,i) => {
                const base = Math.max(1, Math.min(pages-6, page-3))
                const num = Math.min(pages, base + i)
                return <button key={i} className={`btn btn-sm ${num===page?'btn-secondary':'btn-outline-secondary'}`} onClick={()=>{ sp.set('page', String(num)); setSp(sp) }}>{num}</button>
              })}
              <button className="btn btn-sm btn-outline-secondary" disabled={page>=pages} onClick={()=>{ sp.set('page', String(page+1)); setSp(sp) }}>»</button>
            </div>
          )}
          <button className="btn btn-sm btn-outline-secondary ms-2" onClick={()=>{ sp.forEach((_,k)=>sp.delete(k)); setSelectedTags([]); setAiAnswer(''); setAiKeywords([]); setAiMode(false); setAiProgress([]); sp.set('page','1'); setSp(sp) }}>Сбросить</button>
        </div>
        {loading && !aiMode && (
          <div className="masonry">
            {Array.from({length: 9}).map((_,i)=> (
              <div key={i} className="masonry-item">
                <div className="skeleton" style={{height: 120}} />
              </div>
            ))}
          </div>
        )}
        {aiMode && (aiProgress.length > 0 || aiLoading) && (
          <div className="card p-3 mb-2" aria-live="polite">
            <div className="fw-semibold mb-1 d-flex justify-content-between align-items-center">
              <span>Прогресс поиска</span>
              {aiQueryHash && <span className="text-muted" style={{fontSize: '0.85rem'}}>Хэш {aiQueryHash.slice(0, 8)}</span>}
            </div>
            <div className="d-grid gap-1">
              {(progressItems.length ? progressItems : [{ id: 'pending', line: 'Поиск выполняется…', icon: '⏳' }]).map((item, idx) => (
                <div key={item.id} className={`d-flex align-items-start gap-2 ${idx === progressItems.length - 1 && aiLoading ? 'fw-semibold' : ''}`}>
                  <span style={{width: 20}}>{item.icon}</span>
                  <span>{item.line}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {!!aiAnswer && (
          <div className="card p-3 mb-2" aria-live="polite">
            <div className="d-flex align-items-center justify-content-between mb-1 gap-2">
              <div className="fw-semibold">Ответ ИИ</div>
              {canSpeakAnswer && (
                <button
                  type="button"
                  className={`btn btn-sm btn-outline-${speechState === 'speaking' ? 'danger' : 'secondary'}`}
                  onClick={speechState === 'speaking' ? stopSpeakingAnswer : speakAnswer}
                  aria-label={speechState === 'speaking' ? 'Остановить озвучивание' : 'Озвучить ответ'}
                  title={speechState === 'speaking' ? 'Остановить озвучивание' : 'Озвучить ответ'}
                >
                  {speechState === 'speaking' ? '⏹ Стоп' : '🔊 Озвучить'}
                </button>
              )}
            </div>
            {speechError && (
              <div className="text-danger" style={{ fontSize: '0.85rem' }}>{speechError}</div>
            )}
            <div style={{whiteSpace:'pre-wrap'}} dangerouslySetInnerHTML={{__html: (function(){
              const raw = String(aiAnswer||'')
              const tmp = document.createElement('div'); tmp.innerHTML = raw
              const walker = document.createTreeWalker(tmp, NodeFilter.SHOW_ELEMENT, null)
              const allowed = new Set(['A','B','I','EM','STRONG','CODE','PRE','P','BR','UL','OL','LI','SPAN','DIV'])
              const toRemove: Element[] = []
              while (walker.nextNode()){
                const el = walker.currentNode as Element
                if (!allowed.has(el.tagName)) { toRemove.push(el); continue }
                for (const attr of Array.from(el.attributes)){
                  const n = attr.name.toLowerCase()
                  if (n.startsWith('on')) el.removeAttribute(attr.name)
                  if (el.tagName==='A' && n==='href' && /^\s*javascript:/i.test(attr.value)) el.removeAttribute(attr.name)
                  if (el.tagName==='A' && n==='rel') el.setAttribute('rel','noopener noreferrer')
                }
              }
              toRemove.forEach(el=> el.replaceWith(document.createTextNode(el.textContent||'')))
              return tmp.innerHTML
            })() }} />
            {!!aiKeywords?.length && (
              <div className="mt-3">
                <div className="text-muted" style={{fontSize:'0.85rem'}}>Ключевые термины</div>
                <div className="d-flex flex-wrap gap-2 mt-1">
                  {aiKeywords.map((k,i)=>{
                    const positiveKey = `relevant:kw-${k}`
                    const negativeKey = `irrelevant:kw-${k}`
                    const posState = feedbackStatus[positiveKey]
                    const negState = feedbackStatus[negativeKey]
                    const statusGlyph = posState === 'done' ? '✓' : negState === 'done' ? '✗' : (posState === 'loading' || negState === 'loading') ? '…' : ''
                    return (
                      <div key={i} className="d-flex align-items-center gap-1 border rounded px-2 py-1" style={{background:'var(--surface-variant, rgba(0,0,0,0.03))'}}>
                        <span className="tag mb-0" style={{margin:0}}>{k}</span>
                        <button
                          className="btn btn-sm btn-outline-success"
                          disabled={!aiQueryHash || posState === 'loading'}
                          title="Помогает"
                          onClick={()=>handleKeywordFeedback(k, 'relevant')}
                        >👍</button>
                        <button
                          className="btn btn-sm btn-outline-danger"
                          disabled={!aiQueryHash || negState === 'loading'}
                          title="Не подходит"
                          onClick={()=>handleKeywordFeedback(k, 'irrelevant')}
                        >👎</button>
                        {statusGlyph && <span className="text-muted" style={{fontSize:'0.75rem'}}>{statusGlyph}</span>}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
            {!!aiFilteredKeywords?.length && (
              <div className="mt-2">
                <div className="text-muted" style={{fontSize:'0.85rem'}}>Исключённые термины</div>
                <div className="d-flex flex-wrap gap-2 mt-1">
                  {aiFilteredKeywords.map((k,i)=> {
                    const busyKey = `relevant:kw-${k}`
                    const status = feedbackStatus[busyKey]
                    return (
                      <div key={i} className="d-inline-flex align-items-center gap-1 border rounded px-2 py-1" style={{fontSize:12}}>
                        <span style={{textDecoration:'line-through'}}>{k}</span>
                        <button
                          className="btn btn-sm btn-outline-success"
                          style={{fontSize:11, padding:'0 6px'}}
                          disabled={status === 'loading'}
                          onClick={()=>handleKeywordRestore(k)}
                        >
                          Вернуть
                        </button>
                        {status === 'error' && <span className="text-danger">!</span>}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
            {aiSources && aiSources.length>0 && (
              <div className="mt-2">
                <div className="fw-semibold mb-1">Источники</div>
                <ol>
                  {aiSources.map((s:any, i:number) => {
                    const rel = s.rel_path || ''
                    const snips: string[] = Array.isArray(s.snippets) ? s.snippets.slice(0, 3) : []
                    const baseTerms = (aiKeywords||[])
                    // попытаться выбрать термы, которые встречаются в первом сниппете
                    const primary = snips[0] ? baseTerms.filter(k => String(snips[0]).toLowerCase().includes(String(k).toLowerCase())) : []
                    const terms = (primary.length? primary : baseTerms).slice(0,5)
                  const mark = encodeURIComponent(terms.join('|'))
                  const href = rel? `/preview/${encodeURIComponent(rel)}?embedded=1&mark=${mark}` : '#'
                  const title = s.title || rel || `file-${s.file_id}`
                  const esc=(x:string)=>x.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]||c))
                  const hi = (txt:string)=>{
                    if(!txt) return ''
                    const arr = terms.length? terms : baseTerms.slice(0,5)
                    if(!arr.length) return esc(txt)
                    const re=new RegExp('('+arr.map(t=>t.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|')+')','gi')
                    return esc(txt).replace(re,'<mark>$1</mark>')
                  }
                  const irrelevantKey = `irrelevant:file-${s.file_id}`
                  const relevantKey = `relevant:file-${s.file_id}`
                  const isIrrelevant = feedbackStatus[irrelevantKey] === 'done'
                  const isRelevant = feedbackStatus[relevantKey] === 'done'
                  return (
                      <li key={i} className={isIrrelevant ? 'text-muted text-decoration-line-through' : ''}>
                        <a href={href} target="_blank" rel="noopener" className={isIrrelevant ? 'text-muted text-decoration-line-through' : ''}>[{i+1}] {title}</a>
                        {snips.length>0 && (
                          <div className="muted" style={{fontSize:13}}>
                            {snips.map((sn,si)=>(
                              <div key={si} dangerouslySetInnerHTML={{__html: ' — '+hi(String(sn))}} />
                            ))}
                          </div>
                        )}
                        <div className="mt-1 d-flex align-items-center gap-2">
                          <button className="btn btn-sm btn-outline-success" disabled={!aiQueryHash || feedbackStatus[`relevant:file-${s.file_id}`]==='loading'} onClick={()=>handleSourceFeedback(Number(s.file_id), 'relevant')}>Подходит</button>
                          <button className="btn btn-sm btn-outline-danger" disabled={!aiQueryHash || feedbackStatus[`irrelevant:file-${s.file_id}`]==='loading'} onClick={()=>handleSourceFeedback(Number(s.file_id), 'irrelevant')}>Не подходит</button>
                          {(feedbackStatus[`relevant:file-${s.file_id}`]==='done' || feedbackStatus[`irrelevant:file-${s.file_id}`]==='done') && (
                            <span className="text-success" style={{fontSize:'0.8rem'}}>Спасибо!</span>
                          )}
                          {feedbackStatus[`relevant:file-${s.file_id}`]==='error' || feedbackStatus[`irrelevant:file-${s.file_id}`]==='error' ? (
                            <span className="text-danger" style={{fontSize:'0.8rem'}}>Ошибка</span>
                          ) : null}
                          {isIrrelevant && <span className="badge bg-danger-subtle text-danger" style={{fontSize:'0.7rem'}}>Помечено как нерелевантное</span>}
                          {isRelevant && <span className="badge bg-success-subtle text-success" style={{fontSize:'0.7rem'}}>Помечено как релевантное</span>}
                        </div>
                        {s.llm_snippet && (
                          <div className="mt-1" style={{fontSize:13}}>
                            <strong>Ответ LLM:</strong> {s.llm_snippet}
                          </div>
                        )}
                      </li>
                    )
                  })}
                </ol>
              </div>
            )}
          </div>
        )}
        <div className="masonry">
          {items.map((f) => (
            <div key={f.id} className="masonry-item">
              <div className={`card p-2 card-type-${(f.material_type||'document').toLowerCase()}`} style={{display:'flex', flexDirection:'column'}}>
                <div className="d-flex align-items-start justify-content-between">
                  <div className="fw-semibold" style={{whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis'}}>{f.title || (f.rel_path?.split('/').pop() || '—')}</div>
                  {f.material_type && <span className="badge bg-secondary ms-2">{materialTypeRu(f.material_type)}</span>}
                </div>
                <div className="text-secondary" style={{whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis'}}>{[f.author, f.year].filter(Boolean).join(', ') || '—'}</div>
                {((f.keywords||'').trim()) && (
                  <div className="mt-1" style={{maxHeight:48, overflow:'hidden'}}>
                    {(f.keywords||'').split(/[,;]+/).slice(0,10).map((k,i)=>(<span key={i} className="tag">{k.trim()}</span>))}
                  </div>
                )}
                {/* mini previews */}
                {f.rel_path && f.material_type === 'image' && (
                  <div className="mt-2">
                    <img src={`/media/${encodeURIComponent(f.rel_path)}`} alt="prev" style={{maxHeight:140, width:'100%', objectFit:'contain', borderRadius:8, border:'1px solid var(--border)'}} />
                  </div>
                )}
                {f.material_type === 'audio' && f.rel_path && (
                  <div className="media-frame mt-2">
                    <audio src={`/media/${encodeURIComponent(f.rel_path)}`} controls preload="metadata" style={{width:'100%'}} />
                  </div>
                )}
                {(!f.material_type || f.material_type==='document' || f.material_type==='article' || f.material_type==='textbook') && (f.text_excerpt) && (
                  <div className="mt-2 muted" style={{maxHeight:64, overflow:'hidden'}} dangerouslySetInnerHTML={{__html: (()=>{
                    const esc=(s:string)=>s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]||c));
                    const txt = esc(f.text_excerpt||'');
                    const terms = (dq||'').trim().split(/[\s,]+/).filter(t=>t.length>1).slice(0,6);
                    if(!terms.length) return txt;
                    const re = new RegExp('('+terms.map(t=>t.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|')+')','gi');
                    return txt.replace(re,'<mark>$1</mark>');
                  })()}} />
                )}
                <div className="mt-2" style={{maxHeight: 68, overflow:'auto'}}>
                  {(f.tags||[]).map((t,i)=> (
                    <span key={i} className="tag" title={`${t.key}=${t.value}`}>{tagKeyRu(t.key)}:{t.value}</span>
                  ))}
                  <form className="d-inline" onSubmit={async(e:any)=>{ e.preventDefault(); const input = e.currentTarget.querySelector('input'); const v=(input?.value||'').trim(); if(!v) return; const idx=v.indexOf('='); if(idx===-1) return; const key=v.slice(0,idx).trim(); const value=v.slice(idx+1).trim(); const next=[...(f.tags||[]), {key, value}]; await updateTagsInline(f, next); if(input) input.value=''; }}>
                    <input placeholder="ключ=значение" style={{background:'var(--bg)', color:'var(--text)', border:'1px dashed var(--border)', borderRadius:8, padding:'2px 6px', marginLeft:6, width:120}} />
                  </form>
                </div>
                {f.rel_path && (
                  <div className="mt-2 d-flex gap-2 mt-auto">
                    <button className="btn btn-sm btn-outline-secondary" onClick={() => setPreviewRel(f.rel_path!)} title="Предпросмотр">👁️</button>
                    <button className="btn btn-sm btn-outline-secondary" onClick={async(e)=>{ const el=e.currentTarget as HTMLButtonElement; el.disabled=true; const prev=el.textContent||''; el.textContent='…'; const url=`/api/files/${f.id}/refresh?use_llm=1&kws_audio=1&summarize=1`; logAction(`REFRESH ${url}`); try{ const r=await fetch(url, { method:'POST' }); const text = await r.text(); if(r.ok){ let j: any = null; try{ j = JSON.parse(text) }catch{} if(j&&j.file){ setItems(list=>list.map(x=>x.id===j.file.id? j.file : x)) } if (j && Array.isArray(j.log)) { j.log.forEach((L:any)=> logAction(String(L), 'info')) } toasts.push('Теги обновлены','success'); logAction(`OK ${f.id} tags updated`, 'success') } else { toasts.push('Ошибка обновления тегов','error'); logAction(`ERROR ${r.status} ${text.slice(0,500)}`,'error') } } catch(e:any){ toasts.push('Ошибка сети при обновлении','error'); logAction(`NETWORK ${String(e)}`,'error') } finally{ el.disabled=false; el.textContent=prev } }} title="Обновить теги">⟳</button>
                    <button className="btn btn-sm btn-outline-secondary" onClick={()=>openEdit(f)} title="Редактировать">📝</button>
                    <button className="btn btn-sm btn-outline-secondary" onClick={async()=>{
                      logAction(`RENAME-SUGGEST /api/files/${f.id}/rename-suggest`)
                      const s = await (await fetch(`/api/files/${f.id}/rename-suggest`)).json().catch(()=>null)
                      const new_name = s && (s.new_name || s.suggested || s.name)
                      if (!new_name) { alert('Не удалось получить предложение'); return }
                      const edited = prompt('Новое имя файла', String(new_name))
                      if (edited===null) return
                      const finalName = edited.trim()
                      if (!finalName) return
                      logAction(`RENAME /api/files/${f.id}/rename -> ${finalName}`)
                      try{
                        const r = await fetch(`/api/files/${f.id}/rename`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ base: finalName }) })
                        const text = await r.text()
                        if (r.ok){
                          let j: any = null; try{ j = JSON.parse(text) }catch{}
                          if (j && j.file){
                            setItems(list => list.map(x => x.id===j.file.id? j.file : x))
                            // если открыт предпросмотр старого пути, обновить
                            if (previewRel === f.rel_path && j.file.rel_path) setPreviewRel(j.file.rel_path)
                          } else {
                            // подстраховка — получить свежие данные
                            const fresh = await fetch(`/api/files/${f.id}`).then(r=>r.json()).catch(()=>null)
                            if (fresh){ setItems(list => list.map(x => x.id===fresh.id? fresh : x)); if (previewRel === f.rel_path && fresh.rel_path) setPreviewRel(fresh.rel_path) }
                          }
                          toasts.push('Файл переименован','success'); logAction('OK rename','success')
                        } else {
                          toasts.push('Ошибка переименования','error'); logAction(`ERROR ${r.status} ${text.slice(0,500)}`,'error')
                        }
                      }catch(e:any){ toasts.push('Сбой сети при переименовании','error'); logAction(`NETWORK ${String(e)}`,'error') }
                    }} title="Автопереименование">✎</button>
                    <a className="btn btn-sm btn-outline-secondary" href={`/download/${encodeURIComponent(f.rel_path)}`}>Скачать</a>
                    <button
                      className="btn btn-sm btn-outline-danger ms-auto"
                      onClick={async () => {
                        if (!confirm('Удалить запись?')) return
                        try {
                          const resp = await fetch(`/api/files/${f.id}`, { method: 'DELETE' })
                          if (resp.status === 204) {
                            setItems(list => list.filter(x => x.id !== f.id))
                            setTotal(count => Math.max(0, count - 1))
                            setAiSources(src => (Array.isArray(src) ? src.filter((s: any) => s?.file_id !== f.id) : src))
                            setFeedbackStatus(prev => {
                              const next = { ...prev }
                              Object.keys(next).forEach(key => {
                                if (key.startsWith(`relevant:file-${f.id}`) || key.startsWith(`irrelevant:file-${f.id}`)) {
                                  delete next[key]
                                }
                              })
                              return next
                            })
                            if (previewRel === f.rel_path) setPreviewRel(null)
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
                      }}
                      title="Удалить"
                    >
                      ✖
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
        {/* infinite scroll sentinel */}
        {!aiMode && (localPage < pages) && (
          <div ref={sentinelRef as any} className="mt-3 text-center muted">Подгружаю ещё…</div>
        )}
      </div>
    </div>
    {previewRel && (
      <div role="dialog" aria-modal="true" aria-label="Предпросмотр" onClick={(e)=>{ if(e.target===e.currentTarget) setPreviewRel(null) }}
           style={{position:'fixed', inset:0, background:'rgba(0,0,0,.6)', display:'flex', alignItems:'center', justifyContent:'center', zIndex:1050}}>
        <button aria-label="Закрыть" onClick={()=>setPreviewRel(null)}
                style={{position:'absolute', top:16, right:16, border:'1px solid var(--border)', background:'var(--surface)', color:'var(--text)', borderRadius:8, padding:'6px 10px'}}>×</button>
        <div className="card" role="document" style={{width:'90vw', height:'85vh', background:'var(--surface)', borderColor:'var(--border)'}}>
          <div className="card-header d-flex justify-content-between align-items-center">
            <div className="fw-semibold">Предпросмотр</div>
            <button className="btn btn-sm btn-outline-secondary" onClick={()=>setPreviewRel(null)}>Закрыть</button>
          </div>
          <div className="card-body p-0" style={{height:'calc(100% - 56px)'}}>
            <iframe title="preview" src={`/preview/${encodeURIComponent(previewRel)}?embedded=1`} style={{border:0, width:'100%', height:'100%'}} />
          </div>
        </div>
      </div>
    )}
    {editItem && (
      <div role="dialog" aria-modal="true" aria-label="Редактирование" onClick={(e)=>{ if(e.target===e.currentTarget) setEditItem(null) }}
           style={{position:'fixed', inset:0, background:'rgba(0,0,0,.6)', display:'flex', alignItems:'center', justifyContent:'center', zIndex:1060}}>
        <div className="card" role="document" style={{width:'720px', maxWidth:'96vw', background:'var(--surface)', borderColor:'var(--border)'}}>
          <div className="card-header d-flex justify-content-between align-items-center">
            <div className="fw-semibold">Редактирование</div>
            <button className="btn btn-sm btn-outline-secondary" onClick={()=>setEditItem(null)}>Закрыть</button>
          </div>
          <div className="card-body">
            {editForm && (
              <div className="d-grid gap-2">
                <input className="form-control" placeholder="Название" value={editForm.title} onChange={e=>setEditForm({...editForm, title:e.target.value})} />
                <div className="d-flex gap-2">
                  <input className="form-control" placeholder="Автор" value={editForm.author} onChange={e=>setEditForm({...editForm, author:e.target.value})} />
                  <input className="form-control" placeholder="Год" value={editForm.year} onChange={e=>setEditForm({...editForm, year:e.target.value})} />
                </div>
                <div className="d-flex gap-2">
                  <input className="form-control" placeholder="Тип" list="material-type-options" value={editForm.material_type} onChange={e=>setEditForm({...editForm, material_type:e.target.value})} />
                  <input className="form-control" placeholder="Имя файла (без расширения)" value={editForm.filename} onChange={e=>setEditForm({...editForm, filename:e.target.value})} />
                </div>
                <datalist id="material-type-options">
                  {materialTypeOptions.map(opt => (
                    <option key={opt.value} value={opt.label} label={opt.value} />
                  ))}
                </datalist>
                <input className="form-control" placeholder="Ключевые слова" value={editForm.keywords} onChange={e=>setEditForm({...editForm, keywords:e.target.value})} />
                <label className="form-label">Теги</label>
                <div className="mb-2" style={{maxHeight:100, overflow:'auto'}}>
                  {(editForm.tagsText||'').split(/\n|;/).map((s:string, i:number)=>{
                    const v=s.trim(); if(!v) return null; return (
                      <button key={i} className="tag" onClick={(ev)=>{ ev.preventDefault(); const arr=(editForm.tagsText||'').split(/\n|;/).filter((x:string)=>x.trim() && x.trim()!==v); setEditForm({...editForm, tagsText: arr.join('\n')}) }}> {v} × </button>
                    )
                  })}
                </div>
                <div className="d-flex gap-2 mb-2">
                  <input className="form-control" placeholder="ключ" onKeyDown={e=>{ if(e.key==='Enter'){ const key=(e.currentTarget as HTMLInputElement).value.trim(); const val=(e.currentTarget.parentElement?.querySelector('[data-newtag=value]') as HTMLInputElement)?.value.trim(); if(key&&val){ const nt=`${key}=${val}`; const txt=(editForm.tagsText||''); setEditForm({...editForm, tagsText: (txt? txt+'\n':'') + nt}); (e.currentTarget as HTMLInputElement).value=''; if((e.currentTarget.parentElement?.querySelector('[data-newtag=value]') as HTMLInputElement)) (e.currentTarget.parentElement?.querySelector('[data-newtag=value]') as HTMLInputElement).value='' } } }} />
                  <input className="form-control" placeholder="значение" data-newtag="value" onKeyDown={e=>{ if(e.key==='Enter'){ const val=(e.currentTarget as HTMLInputElement).value.trim(); const key=(e.currentTarget.parentElement?.querySelector('input:not([data-newtag])') as HTMLInputElement)?.value.trim(); if(key&&val){ const nt=`${key}=${val}`; const txt=(editForm.tagsText||''); setEditForm({...editForm, tagsText: (txt? txt+'\n':'') + nt}); (e.currentTarget as HTMLInputElement).value=''; if((e.currentTarget.parentElement?.querySelector('input:not([data-newtag])') as HTMLInputElement)) (e.currentTarget.parentElement?.querySelector('input:not([data-newtag])') as HTMLInputElement).value='' } } }} />
                </div>
                <textarea className="form-control" rows={4} placeholder="ключ=значение, по строке" value={editForm.tagsText} onChange={e=>setEditForm({...editForm, tagsText:e.target.value})} />
                <div className="d-flex gap-2">
                  <button className="btn btn-primary" onClick={saveEdit}>Сохранить</button>
                  <button className="btn btn-outline-secondary" onClick={()=>setEditItem(null)}>Отмена</button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    )}
    </>
  );
}
