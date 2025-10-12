import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../ui/Auth'
import { useToasts } from '../ui/Toasts'
import type { MaterialTypeDefinition } from '../utils/materialTypesStore'
import { loadMaterialTypes } from '../utils/materialTypesStore'

type Collection = {
  id: number
  name: string
  slug?: string
  searchable: boolean
  graphable: boolean
  is_private?: boolean
  count?: number
}
type LlmEndpointInfo = {
  id: number
  name: string
  base_url: string
  model: string
  weight: number
  purpose?: string | null
  purposes?: string[]
  provider?: string
}
type LlmPurposeOption = { id: string; label: string }
type LlmProviderOption = { id: string; label: string }
type AiwordAccessUser = { user_id: number; username?: string | null; full_name?: string | null }
type UserSuggestion = { id: number; username: string; full_name?: string | null }
type Settings = {
  scan_root: string
  extract_text: boolean
  lm_base: string
  lm_model: string
  lm_key: string
  lm_provider: string
  transcribe_enabled: boolean
  transcribe_backend: string
  transcribe_model: string
  transcribe_language: string
  summarize_audio: boolean
  audio_keywords_llm: boolean
  vision_images: boolean
  kw_to_tags: boolean
  type_detect_flow: string
  type_llm_override: boolean
  import_subdir: string
  move_on_rename: boolean
  collections_in_dirs?: boolean
  collection_type_subdirs?: boolean
  ocr_langs: string
  pdf_ocr_pages: number
  ocr_first_page_dissertation?: boolean
  prompts?: Record<string, string>
  prompt_defaults?: Record<string, string>
  ai_rerank_llm?: boolean
  collections?: Collection[]
  llm_endpoints?: LlmEndpointInfo[]
  llm_purposes?: LlmPurposeOption[]
  llm_providers?: LlmProviderOption[]
  aiword_users?: AiwordAccessUser[]
  default_use_llm?: boolean
  default_prune?: boolean
  type_dirs?: Record<string, string>
  material_types?: MaterialTypeDefinition[]
}

const fallbackPromptDefaults: Record<string, string> = {
  metadata_system: '',
  summarize_audio_system: '',
  keywords_system: '',
  ai_search_keywords_system: '',
  vision_system: '',
}

const promptLabels: Record<string, string> = {
  metadata_system: 'Промпт metadata_system',
  summarize_audio_system: 'Промпт summarize_audio_system',
  keywords_system: 'Промпт keywords_system',
  ai_search_keywords_system: 'Промпт ai_search_keywords_system',
  vision_system: 'Промпт vision_system',
}

const promptOrder = ['metadata_system', 'summarize_audio_system', 'keywords_system', 'ai_search_keywords_system', 'vision_system']

const boolVal = (value: any, fallback: boolean) => (value === undefined ? fallback : !!value)

const sanitizeStringArray = (value: any): string[] => {
  if (!Array.isArray(value)) return []
  return value.map(item => String(item ?? '').trim()).filter(Boolean)
}

const parseListInput = (value: string): string[] => value.split(/[,;\n]/).map(token => token.trim()).filter(Boolean)

const formatListInput = (value?: string[]): string => (Array.isArray(value) && value.length ? value.join(', ') : '')

const createEmptyMaterialType = (): MaterialTypeDefinition => ({
  key: '',
  label: '',
  description: '',
  llm_hint: '',
  enabled: true,
  priority: 0,
  threshold: 1,
  text_keywords: [],
  filename_keywords: [],
  extensions: [],
  exclude_keywords: [],
  require_extension: false,
  require_filename: false,
  require_text: false,
  extension_weight: 2,
  filename_weight: 1.5,
  text_weight: 1,
  flow: [],
  aliases: [],
  special: {},
})

const normalizeMaterialTypeDefinition = (raw: any): MaterialTypeDefinition => {
  const base = createEmptyMaterialType()
  const key = String(raw?.key ?? '').trim()
  return {
    ...base,
    key,
    label: raw?.label !== undefined && raw?.label !== null ? String(raw.label) : '',
    description: raw?.description !== undefined && raw?.description !== null ? String(raw.description) : '',
    llm_hint: raw?.llm_hint !== undefined && raw?.llm_hint !== null ? String(raw.llm_hint) : '',
    enabled: raw?.enabled === undefined ? base.enabled : !!raw.enabled,
    priority: Number.isFinite(Number(raw?.priority)) ? Number(raw.priority) : base.priority,
    threshold: Number.isFinite(Number(raw?.threshold)) ? Number(raw.threshold) : base.threshold,
    text_keywords: sanitizeStringArray(raw?.text_keywords ?? raw?.textKeywords),
    filename_keywords: sanitizeStringArray(raw?.filename_keywords ?? raw?.filenameKeywords),
    extensions: sanitizeStringArray(raw?.extensions),
    exclude_keywords: sanitizeStringArray(raw?.exclude_keywords ?? raw?.excludeKeywords),
    require_extension: raw?.require_extension === undefined ? base.require_extension : !!raw.require_extension,
    require_filename: raw?.require_filename === undefined ? base.require_filename : !!raw.require_filename,
    require_text: raw?.require_text === undefined ? base.require_text : !!raw.require_text,
    extension_weight: Number.isFinite(Number(raw?.extension_weight)) ? Number(raw.extension_weight) : base.extension_weight,
    filename_weight: Number.isFinite(Number(raw?.filename_weight)) ? Number(raw.filename_weight) : base.filename_weight,
    text_weight: Number.isFinite(Number(raw?.text_weight)) ? Number(raw.text_weight) : base.text_weight,
    flow: sanitizeStringArray(raw?.flow).map(token => token.toLowerCase()),
    aliases: sanitizeStringArray(raw?.aliases),
    special: raw?.special && typeof raw.special === 'object' ? { ...(raw.special as Record<string, any>) } : {},
  }
}

const slugifyTypeKey = (value: string): string => {
  const trimmed = value.trim()
  if (!trimmed) return ''
  const normalized = trimmed.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '')
  if (normalized) return normalized
  return trimmed.replace(/\s+/g, '_').toLowerCase()
}

const prepareMaterialTypeForSave = (entry: MaterialTypeDefinition): Record<string, any> | null => {
  const key = slugifyTypeKey(entry.key || '')
  if (!key) return null
  const toList = (list?: string[]) => (Array.isArray(list) ? list.map(item => item.trim()).filter(Boolean) : [])
  const flow = toList(entry.flow).map(item => item.toLowerCase())
  const prepared: Record<string, any> = {
    key,
    label: (entry.label || '').trim(),
    description: (entry.description || '').trim(),
    llm_hint: (entry.llm_hint || '').trim(),
    enabled: entry.enabled !== false,
    priority: Number.isFinite(entry.priority) ? entry.priority : 0,
    threshold: Number.isFinite(entry.threshold) ? entry.threshold : 1,
    text_keywords: toList(entry.text_keywords),
    filename_keywords: toList(entry.filename_keywords),
    extensions: toList(entry.extensions),
    exclude_keywords: toList(entry.exclude_keywords),
    require_extension: !!entry.require_extension,
    require_filename: !!entry.require_filename,
    require_text: !!entry.require_text,
    extension_weight: Number.isFinite(entry.extension_weight) ? entry.extension_weight : 2,
    filename_weight: Number.isFinite(entry.filename_weight) ? entry.filename_weight : 1.5,
    text_weight: Number.isFinite(entry.text_weight) ? entry.text_weight : 1,
    flow,
    aliases: toList(entry.aliases),
    special: entry.special && typeof entry.special === 'object' ? entry.special : {},
  }
  return prepared
}

const formatSpecial = (value?: Record<string, any>): string => {
  if (!value || !Object.keys(value).length) return ''
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return ''
  }
}

const normalizeSettings = (raw: any): Settings => {
  const promptDefaults = { ...fallbackPromptDefaults, ...(raw?.prompt_defaults || {}) }
  const prompts = { ...promptDefaults, ...(raw?.prompts || {}) }
  const collectionsInDirs = boolVal(raw?.collections_in_dirs, false)
  const materialTypes = Array.isArray(raw?.material_types)
    ? raw.material_types.map((item: any) => normalizeMaterialTypeDefinition(item))
    : []
  const collections = Array.isArray(raw?.collections)
    ? raw.collections.map((col: any) => ({
        id: Number(col?.id ?? 0),
        name: String(col?.name ?? ''),
        slug: col?.slug ? String(col.slug) : undefined,
        searchable: boolVal(col?.searchable, true),
        graphable: boolVal(col?.graphable, true),
        is_private: boolVal(col?.is_private, false),
        count: Number.isFinite(col?.count) ? Number(col.count) : 0,
      }))
    : []

  return {
    scan_root: String(raw?.scan_root || ''),
    extract_text: boolVal(raw?.extract_text, true),
    lm_base: String(raw?.lm_base || ''),
    lm_model: String(raw?.lm_model || ''),
    lm_key: String(raw?.lm_key || ''),
    lm_provider: String(raw?.lm_provider || 'openai'),
    transcribe_enabled: boolVal(raw?.transcribe_enabled, true),
    transcribe_backend: String(raw?.transcribe_backend || 'faster-whisper'),
    transcribe_model: String(raw?.transcribe_model || ''),
    transcribe_language: String(raw?.transcribe_language || 'ru'),
    summarize_audio: boolVal(raw?.summarize_audio, true),
    audio_keywords_llm: boolVal(raw?.audio_keywords_llm, true),
    vision_images: boolVal(raw?.vision_images, false),
    kw_to_tags: boolVal(raw?.kw_to_tags, true),
    type_detect_flow: String(raw?.type_detect_flow || 'extension,filename,heuristics,llm'),
    type_llm_override: boolVal(raw?.type_llm_override, true),
    import_subdir: String(raw?.import_subdir || 'import'),
    move_on_rename: boolVal(raw?.move_on_rename, true),
    collections_in_dirs: collectionsInDirs,
    collection_type_subdirs: collectionsInDirs ? boolVal(raw?.collection_type_subdirs, false) : false,
    ocr_langs: String(raw?.ocr_langs || 'rus+eng'),
    pdf_ocr_pages: Number.isFinite(raw?.pdf_ocr_pages) ? Number(raw.pdf_ocr_pages) : 5,
    ocr_first_page_dissertation: boolVal(raw?.ocr_first_page_dissertation, true),
    prompts,
    prompt_defaults: promptDefaults,
    ai_rerank_llm: boolVal(raw?.ai_rerank_llm, false),
    collections,
    llm_endpoints: Array.isArray(raw?.llm_endpoints)
      ? raw.llm_endpoints.map((ep: any) => ({
          ...ep,
          provider: ep?.provider ? String(ep.provider) : 'openai',
        }))
      : [],
    llm_purposes: Array.isArray(raw?.llm_purposes) ? raw.llm_purposes : [],
    llm_providers: Array.isArray(raw?.llm_providers) ? raw.llm_providers : [],
    aiword_users: Array.isArray(raw?.aiword_users) ? raw.aiword_users : [],
    default_use_llm: boolVal(raw?.default_use_llm, true),
    default_prune: boolVal(raw?.default_prune, true),
    type_dirs: raw?.type_dirs || {},
    material_types: materialTypes,
  }
}

export default function SettingsPage() {
  const { user } = useAuth()
  const toasts = useToasts()
  const isAdmin = user?.role === 'admin'
  const [s, setS] = useState<Settings | null>(null)
  const [saving, setSaving] = useState(false)
  const [reindexUseLLM, setReindexUseLLM] = useState(true)
  const [reindexPrune, setReindexPrune] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [aiwordQuery, setAiwordQuery] = useState('')
  const [aiwordOptions, setAiwordOptions] = useState<UserSuggestion[]>([])
  const [aiwordLoading, setAiwordLoading] = useState(false)
  const [llmWeights, setLlmWeights] = useState<Record<number, string>>({})
  const [deleteCollectionId, setDeleteCollectionId] = useState<number | null>(null)

  useEffect(() => {
    if (!isAdmin) return
    let cancelled = false
    ;(async () => {
      try {
        setError(null)
        const r = await fetch('/api/settings')
        if (cancelled) return
        if (r.status === 403) {
          setError('Недостаточно прав для просмотра настроек')
          setS(null)
          return
        }
        const data = await r.json().catch(() => ({}))
        const normalized = normalizeSettings(data)
        setS(normalized)
        setReindexUseLLM(boolVal(normalized.default_use_llm, true))
        setReindexPrune(boolVal(normalized.default_prune, true))
      } catch {
        if (!cancelled) {
          setError('Не удалось загрузить настройки')
        }
      }
    })()
    return () => { cancelled = true }
  }, [isAdmin])

  useEffect(() => {
    if (!s?.llm_endpoints) {
      setLlmWeights({})
      return
    }
    const map: Record<number, string> = {}
    s.llm_endpoints.forEach(ep => { map[ep.id] = String(ep.weight ?? 1) })
    setLlmWeights(map)
  }, [s?.llm_endpoints])

  const assignedAiwordIds = useMemo(() => new Set((s?.aiword_users || []).map(u => u.user_id)), [s?.aiword_users])
  const duplicateMaterialKeys = useMemo(() => {
    const duplicates = new Set<string>()
    if (!s?.material_types) return duplicates
    const counts = new Map<string, number>()
    s.material_types.forEach(mt => {
      const key = (mt.key || '').trim().toLowerCase()
      if (!key) return
      counts.set(key, (counts.get(key) || 0) + 1)
    })
    counts.forEach((count, key) => {
      if (count > 1) duplicates.add(key)
    })
    return duplicates
  }, [s?.material_types])
  const llmPurposes = useMemo<LlmPurposeOption[]>(() => {
    if (Array.isArray(s?.llm_purposes)) {
      const list = [...s.llm_purposes]
      if (!list.some(p => p.id === 'default')) {
        list.unshift({ id: 'default', label: 'По умолчанию' })
      }
      return list
    }
    return [{ id: 'default', label: 'По умолчанию' }]
  }, [s?.llm_purposes])
  const llmPurposeLabels = useMemo(() => {
    const map = new Map<string, string>()
    llmPurposes.forEach(p => map.set(p.id, p.label))
    return map
  }, [llmPurposes])
  const llmProviderOptions = useMemo<LlmProviderOption[]>(() => {
    const defaults: LlmProviderOption[] = [
      { id: 'openai', label: 'OpenAI-совместимый (LM Studio, OpenAI, Azure)' },
      { id: 'ollama', label: 'Ollama' },
    ]
    const map = new Map(defaults.map(opt => [opt.id, opt] as const))
    if (Array.isArray(s?.llm_providers)) {
      s.llm_providers.forEach(opt => {
        if (opt?.id) {
          map.set(opt.id, { id: opt.id, label: opt.label || opt.id })
        }
      })
    }
    if (s?.lm_provider && !map.has(s.lm_provider)) {
      map.set(s.lm_provider, { id: s.lm_provider, label: s.lm_provider })
    }
    return Array.from(map.values())
  }, [s?.llm_providers, s?.lm_provider])
  const llmProviderLabels = useMemo(() => {
    const map = new Map<string, string>()
    llmProviderOptions.forEach(opt => map.set(opt.id, opt.label))
    return map
  }, [llmProviderOptions])
  const promptDefaults = useMemo(() => ({ ...fallbackPromptDefaults, ...(s?.prompt_defaults || {}) }), [s?.prompt_defaults])
  const promptKeys = useMemo(() => {
    const known = new Set(promptOrder)
    const extras = [...new Set([...Object.keys(promptDefaults), ...Object.keys(s?.prompts || {})])]
      .filter(key => !known.has(key))
      .sort()
    return [...promptOrder, ...extras]
  }, [promptDefaults, s?.prompts])
  const updatePrompt = useCallback((key: string, value: string) => {
    setS(prev => {
      if (!prev) return prev
      return { ...prev, prompts: { ...(prev.prompts || {}), [key]: value } }
    })
  }, [])
  const resetPrompt = useCallback((key: string) => {
    const defaultValue = promptDefaults[key] ?? ''
    setS(prev => {
      if (!prev) return prev
      const nextPrompts = { ...(prev.prompts || {}) }
      nextPrompts[key] = defaultValue
      return { ...prev, prompts: nextPrompts }
    })
  }, [promptDefaults])
  const transcribeBackendOptions = useMemo(() => {
    const options = ['faster-whisper']
    const current = (s?.transcribe_backend || '').trim()
    if (current && !options.includes(current)) {
      options.push(current)
    }
    return options
  }, [s?.transcribe_backend])
  const transcribeModelOptions = useMemo(() => {
    const defaults = ['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3', 'distil-small.en', 'distil-medium.en', 'distil-large-v2']
    const set = new Set(defaults)
    const current = (s?.transcribe_model || '').trim()
    if (current) {
      set.add(current)
    }
    return Array.from(set)
  }, [s?.transcribe_model])

  useEffect(() => {
    if (!isAdmin) return
    if (!aiwordQuery.trim()) {
      setAiwordOptions([])
      setAiwordLoading(false)
      return
    }
    const controller = new AbortController()
    const handle = window.setTimeout(async () => {
      setAiwordLoading(true)
      try {
        const r = await fetch(`/api/admin/users/search?q=${encodeURIComponent(aiwordQuery)}&limit=10`, { signal: controller.signal })
        const data = await r.json().catch(() => ({}))
        if (!controller.signal.aborted && r.ok && data?.ok && Array.isArray(data.users)) {
          const suggestions: UserSuggestion[] = data.users.filter((u: any) => !assignedAiwordIds.has(u.id)).map((u: any) => ({
            id: u.id,
            username: u.username,
            full_name: u.full_name,
          }))
          setAiwordOptions(suggestions)
        } else if (!controller.signal.aborted) {
          setAiwordOptions([])
        }
      } catch (err) {
        if (!(err instanceof DOMException && err.name === 'AbortError')) {
          if (!controller.signal.aborted) {
            setAiwordOptions([])
          }
        }
      } finally {
        if (!controller.signal.aborted) {
          setAiwordLoading(false)
        }
      }
    }, 250)
    return () => {
      controller.abort()
      window.clearTimeout(handle)
    }
  }, [aiwordQuery, assignedAiwordIds, isAdmin])

  const patchLlmEndpoint = useCallback(async (id: number, payload: Record<string, unknown>): Promise<LlmEndpointInfo | null> => {
    try {
      const r = await fetch(`/api/admin/llm-endpoints/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok && data.item) {
        toasts.push('LLM обновлена', 'success')
        return data.item as LlmEndpointInfo
      }
      toasts.push(data?.error || 'Не удалось обновить LLM', 'error')
    } catch {
      toasts.push('Ошибка соединения при обновлении LLM', 'error')
    }
    return null
  }, [toasts])

  const commitLlmWeight = useCallback(async (endpointId: number) => {
    if (!s) return
    const input = llmWeights[endpointId]
    const weight = parseFloat(input)
    const current = (s.llm_endpoints || []).find(ep => ep.id === endpointId)?.weight ?? 1
    if (!Number.isFinite(weight) || weight <= 0) {
      toasts.push('Вес должен быть положительным числом', 'error')
      setLlmWeights(prev => ({ ...prev, [endpointId]: String(current) }))
      return
    }
    const updated = await patchLlmEndpoint(endpointId, { weight })
    if (updated) {
      setS(prev => {
        if (!prev) return prev
        const nextEndpoints = (prev.llm_endpoints || []).map(ep => ep.id === endpointId ? { ...ep, ...updated } : ep)
        return { ...prev, llm_endpoints: nextEndpoints }
      })
      setLlmWeights(prev => ({ ...prev, [endpointId]: String(updated.weight ?? weight) }))
    } else {
      setLlmWeights(prev => ({ ...prev, [endpointId]: String(current) }))
    }
  }, [llmWeights, patchLlmEndpoint, s, toasts])

  const commitLlmPurposes = useCallback(async (endpointId: number, values: string[]) => {
    if (!s) return
    const selected = values.length ? values : ['default']
    const before = (s.llm_endpoints || []).find(ep => ep.id === endpointId)?.purposes || ['default']
    setS(prev => {
      if (!prev) return prev
      const nextEndpoints = (prev.llm_endpoints || []).map(ep => ep.id === endpointId ? { ...ep, purposes: selected, purpose: selected.join(',') } : ep)
      return { ...prev, llm_endpoints: nextEndpoints }
    })
    const updated = await patchLlmEndpoint(endpointId, { purposes: selected })
    if (updated) {
      setS(prev => {
        if (!prev) return prev
        const nextEndpoints = (prev.llm_endpoints || []).map(ep => ep.id === endpointId ? { ...ep, ...updated } : ep)
        return { ...prev, llm_endpoints: nextEndpoints }
      })
    } else {
      setS(prev => {
        if (!prev) return prev
        const nextEndpoints = (prev.llm_endpoints || []).map(ep => ep.id === endpointId ? { ...ep, purposes: before, purpose: before.join(',') } : ep)
        return { ...prev, llm_endpoints: nextEndpoints }
      })
    }
  }, [patchLlmEndpoint, s])

  const addAiwordUser = useCallback((entry: UserSuggestion) => {
    if (!s || assignedAiwordIds.has(entry.id)) return
    const nextEntry: AiwordAccessUser = { user_id: entry.id, username: entry.username, full_name: entry.full_name }
    setS(prev => prev ? { ...prev, aiword_users: [ ...(prev.aiword_users || []), nextEntry ] } : prev)
    setAiwordQuery('')
    setAiwordOptions([])
  }, [assignedAiwordIds, s])

  const removeAiwordUser = useCallback((userId: number) => {
    setS(prev => {
      if (!prev) return prev
      return { ...prev, aiword_users: (prev.aiword_users || []).filter(u => u.user_id !== userId) }
    })
  }, [])

  const addMaterialType = useCallback(() => {
    setS(prev => {
      if (!prev) return prev
      const next = [...(prev.material_types || []), createEmptyMaterialType()]
      return { ...prev, material_types: next }
    })
  }, [])

  const updateMaterialType = useCallback((index: number, value: MaterialTypeDefinition) => {
    setS(prev => {
      if (!prev) return prev
      const list = [...(prev.material_types || [])]
      list[index] = value
      return { ...prev, material_types: list }
    })
  }, [])

  const removeMaterialType = useCallback((index: number) => {
    setS(prev => {
      if (!prev) return prev
      const list = [...(prev.material_types || [])]
      list.splice(index, 1)
      return { ...prev, material_types: list }
    })
  }, [])

  const save = async () => {
    if (!s) return
    setSaving(true)
    try {
      const { llm_endpoints, llm_purposes, llm_providers, aiword_users, prompt_defaults: _promptDefaults, ...rest } = s
      void _promptDefaults
      void llm_endpoints
      void llm_purposes
      void llm_providers
      const payload: any = { ...rest, aiword_users: (aiword_users || []).map(u => u.user_id) }
      payload.prompts = { ...promptDefaults, ...(payload.prompts || {}) }
      payload.collections_in_dirs = !!payload.collections_in_dirs
      payload.collection_type_subdirs = !!payload.collection_type_subdirs
      payload.default_use_llm = reindexUseLLM
      payload.default_prune = reindexPrune
      payload.collections = (payload.collections || []).map((col: Collection) => ({
        id: col.id,
        searchable: !!col.searchable,
        graphable: !!col.graphable,
      }))
      const preparedMaterialTypes = (payload.material_types || [])
        .map((item: MaterialTypeDefinition) => prepareMaterialTypeForSave(item))
        .filter((item): item is Record<string, any> => Boolean(item))
      payload.material_types = preparedMaterialTypes
      const r = await fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      if (r.ok) {
        setS(prev => prev ? { ...prev, material_types: preparedMaterialTypes.map(normalizeMaterialTypeDefinition) } : prev)
        toasts.push('Настройки сохранены', 'success')
        loadMaterialTypes(true).catch(() => {})
      } else {
        toasts.push(r.status === 403 ? 'Недостаточно прав' : 'Ошибка сохранения настроек', 'error')
      }
    } catch {
      toasts.push('Ошибка сохранения настроек', 'error')
    } finally {
      setSaving(false)
    }
  }

  const reindex = async () => {
    try {
      const fd = new FormData()
      if (s?.extract_text) fd.set('extract_text', 'on')
      fd.set('use_llm', reindexUseLLM ? 'on' : 'off')
      if (reindexPrune) fd.set('prune', 'on')
      await fetch('/scan/start', { method: 'POST', body: fd })
      try { window.dispatchEvent(new Event('scan-open')) } catch {}
    } catch {
      alert('Не удалось запустить сканирование')
    }
  }

  const deleteCollection = async (collectionId: number) => {
    const target = (s?.collections || []).find(c => c.id === collectionId)
    if (!target) return
    if (!confirm(`Удалить коллекцию «${target.name}» и все её файлы?`)) return
    setDeleteCollectionId(collectionId)
    try {
      const r = await fetch(`/api/collections/${collectionId}`, { method: 'DELETE' })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok) {
        toasts.push('Коллекция удалена', 'success')
        setS(prev => prev ? { ...prev, collections: (prev.collections || []).filter(c => c.id !== collectionId) } : prev)
      } else {
        toasts.push(data?.error || 'Не удалось удалить коллекцию', 'error')
      }
    } catch {
      toasts.push('Ошибка при удалении коллекции', 'error')
    } finally {
      setDeleteCollectionId(null)
    }
  }

  const backupDb = async () => {
    try {
      const r = await fetch('/admin/backup-db', { method: 'POST' })
      if (!r.ok) { alert('Ошибка резервного копирования'); return }
      const blob = await r.blob()
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = 'catalogue_backup.db'
      a.click()
    } catch {
      alert('Ошибка резервного копирования')
    }
  }

  const clearDb = async () => {
    if (!confirm('Удалить ВСЕ записи? Операция необратима.')) return
    try {
      await fetch('/admin/clear-db', { method: 'POST' })
      alert('База очищена. Перезапустите сканирование.')
    } catch {
      alert('Ошибка очистки базы')
    }
  }

  const importDb = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const fd = new FormData(e.currentTarget)
    try {
      const r = await fetch('/admin/import-db', { method: 'POST', body: fd })
      alert(r.ok ? 'Импорт завершён. Перезапустите приложение.' : 'Ошибка импорта')
    } catch {
      alert('Ошибка импорта')
    }
  }

  if (!isAdmin) return <div className="card p-3">Недостаточно прав.</div>
  if (error) return <div className="card p-3">{error}</div>
  if (!s) return <div className="card p-3">Загрузка настроек…</div>

  const llmEndpoints = s.llm_endpoints || []
  const aiwordUsers = s.aiword_users || []
  const materialTypes = s.material_types || []

  return (
    <div className="d-grid gap-3">
      <div className="card p-3">
        <div className="fw-semibold mb-3 d-flex flex-wrap justify-content-between gap-2 align-items-center">
          <span>Настройки</span>
          <div className="d-flex flex-wrap align-items-center gap-2">
            <button className="btn btn-primary" onClick={save} disabled={saving}>{saving ? 'Сохранение…' : 'Сохранить'}</button>
            <div className="form-check form-switch m-0">
              <input
                className="form-check-input"
                type="checkbox"
                id="reidx_llm"
                checked={reindexUseLLM}
                onChange={e => {
                  setReindexUseLLM(e.target.checked)
                  setS(prev => prev ? { ...prev, default_use_llm: e.target.checked } : prev)
                }}
              />
              <label className="form-check-label" htmlFor="reidx_llm">LLM при переиндексации</label>
            </div>
            <div className="form-check form-switch m-0">
              <input
                className="form-check-input"
                type="checkbox"
                id="reidx_prune"
                checked={reindexPrune}
                onChange={e => {
                  setReindexPrune(e.target.checked)
                  setS(prev => prev ? { ...prev, default_prune: e.target.checked } : prev)
                }}
              />
              <label className="form-check-label" htmlFor="reidx_prune">Удалять отсутствующие файлы</label>
            </div>
            <button className="btn btn-outline-secondary" onClick={reindex}>Переиндексировать библиотеку</button>
          </div>
        </div>
      </div>

      <div className="card p-3">
        <div className="fw-semibold mb-2">Индексатор и OCR</div>
        <div className="row g-3">
          <div className="col-md-8">
            <label className="form-label">Корневая папка</label>
            <input className="form-control" placeholder="C:/путь/к/библиотеке" value={s.scan_root} onChange={e => setS({ ...s, scan_root: e.target.value })} />
          </div>
          <div className="col-md-4 d-flex align-items-end">
            <div className="form-check form-switch">
              <input className="form-check-input" type="checkbox" id="ext" checked={s.extract_text} onChange={e => setS({ ...s, extract_text: e.target.checked })} />
              <label className="form-check-label" htmlFor="ext">Извлекать текст</label>
            </div>
          </div>
          <div className="col-md-4">
            <label className="form-label">OCR языки</label>
            <input className="form-control" placeholder="rus+eng" value={s.ocr_langs} onChange={e => setS({ ...s, ocr_langs: e.target.value })} />
            <div className="form-text">Напр.: rus+eng</div>
          </div>
          <div className="col-md-4">
            <label className="form-label">PDF OCR: кол-во страниц</label>
            <input className="form-control" type="number" min={0} max={20} value={s.pdf_ocr_pages} onChange={e => setS({ ...s, pdf_ocr_pages: parseInt(e.target.value || '0', 10) })} />
          </div>
          <div className="col-md-4">
            <div className="d-flex flex-column gap-2">
              <div className="form-check form-switch">
                <input className="form-check-input" type="checkbox" id="ocr1st" checked={!!s.ocr_first_page_dissertation} onChange={e => setS({ ...s, ocr_first_page_dissertation: e.target.checked })} />
                <label className="form-check-label" htmlFor="ocr1st">OCR 1‑й стр. для диссертаций</label>
              </div>
              <div className="form-check form-switch">
                <input className="form-check-input" type="checkbox" id="move" checked={s.move_on_rename} onChange={e => setS({ ...s, move_on_rename: e.target.checked })} />
                <label className="form-check-label" htmlFor="move">Перемещать при переименовании</label>
              </div>
              <div className="form-check form-switch">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="collectionsDirs"
                  checked={!!s.collections_in_dirs}
                  onChange={e => setS(prev => prev ? { ...prev, collections_in_dirs: e.target.checked, collection_type_subdirs: e.target.checked ? prev.collection_type_subdirs : false } : prev)}
                />
                <label className="form-check-label" htmlFor="collectionsDirs">Коллекции в отдельных папках</label>
              </div>
              <div className="form-check form-switch ms-3">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="collectionsType"
                  checked={!!s.collection_type_subdirs}
                  disabled={!s.collections_in_dirs}
                  onChange={e => setS(prev => prev ? { ...prev, collection_type_subdirs: e.target.checked } : prev)}
                />
                <label className="form-check-label" htmlFor="collectionsType">Подпапки по типам внутри коллекций</label>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card p-3">
        <div className="fw-semibold mb-2">LLM и типизация</div>
        <div className="row g-3">
          <div className="col-md-3">
            <label className="form-label" htmlFor="lm-provider">Тип API</label>
            <select
              id="lm-provider"
              className="form-select"
              value={s.lm_provider}
              onChange={e => setS({ ...s, lm_provider: e.target.value })}
            >
              {llmProviderOptions.map(opt => (
                <option key={opt.id} value={opt.id}>{opt.label}</option>
              ))}
            </select>
          </div>
          <div className="col-md-4">
            <label className="form-label">Базовый URL LLM</label>
            <input className="form-control" placeholder="http://localhost:1234/v1" value={s.lm_base} onChange={e => setS({ ...s, lm_base: e.target.value })} />
          </div>
          <div className="col-md-3">
            <label className="form-label">Модель LLM</label>
            <input className="form-control" placeholder="gpt-4o-mini" value={s.lm_model} onChange={e => setS({ ...s, lm_model: e.target.value })} />
          </div>
          <div className="col-md-2">
            <label className="form-label">LM API Key</label>
            <input className="form-control" placeholder="sk-..." value={s.lm_key} onChange={e => setS({ ...s, lm_key: e.target.value })} />
          </div>
          <div className="col-md-3 d-flex align-items-end">
            <div className="form-check form-switch">
              <input className="form-check-input" type="checkbox" id="vision" checked={s.vision_images} onChange={e => setS({ ...s, vision_images: e.target.checked })} />
              <label className="form-check-label" htmlFor="vision">LLM для изображений (vision)</label>
            </div>
          </div>
          <div className="col-md-9">
            <label className="form-label">Порядок определения типа</label>
            <input className="form-control" placeholder="extension,filename,heuristics" value={s.type_detect_flow} onChange={e => setS({ ...s, type_detect_flow: e.target.value })} />
            <div className="form-check form-switch mt-2">
              <input className="form-check-input" type="checkbox" id="tlo" checked={s.type_llm_override} onChange={e => setS({ ...s, type_llm_override: e.target.checked })} />
              <label className="form-check-label" htmlFor="tlo">LLM может переопределять тип</label>
            </div>
          </div>
          <div className="col-12">
            <div className="form-check form-switch">
              <input className="form-check-input" type="checkbox" id="ktt" checked={s.kw_to_tags} onChange={e => setS({ ...s, kw_to_tags: e.target.checked })} />
              <label className="form-check-label" htmlFor="ktt">Ключевые слова → теги</label>
            </div>
          </div>
          <div className="col-md-3 d-flex align-items-center">
            <div className="form-check form-switch">
              <input className="form-check-input" type="checkbox" id="rerank" checked={!!s.ai_rerank_llm} onChange={e => setS({ ...s, ai_rerank_llm: e.target.checked })} />
              <label className="form-check-label" htmlFor="rerank">LLM‑реранжирование поиска</label>
            </div>
          </div>
      </div>
    </div>

      <div className="card p-3">
        <div className="fw-semibold mb-1">Типы документов и эвристики</div>
        <div className="text-muted mb-3" style={{ fontSize: 13 }}>Настройте ключевые слова, расширения и специальные правила для автоматического определения типа документов.</div>
        <MaterialTypesEditor
          materialTypes={materialTypes}
          onChange={updateMaterialType}
          onRemove={removeMaterialType}
          onAdd={addMaterialType}
          duplicateKeys={duplicateMaterialKeys}
        />
      </div>

      <div className="card p-3">
        <div className="fw-semibold mb-2">Транскрибация и аудио</div>
        <div className="row g-3">
          <div className="col-md-3 d-flex align-items-center">
            <div className="form-check form-switch m-0">
              <input className="form-check-input" type="checkbox" id="transcribe_enabled" checked={s.transcribe_enabled} onChange={e => setS({ ...s, transcribe_enabled: e.target.checked })} />
              <label className="form-check-label" htmlFor="transcribe_enabled">Включить транскрибацию</label>
            </div>
          </div>
          <div className="col-md-3">
            <label className="form-label" htmlFor="transcribe_backend">Бэкенд</label>
            <select
              id="transcribe_backend"
              className="form-select"
              value={s.transcribe_backend}
              onChange={e => setS({ ...s, transcribe_backend: e.target.value })}
            >
              {transcribeBackendOptions.map(option => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
            <div className="form-text">Поддерживается faster-whisper.</div>
          </div>
          <div className="col-md-3">
            <label className="form-label" htmlFor="transcribe_model">Модель</label>
            <input
              id="transcribe_model"
              className="form-control"
              placeholder="small"
              list="transcribe-models"
              value={s.transcribe_model}
              onChange={e => setS({ ...s, transcribe_model: e.target.value })}
            />
            <div className="form-text">Можно указать алиас, repo id или путь к модели.</div>
          </div>
          <div className="col-md-3">
            <label className="form-label" htmlFor="transcribe_language">Язык распознавания</label>
            <input
              id="transcribe_language"
              className="form-control"
              placeholder="ru"
              value={s.transcribe_language}
              onChange={e => setS({ ...s, transcribe_language: e.target.value })}
            />
            <div className="form-text">Напр.: ru, en, auto.</div>
          </div>
          <div className="col-md-3 d-flex align-items-center">
            <div className="form-check form-switch m-0">
              <input className="form-check-input" type="checkbox" id="summ_audio" checked={s.summarize_audio} onChange={e => setS({ ...s, summarize_audio: e.target.checked })} />
              <label className="form-check-label" htmlFor="summ_audio">Суммаризировать аудио</label>
            </div>
          </div>
          <div className="col-md-3 d-flex align-items-center">
            <div className="form-check form-switch m-0">
              <input className="form-check-input" type="checkbox" id="audio_kw" checked={s.audio_keywords_llm} onChange={e => setS({ ...s, audio_keywords_llm: e.target.checked })} />
              <label className="form-check-label" htmlFor="audio_kw">Ключевые слова из аудио</label>
            </div>
          </div>
        </div>
        <datalist id="transcribe-models">
          {transcribeModelOptions.map(option => (
            <option key={option} value={option} />
          ))}
        </datalist>
      </div>

      <div className="card p-3">
        <div className="d-flex flex-wrap justify-content-between align-items-start gap-2 mb-3">
          <div>
            <div className="fw-semibold">Назначение LLM эндпоинтов</div>
            <div className="text-muted" style={{ fontSize: 13 }}>Выберите задачи для каждого эндпоинта. Один тип может обслуживаться несколькими LLM — запросы будут распределяться по очереди.</div>
          </div>
          <Link className="btn btn-sm btn-outline-secondary" to="/admin/llm">Управление эндпоинтами</Link>
        </div>
        <div className="d-grid gap-3">
          {llmEndpoints.length === 0 && (
            <div className="text-muted">Нет сохранённых эндпоинтов. Добавьте их на вкладке «LLM».</div>
          )}
          {llmEndpoints.map(ep => {
            const selected = ep.purposes && ep.purposes.length ? ep.purposes : ['default']
            const chips = selected.map(id => llmPurposeLabels.get(id) || id)
            const providerLabel = llmProviderLabels.get(ep.provider || '') || ep.provider || 'openai'
            return (
              <div key={ep.id} className="border rounded-3 p-3" style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}>
                <div className="d-flex flex-wrap justify-content-between align-items-start gap-3">
                  <div>
                    <div className="fw-semibold">{ep.name}</div>
                    <div className="text-muted" style={{ fontSize: 13 }}>{providerLabel} • {ep.model} • {ep.base_url}</div>
                  </div>
                  <div className="d-flex align-items-center gap-2">
                    <label className="form-label m-0" htmlFor={`llm-weight-${ep.id}`} style={{ fontSize: 13 }}>Вес</label>
                    <input
                      id={`llm-weight-${ep.id}`}
                      className="form-control form-control-sm"
                      style={{ width: 90 }}
                      value={llmWeights[ep.id] ?? ''}
                      onChange={e => setLlmWeights(prev => ({ ...prev, [ep.id]: e.target.value }))}
                      onBlur={() => commitLlmWeight(ep.id)}
                      inputMode="decimal"
                    />
                  </div>
                </div>
                <div className="mt-3">
                  <label className="form-label" htmlFor={`llm-purposes-${ep.id}`}>Типы задач</label>
                  <select
                    id={`llm-purposes-${ep.id}`}
                    className="form-select"
                    multiple
                    value={selected}
                    onChange={e => {
                      const values = Array.from(e.target.selectedOptions).map(o => o.value)
                      commitLlmPurposes(ep.id, values)
                    }}
                    size={Math.min(Math.max(llmPurposes.length, 4), 8)}
                  >
                    {llmPurposes.map(option => (
                      <option key={option.id} value={option.id}>{option.label}</option>
                    ))}
                  </select>
                  <div className="form-text">Назначено: {chips.length ? chips.join(', ') : 'По умолчанию'}</div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      <div className="card p-3">
        <div className="fw-semibold mb-1">Доступ к AIWord</div>
        <div className="text-muted mb-3" style={{ fontSize: 13 }}>Администраторы всегда имеют доступ. Назначьте дополнительных пользователей, начав вводить их имя или логин.</div>
        <div className="d-flex flex-wrap align-items-start gap-2">
          <div className="position-relative" style={{ flex: '1 1 280px', maxWidth: 360 }}>
            <input
              className="form-control"
              placeholder="Начните вводить имя пользователя"
              value={aiwordQuery}
              onChange={e => setAiwordQuery(e.target.value)}
            />
            {(aiwordOptions.length > 0 || aiwordLoading) && (
              <div className="border rounded-3 mt-1" style={{ position: 'absolute', insetInlineStart: 0, top: '100%', width: '100%', background: 'var(--surface)', zIndex: 1500, boxShadow: 'var(--card-shadow)' }}>
                {aiwordLoading && <div className="px-3 py-2 text-muted" style={{ fontSize: 13 }}>Поиск…</div>}
                {!aiwordLoading && aiwordOptions.map((opt, idx) => (
                  <button
                    key={opt.id}
                    type="button"
                    className="btn w-100 text-start"
                    style={{ border: 'none', borderBottom: idx === aiwordOptions.length - 1 ? 'none' : '1px solid var(--border)', borderRadius: 0 }}
                    onClick={() => addAiwordUser(opt)}
                  >
                    <div className="fw-semibold" style={{ fontSize: 13 }}>{opt.full_name || '—'}</div>
                    <div className="text-muted" style={{ fontSize: 12 }}>@{opt.username}</div>
                  </button>
                ))}
                {!aiwordLoading && aiwordOptions.length === 0 && <div className="px-3 py-2 text-muted" style={{ fontSize: 13 }}>Пользователи не найдены</div>}
              </div>
            )}
          </div>
        </div>
        <div className="d-flex flex-wrap gap-2 mt-3">
          {aiwordUsers.map(u => {
            const label = u.full_name?.trim() ? u.full_name : u.username
            return (
              <span key={u.user_id} className="badge bg-primary-subtle text-primary d-flex align-items-center gap-2" style={{ fontSize: 13 }}>
                <span>{label}</span>
                <button type="button" className="btn btn-sm btn-outline-primary" onClick={() => removeAiwordUser(u.user_id)} aria-label="Удалить доступ">×</button>
              </span>
            )
          })}
          {aiwordUsers.length === 0 && <span className="text-muted" style={{ fontSize: 13 }}>Доступ ещё не назначен ни одному пользователю.</span>}
        </div>
      </div>

      <div className="card p-3">
        <div className="fw-semibold mb-2">Коллекции</div>
        <div className="table-responsive">
          <table className="table table-sm align-middle">
            <thead>
              <tr>
                <th>Название</th>
                <th>Файлов</th>
                <th>Поиск</th>
                <th>Граф</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {(s.collections || []).map((c, idx) => (
                <tr key={c.id}>
                  <td>
                    <div className="fw-semibold">{c.name}</div>
                    <div className="text-muted" style={{ fontSize: 12 }}>
                      {c.slug ? `slug: ${c.slug}` : ''}
                    </div>
                  </td>
                  <td className="text-secondary">{c.count ?? 0}</td>
                  <td>
                    <input
                      className="form-check-input"
                      type="checkbox"
                      checked={c.searchable}
                      onChange={e => {
                        const next = [...(s.collections || [])]
                        next[idx] = { ...c, searchable: e.target.checked }
                        setS({ ...s, collections: next })
                      }}
                    />
                  </td>
                  <td>
                    <input
                      className="form-check-input"
                      type="checkbox"
                      checked={c.graphable}
                      onChange={e => {
                        const next = [...(s.collections || [])]
                        next[idx] = { ...c, graphable: e.target.checked }
                        setS({ ...s, collections: next })
                      }}
                    />
                  </td>
                  <td className="text-end">
                    <button
                      className="btn btn-sm btn-outline-danger"
                      onClick={() => deleteCollection(c.id)}
                      disabled={deleteCollectionId === c.id}
                    >
                      {deleteCollectionId === c.id ? '...' : 'Удалить'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <CollectionsEditor
          collections={s.collections || []}
          onSaved={async () => {
            const r = await fetch('/api/settings')
            const j = await r.json()
            const normalized = normalizeSettings(j)
            setS(normalized)
            setReindexUseLLM(boolVal(normalized.default_use_llm, true))
            setReindexPrune(boolVal(normalized.default_prune, true))
          }}
        />
      </div>

      <div className="card p-3">
        <div className="fw-semibold mb-2">Управление базой данных</div>
        <div className="d-flex flex-wrap gap-2 align-items-center">
          <button className="btn btn-outline-primary" onClick={backupDb}>Резервная копия</button>
          <form onSubmit={importDb} className="d-flex gap-2 align-items-center">
            <input className="form-control" type="file" name="dbfile" accept=".db" required />
            <button className="btn btn-outline-secondary" type="submit">Импорт .db</button>
          </form>
          <button className="btn btn-outline-danger" onClick={clearDb}>Очистить базу</button>
        </div>
      </div>

      <div className="card p-3">
        <details>
          <summary className="fw-semibold">Промпты LLM</summary>
          <div className="row g-2 mt-2">
            {promptKeys.map(key => {
              const label = promptLabels[key] || `Промпт ${key}`
              const current = s.prompts?.[key] ?? ''
              const defaultValue = promptDefaults[key] ?? ''
              const isDefault = current === defaultValue
              return (
                <div className="col-md-6" key={key}>
                  <label className="form-label d-flex flex-wrap align-items-center justify-content-between gap-2">
                    <span>{label}</span>
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-secondary"
                      onClick={() => resetPrompt(key)}
                      disabled={isDefault}
                    >
                      По умолчанию
                    </button>
                  </label>
                  <textarea
                    className="form-control"
                    rows={4}
                    value={current}
                    onChange={e => updatePrompt(key, e.target.value)}
                  />
                </div>
              )
            })}
          </div>
        </details>
      </div>
    </div>
  )
}

function CollectionsEditor({ collections, onSaved }: { collections: Collection[]; onSaved: () => void }) {
  const [newName, setNewName] = useState('')
  const [saving, setSaving] = useState(false)
  const save = async () => {
    setSaving(true)
    try {
      const fd = new FormData()
      for (const c of collections) {
        if (c.searchable) fd.set(`search_${c.id}`, 'on')
        if (c.graphable) fd.set(`graph_${c.id}`, 'on')
      }
      if (newName.trim()) fd.set('new_name', newName.trim())
      const r = await fetch('/settings/collections', { method: 'POST', body: fd })
      if (!r.ok) alert('Ошибка сохранения коллекций')
      setNewName('')
      onSaved()
    } catch {
      alert('Ошибка сети при сохранении коллекций')
    } finally {
      setSaving(false)
    }
  }
  return (
    <div className="d-flex flex-wrap align-items-center gap-2">
      <input className="form-control" placeholder="Новая коллекция" value={newName} onChange={e => setNewName(e.target.value)} style={{ maxWidth: 320 }} />
      <button className="btn btn-outline-primary" onClick={save} disabled={saving}>{saving ? 'Сохранение…' : 'Сохранить'}</button>
    </div>
  )
}

type MaterialTypesEditorProps = {
  materialTypes: MaterialTypeDefinition[]
  onChange: (index: number, value: MaterialTypeDefinition) => void
  onRemove: (index: number) => void
  onAdd: () => void
  duplicateKeys: Set<string>
}

type MaterialTypeCardProps = {
  value: MaterialTypeDefinition
  index: number
  onChange: (value: MaterialTypeDefinition) => void
  onRemove: () => void
  disableRemove?: boolean
  duplicateKey?: boolean
}

function MaterialTypesEditor({ materialTypes, onChange, onRemove, onAdd, duplicateKeys }: MaterialTypesEditorProps) {
  if (!materialTypes.length) {
    return (
      <div className="d-grid gap-3">
        <div className="text-muted" style={{ fontSize: 13 }}>Типы ещё не настроены.</div>
        <button className="btn btn-outline-primary" type="button" onClick={onAdd}>Добавить тип</button>
      </div>
    )
  }
  return (
    <div className="d-grid gap-3">
      {materialTypes.map((item, index) => (
        <MaterialTypeCard
          key={item.key ? `${item.key}-${index}` : `material-${index}`}
          value={item}
          index={index}
          onChange={next => onChange(index, next)}
          onRemove={() => onRemove(index)}
          disableRemove={item.key.trim().toLowerCase() === 'document'}
          duplicateKey={item.key ? duplicateKeys.has(item.key.trim().toLowerCase()) : false}
        />
      ))}
      <div>
        <button className="btn btn-outline-primary" type="button" onClick={onAdd}>Добавить тип</button>
      </div>
    </div>
  )
}

function MaterialTypeCard({ value, index, onChange, onRemove, disableRemove, duplicateKey }: MaterialTypeCardProps) {
  const [specialDraft, setSpecialDraft] = useState(() => formatSpecial(value.special))
  const [specialError, setSpecialError] = useState<string | null>(null)

  useEffect(() => {
    if (!specialError) {
      setSpecialDraft(formatSpecial(value.special))
    }
  }, [value.special, specialError])

  const idPrefix = useMemo(() => `mt-${index}-${value.key || 'new'}`, [index, value.key])

  const updateField = useCallback((patch: Partial<MaterialTypeDefinition>) => {
    onChange({ ...value, ...patch })
  }, [onChange, value])

  const handleNumberChange = useCallback((field: keyof MaterialTypeDefinition) => (event: React.ChangeEvent<HTMLInputElement>) => {
    const raw = event.target.value
    if (raw === '') {
      updateField({ [field]: undefined } as Partial<MaterialTypeDefinition>)
      return
    }
    const num = Number(raw)
    if (Number.isFinite(num)) {
      updateField({ [field]: num } as Partial<MaterialTypeDefinition>)
    }
  }, [updateField])

  const handleSpecialBlur = useCallback(() => {
    if (!specialDraft.trim()) {
      setSpecialError(null)
      updateField({ special: {} })
      return
    }
    try {
      const parsed = JSON.parse(specialDraft)
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        setSpecialError(null)
        updateField({ special: parsed })
      } else {
        setSpecialError('Ожидается JSON-объект')
      }
    } catch {
      setSpecialError('Некорректный JSON')
    }
  }, [specialDraft, updateField])

  return (
    <div className="border rounded-3 p-3" style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}>
      <div className="d-flex justify-content-between align-items-start gap-3">
        <div className="flex-grow-1">
          <label className="form-label">Идентификатор (slug)</label>
          <input
            className={`form-control${duplicateKey ? ' is-invalid' : ''}`}
            value={value.key}
            onChange={event => updateField({ key: event.target.value })}
            placeholder="report"
          />
          <div className="form-text">Используется для API, директорий и фильтров.</div>
          {duplicateKey && <div className="invalid-feedback d-block">Ключ уже используется</div>}
        </div>
        <div className="d-flex flex-column align-items-end gap-2">
          <div className="form-check form-switch">
            <input
              className="form-check-input"
              type="checkbox"
              id={`${idPrefix}-enabled`}
              checked={value.enabled !== false}
              onChange={event => updateField({ enabled: event.target.checked })}
            />
            <label className="form-check-label" htmlFor={`${idPrefix}-enabled`}>Активен</label>
          </div>
          <button className="btn btn-sm btn-outline-danger" type="button" onClick={onRemove} disabled={disableRemove}>Удалить</button>
        </div>
      </div>

      <div className="row g-3 mt-2">
        <div className="col-md-4">
          <label className="form-label">Название</label>
          <input className="form-control" value={value.label || ''} onChange={event => updateField({ label: event.target.value })} placeholder="Отчёт" />
          <div className="form-text">Отображается пользователям.</div>
        </div>
        <div className="col-md-4">
          <label className="form-label">LLM подсказка</label>
          <textarea className="form-control" rows={2} value={value.llm_hint || ''} onChange={event => updateField({ llm_hint: event.target.value })} />
        </div>
        <div className="col-md-4">
          <label className="form-label">Описание</label>
          <textarea className="form-control" rows={2} value={value.description || ''} onChange={event => updateField({ description: event.target.value })} />
        </div>
      </div>

      <div className="row g-3 mt-2">
        <div className="col-md-4">
          <label className="form-label">Ключевые слова в тексте</label>
          <textarea className="form-control" rows={3} value={formatListInput(value.text_keywords)} onChange={event => updateField({ text_keywords: parseListInput(event.target.value) })} placeholder="статья, журнал" />
          <div className="form-text">Через запятую или с новой строки.</div>
        </div>
        <div className="col-md-4">
          <label className="form-label">Ключевые слова в имени файла</label>
          <textarea className="form-control" rows={3} value={formatListInput(value.filename_keywords)} onChange={event => updateField({ filename_keywords: parseListInput(event.target.value) })} placeholder="report, отчёт" />
        </div>
        <div className="col-md-4">
          <label className="form-label">Исключающие слова</label>
          <textarea className="form-control" rows={3} value={formatListInput(value.exclude_keywords)} onChange={event => updateField({ exclude_keywords: parseListInput(event.target.value) })} placeholder="шаблон" />
        </div>
      </div>

      <div className="row g-3 mt-2">
        <div className="col-md-4">
          <label className="form-label">Расширения</label>
          <input className="form-control" value={formatListInput(value.extensions)} onChange={event => updateField({ extensions: parseListInput(event.target.value).map(token => token.toLowerCase()) })} placeholder="pdf, docx" />
        </div>
        <div className="col-md-4">
          <label className="form-label">Псевдонимы (aliases)</label>
          <input className="form-control" value={formatListInput(value.aliases)} onChange={event => updateField({ aliases: parseListInput(event.target.value) })} placeholder="alt-1, alt-2" />
        </div>
        <div className="col-md-4">
          <label className="form-label">Порядок этапов (flow)</label>
          <input className="form-control" value={formatListInput(value.flow)} onChange={event => updateField({ flow: parseListInput(event.target.value).map(token => token.toLowerCase()) })} placeholder="extension,filename,heuristics" />
          <div className="form-text">Оставьте пустым, чтобы использовать все этапы.</div>
        </div>
      </div>

      <details className="mt-3">
        <summary className="fw-semibold">Расширенные настройки</summary>
        <div className="row g-3 mt-2">
          <div className="col-md-3">
            <label className="form-label">Приоритет</label>
            <input className="form-control" type="number" value={value.priority ?? ''} onChange={handleNumberChange('priority')} />
          </div>
          <div className="col-md-3">
            <label className="form-label">Порог совпадений</label>
            <input className="form-control" type="number" step="0.1" value={value.threshold ?? ''} onChange={handleNumberChange('threshold')} />
          </div>
          <div className="col-md-2">
            <label className="form-label">Вес расширений</label>
            <input className="form-control" type="number" step="0.1" value={value.extension_weight ?? ''} onChange={handleNumberChange('extension_weight')} />
          </div>
          <div className="col-md-2">
            <label className="form-label">Вес имени</label>
            <input className="form-control" type="number" step="0.1" value={value.filename_weight ?? ''} onChange={handleNumberChange('filename_weight')} />
          </div>
          <div className="col-md-2">
            <label className="form-label">Вес текста</label>
            <input className="form-control" type="number" step="0.1" value={value.text_weight ?? ''} onChange={handleNumberChange('text_weight')} />
          </div>
        </div>
        <div className="d-flex flex-wrap gap-3 mt-3">
          <div className="form-check">
            <input className="form-check-input" type="checkbox" id={`${idPrefix}-req-ext`} checked={!!value.require_extension} onChange={event => updateField({ require_extension: event.target.checked })} />
            <label className="form-check-label" htmlFor={`${idPrefix}-req-ext`}>Требуется совпадение расширения</label>
          </div>
          <div className="form-check">
            <input className="form-check-input" type="checkbox" id={`${idPrefix}-req-name`} checked={!!value.require_filename} onChange={event => updateField({ require_filename: event.target.checked })} />
            <label className="form-check-label" htmlFor={`${idPrefix}-req-name`}>Требуется совпадение имени файла</label>
          </div>
          <div className="form-check">
            <input className="form-check-input" type="checkbox" id={`${idPrefix}-req-text`} checked={!!value.require_text} onChange={event => updateField({ require_text: event.target.checked })} />
            <label className="form-check-label" htmlFor={`${idPrefix}-req-text`}>Требуется совпадение текста</label>
          </div>
        </div>
        <div className="mt-3">
          <label className="form-label">Специальные правила (JSON)</label>
          <textarea
            className={`form-control${specialError ? ' is-invalid' : ''}`}
            rows={4}
            value={specialDraft}
            onChange={event => setSpecialDraft(event.target.value)}
            onBlur={handleSpecialBlur}
            placeholder='{"journal_toc_required": true, "min_toc_entries": 5}'
          />
          {specialError ? (
            <div className="invalid-feedback d-block">{specialError}</div>
          ) : (
            <div className="form-text">Например: {`{"journal_toc_required": true, "min_toc_entries": 5}`}</div>
          )}
        </div>
      </details>
    </div>
  )
}
