import { useMemo, useSyncExternalStore } from 'react'

export type MaterialTypeDefinition = {
  key: string
  label?: string
  description?: string
  llm_hint?: string
  enabled?: boolean
  priority?: number
  threshold?: number
  text_keywords?: string[]
  filename_keywords?: string[]
  extensions?: string[]
  exclude_keywords?: string[]
  require_extension?: boolean
  require_filename?: boolean
  require_text?: boolean
  extension_weight?: number
  filename_weight?: number
  text_weight?: number
  flow?: string[]
  special?: Record<string, any>
  aliases?: string[]
}

const DEFAULT_LABELS: Record<string, string> = {
  dissertation: 'Диссертация',
  dissertation_abstract: 'Автореферат',
  article: 'Статья',
  journal: 'Журнал',
  textbook: 'Учебник',
  monograph: 'Монография',
  report: 'Отчёт',
  patent: 'Патент',
  presentation: 'Презентация',
  proceedings: 'Труды',
  standard: 'Стандарт',
  note: 'Заметка',
  document: 'Документ',
  file: 'Файл',
  audio: 'Аудио',
  image: 'Изображение',
  video: 'Видео',
  other: 'Другое',
}

const DEFAULT_TYPES: MaterialTypeDefinition[] = Object.entries(DEFAULT_LABELS).map(([key, label]) => ({
  key,
  label,
}))

let materialTypesState: MaterialTypeDefinition[] = DEFAULT_TYPES.map(entry => ({ ...entry }))
let labelMap: Record<string, string> = buildLabelMap(materialTypesState)
let loadPromise: Promise<void> | null = null

const subscribers = new Set<() => void>()

function emit(): void {
  subscribers.forEach(listener => listener())
}

function subscribe(listener: () => void): () => void {
  subscribers.add(listener)
  return () => subscribers.delete(listener)
}

function getSnapshot(): MaterialTypeDefinition[] {
  return materialTypesState
}

function sanitizeStringArray(value: any): string[] {
  if (!Array.isArray(value)) return []
  return value.map(item => String(item ?? '').trim()).filter(Boolean)
}

function sanitizeMaterialType(raw: any): MaterialTypeDefinition | null {
  if (!raw || typeof raw !== 'object') return null
  const key = String(raw.key ?? '').trim()
  if (!key) return null
  const entry: MaterialTypeDefinition = {
    key,
  }
  if (raw.label !== undefined && raw.label !== null) entry.label = String(raw.label)
  if (raw.description !== undefined && raw.description !== null) entry.description = String(raw.description)
  if (raw.llm_hint !== undefined && raw.llm_hint !== null) entry.llm_hint = String(raw.llm_hint)
  if (raw.enabled !== undefined) entry.enabled = !!raw.enabled
  if (raw.priority !== undefined && Number.isFinite(Number(raw.priority))) entry.priority = Number(raw.priority)
  if (raw.threshold !== undefined && Number.isFinite(Number(raw.threshold))) entry.threshold = Number(raw.threshold)
  if (raw.extension_weight !== undefined && Number.isFinite(Number(raw.extension_weight))) entry.extension_weight = Number(raw.extension_weight)
  if (raw.filename_weight !== undefined && Number.isFinite(Number(raw.filename_weight))) entry.filename_weight = Number(raw.filename_weight)
  if (raw.text_weight !== undefined && Number.isFinite(Number(raw.text_weight))) entry.text_weight = Number(raw.text_weight)
  if (raw.require_extension !== undefined) entry.require_extension = !!raw.require_extension
  if (raw.require_filename !== undefined) entry.require_filename = !!raw.require_filename
  if (raw.require_text !== undefined) entry.require_text = !!raw.require_text
  const textKeywords = sanitizeStringArray(raw.text_keywords ?? raw.textKeywords)
  if (textKeywords.length) entry.text_keywords = textKeywords
  const filenameKeywords = sanitizeStringArray(raw.filename_keywords ?? raw.filenameKeywords)
  if (filenameKeywords.length) entry.filename_keywords = filenameKeywords
  const extensions = sanitizeStringArray(raw.extensions)
  if (extensions.length) entry.extensions = extensions
  const excludeKeywords = sanitizeStringArray(raw.exclude_keywords ?? raw.excludeKeywords)
  if (excludeKeywords.length) entry.exclude_keywords = excludeKeywords
  const flow = sanitizeStringArray(raw.flow)
  if (flow.length) entry.flow = flow
  const aliases = sanitizeStringArray(raw.aliases)
  if (aliases.length) entry.aliases = aliases
  if (raw.special && typeof raw.special === 'object') {
    entry.special = { ...(raw.special as Record<string, any>) }
  }
  return entry
}

function buildLabelMap(list: MaterialTypeDefinition[], overrides?: Record<string, string> | null): Record<string, string> {
  const map: Record<string, string> = {}
  Object.entries(DEFAULT_LABELS).forEach(([key, label]) => {
    const normalized = key.trim().toLowerCase()
    if (normalized) map[normalized] = label
  })
  list.forEach(entry => {
    const rawKey = String(entry.key ?? '').trim()
    if (!rawKey) return
    const normalized = rawKey.toLowerCase()
    const overrideLabel = overrides && typeof overrides[normalized] === 'string' ? String(overrides[normalized]) : undefined
    const label = overrideLabel || entry.label || DEFAULT_LABELS[normalized] || rawKey
    map[normalized] = label
    const aliases = entry.aliases || []
    aliases.forEach(alias => {
      const aliasNormalized = String(alias ?? '').trim().toLowerCase()
      if (aliasNormalized) {
        map[aliasNormalized] = label
      }
    })
  })
  if (overrides) {
    Object.entries(overrides).forEach(([key, label]) => {
      const normalized = String(key ?? '').trim().toLowerCase()
      if (normalized && label) {
        map[normalized] = String(label)
      }
    })
  }
  return map
}

function ensureDefaults(list: MaterialTypeDefinition[]): MaterialTypeDefinition[] {
  const seen = new Set(list.map(entry => entry.key.trim().toLowerCase()))
  DEFAULT_TYPES.forEach(entry => {
    const normalized = entry.key.trim().toLowerCase()
    if (!seen.has(normalized)) {
      seen.add(normalized)
      list.push({ ...entry })
    }
  })
  return list
}

function getOptionsSnapshot(list: MaterialTypeDefinition[]): { value: string; label: string }[] {
  const unique = new Set<string>()
  const options: { value: string; label: string }[] = []
  list.forEach(entry => {
    const value = String(entry.key ?? '').trim().toLowerCase()
    if (!value || unique.has(value)) return
    unique.add(value)
    if (entry.enabled === false) return
    const label = labelMap[value] || entry.label || DEFAULT_LABELS[value] || value
    options.push({ value, label })
  })
  return options
}

export function useMaterialTypes(): MaterialTypeDefinition[] {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
}

export function getMaterialTypesSnapshot(): MaterialTypeDefinition[] {
  return materialTypesState
}

export function useMaterialTypeOptions(): { value: string; label: string }[] {
  const list = useMaterialTypes()
  return useMemo(() => getOptionsSnapshot(list), [list])
}

export function getMaterialTypeOptionsSnapshot(): { value: string; label: string }[] {
  return getOptionsSnapshot(materialTypesState)
}

export function materialTypeLabel(type: string | null | undefined, fallback: string = 'Другое'): string {
  if (!type) return fallback
  const normalized = String(type).trim().toLowerCase()
  if (!normalized) return fallback
  return labelMap[normalized] || fallback
}

export function materialTypeSlug(input: string | null | undefined): string | null {
  if (!input) return null
  const trimmed = input.trim()
  if (!trimmed) return null
  const normalized = trimmed.toLowerCase()
  if (labelMap[normalized]) return normalized
  const match = Object.entries(labelMap).find(([, label]) => String(label).toLowerCase() === normalized)
  if (match) return match[0]
  return trimmed
}

export function resetMaterialTypes(): void {
  materialTypesState = DEFAULT_TYPES.map(entry => ({ ...entry }))
  labelMap = buildLabelMap(materialTypesState)
  emit()
}

export function updateMaterialTypesFromServer(items: any[], labels?: Record<string, string> | null): void {
  const sanitized = Array.isArray(items)
    ? items.map(sanitizeMaterialType).filter((entry): entry is MaterialTypeDefinition => Boolean(entry))
    : []
  if (!sanitized.length) {
    materialTypesState = DEFAULT_TYPES.map(entry => ({ ...entry }))
    labelMap = buildLabelMap(materialTypesState, labels ?? null)
    emit()
    return
  }
  const seen = new Set<string>()
  const unique: MaterialTypeDefinition[] = []
  sanitized.forEach(entry => {
    const normalized = entry.key.trim().toLowerCase()
    if (!normalized || seen.has(normalized)) return
    seen.add(normalized)
    unique.push(entry)
  })
  materialTypesState = ensureDefaults(unique)
  labelMap = buildLabelMap(materialTypesState, labels ?? null)
  emit()
}

export async function loadMaterialTypes(force = false): Promise<void> {
  if (loadPromise && !force) return loadPromise
  loadPromise = fetch('/api/material-types')
    .then(async response => {
      if (!response.ok) {
        if (response.status === 401 || response.status === 403) {
          resetMaterialTypes()
          return
        }
        throw new Error(`material-types:${response.status}`)
      }
      const data = await response.json().catch(() => null)
      if (data && data.ok && Array.isArray(data.items)) {
        const labels = data.labels && typeof data.labels === 'object' ? data.labels : null
        updateMaterialTypesFromServer(data.items, labels)
      } else {
        resetMaterialTypes()
      }
    })
    .catch(() => {
      resetMaterialTypes()
    })
    .finally(() => {
      loadPromise = null
    })
  return loadPromise
}
