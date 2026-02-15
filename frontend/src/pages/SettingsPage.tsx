import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useAuth } from '../ui/Auth'
import { useToasts } from '../ui/Toasts'
import AdminFacetSettingsPage from './AdminFacetSettingsPage'
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
  context_length?: number | null
  instances?: number
  purpose?: string | null
  purposes?: string[]
  provider?: string
}
type LlmPurposeOption = { id: string; label: string }
type LlmProviderOption = { id: string; label: string }
type AiwordAccessUser = { user_id: number; username?: string | null; full_name?: string | null }
type UserSuggestion = { id: number; username: string; full_name?: string | null }
type RuntimeField = {
  name: string
  api_key?: string
  group: string
  type: string
  description: string
  env_key?: string | null
  runtime_mutable: boolean
  visibility?: 'ui_safe' | 'ui_expert' | 'env_only'
  restart_required?: boolean
  constraints?: Record<string, any>
  ui_component?: string
  depends_on?: Record<string, any> | null
  risk_level?: 'low' | 'medium' | 'high'
  value: any
}
type Settings = {
  scan_root: string
  extract_text: boolean
  lm_base: string
  lm_model: string
  lm_key: string
  lm_provider: string
  rag_embedding_backend: string
  rag_embedding_model: string
  rag_embedding_dim: number
  rag_embedding_batch: number
  rag_embedding_device?: string | null
  rag_embedding_endpoint: string
  rag_embedding_api_key: string
  transcribe_enabled: boolean
  transcribe_backend: string
  transcribe_model: string
  transcribe_language: string
  summarize_audio: boolean
  audio_keywords_llm: boolean
  vision_images: boolean
  kw_to_tags: boolean
  doc_chat_chunk_max_tokens: number
  doc_chat_chunk_overlap: number
  doc_chat_chunk_min_tokens: number
  doc_chat_max_chunks: number
  doc_chat_fallback_chunks: number
  doc_chat_image_min_width: number
  doc_chat_image_min_height: number
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
  ai_rag_retry_enabled?: boolean
  ai_rag_retry_threshold?: number
  ai_query_variants_max?: number
  collections?: Collection[]
  llm_endpoints?: LlmEndpointInfo[]
  llm_purposes?: LlmPurposeOption[]
  llm_providers?: LlmProviderOption[]
  aiword_users?: AiwordAccessUser[]
  default_use_llm?: boolean
  default_prune?: boolean
  type_dirs?: Record<string, string>
  material_types?: MaterialTypeDefinition[]
  runtime_fields?: RuntimeField[]
  settings_hub_v2_enabled?: boolean
  database_dialect?: 'sqlite' | 'postgresql' | string
  database_uri?: string
  database_type_options?: string[]
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

const runtimeGroupLabels: Record<string, string> = {
  llm: 'LLM',
  llm_pool: 'LLM Pool',
  rag: 'RAG',
  doc_chat: 'Чат по документу',
  scan: 'Сканирование',
  ocr: 'OCR',
  transcription: 'Транскрибация',
  vision: 'Vision',
  cache: 'Кеши',
  facets: 'Фасеты',
  type_detection: 'Определение типов',
  feedback: 'Feedback',
  prompts: 'Промпты',
  general: 'Общие',
  system: 'Система',
  security: 'Безопасность',
}

const runtimeFieldI18n: Record<string, { label: string; description: string }> = {
  lmstudio_api_base: { label: 'Базовый URL LLM API', description: 'Адрес OpenAI-совместимого endpoint для LLM-запросов.' },
  lmstudio_model: { label: 'Модель LLM по умолчанию', description: 'Имя модели, используемой в маршруте по умолчанию.' },
  lmstudio_api_key: { label: 'API-ключ LLM', description: 'Ключ доступа к LLM endpoint.' },
  lm_default_provider: { label: 'Провайдер LLM по умолчанию', description: 'Провайдер, используемый как fallback для LLM-маршрута.' },
  lm_max_input_chars: { label: 'Максимум символов во входе LLM', description: 'Ограничение длины текста перед отправкой в модель.' },
  lm_max_output_tokens: { label: 'Максимум выходных токенов LLM', description: 'Ограничение размера ответа модели.' },
  ai_rerank_llm: { label: 'LLM-реранжирование', description: 'Использовать LLM для реранжирования результатов поиска.' },
  azure_openai_api_version: { label: 'Версия Azure OpenAI API', description: 'Версия API для совместимых Azure endpoint.' },
  llm_pool_global_concurrency: { label: 'Глобальная конкуррентность LLM-пула', description: 'Сколько LLM-задач может выполняться одновременно во всем приложении.' },
  llm_pool_per_user_concurrency: { label: 'Конкуррентность LLM-пула на пользователя', description: 'Сколько LLM-задач одновременно разрешено одному пользователю.' },
  llm_queue_max_size: { label: 'Максимальный размер LLM-очереди', description: 'Общий лимит задач в очереди LLM.' },
  llm_queue_per_user_max: { label: 'Лимит LLM-очереди на пользователя', description: 'Максимум задач LLM в очереди для одного пользователя.' },
  llm_request_timeout_sec: { label: 'Таймаут LLM-запроса (сек)', description: 'Максимальное время ожидания ответа LLM.' },
  llm_retry_count: { label: 'Количество повторов LLM-запроса', description: 'Сколько раз повторять запрос при временных ошибках.' },
  llm_retry_backoff_ms: { label: 'Задержка между повторами (мс)', description: 'Пауза перед повторным LLM-запросом.' },
  lmstudio_idle_unload_minutes: { label: 'Автовыгрузка моделей LM Studio (мин)', description: 'Через сколько минут простоя выгружать все модели из LM Studio. 0 — отключено.' },
  rag_embedding_backend: { label: 'RAG backend эмбеддингов', description: 'Источник/механизм генерации эмбеддингов для RAG.' },
  rag_embedding_model: { label: 'RAG модель эмбеддингов', description: 'Имя модели эмбеддингов.' },
  rag_embedding_dim: { label: 'Размерность эмбеддингов', description: 'Длина вектора эмбеддинга.' },
  rag_embedding_batch_size: { label: 'Batch эмбеддингов', description: 'Сколько элементов обрабатывать за один запрос эмбеддингов.' },
  rag_embedding_device: { label: 'Устройство эмбеддингов', description: 'Устройство для локальных эмбеддингов (например, cpu/cuda:0).' },
  rag_embedding_endpoint: { label: 'Endpoint эмбеддингов', description: 'URL endpoint для получения эмбеддингов.' },
  rag_embedding_api_key: { label: 'API-ключ эмбеддингов', description: 'Ключ доступа к endpoint эмбеддингов.' },
  rag_rerank_backend: { label: 'RAG backend реранкера', description: 'Бэкенд для реранжирования RAG-кандидатов.' },
  rag_rerank_model: { label: 'RAG модель реранкера', description: 'Имя модели реранкера.' },
  rag_rerank_device: { label: 'Устройство реранкера', description: 'Устройство выполнения реранкера.' },
  rag_rerank_batch_size: { label: 'Batch реранкера', description: 'Размер пакета при реранжировании.' },
  rag_rerank_max_length: { label: 'Максимальная длина реранкера', description: 'Максимальная длина входа в реранкер.' },
  rag_rerank_max_chars: { label: 'Максимум символов на фрагмент', description: 'Лимит символов фрагмента для реранжирования.' },
  ai_query_variants_max: { label: 'Максимум вариантов запроса', description: 'Сколько вариантов запроса генерировать для AI-поиска.' },
  ai_rag_retry_enabled: { label: 'Автоповтор RAG', description: 'Повторять RAG с расширением контекста при низкой уверенности.' },
  ai_rag_retry_threshold: { label: 'Порог автоповтора RAG', description: 'Порог уверенности, ниже которого запускается повтор.' },
  doc_chat_chunk_max_tokens: { label: 'DocChat: максимум токенов чанка', description: 'Максимальный размер чанка в токенах.' },
  doc_chat_chunk_overlap: { label: 'DocChat: перекрытие чанков', description: 'Сколько токенов повторяется между соседними чанками.' },
  doc_chat_chunk_min_tokens: { label: 'DocChat: минимум токенов чанка', description: 'Минимальный размер чанка в токенах.' },
  doc_chat_max_chunks: { label: 'DocChat: лимит чанков', description: 'Максимум чанков в контексте (0 — без ограничения).' },
  doc_chat_fallback_chunks: { label: 'DocChat: fallback чанки', description: 'Дополнительные чанки при слабом основном результате.' },
  doc_chat_image_min_width: { label: 'DocChat: мин. ширина изображения', description: 'Минимальная ширина изображения для анализа.' },
  doc_chat_image_min_height: { label: 'DocChat: мин. высота изображения', description: 'Минимальная высота изображения для анализа.' },
  scan_root: { label: 'Корневая папка сканирования', description: 'Базовый каталог, который сканирует индексатор.' },
  extract_text: { label: 'Извлекать текст', description: 'Извлекать текст из документов во время сканирования.' },
  import_subdir: { label: 'Подпапка импорта', description: 'Имя подпапки для импортируемых файлов.' },
  collections_in_separate_dirs: { label: 'Коллекции в отдельных папках', description: 'Хранить коллекции в отдельных директориях.' },
  collection_type_subdirs: { label: 'Подпапки по типам', description: 'Создавать подпапки по типам материалов внутри коллекции.' },
  move_on_rename: { label: 'Перемещать при переименовании', description: 'Перемещать файл при изменении его имени/пути.' },
  default_use_llm: { label: 'LLM по умолчанию при переиндексации', description: 'Использовать LLM-обработку при запуске сканирования.' },
  default_prune: { label: 'Удалять отсутствующие файлы', description: 'Удалять из каталога записи для пропавших файлов.' },
  ocr_langs: { label: 'OCR языки', description: 'Языки OCR в формате tesseract, например rus+eng.' },
  pdf_ocr_pages: { label: 'Количество OCR-страниц PDF', description: 'Сколько страниц PDF обрабатывать OCR.' },
  always_ocr_first_page_dissertation: { label: 'OCR первой страницы диссертаций', description: 'Всегда распознавать первую страницу диссертации.' },
  transcribe_enabled: { label: 'Транскрибация включена', description: 'Включить распознавание аудио.' },
  transcribe_backend: { label: 'Бэкенд транскрибации', description: 'Движок, используемый для распознавания аудио.' },
  transcribe_model_path: { label: 'Модель/путь транскрибации', description: 'Путь или идентификатор модели транскрибации.' },
  transcribe_language: { label: 'Язык транскрибации', description: 'Язык распознавания аудио по умолчанию.' },
  summarize_audio: { label: 'Суммаризация аудио', description: 'Генерировать краткое содержание после транскрибации.' },
  audio_keywords_llm: { label: 'Ключевые слова из аудио через LLM', description: 'Извлекать ключевые слова из аудио с помощью LLM.' },
  images_vision_enabled: { label: 'Vision-анализ изображений', description: 'Разрешить анализ изображений через LLM vision.' },
  keywords_to_tags_enabled: { label: 'Ключевые слова в теги', description: 'Конвертировать извлеченные ключевые слова в теги.' },
  llm_cache_enabled: { label: 'LLM-кеш включен', description: 'Кешировать ответы LLM.' },
  llm_cache_ttl_seconds: { label: 'TTL LLM-кеша (сек)', description: 'Время жизни записей LLM-кеша.' },
  llm_cache_max_items: { label: 'Лимит элементов LLM-кеша', description: 'Максимальное число записей в LLM-кеше.' },
  llm_cache_only_mode: { label: 'Только LLM-кеш', description: 'Не ходить в API, использовать только кеш.' },
  search_cache_enabled: { label: 'Кеш поиска включен', description: 'Кешировать результаты поиска.' },
  search_cache_ttl_seconds: { label: 'TTL кеша поиска (сек)', description: 'Время жизни записей кеша поиска.' },
  search_cache_max_items: { label: 'Лимит элементов кеша поиска', description: 'Максимум записей в кеше поиска.' },
  search_facet_tag_keys: { label: 'Разрешенные теги фасетов поиска', description: 'Список ключей тегов для фасетов поиска.' },
  graph_facet_tag_keys: { label: 'Разрешенные теги фасетов графа', description: 'Список ключей тегов для фасетов графа.' },
  search_facet_include_types: { label: 'Фасет типов в поиске', description: 'Показывать фасет по типам материалов.' },
  type_detect_flow: { label: 'Пайплайн определения типа', description: 'Порядок этапов определения типа документа.' },
  type_llm_override: { label: 'LLM может переопределять тип', description: 'Разрешить LLM менять тип после правил.' },
  type_dirs: { label: 'Сопоставление типов и папок', description: 'Карта type -> директория хранения.' },
  material_types: { label: 'Профили типов материалов', description: 'Набор правил и параметров классификации типов.' },
  feedback_train_interval_hours: { label: 'Интервал обучения feedback (ч)', description: 'Как часто запускать дообучение по обратной связи.' },
  feedback_train_cutoff_days: { label: 'Глубина feedback-данных (дни)', description: 'За сколько дней брать данные для обучения.' },
  prompts: { label: 'Промпты', description: 'Кастомные промпты для LLM-задач.' },
}

const getRuntimeFieldLabel = (field: RuntimeField): string =>
  runtimeFieldI18n[field.name]?.label || field.name

const getRuntimeFieldDescription = (field: RuntimeField): string =>
  runtimeFieldI18n[field.name]?.description || field.description || 'Без описания'

const SAFE_GROUPS = new Set(['scan', 'ocr', 'llm', 'rag', 'doc_chat', 'transcription', 'vision'])
const EXPERT_GROUPS = new Set(['llm_pool', 'cache', 'feedback', 'type_detection', 'prompts', 'general', 'system', 'security'])

const stableStringify = (value: any): string => {
  if (value === undefined) return 'undefined'
  try {
    return JSON.stringify(value, Object.keys(value || {}).sort())
  } catch {
    return String(value)
  }
}

const normalizeRuntimeValue = (field: RuntimeField, raw: string | boolean): any => {
  const type = (field.type || '').toLowerCase()
  if (type.includes('bool')) {
    return typeof raw === 'boolean' ? raw : String(raw).trim().toLowerCase() === 'true'
  }
  if (type.includes('int')) {
    const parsed = parseInt(String(raw), 10)
    return Number.isFinite(parsed) ? parsed : field.value
  }
  if (type.includes('float')) {
    const parsed = parseFloat(String(raw))
    return Number.isFinite(parsed) ? parsed : field.value
  }
  if (type.includes('list') || type.includes('dict') || type.includes('optional')) {
    const text = String(raw ?? '').trim()
    if (!text) return null
    try {
      return JSON.parse(text)
    } catch {
      if (type.includes('list')) {
        return text.split(/[,;\n]/).map(token => token.trim()).filter(Boolean)
      }
      return text
    }
  }
  return raw
}

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
    rag_embedding_backend: String(raw?.rag_embedding_backend || 'lm-studio'),
    rag_embedding_model: String(raw?.rag_embedding_model || 'nomic-ai/nomic-embed-text-v1.5-GGUF'),
    rag_embedding_dim: Number.isFinite(raw?.rag_embedding_dim) ? Number(raw.rag_embedding_dim) : 384,
    rag_embedding_batch: Number.isFinite(raw?.rag_embedding_batch) ? Number(raw.rag_embedding_batch) : 32,
    rag_embedding_device: raw?.rag_embedding_device ? String(raw.rag_embedding_device) : '',
    rag_embedding_endpoint: String(raw?.rag_embedding_endpoint || raw?.lm_base || ''),
    rag_embedding_api_key: String(raw?.rag_embedding_api_key || raw?.lm_key || ''),
    transcribe_enabled: boolVal(raw?.transcribe_enabled, true),
    transcribe_backend: String(raw?.transcribe_backend || 'faster-whisper'),
    transcribe_model: String(raw?.transcribe_model || ''),
    transcribe_language: String(raw?.transcribe_language || 'ru'),
    summarize_audio: boolVal(raw?.summarize_audio, true),
    audio_keywords_llm: boolVal(raw?.audio_keywords_llm, true),
    vision_images: boolVal(raw?.vision_images, false),
    kw_to_tags: boolVal(raw?.kw_to_tags, true),
    doc_chat_chunk_max_tokens: Number.isFinite(raw?.doc_chat_chunk_max_tokens)
      ? Number(raw.doc_chat_chunk_max_tokens)
      : 700,
    doc_chat_chunk_overlap: Number.isFinite(raw?.doc_chat_chunk_overlap)
      ? Number(raw.doc_chat_chunk_overlap)
      : 120,
    doc_chat_chunk_min_tokens: Number.isFinite(raw?.doc_chat_chunk_min_tokens)
      ? Number(raw.doc_chat_chunk_min_tokens)
      : 80,
    doc_chat_max_chunks: Number.isFinite(raw?.doc_chat_max_chunks)
      ? Number(raw.doc_chat_max_chunks)
      : 0,
    doc_chat_fallback_chunks: Number.isFinite(raw?.doc_chat_fallback_chunks)
      ? Number(raw.doc_chat_fallback_chunks)
      : 0,
    doc_chat_image_min_width: Number.isFinite(raw?.doc_chat_image_min_width)
      ? Number(raw.doc_chat_image_min_width)
      : 0,
    doc_chat_image_min_height: Number.isFinite(raw?.doc_chat_image_min_height)
      ? Number(raw.doc_chat_image_min_height)
      : 0,
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
    ai_query_variants_max: Number.isFinite(raw?.ai_query_variants_max) ? Number(raw.ai_query_variants_max) : 0,
    ai_rag_retry_enabled: boolVal(raw?.ai_rag_retry_enabled, true),
    ai_rag_retry_threshold: Number.isFinite(raw?.ai_rag_retry_threshold) ? Number(raw.ai_rag_retry_threshold) : 0.6,
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
    runtime_fields: Array.isArray(raw?.runtime_fields) ? raw.runtime_fields : [],
    settings_hub_v2_enabled: raw?.settings_hub_v2_enabled !== false,
    database_dialect: String(raw?.database_dialect || 'sqlite'),
    database_uri: String(raw?.database_uri || ''),
    database_type_options: Array.isArray(raw?.database_type_options) ? raw.database_type_options : ['sqlite', 'postgresql'],
  }
}

const ragEmbeddingBackendOptions: Array<{ id: string; label: string }> = [
  { id: 'lm-studio', label: 'LM Studio / OpenAI совместимый' },
  { id: 'sentence-transformers', label: 'Sentence Transformers' },
  { id: 'hash', label: 'Hash (псевдослучайный, оффлайн)' },
  { id: 'auto', label: 'Auto (по умолчанию)' },
]

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
  const [llmContextDrafts, setLlmContextDrafts] = useState<Record<number, string>>({})
  const [llmInstancesDrafts, setLlmInstancesDrafts] = useState<Record<number, string>>({})
  const [lmstudioProbeUrl, setLmstudioProbeUrl] = useState('')
  const [lmstudioLoading, setLmstudioLoading] = useState(false)
  const [lmstudioModelsByBase, setLmstudioModelsByBase] = useState<Record<string, Array<{ id: string; name: string; loaded?: boolean | null }>>>({})
  const [newMappingPurpose, setNewMappingPurpose] = useState('default')
  const [newMappingModel, setNewMappingModel] = useState('')
  const [newMappingName, setNewMappingName] = useState('')
  const [newMappingWeight, setNewMappingWeight] = useState('1')
  const [newMappingProvider, setNewMappingProvider] = useState('openai')
  const [newMappingContext, setNewMappingContext] = useState('')
  const [newMappingInstances, setNewMappingInstances] = useState('1')
  const [activeTab, setActiveTab] = useState<'safe' | 'expert' | 'llm' | 'facets'>('safe')
  const [runtimeSearch, setRuntimeSearch] = useState('')
  const [runtimeChangedOnly, setRuntimeChangedOnly] = useState(false)
  const [runtimeBaseline, setRuntimeBaseline] = useState<Record<string, any>>({})
  const [dbType, setDbType] = useState<'sqlite' | 'postgresql'>('sqlite')
  const [pgBackupFormat, setPgBackupFormat] = useState<'dump' | 'sql'>('dump')
  const [migrationSqlitePath, setMigrationSqlitePath] = useState('catalogue.db')
  const [migrationPgUrl, setMigrationPgUrl] = useState('')
  const [migrationPgHost, setMigrationPgHost] = useState('localhost')
  const [migrationPgPort, setMigrationPgPort] = useState('5432')
  const [migrationPgDb, setMigrationPgDb] = useState('agregator')
  const [migrationPgUser, setMigrationPgUser] = useState('agregator')
  const [migrationPgPassword, setMigrationPgPassword] = useState('')
  const [migrationMode, setMigrationMode] = useState<'dry-run' | 'run'>('dry-run')
  const [migrationBusy, setMigrationBusy] = useState(false)

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
        const baseline: Record<string, any> = {}
        ;(normalized.runtime_fields || []).forEach((field) => {
          baseline[field.name] = field.value
        })
        setRuntimeBaseline(baseline)
        setReindexUseLLM(boolVal(normalized.default_use_llm, true))
        setReindexPrune(boolVal(normalized.default_prune, true))
        setDbType((normalized.database_dialect === 'postgresql' ? 'postgresql' : 'sqlite'))
        const uri = normalized.database_uri || ''
        setMigrationPgUrl(uri)
        if (uri) {
          try {
            const parsed = new URL(uri)
            setMigrationPgHost(parsed.hostname || 'localhost')
            setMigrationPgPort(parsed.port || '5432')
            setMigrationPgDb((parsed.pathname || '/agregator').replace(/^\//, '') || 'agregator')
            setMigrationPgUser(parsed.username ? decodeURIComponent(parsed.username) : 'agregator')
          } catch {
            // keep defaults when URI is not parseable
          }
        }
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
      setLlmContextDrafts({})
      setLlmInstancesDrafts({})
      return
    }
    const map: Record<number, string> = {}
    const contextMap: Record<number, string> = {}
    const instancesMap: Record<number, string> = {}
    s.llm_endpoints.forEach(ep => { map[ep.id] = String(ep.weight ?? 1) })
    s.llm_endpoints.forEach(ep => { contextMap[ep.id] = ep.context_length ? String(ep.context_length) : '' })
    s.llm_endpoints.forEach(ep => { instancesMap[ep.id] = String(ep.instances ?? 1) })
    setLlmWeights(map)
    setLlmContextDrafts(contextMap)
    setLlmInstancesDrafts(instancesMap)
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
  useEffect(() => {
    if (!llmProviderOptions.length) return
    if (!llmProviderOptions.some(opt => opt.id === newMappingProvider)) {
      setNewMappingProvider(llmProviderOptions[0].id)
    }
  }, [llmProviderOptions, newMappingProvider])
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

  const patchLlmEndpoint = useCallback(async (
    id: number,
    payload: Record<string, unknown>,
    successMessage = 'LLM обновлена',
  ): Promise<LlmEndpointInfo | null> => {
    try {
      const r = await fetch(`/api/admin/llm-endpoints/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data?.ok && data.item) {
        toasts.push(successMessage, 'success')
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

  const probeLmstudioModels = useCallback(async (baseUrlRaw?: string, endpointId?: number) => {
    const baseUrl = (baseUrlRaw || lmstudioProbeUrl || '').trim()
    if (!baseUrl) {
      toasts.push('Укажите base URL LM Studio', 'error')
      return
    }
    setLmstudioLoading(true)
    try {
      const query = new URLSearchParams({ base_url: baseUrl })
      if (endpointId) query.set('endpoint_id', String(endpointId))
      const r = await fetch(`/api/admin/lmstudio/models?${query.toString()}`)
      const data = await r.json().catch(() => ({}))
      if (!r.ok || !data?.ok) {
        throw new Error(data?.error || 'Не удалось получить список моделей LM Studio')
      }
      const items = Array.isArray(data.items) ? data.items : []
      setLmstudioModelsByBase(prev => ({ ...prev, [baseUrl]: items }))
      if (!newMappingModel && items.length > 0) {
        setNewMappingModel(String(items[0].id || ''))
      }
      toasts.push(`Получено моделей: ${items.length}`, 'success')
    } catch (error: any) {
      toasts.push(String(error?.message || error || 'Ошибка проверки моделей'), 'error')
    } finally {
      setLmstudioLoading(false)
    }
  }, [lmstudioProbeUrl, newMappingModel, toasts])

  const loadLmstudioModel = useCallback(async (endpoint: LlmEndpointInfo) => {
    try {
      const contextRaw = llmContextDrafts[endpoint.id]
      const instancesRaw = llmInstancesDrafts[endpoint.id]
      const payload: Record<string, unknown> = { endpoint_id: endpoint.id }
      if (contextRaw && Number.isFinite(Number(contextRaw)) && Number(contextRaw) > 0) {
        payload.context_length = Math.max(256, parseInt(contextRaw, 10))
      }
      if (instancesRaw && Number.isFinite(Number(instancesRaw)) && Number(instancesRaw) > 0) {
        payload.instances = Math.max(1, parseInt(instancesRaw, 10))
      }
      const r = await fetch('/api/admin/lmstudio/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await r.json().catch(() => ({}))
      if (!r.ok || !data?.ok) {
        throw new Error(data?.error || 'Не удалось загрузить модель в LM Studio')
      }
      toasts.push(`Модель ${endpoint.model} загружена`, 'success')
      await probeLmstudioModels(endpoint.base_url, endpoint.id)
    } catch (error: any) {
      toasts.push(String(error?.message || error || 'Ошибка загрузки модели'), 'error')
    }
  }, [llmContextDrafts, llmInstancesDrafts, probeLmstudioModels, toasts])

  const unloadLmstudioModel = useCallback(async (endpoint: LlmEndpointInfo) => {
    try {
      const r = await fetch('/api/admin/lmstudio/models/unload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ endpoint_id: endpoint.id }),
      })
      const data = await r.json().catch(() => ({}))
      if (!r.ok || !data?.ok) {
        throw new Error(data?.error || 'Не удалось выгрузить модель из LM Studio')
      }
      toasts.push(`Модель ${endpoint.model} выгружена`, 'success')
      await probeLmstudioModels(endpoint.base_url, endpoint.id)
    } catch (error: any) {
      toasts.push(String(error?.message || error || 'Ошибка выгрузки модели'), 'error')
    }
  }, [probeLmstudioModels, toasts])

  const createTaskMapping = useCallback(async () => {
    const baseUrl = (lmstudioProbeUrl || '').trim()
    if (!baseUrl) {
      toasts.push('Укажите Base URL LM Studio', 'error')
      return
    }
    if (!newMappingModel.trim()) {
      toasts.push('Выберите модель из списка', 'error')
      return
    }
    const purpose = (newMappingPurpose || 'default').trim()
    const name = (newMappingName || `${purpose}:${newMappingModel}`).trim()
    const payload: Record<string, unknown> = {
      name,
      base_url: baseUrl,
      model: newMappingModel.trim(),
      weight: parseFloat(newMappingWeight || '1') || 1,
      purposes: [purpose],
      provider: newMappingProvider || 'openai',
    }
    if (newMappingContext.trim()) {
      payload.context_length = Math.max(256, parseInt(newMappingContext, 10) || 0)
    }
    if (newMappingInstances.trim()) {
      payload.instances = Math.max(1, parseInt(newMappingInstances, 10) || 1)
    }
    try {
      const r = await fetch('/api/admin/llm-endpoints', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await r.json().catch(() => ({}))
      if (!r.ok || !data?.ok || !data.item) {
        throw new Error(data?.error || 'Не удалось создать маршрут задачи')
      }
      setS(prev => {
        if (!prev) return prev
        return { ...prev, llm_endpoints: [data.item as LlmEndpointInfo, ...(prev.llm_endpoints || [])] }
      })
      setNewMappingName('')
      toasts.push('Маршрут задачи добавлен', 'success')
    } catch (error: any) {
      toasts.push(String(error?.message || error || 'Ошибка добавления маршрута'), 'error')
    }
  }, [
    lmstudioProbeUrl,
    newMappingModel,
    newMappingPurpose,
    newMappingName,
    newMappingWeight,
    newMappingProvider,
    newMappingContext,
    newMappingInstances,
    toasts,
  ])

  const removeTaskMapping = useCallback(async (endpointId: number) => {
    if (!window.confirm('Удалить этот маршрут задачи?')) return
    try {
      const r = await fetch(`/api/admin/llm-endpoints/${endpointId}`, { method: 'DELETE' })
      const data = await r.json().catch(() => ({}))
      if (!r.ok || !data?.ok) {
        throw new Error(data?.error || 'Не удалось удалить маршрут задачи')
      }
      setS(prev => {
        if (!prev) return prev
        return { ...prev, llm_endpoints: (prev.llm_endpoints || []).filter(ep => ep.id !== endpointId) }
      })
      toasts.push('Маршрут удалён', 'success')
    } catch (error: any) {
      toasts.push(String(error?.message || error || 'Ошибка удаления маршрута'), 'error')
    }
  }, [toasts])

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
    const isChanged = (field: RuntimeField) => stableStringify(runtimeBaseline[field.name]) !== stableStringify(field.value)
    const changedRuntime = (s.runtime_fields || []).filter(isChanged)
    if (changedRuntime.length) {
      const preview = changedRuntime.slice(0, 12).map(field => {
        const before = runtimeBaseline[field.name]
        const after = field.value
        return `${field.name}: ${JSON.stringify(before)} -> ${JSON.stringify(after)}`
      }).join('\n')
      const restartHint = changedRuntime.some(field => field.restart_required)
        ? '\n\nЧасть изменений требует перезапуска сервиса.'
        : ''
      const riskHint = changedRuntime.some(field => field.risk_level === 'high')
        ? '\nЕсть high-risk параметры, проверьте значения.'
        : ''
      const approved = window.confirm(`Изменения runtime-полей (${changedRuntime.length}):\n${preview}${restartHint}${riskHint}\n\nСохранить?`)
      if (!approved) return
    }
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
      payload.ai_query_variants_max = Math.max(0, Math.min(3, parseInt(String(payload.ai_query_variants_max ?? 0), 10) || 0))
      payload.ai_rag_retry_enabled = !!payload.ai_rag_retry_enabled
      payload.ai_rag_retry_threshold = Math.max(0, Math.min(1, parseFloat(String(payload.ai_rag_retry_threshold ?? 0.6)) || 0.6))
      payload.rag_embedding_dim = Math.max(1, parseInt(String(payload.rag_embedding_dim || 0), 10) || 1)
      payload.rag_embedding_batch = Math.max(1, parseInt(String(payload.rag_embedding_batch || 0), 10) || 1)
      payload.rag_embedding_device = (payload.rag_embedding_device || '').trim()
      payload.rag_embedding_endpoint = (payload.rag_embedding_endpoint || '').trim()
      payload.rag_embedding_api_key = (payload.rag_embedding_api_key || '').trim()
      const chunkMaxTokensRaw = parseInt(String(payload.doc_chat_chunk_max_tokens ?? ''), 10)
      payload.doc_chat_chunk_max_tokens = Number.isFinite(chunkMaxTokensRaw) ? Math.max(16, chunkMaxTokensRaw) : 700
      const chunkOverlapRaw = parseInt(String(payload.doc_chat_chunk_overlap ?? ''), 10)
      payload.doc_chat_chunk_overlap = Number.isFinite(chunkOverlapRaw) ? Math.max(0, chunkOverlapRaw) : 120
      const chunkMinTokensRaw = parseInt(String(payload.doc_chat_chunk_min_tokens ?? ''), 10)
      payload.doc_chat_chunk_min_tokens = Number.isFinite(chunkMinTokensRaw) ? Math.max(1, chunkMinTokensRaw) : 80
      const docChatMaxChunksRaw = parseInt(String(payload.doc_chat_max_chunks ?? ''), 10)
      payload.doc_chat_max_chunks = Number.isFinite(docChatMaxChunksRaw) ? Math.max(0, docChatMaxChunksRaw) : 0
      const docChatFallbackRaw = parseInt(String(payload.doc_chat_fallback_chunks ?? ''), 10)
      payload.doc_chat_fallback_chunks = Number.isFinite(docChatFallbackRaw) ? Math.max(0, docChatFallbackRaw) : 0
      const imgWidthRaw = parseInt(String(payload.doc_chat_image_min_width ?? ''), 10)
      payload.doc_chat_image_min_width = Number.isFinite(imgWidthRaw) ? Math.max(0, imgWidthRaw) : 0
      const imgHeightRaw = parseInt(String(payload.doc_chat_image_min_height ?? ''), 10)
      payload.doc_chat_image_min_height = Number.isFinite(imgHeightRaw) ? Math.max(0, imgHeightRaw) : 0
      const preparedMaterialTypes = (payload.material_types || [])
        .map((item: MaterialTypeDefinition) => prepareMaterialTypeForSave(item))
        .filter((item): item is Record<string, any> => Boolean(item))
      payload.material_types = preparedMaterialTypes
      const runtimeFieldsPayload: Record<string, any> = {}
      for (const field of s.runtime_fields || []) {
        runtimeFieldsPayload[field.name] = field.value
      }
      payload.runtime_fields = runtimeFieldsPayload
      const r = await fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      if (r.ok) {
        setS(prev => prev ? { ...prev, material_types: preparedMaterialTypes.map(normalizeMaterialTypeDefinition) } : prev)
        const baseline: Record<string, any> = {}
        ;(s.runtime_fields || []).forEach((field) => { baseline[field.name] = field.value })
        setRuntimeBaseline(baseline)
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

  const backupDb = async () => {
    try {
      const payload = {
        db_type: dbType,
        format: dbType === 'postgresql' ? pgBackupFormat : 'db',
      }
      const r = await fetch('/admin/backup-db', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!r.ok) {
        const msg = await r.json().then(v => v?.error || '').catch(() => '')
        alert(msg || 'Ошибка резервного копирования')
        return
      }
      const blob = await r.blob()
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      const ext = dbType === 'postgresql' ? (pgBackupFormat === 'sql' ? 'sql' : 'dump') : 'db'
      a.download = `catalogue_backup.${ext}`
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
    fd.set('db_type', dbType)
    try {
      const r = await fetch('/admin/import-db', { method: 'POST', body: fd })
      if (r.ok) {
        alert('Импорт завершён. Перезапустите приложение.')
        return
      }
      const msg = await r.json().then(v => v?.error || '').catch(() => '')
      alert(msg || 'Ошибка импорта')
    } catch {
      alert('Ошибка импорта')
    }
  }

  const runMigrationWizard = async () => {
    const hasUrl = !!migrationPgUrl.trim()
    if (!hasUrl && (!migrationPgHost.trim() || !migrationPgDb.trim() || !migrationPgUser.trim())) {
      alert('Укажите PostgreSQL URL или поля адрес/база/логин')
      return
    }
    if (migrationMode === 'run' && !confirm('Запустить реальную миграцию SQLite -> PostgreSQL?')) return
    setMigrationBusy(true)
    try {
      const resp = await fetch('/admin/db/migrate-sqlite-to-postgres', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sqlite_path: migrationSqlitePath.trim() || 'catalogue.db',
          postgres_url: migrationPgUrl.trim(),
          postgres_host: migrationPgHost.trim(),
          postgres_port: parseInt(migrationPgPort || '5432', 10) || 5432,
          postgres_db: migrationPgDb.trim(),
          postgres_user: migrationPgUser.trim(),
          postgres_password: migrationPgPassword,
          mode: migrationMode,
        }),
      })
      const data = await resp.json().catch(() => ({}))
      const output = String(data?.output || '').trim()
      const result = data?.ok ? 'успешно' : 'с ошибкой'
      alert(`Миграция (${migrationMode}) завершена ${result}. Код: ${data?.exit_code ?? 'n/a'}\n\n${output || 'Лог пуст'}`)
    } catch {
      alert('Ошибка запуска wizard-миграции')
    } finally {
      setMigrationBusy(false)
    }
  }

  if (!isAdmin) return <div className="card p-3">Недостаточно прав.</div>
  if (error) return <div className="card p-3">{error}</div>
  if (!s) return <div className="card p-3">Загрузка настроек…</div>

  const llmEndpoints = s.llm_endpoints || []
  const probedModels = lmstudioModelsByBase[(lmstudioProbeUrl || '').trim()] || []
  const llmTaskMappings = llmEndpoints.flatMap(ep => {
    const purposes = ep.purposes && ep.purposes.length ? ep.purposes : ['default']
    return purposes.map(purpose => ({
      endpointId: ep.id,
      purpose,
      purposeLabel: llmPurposeLabels.get(purpose) || purpose,
      model: ep.model,
      provider: ep.provider || 'openai',
      baseUrl: ep.base_url,
      weight: ep.weight,
      instances: ep.instances || 1,
      contextLength: ep.context_length || null,
      name: ep.name,
    }))
  })
  const aiwordUsers = s.aiword_users || []
  const materialTypes = s.material_types || []
  const runtimeFields = s.runtime_fields || []
  const runtimeFieldChanged = (field: RuntimeField) => stableStringify(runtimeBaseline[field.name]) !== stableStringify(field.value)
  const runtimeChangedCount = runtimeFields.filter(runtimeFieldChanged).length
  const runtimePendingRestartCount = runtimeFields.filter(field => runtimeFieldChanged(field) && field.restart_required).length
  const runtimeHighRiskCount = runtimeFields.filter(field => runtimeFieldChanged(field) && field.risk_level === 'high').length
  const runtimeSearchNormalized = runtimeSearch.trim().toLowerCase()
  const visibleRuntimeFields = runtimeFields.filter(field => field.visibility !== 'env_only')
  const runtimeFieldsForActiveTab = visibleRuntimeFields.filter(field => {
    const belongsSafe = SAFE_GROUPS.has(field.group || 'general')
    const belongsExpert = EXPERT_GROUPS.has(field.group || 'general') || !belongsSafe
    if (s.settings_hub_v2_enabled !== false) {
      if (activeTab === 'safe' && !belongsSafe) return false
      if (activeTab === 'expert' && !belongsExpert) return false
    }
    return true
  }).filter(field => {
    if (runtimeChangedOnly && !runtimeFieldChanged(field)) return false
    if (!runtimeSearchNormalized) return true
    const hay = `${field.name} ${getRuntimeFieldLabel(field)} ${field.api_key || ''} ${getRuntimeFieldDescription(field)} ${field.group || ''}`.toLowerCase()
    return hay.includes(runtimeSearchNormalized)
  })
  const llmRuntimeFields = visibleRuntimeFields.filter(field => ['llm', 'llm_pool', 'cache'].includes(field.group))
    .filter(field => !['lmstudio_api_base', 'lmstudio_model', 'lmstudio_api_key', 'lm_default_provider'].includes(field.name))
  const runtimeFieldsByGroup = runtimeFieldsForActiveTab.reduce<Record<string, RuntimeField[]>>((acc, field) => {
    const group = field.group || 'general'
    if (!acc[group]) acc[group] = []
    acc[group].push(field)
    return acc
  }, {})

  const updateRuntimeField = (name: string, raw: string | boolean) => {
    setS(prev => {
      if (!prev) return prev
      const nextFields = (prev.runtime_fields || []).map(field => {
        if (field.name !== name) return field
        return { ...field, value: normalizeRuntimeValue(field, raw) }
      })
      return { ...prev, runtime_fields: nextFields }
    })
  }
  const hubV2Enabled = s.settings_hub_v2_enabled !== false
  const safeVisible = !hubV2Enabled || activeTab === 'safe'
  const expertVisible = !hubV2Enabled || activeTab === 'expert'
  const llmVisible = !hubV2Enabled || activeTab === 'llm'
  const facetsVisible = !hubV2Enabled || activeTab === 'facets'

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
        <div className="d-flex flex-wrap gap-2 align-items-center">
          {hubV2Enabled && <button className={`btn btn-sm ${activeTab === 'safe' ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setActiveTab('safe')}>Safe</button>}
          {hubV2Enabled && <button className={`btn btn-sm ${activeTab === 'expert' ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setActiveTab('expert')}>Expert</button>}
          {hubV2Enabled && <button className={`btn btn-sm ${activeTab === 'llm' ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setActiveTab('llm')}>LLM Endpoints</button>}
          {hubV2Enabled && <button className={`btn btn-sm ${activeTab === 'facets' ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setActiveTab('facets')}>Фасеты</button>}
          <div className="ms-auto d-flex flex-wrap gap-2">
            <span className="badge text-bg-secondary">изменено: {runtimeChangedCount}</span>
            {runtimePendingRestartCount > 0 && <span className="badge text-bg-warning">нужен рестарт: {runtimePendingRestartCount}</span>}
            {runtimeHighRiskCount > 0 && <span className="badge text-bg-danger">high-risk: {runtimeHighRiskCount}</span>}
          </div>
        </div>
      </div>

      {safeVisible && (
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
      )}

      {safeVisible && (
      <>
      <div className="card p-3">
        <div className="fw-semibold mb-2">Чат по документу</div>
        <div className="text-muted mb-3" style={{ fontSize: 13 }}>
          Управляйте разбиением текста, ограничениями по чанкам и минимальными размерами изображений для режима «Чат по документу». Fallback чанки добавляются как запас, если режимы не вернули достаточное число фрагментов.
        </div>
        <div className="row g-3">
          <div className="col-md-4">
            <label className="form-label">Максимум токенов на чанк</label>
            <input
              className="form-control"
              type="number"
              min={16}
              value={s.doc_chat_chunk_max_tokens}
              onChange={e => {
                const next = Math.max(16, parseInt(e.target.value || '0', 10) || 0)
                setS({ ...s, doc_chat_chunk_max_tokens: next })
              }}
            />
            <div className="form-text">Граница токенов, после которой начинается новый чанк.</div>
          </div>
          <div className="col-md-4">
            <label className="form-label">Перекрытие токенов</label>
            <input
              className="form-control"
              type="number"
              min={0}
              value={s.doc_chat_chunk_overlap}
              onChange={e => {
                const raw = parseInt(e.target.value || '0', 10)
                const next = Number.isFinite(raw) ? Math.max(0, raw) : 0
                setS({ ...s, doc_chat_chunk_overlap: next })
              }}
            />
            <div className="form-text">Сколько токенов пересекаются между соседними чанками.</div>
          </div>
          <div className="col-md-4">
            <label className="form-label">Минимум токенов в чанке</label>
            <input
              className="form-control"
              type="number"
              min={1}
              value={s.doc_chat_chunk_min_tokens}
              onChange={e => {
                const raw = parseInt(e.target.value || '0', 10)
                const next = Number.isFinite(raw) ? Math.max(1, raw) : 1
                setS({ ...s, doc_chat_chunk_min_tokens: next })
              }}
            />
            <div className="form-text">Не давайте чанкам быть слишком короткими, иначе они будут плохо ранжироваться.</div>
          </div>
          <div className="col-md-4">
            <label className="form-label">Максимум чанков на ответ</label>
            <input
              className="form-control"
              type="number"
              min={0}
              value={s.doc_chat_max_chunks}
              onChange={e => {
                const raw = parseInt(e.target.value || '0', 10)
                const next = Number.isFinite(raw) ? Math.max(0, raw) : 0
                setS({ ...s, doc_chat_max_chunks: next })
              }}
            />
            <div className="form-text">0 — использовать стандартные режимы, иначе ограничить контекст до указанного числа.</div>
          </div>
          <div className="col-md-4">
            <label className="form-label">Fallback чанков</label>
            <input
              className="form-control"
              type="number"
              min={0}
              value={s.doc_chat_fallback_chunks}
              onChange={e => {
                const raw = parseInt(e.target.value || '0', 10)
                const next = Number.isFinite(raw) ? Math.max(0, raw) : 0
                setS({ ...s, doc_chat_fallback_chunks: next })
              }}
            />
            <div className="form-text">
              Запасное количество чанков, которое добавляется, если основной режим не дал достаточный контекст.
            </div>
          </div>
          <div className="col-md-4">
            <label className="form-label">Минимальная ширина изображения</label>
            <input
              className="form-control"
              type="number"
              min={0}
              value={s.doc_chat_image_min_width}
              onChange={e => {
                const raw = parseInt(e.target.value || '0', 10)
                const next = Number.isFinite(raw) ? Math.max(0, raw) : 0
                setS({ ...s, doc_chat_image_min_width: next })
              }}
            />
            <div className="form-text">32 — фильтр по умолчанию, ниже отсекаются артефакты.</div>
          </div>
          <div className="col-md-4">
            <label className="form-label">Минимальная высота изображения</label>
            <input
              className="form-control"
              type="number"
              min={0}
              value={s.doc_chat_image_min_height}
              onChange={e => {
                const raw = parseInt(e.target.value || '0', 10)
                const next = Number.isFinite(raw) ? Math.max(0, raw) : 0
                setS({ ...s, doc_chat_image_min_height: next })
              }}
            />
            <div className="form-text">32 — фильтр по умолчанию; исключаются поля меньше указанной высоты.</div>
          </div>
        </div>
      </div>

      <div className="card p-3">
        <div className="fw-semibold mb-2">RAG и типизация</div>
        <div className="row g-3">
          <div className="col-12">
            <div className="alert alert-light border mb-0">
              Настройки токенизатора перенесены во вкладку <strong>LLM Endpoints</strong> и сохраняются вместе с LLM-конфигурацией.
            </div>
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
          <div className="col-md-3">
            <label className="form-label">Переформулировки запроса (0–3)</label>
            <input
              className="form-control"
              type="number"
              min={0}
              max={3}
              value={s.ai_query_variants_max ?? 0}
              onChange={e => {
                const next = Math.max(0, Math.min(3, parseInt(e.target.value || '0', 10) || 0))
                setS({ ...s, ai_query_variants_max: next })
              }}
            />
            <div className="form-text">Сколько вариантов запроса генерировать для RAG. 0 — без перефразировок.</div>
          </div>
          <div className="col-md-3 d-flex align-items-center">
            <div className="form-check form-switch">
              <input className="form-check-input" type="checkbox" id="ragRetry" checked={!!s.ai_rag_retry_enabled} onChange={e => setS({ ...s, ai_rag_retry_enabled: e.target.checked })} />
              <label className="form-check-label" htmlFor="ragRetry">Автоповтор RAG при высоком риске</label>
            </div>
          </div>
          <div className="col-md-3">
            <label className="form-label">Порог риска (0–1)</label>
            <input
              className="form-control"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={typeof s.ai_rag_retry_threshold === 'number' ? s.ai_rag_retry_threshold : 0.6}
              onChange={e => {
                const raw = parseFloat(e.target.value || '0')
                const next = Number.isFinite(raw) ? Math.max(0, Math.min(1, raw)) : 0.6
                setS({ ...s, ai_rag_retry_threshold: next })
              }}
            />
            <div className="form-text">Если риск ≥ порога, система попробует переформулировать запрос и расширить контекст.</div>
          </div>
      </div>
    </div>
    </>
    )}

      {expertVisible && (
      <div className="card p-3">
        <details>
          <summary className="fw-semibold">Типы документов и эвристики</summary>
          <div className="text-muted mt-2 mb-3" style={{ fontSize: 13 }}>Настройте ключевые слова, расширения и специальные правила для автоматического определения типа документов.</div>
          <div className="mt-2">
            <MaterialTypesEditor
              materialTypes={materialTypes}
              onChange={updateMaterialType}
              onRemove={removeMaterialType}
              onAdd={addMaterialType}
              duplicateKeys={duplicateMaterialKeys}
            />
          </div>
        </details>
      </div>
      )}

      {safeVisible && (
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
      )}

      {llmVisible && (
      <div className="card p-3">
        <div className="d-flex flex-wrap justify-content-between align-items-start gap-2 mb-3">
          <div>
            <div className="fw-semibold">Назначение LLM эндпоинтов</div>
            <div className="text-muted" style={{ fontSize: 13 }}>Выберите задачи для каждого эндпоинта. Один тип может обслуживаться несколькими LLM — запросы будут распределяться по очереди.</div>
          </div>
          <span className="badge text-bg-info">Единая точка управления LLM в этом разделе</span>
        </div>
        <div className="border rounded-3 p-3 mb-3">
          <div className="fw-semibold mb-2">Проверка моделей LM Studio по IP/URL</div>
          <div className="row g-2 align-items-end">
            <div className="col-md-9">
              <label className="form-label">Base URL LM Studio</label>
              <input
                className="form-control"
                placeholder="http://192.168.1.12:1234/v1"
                value={lmstudioProbeUrl}
                onChange={e => setLmstudioProbeUrl(e.target.value)}
              />
            </div>
            <div className="col-md-3 d-grid">
              <button className="btn btn-outline-primary" onClick={() => probeLmstudioModels()} disabled={lmstudioLoading}>
                {lmstudioLoading ? 'Проверка…' : 'Проверить модели'}
              </button>
            </div>
          </div>
          <div className="mt-3 border rounded-3 p-3">
            <div className="fw-semibold mb-2">Настройки токенизатора (RAG)</div>
            <div className="row g-2">
              <div className="col-md-3">
                <label className="form-label" htmlFor="llm-tab-rag-backend">Backend</label>
                <select
                  id="llm-tab-rag-backend"
                  className="form-select"
                  value={s.rag_embedding_backend}
                  onChange={e => setS({ ...s, rag_embedding_backend: e.target.value })}
                >
                  {ragEmbeddingBackendOptions.map(opt => (
                    <option key={opt.id} value={opt.id}>{opt.label}</option>
                  ))}
                </select>
              </div>
              <div className="col-md-3">
                <label className="form-label">Модель токенизатора</label>
                <input
                  className="form-control"
                  placeholder="nomic-ai/nomic-embed-text-v1.5-GGUF"
                  value={s.rag_embedding_model}
                  onChange={e => setS({ ...s, rag_embedding_model: e.target.value })}
                />
              </div>
              <div className="col-md-3">
                <label className="form-label">Endpoint токенизатора</label>
                <input
                  className="form-control"
                  placeholder="http://localhost:1234/v1"
                  value={s.rag_embedding_endpoint}
                  onChange={e => setS({ ...s, rag_embedding_endpoint: e.target.value })}
                />
              </div>
              <div className="col-md-3">
                <label className="form-label">API key токенизатора</label>
                <input
                  className="form-control"
                  placeholder="sk-..."
                  value={s.rag_embedding_api_key}
                  onChange={e => setS({ ...s, rag_embedding_api_key: e.target.value })}
                />
              </div>
              <div className="col-md-2">
                <label className="form-label">Размерность</label>
                <input
                  className="form-control"
                  type="number"
                  min={1}
                  value={s.rag_embedding_dim}
                  onChange={e => setS({ ...s, rag_embedding_dim: parseInt(e.target.value || '0', 10) || 0 })}
                />
              </div>
              <div className="col-md-2">
                <label className="form-label">Batch</label>
                <input
                  className="form-control"
                  type="number"
                  min={1}
                  value={s.rag_embedding_batch}
                  onChange={e => setS({ ...s, rag_embedding_batch: parseInt(e.target.value || '1', 10) || 1 })}
                />
              </div>
              <div className="col-md-4">
                <label className="form-label">Устройство</label>
                <input
                  className="form-control"
                  placeholder="cpu / cuda:0"
                  value={s.rag_embedding_device ?? ''}
                  onChange={e => setS({ ...s, rag_embedding_device: e.target.value })}
                />
              </div>
            </div>
          </div>
          {probedModels.length > 0 && (
            <div className="mt-3">
              <div className="fw-semibold mb-2">Найденные модели</div>
              <div className="d-flex flex-wrap gap-2">
                {probedModels.map(item => (
                  <button
                    key={item.id}
                    type="button"
                    className={`btn btn-sm ${newMappingModel === item.id ? 'btn-primary' : 'btn-outline-secondary'}`}
                    onClick={() => setNewMappingModel(item.id)}
                  >
                    {item.name}{item.loaded ? ' (loaded)' : ''}
                  </button>
                ))}
              </div>
            </div>
          )}
          <div className="mt-3 border rounded-3 p-3">
            <div className="fw-semibold mb-2">Добавить маршрут задачи (задача → модель)</div>
            <div className="row g-2">
              <div className="col-md-3">
                <label className="form-label">Тип задачи</label>
                <select className="form-select" value={newMappingPurpose} onChange={e => setNewMappingPurpose(e.target.value)}>
                  {llmPurposes.map(option => (
                    <option key={option.id} value={option.id}>{option.label}</option>
                  ))}
                </select>
              </div>
              <div className="col-md-3">
                <label className="form-label">Модель</label>
                <select className="form-select" value={newMappingModel} onChange={e => setNewMappingModel(e.target.value)}>
                  <option value="">-- выберите модель --</option>
                  {probedModels.map(item => (
                    <option key={item.id} value={item.id}>{item.name}</option>
                  ))}
                </select>
              </div>
              <div className="col-md-2">
                <label className="form-label">Провайдер</label>
                <select className="form-select" value={newMappingProvider} onChange={e => setNewMappingProvider(e.target.value)}>
                  {llmProviderOptions.map(opt => (
                    <option key={opt.id} value={opt.id}>{opt.label}</option>
                  ))}
                </select>
              </div>
              <div className="col-md-2">
                <label className="form-label">Вес</label>
                <input className="form-control" value={newMappingWeight} onChange={e => setNewMappingWeight(e.target.value)} />
              </div>
              <div className="col-md-2">
                <label className="form-label">Копии</label>
                <input className="form-control" type="number" min={1} value={newMappingInstances} onChange={e => setNewMappingInstances(e.target.value)} />
              </div>
              <div className="col-md-3">
                <label className="form-label">Контекст</label>
                <input className="form-control" type="number" min={256} value={newMappingContext} onChange={e => setNewMappingContext(e.target.value)} />
              </div>
              <div className="col-md-7">
                <label className="form-label">Название маршрута</label>
                <input className="form-control" placeholder="Опционально" value={newMappingName} onChange={e => setNewMappingName(e.target.value)} />
              </div>
              <div className="col-md-2 d-grid align-items-end">
                <button className="btn btn-success" onClick={createTaskMapping}>Добавить</button>
              </div>
            </div>
          </div>
        </div>
        <div className="border rounded-3 p-3 mb-3">
          <div className="fw-semibold mb-2">Текущие маршруты задач</div>
          {llmTaskMappings.length === 0 ? (
            <div className="text-muted">Маршрутов пока нет. Добавьте первый маршрут через форму выше.</div>
          ) : (
            <div className="table-responsive">
              <table className="table table-sm align-middle">
                <thead>
                  <tr>
                    <th>Задача</th>
                    <th>Модель</th>
                    <th>Провайдер</th>
                    <th>Контекст</th>
                    <th>Копии</th>
                    <th>Вес</th>
                    <th>Endpoint</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {llmTaskMappings.map(item => (
                    <tr key={`${item.endpointId}:${item.purpose}`}>
                      <td>{item.purposeLabel}</td>
                      <td>{item.model}</td>
                      <td>{item.provider}</td>
                      <td>{item.contextLength || '—'}</td>
                      <td>{item.instances}</td>
                      <td>{item.weight}</td>
                      <td>{item.name}</td>
                      <td>
                        <button className="btn btn-sm btn-outline-danger" onClick={() => removeTaskMapping(item.endpointId)}>Удалить</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
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
                <div className="row g-2 mt-2">
                  <div className="col-md-6">
                    <label className="form-label">Модель endpoint</label>
                    <select
                      className="form-select"
                      value={ep.model}
                      onChange={e => {
                        const value = e.target.value
                        patchLlmEndpoint(ep.id, { model: value }, 'Модель endpoint обновлена').then((updated) => {
                          if (!updated) return
                          setS(prev => {
                            if (!prev) return prev
                            return {
                              ...prev,
                              llm_endpoints: (prev.llm_endpoints || []).map(item => item.id === ep.id ? { ...item, ...updated } : item),
                            }
                          })
                        })
                      }}
                    >
                      {(lmstudioModelsByBase[ep.base_url] || []).length === 0 && <option value={ep.model}>{ep.model}</option>}
                      {(lmstudioModelsByBase[ep.base_url] || []).map(item => (
                        <option key={item.id} value={item.id}>
                          {item.name}{item.loaded ? ' (loaded)' : ''}
                        </option>
                      ))}
                    </select>
                    <div className="form-text">Список берётся из LM Studio v1 `/api/v1/models`.</div>
                  </div>
                  <div className="col-md-3">
                    <label className="form-label">Контекст при загрузке</label>
                    <input
                      className="form-control"
                      type="number"
                      min={256}
                      value={llmContextDrafts[ep.id] ?? ''}
                      onChange={e => setLlmContextDrafts(prev => ({ ...prev, [ep.id]: e.target.value }))}
                      onBlur={() => {
                        const raw = llmContextDrafts[ep.id]
                        const parsed = raw ? parseInt(raw, 10) : NaN
                        const value = Number.isFinite(parsed) && parsed > 0 ? Math.max(256, parsed) : null
                        patchLlmEndpoint(ep.id, { context_length: value }, 'Контекст endpoint обновлён')
                      }}
                    />
                  </div>
                  <div className="col-md-3">
                    <label className="form-label">Копий модели</label>
                    <input
                      className="form-control"
                      type="number"
                      min={1}
                      value={llmInstancesDrafts[ep.id] ?? '1'}
                      onChange={e => setLlmInstancesDrafts(prev => ({ ...prev, [ep.id]: e.target.value }))}
                      onBlur={() => {
                        const parsed = Math.max(1, parseInt(llmInstancesDrafts[ep.id] || '1', 10) || 1)
                        patchLlmEndpoint(ep.id, { instances: parsed }, 'Число копий endpoint обновлено')
                      }}
                    />
                  </div>
                </div>
                <div className="d-flex flex-wrap gap-2 mt-2">
                  <button className="btn btn-sm btn-outline-secondary" onClick={() => probeLmstudioModels(ep.base_url, ep.id)}>
                    Обновить список моделей по этому URL
                  </button>
                  <button className="btn btn-sm btn-outline-success" onClick={() => loadLmstudioModel(ep)}>
                    Загрузить модель в LM Studio
                  </button>
                  <button className="btn btn-sm btn-outline-warning" onClick={() => unloadLmstudioModel(ep)}>
                    Выгрузить модель
                  </button>
                </div>
              </div>
            )
          })}
        </div>
        <hr className="my-3" />
        <div className="fw-semibold mb-2">LLM Pool и тонкая настройка</div>
        <div className="row g-3">
          {llmRuntimeFields.map(field => {
            const fieldType = (field.type || '').toLowerCase()
            const valueText = String(field.value ?? '')
            return (
              <div key={`llm-runtime-${field.name}`} className="col-md-6">
                <label className="form-label d-flex flex-wrap gap-2 align-items-center">
                  <span>{getRuntimeFieldLabel(field)}</span>
                  {field.restart_required ? <span className="badge text-bg-warning">restart</span> : null}
                </label>
                {fieldType.includes('bool') ? (
                  <div className="form-check form-switch">
                    <input className="form-check-input" type="checkbox" checked={!!field.value} onChange={e => updateRuntimeField(field.name, e.target.checked)} />
                  </div>
                ) : (
                  <input
                    className="form-control"
                    type={fieldType.includes('int') || fieldType.includes('float') ? 'number' : 'text'}
                    value={valueText}
                    onChange={e => updateRuntimeField(field.name, e.target.value)}
                  />
                )}
                <div className="form-text">{getRuntimeFieldDescription(field)}</div>
              </div>
            )
          })}
        </div>
      </div>
      )}

      {expertVisible && (
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
      )}

      {expertVisible && (
      <div className="card p-3">
        <div className="fw-semibold mb-2">Управление базой данных</div>
        <div className="row g-2 mb-3">
          <div className="col-md-4">
            <label className="form-label">Тип БД</label>
            <select
              className="form-select"
              value={dbType}
              onChange={e => setDbType((e.target.value === 'postgresql' ? 'postgresql' : 'sqlite'))}
            >
              <option value="sqlite">SQLite (legacy .db)</option>
              <option value="postgresql">PostgreSQL</option>
            </select>
          </div>
          {dbType === 'postgresql' && (
            <div className="col-md-4">
              <label className="form-label">Формат backup</label>
              <select
                className="form-select"
                value={pgBackupFormat}
                onChange={e => setPgBackupFormat((e.target.value === 'sql' ? 'sql' : 'dump'))}
              >
                <option value="dump">Custom dump (.dump)</option>
                <option value="sql">Plain SQL (.sql)</option>
              </select>
            </div>
          )}
          <div className="col-12">
            <div className="form-text">
              Активное подключение: <code>{s.database_dialect || 'unknown'}</code>
              {s.database_uri ? <> · <code>{s.database_uri}</code></> : null}
            </div>
          </div>
        </div>
        <div className="d-flex flex-wrap gap-2 align-items-center">
          <button className="btn btn-outline-primary" onClick={backupDb}>Резервная копия</button>
          <form onSubmit={importDb} className="d-flex gap-2 align-items-center">
            <input className="form-control" type="file" name="dbfile" accept={dbType === 'postgresql' ? '.dump,.sql,.backup' : '.db'} required />
            <button className="btn btn-outline-secondary" type="submit">
              {dbType === 'postgresql' ? 'Импорт PostgreSQL' : 'Импорт .db'}
            </button>
          </form>
          <button className="btn btn-outline-danger" onClick={clearDb}>Очистить базу</button>
        </div>
      </div>
      )}

      {expertVisible && (
      <div className="card p-3">
        <div className="fw-semibold mb-2">Миграция SQLite → PostgreSQL (Wizard)</div>
        <div className="row g-2">
          <div className="col-md-4">
            <label className="form-label">SQLite файл</label>
            <input
              className="form-control"
              value={migrationSqlitePath}
              onChange={e => setMigrationSqlitePath(e.target.value)}
              placeholder="catalogue.db"
            />
          </div>
          <div className="col-md-8">
            <label className="form-label">PostgreSQL URL (опционально)</label>
            <input
              className="form-control"
              value={migrationPgUrl}
              onChange={e => setMigrationPgUrl(e.target.value)}
              placeholder="postgresql://agregator:agregator@localhost:5432/agregator"
            />
          </div>
          <div className="col-md-4">
            <label className="form-label">Хост / адрес БД</label>
            <input className="form-control" value={migrationPgHost} onChange={e => setMigrationPgHost(e.target.value)} placeholder="localhost" />
            <div className="d-flex gap-2 flex-wrap mt-2">
              <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => setMigrationPgHost('localhost')}>
                localhost
              </button>
              <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => setMigrationPgHost('host.docker.internal')}>
                host.docker.internal
              </button>
              <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => setMigrationPgHost('postgres')}>
                postgres
              </button>
            </div>
          </div>
          <div className="col-md-2">
            <label className="form-label">Порт</label>
            <input className="form-control" value={migrationPgPort} onChange={e => setMigrationPgPort(e.target.value)} placeholder="5432" />
          </div>
          <div className="col-md-3">
            <label className="form-label">Имя БД</label>
            <input className="form-control" value={migrationPgDb} onChange={e => setMigrationPgDb(e.target.value)} placeholder="agregator" />
          </div>
          <div className="col-md-3">
            <label className="form-label">Логин</label>
            <input className="form-control" value={migrationPgUser} onChange={e => setMigrationPgUser(e.target.value)} placeholder="agregator" />
          </div>
          <div className="col-md-4">
            <label className="form-label">Пароль</label>
            <input className="form-control" type="password" value={migrationPgPassword} onChange={e => setMigrationPgPassword(e.target.value)} placeholder="••••••••" />
          </div>
          <div className="col-md-2">
            <label className="form-label">Режим</label>
            <select
              className="form-select"
              value={migrationMode}
              onChange={e => setMigrationMode(e.target.value === 'run' ? 'run' : 'dry-run')}
            >
              <option value="dry-run">dry-run</option>
              <option value="run">run</option>
            </select>
          </div>
        </div>
        <div className="form-text mt-2">
          Wizard запускает `scripts/migrate_sqlite_to_postgres.sh`: backup, проверка подключения, перенос через pgloader (если есть), затем `alembic upgrade head`.
        </div>
        <div className="mt-3">
          <button className="btn btn-outline-primary" onClick={runMigrationWizard} disabled={migrationBusy}>
            {migrationBusy ? 'Выполнение…' : 'Запустить миграцию'}
          </button>
        </div>
      </div>
      )}

      {expertVisible && (
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
      )}

      {expertVisible && (
      <div className="card p-3">
        <details>
          <summary className="fw-semibold">Расширенные runtime-настройки (все параметры)</summary>
          <div className="row g-2 mt-2">
            <div className="col-md-8">
              <input
                className="form-control"
                placeholder="Поиск по имени/описанию/env ключу"
                value={runtimeSearch}
                onChange={e => setRuntimeSearch(e.target.value)}
              />
            </div>
            <div className="col-md-4 d-flex align-items-center">
              <div className="form-check form-switch">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="runtimeChangedOnly"
                  checked={runtimeChangedOnly}
                  onChange={e => setRuntimeChangedOnly(e.target.checked)}
                />
                <label className="form-check-label" htmlFor="runtimeChangedOnly">Только изменённые поля</label>
              </div>
            </div>
          </div>
          <div className="text-muted mt-2 mb-3" style={{ fontSize: 13 }}>
            Здесь отображаются все runtime-параметры сервера с описаниями. Поля синхронизируются с backend и сохраняются через общую кнопку «Сохранить».
          </div>
          <div className="d-grid gap-3">
            {Object.keys(runtimeFieldsByGroup).length === 0 && (
              <div className="text-muted">Нет полей по текущему фильтру.</div>
            )}
            {Object.entries(runtimeFieldsByGroup).map(([group, fields]) => (
              <div key={group} className="border rounded-3 p-3" style={{ borderColor: 'var(--border)' }}>
                <div className="fw-semibold mb-2">{runtimeGroupLabels[group] || group}</div>
                <div className="row g-3">
                  {fields.map(field => {
                    const fieldType = (field.type || '').toLowerCase()
                    const isBool = fieldType.includes('bool')
                    const isNumber = fieldType.includes('int') || fieldType.includes('float')
                    const isStructured = fieldType.includes('list') || fieldType.includes('dict') || fieldType.includes('optional')
                    const valueText = isStructured
                      ? (() => {
                          try {
                            return field.value === null || field.value === undefined ? '' : JSON.stringify(field.value, null, 2)
                          } catch {
                            return String(field.value ?? '')
                          }
                        })()
                      : String(field.value ?? '')
                    return (
                      <div key={field.name} className="col-md-6">
                        <label className="form-label d-flex flex-wrap align-items-center gap-2">
                          <span>{getRuntimeFieldLabel(field)}</span>
                          {field.env_key ? <code>{field.env_key}</code> : null}
                          {runtimeFieldChanged(field) ? <span className="badge text-bg-secondary">changed</span> : null}
                          {field.restart_required ? <span className="badge text-bg-warning">restart</span> : null}
                          {field.risk_level === 'high' ? <span className="badge text-bg-danger">high-risk</span> : null}
                        </label>
                        {isBool ? (
                          <div className="form-check form-switch">
                            <input
                              className="form-check-input"
                              type="checkbox"
                              checked={!!field.value}
                              disabled={!field.runtime_mutable}
                              onChange={e => updateRuntimeField(field.name, e.target.checked)}
                            />
                          </div>
                        ) : isNumber ? (
                          <input
                            className="form-control"
                            type="number"
                            step={fieldType.includes('float') ? '0.01' : '1'}
                            value={valueText}
                            disabled={!field.runtime_mutable}
                            onChange={e => updateRuntimeField(field.name, e.target.value)}
                          />
                        ) : isStructured ? (
                          <textarea
                            className="form-control"
                            rows={4}
                            value={valueText}
                            disabled={!field.runtime_mutable}
                            onChange={e => updateRuntimeField(field.name, e.target.value)}
                          />
                        ) : (
                          <input
                            className="form-control"
                            value={valueText}
                            disabled={!field.runtime_mutable}
                            onChange={e => updateRuntimeField(field.name, e.target.value)}
                          />
                        )}
                        <div className="form-text">{getRuntimeFieldDescription(field)}{field.api_key ? ` (API key: ${field.api_key})` : ''}</div>
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
        </details>
      </div>
      )}

      {facetsVisible && (
        <AdminFacetSettingsPage />
      )}
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
