import React, { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { DataSet, Network } from 'vis-network/standalone'
import { useToasts } from '../ui/Toasts'
import { renderMarkdown } from '../utils/markdown'
import OsintHistoryPanel from './osint/OsintHistoryPanel'

interface OntologyNode {
  id: string
  label: string
  type?: string | null
  summary?: string | null
  aliases?: string[]
  sources?: string[]
  score?: number | null
}

interface OntologyEdge {
  id: string
  from: string
  to: string
  label?: string | null
  description?: string | null
  sources?: string[]
  confidence?: number | null
}

interface OntologyPayload {
  nodes: OntologyNode[]
  edges: OntologyEdge[]
  meta?: Record<string, any> | null
}

interface OsintResultItem {
  id?: number
  rank: number
  title: string
  url: string
  snippet?: string | null
  highlight?: string | null
  score?: number | null
  metadata?: Record<string, any>
}

interface OsintSourceSnapshot {
  id?: number
  source: string
  engine?: string | null
  status: string
  blocked?: boolean
  from_cache?: boolean
  error?: string | null
  metadata?: Record<string, any>
  params?: Record<string, any>
  requested_url?: string | null
  final_url?: string | null
  html_snapshot?: string | null
  text_content?: string | null
  screenshot_path?: string | null
  llm_payload?: string | null
  llm_model?: string | null
  llm_error?: string | null
  created_at?: string | null
  started_at?: string | null
  completed_at?: string | null
  results: OsintResultItem[]
}

interface CombinedResultSourceRef {
  id: string
  label: string
  engine?: string | null
  rank?: number | null
  score?: number | null
}

interface CombinedResultItem {
  id: string
  title: string
  snippet?: string | null
  url?: string | null
  highlight?: string | null
  metadata?: Record<string, any> | null
  sources: CombinedResultSourceRef[]
}

interface OsintScheduleSnapshot {
  id: number
  label?: string | null
  active: boolean
  running?: boolean
  interval_minutes?: number | null
  next_run_at?: string | null
  last_run_at?: string | null
  notify?: boolean
  notify_channel?: string | null
  last_error?: string | null
}

interface OsintProgressEntry {
  status: string
  label?: string
  error?: string
  updated_at?: string
  created_at?: string
  fallback?: boolean
  fallback_url?: string
  final_url?: string | null
  pages_processed?: number
  pages_estimated?: number
  results_parsed?: number
  results_on_page?: number
  results_forwarded?: number
  requested_results?: number
  links_collected?: number
  keywords?: string[] | string | null
  refined_query?: string | null
  original_query?: string | null
  retry_available?: boolean
}

interface OsintJobSnapshot {
  id: number
  query: string
  locale?: string | null
  region?: string | null
  safe?: boolean
  status: string
  error?: string | null
  progress: Record<string, OsintProgressEntry>
  sources_total: number
  sources_completed: number
  created_at?: string | null
  started_at?: string | null
  completed_at?: string | null
  sources: OsintSourceSnapshot[]
  source_specs?: any[]
  params?: Record<string, any>
  analysis?: string | null
  analysis_error?: string | null
  analysis_html?: string | null
  combined_results?: CombinedResultItem[]
  ontology?: OntologyPayload | null
  ontology_error?: string | null
  schedule?: OsintScheduleSnapshot | null
  user_id?: number | null
}

interface ApiJobResponse {
  ok: boolean
  job?: OsintJobSnapshot
  error?: string
  details?: string
}

interface ApiJobsResponse {
  ok: boolean
  items?: OsintJobSnapshot[]
  error?: string
}

const DEFAULT_REMOTE_SOURCES = ['google']
const LOCALE_OPTIONS = ['ru-RU', 'en-US'] as const

const isAbsolutePath = (value: string) => {
  const trimmed = value.trim()
  if (trimmed.startsWith('/') || trimmed.startsWith('~/')) {
    return true
  }
  return /^[A-Za-z]:[\\/]/.test(trimmed)
}

const buildArtifactUrl = (relativePath: string) => {
  if (!relativePath) return ''
  const segments = relativePath.split('/').map(segment => encodeURIComponent(segment))
  return `/api/osint/artifacts/${segments.join('/')}`
}

const formatMetadataValue = (value: unknown): string => {
  if (value === null || value === undefined) return ''
  if (Array.isArray(value)) {
    return value.map(item => formatMetadataValue(item)).filter(Boolean).join(', ')
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
    return entries
      .map(([key, nested]) => {
        const formatted = formatMetadataValue(nested)
        return formatted ? `${key}: ${formatted}` : key
      })
      .join('; ')
  }
  return String(value)
}

const formatTimestamp = (value?: string | null) => {
  if (!value) return ''
  try {
    const date = new Date(value)
    if (Number.isNaN(date.getTime())) return value
    return new Intl.DateTimeFormat('ru-RU', {
      year: 'numeric',
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    }).format(date)
  } catch {
    return value
  }
}

const formatInterval = (minutes?: number | null) => {
  if (!minutes || minutes <= 0) return 'не задан'
  if (minutes % 1440 === 0) {
    const days = minutes / 1440
    return `${days} дн.`
  }
  if (minutes % 60 === 0) {
    const hours = minutes / 60
    return `${hours} ч.`
  }
  return `${minutes} мин.`
}

const normalizeNumber = (value: unknown): number | null => {
  if (value === null || value === undefined) return null
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null
  }
  if (typeof value === 'bigint') {
    return Number(value)
  }
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return null
    const parsed = Number(trimmed)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

const normalizeKeywordsList = (value: unknown): string[] => {
  if (!value) return []
  const seen = new Set<string>()
  const tokens: string[] = []
  const pushToken = (token: string) => {
    const trimmed = token.trim()
    if (!trimmed) return
    const lower = trimmed.toLowerCase()
    if (seen.has(lower)) return
    seen.add(lower)
    tokens.push(trimmed)
  }
  if (Array.isArray(value)) {
    value.forEach(item => pushToken(String(item ?? '')))
    return tokens
  }
  if (typeof value === 'string') {
    value
      .split(/[,;\n]+/g)
      .forEach(item => pushToken(item))
    return tokens
  }
  return tokens
}

const statusBadgeClass = (status?: string) => {
  switch ((status || '').toLowerCase()) {
    case 'completed':
      return 'badge bg-success'
    case 'running':
    case 'queued':
    case 'pending':
      return 'badge bg-primary'
    case 'error':
      return 'badge bg-danger'
    default:
      return 'badge bg-secondary'
  }
}

const ONTOLOGY_COLORS: Record<string, string> = {
  person: '#2d76ff',
  organization: '#ff7f36',
  location: '#1f9d55',
  event: '#be3455',
  object: '#ffb347',
  concept: '#6f42c1',
  other: '#8b949e',
}

function OsintOntologyGraph({ graph, variant = 'default' }: { graph: OntologyPayload; variant?: 'default' | 'fullscreen' }) {
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const nodesSource = Array.isArray(graph.nodes) ? graph.nodes : []
    const edgesSource = Array.isArray(graph.edges) ? graph.edges : []
    if (!nodesSource.length) {
      container.innerHTML = ''
      return
    }

    const containerRect = container.getBoundingClientRect()
    const method = String(graph.meta?.method || '').toLowerCase()
    const fallbackLayout = method === 'fallback'
    const circleLayout = fallbackLayout && nodesSource.length > 1
    const disablePhysics = fallbackLayout
    const fallbackTargets = circleLayout ? nodesSource.filter(node => node.id !== 'query') : []
    const fallbackRadius = circleLayout
      ? Math.max(240, Math.max(containerRect.width, containerRect.height || 0) / 2.3)
      : 0
    let fallbackIndex = 0
    const fallbackTotal = Math.max(1, fallbackTargets.length)

    const buildNodeTitle = (node: OntologyNode) => {
      const chunks: string[] = [node.label]
      if (node.summary) chunks.push(node.summary)
      if (node.aliases && node.aliases.length) chunks.push(`Алиасы: ${node.aliases.join(', ')}`)
      if (node.sources && node.sources.length) chunks.push(`Источники: ${node.sources.join(', ')}`)
      return chunks.join('\n')
    }

    const resolveSize = (node: OntologyNode) => {
      const score = typeof node.score === 'number' && Number.isFinite(node.score) ? node.score : null
      if (score === null) return 18
      if (score <= 1) return 18 + Math.max(0, Math.min(12, score * 20))
      return 18 + Math.max(0, Math.min(12, score * 0.6))
    }

    const nodes = new DataSet(
      nodesSource.map(node => {
        let position: { x?: number; y?: number; fixed?: boolean } = {}
        if (circleLayout) {
          if (node.id === 'query') {
            position = { x: 0, y: 0, fixed: true }
          } else {
            const angle = (2 * Math.PI * fallbackIndex) / fallbackTotal
            fallbackIndex += 1
            const radius = Number.isFinite(fallbackRadius) ? fallbackRadius : 260
            position = {
              x: radius * Math.cos(angle),
              y: radius * Math.sin(angle),
              fixed: true,
            }
          }
        }
        return {
          id: node.id,
          label: node.label,
          title: buildNodeTitle(node),
          color: ONTOLOGY_COLORS[(node.type || '').toLowerCase()] || ONTOLOGY_COLORS.other,
          size: resolveSize(node),
          font: { color: '#0f172a' },
          ...position,
        }
      })
    )
    const edges = new DataSet(
      edgesSource.map(edge => ({
        id: edge.id,
        from: edge.from,
        to: edge.to,
        arrows: 'to',
        smooth: circleLayout ? false : true,
        label: edge.label || undefined,
        title: [edge.description, edge.sources && edge.sources.length ? `Источники: ${edge.sources.join(', ')}` : null]
          .filter(Boolean)
          .join('\n'),
        color: { color: '#6c757d' },
        font: { size: 10, color: '#343a40', strokeWidth: 0, align: 'horizontal' },
      }))
    )

    const network = new Network(
      container,
      { nodes, edges },
      {
        layout: { improvedLayout: !disablePhysics },
        physics: disablePhysics
          ? false
          : {
              stabilization: true,
              solver: 'barnesHut',
              barnesHut: { gravitationalConstant: -18000, springLength: 180, centralGravity: 0.28 },
            },
        interaction: { hover: true, tooltipDelay: 160 },
        edges: { smooth: !circleLayout },
        nodes: { shape: 'dot', borderWidth: 1, font: { color: '#0f172a' } },
      }
    )

    const allNodeIds = nodes.getIds() as string[]
    const basePadding = circleLayout ? 12 : 56

    const fitNetwork = (animate: boolean) => {
      try {
        network.fit({
          nodes: allNodeIds.length ? allNodeIds : undefined,
          animation: animate
            ? { duration: 320, easingFunction: 'easeInOutCubic' }
            : false,
          padding: basePadding,
        })
      } catch {
        // ignore fit errors
      }
    }

    if (!disablePhysics) {
      try {
        network.stabilize(180)
      } catch {
        // ignore stabilize errors
      }
    }

    fitNetwork(true)
    if (circleLayout) {
      try {
        network.moveTo({ position: { x: 0, y: 0 } })
      } catch {
        // ignore move errors
      }
    }
    const delayedFit = window.setTimeout(() => fitNetwork(true), 160)
    const handleResize = () => fitNetwork(false)
    window.addEventListener('resize', handleResize)
    if (!disablePhysics) {
      network.once('stabilized', () => fitNetwork(true))
    }

    return () => {
      window.clearTimeout(delayedFit)
      window.removeEventListener('resize', handleResize)
      try {
        network.destroy()
      } catch (err) {
        // ignore cleanup errors
      }
      nodes.clear()
      edges.clear()
    }
  }, [graph, variant])

  const containerStyle =
    variant === 'fullscreen'
      ? { width: '100%', height: '100%', minHeight: 520 }
      : { width: '100%', minHeight: 520, height: 'clamp(520px, 60vh, 760px)' }

  return <div ref={containerRef} style={containerStyle} />
}

export default function OsintPage() {
  const toasts = useToasts()
  const [query, setQuery] = useState('')
  const [selectedLocales, setSelectedLocales] = useState<string[]>([...LOCALE_OPTIONS])
  const primaryLocale = selectedLocales[0] || LOCALE_OPTIONS[0]
  const toggleLocale = useCallback((value: string) => {
    setSelectedLocales(prev => {
      const hasValue = prev.includes(value)
      if (hasValue && prev.length === 1) {
        return prev
      }
      const nextSet = new Set(prev)
      if (hasValue) {
        nextSet.delete(value)
      } else {
        nextSet.add(value)
      }
      return LOCALE_OPTIONS.filter(option => nextSet.has(option))
    })
  }, [])
  const [safe, setSafe] = useState(false)
  const [maxResults, setMaxResults] = useState<string>('10')
  const [includeHtml, setIncludeHtml] = useState(false)
  const [includeLlm, setIncludeLlm] = useState(false)
  const [buildOntology, setBuildOntology] = useState(false)
  const [scheduleEnabled, setScheduleEnabled] = useState(false)
  const [scheduleInterval, setScheduleInterval] = useState('1440')
  const [scheduleStart, setScheduleStart] = useState('')
  const [scheduleNotify, setScheduleNotify] = useState(false)
  const [scheduleLabel, setScheduleLabel] = useState('')
  const [ontologyFullscreen, setOntologyFullscreen] = useState(false)

  const [useGoogle, setUseGoogle] = useState(true)
  const [useYandex, setUseYandex] = useState(false)
  const [useLocalCatalogue, setUseLocalCatalogue] = useState(false)
  const [useLocalPath, setUseLocalPath] = useState(false)
  const [localPath, setLocalPath] = useState('')
  const [localRecursive, setLocalRecursive] = useState(true)
  const [localLimit, setLocalLimit] = useState('20')
  const [localExclude, setLocalExclude] = useState('')
  const [localOcr, setLocalOcr] = useState(false)

  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [activeJob, setActiveJob] = useState<OsintJobSnapshot | null>(null)
  const [history, setHistory] = useState<OsintJobSnapshot[]>([])
  const [exporting, setExporting] = useState(false)
  const [captchaDialog, setCaptchaDialog] = useState<{ sourceId: string; label: string; url: string | null } | null>(null)
  const [retrying, setRetrying] = useState(false)
  const [remoteBrowserSession, setRemoteBrowserSession] = useState<{
    sourceId: string
    label: string
    viewport: { width: number; height: number }
  } | null>(null)
  const [remoteBrowserTick, setRemoteBrowserTick] = useState(0)
  const [remoteBrowserLoading, setRemoteBrowserLoading] = useState(false)
  const pollingRef = useRef<number | null>(null)
  const remoteBrowserIntervalRef = useRef<number | null>(null)
  const folderInputRef = useRef<HTMLInputElement | null>(null)
  const fallbackNotifiedRef = useRef<Set<string>>(new Set())
  const jobMaxResults = useMemo(() => normalizeNumber(activeJob?.params?.max_results), [activeJob])
  const getSourceStats = useCallback((source: OsintSourceSnapshot) => {
    const meta = source.metadata || {}
    const pages = normalizeNumber(meta.pages_estimated ?? meta.pages_processed)
    const parsed = normalizeNumber(meta.results_parsed ?? meta.results_collected)
    const served = normalizeNumber(
      meta.results_on_page ??
      meta.results_collected ??
      (source.results ? source.results.length : null),
    )
    const forwarded = normalizeNumber(
      meta.results_forwarded ??
      meta.links_collected ??
      (source.results ? source.results.length : null),
    )
    const requestedCandidate =
      normalizeNumber(meta.requested_results) ??
      normalizeNumber(source.params?.max_results)
    const requested = requestedCandidate ?? jobMaxResults ?? null
    const keywords = normalizeKeywordsList(meta.keywords)
    const retry = Boolean(meta.retry_available ?? meta.fallback)
    return { pages, parsed, forwarded, requested, keywords, served, retry }
  }, [jobMaxResults])
  const jobKeywords = useMemo(() => normalizeKeywordsList(activeJob?.params?.keywords), [activeJob])
  const jobStats = useMemo(() => {
    if (!activeJob?.sources || activeJob.sources.length === 0) {
      return null
    }
    let pagesTotal = 0
    let serpTotal = 0
    let forwardedTotal = 0
    let requestedMax = jobMaxResults ?? 0
    activeJob.sources.forEach(source => {
      const stats = getSourceStats(source)
      if (stats.pages && stats.pages > 0) {
        pagesTotal += stats.pages
      }
      if (stats.parsed && stats.parsed > 0) {
        serpTotal += stats.parsed
      }
      if (stats.forwarded && stats.forwarded > 0) {
        forwardedTotal += stats.forwarded
      }
      if (stats.requested && (!requestedMax || stats.requested > requestedMax)) {
        requestedMax = stats.requested
      }
    })
    if (!pagesTotal && !serpTotal && !forwardedTotal && !requestedMax) {
      return null
    }
    return {
      pages: pagesTotal,
      serp: serpTotal,
      forwarded: forwardedTotal,
      requested: requestedMax || null,
    }
  }, [activeJob, getSourceStats, jobMaxResults])

  const buildSourcePayload = useCallback(() => {
    const sources: any[] = []
    if (useGoogle) sources.push({ type: 'engine', engine: 'google' })
    if (useYandex) sources.push({ type: 'engine', engine: 'yandex' })
    const parseLimit = (raw: string, fallback: number, min: number, max: number) => {
      const parsed = Number.parseInt(raw || String(fallback), 10)
      if (Number.isNaN(parsed)) return fallback
      return Math.min(Math.max(parsed, min), max)
    }

    if (useLocalCatalogue) {
      sources.push({
        type: 'local',
        id: 'local-catalog',
        options: {
          mode: 'catalog',
          limit: parseLimit(localLimit, 20, 1, 200),
        },
        label: 'Локальная библиотека',
      })
    }
    if (useLocalPath) {
      const pathValue = localPath.trim()
      if (!pathValue) {
        throw new Error('Укажите путь к локальной или сетевой папке')
      }
      if (!isAbsolutePath(pathValue)) {
        throw new Error('Укажите абсолютный путь к локальной папке (например /Users/ivan/Documents или C:\\data)')
      }
      const excludePatterns = localExclude
        .split(',')
        .map(pattern => pattern.trim())
        .filter(Boolean)
      const filesystemOptions: Record<string, unknown> = {
        mode: 'filesystem',
        path: pathValue,
        recursive: localRecursive,
        limit: parseLimit(localLimit, 20, 1, 200),
      }
      if (excludePatterns.length > 0) {
        filesystemOptions.exclude_patterns = excludePatterns
      }
      if (localOcr) {
        filesystemOptions.ocr = true
      }
      sources.push({
        type: 'local',
        id: 'local-filesystem',
        options: filesystemOptions,
        label: 'Локальная папка',
      })
    }
    if (sources.length === 0) {
      DEFAULT_REMOTE_SOURCES.forEach(engine => sources.push({ type: 'engine', engine }))
    }
    return sources
  }, [useGoogle, useYandex, useLocalCatalogue, useLocalPath, localPath, localRecursive, localLimit, localExclude, localOcr])

  const cancelPolling = useCallback(() => {
    if (pollingRef.current) {
      window.clearTimeout(pollingRef.current)
      pollingRef.current = null
    }
  }, [])

  const handleFolderSelect = useCallback(async () => {
    try {
      if (typeof window !== 'undefined' && 'showDirectoryPicker' in window) {
        const picker = await (window as any).showDirectoryPicker()
        if (picker && picker.name) {
          setLocalPath(`~/${picker.name}`)
        }
        return
      }
    } catch (err) {
      // игнорируем отмену выбора
    }
    folderInputRef.current?.click()
  }, [])

  const handleFolderInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (!files || files.length === 0) return
    const first = files[0]
    const relative = (first as any).webkitRelativePath || ''
    if (relative) {
      const folder = relative.split('/')[0]
      if (folder) {
        setLocalPath(`~/${folder}`)
        return
      }
    }
    if (first.name) {
      setLocalPath(`~/${first.name}`)
    }
  }, [])

  const loadHistory = useCallback(async () => {
    try {
      const response = await fetch('/api/osint/history?limit=10')
      if (!response.ok) throw new Error('request_failed')
      const data: ApiJobsResponse = await response.json()
      if (data.ok && Array.isArray(data.items)) {
        setHistory(data.items)
      }
    } catch {
      setHistory([])
    }
  }, [])

  const handleDeleteJob = useCallback(async (jobId: number) => {
    try {
      const response = await fetch(`/api/osint/jobs/${jobId}`, { method: 'DELETE' })
      if (!response.ok) {
        throw new Error('delete_failed')
      }
      toasts.push('Поиск удалён из истории', 'success')
      setHistory(prev => prev.filter(item => item.id !== jobId))
      if (activeJob?.id === jobId) {
        cancelPolling()
        setActiveJob(null)
      }
      loadHistory()
    } catch {
      toasts.push('Не удалось удалить запись истории', 'error')
    }
  }, [activeJob, cancelPolling, loadHistory, toasts])

  const handleDisableSchedule = useCallback(async (jobId: number, scheduleId?: number) => {
    try {
      const response = await fetch(`/api/osint/jobs/${jobId}/schedule`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ schedule: { id: scheduleId, active: false } }),
      })
      const data: ApiJobResponse = await response.json()
      if (!response.ok || !data.ok || !data.job) {
        throw new Error(data.error || 'disable_failed')
      }
      setActiveJob(prev => (prev && prev.id === data.job?.id ? data.job : prev))
      toasts.push('Расписание отключено', 'success')
      loadHistory()
    } catch (err) {
      toasts.push('Не удалось отключить расписание', 'error')
    }
  }, [loadHistory, toasts])

  const pollJob = useCallback(async (jobId: number) => {
    cancelPolling()
    try {
      const response = await fetch(`/api/osint/jobs/${jobId}`)
      if (!response.ok) throw new Error('request_failed')
      const data: ApiJobResponse = await response.json()
      if (!data.ok || !data.job) throw new Error(data.error || 'invalid_response')
      setActiveJob(data.job)
      const status = data.job.status ? data.job.status.toLowerCase() : ''
      const needsAnalysis = status === 'completed' && !data.job.analysis && !data.job.analysis_error
      const wantsOntology = Boolean(data.job.params?.build_ontology)
      const needsOntology = status === 'completed' && wantsOntology && !data.job.ontology && !data.job.ontology_error
      if (['queued', 'running', 'pending'].includes(status) || needsAnalysis || needsOntology) {
        pollingRef.current = window.setTimeout(() => pollJob(jobId), 2000)
      }
    } catch (err: any) {
      toasts.push('Не удалось обновить прогресс OSINT-задачи', 'error')
    }
  }, [cancelPolling, toasts])

  const handleRetryConfirm = useCallback(async () => {
    if (!activeJob || !captchaDialog) {
      setCaptchaDialog(null)
      return
    }
    setRetrying(true)
    try {
      const response = await fetch(`/api/osint/jobs/${activeJob.id}/sources/${encodeURIComponent(captchaDialog.sourceId)}/retry`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force_refresh: true }),
      })
      const data: ApiJobResponse = await response.json().catch(() => ({} as ApiJobResponse))
      if (!response.ok || !data?.ok || !data.job) {
        throw new Error(data?.error || 'Не удалось повторить поиск')
      }
      setActiveJob(data.job)
      toasts.push('Источник поставлен на повторный поиск', 'info')
      pollJob(data.job.id)
      loadHistory()
      setCaptchaDialog(null)
    } catch (err: any) {
      toasts.push(err?.message || 'Не удалось повторить поиск', 'error')
    } finally {
      setRetrying(false)
    }
  }, [activeJob, captchaDialog, pollJob, toasts, loadHistory])

  useEffect(() => {
    if (!remoteBrowserSession) {
      if (remoteBrowserIntervalRef.current) {
        window.clearInterval(remoteBrowserIntervalRef.current)
        remoteBrowserIntervalRef.current = null
      }
      return
    }
    if (remoteBrowserIntervalRef.current) {
      window.clearInterval(remoteBrowserIntervalRef.current)
    }
    remoteBrowserIntervalRef.current = window.setInterval(() => {
      setRemoteBrowserTick(prev => prev + 1)
    }, 2000)
    return () => {
      if (remoteBrowserIntervalRef.current) {
        window.clearInterval(remoteBrowserIntervalRef.current)
        remoteBrowserIntervalRef.current = null
      }
    }
  }, [remoteBrowserSession])

  useEffect(() => {
    loadHistory()
    return () => cancelPolling()
  }, [loadHistory, cancelPolling])

  useEffect(() => {
    if (!activeJob) return
    const specs = activeJob.source_specs || []
    const engineSources = specs.filter(spec => spec.type === 'engine')
    setUseGoogle(engineSources.some(spec => spec.engine === 'google'))
    setUseYandex(engineSources.some(spec => spec.engine === 'yandex'))

    const localCatalog = specs.find(spec => spec.type === 'local' && spec.options?.mode === 'catalog')
    const localFilesystem = specs.find(spec => spec.type === 'local' && spec.options?.mode === 'filesystem')

    setUseLocalCatalogue(Boolean(localCatalog))
    setUseLocalPath(Boolean(localFilesystem))

    if (localCatalog?.options?.limit !== undefined) {
      setLocalLimit(String(localCatalog.options.limit))
    }
    if (localFilesystem) {
      const options = localFilesystem.options || {}
      if (options.path) setLocalPath(String(options.path))
      if (options.limit !== undefined) setLocalLimit(String(options.limit))
      setLocalRecursive(Boolean(options.recursive))
      if (Array.isArray(options.exclude_patterns)) {
        setLocalExclude(options.exclude_patterns.join(', '))
      } else {
        setLocalExclude('')
      }
      setLocalOcr(Boolean(options.ocr))
    }

    const jobParams = activeJob.params || {}
    setIncludeHtml(Boolean(jobParams.include_html))
    setIncludeLlm(Boolean(jobParams.include_llm_payload))
    setBuildOntology(Boolean(jobParams.build_ontology))
    if (jobParams.max_results !== undefined && jobParams.max_results !== null && jobParams.max_results !== '') {
      setMaxResults(String(jobParams.max_results))
    }

    const scheduleConfig = jobParams.schedule || activeJob.schedule
    if (scheduleConfig && scheduleConfig.interval_minutes) {
      setScheduleEnabled(true)
      setScheduleInterval(String(scheduleConfig.interval_minutes))
      setScheduleNotify(Boolean(scheduleConfig.notify))
      setScheduleLabel(scheduleConfig.label || '')
      if (scheduleConfig.start_at) {
        const start = new Date(scheduleConfig.start_at)
        if (!Number.isNaN(start.getTime())) {
          const iso = start.toISOString()
          setScheduleStart(iso.slice(0, 16))
        }
      }
    } else {
      setScheduleEnabled(false)
    }

    const fallbackSources =
      activeJob.sources?.filter(source => source.metadata?.fallback || source.metadata?.blocked) || []
    fallbackSources.forEach(source => {
      const key = `${activeJob.id}:${source.source}`
      if (!fallbackNotifiedRef.current.has(key)) {
        fallbackNotifiedRef.current.add(key)
        const label = source.metadata?.label || source.source
        const fallbackUrl = source.metadata?.fallback_url || source.final_url || source.requested_url
        const message = source.metadata?.fallback_url
          ? `Источник «${label}» запросил капчу. Откройте оригинал и подтвердите «я не робот».`
          : `Источник «${label}» заблокировал выдачу. Перейдите вручную и пройдите проверку.`
        const action = fallbackUrl ? (
          <a
            href={String(fallbackUrl)}
            target="_blank"
            rel="noopener"
            className="btn btn-sm btn-primary"
          >
            Пройти капчу
          </a>
        ) : undefined
        toasts.push(message, 'info', action)
      }
    })
  }, [activeJob, toasts])

  useEffect(() => {
    if (typeof document === 'undefined') return
    if (!ontologyFullscreen) return
    const original = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = original
    }
  }, [ontologyFullscreen])

  useEffect(() => {
    if (!ontologyFullscreen) return
    if (activeJob?.ontology) return
    setOntologyFullscreen(false)
  }, [ontologyFullscreen, activeJob?.ontology])

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    setError(null)
    const trimmed = query.trim()
    if (!trimmed) {
      setError('Введите поисковый запрос')
      return
    }
    if (selectedLocales.length === 0) {
      setError('Выберите хотя бы одну локаль')
      return
    }
    let sources: any[]
    try {
      sources = buildSourcePayload()
    } catch (err: any) {
      setError(err?.message || 'Ошибка конфигурации источников')
      return
    }
    let schedulePayload: Record<string, any> | undefined
    if (scheduleEnabled) {
      const intervalValue = Number(scheduleInterval)
      if (!Number.isFinite(intervalValue) || intervalValue < 5) {
        setError('Минимальный интервал расписания — 5 минут')
        return
      }
      schedulePayload = {
        interval_minutes: Math.floor(intervalValue),
        notify: scheduleNotify,
      }
      const labelTrimmed = scheduleLabel.trim()
      if (labelTrimmed) {
        schedulePayload.label = labelTrimmed
      }
      const startTrimmed = scheduleStart.trim()
      if (startTrimmed) {
        try {
          const startDate = new Date(startTrimmed)
          if (Number.isNaN(startDate.getTime())) {
            throw new Error('invalid')
          }
          schedulePayload.start_at = startDate.toISOString()
        } catch {
          setError('Некорректная дата запуска для расписания')
          return
        }
      }
    }
    setSubmitting(true)
    cancelPolling()
    try {
      const hasRemoteSources = sources.some(src => src.type === 'engine')
      if (hasRemoteSources && typeof navigator !== 'undefined' && navigator && navigator.onLine === false) {
        setError('нет интернета - нет ответа')
        setSubmitting(false)
        return
      }
      const payload = {
        query: trimmed,
        locale: primaryLocale,
        locales: selectedLocales,
        safe,
        max_results: maxResults.trim() ? Number.isNaN(Number(maxResults)) ? undefined : Number(maxResults) : undefined,
        include_html: includeHtml,
        include_llm_payload: includeLlm,
        build_ontology: buildOntology,
        sources,
        schedule: schedulePayload,
      }
      const response = await fetch('/api/osint/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data: ApiJobResponse = await response.json()
      if (!response.ok || !data.ok || !data.job) {
        throw new Error(data.error || 'Не удалось запустить поиск')
      }
      setActiveJob(data.job)
      toasts.push('OSINT-запрос запущен', 'success')
      pollJob(data.job.id)
      loadHistory()
    } catch (err: any) {
      setError(err?.message || 'Ошибка запуска поиска')
    } finally {
      setSubmitting(false)
    }
  }

  const progressPercent = useMemo(() => {
    if (!activeJob || !activeJob.sources_total) return 0
    return Math.round((activeJob.sources_completed / activeJob.sources_total) * 100)
  }, [activeJob])

  const remoteBrowserSnapshotUrl = useMemo(() => {
    if (!remoteBrowserSession) return ''
    return `/api/osint/browser-session/${encodeURIComponent(remoteBrowserSession.sourceId)}/snapshot?ts=${remoteBrowserTick}`
  }, [remoteBrowserSession, remoteBrowserTick])

  const progressStatsFor = (entry: OsintProgressEntry) => {
    const pages = normalizeNumber(entry.pages_processed ?? entry.pages_estimated)
    const parsed = normalizeNumber(entry.results_parsed ?? entry.links_collected)
    const served = normalizeNumber(entry.results_on_page ?? entry.links_collected)
    const forwarded = normalizeNumber(entry.results_forwarded ?? entry.links_collected)
    const requested = normalizeNumber(entry.requested_results)
    const keywords = normalizeKeywordsList(entry.keywords)
    const refined =
      typeof entry.refined_query === 'string' && entry.refined_query.trim().length > 0
        ? entry.refined_query.trim()
        : undefined
    const retry = Boolean(entry.retry_available ?? entry.fallback)
    return { pages, parsed, served, forwarded, requested, keywords, refined, retry }
  }

  const renderSourceProgress = () => {
    if (!activeJob) return null
    const entries = Object.entries(activeJob.progress || {})
    if (!entries.length) return null
    return (
      <div className="list-group mb-3">
        {entries.map(([sourceId, info]) => {
          const stats = progressStatsFor(info)
          const sourceDetails = activeJob.sources?.find(item => item.source === sourceId)
          const fallbackUrl =
            info.fallback_url ||
            info.final_url ||
            sourceDetails?.metadata?.fallback_url ||
            sourceDetails?.final_url ||
            sourceDetails?.requested_url ||
            null
          const hasPages = stats.pages !== null && stats.pages > 0
          const hasParsed = stats.parsed !== null && stats.parsed > 0
          const hasServed = stats.served !== null && stats.served > 0
          const hasForwarded = stats.forwarded !== null && stats.forwarded > 0
          const hasRequested = stats.requested !== null && stats.requested > 0
          return (
            <div key={sourceId} className="list-group-item d-flex justify-content-between align-items-center gap-3">
              <div>
                <div className="fw-semibold">{info.label || sourceId}</div>
                <div className="small text-muted">
                  {info.status}
                  {info.updated_at ? ` · обновлено: ${formatTimestamp(info.updated_at)}` : ''}
                </div>
                {info.error && <div className="text-danger small">{info.error}</div>}
                {stats.retry && (
                  <div className="text-warning small d-flex flex-wrap align-items-center gap-2 mt-1">
                    <span>Поисковик запросил проверку вручную.</span>
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-primary"
                      onClick={() => openCaptchaDialog(sourceId, info.label || sourceId, fallbackUrl)}
                    >
                      Повторить
                    </button>
                    {fallbackUrl && (
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-secondary"
                      onClick={() => startRemoteBrowser(sourceId, fallbackUrl, info.label || sourceId)}
                    >
                      Открыть браузер
                    </button>
                    )}
                    {fallbackUrl && (
                      <button
                        type="button"
                        className="btn btn-sm btn-outline-secondary"
                        onClick={() => window.open(fallbackUrl, '_blank', 'noopener')}
                      >
                        Открыть капчу
                      </button>
                    )}
                  </div>
                )}
                {(hasPages || hasParsed || hasServed || hasForwarded || hasRequested) && (
                  <div className="d-flex flex-wrap gap-2 small text-muted mt-1">
                    {hasPages && <span className="badge bg-light text-dark border">Страниц: {stats.pages}</span>}
                    {hasParsed && <span className="badge bg-light text-dark border">Ссылок (SERP): {stats.parsed}</span>}
                    {hasServed && <span className="badge bg-light text-dark border">Сохранено: {stats.served}</span>}
                    {hasForwarded && <span className="badge bg-light text-dark border">Передано: {stats.forwarded}</span>}
                    {hasRequested && <span className="badge bg-light text-dark border">Лимит: {stats.requested}</span>}
                  </div>
                )}
                {stats.keywords.length > 0 && (
                  <div className="d-flex flex-wrap gap-2 mt-1">
                    {stats.keywords.map(keyword => (
                      <span key={`${sourceId}-${keyword}`} className="badge bg-light text-muted border">
                        {keyword}
                      </span>
                    ))}
                  </div>
                )}
                {stats.refined && (
                  <div className="small text-muted mt-1">
                    Поиск по: <code>{stats.refined}</code>
                  </div>
                )}
                {stats.requested && stats.forwarded !== null && stats.forwarded < stats.requested && (
                  <div className="text-warning small mt-1">
                    Получено меньше ссылок, чем лимит — возможна капча или блокировка.
                  </div>
                )}
              </div>
              <span className={statusBadgeClass(info.status)}>{info.status}</span>
            </div>
          )
        })}
      </div>
    )
  }

  const handleExportJob = useCallback(async (jobId: number, format: 'markdown' | 'json') => {
    if (typeof document === 'undefined') return
    setExporting(true)
    try {
      const response = await fetch(`/api/osint/jobs/${jobId}/export${format === 'json' ? '?format=json' : ''}`)
      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        throw new Error(data?.error || 'Не удалось выгрузить отчёт')
      }
      const text = await response.text()
      const blob = new Blob(
        [text],
        { type: format === 'json' ? 'application/json;charset=utf-8' : 'text/markdown;charset=utf-8' },
      )
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `osint-job-${jobId}.${format === 'json' ? 'json' : 'md'}`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)
      toasts.push(`Отчёт сохранён в формате ${format === 'json' ? 'JSON' : 'Markdown'}`, 'success')
    } catch (err: any) {
      toasts.push(err?.message || 'Не удалось выгрузить отчёт', 'error')
    } finally {
      setExporting(false)
    }
  }, [toasts])

  const openCaptchaDialog = useCallback((sourceId: string, label: string, url: string | null) => {
    setCaptchaDialog({ sourceId, label, url })
  }, [])

  const closeCaptchaDialog = useCallback(() => {
    if (retrying) return
    setCaptchaDialog(null)
  }, [retrying])

  const startRemoteBrowser = useCallback(async (sourceId: string, url: string | null, label: string) => {
    if (!url) return
    setRemoteBrowserLoading(true)
    try {
      const response = await fetch('/api/osint/browser-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_id: sourceId, url }),
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok || !data?.ok) {
        throw new Error(data?.error || 'browser_session_failed')
      }
      setRemoteBrowserSession({
        sourceId,
        label,
        viewport: (data.viewport && typeof data.viewport === 'object'
          ? data.viewport
          : { width: 1366, height: 768 }) as { width: number; height: number },
      })
      setRemoteBrowserTick(prev => prev + 1)
    } catch (err: any) {
      toasts.push(err?.message || 'Не удалось открыть браузер', 'error')
    } finally {
      setRemoteBrowserLoading(false)
    }
  }, [toasts])

  const closeRemoteBrowser = useCallback(async () => {
    if (!remoteBrowserSession) return
    try {
      await fetch(`/api/osint/browser-session/${encodeURIComponent(remoteBrowserSession.sourceId)}`, {
        method: 'DELETE',
      })
    } catch {
      // ignore
    }
    setRemoteBrowserSession(null)
  }, [remoteBrowserSession])

  const sendRemoteBrowserAction = useCallback(async (action: Record<string, unknown>) => {
    if (!remoteBrowserSession) return
    try {
      await fetch(
        `/api/osint/browser-session/${encodeURIComponent(remoteBrowserSession.sourceId)}/action`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(action),
        },
      )
      setRemoteBrowserTick(prev => prev + 1)
    } catch {
      // ignore
    }
  }, [remoteBrowserSession])

  const handleRemoteBrowserClick = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      if (!remoteBrowserSession) return
      const bounds = event.currentTarget.getBoundingClientRect()
      const relX = (event.clientX - bounds.left) / bounds.width
      const relY = (event.clientY - bounds.top) / bounds.height
      const clampedX = Math.max(0, Math.min(1, relX))
      const clampedY = Math.max(0, Math.min(1, relY))
      sendRemoteBrowserAction({ type: 'click', x: clampedX, y: clampedY })
    },
    [remoteBrowserSession, sendRemoteBrowserAction],
  )

  const renderJobSummary = () => {
    if (!activeJob) return null
    const refinedRaw = typeof activeJob.params?.refined_query === 'string' ? activeJob.params?.refined_query.trim() : ''
    const baseQuery = (activeJob.query || '').trim()
    const refinedDiffers =
      refinedRaw &&
      refinedRaw.localeCompare(baseQuery, undefined, { sensitivity: 'base' }) !== 0
    const requestedLimit = jobStats?.requested ?? jobMaxResults ?? null
    if (!refinedDiffers && jobKeywords.length === 0 && !jobStats) {
      return null
    }
    return (
      <div className="border rounded-3 p-3 mb-3 bg-light">
        <div className="d-flex justify-content-between flex-wrap gap-3">
          <div className="flex-grow-1">
            <div className="small text-muted">Исходный запрос</div>
            <div className="fw-semibold" style={{ whiteSpace: 'pre-wrap' }}>{baseQuery || '—'}</div>
            {refinedDiffers && (
              <div className="small text-muted mt-1">
                В поиске использовано:{' '}
                <code>{refinedRaw}</code>
              </div>
            )}
          </div>
          {(jobStats || requestedLimit) && (
            <div className="text-end small text-muted">
              {jobStats?.serp && jobStats.serp > 0 && <div>Ссылок в SERP: {jobStats.serp}</div>}
              {jobStats?.forwarded && jobStats.forwarded > 0 && <div>Передано в анализ: {jobStats.forwarded}</div>}
              {jobStats?.pages && jobStats.pages > 0 && <div>Страниц выдачи: {jobStats.pages}</div>}
              {requestedLimit && <div>Лимит SERP: {requestedLimit}</div>}
            </div>
          )}
        </div>
        {jobKeywords.length > 0 && (
          <div className="d-flex flex-wrap gap-2 mt-3">
            {jobKeywords.map(keyword => (
              <span key={`job-keyword-${keyword}`} className="badge bg-light text-muted border">
                {keyword}
              </span>
            ))}
          </div>
        )}
        {jobStats && jobStats.requested && jobStats.forwarded !== null && jobStats.forwarded < jobStats.requested && (
          <div className="small text-warning mt-2">
            Получено только {jobStats.forwarded} ссылок из запрошенных {jobStats.requested}. Возможно, поисковик ограничил выдачу (капча или блокировка).
          </div>
        )}
      </div>
    )
  }

  const llmContext = useMemo(() => {
    if (!activeJob?.sources) return []
    return activeJob.sources
      .map(source => {
        const label = source.metadata?.label || source.source
        const forwarded = normalizeNumber(source.metadata?.results_forwarded) ?? source.results.length
        const entries = (source.results || []).slice(0, 3).map(item => ({
          id: `${source.source}-${item.rank}-${item.url}`,
          title: item.title || item.url,
          url: item.url,
          snippet: item.snippet || '',
          highlight: item.highlight || (item.metadata?.highlight as string | undefined) || '',
        }))
        return {
          id: source.source,
          label,
          forwarded,
          fallback: Boolean(source.metadata?.fallback),
          entries: entries.filter(entry => entry.title),
        }
      })
      .filter(entry => entry.entries.length > 0)
  }, [activeJob])

  const renderLlmContext = () => {
    if (!activeJob) return null
    if (llmContext.length === 0) return null
    const totalLinks = llmContext.reduce((acc, block) => acc + (block.forwarded || block.entries.length), 0)
    return (
      <section className="card shadow-sm mb-3">
        <div className="card-body">
          <div className="d-flex justify-content-between align-items-center mb-3">
            <div>
              <h3 className="h6 mb-1">Контекст для LLM</h3>
              <div className="small text-muted">Отправлено ссылок: {totalLinks}</div>
            </div>
          </div>
          <div className="list-group">
            {llmContext.map(block => (
              <div key={block.id} className="list-group-item">
                <div className="d-flex justify-content-between align-items-start gap-2 mb-2">
                  <div className="fw-semibold">{block.label}</div>
                  <span className="badge bg-light text-dark border">
                    {block.forwarded || block.entries.length} ссылок
                  </span>
                </div>
                {block.fallback && (
                  <div className="text-warning small mb-2">
                    Источник выдал страницу проверки — данные могли быть неполными.
                  </div>
                )}
                <ul className="list-unstyled small mb-0">
                  {block.entries.map(entry => {
                    const snippet = entry.highlight?.trim() || entry.snippet
                    return (
                      <li key={entry.id} className="mb-2">
                        {entry.url ? (
                          <a href={entry.url} target="_blank" rel="noopener" className="fw-semibold">
                            {entry.title}
                          </a>
                        ) : (
                          <span className="fw-semibold">{entry.title}</span>
                        )}
                        {snippet && (
                          <div
                            className="text-muted"
                            style={{ whiteSpace: 'pre-wrap' }}
                            dangerouslySetInnerHTML={{ __html: snippet.length > 280 ? `${snippet.slice(0, 280)}…` : snippet }}
                          />
                        )}
                      </li>
                    )
                  })}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>
    )
  }

  const renderCaptchaNotice = () => {
    if (!activeJob) return null
    const sources =
      activeJob.sources?.filter(source => source.metadata?.fallback || source.metadata?.blocked) || []
    if (sources.length === 0) return null
    return (
      <div className="alert alert-warning" role="alert">
        <div className="fw-semibold mb-1">Поисковик запросил проверку «Я не робот»</div>
        <div className="text-muted small mb-2">
          Запросы выполняются из изолированного браузера на сервере; подтверждение «я не робот» в вашем окне не передаётся напрямую, но помогает обезопасить IP-адрес.
          После проверки снова нажмите «Повторить», а при регулярных ограничениях попробуйте указать переменные окружения <code>OSINT_RETRY_USER_AGENTS</code> и <code>OSINT_RETRY_PROXIES</code> или снизить частоту запросов.
        </div>
        <ul className="mb-0">
          {sources.map(source => {
            const label = source.metadata?.label || source.source
            const url = source.metadata?.fallback_url || source.final_url || source.requested_url
            return (
              <li key={source.source} className="small">
                {label}:{' '}
                {url ? 'откройте страницу проверки и подтвердите «я не робот».' : 'откройте выдачу вручную и подтвердите «я не робот».'}
                <button
                  type="button"
                  className="btn btn-sm btn-outline-primary ms-2"
                  onClick={() => openCaptchaDialog(source.source, label, url || null)}
                >
                  Повторить
                </button>
                {url && (
                  <button
                    type="button"
                    className="btn btn-sm btn-outline-secondary ms-2"
                    onClick={() => startRemoteBrowser(source.source, url, label)}
                  >
                    Открыть браузер
                  </button>
                )}
              </li>
            )
          })}
        </ul>
      </div>
    )
  }

  const renderCaptchaModal = () => {
    if (!captchaDialog) return null
    if (typeof document === 'undefined') return null
    const { sourceId, label, url } = captchaDialog
    return createPortal(
      <div
        role="dialog"
        aria-modal="true"
        style={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(15, 23, 42, 0.5)',
          zIndex: 1050,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '1.5rem',
        }}
        onClick={closeCaptchaDialog}
      >
        <div
          className="card shadow-lg"
          style={{ width: 'min(520px, 95vw)' }}
          onClick={event => event.stopPropagation()}
        >
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-3">
              <h5 className="card-title mb-0">Повторный поиск — {label}</h5>
              <button type="button" className="btn-close" aria-label="Закрыть" onClick={closeCaptchaDialog} disabled={retrying} />
            </div>
            <p className="small text-muted mb-2">
              1. Откройте страницу с проверкой Google и подтвердите, что вы не робот — это помогает убедить систему, что трафик исходит от вас, но не снимает блокировку автоматически, поэтому после любой попытки вернитесь и нажмите «Повторить поиск».
            </p>
            {url ? (
              <div className="input-group input-group-sm mb-3">
                <input
                  className="form-control"
                  value={url}
                  readOnly
                  onFocus={event => event.currentTarget.select()}
                />
                <button
                  type="button"
                  className="btn btn-outline-secondary"
                  onClick={() => window.open(url, '_blank', 'noopener')}
                >
                  Открыть
                </button>
              </div>
            ) : (
              <div className="alert alert-warning small">
                Прямая ссылка не сохранена. Откройте оригинальную выдачу вручную и пройдите капчу.
              </div>
            )}
            <p className="small text-muted">
              2. После успешной проверки вернитесь и нажмите «Повторить поиск», чтобы обновить результаты.
            </p>
            <p className="small text-muted mt-2">
              Если блокировка повторяется, попробуйте указать переменные окружения <code>OSINT_RETRY_USER_AGENTS</code> и <code>OSINT_RETRY_PROXIES</code> или уменьшить частоту запросов — повторный запуск под другим агентом или через прокси может пройти без капчи.
            </p>
            <div className="d-flex justify-content-end gap-2 mt-4">
              <button type="button" className="btn btn-outline-secondary" onClick={closeCaptchaDialog} disabled={retrying}>
                Закрыть
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleRetryConfirm}
                disabled={retrying || !activeJob}
              >
                {retrying ? 'Повтор...' : 'Повторить поиск'}
              </button>
            </div>
            <input type="hidden" value={sourceId} readOnly />
          </div>
        </div>
      </div>,
      document.body,
    )
  }

  const renderRemoteBrowserModal = () => {
    if (!remoteBrowserSession) return null
    if (typeof document === 'undefined') return null
    return createPortal(
      <div
        role="dialog"
        aria-modal="true"
        style={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(15, 23, 42, 0.65)',
          zIndex: 1060,
          display: 'flex',
          alignItems: 'stretch',
          justifyContent: 'center',
          padding: '1.5rem',
        }}
        onClick={closeRemoteBrowser}
      >
        <div
          className="card shadow-lg"
          style={{ width: 'min(1100px, 95vw)', height: 'min(780px, 92vh)', display: 'flex', flexDirection: 'column' }}
          onClick={event => event.stopPropagation()}
        >
          <div className="card-body d-flex flex-column">
            <div className="d-flex justify-content-between align-items-start mb-3">
              <div>
                <h5 className="card-title mb-0">Интерактивный браузер — {remoteBrowserSession.label}</h5>
                <div className="small text-muted" style={{ wordBreak: 'break-all' }}>
                  {remoteBrowserSession.sourceId}
                </div>
              </div>
              <div className="d-flex gap-2">
                <button type="button" className="btn btn-sm btn-outline-secondary" onClick={closeRemoteBrowser} disabled={remoteBrowserLoading}>
                  Завершить
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-outline-light"
                  onClick={() => sendRemoteBrowserAction({ type: 'click', x: 0.5, y: 0.5 })}
                  disabled={remoteBrowserLoading}
                >
                  Обновить
                </button>
              </div>
            </div>
            <div
              className="flex-grow-1 border rounded-3 overflow-hidden"
              style={{ minHeight: 0, backgroundColor: '#0f172a' }}
              onClick={handleRemoteBrowserClick}
            >
              {remoteBrowserSnapshotUrl ? (
                <img
                  src={remoteBrowserSnapshotUrl}
                  alt="OSINT remote browser"
                  style={{ width: '100%', height: '100%', objectFit: 'contain', cursor: 'crosshair' }}
                />
              ) : (
                <div className="d-flex flex-column justify-content-center align-items-center h-100 text-muted">
                  Загрузка браузера...
                </div>
              )}
            </div>
            <div className="small text-muted mt-2">
              Кликайте по изображению, чтобы передать координаты в браузер, после прохождения капчи вернитесь к панели и нажмите «Повторить поиск», чтобы сохранить результаты.
            </div>
          </div>
        </div>
      </div>,
      document.body,
    )
  }

  const renderScheduleInfo = () => {
    if (!activeJob?.schedule) return null
    const schedule = activeJob.schedule
    return (
      <div className="card border-0 bg-light mb-3">
        <div className="card-body">
          <div className="d-flex justify-content-between flex-wrap gap-3">
            <div>
              <div className="fw-semibold mb-1">Расписание</div>
              {schedule.label && <div className="text-muted small mb-2">{schedule.label}</div>}
              <ul className="list-unstyled small mb-0">
                <li>Интервал: {formatInterval(schedule.interval_minutes)}</li>
                {schedule.next_run_at && <li>Следующий запуск: {formatTimestamp(schedule.next_run_at)}</li>}
                {schedule.last_run_at && <li>Последний запуск: {formatTimestamp(schedule.last_run_at)}</li>}
                {schedule.notify && <li>Уведомления: включены</li>}
              </ul>
              {schedule.last_error && (
                <div className="text-danger small mt-2">Последняя ошибка: {schedule.last_error}</div>
              )}
            </div>
            <div className="d-flex flex-column align-items-end gap-2">
              <span className={`badge ${schedule.running ? 'bg-primary' : schedule.active ? 'bg-success' : 'bg-secondary'}`}>
                {schedule.running ? 'выполняется' : schedule.active ? 'активно' : 'выключено'}
              </span>
              <button
                type="button"
                className="btn btn-sm btn-outline-danger"
                onClick={() => handleDisableSchedule(activeJob.id, schedule.id)}
              >
                Отключить
              </button>
            </div>
          </div>
        </div>
      </div>
    )
  }

  const renderAnalysis = () => {
    if (!activeJob) return null
    if (activeJob.analysis) {
      const html = activeJob.analysis_html && activeJob.analysis_html.trim()
        ? activeJob.analysis_html
        : renderMarkdown(activeJob.analysis)
      return (
        <section className="card shadow-sm mb-3">
          <div className="card-body">
            <h3 className="h6 mb-2">Автоматический анализ</h3>
            <div
              className="markdown-body"
              style={{ lineHeight: 1.6 }}
              dangerouslySetInnerHTML={{ __html: html }}
            />
          </div>
        </section>
      )
    }
    if (activeJob.analysis_error) {
      return (
        <div className="alert alert-warning" role="alert">
          Анализ результатов OSINT не выполнен: {activeJob.analysis_error}
        </div>
      )
    }
    return null
  }

  const renderOntology = () => {
    if (!activeJob) return null
    const requested = Boolean(activeJob.params?.build_ontology)
    const rawGraph = activeJob.ontology
    const graph = rawGraph
      ? {
          ...rawGraph,
          nodes: Array.isArray(rawGraph.nodes) ? rawGraph.nodes : [],
          edges: Array.isArray(rawGraph.edges) ? rawGraph.edges : [],
        }
      : null
    const hasGraph = Boolean(graph && (graph.nodes.length > 0 || graph.edges.length > 0))

    if (hasGraph && graph) {
      const method = graph.meta?.method
      const fullscreenOverlay = ontologyFullscreen && typeof document !== 'undefined'
        ? createPortal(
            <div
              role="dialog"
              aria-modal="true"
              style={{
                position: 'fixed',
                inset: 0,
                backgroundColor: 'rgba(15, 23, 42, 0.65)',
                zIndex: 1040,
                padding: '2rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
              onClick={() => setOntologyFullscreen(false)}
            >
              <div
                className="card shadow-lg"
                style={{
                  width: 'min(1200px, 92vw)',
                  height: 'min(900px, 92vh)',
                  display: 'flex',
                  flexDirection: 'column',
                }}
                onClick={event => event.stopPropagation()}
              >
                <div className="card-body d-flex flex-column">
                  <div className="d-flex justify-content-between align-items-center mb-3">
                    <h2 className="h5 mb-0">Онтология</h2>
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-light"
                      onClick={() => setOntologyFullscreen(false)}
                    >
                      Закрыть
                    </button>
                  </div>
                  <div className="flex-grow-1" style={{ minHeight: 0, flex: 1 }}>
                    <OsintOntologyGraph graph={graph} variant="fullscreen" />
                  </div>
                  {method === 'fallback' && (
                    <div className="text-muted small mt-3">
                      Использовано резервное построение графа (LLM недоступен).
                    </div>
                  )}
                </div>
              </div>
            </div>,
            document.body
          )
        : null

      return (
        <>
          <section className="card shadow-sm mb-3">
            <div className="card-body">
              <div className="d-flex align-items-start justify-content-between mb-3">
                <h3 className="h6 mb-0">Онтология</h3>
                <div className="d-flex gap-2">
                  <button
                    type="button"
                    className="btn btn-sm btn-outline-secondary"
                    onClick={() => setOntologyFullscreen(true)}
                  >
                    На весь экран
                  </button>
                </div>
              </div>
              <OsintOntologyGraph graph={graph} />
              {method === 'fallback' && (
                <div className="text-muted small mt-2">
                  Использовано резервное построение графа (LLM недоступен).
                </div>
              )}
            </div>
          </section>
          {fullscreenOverlay}
        </>
      )
    }
    if (activeJob.ontology_error) {
      return (
        <div className="alert alert-warning" role="alert">
          Онтология не построена: {activeJob.ontology_error}
        </div>
      )
    }
    if (requested) {
      if (graph && !hasGraph) {
        return (
          <section className="card shadow-sm mb-3">
            <div className="card-body">
              <h3 className="h6 mb-2">Онтология</h3>
              <div className="text-muted">Сущности не обнаружены в доступных результатах.</div>
            </div>
          </section>
        )
      }
      if (!graph) {
        return (
          <div className="alert alert-info" role="alert">
            Онтология строится, обновите страницу через несколько секунд.
          </div>
        )
      }
    }
    return null
  }

  const renderCombinedResults = () => {
    if (!activeJob) return null
    const items = activeJob.combined_results || []
    if (!Array.isArray(items) || items.length === 0) return null
    return (
      <section className="card shadow-sm mb-3">
        <div className="card-body">
          <h3 className="h6 mb-2">Сводные результаты</h3>
          <div className="list-group">
            {items.map(item => {
              const snippet = (item.snippet || '').trim()
              const highlightHtmlRaw = item.highlight || (item.metadata?.highlight as string | undefined)
              const highlightHtml = typeof highlightHtmlRaw === 'string' && highlightHtmlRaw.trim().length > 0
                ? highlightHtmlRaw
                : null
              const snippetPreview = snippet.length > 220 ? `${snippet.slice(0, 220)}…` : snippet
              return (
                <div key={item.id} className="list-group-item">
                  <div className="d-flex justify-content-between align-items-start gap-3">
                    <div className="flex-grow-1">
                      {item.url ? (
                        <a href={item.url} target="_blank" rel="noopener" className="fw-semibold">
                          {item.title || item.url}
                        </a>
                      ) : (
                        <span className="fw-semibold">{item.title}</span>
                      )}
                      {highlightHtml && (
                        <div
                          className="small text-body-secondary mb-2"
                          style={{ whiteSpace: 'pre-wrap' }}
                          dangerouslySetInnerHTML={{ __html: highlightHtml }}
                        />
                      )}
                      {snippet && (
                        <details className="mb-2">
                          <summary style={{ cursor: 'pointer' }}>{snippetPreview}</summary>
                          <div className="mt-2 small" style={{ whiteSpace: 'pre-wrap' }}>{snippet}</div>
                        </details>
                      )}
                      {item.metadata && Object.keys(item.metadata).length > 0 && (
                      <ul className="list-inline small text-muted mb-0">
                        {Object.entries(item.metadata).map(([key, value]) => {
                          if (key === 'screenshot_path' && typeof value === 'string' && value) {
                            return (
                              <li key={key} className="list-inline-item">
                                <span className="fw-semibold">скриншот</span>:{' '}
                                <a href={buildArtifactUrl(value)} target="_blank" rel="noopener">открыть</a>
                              </li>
                            )
                          }
                          return (
                            <li key={key} className="list-inline-item">
                              <span className="fw-semibold">{key}</span>: {formatMetadataValue(value)}
                            </li>
                          )
                        })}
                      </ul>
                      )}
                    </div>
                    <div className="text-end small" style={{ minWidth: 160 }}>
                      {item.sources.map(src => (
                        <div key={`${item.id}-${src.id}-${src.rank ?? 'ref'}`} className="badge bg-secondary text-wrap mb-1">
                          {src.label || src.id}
                          {typeof src.rank === 'number' ? ` · #${src.rank}` : ''}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </section>
    )
  }

  const renderSourceResults = () => {
    if (!activeJob) return null
    if (!activeJob.sources || activeJob.sources.length === 0) {
      return <div className="alert alert-secondary">Результаты пока отсутствуют</div>
    }
    return activeJob.sources.map(source => {
      const progressInfo = activeJob.progress?.[source.source]
      const label = source.metadata?.label || progressInfo?.label || source.source
      const stats = getSourceStats(source)
      const htmlArtifact = typeof source.metadata?.html_artifact === 'string' ? source.metadata?.html_artifact : null
      const textArtifact = typeof source.metadata?.text_artifact === 'string' ? source.metadata?.text_artifact : null
      const metadataArtifact = typeof source.metadata?.metadata_artifact === 'string' ? source.metadata?.metadata_artifact : null
      return (
        <div key={source.source} className="card shadow-sm mb-3">
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start gap-2 mb-2">
              <div>
                <h3 className="h6 mb-1">{label}</h3>
                <div className="small text-muted">
                  {source.engine ? `Поисковая система: ${source.engine}` : 'Локальный источник'}
                </div>
                {source.error && <div className="text-danger small">{source.error}</div>}
                {source.metadata?.fallback && (
                  <div className="text-warning small d-flex flex-wrap align-items-center gap-2">
                    <span>Поисковая система вернула страницу с проверкой.</span>
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-primary"
                      onClick={() => openCaptchaDialog(
                        source.source,
                        label || source.source,
                        source.metadata?.fallback_url || source.final_url || source.requested_url || null,
                      )}
                    >
                      Повторить
                    </button>
                    {(source.metadata?.fallback_url || source.final_url || source.requested_url) && (
                      <button
                        type="button"
                        className="btn btn-sm btn-outline-secondary"
                        onClick={() => startRemoteBrowser(
                          source.source,
                          source.metadata?.fallback_url || source.final_url || source.requested_url || '',
                          label || source.source,
                        )}
                      >
                        Открыть браузер
                      </button>
                    )}
                    {source.metadata?.fallback_url || source.final_url || source.requested_url ? (
                      <button
                        type="button"
                        className="btn btn-sm btn-outline-secondary"
                        onClick={() => window.open(
                          String(source.metadata?.fallback_url || source.final_url || source.requested_url),
                          '_blank',
                          'noopener',
                        )}
                      >
                        Открыть капчу
                      </button>
                    ) : null}
                  </div>
                )}
                {(stats.pages || stats.parsed || stats.served || stats.forwarded || stats.requested) && (
                  <ul className="list-inline small text-muted mb-2">
                    {stats.pages && <li className="list-inline-item">Страниц: {stats.pages}</li>}
                    {stats.parsed && <li className="list-inline-item">Ссылок с SERP: {stats.parsed}</li>}
                    {stats.served && <li className="list-inline-item">Сохранено: {stats.served}</li>}
                    {stats.forwarded && <li className="list-inline-item">Передано в ответ: {stats.forwarded}</li>}
                    {stats.requested && <li className="list-inline-item">Лимит: {stats.requested}</li>}
                  </ul>
                )}
                {stats.keywords.length > 0 && (
                  <div className="d-flex flex-wrap gap-2 mb-2">
                    {stats.keywords.map(keyword => (
                      <span key={`${source.source}-keyword-${keyword}`} className="badge bg-light text-muted border">
                        {keyword}
                      </span>
                    ))}
                  </div>
                )}
                {stats.requested && stats.forwarded !== null && stats.forwarded < stats.requested && (
                  <div className="text-warning small mb-2">
                    Ссылок меньше лимита — вероятно, поисковик показал капчу или ограничил выдачу.
                  </div>
                )}
                {(htmlArtifact || textArtifact || metadataArtifact || source.screenshot_path) && (
                  <div className="d-flex flex-wrap gap-3 small text-muted mt-2">
                    {htmlArtifact && (
                      <a href={buildArtifactUrl(htmlArtifact)} target="_blank" rel="noopener">
                        HTML-снимок
                      </a>
                    )}
                    {textArtifact && (
                      <a href={buildArtifactUrl(textArtifact)} target="_blank" rel="noopener">
                        Текстовое извлечение
                      </a>
                    )}
                    {metadataArtifact && (
                      <a href={buildArtifactUrl(metadataArtifact)} target="_blank" rel="noopener">
                        JSON метаданных
                      </a>
                    )}
                    {source.screenshot_path && (
                      <a href={buildArtifactUrl(source.screenshot_path)} target="_blank" rel="noopener">
                        Скриншот
                      </a>
                    )}
                  </div>
                )}
              </div>
              <span className={statusBadgeClass(source.status)}>{source.status}</span>
            </div>
          {source.text_content && (
            <details className="mt-2">
              <summary className="small" style={{ cursor: 'pointer' }}>Текстовое извлечение</summary>
              <div className="mt-2 small" style={{ whiteSpace: 'pre-wrap' }}>{source.text_content}</div>
            </details>
          )}
          {source.screenshot_path && (
            <div className="mt-3">
              <img
                src={buildArtifactUrl(source.screenshot_path)}
                alt={`Скриншот ${label || source.source}`}
                className="img-fluid rounded border"
                loading="lazy"
              />
            </div>
          )}
          {source.results.length === 0 && (
            <div className="text-muted">Совпадений не найдено</div>
          )}
          {source.results.length > 0 && (
            <div className="list-group">
              {source.results.map(item => (
                <div key={`${source.source}-${item.rank}-${item.url}`} className="list-group-item">
                  <div className="d-flex align-items-center gap-2 mb-1">
                    <span className="badge bg-primary">{item.rank}</span>
                    <a href={item.url} target="_blank" rel="noopener" className="fw-semibold">
                      {item.title || item.url}
                    </a>
                  </div>
                  {(() => {
                    const highlightSource = item.highlight || (item.metadata?.highlight as string | undefined)
                    if (!highlightSource || highlightSource.trim().length === 0) {
                      return null
                    }
                    return (
                      <div
                        className="small text-body-secondary mb-2"
                        style={{ whiteSpace: 'pre-wrap' }}
                        dangerouslySetInnerHTML={{ __html: highlightSource }}
                      />
                    )
                  })()}
                  {item.snippet && !(item.highlight || item.metadata?.highlight) && <p className="mb-2">{item.snippet}</p>}
                  {item.metadata && Object.keys(item.metadata).length > 0 && (
                    <ul className="list-inline small mb-0 text-muted">
                      {Object.entries(item.metadata)
                        .filter(([key]) => key !== 'highlight')
                        .map(([key, value]) => {
                          if (key === 'screenshot_path' && typeof value === 'string' && value) {
                            return (
                              <li key={key} className="list-inline-item">
                                <span className="fw-semibold">скриншот</span>:{' '}
                                <a href={buildArtifactUrl(value)} target="_blank" rel="noopener">открыть</a>
                              </li>
                            )
                          }
                          return (
                            <li key={key} className="list-inline-item">
                              <span className="fw-semibold">{key}</span>: {Array.isArray(value) ? value.join(', ') : String(value)}
                            </li>
                          )
                        })}
                    </ul>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      )
    })
  }

  return (
    <div className="container-fluid osint-page-glass">
      <div className="d-flex align-items-center justify-content-between flex-wrap gap-2 mb-4">
        <div>
          <h1 className="h3 mb-1">OSINT-поиск</h1>
      
        </div>
      </div>

      <div className="row g-4">
        <div className="col-12 col-lg-7 col-xl-8">
          <section className="card shadow-sm">
            <div className="card-body">
              <h2 className="h5 mb-3">Новый запрос</h2>
              <form className="d-grid gap-3" onSubmit={handleSubmit}>
                <div>
                  <label htmlFor="osint-query" className="form-label fw-semibold">Запрос</label>
                  <textarea
                    id="osint-query"
                    className="form-control"
                    rows={3}
                    value={query}
                    onChange={event => setQuery(event.target.value)}
                    disabled={submitting}
                    placeholder="Например: финансирование университетских исследований 2024 site:.gov"
                  />
                </div>
                <div className="row g-3">
                  <div className="col-12 col-md-4">
                    <label className="form-label fw-semibold">Локаль</label>
                    <div className="d-flex flex-wrap gap-3">
                      {LOCALE_OPTIONS.map(option => (
                        <div className="form-check" key={option}>
                          <input
                            id={`osint-locale-${option}`}
                            type="checkbox"
                            className="form-check-input"
                            checked={selectedLocales.includes(option)}
                            onChange={() => toggleLocale(option)}
                            disabled={submitting}
                          />
                          <label className="form-check-label" htmlFor={`osint-locale-${option}`}>
                            {option}
                          </label>
                        </div>
                      ))}
                    </div>
                    <div className="form-text">Можно выбрать сразу несколько локалей; минимум одна.</div>
                  </div>
                  <div className="col-12 col-md-4">
                    <label htmlFor="osint-max-results" className="form-label fw-semibold">Максимум результатов (поиск)</label>
                    <input
                      id="osint-max-results"
                      type="number"
                      min={1}
                      max={50}
                      className="form-control"
                      value={maxResults}
                      onChange={event => setMaxResults(event.target.value)}
                      disabled={submitting}
                      placeholder="10"
                    />
                  </div>
                  <div className="col-12 col-md-4 d-flex align-items-end gap-3">
                    <div className="form-check">
                      <input
                        id="osint-safe"
                        type="checkbox"
                        className="form-check-input"
                        checked={safe}
                        onChange={event => setSafe(event.target.checked)}
                        disabled={submitting}
                      />
                      <label className="form-check-label" htmlFor="osint-safe">Safe search</label>
                    </div>
                  </div>
                </div>

                <div className="d-flex flex-wrap gap-4 align-items-start">
                  <div className="flex-grow-1 min-w-0">
                    <div className="fw-semibold mb-1">Внешние источники</div>
                    <div className="d-flex flex-wrap gap-3">
                      <div className="form-check">
                        <input
                          id="src-google"
                          type="checkbox"
                          className="form-check-input"
                          checked={useGoogle}
                          onChange={event => setUseGoogle(event.target.checked)}
                          disabled={submitting}
                        />
                        <label className="form-check-label" htmlFor="src-google">Google</label>
                      </div>
                      <div className="form-check">
                        <input
                          id="src-yandex"
                          type="checkbox"
                          className="form-check-input"
                          checked={useYandex}
                          onChange={event => setUseYandex(event.target.checked)}
                          disabled={submitting}
                        />
                        <label className="form-check-label" htmlFor="src-yandex">Yandex</label>
                      </div>
                    </div>
                  </div>
                  <div className="flex-grow-1 min-w-0">
                    <div className="fw-semibold mb-1">Локальные источники</div>
                    <div className="d-flex flex-wrap align-items-center gap-3">
                      <div className="form-check">
                        <input
                          id="src-local-catalog"
                          type="checkbox"
                          className="form-check-input"
                          checked={useLocalCatalogue}
                          onChange={event => setUseLocalCatalogue(event.target.checked)}
                          disabled={submitting}
                        />
                        <label className="form-check-label" htmlFor="src-local-catalog">Каталог Agregator</label>
                      </div>
                      <div className="form-check">
                        <input
                          id="src-local-path"
                          type="checkbox"
                          className="form-check-input"
                          checked={useLocalPath}
                          onChange={event => setUseLocalPath(event.target.checked)}
                          disabled={submitting}
                        />
                        <label className="form-check-label" htmlFor="src-local-path">Папка / сетевой ресурс</label>
                      </div>
                      {useLocalPath && (
                        <div className="w-100 ms-2 mt-2">
                          <label htmlFor="local-path" className="form-label">Путь к папке</label>
                          <div className="input-group">
                            <input
                              id="local-path"
                              className="form-control"
                              value={localPath}
                              onChange={event => setLocalPath(event.target.value)}
                              disabled={submitting}
                              placeholder="/Users/ivan/Documents или C:\\data (только абсолютный путь)"
                            />
                            <button
                              className="btn btn-outline-secondary"
                              type="button"
                              onClick={handleFolderSelect}
                              disabled={submitting}
                            >
                              Выбрать папку
                            </button>
                          </div>
                          <div className="form-text">
                            Укажите абсолютный путь (можно с ~/ для домашней папки).
                          </div>
                          <input
                            ref={folderInputRef}
                            type="file"
                            style={{ display: 'none' }}
                            multiple
                            onChange={handleFolderInputChange}
                            {...({ webkitdirectory: '' as any, directory: '' as any })}
                          />
                          <div className="form-check mt-2">
                            <input
                              id="local-recursive"
                              type="checkbox"
                              className="form-check-input"
                              checked={localRecursive}
                              onChange={event => setLocalRecursive(event.target.checked)}
                              disabled={submitting}
                            />
                            <label className="form-check-label" htmlFor="local-recursive">Рекурсивный поиск</label>
                          </div>
                        </div>
                      )}
                      {(useLocalCatalogue || useLocalPath) && (
                        <div className="w-100 ms-2 mt-2">
                          <label htmlFor="local-limit" className="form-label">Лимит результатов (локальные источники)</label>
                          <input
                            id="local-limit"
                            type="number"
                            min={1}
                            max={200}
                            className="form-control"
                            value={localLimit}
                            onChange={event => setLocalLimit(event.target.value)}
                            disabled={submitting}
                            placeholder="20"
                          />
                        </div>
                      )}
                      {useLocalPath && (
                        <div className="w-100 ms-2 mt-2">
                          <label htmlFor="local-exclude" className="form-label">Исключить файлы (маски через запятую)</label>
                          <input
                            id="local-exclude"
                            className="form-control"
                            value={localExclude}
                            onChange={event => setLocalExclude(event.target.value)}
                            disabled={submitting}
                            placeholder="*.tmp, */cache/*"
                          />
                        </div>
                      )}
                      {useLocalPath && (
                        <div className="form-check ms-2 mt-2">
                          <input
                            id="local-ocr"
                            type="checkbox"
                            className="form-check-input"
                            checked={localOcr}
                            onChange={event => setLocalOcr(event.target.checked)}
                            disabled={submitting}
                          />
                          <label className="form-check-label" htmlFor="local-ocr">Включить OCR для PDF</label>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                <div className="d-flex flex-wrap gap-3">
                  <div className="form-check">
                    <input
                      id="include-html"
                      type="checkbox"
                      className="form-check-input"
                      checked={includeHtml}
                      onChange={event => setIncludeHtml(event.target.checked)}
                      disabled={submitting}
                    />
                    <label className="form-check-label" htmlFor="include-html">Сохранять HTML-слепок</label>
                  </div>
                  <div className="form-check">
                    <input
                      id="include-llm"
                      type="checkbox"
                      className="form-check-input"
                      checked={includeLlm}
                      onChange={event => setIncludeLlm(event.target.checked)}
                      disabled={submitting}
                    />
                    <label className="form-check-label" htmlFor="include-llm">Сохранять сырой ответ LLM</label>
                  </div>
                  <div className="form-check">
                    <input
                      id="build-ontology"
                      type="checkbox"
                      className="form-check-input"
                      checked={buildOntology}
                      onChange={event => setBuildOntology(event.target.checked)}
                      disabled={submitting}
                    />
                    <label className="form-check-label" htmlFor="build-ontology">Построить онтологию</label>
                  </div>
                </div>

                <div className="card border-0 bg-light">
                  <div className="card-body">
                    <div className="form-check form-switch mb-3">
                      <input
                        id="schedule-enabled"
                        type="checkbox"
                        className="form-check-input"
                        checked={scheduleEnabled}
                        onChange={event => setScheduleEnabled(event.target.checked)}
                        disabled={submitting}
                      />
                      <label className="form-check-label" htmlFor="schedule-enabled">Запускать по расписанию</label>
                    </div>
                    {scheduleEnabled && (
                      <div className="row g-3">
                        <div className="col-12 col-md-4">
                          <label htmlFor="schedule-interval" className="form-label">Интервал (минуты)</label>
                          <input
                            id="schedule-interval"
                            type="number"
                            min={5}
                            className="form-control"
                            value={scheduleInterval}
                            onChange={event => setScheduleInterval(event.target.value)}
                            disabled={submitting}
                          />
                        </div>
                        <div className="col-12 col-md-4">
                          <label htmlFor="schedule-start" className="form-label">Начать с</label>
                          <input
                            id="schedule-start"
                            type="datetime-local"
                            className="form-control"
                            value={scheduleStart}
                            onChange={event => setScheduleStart(event.target.value)}
                            disabled={submitting}
                          />
                        </div>
                        <div className="col-12 col-md-4">
                          <label htmlFor="schedule-label" className="form-label">Метка</label>
                          <input
                            id="schedule-label"
                            className="form-control"
                            value={scheduleLabel}
                            onChange={event => setScheduleLabel(event.target.value)}
                            disabled={submitting}
                            placeholder="Например: Ежедневно"
                          />
                        </div>
                        <div className="col-12">
                          <div className="form-check">
                            <input
                              id="schedule-notify"
                              type="checkbox"
                              className="form-check-input"
                              checked={scheduleNotify}
                              onChange={event => setScheduleNotify(event.target.checked)}
                              disabled={submitting}
                            />
                            <label className="form-check-label" htmlFor="schedule-notify">Уведомлять после выполнения</label>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {error && <div className="alert alert-danger mb-0">{error}</div>}

                <div className="d-flex align-items-center gap-3">
                  <button type="submit" className="btn btn-primary" disabled={submitting}>
                    {submitting ? 'Запуск…' : 'Запустить поиск'}
                  </button>
                  <button
                    type="button"
                    className="btn btn-outline-secondary"
                    onClick={() => {
                      setQuery('')
                      setError(null)
                      setActiveJob(null)
                      cancelPolling()
                    }}
                    disabled={submitting}
                  >
                    Очистить
                  </button>
                </div>
              </form>
            </div>
          </section>

          {activeJob && (
            <section className="card shadow-sm mt-4">
              <div className="card-body">
                <div className="d-flex justify-content-between align-items-center mb-3">
                  <div>
                    <h2 className="h5 mb-1">Текущая задача</h2>
                    <div className="text-muted small">
                      Статус: {activeJob.status}
                      {activeJob.completed_at ? ` · завершена: ${formatTimestamp(activeJob.completed_at)}` : ''}
                    </div>
                  </div>
                  <div className="d-flex align-items-center gap-2 flex-wrap">
                    <div className="btn-group">
                      <button
                        type="button"
                        className="btn btn-sm btn-outline-secondary"
                        onClick={() => handleExportJob(activeJob.id, 'markdown')}
                        disabled={exporting}
                      >
                        {exporting ? 'Экспорт…' : 'Markdown'}
                      </button>
                      <button
                        type="button"
                        className="btn btn-sm btn-outline-secondary"
                        onClick={() => handleExportJob(activeJob.id, 'json')}
                        disabled={exporting}
                      >
                        {exporting ? '...' : 'JSON'}
                      </button>
                    </div>
                    <span className={statusBadgeClass(activeJob.status)}>{activeJob.status}</span>
                  </div>
                </div>
                <div className="progress mb-3" role="progressbar" aria-valuenow={progressPercent} aria-valuemin={0} aria-valuemax={100}
                  style={{ height: 10 }}>
                  <div className="progress-bar" style={{ width: `${progressPercent}%` }} />
                </div>
                {renderJobSummary()}
                {renderCaptchaNotice()}
                {renderSourceProgress()}
                {renderScheduleInfo()}
                {renderAnalysis()}
                {renderLlmContext()}
                {renderOntology()}
                {renderCombinedResults()}
                {renderSourceResults()}
              </div>
            </section>
          )}
        </div>

        <div className="col-12 col-lg-5 col-xl-4">
          <OsintHistoryPanel
            history={history}
            activeJobId={activeJob?.id}
            onRefresh={loadHistory}
            onSelect={(job) => {
              setActiveJob(job)
              pollJob(job.id)
            }}
            onDelete={handleDeleteJob}
            statusBadgeClass={statusBadgeClass}
            formatTimestamp={formatTimestamp}
            formatInterval={formatInterval}
          />
        </div>
      </div>
      {renderCaptchaModal()}
      {renderRemoteBrowserModal()}
    </div>
  )
}
