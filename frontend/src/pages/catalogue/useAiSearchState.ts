/**
 * AI search state management — streaming responses, RAG context, feedback.
 *
 * Extracted from `useCatalogueState` to isolate the AI-specific state:
 * model toggles, streaming NDJSON handling, RAG context display, keyword
 * feedback, and text-to-speech.
 */

import { useCallback, useRef, useState } from 'react'
import type { RagContextEntry, RagContextGroup, RagRisk, RagSourceEntry, RagValidationResult } from './types'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AiSearchSettings {
  aiTopK: number
  setAiTopK: (v: number) => void
  aiDeepSearch: boolean
  setAiDeepSearch: (v: boolean) => void
  aiUseTags: boolean
  setAiUseTags: (v: boolean) => void
  aiUseText: boolean
  setAiUseText: (v: boolean) => void
  aiMaxCandidates: number
  setAiMaxCandidates: (v: number) => void
  aiChunkChars: number
  setAiChunkChars: (v: number) => void
  aiMaxChunks: number
  setAiMaxChunks: (v: number) => void
  aiMaxSnippets: number
  setAiMaxSnippets: (v: number) => void
  aiFullText: boolean
  setAiFullText: (v: boolean) => void
  aiUseLlmSnippets: boolean
  setAiUseLlmSnippets: (v: boolean) => void
  aiUseRag: boolean
  setAiUseRag: (v: boolean) => void
  aiAllLanguages: boolean
  setAiAllLanguages: (v: boolean) => void
  showAiSettings: boolean
  setShowAiSettings: (v: boolean) => void
}

export interface AiSearchResults {
  aiLoading: boolean
  aiProgress: string[]
  aiAnswer: string
  aiKeywords: string[]
  aiSources: any[]
  aiFilteredKeywords: string[]
  aiQueryHash: string
  feedbackStatus: Record<string, string>
  ragContext: RagContextEntry[]
  ragContextGroups: RagContextGroup[]
  ragValidation: RagValidationResult | null
  ragRisk: RagRisk | null
  ragSources: RagSourceEntry[]
  ragRetry: boolean
  ragWarnings: string[]
  ragNotes: string[]
  ragFallback: boolean
  ragSessionId: number | null
  ragHintVisible: boolean
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useAiSearchState() {
  const [aiMode, setAiMode] = useState(false)
  const [aiLoading, setAiLoading] = useState(false)
  const [aiProgress, setAiProgress] = useState<string[]>([])
  const [aiTopK, setAiTopK] = useState(3)
  const [aiDeepSearch, setAiDeepSearch] = useState(true)
  const [aiUseTags, setAiUseTags] = useState(true)
  const [aiUseText, setAiUseText] = useState(true)
  const [aiMaxCandidates, setAiMaxCandidates] = useState(15)
  const [aiChunkChars, setAiChunkChars] = useState(5000)
  const [aiMaxChunks, setAiMaxChunks] = useState(40)
  const [aiMaxSnippets, setAiMaxSnippets] = useState(3)
  const [aiFullText, setAiFullText] = useState(false)
  const [aiUseLlmSnippets, setAiUseLlmSnippets] = useState(false)
  const [aiUseRag, setAiUseRag] = useState(() => {
    if (typeof window === 'undefined') return false
    try {
      return localStorage.getItem('catalogue.useRag') === '1'
    } catch {
      return false
    }
  })
  const [aiAllLanguages, setAiAllLanguages] = useState(false)
  const [showAiSettings, setShowAiSettings] = useState(false)

  // Results state
  const [aiAnswer, setAiAnswer] = useState('')
  const [aiKeywords, setAiKeywords] = useState<string[]>([])
  const [aiSources, setAiSources] = useState<any[]>([])
  const [aiFilteredKeywords, setAiFilteredKeywords] = useState<string[]>([])
  const [aiQueryHash, setAiQueryHash] = useState('')
  const [feedbackStatus, setFeedbackStatus] = useState<Record<string, string>>({})

  // RAG context
  const [ragContext, setRagContext] = useState<RagContextEntry[]>([])
  const [ragContextGroups, setRagContextGroups] = useState<RagContextGroup[]>([])
  const [ragValidation, setRagValidation] = useState<RagValidationResult | null>(null)
  const [ragRisk, setRagRisk] = useState<RagRisk | null>(null)
  const [ragSources, setRagSources] = useState<RagSourceEntry[]>([])
  const [ragRetry, setRagRetry] = useState(false)
  const [ragWarnings, setRagWarnings] = useState<string[]>([])
  const [ragNotes, setRagNotes] = useState<string[]>([])
  const [ragFallback, setRagFallback] = useState(false)
  const [ragSessionId, setRagSessionId] = useState<number | null>(null)
  const [ragHintVisible, setRagHintVisible] = useState(() => {
    if (typeof window === 'undefined') return true
    try {
      return localStorage.getItem('catalogue.ragHintSeen') !== '1'
    } catch {
      return true
    }
  })

  // Speech state
  const [speechSupported] = useState(
    () =>
      typeof window !== 'undefined' &&
      'speechSynthesis' in window &&
      typeof SpeechSynthesisUtterance !== 'undefined',
  )
  const [autoSpeakAnswer, setAutoSpeakAnswer] = useState(() => {
    if (typeof window === 'undefined') return false
    try {
      return localStorage.getItem('catalogue.autoSpeakAnswer') === '1'
    } catch {
      return false
    }
  })
  const [speechState, setSpeechState] = useState<'idle' | 'speaking'>('idle')
  const [speechError, setSpeechError] = useState<string | null>(null)
  const speechUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null)

  const resetAiState = useCallback(() => {
    setAiAnswer('')
    setAiKeywords([])
    setAiSources([])
    setAiFilteredKeywords([])
    setAiQueryHash('')
    setFeedbackStatus({})
    setAiProgress([])
    setRagContext([])
    setRagContextGroups([])
    setRagValidation(null)
    setRagRisk(null)
    setRagSources([])
    setRagRetry(false)
    setRagWarnings([])
    setRagNotes([])
    setRagFallback(false)
    setRagSessionId(null)
  }, [])

  const dismissRagHint = useCallback(() => {
    setRagHintVisible(false)
    try {
      localStorage.setItem('catalogue.ragHintSeen', '1')
    } catch {
      // ignore
    }
  }, [])

  return {
    aiMode,
    setAiMode,
    aiLoading,
    setAiLoading,
    aiProgress,
    setAiProgress,
    settings: {
      aiTopK,
      setAiTopK,
      aiDeepSearch,
      setAiDeepSearch,
      aiUseTags,
      setAiUseTags,
      aiUseText,
      setAiUseText,
      aiMaxCandidates,
      setAiMaxCandidates,
      aiChunkChars,
      setAiChunkChars,
      aiMaxChunks,
      setAiMaxChunks,
      aiMaxSnippets,
      setAiMaxSnippets,
      aiFullText,
      setAiFullText,
      aiUseLlmSnippets,
      setAiUseLlmSnippets,
      aiUseRag,
      setAiUseRag,
      aiAllLanguages,
      setAiAllLanguages,
      showAiSettings,
      setShowAiSettings,
    },
    results: {
      aiAnswer,
      setAiAnswer,
      aiKeywords,
      setAiKeywords,
      aiSources,
      setAiSources,
      aiFilteredKeywords,
      setAiFilteredKeywords,
      aiQueryHash,
      setAiQueryHash,
      feedbackStatus,
      setFeedbackStatus,
    },
    rag: {
      ragContext,
      setRagContext,
      ragContextGroups,
      setRagContextGroups,
      ragValidation,
      setRagValidation,
      ragRisk,
      setRagRisk,
      ragSources,
      setRagSources,
      ragRetry,
      setRagRetry,
      ragWarnings,
      setRagWarnings,
      ragNotes,
      setRagNotes,
      ragFallback,
      setRagFallback,
      ragSessionId,
      setRagSessionId,
      ragHintVisible,
      dismissRagHint,
    },
    speech: {
      speechSupported,
      autoSpeakAnswer,
      setAutoSpeakAnswer,
      speechState,
      setSpeechState,
      speechError,
      setSpeechError,
      speechUtteranceRef,
    },
    resetAiState,
  }
}
