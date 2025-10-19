export type Tag = { key: string; value: string }

export type FileItem = {
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

export type CollectionOption = { id: number; name: string }

export type FacetSuggestion = {
  kind: 'author' | 'year' | 'tag'
  value: string
  label: string
  key?: string
}

export type FacetData = {
  types: [string | null, number][]
  tag_facets: Record<string, [string, number][]>
  authors: [string, number][]
  years: [string, number][]
  suggestions: FacetSuggestion[]
  include_types?: boolean
  allowed_keys?: string[] | null
}

export type ProgressItem = {
  id: string
  line: string
  icon: string
}

export type RagContextEntry = {
  doc_id: number
  chunk_id: number
  title: string
  language: string
  section_path?: string | null
  translation_hint?: string
  score_dense?: number
  score_sparse?: number
  combined_score?: number
  reasoning_hint?: string
  token_estimate?: number
  preview?: string
  content?: string
  url?: string | null
  extra?: Record<string, unknown>
}

export type RagSourceEntry = {
  doc_id: number
  chunk_id: number
  title: string
  section_path?: string | null
  combined_score?: number
}

export type RagRiskSection = {
  doc_id: number
  chunk_id: number
  combined_score?: number
  reasoning_hint?: string
}

export type RagRisk = {
  score?: number
  level?: string
  reasons?: string[]
  flagged_refs?: { doc_id: number; chunk_id: number }[]
  hallucination_warning?: boolean
  top_sections?: RagRiskSection[]
}

export type RagSearchResponse = {
  rag_retry?: boolean
}

export type RagValidationResult = {
  is_empty?: boolean
  missing_citations?: boolean
  unknown_citations?: [number, number][]
  extra_citations?: [number, number][]
  hallucination_warning?: boolean
  facts_with_issues?: string[]
}
