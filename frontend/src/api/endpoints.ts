/**
 * Typed API endpoint functions for the Agregator backend.
 *
 * Each function wraps `apiFetch` with proper typing for both request and
 * response payloads.  Components should import from here instead of calling
 * `fetch()` directly.
 */

import { apiFetch, type ApiFetchOptions } from './client'

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

export interface AuthUser {
  id: number
  username: string
  role: string
  full_name: string | null
  aiword_access: boolean
  can_upload: boolean
  permissions: {
    can_admin: boolean
    can_edit: boolean
    can_view: boolean
  }
}

export interface LoginResponse {
  ok: boolean
  user?: AuthUser
  error?: string
}

export function login(username: string, password: string): Promise<LoginResponse> {
  return apiFetch('/api/auth/login', {
    method: 'POST',
    json: { username, password },
  })
}

export function logout(): Promise<{ ok: boolean }> {
  return apiFetch('/api/auth/logout', { method: 'POST' })
}

export function getMe(): Promise<{ ok: boolean; user?: AuthUser }> {
  return apiFetch('/api/auth/me')
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

export interface FileItem {
  id: number
  title: string
  author: string
  filename: string
  material_type: string
  year: string | null
  size: number
  collection_id: number | null
  tags: Array<{ key: string; value: string }>
  [key: string]: unknown
}

export interface FilesListResponse {
  files: FileItem[]
  total: number
  page: number
  pages: number
}

export function listFiles(params?: URLSearchParams): Promise<FilesListResponse> {
  const qs = params ? `?${params}` : ''
  return apiFetch(`/api/files${qs}`)
}

export function getFile(id: number): Promise<FileItem> {
  return apiFetch(`/api/files/${id}`)
}

export function updateFile(id: number, data: Partial<FileItem>): Promise<FileItem> {
  return apiFetch(`/api/files/${id}`, { method: 'PUT', json: data })
}

export function deleteFile(id: number): Promise<{ ok: boolean }> {
  return apiFetch(`/api/files/${id}`, { method: 'DELETE' })
}

// ---------------------------------------------------------------------------
// Collections
// ---------------------------------------------------------------------------

export interface Collection {
  id: number
  name: string
  slug: string
  file_count: number
  owner_id: number | null
  is_private: boolean
}

export function listCollections(): Promise<Collection[]> {
  return apiFetch('/api/collections')
}

export function createCollection(name: string): Promise<Collection> {
  return apiFetch('/api/collections', { method: 'POST', json: { name } })
}

export function deleteCollection(id: number): Promise<{ ok: boolean }> {
  return apiFetch(`/api/collections/${id}`, { method: 'DELETE' })
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

export interface SearchResponse {
  files: FileItem[]
  total: number
  query: string
}

export function search(params: URLSearchParams): Promise<SearchResponse> {
  return apiFetch(`/api/search?${params}`)
}

// ---------------------------------------------------------------------------
// AI Search
// ---------------------------------------------------------------------------

export interface AiSearchPayload {
  query: string
  stream?: boolean
  top_k?: number
  deep_search?: boolean
  use_tags?: boolean
  use_text?: boolean
  use_rag?: boolean
  full_text?: boolean
  max_candidates?: number
  chunk_chars?: number
  max_chunks?: number
  max_snippets?: number
  use_llm_snippets?: boolean
  all_languages?: boolean
}

export interface AiSearchResult {
  ok: boolean
  answer?: string
  keywords?: string[]
  sources?: Array<{
    id: number
    title: string
    score: number
    snippet: string
    [key: string]: unknown
  }>
  query_hash?: string
  rag_context?: unknown[]
  [key: string]: unknown
}

export function aiSearch(payload: AiSearchPayload): Promise<AiSearchResult> {
  return apiFetch('/api/ai-search', { method: 'POST', json: payload })
}

/**
 * Stream AI search results (returns raw Response for NDJSON parsing).
 */
export function aiSearchStream(
  payload: AiSearchPayload,
  signal?: AbortSignal,
): Promise<Response> {
  return apiFetch('/api/ai-search', {
    method: 'POST',
    json: { ...payload, stream: true },
    raw: true,
    signal,
  }) as Promise<Response>
}

// ---------------------------------------------------------------------------
// AI Search Feedback
// ---------------------------------------------------------------------------

export function submitAiSearchFeedback(data: {
  file_id: number
  query_hash: string
  relevant: boolean
}): Promise<{ ok: boolean }> {
  return apiFetch('/api/ai-search/feedback', { method: 'POST', json: data })
}

export function submitSearchFeedback(data: {
  file_id: number
  query: string
  relevant: boolean
}): Promise<{ ok: boolean }> {
  return apiFetch('/api/search/feedback', { method: 'POST', json: data })
}

// ---------------------------------------------------------------------------
// Doc Chat
// ---------------------------------------------------------------------------

export function docChatDocuments(): Promise<{ ok: boolean; documents: unknown[] }> {
  return apiFetch('/api/doc-chat/documents')
}

export function docChatPrepare(fileIds: number[]): Promise<{ ok: boolean; session_id: string }> {
  return apiFetch('/api/doc-chat/prepare', { method: 'POST', json: { file_ids: fileIds } })
}

export function docChatAsk(data: {
  question: string
  session_id?: string
  mode?: string
}): Promise<{ ok: boolean; answer: string; [key: string]: unknown }> {
  return apiFetch('/api/doc-chat/ask', { method: 'POST', json: data })
}

export function docChatClear(): Promise<{ ok: boolean }> {
  return apiFetch('/api/doc-chat/clear', { method: 'POST', json: {} })
}

export function docChatPreferences(
  update?: Record<string, string>,
): Promise<Record<string, string>> {
  if (update) {
    return apiFetch('/api/doc-chat/preferences', { method: 'POST', json: update })
  }
  return apiFetch('/api/doc-chat/preferences')
}

// ---------------------------------------------------------------------------
// Facets & Material Types
// ---------------------------------------------------------------------------

export interface FacetData {
  types?: Array<{ key: string; label: string; count: number }>
  tags?: Array<{ key: string; value: string; count: number }>
  authors?: Array<{ name: string; count: number }>
  years?: Array<{ year: string; count: number }>
}

export function getFacets(params?: URLSearchParams): Promise<FacetData> {
  const qs = params ? `?${params}` : ''
  return apiFetch(`/api/facets${qs}`)
}

export function getMaterialTypes(): Promise<Array<{ key: string; label: string }>> {
  return apiFetch('/api/material-types')
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

export function getSettings(): Promise<Record<string, unknown>> {
  return apiFetch('/api/settings')
}

export function updateSettings(data: Record<string, unknown>): Promise<{ ok: boolean }> {
  return apiFetch('/api/settings', { method: 'POST', json: data })
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export function healthCheck(): Promise<{ status: string }> {
  return apiFetch('/health')
}
