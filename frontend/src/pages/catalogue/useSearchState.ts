/**
 * Search parameters and URL state management.
 *
 * Extracted from `useCatalogueState` to isolate search query, filters,
 * debouncing, and URL <-> state synchronization logic.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

// ---------------------------------------------------------------------------
// Debounce hook
// ---------------------------------------------------------------------------

export function useDebounced<T>(value: T, delay = 400): T {
  const [state, setState] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setState(value), delay)
    return () => clearTimeout(id)
  }, [value, delay])
  return state
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SearchParams {
  sp: URLSearchParams
  setSp: ReturnType<typeof useSearchParams>[1]
  q: string
  type: string
  year_from: string
  year_to: string
  size_min: string
  size_max: string
  metadataQuality: string
  collectionId: string
  dq: string
  params: URLSearchParams
  commit: string
}

export interface PaginationState {
  page: number
  pages: number
  perPage: number
  offset: number
  localPage: number
  canLoadMore: boolean
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useSearchState() {
  const [sp, setSp] = useSearchParams()
  const q = sp.get('q') || ''
  const type = sp.get('type') || ''
  const year_from = sp.get('year_from') || ''
  const year_to = sp.get('year_to') || ''
  const size_min = sp.get('size_min') || ''
  const size_max = sp.get('size_max') || ''
  const metadataQuality = sp.get('metadata_quality') || ''
  const collectionId = sp.get('collection_id') || ''
  const page = Math.max(parseInt(sp.get('page') || '1'), 1)
  const commit = sp.get('commit') || ''
  const perPage = 50
  const offset = (page - 1) * perPage

  const dq = useDebounced(q, 350)

  const [localPage, setLocalPage] = useState(page)

  const params = useMemo(() => {
    const p = new URLSearchParams()
    if (dq) p.set('q', dq)
    if (dq) p.set('smart', '1')
    if (type) p.set('type', type)
    if (year_from) p.set('year_from', year_from)
    if (year_to) p.set('year_to', year_to)
    if (size_min) p.set('size_min', size_min)
    if (size_max) p.set('size_max', size_max)
    if (metadataQuality) p.set('metadata_quality', metadataQuality)
    if (collectionId) p.set('collection_id', collectionId)
    return p
  }, [dq, type, year_from, year_to, size_min, size_max, metadataQuality, collectionId])

  const searchParams: SearchParams = {
    sp,
    setSp,
    q,
    type,
    year_from,
    year_to,
    size_min,
    size_max,
    metadataQuality,
    collectionId,
    dq,
    params,
    commit,
  }

  return {
    searchParams,
    page,
    perPage,
    offset,
    localPage,
    setLocalPage,
  }
}
