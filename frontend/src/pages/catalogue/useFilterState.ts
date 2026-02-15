/**
 * Facet filtering state and tag selection management.
 *
 * Extracted from `useCatalogueState` to isolate facet loading, tag
 * selection, and collection management.
 */

import { useCallback, useEffect, useState } from 'react'
import type { CollectionOption, FacetData } from './types'

export interface FilterState {
  facets: FacetData | null
  facetsLoading: boolean
  selectedTags: string[]
  setSelectedTags: React.Dispatch<React.SetStateAction<string[]>>
  collections: CollectionOption[]
  removeTag: (tag: string) => void
  addTagFilter: (tag: string) => void
  loadFacets: (params: URLSearchParams) => Promise<void>
}

export function useFilterState(
  sp: URLSearchParams,
  setSp: (updater: (prev: URLSearchParams) => URLSearchParams) => void,
): FilterState {
  const [facets, setFacets] = useState<FacetData | null>(null)
  const [facetsLoading, setFacetsLoading] = useState(false)
  const [selectedTags, setSelectedTags] = useState<string[]>(() => sp.getAll('tag'))
  const [collections, setCollections] = useState<CollectionOption[]>([])

  // Load collections once
  useEffect(() => {
    fetch('/api/collections')
      .then(r => r.json())
      .then(data => {
        const cols = Array.isArray(data) ? data : data.collections || data.items || []
        setCollections(
          cols.map((c: any) => ({
            id: c.id,
            name: c.name,
            slug: c.slug,
            count: c.file_count || c.count || 0,
          })),
        )
      })
      .catch(() => setCollections([]))
  }, [])

  const removeTag = useCallback(
    (tag: string) => {
      setSelectedTags(prev => prev.filter(t => t !== tag))
      setSp((prev: URLSearchParams) => {
        const next = new URLSearchParams(prev)
        const tags = next.getAll('tag').filter(t => t !== tag)
        next.delete('tag')
        tags.forEach(t => next.append('tag', t))
        return next
      })
    },
    [setSp],
  )

  const addTagFilter = useCallback(
    (tag: string) => {
      setSelectedTags(prev => {
        if (prev.includes(tag)) return prev
        return [...prev, tag]
      })
      setSp((prev: URLSearchParams) => {
        const next = new URLSearchParams(prev)
        if (!next.getAll('tag').includes(tag)) {
          next.append('tag', tag)
        }
        return next
      })
    },
    [setSp],
  )

  const loadFacets = useCallback(async (params: URLSearchParams) => {
    setFacetsLoading(true)
    try {
      const qs = new URLSearchParams(params)
      qs.set('context', 'search')
      const resp = await fetch(`/api/facets?${qs}`)
      if (resp.ok) {
        const data = await resp.json()
        setFacets(data)
      }
    } catch {
      // silently ignore
    } finally {
      setFacetsLoading(false)
    }
  }, [])

  return {
    facets,
    facetsLoading,
    selectedTags,
    setSelectedTags,
    collections,
    removeTag,
    addTagFilter,
    loadFacets,
  }
}
