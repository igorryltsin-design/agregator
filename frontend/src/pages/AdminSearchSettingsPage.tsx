import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useAuth } from '../ui/Auth'
import { useToasts } from '../ui/Toasts'

type SearchTypeOption = { id: string; label: string }
type TagKeyOption = { key: string; count: number }
type SearchConfigPayload = {
  search_types?: string[]
  facet_include_types?: boolean
  facet_tag_keys?: string[] | null
  supported_search_types?: SearchTypeOption[]
  tag_key_options?: TagKeyOption[]
}

type FacetMode = 'all' | 'custom' | 'none'

const defaultSearchTypeOptions: SearchTypeOption[] = [
  { id: 'classic', label: 'Классический поиск' },
  { id: 'ai', label: 'Поиск ИИ' },
]

const defaultConfig = {
  search_types: ['classic', 'ai'],
  facet_include_types: true,
  facet_tag_keys: null as string[] | null,
}

export default function AdminSearchSettingsPage() {
  const { user } = useAuth()
  const toasts = useToasts()
  const isAdmin = user?.role === 'admin'
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  const [searchTypeOptions, setSearchTypeOptions] = useState<SearchTypeOption[]>(defaultSearchTypeOptions)
  const [searchTypes, setSearchTypes] = useState<string[]>(defaultConfig.search_types)
  const [facetIncludeTypes, setFacetIncludeTypes] = useState<boolean>(defaultConfig.facet_include_types)
  const [facetMode, setFacetMode] = useState<FacetMode>('all')
  const [selectedTagKeys, setSelectedTagKeys] = useState<string[]>([])
  const [tagOptions, setTagOptions] = useState<TagKeyOption[]>([])
  const [tagFilter, setTagFilter] = useState('')
  const [newTagKey, setNewTagKey] = useState('')

  const applyConfig = useCallback((config: SearchConfigPayload) => {
    const nextTypes = Array.isArray(config.search_types) && config.search_types.length
      ? config.search_types.map((id: any) => String(id)).filter(Boolean)
      : defaultConfig.search_types
    setSearchTypes(nextTypes)
    setFacetIncludeTypes(Boolean(config.facet_include_types))

    if (config.facet_tag_keys === null) {
      setFacetMode('all')
      setSelectedTagKeys([])
    } else if (Array.isArray(config.facet_tag_keys)) {
      const cleaned = config.facet_tag_keys.map((key: any) => String(key)).filter(Boolean)
      if (cleaned.length) {
        setFacetMode('custom')
        setSelectedTagKeys(cleaned)
      } else {
        setFacetMode('none')
        setSelectedTagKeys([])
      }
    }

    if (Array.isArray(config.supported_search_types) && config.supported_search_types.length) {
      setSearchTypeOptions(config.supported_search_types.map(opt => ({ id: String(opt.id), label: String(opt.label || opt.id) })))
    }
    if (Array.isArray(config.tag_key_options) && config.tag_key_options.length) {
      const sorted = [...config.tag_key_options].sort((a, b) => (b.count || 0) - (a.count || 0))
      setTagOptions(sorted)
    }
  }, [])

  useEffect(() => {
    if (!isAdmin) return
    let cancelled = false
    const load = async () => {
      setLoading(true)
      try {
        const resp = await fetch('/api/search-config?include_meta=1')
        const data = await resp.json().catch(() => ({}))
        if (!cancelled && resp.ok && data?.ok && data.config) {
          applyConfig(data.config as SearchConfigPayload)
        } else if (!cancelled && !resp.ok) {
          toasts.push(data?.error || 'Не удалось загрузить настройки поиска', 'error')
        }
      } catch {
        if (!cancelled) {
          toasts.push('Ошибка соединения при загрузке настроек поиска', 'error')
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }
    load()
    return () => { cancelled = true }
  }, [applyConfig, isAdmin, toasts])

  const toggleSearchType = (id: string) => {
    setSearchTypes(prev => {
      const exists = prev.includes(id)
      if (exists) {
        if (prev.length <= 1) {
          toasts.push('Оставьте хотя бы один тип поиска', 'error')
          return prev
        }
        return prev.filter(x => x !== id)
      }
      return [...prev, id]
    })
  }

  const toggleTagKey = (key: string) => {
    setSelectedTagKeys(prev => prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key])
  }

  const addTagKey = () => {
    const value = newTagKey.trim()
    if (!value) return
    setSelectedTagKeys(prev => prev.includes(value) ? prev : [...prev, value])
    setNewTagKey('')
  }

  const optionsForDisplay = useMemo(() => {
    const map = new Map<string, TagKeyOption>()
    tagOptions.forEach(opt => map.set(opt.key, opt))
    selectedTagKeys.forEach(key => {
      if (!map.has(key)) {
        map.set(key, { key, count: 0 })
      }
    })
    let list = Array.from(map.values())
    const filter = tagFilter.trim().toLowerCase()
    if (filter) {
      list = list.filter(item => item.key.toLowerCase().includes(filter))
    }
    list.sort((a, b) => {
      const diff = (b.count || 0) - (a.count || 0)
      if (diff !== 0) return diff
      return a.key.localeCompare(b.key)
    })
    return list
  }, [tagOptions, selectedTagKeys, tagFilter])

  const handleSave = async () => {
    if (!isAdmin) return
    if (!searchTypes.length) {
      toasts.push('Выберите хотя бы один тип поиска', 'error')
      return
    }
    const payload: { search_types: string[]; facet_include_types: boolean; facet_tag_keys: string[] | null } = {
      search_types: searchTypes,
      facet_include_types: facetIncludeTypes,
      facet_tag_keys: null,
    }
    if (facetMode === 'custom') {
      payload.facet_tag_keys = selectedTagKeys
    } else if (facetMode === 'none') {
      payload.facet_tag_keys = []
    }
  
    setSaving(true)
    try {
      const resp = await fetch('/api/search-config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await resp.json().catch(() => ({}))
      if (resp.ok && data?.ok && data.config) {
        applyConfig(data.config as SearchConfigPayload)
        toasts.push('Настройки поиска обновлены', 'success')
      } else {
        toasts.push(data?.error || 'Не удалось сохранить настройки поиска', 'error')
      }
    } catch {
      toasts.push('Ошибка соединения при сохранении настроек', 'error')
    } finally {
      setSaving(false)
    }
  }

  if (!isAdmin) return <div className="card p-3">Недостаточно прав.</div>

  return (
    <div className="card p-3">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h2 className="h5 m-0">Настройки поиска</h2>
        <button className="btn btn-primary" onClick={handleSave} disabled={saving || loading}>
          {saving ? 'Сохранение…' : 'Сохранить'}
        </button>
      </div>
      {loading ? (
        <div>Загрузка…</div>
      ) : (
        <div className="row g-4">
          <div className="col-12 col-lg-4">
            <div className="border rounded-3 p-3 h-100">
              <div className="fw-semibold mb-2">Типы поиска</div>
              <div className="text-secondary mb-3" style={{ fontSize: '0.9rem' }}>
                Выберите режимы, доступные пользователям. Минимум один вариант должен быть включён.
              </div>
              <div className="d-grid gap-2">
                {searchTypeOptions.map(opt => (
                  <label key={opt.id} className="form-check">
                    <input
                      type="checkbox"
                      className="form-check-input"
                      checked={searchTypes.includes(opt.id)}
                      onChange={() => toggleSearchType(opt.id)}
                    />
                    <span className="form-check-label">{opt.label}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
          <div className="col-12 col-lg-8">
            <div className="border rounded-3 p-3 h-100">
              <div className="fw-semibold mb-2">Фасеты</div>
              <div className="mb-3">
                <label className="form-check form-switch">
                  <input
                    className="form-check-input"
                    type="checkbox"
                    checked={facetIncludeTypes}
                    onChange={e => setFacetIncludeTypes(e.target.checked)}
                  />
                  <span className="form-check-label">Показывать фасет по типам материалов</span>
                </label>
              </div>
              <div className="mb-3">
                <div className="fw-semibold" style={{ fontSize: '0.95rem' }}>Фасеты по тегам</div>
                <div className="d-flex flex-column gap-2 mt-2">
                  <label className="form-check">
                    <input
                      type="radio"
                      className="form-check-input"
                      checked={facetMode === 'all'}
                      onChange={() => setFacetMode('all')}
                    />
                    <span className="form-check-label">Все теги (по умолчанию)</span>
                  </label>
                  <label className="form-check">
                    <input
                      type="radio"
                      className="form-check-input"
                      checked={facetMode === 'custom'}
                      onChange={() => setFacetMode('custom')}
                    />
                    <span className="form-check-label">Только выбранные теги</span>
                  </label>
                  <label className="form-check">
                    <input
                      type="radio"
                      className="form-check-input"
                      checked={facetMode === 'none'}
                      onChange={() => setFacetMode('none')}
                    />
                    <span className="form-check-label">Отключить теговые фасеты</span>
                  </label>
                </div>
              </div>
              {facetMode === 'custom' && (
                <div className="d-flex flex-column gap-3">
                  <div>
                    <label className="form-label">Выбранные теги</label>
                    <div className="d-flex flex-wrap gap-2">
                      {selectedTagKeys.length === 0 && (
                        <span className="text-secondary" style={{ fontSize: '0.9rem' }}>Выберите теги ниже или добавьте вручную.</span>
                      )}
                      {selectedTagKeys.map(key => (
                        <span key={key} className="badge bg-secondary-subtle text-body d-inline-flex align-items-center gap-2">
                          {key}
                          <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => toggleTagKey(key)}>×</button>
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="row g-2">
                    <div className="col-12 col-md-6">
                      <label className="form-label">Фильтр по названию</label>
                      <input className="form-control" value={tagFilter} onChange={e => setTagFilter(e.target.value)} placeholder="Например, author" />
                    </div>
                    <div className="col-12 col-md-6">
                      <label className="form-label">Добавить тег вручную</label>
                      <div className="input-group">
                        <input className="form-control" value={newTagKey} onChange={e => setNewTagKey(e.target.value)} placeholder="tag_key" onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); addTagKey() } }} />
                        <button className="btn btn-outline-secondary" type="button" onClick={addTagKey}>Добавить</button>
                      </div>
                    </div>
                  </div>
                  <div>
                    <div className="fw-semibold mb-2" style={{ fontSize: '0.95rem' }}>Популярные теги</div>
                    <div className="d-grid gap-2" style={{ maxHeight: 260, overflowY: 'auto' }}>
                      {optionsForDisplay.length === 0 && (
                        <div className="text-secondary" style={{ fontSize: '0.9rem' }}>Совпадений не найдено.</div>
                      )}
                      {optionsForDisplay.map(opt => (
                        <label key={opt.key} className="form-check d-flex justify-content-between align-items-center">
                          <span>
                            <input
                              type="checkbox"
                              className="form-check-input me-2"
                              checked={selectedTagKeys.includes(opt.key)}
                              onChange={() => toggleTagKey(opt.key)}
                            />
                            <span className="form-check-label">{opt.key}</span>
                          </span>
                          <span className="text-secondary" style={{ fontSize: '0.85rem' }}>{opt.count}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
