import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useAuth } from '../ui/Auth'
import { useToasts } from '../ui/Toasts'
import { tagKeyRu } from '../utils/locale'

type TagKeyOption = { key: string; count: number; samples?: string[] }
type FacetMode = 'all' | 'custom' | 'none'
type GraphFacetMode = 'all' | 'custom'

type FacetConfigResponse = {
  ok?: boolean
  error?: string
  config?: {
    search?: {
      include_types?: boolean
      tag_keys?: string[] | null
    }
    graph?: {
      tag_keys?: string[] | null
    }
  }
  options?: TagKeyOption[]
}

function toMode(keys: string[] | null | undefined): { mode: FacetMode; list: string[] } {
  if (keys === null || keys === undefined) {
    return { mode: 'all', list: [] }
  }
  if (Array.isArray(keys) && keys.length) {
    return { mode: 'custom', list: keys }
  }
  return { mode: 'none', list: [] }
}

function uniqueKeys(list: string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  list.forEach(item => {
    const key = item.trim()
    if (!key || seen.has(key)) return
    seen.add(key)
    out.push(key)
  })
  return out
}

function extendOptions(options: TagKeyOption[], extra: string[]): TagKeyOption[] {
  const map = new Map<string, TagKeyOption>()
  options.forEach(opt => {
    const key = opt.key.trim()
    if (!key) return
    map.set(key, { key, count: opt.count, samples: Array.isArray(opt.samples) ? opt.samples : [] })
  })
  extra.forEach(key => {
    const trimmed = key.trim()
    if (!trimmed || map.has(trimmed)) return
    map.set(trimmed, { key: trimmed, count: 0, samples: [] })
  })
  return Array.from(map.values())
}

export default function AdminFacetSettingsPage() {
  const { user } = useAuth()
  const toasts = useToasts()
  const isAdmin = user?.role === 'admin'
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  const [includeTypes, setIncludeTypes] = useState(true)
  const [searchMode, setSearchMode] = useState<FacetMode>('all')
  const [graphMode, setGraphMode] = useState<GraphFacetMode>('all')
  const [searchKeys, setSearchKeys] = useState<string[]>([])
  const [graphKeys, setGraphKeys] = useState<string[]>([])
  const [tagOptions, setTagOptions] = useState<TagKeyOption[]>([])
  const [searchFilter, setSearchFilter] = useState('')
  const [graphFilter, setGraphFilter] = useState('')
  const [searchManual, setSearchManual] = useState('')
  const [graphManual, setGraphManual] = useState('')

  const loadConfig = useCallback(async () => {
    if (!isAdmin) return
    setLoading(true)
    try {
      const resp = await fetch('/api/facet-config')
      const data: FacetConfigResponse = await resp.json().catch(() => ({}))
      if (!resp.ok || !data?.ok) {
        throw new Error(data?.error || 'Не удалось загрузить настройки фасетов')
      }
      const searchInfo = toMode(data.config?.search?.tag_keys)
      const graphInfo = toMode(data.config?.graph?.tag_keys)
      setIncludeTypes(data.config?.search?.include_types ?? true)
      setSearchMode(searchInfo.mode)
      setGraphMode(graphInfo.mode === 'none' ? 'all' : graphInfo.mode)
      setSearchKeys(uniqueKeys(searchInfo.list))
      setGraphKeys(uniqueKeys(graphInfo.list))
      const options = Array.isArray(data.options) ? data.options : []
      const normalized = options.map(opt => ({
        key: String(opt.key || '').trim(),
        count: Math.max(0, Number(opt.count || 0)),
        samples: Array.isArray(opt.samples) ? opt.samples.map(v => String(v || '').trim()).filter(Boolean) : [],
      })).filter(opt => opt.key)
      const sorted = [...normalized].sort((a, b) => (b.count || 0) - (a.count || 0))
      setTagOptions(sorted)
    } catch (error: any) {
      toasts.push(String(error?.message || error || 'Ошибка загрузки настроек фасетов'), 'error')
    } finally {
      setLoading(false)
    }
  }, [isAdmin, toasts])

  useEffect(() => {
    loadConfig()
  }, [loadConfig])

  const normalize = useCallback((value: string) => value.normalize('NFD').replace(/[\u0300-\u036f]/g, ''), [])

  const matchesFilter = useCallback((opt: TagKeyOption, term: string) => {
    if (!term) return true
    const key = normalize(opt.key.toLowerCase())
    if (key.includes(term)) return true
    const label = normalize(tagKeyRu(opt.key).toLowerCase())
    if (label.includes(term)) return true
    const samples = Array.isArray(opt.samples) ? opt.samples : []
    return samples.some(sample => normalize(sample.toLowerCase()).includes(term))
  }, [normalize])

  const searchOptionList = useMemo(() => {
    const list = extendOptions(tagOptions, searchKeys)
    const term = normalize(searchFilter.trim().toLowerCase())
    const filtered = term ? list.filter(opt => matchesFilter(opt, term)) : list
    return filtered.sort((a, b) => {
      const diff = (b.count || 0) - (a.count || 0)
      if (diff !== 0) return diff
      return a.key.localeCompare(b.key)
    })
  }, [tagOptions, searchKeys, searchFilter, matchesFilter, normalize])

  const graphOptionList = useMemo(() => {
    const list = extendOptions(tagOptions, graphKeys)
    const term = normalize(graphFilter.trim().toLowerCase())
    const filtered = term ? list.filter(opt => matchesFilter(opt, term)) : list
    return filtered.sort((a, b) => {
      const diff = (b.count || 0) - (a.count || 0)
      if (diff !== 0) return diff
      return a.key.localeCompare(b.key)
    })
  }, [tagOptions, graphKeys, graphFilter, matchesFilter, normalize])

  const toggleSearchKey = (key: string) => {
    setSearchKeys(prev => prev.includes(key) ? prev.filter(item => item !== key) : [...prev, key])
  }

  const toggleGraphKey = (key: string) => {
    setGraphKeys(prev => prev.includes(key) ? prev.filter(item => item !== key) : [...prev, key])
  }

  const addSearchKey = () => {
    const value = searchManual.trim()
    if (!value) return
    setSearchKeys(prev => prev.includes(value) ? prev : [...prev, value])
    setSearchManual('')
    setSearchMode('custom')
  }

  const addGraphKey = () => {
    const value = graphManual.trim()
    if (!value) return
    setGraphKeys(prev => prev.includes(value) ? prev : [...prev, value])
    setGraphManual('')
    setGraphMode('custom')
  }

  const handleSave = async () => {
    if (!isAdmin) return
    if (searchMode === 'custom' && searchKeys.length === 0) {
      toasts.push('Выберите хотя бы один тег для поиска или включите режим "Все теги"', 'error')
      return
    }
    if (graphMode === 'custom' && graphKeys.length === 0) {
      toasts.push('Добавьте минимум один тег для графа или выберите "Все теги"', 'error')
      return
    }
    const payload = {
      search: {
        include_types: includeTypes,
        tag_keys: searchMode === 'all' ? null : (searchMode === 'custom' ? uniqueKeys(searchKeys) : []),
      },
      graph: {
        tag_keys: graphMode === 'all' ? null : uniqueKeys(graphKeys),
      }
    }
    setSaving(true)
    try {
      const resp = await fetch('/api/facet-config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data: FacetConfigResponse = await resp.json().catch(() => ({}))
      if (!resp.ok || !data?.ok) {
        throw new Error(data?.error || 'Не удалось сохранить настройки фасетов')
      }
      const searchInfo = toMode(data.config?.search?.tag_keys)
      const graphInfo = toMode(data.config?.graph?.tag_keys)
      setIncludeTypes(data.config?.search?.include_types ?? true)
      setSearchMode(searchInfo.mode)
      setGraphMode(graphInfo.mode === 'none' ? 'all' : graphInfo.mode)
      setSearchKeys(uniqueKeys(searchInfo.list))
      setGraphKeys(uniqueKeys(graphInfo.list))
      const options = Array.isArray(data.options) ? data.options : []
      const normalized = options.map(opt => ({
        key: String(opt.key || '').trim(),
        count: Math.max(0, Number(opt.count || 0)),
        samples: Array.isArray(opt.samples) ? opt.samples.map(v => String(v || '').trim()).filter(Boolean) : [],
      })).filter(opt => opt.key)
      const sorted = [...normalized].sort((a, b) => (b.count || 0) - (a.count || 0))
      setTagOptions(sorted)
      toasts.push('Настройки фасетов обновлены', 'success')
    } catch (error: any) {
      toasts.push(String(error?.message || error || 'Ошибка сохранения настроек'), 'error')
    } finally {
      setSaving(false)
    }
  }

  if (!isAdmin) {
    return <div className="card p-3">Недостаточно прав.</div>
  }

  return (
    <div className="card p-3 facet-settings">
      <div className="d-flex justify-content-between align-items-center mb-3 flex-wrap gap-2">
        <h2 className="h5 m-0">Настройки фасетов</h2>
        <button className="btn btn-primary" onClick={handleSave} disabled={saving || loading}>
          {saving ? 'Сохранение…' : 'Сохранить'}
        </button>
      </div>
      {loading ? (
        <div>Загрузка…</div>
      ) : (
        <div className="row g-4">
          <div className="col-12 col-lg-6">
            <div className="border rounded-3 p-3 h-100">
              <div className="fw-semibold mb-2">Поиск</div>
              <div className="text-secondary mb-3" style={{ fontSize: '0.9rem' }}>
                Управляйте фасетами, доступными на странице поиска. Оставьте только нужные поля, чтобы разгрузить интерфейс.
              </div>
              <label className="form-check form-switch mb-3 facet-switch">
                <input
                  className="form-check-input"
                  type="checkbox"
                  checked={includeTypes}
                  onChange={e => setIncludeTypes(e.target.checked)}
                />
                <span className="form-check-label">Показывать фасет по типам материалов</span>
              </label>
              <div className="d-flex flex-column gap-2 mb-3">
                <label className="form-check facet-radio">
                  <input
                    type="radio"
                    className="form-check-input"
                    checked={searchMode === 'all'}
                    onChange={() => setSearchMode('all')}
                  />
                  <span className="form-check-label">Все теги</span>
                </label>
                <label className="form-check facet-radio">
                  <input
                    type="radio"
                    className="form-check-input"
                    checked={searchMode === 'custom'}
                    onChange={() => setSearchMode('custom')}
                  />
                  <span className="form-check-label">Только выбранные</span>
                </label>
                <label className="form-check facet-radio">
                  <input
                    type="radio"
                    className="form-check-input"
                    checked={searchMode === 'none'}
                    onChange={() => setSearchMode('none')}
                  />
                  <span className="form-check-label">Отключить фасеты по тегам</span>
                </label>
              </div>
              {searchMode === 'custom' && (
                <div className="d-flex flex-column gap-2">
                  <div>
                    <div className="fw-semibold mb-2" style={{ fontSize: '0.95rem' }}>Выбранные теги</div>
                    <div className="d-flex flex-wrap gap-2">
                      {searchKeys.length === 0 && <span className="text-secondary" style={{ fontSize: '0.9rem' }}>Добавьте теги из списка ниже.</span>}
                      {searchKeys.map(key => (
                        <span key={key} className="badge bg-secondary-subtle text-body d-inline-flex align-items-center gap-2">
                          {key}
                          <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => toggleSearchKey(key)}>×</button>
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="row g-2">
                    <div className="col-12 col-md-6 compact-input">
                      <label className="form-label text-secondary mb-1" style={{ fontSize: '0.8rem' }}>Фильтр по названию</label>
                      <input className="form-control form-control-sm" value={searchFilter} onChange={e => setSearchFilter(e.target.value)} placeholder="Например, author" />
                    </div>
                    <div className="col-12 col-md-6 compact-input">
                      <label className="form-label text-secondary mb-1" style={{ fontSize: '0.8rem' }}>Добавить тег вручную</label>
                      <div className="input-group input-group-sm">
                        <input
                          className="form-control"
                          value={searchManual}
                          onChange={e => setSearchManual(e.target.value)}
                          placeholder="tag_key"
                          onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); addSearchKey() } }}
                        />
                        <button className="btn btn-outline-secondary" type="button" onClick={addSearchKey}>Добавить</button>
                      </div>
                    </div>
                  </div>
                  <div>
                    <div className="fw-semibold mb-2" style={{ fontSize: '0.95rem' }}>Популярные теги</div>
                    <div className="d-grid gap-1" style={{ maxHeight: 220, overflowY: 'auto' }}>
                      {searchOptionList.length === 0 && <div className="text-secondary" style={{ fontSize: '0.9rem' }}>Совпадений не найдено.</div>}
                      {searchOptionList.map(opt => (
                        <label key={opt.key} className="form-check facet-option">
                          <span className="facet-option-label">
                            <input
                              type="checkbox"
                              className="form-check-input"
                              checked={searchKeys.includes(opt.key)}
                              onChange={() => toggleSearchKey(opt.key)}
                            />
                            <span className="form-check-label">{opt.key}</span>
                          </span>
                          <span className="text-secondary facet-option-count">{opt.count}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
          <div className="col-12 col-lg-6">
            <div className="border rounded-3 p-3 h-100">
              <div className="fw-semibold mb-2">Граф</div>
              <div className="text-secondary mb-3" style={{ fontSize: '0.9rem' }}>
                Сократите набор тегов для графа, чтобы визуализация оставалась читаемой.
              </div>
              <div className="d-flex flex-column gap-2 mb-3">
                <label className="form-check facet-radio">
                  <input
                    type="radio"
                    className="form-check-input"
                    checked={graphMode === 'all'}
                    onChange={() => setGraphMode('all')}
                  />
                  <span className="form-check-label">Все теги</span>
                </label>
                <label className="form-check facet-radio">
                  <input
                    type="radio"
                    className="form-check-input"
                    checked={graphMode === 'custom'}
                    onChange={() => setGraphMode('custom')}
                  />
                  <span className="form-check-label">Только выбранные</span>
                </label>
              </div>
              {graphMode === 'custom' && (
                <div className="d-flex flex-column gap-2">
                  <div>
                    <div className="fw-semibold mb-2" style={{ fontSize: '0.95rem' }}>Выбранные теги</div>
                    <div className="d-flex flex-wrap gap-2">
                      {graphKeys.length === 0 && <span className="text-secondary" style={{ fontSize: '0.9rem' }}>Выберите теги для графа.</span>}
                      {graphKeys.map(key => (
                        <span key={key} className="badge bg-secondary-subtle text-body d-inline-flex align-items-center gap-2">
                          {key}
                          <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => toggleGraphKey(key)}>×</button>
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="row g-2">
                    <div className="col-12 col-md-6 compact-input">
                      <label className="form-label text-secondary mb-1" style={{ fontSize: '0.8rem' }}>Фильтр по названию</label>
                      <input className="form-control form-control-sm" value={graphFilter} onChange={e => setGraphFilter(e.target.value)} placeholder="Например, advisor" />
                    </div>
                    <div className="col-12 col-md-6 compact-input">
                      <label className="form-label text-secondary mb-1" style={{ fontSize: '0.8rem' }}>Добавить тег вручную</label>
                      <div className="input-group input-group-sm">
                        <input
                          className="form-control"
                          value={graphManual}
                          onChange={e => setGraphManual(e.target.value)}
                          placeholder="tag_key"
                          onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); addGraphKey() } }}
                        />
                        <button className="btn btn-outline-secondary" type="button" onClick={addGraphKey}>Добавить</button>
                      </div>
                    </div>
                  </div>
                  <div>
                    <div className="fw-semibold mb-2" style={{ fontSize: '0.95rem' }}>Популярные теги</div>
                    <div className="d-grid gap-1" style={{ maxHeight: 220, overflowY: 'auto' }}>
                      {graphOptionList.length === 0 && <div className="text-secondary" style={{ fontSize: '0.9rem' }}>Совпадений не найдено.</div>}
                      {graphOptionList.map(opt => (
                        <label key={opt.key} className="form-check facet-option">
                          <span className="facet-option-label">
                            <input
                              type="checkbox"
                              className="form-check-input"
                              checked={graphKeys.includes(opt.key)}
                              onChange={() => toggleGraphKey(opt.key)}
                            />
                            <span className="form-check-label">{opt.key}</span>
                          </span>
                          <span className="text-secondary facet-option-count">{opt.count}</span>
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
