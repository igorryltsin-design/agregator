import React, { useMemo, useState } from 'react'
import type { FacetData, FacetSuggestion } from './types'
import { materialTypeRu, tagKeyRu } from '../../utils/locale'
import LoadingState from '../../ui/LoadingState'

type FilterSidebarProps = {
  facets: FacetData | null
  typeFilter: string
  onTypeChange: (value: string | null) => void
  selectedTags: string[]
  onToggleTag: (tag: string) => void
  onReset: () => void
  canReset: boolean
  isLoading: boolean
  yearFrom: string
  yearTo: string
  onYearSelect: (year: string | null) => void
  onSuggestion?: (suggestion: FacetSuggestion) => void
}

const FilterSidebar: React.FC<FilterSidebarProps> = ({
  facets,
  typeFilter,
  onTypeChange,
  selectedTags,
  onToggleTag,
  onReset,
  canReset,
  isLoading,
  yearFrom,
  yearTo,
  onYearSelect,
  onSuggestion,
}) => {
  const [tagSearch, setTagSearch] = useState('')
  const [expandedKeys, setExpandedKeys] = useState<Record<string, boolean>>({})
  const [authorsExpanded, setAuthorsExpanded] = useState(false)
  const [yearsExpanded, setYearsExpanded] = useState(false)

  const filteredFacets = useMemo(() => {
    if (!facets) return null
    const query = tagSearch.trim().toLowerCase()
    if (!query) return facets.tag_facets
    const result: typeof facets.tag_facets = {}
    Object.entries(facets.tag_facets).forEach(([key, values]) => {
      const matches = values.filter(([value]) => String(value).toLowerCase().includes(query) || tagKeyRu(key).toLowerCase().includes(query))
      if (matches.length) {
        result[key] = matches
      }
    })
    return result
  }, [facets, tagSearch])

  const activeYear = useMemo(() => {
    if (yearFrom && yearTo && yearFrom === yearTo) return yearFrom
    if (yearFrom && !yearTo) return yearFrom
    if (yearTo && !yearFrom) return yearTo
    return ''
  }, [yearFrom, yearTo])

  const handleSuggestionClick = (suggestion: FacetSuggestion) => {
    if (suggestion.kind === 'author') {
      onToggleTag(`author=${suggestion.value}`)
      return
    }
    if (suggestion.kind === 'year') {
      onYearSelect(activeYear === suggestion.value ? null : suggestion.value)
      return
    }
    if (suggestion.kind === 'tag' && suggestion.key) {
      onToggleTag(`${suggestion.key}=${suggestion.value}`)
      return
    }
    if (onSuggestion) {
      onSuggestion(suggestion)
    }
  }

  return (
    <div className="card p-3" style={{ position: 'sticky', top: 8, maxHeight: 'calc(100vh - 120px)', overflow: 'auto' }} aria-busy={isLoading}>
      <div className="d-flex justify-content-between align-items-center mb-2">
        <div className="fw-semibold">Фильтры</div>
        <button className="btn btn-sm btn-outline-secondary" onClick={onReset} disabled={!canReset}>
          Сбросить
        </button>
      </div>
      {isLoading && <LoadingState variant="inline" lines={4} />}
      {!isLoading && !facets && (
        <div className="text-secondary">Нет данных фасетов.</div>
      )}
      {facets && (
        <div className="d-flex flex-column gap-3">
          {Array.isArray(facets.suggestions) && facets.suggestions.length > 0 && (
            <div>
              <div className="mb-2">Подсказки</div>
              <div className="d-flex flex-wrap gap-2">
                {facets.suggestions.map((suggestion, idx) => {
                  const isActive =
                    (suggestion.kind === 'author' && selectedTags.includes(`author=${suggestion.value}`)) ||
                    (suggestion.kind === 'year' && activeYear === suggestion.value) ||
                    (suggestion.kind === 'tag' && suggestion.key && selectedTags.includes(`${suggestion.key}=${suggestion.value}`))
                  return (
                    <button
                      key={`${suggestion.kind}-${suggestion.value}-${idx}`}
                      className={`btn btn-sm ${isActive ? 'btn-secondary' : 'btn-outline-primary'}`}
                      type="button"
                      onClick={() => handleSuggestionClick(suggestion)}
                    >
                      {suggestion.label}
                    </button>
                  )
                })}
              </div>
            </div>
          )}

          {facets.include_types !== false && facets.types.length > 0 && (
            <div>
              <div className="mb-2">Типы материалов</div>
              <div className="d-flex flex-column gap-2">
                {facets.types.map(([value, count], idx) => {
                  const stringValue = String(value || '')
                  const active = typeFilter === stringValue
                  return (
                    <button
                      key={idx}
                      className={`btn btn-sm ${active ? 'btn-secondary' : 'btn-outline-secondary'} d-flex justify-content-between align-items-center`}
                      onClick={() => onTypeChange(active ? null : stringValue)}
                    >
                      <span>{materialTypeRu(value, 'Другое')}</span>
                      <span className="text-secondary">{count}</span>
                    </button>
                  )
                })}
              </div>
            </div>
          )}

          {Array.isArray(facets.authors) && facets.authors.length > 0 && (
            <div>
              <div className="mb-2 d-flex justify-content-between align-items-center">
                <span>Авторы</span>
                {facets.authors.length > 10 && (
                  <button className="btn btn-link btn-sm p-0" onClick={() => setAuthorsExpanded(prev => !prev)}>
                    {authorsExpanded ? 'Свернуть' : 'Ещё'}
                  </button>
                )}
              </div>
              <div className="d-flex flex-column gap-2">
                {(authorsExpanded ? facets.authors : facets.authors.slice(0, 10)).map(([author, count]) => {
                  const tagValue = `author=${author}`
                  const active = selectedTags.includes(tagValue)
                  return (
                    <button
                      key={author}
                      className={`btn btn-sm ${active ? 'btn-secondary' : 'btn-outline-secondary'} d-flex justify-content-between align-items-center`}
                      type="button"
                      onClick={() => onToggleTag(tagValue)}
                    >
                      <span className="text-truncate" style={{ maxWidth: '85%' }}>{author}</span>
                      <span className="text-secondary">{count}</span>
                    </button>
                  )
                })}
              </div>
            </div>
          )}

          {Array.isArray(facets.years) && facets.years.length > 0 && (
            <div>
              <div className="mb-2 d-flex justify-content-between align-items-center">
                <span>Годы</span>
                {facets.years.length > 12 && (
                  <button className="btn btn-link btn-sm p-0" onClick={() => setYearsExpanded(prev => !prev)}>
                    {yearsExpanded ? 'Свернуть' : 'Ещё'}
                  </button>
                )}
              </div>
              <div className="d-flex flex-wrap gap-2">
                {(yearsExpanded ? facets.years : facets.years.slice(0, 12)).map(([year, count]) => {
                  const active = activeYear === year
                  return (
                    <button
                      key={year}
                      className={`btn btn-sm ${active ? 'btn-secondary' : 'btn-outline-secondary'}`}
                      type="button"
                      onClick={() => onYearSelect(active ? null : year)}
                    >
                      {year} <span className="text-secondary">({count})</span>
                    </button>
                  )
                })}
              </div>
            </div>
          )}

          <div>
            <div className="mb-2">Теги</div>
            <input
              type="search"
              className="form-control form-control-sm mb-2"
              placeholder="Поиск по тегам"
              value={tagSearch}
              onChange={event => setTagSearch(event.target.value)}
            />
            {filteredFacets && Object.keys(filteredFacets).length === 0 && (
              <div className="text-secondary" style={{ fontSize: '0.9rem' }}>Ничего не найдено.</div>
            )}
            {filteredFacets && Object.entries(filteredFacets).map(([key, values]) => {
              const expanded = expandedKeys[key]
              const visibleValues = expanded ? values : values.slice(0, 10)
              const hasMore = values.length > visibleValues.length
              return (
                <div key={key} className="mb-3">
                  <div className="text-secondary d-flex justify-content-between align-items-center">
                    <span>{tagKeyRu(key)}</span>
                    {hasMore && (
                      <button
                        className="btn btn-link btn-sm p-0"
                        onClick={() => setExpandedKeys(prev => ({ ...prev, [key]: !expanded }))}
                      >
                        {expanded ? 'Свернуть' : 'Ещё'}
                      </button>
                    )}
                  </div>
                  <div className="d-flex flex-wrap gap-2 mt-1">
                    {visibleValues.map(([value, count], idx) => {
                      const tagValue = `${key}=${value}`
                      const active = selectedTags.includes(tagValue)
                      return (
                        <button
                          key={idx}
                          className={`btn btn-sm ${active ? 'btn-secondary' : 'btn-outline-secondary'}`}
                          onClick={() => onToggleTag(tagValue)}
                        >
                          {value} <span className="text-secondary">({count})</span>
                        </button>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

export default FilterSidebar
