import React from 'react'
import { materialTypeOptions, tagKeyRu } from '../../utils/locale'
import { useCatalogueState } from './useCatalogueState'
import AiPanel from './AiPanel'
import FileCard from './FileCard'
import PreviewModal from './PreviewModal'
import FilterSidebar from './FilterSidebar'

export default function CatalogueView() {
  const catalogue = useCatalogueState()
  const { searchParams, selectors, list, ai, pagination, modals, helpers, facets, facetsLoading, collections } = catalogue

  const { sp, setSp, dq, type, collectionId } = searchParams
  const { selectedTags, setSelectedTags, removeTag } = selectors
  const { items, total, loading, sentinelRef } = list
  const { aiMode, resetAiState } = ai
  const { page, pages, canLoadMore } = pagination
  const { previewRel, setPreviewRel, editItem, setEditItem, editForm, setEditForm, saveEdit, openEdit } = modals
  const { updateTagsInline, refreshFile, renameFile, deleteFile } = helpers

  const handleCollectionChange = (value: string) => {
    const next = new URLSearchParams(sp)
    if (value) {
      next.set('collection_id', value)
    } else {
      next.delete('collection_id')
    }
    next.set('page', '1')
    setSp(next)
  }

  const hasFacetFilters = Boolean(type) || selectedTags.length > 0

  const handleResetFacets = () => {
    if (!hasFacetFilters) return
    const next = new URLSearchParams(sp)
    next.delete('type')
    next.delete('tag')
    next.set('page', '1')
    setSelectedTags([])
    setSp(next)
  }

  const handleResetAll = () => {
    const next = new URLSearchParams()
    next.set('page', '1')
    setSelectedTags([])
    resetAiState()
    setSp(next)
  }

  const handleTypeChange = (value: string | null) => {
    const next = new URLSearchParams(sp)
    if (value) {
      next.set('type', value)
    } else {
      next.delete('type')
    }
    next.set('page', '1')
    setSp(next)
  }

  const handleTagToggle = (tag: string) => {
    const nextTags = selectedTags.includes(tag)
      ? selectedTags.filter(x => x !== tag)
      : [...selectedTags, tag]
    setSelectedTags(nextTags)
    const nextParams = new URLSearchParams(sp)
    nextParams.delete('tag')
    nextTags.forEach(value => nextParams.append('tag', value))
    nextParams.set('page', '1')
    setSp(nextParams)
  }

  return (
    <>
      <div className="row g-3">
        <div className="col-12 col-lg-3">
          <FilterSidebar
            facets={facets}
            typeFilter={type}
            onTypeChange={handleTypeChange}
            selectedTags={selectedTags}
            onToggleTag={handleTagToggle}
            onReset={handleResetFacets}
            canReset={hasFacetFilters}
            isLoading={facetsLoading}
          />
        </div>
        <div className="col-12 col-lg-9">
          <div className="d-flex flex-wrap gap-2 align-items-center mb-2">
            <div className="form-floating" style={{ minWidth: 220 }}>
              <select className="form-select" id="collectionSelect" value={collectionId} onChange={event => handleCollectionChange(event.target.value)}>
                <option value="">Все доступные</option>
                {collections.map(col => (
                  <option key={col.id} value={col.id}>{col.name}</option>
                ))}
              </select>
              <label htmlFor="collectionSelect">Коллекция</label>
            </div>
          </div>

          <AiPanel ai={ai} />

          <div className="floating-search mb-2 d-flex align-items-center justify-content-between" role="toolbar" aria-label="Выбранные фильтры">
            <div className="d-flex flex-wrap align-items-center" style={{ gap: 8 }}>
              {selectedTags.map((tag, index) => {
                const [rawKey, ...rest] = tag.split('=')
                const value = rest.join('=')
                const keyLabel = tagKeyRu(rawKey || tag)
                const label = value ? `${keyLabel}: ${value}` : keyLabel
                return (
                  <span key={index} className="tag" aria-label={`Фильтр ${label}`}>
                    {label}{' '}
                    <button className="btn btn-sm btn-outline-secondary ms-1" onClick={() => removeTag(index)} aria-label="Снять фильтр">×</button>
                  </span>
                )
              })}
              <span className="muted ms-2">Найдено: {total}</span>
            </div>
            {pages > 1 && (
              <div className="btn-group">
                <button className="btn btn-sm btn-outline-secondary" disabled={page <= 1} onClick={() => { const next = new URLSearchParams(sp); next.set('page', String(page - 1)); setSp(next) }}>«</button>
                {Array.from({ length: Math.min(7, pages) }).map((_, idx) => {
                  const base = Math.max(1, Math.min(pages - 6, page - 3))
                  const num = Math.min(pages, base + idx)
                  return (
                    <button
                      key={idx}
                      className={`btn btn-sm ${num === page ? 'btn-secondary' : 'btn-outline-secondary'}`}
                      onClick={() => { const next = new URLSearchParams(sp); next.set('page', String(num)); setSp(next) }}
                    >
                      {num}
                    </button>
                  )
                })}
                <button className="btn btn-sm btn-outline-secondary" disabled={page >= pages} onClick={() => { const next = new URLSearchParams(sp); next.set('page', String(page + 1)); setSp(next) }}>»</button>
              </div>
            )}
            <button className="btn btn-sm btn-outline-secondary ms-2" onClick={handleResetAll}>Сбросить</button>
          </div>

          {loading && !aiMode && (
            <div className="masonry">
              {Array.from({ length: 9 }).map((_, idx) => (
                <div key={idx} className="masonry-item">
                  <div className="skeleton" style={{ height: 120 }} />
                </div>
              ))}
            </div>
          )}

          <div className="masonry">
            {items.map(file => (
              <FileCard
                key={file.id}
                file={file}
                query={dq || ''}
                onPreview={setPreviewRel}
                onEdit={openEdit}
                onRefresh={refreshFile}
                onRename={renameFile}
                onDelete={deleteFile}
                onTagSubmit={updateTagsInline}
              />
            ))}
          </div>

          {!aiMode && canLoadMore && (
            <div ref={sentinelRef as any} className="mt-3 text-center muted">Подгружаю ещё…</div>
          )}
        </div>
      </div>

      <PreviewModal relPath={previewRel} onClose={() => setPreviewRel(null)} />

      {editItem && editForm && (
        <div role="dialog" aria-modal="true" aria-label="Редактирование" onClick={event => { if (event.target === event.currentTarget) setEditItem(null) }}
             style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1060 }}>
          <div className="card" role="document" style={{ width: '720px', maxWidth: '96vw', background: 'var(--surface)', borderColor: 'var(--border)' }}>
            <div className="card-header d-flex justify-content-between align-items-center">
              <div className="fw-semibold">Редактирование</div>
              <button className="btn btn-sm btn-outline-secondary" onClick={() => setEditItem(null)}>Закрыть</button>
            </div>
            <div className="card-body">
              <div className="d-grid gap-2">
                <input className="form-control" placeholder="Название" value={editForm.title} onChange={event => setEditForm({ ...editForm, title: event.target.value })} />
                <div className="d-flex gap-2">
                  <input className="form-control" placeholder="Автор" value={editForm.author} onChange={event => setEditForm({ ...editForm, author: event.target.value })} />
                  <input className="form-control" placeholder="Год" value={editForm.year} onChange={event => setEditForm({ ...editForm, year: event.target.value })} />
                </div>
                <div className="d-flex gap-2">
                  <input className="form-control" placeholder="Тип" list="material-type-options" value={editForm.material_type} onChange={event => setEditForm({ ...editForm, material_type: event.target.value })} />
                  <input className="form-control" placeholder="Имя файла (без расширения)" value={editForm.filename} onChange={event => setEditForm({ ...editForm, filename: event.target.value })} />
                </div>
                <datalist id="material-type-options">
                  {materialTypeOptions.map(opt => (
                    <option key={opt.value} value={opt.label} label={opt.value} />
                  ))}
                </datalist>
                <input className="form-control" placeholder="Ключевые слова" value={editForm.keywords} onChange={event => setEditForm({ ...editForm, keywords: event.target.value })} />
                <label className="form-label">Теги</label>
                <div className="mb-2" style={{ maxHeight: 100, overflow: 'auto' }}>
                  {(editForm.tagsText || '').split(/\n|;/).map((row: string, idx: number) => {
                    const trimmed = row.trim()
                    if (!trimmed) return null
                    return (
                      <button key={idx} className="tag" onClick={event => { event.preventDefault(); const arr = (editForm.tagsText || '').split(/\n|;/).filter((value: string) => value.trim() && value.trim() !== trimmed); setEditForm({ ...editForm, tagsText: arr.join('\n') }) }}> {trimmed} × </button>
                    )
                  })}
                </div>
                <div className="d-flex gap-2 mb-2">
                  <input className="form-control" placeholder="ключ" onKeyDown={event => {
                    if (event.key === 'Enter') {
                      const key = (event.currentTarget as HTMLInputElement).value.trim()
                      const val = (event.currentTarget.parentElement?.querySelector('[data-newtag="value"]') as HTMLInputElement)?.value.trim()
                      if (key && val) {
                        const nt = `${key}=${val}`
                        const txt = (editForm.tagsText || '')
                        setEditForm({ ...editForm, tagsText: (txt ? txt + '\n' : '') + nt })
                        ;(event.currentTarget as HTMLInputElement).value = ''
                        if ((event.currentTarget.parentElement?.querySelector('[data-newtag="value"]') as HTMLInputElement)) {
                          (event.currentTarget.parentElement?.querySelector('[data-newtag="value"]') as HTMLInputElement).value = ''
                        }
                      }
                    }
                  }} />
                  <input className="form-control" placeholder="значение" data-newtag="value" onKeyDown={event => {
                    if (event.key === 'Enter') {
                      const val = (event.currentTarget as HTMLInputElement).value.trim()
                      const key = (event.currentTarget.parentElement?.querySelector('input:not([data-newtag])') as HTMLInputElement)?.value.trim()
                      if (key && val) {
                        const nt = `${key}=${val}`
                        const txt = (editForm.tagsText || '')
                        setEditForm({ ...editForm, tagsText: (txt ? txt + '\n' : '') + nt })
                        ;(event.currentTarget as HTMLInputElement).value = ''
                        if ((event.currentTarget.parentElement?.querySelector('input:not([data-newtag])') as HTMLInputElement)) {
                          (event.currentTarget.parentElement?.querySelector('input:not([data-newtag])') as HTMLInputElement).value = ''
                        }
                      }
                    }
                  }} />
                </div>
                <textarea className="form-control" rows={4} placeholder="ключ=значение, по строке" value={editForm.tagsText} onChange={event => setEditForm({ ...editForm, tagsText: event.target.value })} />
                <div className="d-flex gap-2">
                  <button className="btn btn-primary" onClick={saveEdit}>Сохранить</button>
                  <button className="btn btn-outline-secondary" onClick={() => setEditItem(null)}>Отмена</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
