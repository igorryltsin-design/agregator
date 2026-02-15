import React, { useState } from 'react'
import type { FileItem, Tag } from './types'
import { materialTypeRu, tagKeyRu } from '../../utils/locale'

type FileCardProps = {
  file: FileItem
  query: string
  onPreview: (relPath: string) => void
  onEdit: (file: FileItem) => void
  onRefresh: (file: FileItem) => void
  onRename: (file: FileItem) => void
  onDelete: (file: FileItem) => void
  onTagSubmit: (file: FileItem, tags: Tag[]) => void | Promise<void>
  onStartChat: (file: FileItem) => void
  onSearchFeedback?: (file: FileItem, action: 'click' | 'relevant' | 'irrelevant') => void
}

const FileCard: React.FC<FileCardProps> = ({ file, query, onPreview, onEdit, onRefresh, onRename, onDelete, onTagSubmit, onStartChat, onSearchFeedback }) => {
  const [refreshing, setRefreshing] = useState(false)
  const [renaming, setRenaming] = useState(false)

  const highlightSnippet = (text?: string | null) => {
    if (!text) return ''
    const esc = (value: string) => value.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c] || c))
    const words = (query || '').trim().split(/[\s,]+/).filter(term => term.length > 1).slice(0, 6)
    if (!words.length) return esc(text)
    const re = new RegExp('(' + words.map(word => word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')', 'gi')
    return esc(text).replace(re, '<mark>$1</mark>')
  }

  const handleTagSubmit: React.FormEventHandler<HTMLFormElement> = async event => {
    event.preventDefault()
    const input = event.currentTarget.querySelector('input') as HTMLInputElement | null
    const value = (input?.value || '').trim()
    if (!value) return
    const idx = value.indexOf('=')
    if (idx === -1) return
    const key = value.slice(0, idx).trim()
    const tagValue = value.slice(idx + 1).trim()
    if (!key || !tagValue) return
    const nextTags = [...(file.tags || []), { key, value: tagValue }]
    await onTagSubmit(file, nextTags)
    if (input) input.value = ''
  }

  const hasRelPath = Boolean(file.rel_path)

  return (
    <div className="masonry-item">
      <article className={`card p-2 file-card card-type-${(file.material_type || 'document').toLowerCase()}`} style={{ display: 'flex', flexDirection: 'column' }}>
        <div className="d-flex align-items-start justify-content-between">
          <div className="fw-semibold file-card__title" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {file.title || (file.rel_path?.split('/').pop() || '—')}
          </div>
          <div className="d-flex align-items-center gap-1 file-card__badges">
            {file.metadata_quality?.bucket === 'low' && (
              <span className="badge bg-warning text-dark ms-2" title={`Заполнено ${file.metadata_quality.filled}/${file.metadata_quality.total}`}>Низкое качество</span>
            )}
            {file.material_type && <span className="badge bg-secondary ms-2">{materialTypeRu(file.material_type)}</span>}
          </div>
        </div>
        <div className="text-secondary file-card__meta" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {[file.author, file.year].filter(Boolean).join(', ') || '—'}
        </div>
        {(file.keywords || '').trim() && (
          <div className="mt-1 file-card__keywords" style={{ maxHeight: 48, overflow: 'hidden' }}>
            {(file.keywords || '').split(/[,;]+/).slice(0, 10).map((keyword, idx) => (
              <span key={idx} className="tag">{keyword.trim()}</span>
            ))}
          </div>
        )}
        {file.rel_path && file.material_type === 'image' && (
          <div className="mt-2">
            <img src={`/media/${encodeURIComponent(file.rel_path)}`} alt="prev" style={{ maxHeight: 140, width: '100%', objectFit: 'contain', borderRadius: 8, border: '1px solid var(--border)' }} />
          </div>
        )}
        {file.material_type === 'audio' && file.rel_path && (
          <div className="media-frame mt-2">
            <audio src={`/media/${encodeURIComponent(file.rel_path)}`} controls preload="metadata" style={{ width: '100%' }} />
          </div>
        )}
        {(!file.material_type || file.material_type === 'document' || file.material_type === 'article' || file.material_type === 'textbook') && file.text_excerpt && (
          <div className="mt-2 muted file-card__excerpt" style={{ maxHeight: 64, overflow: 'hidden' }} dangerouslySetInnerHTML={{ __html: highlightSnippet(file.text_excerpt) }} />
        )}
        <div className="mt-2 file-card__tags" style={{ maxHeight: 68, overflow: 'auto' }}>
          {(file.tags || []).map((tag, idx) => (
            <span key={idx} className="tag" title={`${tag.key}=${tag.value}`}>{tagKeyRu(tag.key)}:{tag.value}</span>
          ))}
          <form className="d-inline" onSubmit={handleTagSubmit}>
            <input
              placeholder="ключ=значение"
              style={{ background: 'var(--bg)', color: 'var(--text)', border: '1px dashed var(--border)', borderRadius: 8, padding: '2px 6px', marginLeft: 6, width: 120 }}
            />
          </form>
        </div>
        <div className="mt-2 d-flex gap-2 mt-auto file-card__actions">
          {hasRelPath && <button className="btn btn-sm btn-outline-secondary" onClick={() => { onSearchFeedback?.(file, 'click'); onPreview(file.rel_path!) }} title="Предпросмотр">👁️</button>}
          <button className="btn btn-sm btn-outline-primary" onClick={() => { onSearchFeedback?.(file, 'relevant'); onStartChat(file) }} title="Чат с документом">💬</button>
          <button className="btn btn-sm btn-outline-secondary" disabled={refreshing} onClick={async () => { if (refreshing) return; setRefreshing(true); try { await onRefresh(file) } finally { setRefreshing(false) } }} title="Обновить теги">{refreshing ? '…' : '⟳'}</button>
          <button className="btn btn-sm btn-outline-secondary" onClick={() => onEdit(file)} title="Редактировать">📝</button>
          <button className="btn btn-sm btn-outline-secondary" disabled={renaming} onClick={async () => { if (renaming) return; setRenaming(true); try { await onRename(file) } finally { setRenaming(false) } }} title="Автопереименование">{renaming ? '…' : '✎'}</button>
          {hasRelPath && <a className="btn btn-sm btn-outline-secondary" href={`/download/${encodeURIComponent(file.rel_path!)}`} onClick={() => onSearchFeedback?.(file, 'click')}>Скачать</a>}
          <button className="btn btn-sm btn-outline-danger ms-auto" onClick={() => onDelete(file)} title="Удалить">✖</button>
        </div>
      </article>
    </div>
  )
}

export default FileCard
