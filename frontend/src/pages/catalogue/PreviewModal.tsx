import React from 'react'

type PreviewModalProps = {
  relPath: string | null
  onClose: () => void
}

const PreviewModal: React.FC<PreviewModalProps> = ({ relPath, onClose }) => {
  if (!relPath) return null
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Предпросмотр"
      onClick={event => { if (event.target === event.currentTarget) onClose() }}
      style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1050 }}
    >
      <button
        aria-label="Закрыть"
        onClick={onClose}
        style={{ position: 'absolute', top: 16, right: 16, border: '1px solid var(--border)', background: 'var(--surface)', color: 'var(--text)', borderRadius: 8, padding: '6px 10px' }}
      >
        ×
      </button>
      <div className="card" role="document" style={{ width: '90vw', height: '85vh', background: 'var(--surface)', borderColor: 'var(--border)' }}>
        <div className="card-header d-flex justify-content-between align-items-center">
          <div className="fw-semibold">Предпросмотр</div>
          <button className="btn btn-sm btn-outline-secondary" onClick={onClose}>Закрыть</button>
        </div>
        <div className="card-body p-0" style={{ height: 'calc(100% - 56px)' }}>
          <iframe title="preview" src={`/preview/${encodeURIComponent(relPath)}?embedded=1`} style={{ border: 0, width: '100%', height: '100%' }} />
        </div>
      </div>
    </div>
  )
}

export default PreviewModal
