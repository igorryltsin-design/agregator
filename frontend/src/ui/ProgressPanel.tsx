import React from 'react'

export type ProgressBullet = {
  id: string
  text: string
  icon?: React.ReactNode
  active?: boolean
}

export type ProgressPanelProps = {
  title: string
  caption?: string
  bullets?: ProgressBullet[]
  progress?: { percent: number; label?: string }
  footer?: React.ReactNode
  className?: string
  style?: React.CSSProperties
  actions?: React.ReactNode
}

const ProgressPanel: React.FC<ProgressPanelProps> = ({ title, caption, bullets, progress, footer, className, style, actions }) => {
  const hasBullets = Array.isArray(bullets) && bullets.length > 0
  return (
    <div className={`card p-3 progress-panel ${className || ''}`} style={style}>
      <div className="d-flex justify-content-between align-items-start mb-2">
        <div>
          <div className="fw-semibold">{title}</div>
          {caption && <div className="text-muted" style={{ fontSize: '0.85rem' }}>{caption}</div>}
        </div>
        {(actions) && (
          <div className="d-flex align-items-center gap-2">
            {actions}
          </div>
        )}
      </div>
      {progress && (
        <div className="mb-2">
          <div className="progress" style={{ height: 6 }}>
            <div className="progress-bar" role="progressbar" style={{ width: `${Math.min(100, Math.max(0, progress.percent))}%` }} aria-valuenow={progress.percent} aria-valuemin={0} aria-valuemax={100} />
          </div>
          {progress.label && <div className="text-muted mt-1" style={{ fontSize: 12 }}>{progress.label}</div>}
        </div>
      )}
      {hasBullets && (
        <div className="d-grid gap-1">
          {bullets!.map(item => (
            <div key={item.id} className={`d-flex align-items-start gap-2 ${item.active ? 'fw-semibold' : ''}`}>
              {item.icon && <span style={{ width: 20 }}>{item.icon}</span>}
              <span>{item.text}</span>
            </div>
          ))}
        </div>
      )}
      {footer && <div className="mt-3 d-flex flex-wrap gap-2">{footer}</div>}
    </div>
  )
}

export default ProgressPanel
