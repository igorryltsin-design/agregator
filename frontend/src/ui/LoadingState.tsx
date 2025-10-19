import React from 'react'

type LoadingStateProps = {
  title?: string
  description?: string
  lines?: number
  className?: string
  style?: React.CSSProperties
  variant?: 'card' | 'inline'
}

const LoadingState: React.FC<LoadingStateProps> = ({
  title,
  description,
  lines = 3,
  className,
  style,
  variant = 'card',
}) => {
  const count = Math.max(1, Math.min(10, Math.floor(lines)))
  const Container: React.ElementType = 'div'
  const baseClass =
    variant === 'card'
      ? 'loading-state loading-state--card card p-3'
      : 'loading-state loading-state--inline'
  const containerClass = className ? `${baseClass} ${className}` : baseClass

  return (
    <Container
      className={containerClass}
      style={style}
      role="status"
      aria-live="polite"
    >
      {title && <div className="fw-semibold mb-1">{title}</div>}
      {description && <div className="text-body-secondary mb-2 small">{description}</div>}
      <div className="d-grid gap-2">
        {Array.from({ length: count }).map((_, idx) => (
          <div
            key={idx}
            className="skeleton skeleton-line"
            style={{ height: idx === 0 ? 18 : 12, width: `${100 - Math.min(35, idx * 12)}%` }}
          />
        ))}
      </div>
    </Container>
  )
}

export default LoadingState
