import React from 'react'
import { EmptyChat } from '../../ui/EmptyState'

type OsintHistoryJob = {
  id: number
  query: string
  status: string
  sources_completed: number
  sources_total: number
  created_at?: string | null
  schedule?: {
    active?: boolean
    interval_minutes?: number | null
    next_run_at?: string | null
  } | null
}

type Props = {
  history: OsintHistoryJob[]
  activeJobId?: number | null
  onRefresh: () => void
  onSelect: (job: OsintHistoryJob) => void
  onDelete: (jobId: number) => void
  statusBadgeClass: (status: string) => string
  formatTimestamp: (value?: string | null) => string
  formatInterval: (minutes?: number | null) => string
}

export default function OsintHistoryPanel({
  history,
  activeJobId,
  onRefresh,
  onSelect,
  onDelete,
  statusBadgeClass,
  formatTimestamp,
  formatInterval,
}: Props) {
  return (
    <section className="card shadow-sm osint-history-card">
      <div className="card-body">
        <div className="d-flex align-items-center justify-content-between mb-3">
          <h2 className="h5 mb-0">История</h2>
          <button type="button" className="btn btn-sm btn-outline-secondary" onClick={onRefresh}>
            Обновить
          </button>
        </div>
        {history.length === 0 && <EmptyChat />}
        {history.length > 0 && (
          <div className="list-group">
            {history.map(job => {
              const isActive = activeJobId === job.id
              return (
                <div
                  key={job.id}
                  className={`list-group-item list-group-item-action ${isActive ? 'active' : ''}`}
                  role="button"
                  onClick={() => onSelect(job)}
                >
                  <div className="d-flex justify-content-between align-items-center gap-3">
                    <div className="text-start flex-grow-1 min-w-0">
                      <div className="fw-semibold text-break" style={{ overflowWrap: 'anywhere' }}>
                        {job.query}
                      </div>
                      <div className="small text-muted">
                        {job.sources_completed}/{job.sources_total} · {formatTimestamp(job.created_at)}
                      </div>
                      {job.schedule?.active && (
                        <div className="small text-warning mt-1">
                          Расписание: {formatInterval(job.schedule.interval_minutes)}
                          {job.schedule.next_run_at ? ` · следующий: ${formatTimestamp(job.schedule.next_run_at)}` : ''}
                        </div>
                      )}
                    </div>
                    <div className="d-flex align-items-center gap-2">
                      <span className={statusBadgeClass(job.status)}>{job.status}</span>
                      <button
                        type="button"
                        className="btn btn-sm btn-outline-danger"
                        onClick={event => {
                          event.stopPropagation()
                          onDelete(job.id)
                        }}
                        aria-label="Удалить поиск"
                      >
                        ×
                      </button>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </section>
  )
}
