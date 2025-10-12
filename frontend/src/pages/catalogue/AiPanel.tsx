import React from 'react'
import type { useCatalogueState } from './useCatalogueState'
import ProgressPanel, { ProgressBullet } from '../../ui/ProgressPanel'

type AiState = ReturnType<typeof useCatalogueState>['ai']

type AiPanelProps = {
  ai: AiState
}

const AiPanel: React.FC<AiPanelProps> = ({ ai }) => {
  const {
    aiMode,
    setAiMode,
    showAiSettings,
    setShowAiSettings,
    aiTopK,
    setAiTopK,
    aiMaxCandidates,
    setAiMaxCandidates,
    aiDeepSearch,
    setAiDeepSearch,
    aiMaxChunks,
    setAiMaxChunks,
    aiChunkChars,
    setAiChunkChars,
    aiMaxSnippets,
    setAiMaxSnippets,
    aiFullText,
    setAiFullText,
    aiUseLlmSnippets,
    setAiUseLlmSnippets,
    aiAllLanguages,
    setAiAllLanguages,
    aiUseTags,
    setAiUseTags,
    aiUseText,
    setAiUseText,
    aiLanguageOptions,
    aiLoading,
    aiProgress,
    progressItems,
    aiAnswer,
    safeAiAnswer,
    aiKeywords,
    aiFilteredKeywords,
    aiSources,
    aiQueryHash,
    feedbackStatus,
    aiLoadingState: { speechSupported, autoSpeakAnswer, setAutoSpeakAnswer, speechState, speechError },
    handlers: { handleSourceFeedback, handleKeywordFeedback, handleKeywordRestore, speakAnswer, stopSpeakingAnswer },
    canSpeakAnswer,
  } = ai

  return (
    <>
      <div className="mb-2 d-flex align-items-center gap-2">
        <div className="form-check form-switch">
          <input className="form-check-input" type="checkbox" id="ai" checked={aiMode} onChange={e => setAiMode(e.target.checked)} />
          <label className="form-check-label" htmlFor="ai">Поиск ИИ</label>
        </div>
        {aiMode && (
          <>
            <button className="btn btn-sm btn-outline-secondary" onClick={() => setShowAiSettings(v => !v)}>
              {showAiSettings ? 'Скрыть настройки' : 'Настройки ИИ'}
            </button>
            <span className="muted" style={{ fontSize: '0.9rem' }}>
              Топ {aiTopK} · кандидатов {aiMaxCandidates} · {aiDeepSearch ? 'глубокий' : 'быстрый'}{aiFullText ? ' · полный текст' : ''}{aiUseLlmSnippets ? ' · сниппеты LLM' : ''}{aiAllLanguages ? ' · все языки' : ''}
            </span>
          </>
        )}
      </div>

      {aiMode && showAiSettings && (
        <div className="card p-3 mb-2 bg-light">
          <div className="row g-3">
            <div className="col-12 col-lg-3">
              <label className="form-label">Топ K (1–5)</label>
              <input
                type="number"
                className="form-control"
                min={1}
                max={5}
                value={aiTopK}
                onChange={e => {
                  const parsed = parseInt(e.target.value, 10)
                  if (!Number.isNaN(parsed)) setAiTopK(Math.max(1, Math.min(5, parsed)))
                }}
              />
            </div>
            <div className="col-12 col-lg-3">
              <label className="form-label">Кандидатов</label>
              <input
                type="number"
                className="form-control"
                min={3}
                max={30}
                value={aiMaxCandidates}
                onChange={e => {
                  const parsed = parseInt(e.target.value, 10)
                  if (!Number.isNaN(parsed)) setAiMaxCandidates(Math.max(3, Math.min(30, parsed)))
                }}
              />
            </div>
            <div className="col-12 col-lg-3">
              <label className="form-label">Чанк (символов)</label>
              <input
                type="number"
                className="form-control"
                min={1000}
                max={8000}
                step={500}
                value={aiChunkChars}
                onChange={e => {
                  const parsed = parseInt(e.target.value, 10)
                  if (!Number.isNaN(parsed)) setAiChunkChars(Math.max(1000, Math.min(8000, parsed)))
                }}
              />
            </div>
            <div className="col-12 col-lg-3">
              <label className="form-label">Макс. чанков</label>
              <input
                type="number"
                className="form-control"
                min={10}
                max={80}
                value={aiMaxChunks}
                onChange={e => {
                  const parsed = parseInt(e.target.value, 10)
                  if (!Number.isNaN(parsed)) setAiMaxChunks(Math.max(10, Math.min(80, parsed)))
                }}
              />
            </div>
            <div className="col-12 col-lg-3">
              <label className="form-label">Сниппетов</label>
              <input
                type="number"
                className="form-control"
                min={1}
                max={5}
                value={aiMaxSnippets}
                onChange={e => {
                  const parsed = parseInt(e.target.value, 10)
                  if (!Number.isNaN(parsed)) setAiMaxSnippets(Math.max(1, Math.min(5, parsed)))
                }}
              />
            </div>
            <div className="col-12 col-lg-3 d-flex flex-column justify-content-center">
              <div className="form-check form-switch">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="aiDeepSearch"
                  checked={aiDeepSearch}
                  onChange={e => setAiDeepSearch(e.target.checked)}
                />
                <label className="form-check-label" htmlFor="aiDeepSearch">Глубокий поиск</label>
              </div>
              <small className="text-muted">Проходит контекст шире и использует rerank, но ищет дольше.</small>
            </div>
            <div className="col-12 col-lg-3 d-flex flex-column justify-content-center">
              <div className="form-check form-switch">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="aiFullText"
                  checked={aiFullText}
                  onChange={e => setAiFullText(e.target.checked)}
                />
                <label className="form-check-label" htmlFor="aiFullText">Полный текст</label>
              </div>
              <small className="text-muted">Читает сам файл вместо кэша. Медленнее, но точнее.</small>
            </div>
            <div className="col-12 col-lg-3 d-flex flex-column justify-content-center">
              <div className="form-check form-switch">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="aiLlmSnippets"
                  checked={aiUseLlmSnippets}
                  onChange={e => setAiUseLlmSnippets(e.target.checked)}
                />
                <label className="form-check-label" htmlFor="aiLlmSnippets">LLM сниппеты</label>
              </div>
              <small className="text-muted">Формирует короткий пересказ для выдачи (медленнее, но информативнее).</small>
            </div>
            <div className="col-12 col-lg-3">
              <div className="form-check">
                <input className="form-check-input" type="checkbox" id="aiTags" checked={aiUseTags} onChange={e => setAiUseTags(e.target.checked)} />
                <label className="form-check-label" htmlFor="aiTags">Учитывать теги</label>
              </div>
              <div className="form-check">
                <input className="form-check-input" type="checkbox" id="aiText" checked={aiUseText} onChange={e => setAiUseText(e.target.checked)} />
                <label className="form-check-label" htmlFor="aiText">Учитывать метаданные</label>
              </div>
              <small className="text-muted">Выключите, если нужна чистая работа только по одному источнику.</small>
            </div>
            <div className="col-12 col-lg-3 d-flex flex-column justify-content-center">
              <div className="form-check form-switch">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="aiAllLanguages"
                  checked={aiAllLanguages}
                  disabled={aiLanguageOptions.length === 0}
                  onChange={e => setAiAllLanguages(e.target.checked)}
                />
                <label className="form-check-label" htmlFor="aiAllLanguages">Искать по всем языкам</label>
              </div>
              <small className="text-muted">
                {aiLanguageOptions.length ? `Доступно: ${aiLanguageOptions.join(', ')}` : 'Нет языковых тегов'}
              </small>
            </div>
            <div className="col-12 col-lg-3 d-flex flex-column justify-content-center">
              <div className="form-check form-switch">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="aiAutoSpeak"
                  checked={autoSpeakAnswer}
                  disabled={!speechSupported}
                  onChange={e => setAutoSpeakAnswer(e.target.checked)}
                />
                <label className="form-check-label" htmlFor="aiAutoSpeak">Автоматически озвучивать ответ</label>
              </div>
              {speechSupported ? (
                <small className="text-muted">Старт озвучки при появлении нового ответа.</small>
              ) : (
                <small className="text-muted">Синтез речи недоступен в этом браузере.</small>
              )}
            </div>
          </div>
        </div>
      )}

      {aiMode && (aiProgress.length > 0 || aiLoading) && (
        <ProgressPanel
          className="mb-2"
          title="Прогресс поиска"
          caption={aiQueryHash ? `Хэш ${aiQueryHash.slice(0, 8)}` : undefined}
          bullets={(function (): ProgressBullet[] {
            const source = progressItems.length ? progressItems : [{ id: 'pending', line: 'Поиск выполняется…', icon: '⏳' }]
            return source.map((item, index) => ({
              id: item.id,
              text: item.line,
              icon: <span>{item.icon}</span>,
              active: aiLoading && index === source.length - 1,
            }))
          })()}
        />
      )}

      {!!aiAnswer && (
        <div className="card p-3 mb-2" aria-live="polite">
          <div className="d-flex align-items-center justify-content-between mb-1 gap-2">
            <div className="fw-semibold">Ответ ИИ</div>
            {canSpeakAnswer && (
              <button
                type="button"
                className={`btn btn-sm btn-outline-${speechState === 'speaking' ? 'danger' : 'secondary'}`}
                onClick={speechState === 'speaking' ? stopSpeakingAnswer : speakAnswer}
                aria-label={speechState === 'speaking' ? 'Остановить озвучивание' : 'Озвучить ответ'}
                title={speechState === 'speaking' ? 'Остановить озвучивание' : 'Озвучить ответ'}
              >
                {speechState === 'speaking' ? '⏹ Стоп' : '🔊 Озвучить'}
              </button>
            )}
          </div>
          {speechError && (
            <div className="text-danger" style={{ fontSize: '0.85rem' }}>{speechError}</div>
          )}
          <div style={{ whiteSpace: 'pre-wrap' }} dangerouslySetInnerHTML={{ __html: safeAiAnswer }} />
          {!!aiKeywords?.length && (
            <div className="mt-3">
              <div className="text-muted" style={{ fontSize: '0.85rem' }}>Ключевые термины</div>
              <div className="d-flex flex-wrap gap-2 mt-1">
                {aiKeywords.map((keyword, idx) => {
                  const positiveKey = `relevant:kw-${keyword}`
                  const negativeKey = `irrelevant:kw-${keyword}`
                  const posState = feedbackStatus[positiveKey]
                  const negState = feedbackStatus[negativeKey]
                  const statusGlyph = posState === 'done' ? '✓' : negState === 'done' ? '✗' : (posState === 'loading' || negState === 'loading') ? '…' : ''
                  return (
                    <div key={idx} className="d-flex align-items-center gap-1 border rounded px-2 py-1" style={{ background: 'var(--surface-variant, rgba(0,0,0,0.03))' }}>
                      <span className="tag mb-0" style={{ margin: 0 }}>{keyword}</span>
                      <button
                        className="btn btn-sm btn-outline-success"
                        disabled={!aiQueryHash || posState === 'loading'}
                        title="Помогает"
                        onClick={() => handleKeywordFeedback(keyword, 'relevant')}
                      >👍</button>
                      <button
                        className="btn btn-sm btn-outline-danger"
                        disabled={!aiQueryHash || negState === 'loading'}
                        title="Не подходит"
                        onClick={() => handleKeywordFeedback(keyword, 'irrelevant')}
                      >👎</button>
                      {statusGlyph && <span className="text-muted" style={{ fontSize: '0.75rem' }}>{statusGlyph}</span>}
                    </div>
                  )
                })}
              </div>
            </div>
          )}
          {!!aiFilteredKeywords?.length && (
            <div className="mt-2">
              <div className="text-muted" style={{ fontSize: '0.85rem' }}>Исключённые термины</div>
              <div className="d-flex flex-wrap gap-2 mt-1">
                {aiFilteredKeywords.map((keyword, idx) => {
                  const busyKey = `relevant:kw-${keyword}`
                  const status = feedbackStatus[busyKey]
                  return (
                    <div key={idx} className="d-inline-flex align-items-center gap-1 border rounded px-2 py-1" style={{ fontSize: 12 }}>
                      <span style={{ textDecoration: 'line-through' }}>{keyword}</span>
                      <button
                        className="btn btn-sm btn-outline-success"
                        style={{ fontSize: 11, padding: '0 6px' }}
                        disabled={status === 'loading'}
                        onClick={() => handleKeywordRestore(keyword)}
                      >
                        Вернуть
                      </button>
                      {status === 'error' && <span className="text-danger">!</span>}
                    </div>
                  )
                })}
              </div>
            </div>
          )}
          {aiSources && aiSources.length > 0 && (
            <div className="mt-2">
              <div className="fw-semibold mb-1">Источники</div>
              <ol>
                {aiSources.map((source: any, idx: number) => {
                  const rel = source.rel_path || ''
                  const snippets: string[] = Array.isArray(source.snippets) ? source.snippets.slice(0, 3) : []
                  const baseTerms = aiKeywords || []
                  const primary = snippets[0] ? baseTerms.filter(k => String(snippets[0]).toLowerCase().includes(String(k).toLowerCase())) : []
                  const terms = (primary.length ? primary : baseTerms).slice(0, 5)
                  const mark = encodeURIComponent(terms.join('|'))
                  const href = rel ? `/preview/${encodeURIComponent(rel)}?embedded=1&mark=${mark}` : '#'
                  const title = source.title || rel || `file-${source.file_id}`
                  const esc = (value: string) => value.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c] || c))
                  const hi = (text: string) => {
                    if (!text) return ''
                    const arr = terms.length ? terms : baseTerms.slice(0, 5)
                    if (!arr.length) return esc(text)
                    const re = new RegExp('(' + arr.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')', 'gi')
                    return esc(text).replace(re, '<mark>$1</mark>')
                  }
                  const irrelevantKey = `irrelevant:file-${source.file_id}`
                  const relevantKey = `relevant:file-${source.file_id}`
                  const isIrrelevant = feedbackStatus[irrelevantKey] === 'done'
                  const isRelevant = feedbackStatus[relevantKey] === 'done'
                  return (
                    <li key={idx} className={isIrrelevant ? 'text-muted text-decoration-line-through' : ''}>
                      <a href={href} target="_blank" rel="noopener" className={isIrrelevant ? 'text-muted text-decoration-line-through' : ''}>[{idx + 1}] {title}</a>
                      {snippets.length > 0 && (
                        <div className="muted" style={{ fontSize: 13 }}>
                          {snippets.map((snippet, snippetIdx) => (
                            <div key={snippetIdx} dangerouslySetInnerHTML={{ __html: ' — ' + hi(String(snippet)) }} />
                          ))}
                        </div>
                      )}
                      <div className="mt-1 d-flex align-items-center gap-2">
                        <button className="btn btn-sm btn-outline-success" disabled={!aiQueryHash || feedbackStatus[`relevant:file-${source.file_id}`] === 'loading'} onClick={() => handleSourceFeedback(Number(source.file_id), 'relevant')}>Подходит</button>
                        <button className="btn btn-sm btn-outline-danger" disabled={!aiQueryHash || feedbackStatus[`irrelevant:file-${source.file_id}`] === 'loading'} onClick={() => handleSourceFeedback(Number(source.file_id), 'irrelevant')}>Не подходит</button>
                        {(feedbackStatus[`relevant:file-${source.file_id}`] === 'done' || feedbackStatus[`irrelevant:file-${source.file_id}`] === 'done') && (
                          <span className="text-success" style={{ fontSize: '0.8rem' }}>Спасибо!</span>
                        )}
                        {(feedbackStatus[`relevant:file-${source.file_id}`] === 'error' || feedbackStatus[`irrelevant:file-${source.file_id}`] === 'error') && (
                          <span className="text-danger" style={{ fontSize: '0.8rem' }}>Ошибка</span>
                        )}
                        {isIrrelevant && <span className="badge bg-danger-subtle text-danger" style={{ fontSize: '0.7rem' }}>Помечено как нерелевантное</span>}
                        {isRelevant && <span className="badge bg-success-subtle text-success" style={{ fontSize: '0.7rem' }}>Помечено как релевантное</span>}
                      </div>
                      {source.llm_snippet && (
                        <div className="mt-1" style={{ fontSize: 13 }}>
                          <strong>Ответ LLM:</strong> {source.llm_snippet}
                        </div>
                      )}
                    </li>
                  )
                })}
              </ol>
            </div>
          )}
        </div>
      )}
    </>
  )
}

export default AiPanel
