import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, Outlet, useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import { useToasts } from '../ui/Toasts'
import { useAuth } from '../ui/Auth'
import VoiceSearchButton from '../ui/VoiceSearchButton'
import ProgressPanel, { ProgressBullet } from '../ui/ProgressPanel'
import agregatorLogo from '../../logo/agregator.png'
import aiWordLogo from '../../logo/AIWord.png'

export default function App() {
  const [sp, setSp] = useSearchParams()
  const nav = useNavigate()
  const location = useLocation()
  const { user, loading, logout } = useAuth()
  const q = sp.get('q') || ''
  const [searchDraft, setSearchDraft] = useState(q)
  const [theme, setTheme] = useState<string>(()=>{
    const saved = localStorage.getItem('theme')
    if (saved) return saved
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
    return prefersDark ? 'dark' : 'light'
  })
  const searchRef = useRef<HTMLInputElement|null>(null)
  const [adminMenuOpen, setAdminMenuOpen] = useState(false)
  const [scanOpen, setScanOpen] = useState(false)
  const [scanMin, setScanMin] = useState(false)
  const [scanRunning, setScanRunning] = useState(false)
  const [scanStat, setScanStat] = useState<any>(null)
  const scanMetrics = useMemo(() => {
    if (!scanStat) return null
    const processed = Number(scanStat.processed || 0)
    const total = Number(scanStat.total || 0)
    const added = Number(scanStat.added || 0)
    const updated = Number(scanStat.updated || 0)
    const current = scanStat.current ? String(scanStat.current) : ''
    const etaSeconds = Number(scanStat.eta_seconds || 0)
    const percent = total > 0 ? Math.min(100, Math.round((processed * 100) / total)) : 0
    const fmt = (s: number) => {
      const hh = Math.floor(s / 3600)
      const mm = Math.floor((s % 3600) / 60)
      const ss = Math.floor(s % 60)
      return `${hh.toString().padStart(2, '0')}:${mm.toString().padStart(2, '0')}:${ss.toString().padStart(2, '0')}`
    }
    const etaText = etaSeconds > 0 ? `, ETA: ${fmt(etaSeconds)}` : ''
    return {
      percent,
      summary: total > 0 ? `Обработано ${processed}/${total} (${percent}%)` : 'Прогресс недоступен',
      detail: `Добавлено ${added}, обновлено ${updated}${etaText}`,
      current,
    }
  }, [scanStat])
  const scanScopeCaption = useMemo(() => {
    if (!scanStat?.scope?.label) return undefined
    const count = typeof scanStat.scope.count === 'number' ? ` · файлов: ${scanStat.scope.count}` : ''
    return `Область: ${scanStat.scope.label}${count}`
  }, [scanStat])
  const scanBullets = useMemo<ProgressBullet[]>(() => {
    if (!scanMetrics) return []
    const items: ProgressBullet[] = [{ id: 'detail', text: scanMetrics.detail }]
    if (scanMetrics.current) {
      items.push({ id: 'current', text: `Текущий файл: ${scanMetrics.current}` })
    }
    return items
  }, [scanMetrics])
  const [showHelp, setShowHelp] = useState(false)
  const toasts = useToasts()
  const isAdmin = user?.role === 'admin'
  const canUseAiword = !!user?.aiword_access
  const canImport = !!user?.can_import || isAdmin
  const adminMenuRef = useRef<HTMLDivElement | null>(null)
  const helpDialogRef = useRef<HTMLDivElement | null>(null)
  const hotkeySequenceRef = useRef<string | null>(null)
  const hotkeyTimerRef = useRef<number | null>(null)
  const helpTitleId = 'agregator-help-title'
  const helpDescId = 'agregator-help-desc'
  const iconButtonClass = 'btn btn-outline-secondary icon-only'

  useEffect(() => {
    if (!loading && !user) {
      const from = location.pathname + location.search
      nav('/login', { replace: true, state: { from } })
    }
  }, [loading, user, location.pathname, location.search, nav])

  useEffect(() => {
    if (!loading && user) {
      document.documentElement.setAttribute('data-theme', theme)
      localStorage.setItem('theme', theme)
    }
  }, [theme, loading, user])

  useEffect(() => {
    setSearchDraft(q)
  }, [q])

  useEffect(() => {
    if (!adminMenuOpen) return
    const onClickOutside = (event: MouseEvent) => {
      if (!adminMenuRef.current?.contains(event.target as Node)) {
        setAdminMenuOpen(false)
      }
    }
    const onKeyClose = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setAdminMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', onClickOutside)
    document.addEventListener('keydown', onKeyClose)
    return () => {
      document.removeEventListener('mousedown', onClickOutside)
      document.removeEventListener('keydown', onKeyClose)
    }
  }, [adminMenuOpen])

  useEffect(() => {
    setAdminMenuOpen(false)
  }, [location.pathname, location.search])

  useEffect(() => {
    const resetSequence = () => {
      if (hotkeyTimerRef.current) {
        window.clearTimeout(hotkeyTimerRef.current)
        hotkeyTimerRef.current = null
      }
      hotkeySequenceRef.current = null
    }

    const isEditableTarget = (node: EventTarget | null) => {
      if (!(node instanceof HTMLElement)) return false
      const tag = node.tagName
      return node.isContentEditable || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || node.getAttribute('role') === 'textbox'
    }

    const navigateByHotkey = (key: string) => {
      const mapping: Array<{ key: string; path: string; requireAdmin?: boolean; requireImport?: boolean }> = [
        { key: 'g', path: '/' },
        { key: 's', path: '/stats' },
        { key: 'i', path: '/ingest', requireImport: true },
        { key: 'a', path: '/admin/status', requireAdmin: true },
        { key: 't', path: '/admin/tasks', requireAdmin: true },
        { key: 'l', path: '/admin/logs', requireAdmin: true },
        { key: 'm', path: '/admin/ai-metrics', requireAdmin: true },
        { key: 'c', path: '/admin/collections', requireAdmin: true },
        { key: 'u', path: '/users', requireAdmin: true },
      ]
      const entry = mapping.find(item => item.key === key.toLowerCase())
      if (!entry) return false
      if (entry.requireAdmin && !isAdmin) return false
      if (entry.requireImport && !canImport) return false
      nav(entry.path)
      setAdminMenuOpen(false)
      return true
    }

    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return
      const target = event.target as HTMLElement | null
      const isEditable = isEditableTarget(target)
      const plain = !event.metaKey && !event.ctrlKey && !event.altKey

      if (plain && event.key === 'Escape') {
        if (searchRef.current === document.activeElement) {
          searchRef.current?.blur()
        }
        setAdminMenuOpen(false)
        resetSequence()
        return
      }

      if (!isEditable) {
        if (plain && event.key === '/') {
          event.preventDefault()
          searchRef.current?.focus()
          resetSequence()
          return
        }
        if (event.shiftKey && !event.ctrlKey && !event.metaKey && event.key === '?') {
          event.preventDefault()
          setShowHelp(true)
          resetSequence()
          return
        }
        if (plain && event.key.toLowerCase() === 'g') {
          hotkeySequenceRef.current = 'g'
          if (hotkeyTimerRef.current) window.clearTimeout(hotkeyTimerRef.current)
          hotkeyTimerRef.current = window.setTimeout(resetSequence, 1500)
          return
        }
        if (hotkeySequenceRef.current === 'g' && !event.metaKey && !event.ctrlKey) {
          const handled = navigateByHotkey(event.key)
          resetSequence()
          if (handled) {
            event.preventDefault()
          }
          return
        }
      } else if (hotkeySequenceRef.current) {
        resetSequence()
      }
    }

    const onScanOpen = () => setScanOpen(true)
    window.addEventListener('keydown', onKey)
    window.addEventListener('scan-open', onScanOpen as any)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('scan-open', onScanOpen as any)
      resetSequence()
    }
  }, [canImport, isAdmin, nav, setAdminMenuOpen, setShowHelp])

  useEffect(() => {
    if (!scanOpen) return
    let timer: ReturnType<typeof setTimeout> | null = null
    const poll = async () => {
      try {
        const r = await fetch('/scan/status')
        if (!r.ok) throw new Error('status')
        const j = await r.json()
        setScanRunning(!!j.running)
        setScanStat(j)
      } catch {
        // Игнорируем временные ошибки опроса статуса сканирования
      } finally {
        timer = setTimeout(poll, 1500)
      }
    }
    poll()
    return () => { if (timer) clearTimeout(timer) }
  }, [scanOpen])

  useEffect(() => {
    if (!showHelp) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setShowHelp(false)
      }
    }
    const node = helpDialogRef.current
    node?.focus()
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('keydown', onKey)
    }
  }, [showHelp])

  const startScan = async () => {
    if (!isAdmin) {
      toasts.push('Недостаточно прав для запуска сканирования', 'error')
      return
    }
    try {
      await fetch('/scan/start', { method: 'POST' })
      setScanOpen(true)
      setScanMin(false)
      toasts.push('Сканирование запущено', 'success')
    } catch {
      toasts.push('Не удалось запустить сканирование', 'error')
    }
  }

  const cancelScan = async () => {
    if (!isAdmin) {
      toasts.push('Недостаточно прав', 'error')
      return
    }
    try {
      await fetch('/scan/cancel', { method: 'POST' })
      toasts.push('Сканирование остановлено', 'success')
    } catch {
      toasts.push('Ошибка отмены сканирования', 'error')
    }
  }

  const commitSearch = useCallback((value: string, replace = true) => {
    const next = new URLSearchParams(sp)
    const trimmed = value.trim()
    if (trimmed) {
      next.set('q', trimmed)
    } else {
      next.delete('q')
    }
    next.set('page', '1')
    next.set('commit', String(Date.now()))
    setSp(next, { replace })
  }, [sp, setSp])

  const handleVoiceSearch = useCallback((text: string) => {
    const normalized = text.trim()
    if (!normalized) {
      return
    }
    setSearchDraft(normalized)
    commitSearch(normalized, true)
  }, [commitSearch])

  const handleVoiceError = useCallback((message: string) => {
    if (!message) return
    toasts.push(message, 'error')
  }, [toasts])

  const handleLogout = async () => {
    setAdminMenuOpen(false)
    await logout()
    nav('/login', { replace: true })
  }

  const adminMenuItems = useMemo(() => ([
    { to: 'admin/status', label: 'Состояние' },
    { to: 'settings', label: 'Настройки' },
    { to: 'users', label: 'Пользователи' },
    { to: 'admin/tasks', label: 'Задачи' },
    { to: 'admin/logs', label: 'Логи' },
    { to: 'admin/llm', label: 'LLM' },
    { to: 'admin/ai-metrics', label: 'AI метрики' },
    { to: 'admin/facets', label: 'Фасеты' },
    { to: 'admin/collections', label: 'Коллекции' },
  ]), [])

  if (loading) {
    return (
      <div className="d-flex align-items-center justify-content-center" style={{ minHeight: '100vh' }}>
        Загрузка интерфейса…
      </div>
    )
  }

  if (!user) {
    return null
  }

  return (
    <div>
      <nav className="navbar navbar-expand px-3 py-2 mb-3">
        <div className="d-flex align-items-center gap-2 w-100">
          <Link className="navbar-brand m-0" to="/" style={{ color: 'var(--text)' }}>
            <span className="app-brand-badge">
              <img src={agregatorLogo} alt="Agregator" />
            </span>
          </Link>
          <div className="d-flex align-items-center gap-2" style={{ width: 600, maxWidth: '60%', flex: '0 1 auto' }}>
            <div className="position-relative flex-grow-1">
              <input
                className="form-control pe-5"
                placeholder="Поиск…"
                value={searchDraft}
                ref={searchRef}
                onChange={e => setSearchDraft(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); commitSearch(searchDraft, true) } }}
              />
              <span className="hotkey-hint d-none d-md-inline" aria-hidden="true" style={{ position: 'absolute', top: '50%', right: 12, transform: 'translateY(-50%)' }}>/</span>
            </div>
            <button
              className={iconButtonClass}
              type="button"
              onClick={() => commitSearch(searchDraft, true)}
              aria-label="Выполнить поиск"
              data-tooltip="Выполнить поиск"
            >
              <span className="icon-glyph" aria-hidden="true">🔍</span>
            </button>
          </div>
          <VoiceSearchButton onTranscribed={handleVoiceSearch} onError={handleVoiceError} />
          <div className="ms-auto d-flex align-items-center gap-2 flex-wrap justify-content-end" style={{ rowGap: '0.3rem' }}>
            <Link
              className={`${iconButtonClass} doc-chat-nav-icon`}
              to="doc-chat"
              aria-label="Чат по документу"
              data-tooltip="Чат по документу"
            >
              <span className="icon-glyph" aria-hidden="true">💬</span>
            </Link>
            <button
              className={iconButtonClass}
              type="button"
              onClick={() => setShowHelp(true)}
              aria-label="Справка по Agregator"
              data-tooltip="Справка"
            >
              <span className="icon-glyph" aria-hidden="true">❔</span>
            </button>
            <button
              className={iconButtonClass}
              type="button"
              onClick={()=> setTheme(t => t==='dark'?'light':'dark')}
              aria-label="Переключить тему"
              data-tooltip={theme==='dark' ? 'Светлая тема' : 'Тёмная тема'}
            >
              <span className="icon-glyph" aria-hidden="true">{theme==='dark'?'🌙':'☀️'}</span>
            </button>
            <Link className={iconButtonClass} to="graph" aria-label="Граф" data-tooltip="Граф">
              <span className="icon-glyph" aria-hidden="true">🕸️</span>
            </Link>
            <Link className={iconButtonClass} to="stats" aria-label="Статистика" data-tooltip="Статистика">
              <span className="icon-glyph" aria-hidden="true">📊</span>
            </Link>
            {canImport && (
              <Link className={iconButtonClass} to="ingest" aria-label="Импорт" data-tooltip="Импорт">
                <span className="icon-glyph" aria-hidden="true">📥</span>
              </Link>
            )}
            {isAdmin && (
              <div className="position-relative" ref={adminMenuRef}>
                <button className="btn btn-outline-secondary" type="button" onClick={() => setAdminMenuOpen(v => !v)} aria-expanded={adminMenuOpen}>
                  Администрирование ▾
                </button>
                {adminMenuOpen && (
                  <div className="border rounded-3" style={{ position: 'absolute', top: 'calc(100% + 4px)', right: 0, minWidth: 200, background: 'var(--surface)', boxShadow: 'var(--card-shadow)', zIndex: 1500 }}>
                    {adminMenuItems.map(item => (
                      <Link
                        key={item.to}
                        className="d-block px-3 py-2 text-decoration-none"
                        style={{ color: 'var(--text)' }}
                        to={item.to}
                        onClick={() => setAdminMenuOpen(false)}
                      >
                        {item.label}
                      </Link>
                    ))}
                  </div>
                )}
              </div>
            )}
            <Link className={iconButtonClass} to="profile" aria-label="Профиль" data-tooltip="Профиль">
              <span className="icon-glyph" aria-hidden="true">👤</span>
            </Link>
            <span className="badge bg-secondary text-uppercase" style={{ letterSpacing: 0.3 }}>
              {user.role === 'admin' ? 'Админ' : user.role === 'editor' ? 'Редактор' : 'Наблюдатель'}
            </span>
            <button className={iconButtonClass} type="button" onClick={handleLogout} aria-label="Выйти" data-tooltip="Выйти">
              <span className="icon-glyph" aria-hidden="true">🚪</span>
            </button>
            {canUseAiword && (
              <a
                className="aiword-launch-link d-flex align-items-center"
                href="/aiword"
                target="_blank"
                rel="noopener"
                aria-label="AIWord"
                data-tooltip="AIWord"
              >
                <span className="aiword-logo-badge aiword-logo-badge--nav">
                  <img src={aiWordLogo} alt="AIWord" />
                </span>
                <span className="visually-hidden">AIWord</span>
              </a>
            )}
          </div>
        </div>
      </nav>
      {showHelp && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby={helpTitleId}
          aria-describedby={helpDescId}
          className="position-fixed top-0 start-0 w-100 h-100 d-flex align-items-center justify-content-center"
          style={{ background: 'rgba(0, 0, 0, 0.6)', zIndex: 2000 }}
          onClick={() => setShowHelp(false)}
        >
          <div
            ref={helpDialogRef}
            tabIndex={-1}
            className="rounded-4 shadow-lg"
            style={{ background: 'var(--surface)', color: 'var(--text)', maxWidth: 720, width: '90%', padding: '24px 28px', border: '1px solid var(--border)' }}
            onClick={event => event.stopPropagation()}
          >
            <div className="d-flex justify-content-between align-items-start mb-3 gap-3">
              <div>
                <h2 id={helpTitleId} className="h4 mb-2" style={{ color: 'var(--text)' }}>Agregator — справка</h2>
                <p id={helpDescId} className="mb-0 muted" style={{ color: 'var(--muted)' }}>
                  Краткое описание назначения каталога, основных разделов интерфейса и полезных горячих клавиш.
                </p>
              </div>
              <button className="btn btn-outline-secondary" type="button" onClick={() => setShowHelp(false)} aria-label="Закрыть справку">✕</button>
            </div>
            <div className="d-grid gap-3" style={{ fontSize: 14, lineHeight: 1.5 }}>
              <section>
                <h3 className="h6 mb-2" style={{ color: 'var(--text)' }}>О приложении</h3>
                <p className="mb-0">
                  Agregator — локальный каталог научных материалов с поддержкой полнотекстового и AI-поиска, расширенными
                  метаданными и управлением доступом. Приложение индексирует выбранные директории, извлекает текст и теги,
                  строит связи «файл ↔ коллекция ↔ теги» и предоставляет единый интерфейс для работы команды.
                </p>
              </section>
              <section>
                <h3 className="h6 mb-2" style={{ color: 'var(--text)' }}>Главные разделы интерфейса</h3>
                <ul className="mb-0" style={{ paddingLeft: 18 }}>
                  <li><strong>Поиск</strong> — основная лента материалов с фильтрами по типу, тегам, коллекциям и AI-ответами.</li>
                  <li><strong>Граф</strong> — визуализация связей между файлами, тегами и коллекциями для быстрого анализа тем.</li>
                  <li><strong>Статистика</strong> — обзор по количеству материалов, динамике пополнения и заполненности метаданных.</li>
                  {canImport && <li><strong>Импорт</strong> — загрузка новых файлов, создание коллекций и запуск пакетного сканирования.</li>}
                  <li><strong>Профиль</strong> — персональные настройки, API-ключи и права доступа.</li>
                  {isAdmin && <li><strong>Администрирование</strong> — управление пользователями, задачами сканирования, журналом действий и LLM-эндпоинтами.</li>}
                  {canUseAiword && <li><strong>AIWord</strong> — отдельный редактор научных текстов, открывается в новой вкладке.</li>}
                </ul>
              </section>
              <section>
                <h3 className="h6 mb-2" style={{ color: 'var(--text)' }}>Режимы поиска</h3>
                <p className="mb-2">
                  В строке поиска доступны два сценария: классический полнотекстовый поиск и AI-поиск с генерацией кратких ответов.
                  Между режимами можно переключаться над результатами выдачи.
                </p>
                <ul className="mb-0" style={{ paddingLeft: 18 }}>
                  <li>
                    <strong>Классический поиск</strong> — обрабатывает запрос мгновенно, использует морфологию, синонимы и фильтры.
                    Результаты сортируются по релевантности и дате; запуск осуществляется клавишей Enter или иконкой 🔍, после чего
                    выдачу можно уточнять тегами, коллекциями и годами.
                  </li>
                  <li>
                    <strong>AI-поиск</strong> — расширяет запрос LLM, реранжирует найденные материалы и формирует краткий ответ с цитатами.
                    Работает чуть медленнее, отображает индикатор прогресса и стримит результат; использует модели, заданные
                    администратором в разделе «LLM».
                  </li>
                </ul>
              </section>
              <section>
                <h3 className="h6 mb-2" style={{ color: 'var(--text)' }}>Сканирование и обновление каталога</h3>
                <p className="mb-0">
                  Используйте кнопку «Сканировать» в правом нижнем углу (или раздел «Импорт») для запуска обхода файлов. Во время
                  процесса отображается статистика и прогресс. Доступно ручное обновление отдельного файла через карточку материала.
                </p>
              </section>
              <section>
                <h3 className="h6 mb-2" style={{ color: 'var(--text)' }}>AI-помощь</h3>
                <p className="mb-0">
                  AI-ответы используют те же отфильтрованные документы, поэтому классический поиск помогает подготовить выдачу.
                  Вкладка AI отображает ход обработки, подсвечивает источники и доступна только при настроенных LLM-эндпоинтах в
                  админ-панели.
                </p>
              </section>
              <section>
                <h3 className="h6 mb-2" style={{ color: 'var(--text)' }}>Горячие клавиши</h3>
                <ul className="mb-0" style={{ paddingLeft: 18 }}>
                  <li><code>/</code> — фокус на поле поиска.</li>
                  <li><code>Shift + ?</code> — открыть эту справку.</li>
                  <li><code>Esc</code> — закрыть выпадающие окна и диалоги.</li>
                  <li><code>g</code> затем <code>g</code> — перейти в каталог материалов.</li>
                  <li><code>g</code> затем <code>s</code> — открыть статистику.</li>
                  {canImport && <li><code>g</code> затем <code>i</code> — перейти в раздел импорта.</li>}
                  {isAdmin && <li><code>g</code> затем <code>a</code> — открыть «Состояние сервиса».</li>}
                  {isAdmin && <li><code>g</code> затем <code>t</code> — перейти к задачам.</li>}
                  {isAdmin && <li><code>g</code> затем <code>m</code> — открыть AI-метрики.</li>}
                  {isAdmin && <li><code>g</code> затем <code>c</code> — управление коллекциями.</li>}
                  {isAdmin && <li><code>g</code> затем <code>u</code> — список пользователей.</li>}
                </ul>
              </section>
            </div>
            <div className="text-end mt-4">
              <button className="btn btn-primary" type="button" onClick={() => setShowHelp(false)}>
                Понятно
              </button>
            </div>
          </div>
        </div>
      )}
      <div className="container-fluid">
        <Outlet />
      </div>

      {scanOpen && (
        <div role="dialog" aria-modal="false" style={{ position: 'fixed', right: 16, bottom: 16, width: scanMin ? 320 : 420, zIndex: 1600 }}>
          {scanMin ? (
            <div className="card p-3" style={{ boxShadow: 'var(--card-shadow)' }}>
              <div className="d-flex align-items-start justify-content-between gap-3">
                <div className="d-grid gap-1">
                  <div>{scanRunning ? 'Сканирование…' : 'Сканер остановлен'}</div>
                  {scanScopeCaption && <div className="muted" style={{ fontSize: 12 }}>{scanScopeCaption}</div>}
                  {scanMetrics && <div className="muted" style={{ fontSize: 12 }}>{scanMetrics.summary}</div>}
                </div>
                <div className="btn-group btn-group-sm">
                  <button className="btn btn-outline-secondary" onClick={() => setScanMin(false)}>Развернуть</button>
                  <button className="btn btn-outline-secondary" onClick={() => setScanOpen(false)}>Закрыть</button>
                </div>
              </div>
            </div>
          ) : (
            <ProgressPanel
              title={scanRunning ? 'Сканирование…' : 'Сканер остановлен'}
              caption={scanScopeCaption}
              progress={scanMetrics ? { percent: scanMetrics.percent, label: scanMetrics.summary } : undefined}
              bullets={scanBullets}
              footer={isAdmin ? (
                <div className="d-flex gap-2">
                  <button className="btn btn-sm btn-outline-secondary" disabled={scanRunning} onClick={startScan}>Запустить</button>
                  <button className="btn btn-sm btn-outline-secondary" disabled={!scanRunning} onClick={cancelScan}>Остановить</button>
                </div>
              ) : undefined}
              actions={
                <div className="btn-group btn-group-sm">
                  <button className="btn btn-outline-secondary" onClick={() => setScanMin(true)}>Свернуть</button>
                  <button className="btn btn-outline-secondary" onClick={() => setScanOpen(false)}>Закрыть</button>
                </div>
              }
              className="mb-0"
              style={{ boxShadow: 'var(--card-shadow)' }}
            />
          )}
        </div>
      )}

    </div>
  )
}
