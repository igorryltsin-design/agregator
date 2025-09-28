import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../ui/Auth'
import agregatorLogo from '../../logo/agregator.png'

type LoginBackground = {
  name: string
  label: string
  url: string
}

const DEFAULT_BACKGROUNDS: LoginBackground[] = [
  { name: 'acsi.html', label: 'ASCII абстракции', url: '/login-backgrounds/acsi.html' },
  { name: 'Labirint.html', label: 'Лабиринт', url: '/login-backgrounds/Labirint.html' },
  { name: 'Cloud.html', label: 'Поток символов', url: '/login-backgrounds/Cloud.html' },
  { name: 'fractal.html', label: 'Фрактальное поле', url: '/login-backgrounds/fractal.html' }
]

const uniqueBackgrounds = (list: LoginBackground[]) =>
  list.filter((bg, idx, arr) => arr.findIndex(item => item.name === bg.name) === idx)

export default function LoginPage(){
  const { login, user, loading } = useAuth()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [backgrounds, setBackgrounds] = useState<LoginBackground[]>(DEFAULT_BACKGROUNDS)
  const [backgroundIndex, setBackgroundIndex] = useState(() => (
    DEFAULT_BACKGROUNDS.length ? Math.floor(Math.random() * DEFAULT_BACKGROUNDS.length) : 0
  ))
  const [backgroundNonce, setBackgroundNonce] = useState(0)
  const backgroundsInitialised = useRef(false)

  const navigate = useNavigate()
  const location = useLocation()
  const from = (location.state as { from?: string } | undefined)?.from || '/'

  useEffect(() => {
    if (!loading && user) {
      navigate(from, { replace: true })
    }
  }, [loading, user, navigate, from])

  const fetchBackgrounds = useCallback(async (signal?: AbortSignal) => {
    try {
      const res = await fetch('/api/login-backgrounds', {
        cache: 'no-store',
        credentials: 'same-origin',
        signal
      })
      if (!res.ok) throw new Error(`status ${res.status}`)
      const payload = await res.json()
      if (signal?.aborted) return [] as LoginBackground[]
      const items = (Array.isArray(payload) ? payload : []).filter((item): item is LoginBackground => {
        return item && typeof item === 'object' && typeof item.name === 'string' && typeof item.url === 'string'
      }).map(item => ({
        name: item.name,
        url: item.url,
        label: typeof item.label === 'string' && item.label.trim() ? item.label.trim() : item.name
      }))
      const merged = items.length ? items : DEFAULT_BACKGROUNDS
      const deduped = uniqueBackgrounds(merged)
      setBackgrounds(deduped)
      setBackgroundIndex(prev => {
        if (!deduped.length) {
          backgroundsInitialised.current = true
          return 0
        }
        if (!backgroundsInitialised.current) {
          backgroundsInitialised.current = true
          return Math.floor(Math.random() * deduped.length)
        }
        return Math.min(prev, deduped.length - 1)
      })
      return deduped
    } catch (err) {
      if (signal?.aborted) return [] as LoginBackground[]
      console.warn('[login] loading backgrounds failed', err)
      const deduped = uniqueBackgrounds(DEFAULT_BACKGROUNDS)
      setBackgrounds(deduped)
      setBackgroundIndex(prev => {
        if (!deduped.length) {
          backgroundsInitialised.current = true
          return 0
        }
        if (!backgroundsInitialised.current) {
          backgroundsInitialised.current = true
          return Math.floor(Math.random() * deduped.length)
        }
        return Math.min(prev, deduped.length - 1)
      })
      return deduped
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    fetchBackgrounds(controller.signal).catch(() => {})
    return () => controller.abort()
  }, [fetchBackgrounds])

  const handleBackgroundToggle = useCallback(() => {
    const count = backgrounds.length
    if (count === 0) {
      fetchBackgrounds().catch(() => {})
      return
    }
    if (count === 1) {
      setBackgroundNonce(n => n + 1)
      return
    }
    setBackgroundIndex(idx => (idx + 1) % count)
  }, [backgrounds, fetchBackgrounds])

  const safeIndex = backgrounds.length ? backgroundIndex % backgrounds.length : 0
  const currentBackground = backgrounds.length ? backgrounds[safeIndex] : undefined
  const backgroundKey = `${currentBackground?.name ?? 'void'}-${backgroundNonce}-${safeIndex}`
  const backgroundTitle = currentBackground ? `Фон: ${currentBackground.label}` : 'Фон не найден'
  const submit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    setBusy(true)
    setError('')
    const res = await login(username.trim(), password)
    setBusy(false)
    if (res.ok) {
      navigate(from, { replace: true })
    } else {
      setError(res.error || 'Не удалось войти')
    }
  }

  return (
    <div className="login-screen">
      <div className="login-background-host" aria-hidden="true">
        <div className="login-background-fallback"></div>
        {currentBackground ? (
          <iframe
            key={backgroundKey}
            src={`${currentBackground.url}?slot=${encodeURIComponent(backgroundKey)}`}
            title={backgroundTitle}
            className="login-background-frame"
            tabIndex={-1}
            aria-hidden="true"
            loading="lazy"
          />
        ) : null}
        <div className="login-background-veil" aria-hidden="true"></div>
      </div>
      <div className="login-card card p-4">
        <div className="text-center mb-3">
          <button
            type="button"
            className="app-brand-badge app-brand-badge--lg login-logo-toggle"
            tabIndex={-1}
            onMouseDown={event => {
              event.preventDefault()
              event.currentTarget.blur()
            }}
            onClick={() => {
              handleBackgroundToggle()
            }}
            title="Сменить фон"
            aria-label="Сменить фон"
          >
            <img src={agregatorLogo} alt="Agregator" />
          </button>
        </div>
        <div className="fw-semibold fs-5 mb-3 text-center">Вход в систему</div>
  
        <form className="d-grid gap-3" onSubmit={submit}>
          <div>
            <label className="form-label">Логин</label>
            <input className="form-control" value={username} onChange={e=>setUsername(e.target.value)} autoFocus autoComplete="username" />
          </div>
          <div>
            <label className="form-label">Пароль</label>
            <input className="form-control" type="password" value={password} onChange={e=>setPassword(e.target.value)} autoComplete="current-password" />
          </div>
          {error && <div className="alert alert-danger py-2 m-0">{error}</div>}
          <button className="btn btn-primary" type="submit" disabled={busy}>{busy ? 'Вход…' : 'Войти'}</button>
        </form>
        <div className="text-muted mt-3" style={{ fontSize: 13 }}>
          Доступ предоставляется администраторами. После входа будут доступны функции согласно вашей роли.
        </div>
      </div>
    </div>
  )
}
