import React, { createContext, useCallback, useContext, useMemo, useState } from 'react'

type Toast = { id: number; kind?: 'success'|'error'|'info'; text: string; actions?: React.ReactNode }
type Ctx = { push: (text: string, kind?: Toast['kind'], actions?: React.ReactNode) => void }
const ToastCtx = createContext<Ctx>({ push: () => {} })

export function useToasts(){ return useContext(ToastCtx) }

export default function ToastProvider({ children }: { children: React.ReactNode }){
  const [items, setItems] = useState<Toast[]>([])
  const push = useCallback((text: string, kind: Toast['kind']='info', actions?: React.ReactNode) => {
    const id = Date.now() + Math.floor(Math.random()*1000)
    setItems(arr => [...arr, { id, kind, text, actions }])
    const timeout = actions ? 10000 : 4000
    window.setTimeout(() => setItems(arr => arr.filter(x => x.id !== id)), timeout)
  }, [])
  const ctx = useMemo(() => ({ push }), [push])
  return (
    <ToastCtx.Provider value={ctx}>
      {children}
      <div className="app-toast-stack" aria-live="polite" aria-atomic="false">
        {items.map(t => (
          <div key={t.id} role="status" className={`app-toast app-toast--${t.kind || 'info'}`}>
            <div className="app-toast__row">
              <div>
                <strong className="me-2">{t.kind==='error'?'Ошибка': t.kind==='success'?'Готово':'Сообщение'}</strong>
                <span>{t.text}</span>
              </div>
              {t.actions && <div className="app-toast__actions">{t.actions}</div>}
            </div>
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  )
}
