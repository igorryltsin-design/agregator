import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'

type VoiceSearchButtonProps = {
  onTranscribed: (text: string) => void
  onError?: (message: string) => void
  disabled?: boolean
}

type RecorderState = 'idle' | 'recording' | 'processing'

export default function VoiceSearchButton({ onTranscribed, onError, disabled }: VoiceSearchButtonProps) {
  const [recorderState, setRecorderState] = useState<RecorderState>('idle')
  const [supported, setSupported] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) && typeof window.MediaRecorder !== 'undefined'
  })
  const [permissionDenied, setPermissionDenied] = useState<boolean>(false)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const stopTimerRef = useRef<number | null>(null)

  useEffect(() => {
    setSupported(prev => {
      if (prev) return prev
      if (typeof window === 'undefined') return false
      return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) && typeof window.MediaRecorder !== 'undefined'
    })
  }, [])

  useEffect(() => {
    return () => {
      if (stopTimerRef.current !== null) {
        window.clearTimeout(stopTimerRef.current)
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        try {
          mediaRecorderRef.current.stop()
        } catch (e) {
          // Игнорируем ошибку остановки медиарекордера
        }
      }
      mediaRecorderRef.current = null
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop())
        mediaStreamRef.current = null
      }
    }
  }, [])

  const notifyError = useCallback((message: string) => {
    onError?.(message)
  }, [onError])

  const stopRecordingInternal = useCallback((opts?: { cancelled?: boolean }) => {
    if (stopTimerRef.current !== null) {
      window.clearTimeout(stopTimerRef.current)
      stopTimerRef.current = null
    }
    const recorder = mediaRecorderRef.current
    if (!recorder) return
    const needsStop = recorder.state === 'recording'
    if (needsStop) {
      setRecorderState(prev => (opts?.cancelled ? 'idle' : 'processing'))
      try {
        recorder.stop()
      } catch (err) {
        notifyError('Не удалось остановить запись')
        setRecorderState('idle')
      }
    }
  }, [notifyError])

  const cleanupStream = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop())
      mediaStreamRef.current = null
    }
  }, [])

  const startRecording = useCallback(async () => {
    if (recorderState !== 'idle' || disabled || !supported) return
    setPermissionDenied(false)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaStreamRef.current = stream

      const preferredMime = (() => {
        if (typeof window === 'undefined' || !window.MediaRecorder) return ''
        if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) return 'audio/webm;codecs=opus'
        if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) return 'audio/ogg;codecs=opus'
        if (MediaRecorder.isTypeSupported('audio/webm')) return 'audio/webm'
        return ''
      })()

      const recorder = preferredMime
        ? new MediaRecorder(stream, { mimeType: preferredMime })
        : new MediaRecorder(stream)
      mediaRecorderRef.current = recorder
      chunksRef.current = []

      recorder.ondataavailable = event => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      recorder.onerror = event => {
        notifyError('Ошибка записи аудио')
        setRecorderState('idle')
        cleanupStream()
      }

      recorder.onstop = async () => {
        const blob = chunksRef.current.length
          ? new Blob(chunksRef.current, { type: recorder.mimeType || preferredMime || 'audio/webm' })
          : null
        chunksRef.current = []
        cleanupStream()
        if (recorderState === 'recording') {
          setRecorderState('idle')
        }
        if (!blob || blob.size === 0) {
          setRecorderState('idle')
          return
        }
        setRecorderState('processing')
        try {
          const formData = new FormData()
          const extension = (recorder.mimeType || preferredMime || '').includes('ogg') ? 'ogg' : 'webm'
          formData.append('audio', blob, `voice-search.${extension}`)
          const response = await fetch('/api/voice-search', { method: 'POST', body: formData })
          let payload: any = null
          try {
            payload = await response.json()
          } catch (_jsonError) {
            payload = null
          }
          if (!response.ok || !payload || payload.ok === false) {
            const errorMessage = (payload && payload.error) || 'Не удалось распознать голосовой запрос'
            notifyError(errorMessage)
            setRecorderState('idle')
            return
          }
          const text = String(payload.text || '').trim()
          if (!text) {
            if (payload.warning) {
              notifyError(String(payload.warning))
            } else {
              notifyError('Речь не распознана')
            }
            setRecorderState('idle')
            return
          }
          onTranscribed(text)
          if (payload.warning) {
            notifyError(String(payload.warning))
          }
          setRecorderState('idle')
        } catch (err) {
          notifyError('Не удалось отправить запись на сервер')
          setRecorderState('idle')
        }
      }

      recorder.start()
      setRecorderState('recording')
      stopTimerRef.current = window.setTimeout(() => {
        stopRecordingInternal()
      }, 15000)
    } catch (error) {
      cleanupStream()
      const denied = error instanceof DOMException && (error.name === 'NotAllowedError' || error.name === 'SecurityError')
      setPermissionDenied(denied)
      if (denied) {
        notifyError('Доступ к микрофону запрещён браузером')
      } else {
        notifyError('Не удалось получить доступ к микрофону')
      }
      setRecorderState('idle')
    }
  }, [cleanupStream, disabled, notifyError, recorderState, stopRecordingInternal, supported])

  const stopRecording = useCallback(() => {
    stopRecordingInternal()
  }, [stopRecordingInternal])

  const buttonLabel = useMemo(() => {
    if (!supported) return 'Голосовой поиск недоступен'
    if (permissionDenied) return 'Микрофон заблокирован'
    if (recorderState === 'processing') return 'Распознаём…'
    return recorderState === 'recording' ? 'Остановить запись' : 'Голосовой поиск'
  }, [permissionDenied, recorderState, supported])

  const isBusy = recorderState === 'processing'
  const isRecording = recorderState === 'recording'

  const icon = useMemo(() => {
    if (isBusy) {
      return (
        <span className="icon-glyph" aria-hidden="true">
          <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
        </span>
      )
    }
    if (isRecording) {
      return <span className="icon-glyph" aria-hidden="true">⏹</span>
    }
    return <span className="icon-glyph" aria-hidden="true">🎤</span>
  }, [isBusy, isRecording])

  if (!supported) {
    return (
      <button
        type="button"
        className="btn btn-outline-secondary icon-only"
        disabled
        title={buttonLabel}
        aria-disabled="true"
      >
        {icon}
      </button>
    )
  }

  return (
    <button
      type="button"
      className={`btn btn-outline-${isRecording ? 'danger' : 'secondary'} icon-only`}
      onClick={isRecording ? stopRecording : startRecording}
      disabled={disabled || isBusy}
      aria-pressed={isRecording}
      aria-label={buttonLabel}
      title={buttonLabel}
    >
      {icon}
    </button>
  )
}
