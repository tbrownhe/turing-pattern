import { useEffect, useRef, useState } from 'react'
import './App.css'
import {
  controlsMessage,
  DEFAULT_CONTROLS,
  updateControl,
  type ControlKey,
  type Controls,
} from './protocol'

type ConnectionState =
  | 'connecting'
  | 'live'
  | 'paused'
  | 'busy'
  | 'reconnecting'
  | 'timed-out'
  | 'failed'

type ServerMessage =
  | {
      type: 'ready'
      protocol_version: 1
      preview_size: number
      frame_rate: number
    }
  | { type: 'error'; error: { code: string; message: string } }

const statusText: Record<ConnectionState, string> = {
  connecting: 'Connecting to the live simulator…',
  live: 'Live simulation running',
  paused: 'Simulation paused',
  busy: 'The server is full right now. Try again in a moment.',
  reconnecting: 'Connection lost. Reconnecting…',
  'timed-out': 'This session timed out while idle.',
  failed: 'The live simulator is unavailable.',
}

interface SliderProps {
  control: ControlKey
  label: string
  min: number
  max: number
  step: number
  value: number
  onChange: (key: ControlKey, value: number) => void
}

function Slider({ control, label, min, max, step, value, onChange }: SliderProps) {
  const inputId = `control-${control}`
  return (
    <div className="slider-row">
      <div className="slider-label">
        <label htmlFor={inputId}>{label}</label>
        <output htmlFor={inputId}>{value.toFixed(4)}</output>
      </div>
      <input
        id={inputId}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(control, Number(event.target.value))}
      />
    </div>
  )
}

function App() {
  const [controls, setControls] = useState<Controls>(DEFAULT_CONTROLS)
  const [connectionState, setConnectionState] =
    useState<ConnectionState>('connecting')
  const [statusDetail, setStatusDetail] = useState('')
  const [previewSize, setPreviewSize] = useState(256)
  const [hasFrame, setHasFrame] = useState(false)
  const [reconnectKey, setReconnectKey] = useState(0)
  const [userPaused, setUserPaused] = useState(false)

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const socketRef = useRef<WebSocket | null>(null)
  const controlsRef = useRef<Controls>(DEFAULT_CONTROLS)
  const controlsTimerRef = useRef<number | null>(null)
  const seedRef = useRef(0)
  const pausedRef = useRef(false)
  const latestFrameRef = useRef(0)

  const send = (message: object) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(message))
    }
  }

  const queueControls = (nextControls: Controls) => {
    controlsRef.current = nextControls
    setControls(nextControls)
    if (controlsTimerRef.current !== null) return

    controlsTimerRef.current = window.setTimeout(() => {
      controlsTimerRef.current = null
      send(controlsMessage(controlsRef.current))
    }, 50)
  }

  const handleControlChange = (key: ControlKey, value: number) => {
    queueControls(updateControl(controlsRef.current, key, value))
  }

  const togglePaused = () => {
    const nextPaused = !pausedRef.current
    pausedRef.current = nextPaused
    setUserPaused(nextPaused)
    send({ type: nextPaused ? 'pause' : 'resume' })
    setConnectionState(nextPaused ? 'paused' : 'live')
  }

  const reset = () => {
    const seed = crypto.getRandomValues(new Uint32Array(1))[0]
    seedRef.current = seed
    send({ type: 'reset', seed })
  }

  const perturb = () => send({ type: 'perturb', noise: 0.25 })

  const downloadPreview = () => {
    const canvas = canvasRef.current
    if (!canvas || !hasFrame) return
    canvas.toBlob((blob) => {
      if (!blob) return
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `turing-preview-${canvas.width}x${canvas.height}.png`
      anchor.click()
      URL.revokeObjectURL(url)
    }, 'image/png')
  }

  useEffect(() => {
    let disposed = false
    let reconnectTimer: number | undefined
    let reconnectAttempts = 0
    let terminalError = false

    const connect = () => {
      if (disposed) return
      setConnectionState(reconnectAttempts > 0 ? 'reconnecting' : 'connecting')
      setStatusDetail('')

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const websocket = new WebSocket(`${protocol}//${window.location.host}/ws`)
      websocket.binaryType = 'blob'
      socketRef.current = websocket

      websocket.onopen = () => {
        reconnectAttempts = 0
        websocket.send(
          JSON.stringify({
            type: 'start',
            protocol_version: 1,
            controls: controlsRef.current,
            seed: seedRef.current,
          }),
        )
      }

      websocket.onmessage = async (event) => {
        if (typeof event.data === 'string') {
          let message: ServerMessage
          try {
            message = JSON.parse(event.data) as ServerMessage
          } catch {
            terminalError = true
            setConnectionState('failed')
            setStatusDetail('The server sent an unreadable response.')
            websocket.close(1008)
            return
          }

          if (message.type === 'ready') {
            setPreviewSize(message.preview_size)
            setConnectionState(pausedRef.current ? 'paused' : 'live')
            if (pausedRef.current || document.hidden) {
              websocket.send(JSON.stringify({ type: 'pause' }))
            }
            return
          }

          terminalError = true
          setStatusDetail(message.error.message)
          if (message.error.code === 'server_busy') setConnectionState('busy')
          else if (message.error.code === 'idle_timeout')
            setConnectionState('timed-out')
          else setConnectionState('failed')
          return
        }

        const sequence = ++latestFrameRef.current
        try {
          const bitmap = await createImageBitmap(event.data as Blob)
          if (disposed || sequence !== latestFrameRef.current) {
            bitmap.close()
            return
          }
          const canvas = canvasRef.current
          const context = canvas?.getContext('2d')
          if (canvas && context) {
            canvas.width = bitmap.width
            canvas.height = bitmap.height
            context.drawImage(bitmap, 0, 0)
            setHasFrame(true)
          }
          bitmap.close()
        } catch {
          setStatusDetail('A preview frame could not be decoded.')
        }
      }

      websocket.onclose = (event) => {
        if (socketRef.current === websocket) socketRef.current = null
        if (disposed) return
        if (terminalError || event.code === 1008 || event.code === 1013) return
        if (reconnectAttempts >= 5) {
          setConnectionState('failed')
          setStatusDetail('Reconnect attempts were exhausted.')
          return
        }

        setConnectionState('reconnecting')
        const delay = Math.min(1000 * 2 ** reconnectAttempts, 10_000)
        reconnectAttempts += 1
        reconnectTimer = window.setTimeout(connect, delay)
      }

      websocket.onerror = () => websocket.close()
    }

    const handleVisibilityChange = () => {
      if (document.hidden) {
        send({ type: 'pause' })
      } else if (!pausedRef.current) {
        send({ type: 'resume' })
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)
    connect()

    return () => {
      disposed = true
      terminalError = true
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer)
      if (controlsTimerRef.current !== null) {
        window.clearTimeout(controlsTimerRef.current)
        controlsTimerRef.current = null
      }
      latestFrameRef.current += 1
      socketRef.current?.close(1000, 'Page closed')
      socketRef.current = null
    }
  }, [reconnectKey])

  const canControl = connectionState === 'live' || connectionState === 'paused'
  const canReconnect =
    connectionState === 'busy' ||
    connectionState === 'timed-out' ||
    connectionState === 'failed'

  return (
    <main className="app-shell">
      <header className="hero">
        <p className="eyebrow">Reaction + diffusion</p>
        <h1>Gray-Scott Pattern Lab</h1>
        <p className="intro">
          Tune the chemistry at each edge and watch order emerge from noise.
        </p>
      </header>

      <section className="status-bar" aria-live="polite">
        <span className={`status-dot status-${connectionState}`} aria-hidden="true" />
        <div>
          <strong>{statusText[connectionState]}</strong>
          {statusDetail && <p>{statusDetail}</p>}
        </div>
        {canReconnect && (
          <button className="button button-small" onClick={() => setReconnectKey((v) => v + 1)}>
            Try again
          </button>
        )}
      </section>

      <section className="lab" aria-label="Pattern controls and live preview">
        <aside className="control-panel">
          <div className="panel-heading">
            <span>Horizontal chemistry</span>
            <small>left → right</small>
          </div>
          <Slider control="F1" label="Feed · left" min={0} max={0.1} step={0.001} value={controls.F1} onChange={handleControlChange} />
          <Slider control="F2" label="Feed · right" min={0} max={0.1} step={0.001} value={controls.F2} onChange={handleControlChange} />
          <Slider control="K1" label="Kill · left" min={0} max={0.1} step={0.001} value={controls.K1} onChange={handleControlChange} />
          <Slider control="K2" label="Kill · right" min={0} max={0.1} step={0.001} value={controls.K2} onChange={handleControlChange} />
        </aside>

        <div className="preview-column">
          <div className={`preview-frame ${hasFrame ? 'has-frame' : ''}`}>
            <canvas
              ref={canvasRef}
              width={previewSize}
              height={previewSize}
              role="img"
              aria-label="Live grayscale Turing pattern preview"
            />
            {!hasFrame && <span>Waiting for the first pattern…</span>}
          </div>
          <div className="actions">
            <button className="button button-primary" onClick={togglePaused} disabled={!canControl}>
              {userPaused ? 'Resume' : 'Pause'}
            </button>
            <button className="button" onClick={reset} disabled={!canControl}>New seed</button>
            <button className="button" onClick={perturb} disabled={!canControl}>Perturb</button>
            <button className="button" onClick={downloadPreview} disabled={!hasFrame}>Download preview</button>
          </div>
          <p className="preview-note">
            Preview: {previewSize} × {previewSize}px. High-resolution rendering comes after the safety foundation.
          </p>
        </div>

        <aside className="control-panel">
          <div className="panel-heading">
            <span>Vertical diffusion</span>
            <small>top → bottom</small>
          </div>
          <Slider control="Du1" label="U diffusion · top" min={0} max={1} step={0.001} value={controls.Du1} onChange={handleControlChange} />
          <Slider control="Du2" label="U diffusion · bottom" min={0} max={1} step={0.001} value={controls.Du2} onChange={handleControlChange} />
          <Slider control="Dv1" label="V diffusion · top" min={0} max={1} step={0.001} value={controls.Dv1} onChange={handleControlChange} />
          <Slider control="Dv2" label="V diffusion · bottom" min={0} max={1} step={0.001} value={controls.Dv2} onChange={handleControlChange} />
        </aside>
      </section>
    </main>
  )
}

export default App
