import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
} from 'react'
import './App.css'
import RenderStudio from './RenderStudio'
import {
  MAX_PNG_RECIPE_BYTES,
  importPngRecipe,
  type ImportedRenderSettings,
} from './pngRecipe'
import {
  controlsMessage,
  updateControl,
  type ControlKey,
  type Controls,
} from './protocol'
import {
  CURRENT_ENGINE_VERSION,
  PRESETS,
  RECIPE_STORAGE_KEY,
  loadInitialRecipe,
  parseRecipeJson,
  recipeFilename,
  recipeForPreset,
  recipeUrl,
  serializeRecipe,
  type PatternRecipe,
  type PresetId,
} from './recipe'

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
      engine_version?: string
      preview_size: number
      frame_rate: number
      iteration?: number
    }
  | {
      type: 'frame'
      frame_id: number
      iteration: number
      controls_revision?: number
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

const HISTORY_LIMIT = 30
const APPLICATION_VERSION =
  import.meta.env.VITE_APP_VERSION || 'development'

interface ComparisonSnapshot {
  recipe: PatternRecipe
  imageUrl: string
}

interface PendingFrameMetadata {
  frameId: number
  iteration: number
  controlsRevision: number
}

interface PendingFrame {
  blob: Blob
  metadata: PendingFrameMetadata | undefined
}

function controlsHaveGradient(controls: Controls): boolean {
  return (
    controls.F1 !== controls.F2 ||
    controls.K1 !== controls.K2 ||
    controls.Du1 !== controls.Du2 ||
    controls.Dv1 !== controls.Dv2
  )
}

function cloneRecipe(recipe: PatternRecipe): PatternRecipe {
  return { ...recipe, controls: { ...recipe.controls } }
}

function recipesMatch(first: PatternRecipe, second: PatternRecipe): boolean {
  return serializeRecipe(first) === serializeRecipe(second)
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

function loadRecipeForApp() {
  try {
    return loadInitialRecipe(window.location.search, window.localStorage)
  } catch {
    return loadInitialRecipe(window.location.search)
  }
}

function App() {
  const [loadedRecipe] = useState(loadRecipeForApp)
  const [recipe, setRecipe] = useState<PatternRecipe>(loadedRecipe.recipe)
  const [recipeName, setRecipeName] = useState(loadedRecipe.recipe.name)
  const [seedDraft, setSeedDraft] = useState(String(loadedRecipe.recipe.seed))
  const [recipeNotice, setRecipeNotice] = useState(loadedRecipe.warning)
  const [engineVersion, setEngineVersion] = useState(
    loadedRecipe.recipe.engine_version,
  )
  const [connectionState, setConnectionState] =
    useState<ConnectionState>('connecting')
  const [statusDetail, setStatusDetail] = useState('')
  const [previewSize, setPreviewSize] = useState(256)
  const [hasFrame, setHasFrame] = useState(false)
  const [reconnectKey, setReconnectKey] = useState(0)
  const [userPaused, setUserPaused] = useState(false)
  const [gradientEditing, setGradientEditing] = useState(() =>
    controlsHaveGradient(loadedRecipe.recipe.controls),
  )
  const [previewZoom, setPreviewZoom] = useState(1)
  const [previewContrast, setPreviewContrast] = useState(1)
  const [comparison, setComparison] = useState<ComparisonSnapshot | null>(null)
  const [showBefore, setShowBefore] = useState(false)
  const [historyCounts, setHistoryCounts] = useState({ undo: 0, redo: 0 })
  const [liveIteration, setLiveIteration] = useState(0)
  const [renderRecipe, setRenderRecipe] = useState(() => cloneRecipe(loadedRecipe.recipe))
  const [renderReferenceSteps, setRenderReferenceSteps] = useState(0)
  const [renderHandoffVersion, setRenderHandoffVersion] = useState(0)
  const [importedRenderSettings, setImportedRenderSettings] =
    useState<ImportedRenderSettings | null>(null)

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const advancedRef = useRef<HTMLDetailsElement | null>(null)
  const importRef = useRef<HTMLInputElement | null>(null)
  const socketRef = useRef<WebSocket | null>(null)
  const recipeRef = useRef<PatternRecipe>(loadedRecipe.recipe)
  const controlsTimerRef = useRef<number | null>(null)
  const pausedRef = useRef(false)
  const controlsRevisionRef = useRef(0)
  const engineVersionRef = useRef(loadedRecipe.recipe.engine_version)
  const undoStackRef = useRef<PatternRecipe[]>([])
  const redoStackRef = useRef<PatternRecipe[]>([])
  const pendingFrameMetadataRef = useRef<PendingFrameMetadata[]>([])

  const send = useCallback((message: object) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(message))
    }
  }, [])

  const sendControls = useCallback((controls: Controls) => {
    controlsRevisionRef.current += 1
    send(controlsMessage(controls, controlsRevisionRef.current))
  }, [send])

  const commitRecipe = useCallback((nextRecipe: PatternRecipe) => {
    recipeRef.current = nextRecipe
    setRecipe(nextRecipe)
    setRecipeName(nextRecipe.name)
    setSeedDraft(String(nextRecipe.seed))
  }, [])

  const recordRecipe = useCallback(
    (nextRecipe: PatternRecipe) => {
      const current = recipeRef.current
      if (recipesMatch(current, nextRecipe)) return
      undoStackRef.current = [
        ...undoStackRef.current.slice(-(HISTORY_LIMIT - 1)),
        cloneRecipe(current),
      ]
      redoStackRef.current = []
      setHistoryCounts({ undo: undoStackRef.current.length, redo: 0 })
      commitRecipe(nextRecipe)
    },
    [commitRecipe],
  )

  const clearQueuedControls = useCallback(() => {
    if (controlsTimerRef.current !== null) {
      window.clearTimeout(controlsTimerRef.current)
      controlsTimerRef.current = null
    }
  }, [])

  const applyRecipe = useCallback(
    (nextRecipe: PatternRecipe, notice?: string) => {
      clearQueuedControls()
      const normalized = {
        ...nextRecipe,
        engine_version: engineVersionRef.current,
        controls: { ...nextRecipe.controls },
      }
      recordRecipe(normalized)
      setGradientEditing(controlsHaveGradient(normalized.controls))
      setShowBefore(false)
      sendControls(normalized.controls)
      send({ type: 'reset', seed: normalized.seed })
      setLiveIteration(0)
      if (notice) setRecipeNotice(notice)
    },
    [clearQueuedControls, recordRecipe, send, sendControls],
  )

  const queueControls = (nextControls: Controls) => {
    const nextRecipe: PatternRecipe = {
      ...recipeRef.current,
      engine_version: engineVersionRef.current,
      preset: 'custom',
      controls: nextControls,
    }
    recordRecipe(nextRecipe)
    if (controlsTimerRef.current !== null) return

    controlsTimerRef.current = window.setTimeout(() => {
      controlsTimerRef.current = null
      sendControls(recipeRef.current.controls)
    }, 50)
  }

  const handleControlChange = (key: ControlKey, value: number) => {
    queueControls(updateControl(recipeRef.current.controls, key, value))
  }

  const handleUniformControlChange = (key: ControlKey, value: number) => {
    const controls = { ...recipeRef.current.controls }
    if (key === 'F1') controls.F1 = controls.F2 = value
    else if (key === 'K1') controls.K1 = controls.K2 = value
    else if (key === 'Du1') controls.Du1 = controls.Du2 = value
    else if (key === 'Dv1') controls.Dv1 = controls.Dv2 = value
    queueControls(controls)
  }

  const selectPreset = (presetId: PresetId) => {
    applyRecipe(
      recipeForPreset(presetId, recipeRef.current.seed, engineVersionRef.current),
      'Preset applied and restarted from the current seed.',
    )
  }

  const makeUniform = () => {
    const current = recipeRef.current.controls
    queueControls({
      ...current,
      F2: current.F1,
      K2: current.K1,
      Du2: current.Du1,
      Dv2: current.Dv1,
    })
    setGradientEditing(false)
    setRecipeNotice('Edge gradients removed. The current simulation keeps evolving.')
  }

  const editGradients = () => {
    setGradientEditing(true)
    if (advancedRef.current) advancedRef.current.open = true
    setRecipeNotice('Edge editing enabled. Set each side independently in Advanced chemistry.')
  }

  const togglePaused = () => {
    const nextPaused = !pausedRef.current
    pausedRef.current = nextPaused
    setUserPaused(nextPaused)
    send({ type: nextPaused ? 'pause' : 'resume' })
    setConnectionState(nextPaused ? 'paused' : 'live')
  }

  const restartCurrentSeed = () => {
    send({ type: 'reset', seed: recipeRef.current.seed })
    setLiveIteration(0)
    setRecipeNotice(`Restarted deterministically with seed ${recipeRef.current.seed}.`)
  }

  const applySeed = (event: FormEvent) => {
    event.preventDefault()
    const seed = Number(seedDraft)
    if (!Number.isInteger(seed) || seed < 0 || seed > 4_294_967_295) {
      setRecipeNotice('Seed must be a whole number from 0 through 4294967295.')
      return
    }
    const nextRecipe = { ...recipeRef.current, seed }
    recordRecipe(nextRecipe)
    send({ type: 'reset', seed })
    setLiveIteration(0)
    setRecipeNotice(`Restarted with seed ${seed}.`)
  }

  const randomSeed = () => {
    const seed = crypto.getRandomValues(new Uint32Array(1))[0]
    const nextRecipe = { ...recipeRef.current, seed }
    recordRecipe(nextRecipe)
    send({ type: 'reset', seed })
    setLiveIteration(0)
    setRecipeNotice(`New random seed: ${seed}.`)
  }

  const perturb = () => {
    send({ type: 'perturb', noise: 0.25 })
    setRecipeNotice('Added noise to the current state without changing its recipe.')
  }

  const stepOnce = () => {
    if (!pausedRef.current) {
      pausedRef.current = true
      setUserPaused(true)
      setConnectionState('paused')
      send({ type: 'pause' })
    }
    send({ type: 'step' })
    setRecipeNotice('Advanced the paused simulation by one numerical iteration.')
  }

  const restoreHistoryRecipe = (nextRecipe: PatternRecipe, notice: string) => {
    clearQueuedControls()
    const normalized = {
      ...cloneRecipe(nextRecipe),
      engine_version: engineVersionRef.current,
    }
    commitRecipe(normalized)
    setGradientEditing(controlsHaveGradient(normalized.controls))
    setShowBefore(false)
    sendControls(normalized.controls)
    send({ type: 'reset', seed: normalized.seed })
    setLiveIteration(0)
    setRecipeNotice(notice)
  }

  const undoRecipe = () => {
    const previous = undoStackRef.current.pop()
    if (!previous) return
    redoStackRef.current = [
      ...redoStackRef.current.slice(-(HISTORY_LIMIT - 1)),
      cloneRecipe(recipeRef.current),
    ]
    setHistoryCounts({
      undo: undoStackRef.current.length,
      redo: redoStackRef.current.length,
    })
    restoreHistoryRecipe(previous, 'Undid the last recipe change and restarted.')
  }

  const redoRecipe = () => {
    const next = redoStackRef.current.pop()
    if (!next) return
    undoStackRef.current = [
      ...undoStackRef.current.slice(-(HISTORY_LIMIT - 1)),
      cloneRecipe(recipeRef.current),
    ]
    setHistoryCounts({
      undo: undoStackRef.current.length,
      redo: redoStackRef.current.length,
    })
    restoreHistoryRecipe(next, 'Redid the recipe change and restarted.')
  }

  const captureBefore = () => {
    const canvas = canvasRef.current
    if (!canvas || !hasFrame) return
    setComparison({
      recipe: cloneRecipe(recipeRef.current),
      imageUrl: canvas.toDataURL('image/png'),
    })
    setShowBefore(false)
    setRecipeNotice('Saved the current frame and recipe as the before comparison.')
  }

  const restoreBefore = () => {
    if (!comparison) return
    applyRecipe(comparison.recipe, 'Restored the before recipe and restarted it.')
  }

  const saveRecipeName = () => {
    const name = recipeName.trim()
    if (!name) {
      setRecipeName(recipeRef.current.name)
      setRecipeNotice('Recipe names cannot be empty.')
      return
    }
    recordRecipe({ ...recipeRef.current, name: name.slice(0, 80) })
  }

  const copyRecipeLink = async () => {
    const url = recipeUrl(recipeRef.current, window.location.href)
    try {
      await navigator.clipboard.writeText(url)
      setRecipeNotice('Share link copied to the clipboard.')
    } catch {
      setRecipeNotice('Clipboard access was unavailable. The address bar still contains the recipe link.')
    }
  }

  const exportRecipe = () => {
    const blob = new Blob([`${serializeRecipe(recipeRef.current)}\n`], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = recipeFilename(recipeRef.current)
    anchor.click()
    URL.revokeObjectURL(url)
    setRecipeNotice('Recipe JSON downloaded.')
  }

  const importRecipe = async (event: ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget
    const file = input.files?.[0]
    input.value = ''
    if (!file) return
    try {
      const isPng = file.type === 'image/png' || file.name.toLowerCase().endsWith('.png')
      if (isPng && file.size > MAX_PNG_RECIPE_BYTES) {
        throw new Error('PNG is too large to import (64 MB maximum).')
      }
      const pngImport = isPng
        ? importPngRecipe(await file.arrayBuffer(), file.name)
        : null
      const imported = pngImport?.recipe ?? parseRecipeJson(await file.text())
      const source = pngImport?.renderSettings
        ? 'PNG recipe and render settings'
        : isPng
          ? 'PNG recipe'
          : 'Recipe'
      const versionNote =
        imported.engine_version === engineVersionRef.current
          ? `${source} imported and restarted.`
          : `${source} imported from engine ${imported.engine_version} and loaded with ${engineVersionRef.current}.`
      applyRecipe(imported, versionNote)
      if (pngImport?.renderSettings) {
        setRenderRecipe(cloneRecipe({
          ...imported,
          engine_version: engineVersionRef.current,
        }))
        setRenderReferenceSteps(pngImport.renderSettings.developmentSteps)
        setImportedRenderSettings(pngImport.renderSettings)
        setRenderHandoffVersion((value) => value + 1)
      }
    } catch (error) {
      setRecipeNotice(
        error instanceof Error ? `Import failed: ${error.message}` : 'Recipe import failed.',
      )
    }
  }

  const downloadPreview = () => {
    const canvas = canvasRef.current
    if (!canvas || !hasFrame) return
    canvas.toBlob((blob) => {
      if (!blob) return
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `turing-preview-${canvas.width}x${canvas.height}-seed-${recipeRef.current.seed}.png`
      anchor.click()
      URL.revokeObjectURL(url)
    }, 'image/png')
  }

  const handOffToRenderer = () => {
    setRenderRecipe(cloneRecipe(recipeRef.current))
    setRenderReferenceSteps(liveIteration)
    setImportedRenderSettings(null)
    setRenderHandoffVersion((value) => value + 1)
  }

  useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        window.localStorage.setItem(RECIPE_STORAGE_KEY, serializeRecipe(recipe))
      } catch {
        // A private browsing policy may disable storage; the URL remains the fallback.
      }
      try {
        window.history.replaceState(null, '', recipeUrl(recipe, window.location.href))
      } catch {
        // Persistence should never interrupt a live session.
      }
    }, 100)
    return () => window.clearTimeout(timer)
  }, [recipe])

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
      pendingFrameMetadataRef.current = []
      let decodingFrame = false
      let queuedFrame: PendingFrame | null = null
      let droppedFrames = 0

      const updateFrameDiagnostics = () => {
        const canvas = canvasRef.current
        if (!canvas) return
        canvas.dataset.pendingFrames = queuedFrame ? '1' : '0'
        canvas.dataset.droppedFrames = String(droppedFrames)
      }

      const paintLatestFrame = async (candidate: PendingFrame) => {
        if (decodingFrame) {
          if (queuedFrame) droppedFrames += 1
          queuedFrame = candidate
          updateFrameDiagnostics()
          return
        }

        decodingFrame = true
        let current: PendingFrame | null = candidate
        while (current && !disposed && socketRef.current === websocket) {
          queuedFrame = null
          updateFrameDiagnostics()
          let bitmap: ImageBitmap
          try {
            bitmap = await createImageBitmap(current.blob)
          } catch {
            setStatusDetail('A preview frame could not be decoded.')
            current = queuedFrame
            continue
          }

          if (disposed || socketRef.current !== websocket) {
            bitmap.close()
            break
          }
          if (queuedFrame) {
            droppedFrames += 1
            bitmap.close()
            current = queuedFrame
            continue
          }

          const canvas = canvasRef.current
          const context = canvas?.getContext('2d')
          if (canvas && context) {
            canvas.width = bitmap.width
            canvas.height = bitmap.height
            context.drawImage(bitmap, 0, 0)
            if (current.metadata) {
              canvas.dataset.frameId = String(current.metadata.frameId)
              canvas.dataset.controlsRevision = String(current.metadata.controlsRevision)
              setLiveIteration(current.metadata.iteration)
            }
            setHasFrame(true)
          }
          bitmap.close()
          current = queuedFrame
        }
        decodingFrame = false
        updateFrameDiagnostics()
      }

      websocket.onopen = () => {
        reconnectAttempts = 0
        websocket.send(
          JSON.stringify({
            type: 'start',
            protocol_version: 1,
            controls: recipeRef.current.controls,
            seed: recipeRef.current.seed,
          }),
        )
      }

      websocket.onmessage = (event) => {
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
            const readyEngineVersion =
              message.engine_version ?? CURRENT_ENGINE_VERSION
            setPreviewSize(message.preview_size)
            setConnectionState(pausedRef.current ? 'paused' : 'live')
            engineVersionRef.current = readyEngineVersion
            setEngineVersion(readyEngineVersion)
            setLiveIteration(message.iteration ?? 0)
            if (recipeRef.current.engine_version !== readyEngineVersion) {
              const previousVersion = recipeRef.current.engine_version
              commitRecipe({
                ...recipeRef.current,
                engine_version: readyEngineVersion,
              })
              if (previousVersion !== CURRENT_ENGINE_VERSION) {
                setRecipeNotice(
                  `This recipe was created with engine ${previousVersion}; the current server uses ${readyEngineVersion}.`,
                )
              }
            }
            if (pausedRef.current || document.hidden) {
              websocket.send(JSON.stringify({ type: 'pause' }))
            }
            return
          }

          if (message.type === 'frame') {
            pendingFrameMetadataRef.current.push({
              frameId: message.frame_id,
              iteration: message.iteration,
              controlsRevision: message.controls_revision ?? 0,
            })
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

        void paintLatestFrame({
          blob: event.data as Blob,
          metadata: pendingFrameMetadataRef.current.shift(),
        })
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
      if (document.hidden) send({ type: 'pause' })
      else if (!pausedRef.current) send({ type: 'resume' })
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)
    connect()

    return () => {
      disposed = true
      terminalError = true
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer)
      clearQueuedControls()
      socketRef.current?.close(1000, 'Page closed')
      socketRef.current = null
    }
  }, [clearQueuedControls, commitRecipe, reconnectKey, send])

  const canControl = connectionState === 'live' || connectionState === 'paused'
  const canReconnect =
    connectionState === 'busy' ||
    connectionState === 'timed-out' ||
    connectionState === 'failed'
  const selectedPreset = PRESETS.find((preset) => preset.id === recipe.preset)
  const hasGradient = controlsHaveGradient(recipe.controls)

  return (
    <main className="app-shell">
      <header className="hero">
        <p className="eyebrow">Gray-Scott reaction-diffusion</p>
        <h1>Turing Pattern Generator</h1>
        <p className="intro">
          Explore a live pattern family, find a beautiful accident, and preserve its
          exact recipe for a fresh high-resolution tattoo or print design.
        </p>
      </header>

      <section className="status-bar" aria-live="polite">
        <span className={`status-dot status-${connectionState}`} aria-hidden="true" />
        <div>
          <strong>{statusText[connectionState]}</strong>
          {statusDetail && <p>{statusDetail}</p>}
        </div>
        {canReconnect && (
          <button className="button button-small" onClick={() => setReconnectKey((value) => value + 1)}>
            Try again
          </button>
        )}
      </section>

      <section id="live-lab" className="lab" aria-label="Pattern recipe and live preview">
        <aside className="control-panel basic-panel">
          <div className="panel-heading">
            <span>Recipe</span>
            <small>version 1</small>
          </div>

          <label className="field-label" htmlFor="recipe-name">Name</label>
          <input
            id="recipe-name"
            className="text-input"
            maxLength={80}
            value={recipeName}
            onChange={(event) => setRecipeName(event.target.value)}
            onBlur={saveRecipeName}
          />

          <label className="field-label" htmlFor="pattern-preset">Pattern family</label>
          <select
            id="pattern-preset"
            className="select-input"
            value={recipe.preset}
            onChange={(event) => {
              if (event.target.value !== 'custom') selectPreset(event.target.value as PresetId)
            }}
          >
            {PRESETS.map((preset) => (
              <option key={preset.id} value={preset.id}>{preset.name}</option>
            ))}
            <option value="custom">Custom recipe</option>
          </select>
          <p className="field-help">
            {selectedPreset?.description ?? 'Your own blend of feed, kill, and diffusion values.'}
          </p>

          <form className="seed-form" onSubmit={applySeed}>
            <label className="field-label" htmlFor="recipe-seed">Seed</label>
            <div className="input-action-row">
              <input
                id="recipe-seed"
                className="text-input numeric-input"
                type="number"
                min={0}
                max={4_294_967_295}
                step={1}
                value={seedDraft}
                onChange={(event) => setSeedDraft(event.target.value)}
              />
              <button className="button button-small" type="submit" disabled={!canControl}>Apply</button>
            </div>
          </form>

          <div className="recipe-summary">
            <span>
              {gradientEditing
                ? hasGradient
                  ? 'Edge gradients active'
                  : 'Edge editing ready'
                : 'Uniform chemistry'}
            </span>
            <small>Engine {engineVersion}</small>
          </div>
          <div className="mode-buttons" aria-label="Chemistry layout">
            <button
              className="button"
              aria-pressed={!gradientEditing}
              onClick={makeUniform}
              disabled={!canControl}
            >
              Uniform
            </button>
            <button
              className="button"
              aria-pressed={gradientEditing}
              onClick={editGradients}
            >
              Edge gradients
            </button>
          </div>

          <fieldset className="simulation-controls">
            <legend>Simulation</legend>
            <div className="actions">
              <button className="button button-primary" onClick={togglePaused} disabled={!canControl}>
                {userPaused ? 'Resume' : 'Pause'}
              </button>
              <button className="button" onClick={stepOnce} disabled={!canControl}>Step once</button>
              <button className="button" onClick={restartCurrentSeed} disabled={!canControl}>Restart same seed</button>
              <button className="button" onClick={randomSeed} disabled={!canControl}>Random seed</button>
              <button className="button" onClick={perturb} disabled={!canControl}>Perturb state</button>
              <button className="button" onClick={downloadPreview} disabled={!hasFrame}>Download preview</button>
            </div>
            <div className="history-actions" aria-label="Recipe history">
              <button className="button button-small" onClick={undoRecipe} disabled={!canControl || historyCounts.undo === 0}>
                Undo recipe
              </button>
              <button className="button button-small" onClick={redoRecipe} disabled={!canControl || historyCounts.redo === 0}>
                Redo recipe
              </button>
              <span>Up to {HISTORY_LIMIT} recipe changes</span>
            </div>
            <div className="comparison-actions" aria-label="Before and after comparison">
              <button className="button button-small" onClick={captureBefore} disabled={!hasFrame}>
                Set current as before
              </button>
              {comparison && (
                <>
                  <button className="button button-small" onClick={() => setShowBefore((value) => !value)}>
                    {showBefore ? 'View current' : 'View before'}
                  </button>
                  <button className="button button-small" onClick={restoreBefore} disabled={!canControl}>
                    Restore before recipe
                  </button>
                </>
              )}
            </div>
            <p className="preview-note">
              Preview: {previewSize} × {previewSize}px. Control changes evolve the current state;
              presets and seed changes restart it. Perturbation changes the state, not the saved recipe.
            </p>
          </fieldset>

          <fieldset className="display-controls">
            <legend>Preview display</legend>
            <div className="slider-label">
              <label htmlFor="preview-zoom">Zoom</label>
              <output htmlFor="preview-zoom">{previewZoom.toFixed(1)}×</output>
            </div>
            <input
              id="preview-zoom"
              type="range"
              min={1}
              max={3}
              step={0.1}
              value={previewZoom}
              onChange={(event) => setPreviewZoom(Number(event.target.value))}
            />
            <div className="slider-label display-slider-label">
              <label htmlFor="preview-contrast">Contrast</label>
              <output htmlFor="preview-contrast">{previewContrast.toFixed(1)}×</output>
            </div>
            <input
              id="preview-contrast"
              type="range"
              min={0.5}
              max={2.5}
              step={0.1}
              value={previewContrast}
              onChange={(event) => setPreviewContrast(Number(event.target.value))}
            />
            <p className="field-help">
              Display only. These controls do not change the recipe or downloaded pixels.
            </p>
          </fieldset>

          <div className="share-actions" aria-label="Recipe sharing">
            <button className="button" onClick={copyRecipeLink}>Copy link</button>
            <button className="button" onClick={exportRecipe}>Export JSON</button>
            <button className="button" onClick={() => importRef.current?.click()}>Import JSON/PNG</button>
            <input
              ref={importRef}
              className="visually-hidden"
              type="file"
              accept="application/json,image/png,.json,.png"
              onChange={importRecipe}
              tabIndex={-1}
            />
          </div>
          <p className="recipe-notice" aria-live="polite">{recipeNotice}</p>
        </aside>

        <div className="preview-column">
          <div className="preview-sticky">
            <div className="preview-heading">
              <strong>Live preview</strong>
              <output aria-label="Live simulation iteration">Step {liveIteration.toLocaleString()}</output>
            </div>
            <div className={`preview-frame ${hasFrame ? 'has-frame' : ''}`}>
              {comparison && showBefore && (
                <img
                  src={comparison.imageUrl}
                  alt="Saved before comparison"
                  style={{
                    filter: `contrast(${previewContrast})`,
                    transform: `scale(${previewZoom})`,
                  }}
                />
              )}
              <canvas
                ref={canvasRef}
                width={previewSize}
                height={previewSize}
                role="img"
                aria-label="Live grayscale Turing pattern preview"
                hidden={showBefore}
                style={{
                  filter: `contrast(${previewContrast})`,
                  transform: `scale(${previewZoom})`,
                }}
              />
              {!hasFrame && !showBefore && <span>Waiting for the first pattern…</span>}
            </div>
          </div>
        </div>

      <details ref={advancedRef} className="advanced-panel">
        <summary>
          <span>Advanced chemistry</span>
          <small>Exact Gray-Scott controls</small>
        </summary>
        <p className="advanced-intro">
          Feed and kill shape the reaction. U and V diffusion control how quickly each
          concentration spreads. Different endpoint values create a gradient across the image.
        </p>
        {gradientEditing ? (
          <div className="advanced-grid">
            <section className="control-panel">
            <div className="panel-heading">
              <span>Horizontal reaction</span>
              <small>left → right</small>
            </div>
            <Slider control="F1" label="Feed · left edge" min={0} max={0.1} step={0.001} value={recipe.controls.F1} onChange={handleControlChange} />
            <Slider control="F2" label="Feed · right edge" min={0} max={0.1} step={0.001} value={recipe.controls.F2} onChange={handleControlChange} />
            <Slider control="K1" label="Kill · left edge" min={0} max={0.1} step={0.001} value={recipe.controls.K1} onChange={handleControlChange} />
            <Slider control="K2" label="Kill · right edge" min={0} max={0.1} step={0.001} value={recipe.controls.K2} onChange={handleControlChange} />
            </section>

            <section className="control-panel">
            <div className="panel-heading">
              <span>Vertical diffusion</span>
              <small>top → bottom</small>
            </div>
            <Slider control="Du1" label="U diffusion · top edge" min={0} max={1} step={0.001} value={recipe.controls.Du1} onChange={handleControlChange} />
            <Slider control="Du2" label="U diffusion · bottom edge" min={0} max={1} step={0.001} value={recipe.controls.Du2} onChange={handleControlChange} />
            <Slider control="Dv1" label="V diffusion · top edge" min={0} max={1} step={0.001} value={recipe.controls.Dv1} onChange={handleControlChange} />
            <Slider control="Dv2" label="V diffusion · bottom edge" min={0} max={1} step={0.001} value={recipe.controls.Dv2} onChange={handleControlChange} />
            </section>
          </div>
        ) : (
          <div className="advanced-grid uniform-grid">
            <section className="control-panel">
              <div className="panel-heading">
                <span>Uniform reaction</span>
                <small>same at every edge</small>
              </div>
              <Slider control="F1" label="Feed" min={0} max={0.1} step={0.001} value={recipe.controls.F1} onChange={handleUniformControlChange} />
              <Slider control="K1" label="Kill" min={0} max={0.1} step={0.001} value={recipe.controls.K1} onChange={handleUniformControlChange} />
            </section>
            <section className="control-panel">
              <div className="panel-heading">
                <span>Uniform diffusion</span>
                <small>same at every edge</small>
              </div>
              <Slider control="Du1" label="U diffusion" min={0} max={1} step={0.001} value={recipe.controls.Du1} onChange={handleUniformControlChange} />
              <Slider control="Dv1" label="V diffusion" min={0} max={1} step={0.001} value={recipe.controls.Dv1} onChange={handleUniformControlChange} />
            </section>
          </div>
        )}
      </details>
      </section>

      <RenderStudio
        recipe={renderRecipe}
        liveSteps={renderReferenceSteps}
        handoffVersion={renderHandoffVersion}
        importedSettings={importedRenderSettings}
        onImportLiveSettings={handOffToRenderer}
        onImportRecipeFile={() => importRef.current?.click()}
      />

      <section className="pattern-context" aria-labelledby="pattern-context-title">
        <div className="context-intro">
          <p className="eyebrow">Pattern formation in nature and technology</p>
          <h2 id="pattern-context-title">Why spots, stripes, and mazes keep appearing</h2>
          <p>
            Alan Turing showed how locally interacting, diffusing substances can turn
            uniform fields into organized patterns. This lab uses the Gray-Scott model
            to explore that idea—not to reproduce any one organism exactly.
          </p>
        </div>

        <figure className="pattern-example">
          <img
            src="/pattern-example.png"
            width="1024"
            height="256"
            loading="lazy"
            alt="Black-and-white Gray-Scott field shifting from fine spots through maze-like stripes to dense spots"
          />
          <figcaption>
            One simulated field can move between dots, branching stripes, and maze-like
            forms as its feed, kill, and diffusion values vary across space.
          </figcaption>
        </figure>

        <div className="pattern-example-grid">
          <article>
            <p className="example-kind">Biology · fish skin</p>
            <h3>Giant pufferfish patterns</h3>
            <p>
              Pufferfish relatives display spots, reticulations, and maze-like skin
              markings. Fish-skin studies show that pigment-cell interactions can create
              the short-range activation and long-range inhibition associated with
              Turing patterns, although those cells are not the Gray-Scott chemicals.
            </p>
            <a
              href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8580470/"
              target="_blank"
              rel="noreferrer"
            >
              Explore the fish-skin research
            </a>
          </article>

          <article>
            <p className="example-kind">Biology · coat markings</p>
            <h3>Leopard and cat spots</h3>
            <p>
              Cat-family markings helped inspire reaction-diffusion models. Research in
              developing cat skin found a Dkk4 molecular pre-pattern and proposed
              short-range Wnt activators working with longer-range inhibitors. Real coat
              development is richer than this simulation, but the organizing logic is related.
            </p>
            <a
              href="https://www.nature.com/articles/s41467-021-25348-2"
              target="_blank"
              rel="noreferrer"
            >
              Read the cat-pattern study
            </a>
          </article>

          <article>
            <p className="example-kind">Engineering analogue</p>
            <h3>Block-copolymer nanolithography</h3>
            <p>
              Chemically distinct polymer blocks can phase-separate into nanoscale
              domains. Templates guide that self-assembly into sub-10-nanometer features
              studied for semiconductor and data-storage patterning. The kinship here is
              self-organization, not an identical reaction-diffusion mechanism.
            </p>
            <a
              href="https://www.nist.gov/programs-projects/directed-self-assembly-block-copolymers-nanopatterning"
              target="_blank"
              rel="noreferrer"
            >
              See NIST's patterning overview
            </a>
          </article>
        </div>

        <p className="context-note">
          This project began as a way to explore tattoo ideas. The generator turns that
          visual vocabulary into reproducible recipes while leaving room for the happy
          accidents that make generative art worth exploring.
        </p>
      </section>

      <footer className="site-footer">
        <span>Application {APPLICATION_VERSION} · Engine {CURRENT_ENGINE_VERSION}</span>
        <a
          href="https://github.com/tbrownhe/turing-pattern/issues/new/choose"
          target="_blank"
          rel="noreferrer"
        >
          Project feedback and issues
        </a>
      </footer>
    </main>
  )
}

export default App
