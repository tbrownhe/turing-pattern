import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
} from 'react'
import './App.css'
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

interface ComparisonSnapshot {
  recipe: PatternRecipe
  imageUrl: string
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

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const advancedRef = useRef<HTMLDetailsElement | null>(null)
  const importRef = useRef<HTMLInputElement | null>(null)
  const socketRef = useRef<WebSocket | null>(null)
  const recipeRef = useRef<PatternRecipe>(loadedRecipe.recipe)
  const controlsTimerRef = useRef<number | null>(null)
  const pausedRef = useRef(false)
  const latestFrameRef = useRef(0)
  const engineVersionRef = useRef(loadedRecipe.recipe.engine_version)
  const undoStackRef = useRef<PatternRecipe[]>([])
  const redoStackRef = useRef<PatternRecipe[]>([])

  const send = useCallback((message: object) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(message))
    }
  }, [])

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
      send(controlsMessage(normalized.controls))
      send({ type: 'reset', seed: normalized.seed })
      if (notice) setRecipeNotice(notice)
    },
    [clearQueuedControls, recordRecipe, send],
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
      send(controlsMessage(recipeRef.current.controls))
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
    setRecipeNotice(`Restarted with seed ${seed}.`)
  }

  const randomSeed = () => {
    const seed = crypto.getRandomValues(new Uint32Array(1))[0]
    const nextRecipe = { ...recipeRef.current, seed }
    recordRecipe(nextRecipe)
    send({ type: 'reset', seed })
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
    send(controlsMessage(normalized.controls))
    send({ type: 'reset', seed: normalized.seed })
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
      const imported = parseRecipeJson(await file.text())
      const versionNote =
        imported.engine_version === engineVersionRef.current
          ? 'Recipe imported and restarted.'
          : `Recipe imported from engine ${imported.engine_version} and loaded with ${engineVersionRef.current}.`
      applyRecipe(imported, versionNote)
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
            const readyEngineVersion =
              message.engine_version ?? CURRENT_ENGINE_VERSION
            setPreviewSize(message.preview_size)
            setConnectionState(pausedRef.current ? 'paused' : 'live')
            engineVersionRef.current = readyEngineVersion
            setEngineVersion(readyEngineVersion)
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
      latestFrameRef.current += 1
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
        <p className="eyebrow">Reaction + diffusion</p>
        <h1>Gray-Scott Pattern Lab</h1>
        <p className="intro">
          Start with a pattern family, find a beautiful accident, and keep its exact recipe.
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

      <section className="lab" aria-label="Pattern recipe and live preview">
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
            <button className="button" onClick={() => importRef.current?.click()}>Import JSON</button>
            <input
              ref={importRef}
              className="visually-hidden"
              type="file"
              accept="application/json,.json"
              onChange={importRecipe}
              tabIndex={-1}
            />
          </div>
          <p className="recipe-notice" aria-live="polite">{recipeNotice}</p>
        </aside>

        <div className="preview-column">
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
        </div>
      </section>

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
    </main>
  )
}

export default App
