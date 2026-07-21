import { useEffect, useState, type FormEvent } from 'react'
import type { ImportedRenderSettings } from './pngRecipe'
import type { PatternRecipe } from './recipe'

type PhysicalUnit = 'in' | 'cm'
type Quality = 'draft' | 'studio' | 'fine'
type FeatureScale = 0.5 | 1 | 2
type Framing = 'crop' | 'fit' | 'extend'

interface RenderPlan {
  accepted: boolean
  issues: string[]
  quality_label: string
  pixels_per_inch: number
  feature_scale: number
  scale_model_status: string
  development_steps: number
  output_width: number
  output_height: number
  simulation_width: number
  simulation_height: number
  simulation_pixels: number
  bicubic_upsample: number
  estimated_seconds_low: number
  estimated_seconds_high: number
  estimated_memory_bytes: number
  resource_class: string
  engine_version: string
}

interface TimeStudyCheckpoint {
  steps: number
  image_url: string
}

interface TimeStudyResponse {
  engine_version: string
  simulation_size: number
  seed: number
  checkpoints: TimeStudyCheckpoint[]
}

type RenderJobState =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'expired'
  | 'interrupted'

interface RenderJob {
  id: string
  state: RenderJobState
  progress_steps: number
  requested_steps: number
  progress_percent: number
  queue_position: number | null
  cancel_requested: boolean
  error: string | null
  artifact_available: boolean
  artifact_url: string | null
  expires_at: string | null
  plan: RenderPlan
}

interface RenderStudioProps {
  recipe: PatternRecipe
  liveSteps: number
  handoffVersion: number
  importedSettings?: ImportedRenderSettings | null
  onImportLiveSettings: () => void
  onImportRecipeFile: () => void
}

const qualityOptions: ReadonlyArray<{
  value: Quality
  label: string
  detail: string
}> = [
  { value: 'draft', label: 'Draft', detail: '150 pixels per inch' },
  { value: 'studio', label: 'Studio', detail: '300 pixels per inch' },
  { value: 'fine', label: 'Fine', detail: '600 pixels per inch' },
]

const RENDER_JOB_STORAGE_KEY = 'turing-pattern.render-job.v1'

function apiError(value: unknown, fallback: string): string {
  if (typeof value !== 'object' || value === null) return fallback
  const detail = (value as { detail?: unknown }).detail
  if (typeof detail === 'string') return detail
  if (typeof detail === 'object' && detail !== null) {
    const message = (detail as { message?: unknown }).message
    if (typeof message === 'string') return message
  }
  return fallback
}

function studyCheckpoints(target: number): number[] {
  const studyEnd = Math.max(400, Math.min(20_000, target))
  return [...new Set([0.25, 0.5, 0.75, 1].map((fraction) =>
    Math.max(100, Math.round((studyEnd * fraction) / 100) * 100),
  ))].sort((left, right) => left - right)
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds} sec`
  const minutes = Math.ceil(seconds / 60)
  return `${minutes} min`
}

function formatMegabytes(bytes: number): string {
  return `${Math.ceil(bytes / 1_000_000)} MB`
}

export default function RenderStudio({
  recipe,
  liveSteps,
  handoffVersion,
  importedSettings,
  onImportLiveSettings,
  onImportRecipeFile,
}: RenderStudioProps) {
  const [width, setWidth] = useState('6')
  const [height, setHeight] = useState('6')
  const [unit, setUnit] = useState<PhysicalUnit>('in')
  const [quality, setQuality] = useState<Quality>('studio')
  const [featureScale, setFeatureScale] = useState<FeatureScale>(1)
  const [developmentSteps, setDevelopmentSteps] = useState('5000')
  const [framing, setFraming] = useState<Framing>('crop')
  const [plan, setPlan] = useState<RenderPlan | null>(null)
  const [planError, setPlanError] = useState('')
  const [planning, setPlanning] = useState(false)
  const [study, setStudy] = useState<TimeStudyResponse | null>(null)
  const [studyError, setStudyError] = useState('')
  const [studying, setStudying] = useState(false)
  const [job, setJob] = useState<RenderJob | null>(null)
  const [jobError, setJobError] = useState('')
  const [queueing, setQueueing] = useState(false)

  useEffect(() => {
    const jobId = window.localStorage.getItem(RENDER_JOB_STORAGE_KEY)
    if (!jobId) return
    let disposed = false
    fetch(`/api/v1/renders/${jobId}`)
      .then(async (response) => {
        if (!response.ok) throw new Error('Saved render job was not found.')
        return response.json() as Promise<RenderJob>
      })
      .then((savedJob) => {
        if (!disposed) setJob(savedJob)
      })
      .catch(() => window.localStorage.removeItem(RENDER_JOB_STORAGE_KEY))
    return () => {
      disposed = true
    }
  }, [])

  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.state)) return
    const timer = window.setTimeout(async () => {
      try {
        const response = await fetch(`/api/v1/renders/${job.id}`)
        const body: unknown = await response.json()
        if (!response.ok) throw new Error(apiError(body, 'Render status is unavailable.'))
        setJob(body as RenderJob)
      } catch (error) {
        setJobError(error instanceof Error ? error.message : 'Render status is unavailable.')
      }
    }, 1000)
    return () => window.clearTimeout(timer)
  }, [job])

  useEffect(() => {
    if (handoffVersion === 0) return
    if (importedSettings) {
      setWidth(String(importedSettings.width))
      setHeight(String(importedSettings.height))
      setUnit(importedSettings.unit)
      setQuality(importedSettings.quality)
      setFeatureScale(importedSettings.featureScale)
      setDevelopmentSteps(String(importedSettings.developmentSteps))
      setFraming(importedSettings.framing)
    } else {
      setDevelopmentSteps(String(Math.max(100, Math.min(20_000, liveSteps || 5000))))
    }
    setPlan(null)
    setStudy(null)
    setPlanError('')
    setStudyError('')
  }, [handoffVersion, importedSettings, liveSteps])

  const requestBody = () => ({
    controls: recipe.controls,
    seed: recipe.seed,
    recipe_name: recipe.name,
    recipe_preset: recipe.preset,
    width: Number(width),
    height: Number(height),
    unit,
    quality,
    feature_scale: featureScale,
    development_steps: Number(developmentSteps),
    framing,
  })

  const reviewPlan = async (event: FormEvent) => {
    event.preventDefault()
    setPlanning(true)
    setPlanError('')
    setPlan(null)
    try {
      const response = await fetch('/api/v1/render-plans', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody()),
      })
      const body: unknown = await response.json()
      if (!response.ok) throw new Error(apiError(body, 'The render plan is invalid.'))
      setPlan(body as RenderPlan)
    } catch (error) {
      setPlanError(error instanceof Error ? error.message : 'The render plan could not be checked.')
    } finally {
      setPlanning(false)
    }
  }

  const runTimeStudy = async () => {
    const target = Number(developmentSteps)
    if (!Number.isInteger(target) || target < 100 || target > 20_000) {
      setStudyError('Development steps must be a whole number from 100 through 20,000.')
      return
    }
    setStudying(true)
    setStudyError('')
    setStudy(null)
    try {
      const response = await fetch('/api/v1/time-studies', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          controls: recipe.controls,
          seed: recipe.seed,
          checkpoints: studyCheckpoints(target),
        }),
      })
      const body: unknown = await response.json()
      if (!response.ok) throw new Error(apiError(body, 'The time study could not run.'))
      setStudy(body as TimeStudyResponse)
    } catch (error) {
      setStudyError(error instanceof Error ? error.message : 'The time study could not run.')
    } finally {
      setStudying(false)
    }
  }

  const queueRender = async () => {
    setQueueing(true)
    setJobError('')
    try {
      const response = await fetch('/api/v1/renders', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody()),
      })
      const body: unknown = await response.json()
      if (!response.ok) throw new Error(apiError(body, 'The render could not be queued.'))
      const queuedJob = body as RenderJob
      setJob(queuedJob)
      window.localStorage.setItem(RENDER_JOB_STORAGE_KEY, queuedJob.id)
    } catch (error) {
      setJobError(error instanceof Error ? error.message : 'The render could not be queued.')
    } finally {
      setQueueing(false)
    }
  }

  const cancelRender = async () => {
    if (!job) return
    setJobError('')
    try {
      const response = await fetch(`/api/v1/renders/${job.id}`, { method: 'DELETE' })
      const body: unknown = await response.json()
      if (!response.ok) throw new Error(apiError(body, 'The render could not be cancelled.'))
      setJob(body as RenderJob)
    } catch (error) {
      setJobError(error instanceof Error ? error.message : 'The render could not be cancelled.')
    }
  }

  return (
    <section id="render-studio" className="render-studio" aria-labelledby="render-studio-title">
      <div className="render-heading">
        <div>
          <p className="eyebrow">High-resolution render</p>
          <h2 id="render-studio-title">Plan a fresh realization</h2>
        </div>
        <div className="render-source">
          <strong>{recipe.name}</strong>
          <span>Seed {recipe.seed} · live reference {liveSteps.toLocaleString()} steps</span>
          <div className="render-source-actions">
            <button className="button button-primary button-small" type="button" onClick={onImportLiveSettings}>
              Import current Live Lab settings
            </button>
            <button className="button button-small" type="button" onClick={onImportRecipeFile}>
              Load saved JSON/PNG
            </button>
          </div>
        </div>
      </div>
      <p className="render-intro">
        The renderer will start this recipe from its seeded initial state. The live image is
        a parameter-space reference, not the image being enlarged.
      </p>

      <div className="render-workspace">
        <form className="render-form" onSubmit={reviewPlan}>
          <fieldset className="render-fieldset">
            <legend>Physical size</legend>
            <div className="dimension-grid">
              <label>
                <span>Width</span>
                <input className="text-input numeric-input" type="number" min="0.1" max="100" step="0.1" value={width} onChange={(event) => setWidth(event.target.value)} />
              </label>
              <label>
                <span>Height</span>
                <input className="text-input numeric-input" type="number" min="0.1" max="100" step="0.1" value={height} onChange={(event) => setHeight(event.target.value)} />
              </label>
              <label>
                <span>Units</span>
                <select className="select-input" value={unit} onChange={(event) => setUnit(event.target.value as PhysicalUnit)}>
                  <option value="in">inches</option>
                  <option value="cm">centimetres</option>
                </select>
              </label>
            </div>
          </fieldset>

          <fieldset className="render-fieldset">
            <legend>Output detail</legend>
            <div className="choice-grid quality-grid">
              {qualityOptions.map((option) => (
                <label key={option.value} className="choice-card">
                  <input type="radio" name="quality" value={option.value} checked={quality === option.value} onChange={() => setQuality(option.value)} />
                  <strong>{option.label}</strong>
                  <small>{option.detail}</small>
                </label>
              ))}
            </div>
          </fieldset>

          <fieldset className="render-fieldset">
            <legend>Feature scale</legend>
            <div className="choice-grid">
              {([
                [0.5, 'Fine', 'smaller, denser motifs'],
                [1, 'Original', 'live-equivalent scale'],
                [2, 'Bold', 'larger, broader motifs'],
              ] as const).map(([value, label, detail]) => (
                <label key={value} className="choice-card">
                  <input type="radio" name="feature-scale" value={value} checked={featureScale === value} onChange={() => setFeatureScale(value)} />
                  <strong>{label} · {value}×</strong>
                  <small>{detail}</small>
                </label>
              ))}
            </div>
            <p className="field-help">Original 1× is executable now. Fine and Bold remain blocked until their numerical scale mappings are calibrated.</p>
          </fieldset>

          <fieldset className="render-fieldset">
            <legend>Pattern development</legend>
            <label className="field-label" htmlFor="development-steps">Evolution steps</label>
            <input id="development-steps" className="text-input numeric-input" type="number" min="100" max="20000" step="100" value={developmentSteps} onChange={(event) => setDevelopmentSteps(event.target.value)} />
            <p className="field-help">Early termination is a creative control. The render ends at exactly this requested development point.</p>
            <button className="button button-full time-study-button" type="button" onClick={runTimeStudy} disabled={studying}>
              {studying ? 'Running bounded time study…' : 'Preview development stages'}
            </button>
            {studyError && <p className="form-error" role="alert">{studyError}</p>}
          </fieldset>

          <fieldset className="render-fieldset">
            <legend>Framing</legend>
            <select className="select-input" value={framing} onChange={(event) => setFraming(event.target.value as Framing)}>
              <option value="crop">Crop to fill</option>
              <option value="fit">Fit without stretching</option>
              <option value="extend">Extend the simulated domain</option>
            </select>
          </fieldset>

          <button className="button button-primary button-full" type="submit" disabled={planning}>
            {planning ? 'Checking plan…' : 'Review render plan'}
          </button>
          {planError && <p className="form-error" role="alert">{planError}</p>}
        </form>

        <div className="render-results">
          <section className="time-study" aria-labelledby="time-study-title">
            <div className="result-heading">
              <h3 id="time-study-title">Development time study</h3>
              {study && <span>{study.simulation_size} × {study.simulation_size}px</span>}
            </div>
            {study ? (
              <div className="study-grid">
                {study.checkpoints.map((checkpoint) => (
                  <button key={checkpoint.steps} className="study-card" type="button" onClick={() => setDevelopmentSteps(String(checkpoint.steps))}>
                    <img src={checkpoint.image_url} alt={`Fresh recipe after ${checkpoint.steps.toLocaleString()} steps`} />
                    <span>{checkpoint.steps.toLocaleString()} steps</span>
                  </button>
                ))}
              </div>
            ) : (
              <p className="empty-result">Run one fresh low-resolution simulation to compare several early-termination points. Selecting a frame sets the development steps.</p>
            )}
          </section>

          <section className={`plan-result ${plan?.accepted === false ? 'plan-rejected' : ''}`} aria-labelledby="plan-result-title">
            <div className="result-heading">
              <h3 id="plan-result-title">Validated render plan</h3>
              {plan && <span className={`resource-badge resource-${plan.resource_class}`}>{plan.resource_class}</span>}
            </div>
            {plan ? (
              <>
                <dl className="plan-facts">
                  <div><dt>Output</dt><dd>{plan.output_width.toLocaleString()} × {plan.output_height.toLocaleString()} px</dd></div>
                  <div><dt>Simulation</dt><dd>{plan.simulation_width.toLocaleString()} × {plan.simulation_height.toLocaleString()} cells</dd></div>
                  <div><dt>Finish</dt><dd>{plan.bicubic_upsample}× bicubic</dd></div>
                  <div><dt>Development</dt><dd>{plan.development_steps.toLocaleString()} steps</dd></div>
                  <div><dt>Estimate</dt><dd>{formatDuration(plan.estimated_seconds_low)}–{formatDuration(plan.estimated_seconds_high)}</dd></div>
                  <div><dt>Working memory</dt><dd>about {formatMegabytes(plan.estimated_memory_bytes)}</dd></div>
                </dl>
                {plan.issues.length > 0 && (
                  <ul className="plan-issues">
                    {plan.issues.map((issue) => <li key={issue}>{issue}</li>)}
                  </ul>
                )}
                <p className="field-help">Estimate calibrated from the OptiPlex 256² benchmark. Queue conditions and cache behavior can change actual time.</p>
                <button
                  className="button button-primary button-full"
                  type="button"
                  onClick={queueRender}
                  disabled={!plan.accepted || queueing || Boolean(job && ['queued', 'running'].includes(job.state))}
                >
                  {queueing ? 'Joining bounded queue…' : 'Queue high-resolution render'}
                </button>
                {job && (
                  <div className={`job-status job-${job.state}`} aria-live="polite">
                    <div className="job-status-heading">
                      <strong>{job.state === 'queued' ? `Queued${job.queue_position ? ` · position ${job.queue_position}` : ''}` : job.state}</strong>
                      <span>{job.progress_steps.toLocaleString()} / {job.requested_steps.toLocaleString()} steps</span>
                    </div>
                    <progress max={100} value={job.progress_percent}>{job.progress_percent}%</progress>
                    {job.cancel_requested && <p>Cancellation requested; the current numerical chunk will finish safely.</p>}
                    {job.error && <p className="form-error">{job.error}</p>}
                    <div className="job-actions">
                      {['queued', 'running'].includes(job.state) && (
                        <button className="button button-small" type="button" onClick={cancelRender}>Cancel and discard</button>
                      )}
                      {job.artifact_available && job.artifact_url && (
                        <a className="button button-primary" href={job.artifact_url}>Download completed PNG</a>
                      )}
                    </div>
                  </div>
                )}
                {jobError && <p className="form-error" role="alert">{jobError}</p>}
                <p className="queue-note">Jobs persist across refreshes and API restarts. The worker admits one bounded job at a time and releases completed artifacts after their configured lifetime.</p>
              </>
            ) : (
              <p className="empty-result">Review the plan to resolve physical dimensions into a bounded numerical grid before any expensive work starts.</p>
            )}
          </section>
        </div>
      </div>
    </section>
  )
}
