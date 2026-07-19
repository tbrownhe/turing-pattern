import { performance } from 'node:perf_hooks'
import process from 'node:process'
import { chromium } from 'playwright'

const DEFAULT_CONTROLS = {
  F1: 0.04,
  F2: 0.08,
  K1: 0.056,
  K2: 0.074,
  Du1: 0.7,
  Du2: 0.7,
  Dv1: 0.25,
  Dv2: 0.25,
}

const TARGETS = {
  first_preview_ms: 2_000,
  control_paint_p95_ms: 500,
  max_pending_frames: 1,
  heap_growth_bytes: 16 * 1024 * 1024,
  public_health_p95_ms: 250,
  static_ui_p95_ms: 250,
  render_progress_gap_ms: 5_000,
}

function parseArguments(argv) {
  const options = {
    url: 'http://127.0.0.1:3000',
    durationSeconds: 1_800,
    sampleIntervalSeconds: 5,
    activityIntervalSeconds: 300,
    controlTrials: 5,
    cpuThrottle: 1,
    headed: false,
    render: true,
    enforce: false,
  }
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]
    const value = argv[index + 1]
    if (argument === '--url') options.url = value, index += 1
    else if (argument === '--duration') options.durationSeconds = Number(value), index += 1
    else if (argument === '--sample-interval') options.sampleIntervalSeconds = Number(value), index += 1
    else if (argument === '--activity-interval') options.activityIntervalSeconds = Number(value), index += 1
    else if (argument === '--control-trials') options.controlTrials = Number(value), index += 1
    else if (argument === '--cpu-throttle') options.cpuThrottle = Number(value), index += 1
    else if (argument === '--headed') options.headed = true
    else if (argument === '--skip-render') options.render = false
    else if (argument === '--enforce') options.enforce = true
    else if (argument === '--help') {
      console.log(`Usage: npm run budget -- [options]

  --url URL                 Public site URL (default http://127.0.0.1:3000)
  --duration SECONDS        Browser observation time (default 1800)
  --sample-interval SECONDS Memory/frame sample interval (default 5)
  --activity-interval SEC   Control activity interval, below server idle timeout (default 300)
  --control-trials COUNT    Initial control-to-paint trials (default 5)
  --cpu-throttle RATE       Chromium CPU throttling factor (default 1)
  --skip-render             Do not exercise queued-render progress
  --headed                  Show Chromium
  --enforce                 Exit non-zero when an initial target is missed`)
      process.exit(0)
    } else throw new Error(`Unknown argument: ${argument}`)
  }
  if (!Number.isFinite(options.durationSeconds) || options.durationSeconds < 1) {
    throw new Error('--duration must be at least one second')
  }
  if (!Number.isFinite(options.sampleIntervalSeconds) || options.sampleIntervalSeconds <= 0) {
    throw new Error('--sample-interval must be positive')
  }
  if (!Number.isFinite(options.activityIntervalSeconds) || options.activityIntervalSeconds < 10) {
    throw new Error('--activity-interval must be at least ten seconds')
  }
  if (!Number.isInteger(options.controlTrials) || options.controlTrials < 1 || options.controlTrials > 50) {
    throw new Error('--control-trials must be an integer from 1 through 50')
  }
  if (!Number.isFinite(options.cpuThrottle) || options.cpuThrottle < 1 || options.cpuThrottle > 20) {
    throw new Error('--cpu-throttle must be between 1 and 20')
  }
  options.url = options.url.replace(/\/$/, '')
  return options
}

function percentile(values, fraction) {
  if (values.length === 0) return 0
  const ordered = [...values].sort((left, right) => left - right)
  return ordered[Math.min(ordered.length - 1, Math.floor(ordered.length * fraction))]
}

function round(value, digits = 2) {
  return Number(value.toFixed(digits))
}

async function timedFetch(url) {
  const started = performance.now()
  const response = await fetch(url, { cache: 'no-store' })
  await response.arrayBuffer()
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`)
  return performance.now() - started
}

async function measureFirstPreview(page, url) {
  const started = performance.now()
  await page.goto(url, { waitUntil: 'domcontentloaded' })
  const canvas = page.getByRole('img', { name: 'Live grayscale Turing pattern preview' })
  await canvas.waitFor({ state: 'visible' })
  await page.waitForFunction(
    () => Boolean(document.querySelector('canvas[aria-label="Live grayscale Turing pattern preview"]')?.dataset.frameId),
    undefined,
    { timeout: 15_000 },
  )
  return { elapsedMs: performance.now() - started, canvas }
}

function metricMap(response) {
  return Object.fromEntries(response.metrics.map(({ name, value }) => [name, value]))
}

async function measureControlPaint(page, slider, value) {
  const canvas = page.getByRole('img', { name: 'Live grayscale Turing pattern preview' })
  const previousRevision = Number(await canvas.getAttribute('data-controls-revision') ?? 0)
  const started = performance.now()
  await slider.fill(value)
  await page.waitForFunction(
    ({ selector, revision }) => Number(document.querySelector(selector)?.dataset.controlsRevision ?? 0) > revision,
    { selector: 'canvas[aria-label="Live grayscale Turing pattern preview"]', revision: previousRevision },
    { timeout: TARGETS.control_paint_p95_ms * 4 },
  )
  return performance.now() - started
}

async function measureRenderProgress(page, baseUrl) {
  const payload = {
    controls: DEFAULT_CONTROLS,
    seed: 5060,
    width: 4,
    height: 4,
    unit: 'in',
    quality: 'draft',
    feature_scale: 1,
    development_steps: 2_000,
    framing: 'crop',
  }
  const submitted = await page.evaluate(async (body) => {
    const response = await fetch('/api/v1/renders', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    return { status: response.status, body: await response.json() }
  }, payload)
  if (submitted.status !== 202) {
    throw new Error(`Render submission returned HTTP ${submitted.status}: ${JSON.stringify(submitted.body)}`)
  }

  const jobId = submitted.body.id
  const started = performance.now()
  let lastProgress = -1
  const progressChanges = []
  const healthLatencies = []
  const staticLatencies = []
  let job = submitted.body
  while (['queued', 'running'].includes(job.state)) {
    const [statusResult, healthMs, staticMs] = await Promise.all([
      page.evaluate(async (id) => {
        const response = await fetch(`/api/v1/renders/${id}`, { cache: 'no-store' })
        return { status: response.status, body: await response.json() }
      }, jobId),
      timedFetch(`${baseUrl}/healthz`),
      timedFetch(`${baseUrl}/`),
    ])
    if (statusResult.status !== 200) throw new Error(`Render status returned HTTP ${statusResult.status}`)
    job = statusResult.body
    healthLatencies.push(healthMs)
    staticLatencies.push(staticMs)
    if (job.progress_steps !== lastProgress) {
      lastProgress = job.progress_steps
      progressChanges.push({ at_ms: performance.now() - started, steps: lastProgress })
    }
    if (performance.now() - started > 120_000) {
      await page.evaluate((id) => fetch(`/api/v1/renders/${id}`, { method: 'DELETE' }), jobId)
      throw new Error('Budget render exceeded the runner\'s two-minute safety limit')
    }
    if (['queued', 'running'].includes(job.state)) await new Promise((resolve) => setTimeout(resolve, 250))
  }
  if (job.state !== 'completed') throw new Error(`Budget render ended in state ${job.state}: ${job.error}`)

  const artifactBytes = await page.evaluate(async (url) => {
    const response = await fetch(url)
    if (!response.ok) throw new Error(`Artifact returned HTTP ${response.status}`)
    return (await response.arrayBuffer()).byteLength
  }, job.artifact_url)
  const durationMs = performance.now() - started
  const changeTimes = [0, ...progressChanges.map((change) => change.at_ms), durationMs]
  const progressGaps = changeTimes.slice(1).map((value, index) => value - changeTimes[index])
  return {
    job_id: jobId,
    duration_ms: round(durationMs),
    progress_updates: progressChanges.length,
    max_progress_gap_ms: round(Math.max(...progressGaps)),
    public_health_p95_ms: round(percentile(healthLatencies, 0.95)),
    static_ui_p95_ms: round(percentile(staticLatencies, 0.95)),
    artifact_bytes: artifactBytes,
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2))
  await timedFetch(`${options.url}/`)
  const browser = await chromium.launch({ headless: !options.headed })
  try {
    const context = await browser.newContext({ viewport: { width: 390, height: 844 } })
    const warmupPage = await context.newPage()
    const coldPreview = await measureFirstPreview(warmupPage, options.url)
    await warmupPage.close()
    await new Promise((resolve) => setTimeout(resolve, 100))

    const page = await context.newPage()
    const cdp = await context.newCDPSession(page)
    await cdp.send('Performance.enable')
    if (options.cpuThrottle > 1) {
      await cdp.send('Emulation.setCPUThrottlingRate', { rate: options.cpuThrottle })
    }

    const warmPreview = await measureFirstPreview(page, options.url)
    const canvas = warmPreview.canvas

    await page.getByText('Advanced chemistry', { exact: true }).click()
    const feedSlider = page.getByRole('slider', { name: /Feed.*left/ })
    const controlLatencies = []
    for (let trial = 0; trial < options.controlTrials; trial += 1) {
      controlLatencies.push(await measureControlPaint(page, feedSlider, trial % 2 === 0 ? '0.051' : '0.052'))
    }

    const render = options.render ? await measureRenderProgress(page, options.url) : null

    await cdp.send('HeapProfiler.collectGarbage')
    const memoryStart = metricMap(await cdp.send('Performance.getMetrics'))
    const samples = []
    let lastFrameId = Number(await canvas.getAttribute('data-frame-id') ?? 0)
    let frameRegressions = 0
    const sessionStarted = performance.now()
    let nextActivityAt = options.activityIntervalSeconds * 1_000
    while (performance.now() - sessionStarted < options.durationSeconds * 1_000) {
      const elapsed = performance.now() - sessionStarted
      if (elapsed >= nextActivityAt) {
        const value = (Math.floor(elapsed / (options.activityIntervalSeconds * 1_000)) % 2 === 0) ? '0.051' : '0.052'
        controlLatencies.push(await measureControlPaint(page, feedSlider, value))
        nextActivityAt += options.activityIntervalSeconds * 1_000
      }
      const metrics = metricMap(await cdp.send('Performance.getMetrics'))
      const frameId = Number(await canvas.getAttribute('data-frame-id') ?? 0)
      if (frameId < lastFrameId) frameRegressions += 1
      lastFrameId = frameId
      samples.push({
        at_seconds: round(elapsed / 1_000),
        frame_id: frameId,
        pending_frames: Number(await canvas.getAttribute('data-pending-frames') ?? 0),
        dropped_frames: Number(await canvas.getAttribute('data-dropped-frames') ?? 0),
        js_heap_used_bytes: Math.round(metrics.JSHeapUsedSize ?? 0),
        nodes: Math.round(metrics.Nodes ?? 0),
      })
      const remaining = options.durationSeconds * 1_000 - (performance.now() - sessionStarted)
      if (remaining > 0) await new Promise((resolve) => setTimeout(resolve, Math.min(options.sampleIntervalSeconds * 1_000, remaining)))
    }
    await cdp.send('HeapProfiler.collectGarbage')
    const memoryEnd = metricMap(await cdp.send('Performance.getMetrics'))

    const measurements = {
      first_session_preview_ms: round(coldPreview.elapsedMs),
      warm_session_preview_ms: round(warmPreview.elapsedMs),
      control_paint_samples_ms: controlLatencies.map((value) => round(value)),
      control_paint_p95_ms: round(percentile(controlLatencies, 0.95)),
      session_duration_seconds: options.durationSeconds,
      sample_count: samples.length,
      max_pending_frames: Math.max(...samples.map((sample) => sample.pending_frames), 0),
      dropped_frames: Math.max(...samples.map((sample) => sample.dropped_frames), 0),
      frame_id_regressions: frameRegressions,
      heap_start_bytes: Math.round(memoryStart.JSHeapUsedSize ?? 0),
      heap_end_bytes: Math.round(memoryEnd.JSHeapUsedSize ?? 0),
      heap_growth_bytes: Math.round((memoryEnd.JSHeapUsedSize ?? 0) - (memoryStart.JSHeapUsedSize ?? 0)),
      node_growth: Math.round((memoryEnd.Nodes ?? 0) - (memoryStart.Nodes ?? 0)),
      render,
    }
    const checks = {
      first_preview: measurements.warm_session_preview_ms <= TARGETS.first_preview_ms,
      control_paint: measurements.control_paint_p95_ms <= TARGETS.control_paint_p95_ms,
      bounded_frame_queue: measurements.max_pending_frames <= TARGETS.max_pending_frames,
      monotonic_paint: measurements.frame_id_regressions === 0,
      stable_heap: measurements.heap_growth_bytes <= TARGETS.heap_growth_bytes,
      render_progress: render === null || render.duration_ms <= TARGETS.render_progress_gap_ms || (
        render.progress_updates >= 2 && render.max_progress_gap_ms <= TARGETS.render_progress_gap_ms
      ),
      public_health_under_render: render === null || render.public_health_p95_ms <= TARGETS.public_health_p95_ms,
      static_ui_under_render: render === null || render.static_ui_p95_ms <= TARGETS.static_ui_p95_ms,
    }
    const report = {
      label: 'chromium-server-fallback',
      url: options.url,
      timestamp: new Date().toISOString(),
      viewport: { width: 390, height: 844 },
      cpu_throttle: options.cpuThrottle,
      targets: TARGETS,
      measurements,
      checks,
      all_targets_met: Object.values(checks).every(Boolean),
    }
    console.log(JSON.stringify(report, null, 2))
    await context.close()
    if (options.enforce && !report.all_targets_met) process.exitCode = 1
  } finally {
    await browser.close()
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack : error)
  process.exitCode = 1
})
