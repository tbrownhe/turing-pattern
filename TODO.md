# Path to Turing Pattern Glory

This is the working roadmap from the July 2026 repository audit. The goal is not
to turn this small project into a distributed platform. It is to make one modest
Linux server feel fast, safe, predictable, and delightful for two jobs:

1. explore a Gray-Scott pattern interactively; and
2. reproduce and export a print/tattoo-quality result.

## North star

A visitor can discover an interesting pattern without understanding the equations,
an expert can tune every meaningful parameter, and either person can export the
same result later from a recorded seed and parameter set. One abusive or malformed
client cannot make the site unavailable to everyone else.

The recommended architecture is:

```text
browser-side live preview (preferred) ----> shareable parameters + seed
                 |                                      |
                 +----> bounded render API ----> one queued high-resolution job
                                                at a time (initially)
```

Keep the current server-side live mode as a fallback until a WebGL2/WebGPU or WASM
preview proves itself on representative browsers. Do not increase public server
concurrency until it has been measured on the OptiPlex 5060.

## Audit snapshot

What is already good:

- The codebase is small and understandable, with a clean frontend/backend boundary.
- Production does not publish the backend directly on a host port; Traefik is the
  public entry point.
- The existing semaphore and idle timeout show the right resource-management intent.
- `.env` is ignored and is not tracked by Git.
- The frontend has a lockfile, strict TypeScript settings, and a multi-stage image.
- `npm run build` succeeds. `npm run lint` succeeds with one hook-dependency warning.

The most important findings:

| Severity | Finding | Consequence |
| --- | --- | --- |
| Critical | `sim.step()` and PNG encoding run inside the async WebSocket loop. | Admitted simulations block the event loop and each other; connection handling and control messages become sluggish under load. |
| Critical | `/generate` is expensive, synchronous, public, and does not use the semaphore. | Requests can consume the thread pool/CPU without a bounded queue, defeating the live-session limit. |
| Critical | WebSocket JSON is passed to `TuringSimulator(**msg)` without a schema. The client can supply `shape`; values are not bounded or required to be finite. | A client can request huge allocations, crash a simulation, or keep it in a reset/error loop. |
| High | The semaphore is process-local and fixed at four. | Worker-count changes multiply the limit; four CPU simulations may already be too many for this host. |
| High | WebSocket origins are not checked. HTTP CORS allows only `localhost:5173`, while local Compose uses port 3000 and the production UI is on another origin. | Browser protections are inconsistent: REST can fail for valid users, while arbitrary sites can open costly WebSockets. |
| High | Slider handlers call `setState()` and then send a closure containing the previous state. | The server receives controls one interaction behind the UI. |
| High | A new object URL is created for every received frame and never revoked. | A long live session leaks browser memory. |
| High | There is no backend test suite, frontend behavior test, CI workflow, load test, or numerical regression fixture. | Performance, correctness, and security fixes cannot be changed confidently. |
| Medium | Every frame is PNG-encoded and sent as an `<img>` blob at a nominal 20 FPS. Four full 2-D parameter maps and many temporary `np.roll` arrays are allocated. | CPU, memory bandwidth, compression, network, and browser decoding are doing avoidable work. |
| Medium | Randomness is not seeded, and the web output omits the parameter metadata supported by the original script. | Attractive results cannot be reproduced reliably. |
| Medium | Images run as root, dependencies/base images are loosely pinned, health checks and container resource limits are absent, and build tools remain in the backend image. | Deployments are less reproducible and a compromise or runaway process has a larger blast radius. |
| Medium | The README commands/paths no longer match the repository and its equations contain encoding damage. | New contributors cannot follow the documented local workflow. |

Traefik reduces direct exposure, but it does not validate simulation inputs, make
CPU work asynchronous, bound memory, or rate-limit work performed after a WebSocket
handshake. Those controls must live in the application/container design too.

## P0 — Make the current service safe and truthful

Complete these before advertising high-resolution generation or increasing traffic.

Completed on the `overhaul` branch. P0 deliberately keeps `/api/v1/generate` as a
fixed-size, capacity-gated `POST`; the durable `202` render queue and hard worker
timeouts remain P2 work. Browser-side/WASM simulation was not started in this phase.

### [x] P0.1 Put every compute path behind one capacity policy

Files: `backend/app/api/main.py`, `backend/app/core/turing.py`, Compose configuration.

- Temporarily disable `/generate`, or put it behind the same bounded scheduler as
  live work. A normal request must never create unbounded queued CPU work.
- Move simulation stepping and image encoding off the async event-loop thread. For
  the short term, use a dedicated, bounded executor; do not use the unbounded default
  thread pool as the queue. For high-resolution work, prefer an isolated worker
  process with a hard timeout.
- Make live-session count, render-worker count, maximum queue length, frame rate,
  idle timeout, and job timeout environment settings with conservative defaults.
- Run one API process while the admission controller is in memory. Document that
  adding Uvicorn workers multiplies process-local limits. If multi-process serving is
  ever needed, move admission state to a single render service or another shared
  coordinator first.
- Return explicit `429`/`503` responses (and structured WebSocket errors) with a
  retry hint when capacity is exhausted. Never silently accumulate work.
- Apply per-IP handshake/request rate limits and connection limits at Traefik, but
  treat them as a second layer rather than the compute quota itself.

Done when:

- A health request and a new connection remain responsive while the configured
  number of simulations/renders is busy.
- One more client is rejected quickly and cleanly without increasing memory/queue
  length.
- Live and render work together cannot exceed the measured CPU and memory budget.
- Disconnect, timeout, invalid input, cancellation, and internal exceptions always
  release their capacity slot.

### [x] P0.2 Define and validate the public protocol

Files: `backend/app/api/main.py`; add a small `schemas.py` rather than accepting raw
dictionaries throughout the application.

- Create strict Pydantic models for initial controls, control updates, seed/reset,
  and render requests. Use a discriminated `type` field for WebSocket messages.
- Require all numeric inputs to be finite. Set scientifically useful minimums and
  maximums for `F`, `k`, `Du`, and `Dv`; cap steps, dimensions, pixel count, upsample
  factor, and message size.
- Do not accept `shape`, `rgb`, or other constructor options from live clients.
  The server owns preview resolution and resource settings.
- Reject missing/unknown fields and return a stable machine-readable error instead
  of letting `KeyError`, `TypeError`, NumPy, or Pillow fail later.
- Make heavy generation a `POST`, not a side-effecting/cacheable `GET`. Return `202`
  for queued renders rather than holding an HTTP connection through a long job.
- Version the protocol (`/api/v1` and/or an explicit WebSocket protocol version).
- Never log full binary frames or untrusted payloads. Give requests, sessions, and
  jobs opaque IDs for diagnosis.

Done when:

- Tests cover missing keys, extra keys, strings, `NaN`, infinities, negative values,
  extremes, oversized shapes/messages, and invalid message types.
- No public input can control allocation size outside documented limits.
- Every expected client error is a 4xx/structured socket error; it does not emit a
  backend traceback or consume a slot after closure.

### [x] P0.3 Fix WebSocket lifecycle and current UI correctness

Files: `frontend/src/App.tsx`, `backend/app/api/main.py`.

- Send the new slider value, not React state from the previous render. Prefer one
  typed controls object and a debounced/throttled effect over eight duplicated
  handlers.
- Keep only the newest unsent control update so dragging a slider cannot build a
  backlog. Define whether a control change evolves the existing state or restarts it.
- Revoke the prior blob URL after the replacement image has loaded and on unmount,
  or decode frames with `createImageBitmap` and draw to a canvas.
- Add connection states: connecting, live, paused, busy, reconnecting, timed out,
  and failed. Replace `alert()` with an inline, accessible status message.
- Add pause/resume and stop sending/processing frames when the tab is hidden or the
  user pauses. Retry transient disconnects with bounded exponential backoff; do not
  retry capacity/validation failures in a tight loop.
- On the server, observe exceptions from the receiver task, cancel and await sibling
  tasks, use monotonic time for durations, and handle disconnect during the initial
  message. Make timeout text agree with the configured value.
- Decide what **Shake** means. The current method adds noise to the current arrays;
  expose separate **Reseed/reset** and **Perturb** actions if both are useful.

Done when:

- Automated UI tests prove a slider update sends the value displayed on screen.
- A 30-minute live session has stable browser memory and no growing message queue.
- Mount/unmount, network loss, server-busy, idle-timeout, and malformed-message paths
  leave no orphan task or capacity slot.

### [x] P0.4 Use one explicit same-origin security boundary

Files: `docker-compose.yml`, `docker-compose.local.yml`, `frontend/nginx.conf`,
`backend/app/api/main.py`.

- Prefer serving `/api/v1/*` and `/ws` on the frontend origin through Traefik/Nginx.
  This removes most CORS configuration and avoids exposing a separate public API
  hostname merely for browser traffic.
- Until that migration, read an exact HTTP origin allowlist from configuration and
  include both the real local frontend and production frontend. Do not use `*` with
  credentials.
- Validate the WebSocket `Origin` header against its own allowlist before admitting
  compute. CORS middleware does not protect WebSockets.
- Decide whether FastAPI `/docs`, `/redoc`, and the schema are intentionally public.
  Disable or protect them in production if not. The copied
  `nginx-backend-not-found.conf` is not included by the current Nginx config and does
  not protect the separately routed backend.
- Define the referenced Traefik `rate-limit` and `https-redirect` middleware in the
  documented deployment configuration, including burst behavior and trusted proxy
  handling. Do not assume a name alone creates a middleware.
- Add response security headers at the edge: a tested Content Security Policy,
  `X-Content-Type-Options`, `Referrer-Policy`, and appropriate framing/permissions
  policy. Remove the runtime Tailwind CDN dependency so CSP can be strict.

Done when:

- Production and local browser traffic work without permissive CORS.
- A WebSocket from an unapproved origin is rejected before acquiring a simulation
  slot, while a non-browser client is governed by documented policy.
- The deployed routes and headers are verified with an integration/smoke test.

## P1 — Build a fast, testable foundation

P1's server-side foundation is complete on the `overhaul` branch: the shared
seeded engine, measured CPU optimization, numerical/API/UI tests, load tooling,
CI gates, hash locks, non-root/read-only containers, structured logs, metrics, and
operations notes are in place. The production OptiPlex engine baseline and deployed
capacity probe are recorded. Browser-side preview work is deliberately separated
into P2.4 so it can be explored and taught without obscuring the stable fallback.

### [x] P1.1 Separate the numerical engine from transport and files

- Introduce typed `SimulationConfig` and `SimulationState` objects shared by batch
  and live engines. Keep FastAPI, Pillow encoding, filesystem paths, and logging out
  of numerical functions.
- Use `numpy.random.Generator` with an explicit seed. Record engine version, seed,
  simulation grid, iteration count, parameters, interpolation, normalization, and
  output transform in every export.
- Define boundary conditions (currently periodic through `np.roll`), time step, seed
  behavior, and parameter-gradient semantics in code/docs.
- Guard normalization when `max == min`; detect all non-finite values, not only NaNs.
  Fail a render clearly instead of recursively/randomly resetting into a different
  result. Decide and document whether concentration clipping is part of the model.
- Replace mutable list defaults with immutable/default-factory configuration.
- Replace `print("Step ...")` with optional progress callbacks and structured logs.
- Fix the standalone entry point so config and output paths resolve independently of
  the current working directory.

Done when:

- The same config and seed yield a byte-identical or explicitly tolerance-bounded
  concentration map on the supported platform.
- Numerical unit tests cover Laplacian boundaries, parameter interpolation, seeding,
  collapse/non-finite handling, normalization, and metadata round-tripping.
- Batch, API render, and live preview call the same tested stepping primitives.

### [x] P1.2 Benchmark before optimizing the engine

CPU benchmark/refactor work is complete with workstation and production OptiPlex
data in `BENCHMARK.md`. The deployed load probe also validates the conservative
two-job capacity policy. Browser decode/paint measurements and a browser-side engine
are product-performance work in P2.3 and P2.4, not unfinished CPU optimization.

- Add a benchmark script that records CPU model, NumPy backend/thread count, grid,
  dtype, iterations/second, frame-encoding time, frame bytes, peak RSS, and concurrent
  session behavior. Save an OptiPlex baseline in the repository.
- Compare `float32` with `float64`; broadcasting four 1-D control vectors with the
  current repeated 2-D maps; and a preallocated/convolution or compiled Laplacian
  with the current nested `np.roll` implementation.
- Verify which BLAS NumPy actually uses. Remove the `mkl` package unless the shipped
  NumPy is demonstrably linked to and helped by it. Consider eliminating SciPy if
  small NumPy/Pillow replacements for interpolation and output resizing are faster
  and simpler.

Done when:

- A written benchmark chooses preview resolution, FPS, server concurrency, dtype,
  and worker/container limits for this exact host.
- Optimization changes include before/after time, memory, and output-difference data.
- The configured capacity probe rejects excess work while health traffic remains
  responsive and admitted sessions continue producing frames.

### [x] P1.3 Add the safety net

- Add `pytest` unit/API/WebSocket tests, including capacity, cancellation, timeout,
  origin, and validation cases.
- Add React tests for controls and socket lifecycle plus one browser end-to-end smoke
  test against the Compose stack.
- Add deterministic small golden arrays/metadata rather than large opaque screenshots.
  Use perceptual/tolerance checks only where byte equality is not a real contract.
- Add a load test for maximum live sessions, one excess session, render saturation,
  disconnect storms, and malicious message rates. Capture event-loop lag, RSS, CPU,
  and rejection latency.
- Add CI for backend lint/typecheck/tests, frontend lint/test/build, Compose validation,
  dependency review, container build, and image scanning.
- Make warnings fail CI once the existing hook warning is fixed.

Done when:

- A clean clone has one documented command for all checks.
- CI blocks a regression in input bounds, capacity release, determinism, or frontend
  socket cleanup.
- The load test completes within configured resource limits with graceful rejections.

### [x] P1.4 Harden and make deployments reproducible

- Lock Python direct and transitive dependencies with hashes. Use `npm ci`, not
  `npm install`, in the frontend image. Pin base-image versions/digests and use a
  project-specific frontend image name rather than tagging the built image as
  `nginx:alpine`.
- Add `.dockerignore` files so local environments, caches, output images, and secrets
  cannot enter build contexts.
- Build Python wheels/dependencies in a builder stage; keep compilers and headers out
  of runtime. Run both images as non-root, use read-only root filesystems where
  possible, drop Linux capabilities, set `no-new-privileges`, and use bounded tmpfs
  storage for temporary renders.
- Set measured CPU, memory, PID, file-descriptor, and log-rotation limits. Use a
  restart policy that does not hide a deterministic crash loop and configure graceful
  shutdown long enough to cancel/clean jobs.
- Add liveness and readiness endpoints. Readiness should reflect whether the service
  can accept API traffic, not promise that render capacity is free.
- Bind local-development ports to `127.0.0.1` unless LAN exposure is intentional.
- Automate dependency/container vulnerability scanning and document a regular update
  cadence. Do not commit generated `.env` values; provide `.env.example` with safe
  placeholders and validation for required settings.

Done when:

- Images build from lockfiles, run without root, pass health checks, and start from a
  clean checkout with documented configuration.
- A worst-case allowed request stays inside container/host limits and leaves enough
  capacity for Traefik and the OS.
- Rollback to the prior tagged image is documented and tested.

### [x] P1.5 Add useful observability without building an observability empire

- Emit structured logs for session/job start, completion, cancellation, rejection,
  duration, and error using IDs rather than client payloads.
- Track active/queued/rejected sessions and jobs, step time, encode time, event-loop
  lag, bytes/frame, render duration, process RSS, and numerical failures.
- Add a small admin-only status view or scrape endpoint and a disk/RSS alert. Keep
  metrics labels low-cardinality; never label by session/job ID.
- Cap log size and avoid per-iteration production logs.

Done when:

- A slow site can be classified as compute saturation, queueing, encoding/network,
  numerical failure, or client disconnect from one short diagnostic session.

## P2 — Deliver the creative product

### [x] P2.1 Design two clear modes: Live Lab and High-Resolution Render

#### [x] P2.1a Build reproducible Live Lab recipes

- A strict versioned recipe records its name, preset, seed, engine version, and every
  numerical control without storing simulation arrays.
- Curated mixed, spots, worms, coral, and maze starting points include plain-language
  descriptions.
- Share URLs take precedence over validated local storage. Copy-link and strict JSON
  import/export make recipes portable.
- Basic recipe and seed controls are separated from the exact endpoint chemistry in
  a collapsible Advanced panel.
- Deterministic restart, random seed, and state-only perturbation are distinct actions.
- Frontend tests cover validation, serialization, restoration, presets, and resets.

#### [x] P2.1b Add complete server-backed Live Lab interactions

- Add display-only preview zoom and contrast without misrepresenting them as
  numerical controls or changing downloaded pixels.
- Support uniform chemistry and explicit edge-gradient editing without exposing
  internal `1`/`2` suffixes in the UI.
- Add a server-bounded single iteration step that automatically pauses and does not
  accept a client-controlled work count.
- Keep at most 30 recipe changes for undo/redo. Restoring history restarts the exact
  recipe rather than pretending to restore transient simulation arrays.
- Keep one bounded before snapshot for visual comparison and recipe restoration.

#### [x] P2.1c Validate Live Lab accessibility and devices

- Validated the responsive workspace on desktop and a real phone over the LAN. The
  pattern is a viewport-bounded sticky preview while recipe, simulation, display,
  history, comparison, sharing, and advanced chemistry controls remain in the normal
  scrolling control flow.
- Added a 390 x 844 Chromium regression test that reaches the final advanced control,
  keeps the complete preview in-frame, confirms the main simulation controls remain
  accessible, and prevents buttons from returning to the sticky preview.
- Preserved native labeled inputs, keyboard-visible focus, semantic status
  announcements, DOM reading order, high-contrast controls, responsive reflow for
  narrow/zoomed layouts, and the reduced-motion override.

#### [x] P2.1d Design the High-Resolution Render experience

- Added an authoritative iteration counter paired with each painted live frame and a
  clear recipe handoff that treats the count as a development reference, not an
  attempt to enlarge the live image.
- Added a bounded fresh-run time study that captures two to six ordered checkpoints
  from one simulation. Selecting a thumbnail sets the high-resolution termination
  step, making early termination a first-class creative control.
- Added physical dimensions in inches/centimetres, Draft/Studio/Fine output detail,
  explicit feature scale, framing, and exact development steps. The validated plan
  separates numerical and output dimensions, the 2x bicubic finish, measured time
  range, estimated memory, resource class, and configured rejection reasons.
- Feature scale is versioned in the plan. Original 1x is executable; Fine 0.5x and
  Bold 2x remain truthfully blocked until a measured spatial-scale implementation is
  available.
- Keep grayscale PNG first. Threshold/invert/levels, transparent blackwork, palette
  mapping, and seamless tiling remain demand-driven follow-ups after the core job
  path. P2.2 supplies queued/running/progress/cancel/expiry states and artifact metadata.

Done when:

- A first-time user can create, alter, save, restore, share, and export a pattern
  without reading the Gray-Scott equations.
- An advanced user can reproduce every control exposed by the current engine.
- Download never points at the placeholder or a stale preview and clearly states its
  pixel dimensions.

### [x] P2.2 Implement high-resolution rendering as a bounded job system

- Added persistent SQLite job metadata, one background worker, a finite waiting queue,
  per-client active limits, `202` submission, status, cancellation, and artifact APIs.
- Jobs share the global compute gate and execute in configured numerical chunks so
  progress, cooperative cancellation, timeout, and graceful shutdown checks cannot
  create an unbounded work request.
- Enforced dimensions, output edge, simulation cells, steps, runtime, queue, client,
  artifact count, and artifact lifetime limits. Original 1x renders execute; requests
  for uncalibrated 0.5x/2x feature scale are rejected before admission.
- Generate into a private persistent volume, remove crash leftovers, atomically publish
  completed PNGs, embed the exact recipe/plan/actual steps and physical resolution,
  expire old files, and never derive filesystem paths from user text.
- Refresh recovers the latest opaque job ID and state. API restart marks running work
  interrupted while preserving queued work and completed artifacts.
- Keep tiling/streaming deferred until periodic boundaries and global normalization can
  be shown to remain visually correct.

Done when:

- Queue length and worst-case memory/disk use are mathematically bounded from config.
- Refreshing the page can recover a job state; cancel and expiry reclaim resources.
- Two identical recipes (including seed/engine version) produce reproducible outputs.
- A render cannot make live/static/health traffic unavailable.

### [x] P2.3 Establish experience and performance budgets

Production OptiPlex measurements, a 30-minute automated public-path browser soak,
and a named iPhone interaction pass now support the accepted initial budgets:

- Added applied-control revisions to frame metadata and the painted canvas so
  control-to-paint latency measures a frame that actually used the requested values.
- Replaced concurrent image decoding with a one-item latest-frame queue. Slow decode
  drops superseded work, exposes diagnostics, and cannot accumulate stale previews.
- Added a repeatable Chromium study for first preview, control response, queue depth,
  retained heap, render progress, and public health/static latency, plus a
  maximum-valid-grid backend RSS/event-loop probe.
- The production OptiPlex maximum 1,024 x 1,024 numerical grid peaked at 122,363,904
  bytes RSS (5.70% of the 2 GiB limit), with 0.66 ms maximum event-loop lag and
  1.00 ms health p95. The server-memory budget is validated with ample headroom.
- The 30-minute public-path Chromium soak completed through Cloudflare with 223.97 ms
  control-to-painted-frame p95, zero control timeouts, zero frame regressions,
  515,608 bytes retained heap growth, and 25.20 ms public-health p95 during a render.
  All automated experience targets passed.
- The configured-capacity production probe admitted two live sessions, rejected one
  excess session, sustained about 9.9 FPS per session, and returned all session slots.
  Peak backend use was 1.32 cores and 72.18 MiB. A focused origin run returned the
  expected two `503` render rejections with 2.83 ms health p95, zero probe errors,
  and no numerical failures.
- A roughly 15-minute iPhone 15 Pro pass on iOS 26.5.2 and Chrome 150.0.7871.113
  consumed three battery percentage points, remained cool, logged no JavaScript
  errors, recovered cleanly after a one-minute background interval, and completed,
  downloaded, and opened a high-resolution render. Exact transient simulation state
  did not survive reconnection, which is expected; the recipe remains the durable
  contract.
- The shorter high-end-phone pass is accepted alongside the automated 30-minute
  mobile-viewport heap/queue trace. Longer and lower-end device coverage belongs to
  P3.4 artist validation and does not block the initial production budget.

- meaningful first preview within 2 seconds on a warm service;
- visible response to a control change within 150 ms for browser-side preview, or
  within 500 ms for the server fallback;
- no stale-frame queue and stable browser memory during a 30-minute session;
- static UI and health endpoints remain responsive under configured maximum work;
- render progress changes often enough that a user can tell it is alive;
- worst-case valid render stays below 70% of the container memory limit, leaving
  recovery headroom.

### [ ] P2.4 Browser-side live simulation (deferred, demand-triggered)

Decision recorded after the production 30-minute soak: keep the server-side live
simulation as the production default. It meets every current experience budget, and
a browser engine would optimize hypothetical demand at substantial implementation
and compatibility cost. This item does not block P3.

Reopen the browser-engine work when production evidence shows at least one of:

- organic sessions are rejected by the two-job capacity limit;
- concurrent live sessions are routinely at capacity;
- control-to-painted-frame p95 exceeds 500 ms under real traffic;
- live-frame streaming consumes an uncomfortable share of origin upload bandwidth;
- hosting cost or reliability makes client-side compute materially valuable; or
- the WebGL/WASM learning work becomes an explicit project goal again.

When reopened:

- Start with a small numerical kernel and fixture comparison. Document how memory,
  typed arrays, JavaScript/WASM boundaries, and rendering fit together before
  changing the production UI.
- Prototype WebGL2 as the broadly available GPU baseline. Evaluate WASM for CPU
  portability and orchestration, and WebGPU only as an optional faster path.
- Benchmark the whole browser frame path: step, normalize, upload/draw, input latency,
  memory growth, and behavior in background tabs.
- Compare representative concentration maps and visual output against the existing
  deterministic engine fixtures with an explicit tolerance.
- Test adaptive frame rate and dynamic steps-per-frame. Drop stale preview work
  instead of accumulating latency.
- Keep the server WebSocket engine as a selectable fallback until representative
  desktop and mobile browsers pass correctness, performance, and lifecycle checks.

Done when:

- The implementation and benchmark make the browser/server tradeoff understandable,
  not magical, and record why a particular WebGL/WASM boundary was selected.
- A compatibility failure falls back cleanly without losing the current recipe.
- Making browser-side preview the default is a measured, reversible decision.

## P3 — Polish, operations, and community

### [ ] P3.1 Repair documentation and contributor experience

- Replace the stale root instructions (`conda install -f`, `src/turing.py`) with the
  tested current local, Compose, test, benchmark, and production workflows.
- Fix README character encoding and consistently use **Gray-Scott**.
- Replace the untouched Vite template README with project-specific frontend notes.
- Document model equations, periodic boundaries, useful/stable ranges, gradients,
  true resolution versus upsampling, reproducibility guarantees, and expected render
  times from the production host.
- Add architecture, WebSocket/render protocol, environment-variable, capacity-tuning,
  backup/cleanup, deployment, rollback, and incident notes.
- Add screenshots and tell the tattoo origin story prominently; it is the most human
  and memorable reason for the project to exist.

### [ ] P3.2 Add release and maintenance discipline

- Use semantic application/engine versions and immutable container tags.
- Add a staging smoke test, production post-deploy health check, and rollback command.
- Schedule dependency updates and quarterly restore/load/security checks.
- Publish a short privacy/retention statement before storing recipes, IP-derived
  quotas, analytics, or user accounts. Avoid accounts until a real feature requires
  them.
- Add an abuse contact and a graceful maintenance/busy page.

### [ ] P3.3 Add a privacy-conscious daily operations digest

- Use Cloudflare's aggregated hostname analytics for estimated visits, requests, and
  transfer rather than storing visitor IP addresses or inventing user identities.
- Persist restart-safe daily application aggregates in `/var/lib/turing`: admitted
  and rejected live sessions, total live duration/frame bytes, peak compute use,
  render requests and outcomes, render duration, and numerical/internal failures.
- Do not retain raw IPs, user agents, recipes, seeds, or per-visitor histories for
  reporting. Document the aggregate telemetry in the privacy statement.
- Run reporting as a separate scheduled job, never in the web request path. Give its
  Cloudflare token analytics-read access only and keep mail credentials out of the
  image and repository.
- Send one plain-text report for the preceding complete day through an SMTP relay or
  transactional mail API. Persist the reported date and delivery result so retries
  cannot send more than one successful digest per day.
- Include visits/page views, accepted/rejected sessions, live minutes, peak capacity,
  approximate streaming bytes, render requested/completed/failed/cancelled counts,
  render timing, errors, and backend restarts. Keep outage alerting separate.

Done when:

- A restart during the day does not lose or double-count application totals.
- A retry or overlapping scheduler invocation cannot send a duplicate daily email.
- No personal data is introduced, and secrets are least-privilege and recoverable.
- The digest provides enough evidence to decide whether P2.4 should be reopened.

### [ ] P3.4 Validate features with actual artists

- Watch a few people create a pattern without coaching. Record where parameter names,
  gradient direction, waiting, and download quality confuse them.
- Include at least one lower-end or mid-range physical phone in longer-session
  testing; record browser lifecycle, thermal behavior, battery use, and responsiveness.
- Ask tattoo artists what formats, DPI guidance, threshold controls, line cleanup,
  repeatability, and stencil workflows are actually useful before building SVG or
  elaborate editing features.
- Favor curated presets and excellent exports over adding more unexplained sliders.

## Recommended implementation order

Keep early pull requests small enough to measure and reverse:

1. **Protocol and correctness:** schemas/bounds, origin checks, slider stale-state fix,
   blob cleanup, and tests for those behaviors.
2. **Capacity and isolation:** disable or queue `/generate`, offload live compute from
   the event loop, robust cancellation, health/metrics, and an OptiPlex load baseline.
3. **Reproducibility:** seeded engine/config split, numerical fixtures, metadata, and
   a locked/reproducible container build.
4. **Product split:** polished Live Lab plus a one-worker render-job API and UI.
5. **Operational evidence:** add the privacy-conscious daily digest and observe real
   traffic before changing the simulation architecture.
6. **Conditional server relief:** reopen browser-side simulation only when measured
   demand or the explicit learning goal justifies it; retain the backend fallback.
7. **Creative polish:** presets, sharing, blackwork/export controls, accessibility,
   documentation, and artist feedback.

## Definition of “glory”

This roadmap is complete when the site is fun before it is technical: it responds
immediately, explains itself, remembers the exact recipe for a beautiful accident,
and produces a trustworthy high-resolution file. Operationally, the same success
means compute, memory, queueing, storage, and client lifetimes are bounded; malformed
or excess work is rejected clearly; and the OptiPlex remains observable and healthy
without constant babysitting.
