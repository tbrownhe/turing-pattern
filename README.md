# Gray-Scott Turing Pattern Lab

An interactive reaction-diffusion playground for discovering organic textures and
exporting reproducible patterns. This began as a local Python script for tattoo
ideas; one of its patterns now lives on the author's arm. The current project turns
that experiment into a deliberately small, self-hosted web application.

![Turing pattern](static/turing.png)

## What works today

- A live 256×256 simulation with horizontal feed/kill gradients and vertical
  diffusion gradients.
- Curated pattern families plus named, versioned recipes containing the exact seed,
  controls, and engine version.
- URL and local restoration, copy-link sharing, and strict JSON recipe import/export.
- Pause, one-iteration step, deterministic restart, random seed, state-only
  perturbation, reconnect, and leak-free canvas rendering.
- Bounded recipe undo/redo and one-frame before/after comparison.
- Display-only preview zoom and contrast that never alter the saved numerical recipe.
- A bounded PNG endpoint at `POST /api/v1/generate` for the current fixed-size
  export. The full queued high-resolution workflow is still on the roadmap.
- Strict versioned inputs, WebSocket origin checks, a shared compute limit, and
  CPU work isolated from FastAPI's event loop.
- Same-origin production routing: Traefik exposes Nginx, and Nginx proxies `/api`
  and `/ws` to a private backend container.

See [TODO.md](TODO.md) for the measured-performance, high-resolution, and eventual
browser-side simulation roadmap. The WASM/WebGL exploration has intentionally not
started yet; it will be treated as a learning-oriented prototype rather than a
silent rewrite.

## Run locally with Docker

```console
docker compose -f docker-compose.local.yml up --build
```

Open <http://localhost:3000>. The backend is also bound to
<http://127.0.0.1:8000> for local API inspection, and its docs are available at
<http://127.0.0.1:8000/docs>. Both published ports bind only to loopback.

### Test from another device on the LAN

LAN exposure is opt-in. Keep the backend private and publish only the same-origin
frontend proxy. In `.env`, set:

```dotenv
TURING_LOCAL_BIND_ADDRESS=0.0.0.0
TURING_LOCAL_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://192.168.1.71:3000
```

Replace `192.168.1.71` if the development machine's address changes, then recreate
the local containers:

```console
docker compose -f docker-compose.local.yml up -d --build --force-recreate
```

Open <http://192.168.1.71:3000> on the phone. If the page is still unreachable,
allow inbound TCP port 3000 on the development machine's **private** firewall
profile and confirm both devices are on the same non-isolated LAN. Do not expose
port 8000; Nginx already proxies `/api` and `/ws` to the private backend.

## Develop without rebuilding containers

Backend, from the repository root:

```console
cd backend
uv run --with-requirements requirements-dev.lock \
  python -m uvicorn app.api.main:app --reload
```

Frontend, in another terminal:

```console
cd frontend
npm ci
npm run dev
```

Vite proxies `/api` and `/ws` to the local backend, so the browser still uses a
single origin at <http://localhost:5173>.

Run the checks:

```console
python scripts/check.py
```

This uses the hash-locked Python dependency set and runs backend lint, formatting,
type checking and tests plus frontend audit, tests, lint and build. See
[BENCHMARK.md](BENCHMARK.md) for engine measurements and
[docs/OPERATIONS.md](docs/OPERATIONS.md) for deployment, tuning and rollback.
The first run also downloads Playwright's pinned Chromium build for the real-browser
live-frame smoke test.

## Live Lab recipes

The basic panel starts from mixed, spots, worms, coral, maze, or the order–disorder
transition pictured above, and keeps the exact numerical controls in the Advanced
panel. A recipe contains only compact, validated metadata—never the evolving
concentration arrays:

```json
{
  "recipe_version": 1,
  "engine_version": "2.0.0",
  "name": "Branching coral",
  "preset": "coral",
  "seed": 42,
  "controls": {
    "F1": 0.054,
    "F2": 0.058,
    "K1": 0.061,
    "K2": 0.064,
    "Du1": 0.74,
    "Du2": 0.66,
    "Dv1": 0.27,
    "Dv2": 0.23
  }
}
```

Shared URL recipes take precedence over the last recipe in local storage. Imported
JSON is rejected if it has unknown fields, an unsupported version, unsafe values,
or an invalid seed. Changing a raw control evolves the current state; applying a
preset or seed restarts deterministically. Perturbation intentionally changes only
the current state and therefore does not alter the saved recipe.

## Public protocol

The browser opens `/ws` and first sends protocol version 1:

```json
{
  "type": "start",
  "protocol_version": 1,
  "seed": 0,
  "controls": {
    "F1": 0.04,
    "F2": 0.08,
    "K1": 0.056,
    "K2": 0.074,
    "Du1": 0.7,
    "Du2": 0.7,
    "Dv1": 0.25,
    "Dv2": 0.25
  }
}
```

Subsequent message types are `controls`, `pause`, `resume`, `step`, `reset`, and
`perturb`. A `step` advances exactly one numerical iteration and leaves the session
paused; clients cannot supply an iteration count.
Unknown fields, non-finite values, and values outside the documented schema are
rejected. Clients cannot choose simulation allocation size. The server's `ready`
message identifies the engine version recorded by the recipe UI.

The fixed-size PNG endpoint accepts the same controls and a seed:

```console
curl -X POST http://127.0.0.1:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"seed":7,"controls":{"F1":0.04,"F2":0.08,"K1":0.056,"K2":0.074,"Du1":0.7,"Du2":0.7,"Dv1":0.25,"Dv2":0.25}}' \
  --output pattern.png
```

PNG exports contain the recipe in the `TuringParams` text field.

## Capacity and security model

One Uvicorn process owns one in-memory admission controller. Live sessions and PNG
requests share its slots, and all NumPy/Pillow work runs in a dedicated bounded
executor. Do not add Uvicorn workers: each process would create another independent
limit. A future durable render service will own multi-process high-resolution work.

The production defaults admit two compute jobs and two short-lived waiters. Excess
work is rejected with HTTP `503` or WebSocket close code `1013`. Traefik adds request
and in-flight limits, while Nginx adds request-size limits and security headers.
Application validation remains necessary because edge limits do not govern work
performed after a WebSocket handshake.

Copy `.env.example` to `.env` and set at least `STACK_NAME` and `DOMAIN`. Important
tuning variables are:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `TURING_MAX_COMPUTE_JOBS` | `2` | Total admitted live sessions and renders |
| `TURING_MAX_COMPUTE_WAITERS` | `2` | Maximum short admission waiting room |
| `TURING_COMPUTE_WORKERS` | `2` | Dedicated CPU worker threads; must not exceed jobs |
| `TURING_ADMISSION_TIMEOUT_SECONDS` | `0.25` | How quickly excess work is rejected |
| `TURING_IDLE_TIMEOUT_SECONDS` | `600` | Live session inactivity limit |
| `TURING_FRAME_RATE` | `10` | Maximum server preview frames per second |
| `TURING_STEPS_PER_FRAME` | `25` | Numerical iterations between preview frames |
| `TURING_LOG_LEVEL` | `INFO` | Structured JSON log threshold |
| `OPENBLAS_NUM_THREADS` | `1` | Native threads per compute worker |
| `OMP_NUM_THREADS` | `1` | OpenMP threads per compute worker |
| `BACKEND_CPU_LIMIT` | `4.0` | Backend container CPU ceiling |
| `BACKEND_MEMORY_LIMIT` | `2g` | Backend container memory ceiling |

Production API docs are disabled. Browser WebSocket origins must exactly match the
configured public origin; originless non-browser clients are allowed by default but
remain subject to the same protocol and compute limits.

## Model

The Gray-Scott model simulates two concentrations, `U` and `V`:

- `∂U/∂t = Du ∇²U - UV² + F(1 - U)`
- `∂V/∂t = Dv ∇²V + UV² - (F + k)V`

`Du` and `Dv` are diffusion rates, `F` replenishes `U`, and `k` removes `V`.
Autocatalysis through `UV²` produces spots, stripes, worms, and maze-like structures.
The current discrete Laplacian uses periodic boundaries. Endpoint controls are
linearly interpolated across the image. A reset recreates the exact seeded initial
state; perturbation instead adds seeded noise to the current evolving state. The
engine does not clip concentrations, and non-finite results fail explicitly.

The original batch configuration remains in
`backend/app/core/turing_parameters.json`; its standalone entry point resolves
configuration and output paths independently of the current working directory.

## Deployment

Production expects an existing Docker network named `traefik-public` and Traefik
entrypoints named `http` and `https`, plus a certificate resolver named `le`.

```console
cp .env.example .env
# edit DOMAIN and STACK_NAME
docker compose config
docker compose up -d --build
```

Only the frontend joins `traefik-public`. The backend is reachable from Nginx on the
private Compose network and has health checks, CPU/memory/PID limits, bounded logs,
and `no-new-privileges` enabled.

## License

MIT. See [LICENSE](LICENSE).
