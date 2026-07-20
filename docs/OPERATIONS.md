# Operations

## Release model

The application follows semantic versioning and is currently `1.0.0`. The numerical
engine has its own semantic version (`2.0.0`), while recipes, WebSocket messages, and
render metadata carry small independent format versions. Change the engine version
only when numerical output or reproducibility changes.

For a release, update the application version in `backend/app/__init__.py`,
`frontend/package.json`, its lockfile, and `.env.example`; run the complete check;
then create a matching Git tag. Set production `TAG` to that immutable release tag or
an exact commit SHA—never `latest`, `prod`, or a moving branch name. Both images carry
the application version as an OCI label.

Before production, exercise the candidate through a staging hostname with the same
proxy/security topology:

```console
uv run --with websockets python scripts/smoke_test.py \
  --url https://staging.example.com
```

The smoke test checks the static UI, CSP, public health route, deterministic PNG, and
a real WebSocket frame. It is safe for a dedicated staging service; do not point
automated high-rate load tests at production through Cloudflare.

## Deploy

Use an immutable `TAG` (a release or commit SHA), never `latest`:

```console
cp .env.example .env
# Set DOMAIN, STACK_NAME, and TAG.
docker compose config
docker compose build
docker compose up -d
docker compose ps
```

The API runs as UID/GID 10001 and Nginx runs as its unprivileged `nginx` user on
port 8080. Both root filesystems are read-only, all Linux capabilities are dropped,
and only bounded `/tmp` tmpfs storage plus the backend's `/var/lib/turing`
`render-data` volume are writable. That volume owns the render-job SQLite database
and completed PNG artifacts; include it in backups. Do not add Uvicorn workers: the
capacity gate and render worker are process-local.

Before changing the stack, record the current commit and image tag. After deployment,
run the public smoke test and keep the prior tag available until the new version has
survived a real live session and queued render.

## Diagnose

Public checks go through Nginx:

```console
curl -fsS https://$DOMAIN/healthz
uv run --with websockets python scripts/smoke_test.py --url https://$DOMAIN
```

`/readyz` and `/metrics` are backend-private and available from the host/container
network. Metrics distinguish active/waiting work, rejections, numerical failures,
render jobs queued/started/completed/cancelled/failed, step/encode/render time,
event-loop lag, bytes per frame, and RSS. Session/request IDs appear only in
structured logs, never as metric labels.

Useful commands:

```console
docker compose ps
docker compose logs --tail=200 backend
docker compose exec backend python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/readyz').read().decode())"
docker compose exec backend python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/metrics').read().decode())"
```

Classify a slow service in this order: event-loop lag, active/waiting/rejected work,
step time, encode time and frame size, RSS, then client/network behavior. Alert when
RSS remains above 70% of the 2 GiB limit or event-loop lag remains above 250 ms.

## Tune on the OptiPlex

Run `scripts/engine_benchmark.py` and `scripts/load_test.py` from
[BENCHMARK.md](BENCHMARK.md), then run the browser and maximum-grid studies in
[BUDGETS.md](BUDGETS.md). Increase only one of worker count, admission count,
resolution, FPS, or steps per frame at a time. Health latency and RSS must retain
headroom for Traefik and the OS. A busy response is healthy overload behavior.

## Roll back

Keep the previous immutable tag. To roll back code and both images:

```console
# Set TAG to the prior known-good tag in .env.
docker compose pull
docker compose up -d --no-build
docker compose ps
curl -fsS https://$DOMAIN/healthz
```

Preserve the `render-data` volume during rollback. The current schema is additive;
startup retains queued and completed jobs and marks work that was running during a
restart as interrupted. A future incompatible schema change must include an explicit
migration and rollback procedure. Test this procedure in staging before each release.

## Back up and restore

The `render-data` Docker volume contains both SQLite databases, delivery claims, and
unexpired artifacts. Identify its exact host volume name without guessing:

```console
docker inspect "$(docker compose ps -q backend)" \
  --format '{{range .Mounts}}{{if eq .Destination "/var/lib/turing"}}{{println .Name}}{{end}}{{end}}'
```

Use the host's existing Docker-volume backup process to snapshot that volume while
the backend and reporter are stopped. A useful backup must preserve file ownership
for UID/GID 10001 and include `render-jobs.sqlite3`, `usage.sqlite3`, their SQLite
sidecars if present, and `artifacts/`. Never copy only the main SQLite files while a
writer is running.

Test restoration quarterly into a disposable volume and start a one-off backend
against it before treating the backup as valid. Restoring production is a maintenance
operation: stop the backend and reporter, retain the damaged volume until validation
is complete, restore into a new volume, verify ownership, then run readiness, metrics,
live-session, queued-render, and report dry-run checks.

## Incident guide

- **Busy but healthy:** fast `503`/WebSocket `1013` responses with responsive health
  checks mean admission control is working. Inspect active/waiting metrics before
  changing capacity.
- **Backend unhealthy:** preserve logs and metrics, restart once, then roll back to
  the previous immutable tag if health does not recover. Do not delete the volume.
- **Disk pressure:** inspect the named volume and Docker logs. Let configured TTL and
  artifact/history limits clean up; do not edit SQLite or remove active artifacts.
- **Numerical failures:** capture the opaque request/session/job ID and engine version,
  not the user's full recipe. Reproduce only from recipe data the user elects to share.
- **Reporting failure:** inspect `turing-report.service`. A claimed failed date is not
  automatically retried because SMTP acceptance may have been ambiguous; use a dry
  run for diagnosis and leave the recorded claim intact.
- **Planned maintenance:** leave the frontend online when possible. It preserves the
  recipe controls and presents an unavailable/busy status with a manual retry while
  the backend is stopped, rather than replacing the site with an opaque proxy error.

## Maintenance

- Monthly: refresh Python locks and `package-lock.json`, run audits, CI, and image
  scanning, then benchmark unexpected numerical/performance changes.
- Quarterly: run the OptiPlex load test, rollback drill, and 30-minute browser
  memory session.
- After every deploy: verify health, one WebSocket frame, security headers, and one
  deterministic PNG recipe using `scripts/smoke_test.py`, then submit and download a
  small queued high-resolution render.
- Regularly verify free space on the Docker volume. Completed artifacts are bounded
  by both `TURING_MAX_RENDER_ARTIFACTS` and
  `TURING_RENDER_ARTIFACT_TTL_SECONDS`, and terminal metadata is bounded by
  `TURING_MAX_RENDER_JOB_HISTORY`; do not manually edit `render-jobs.sqlite3`.
