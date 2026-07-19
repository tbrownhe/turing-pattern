# Operations

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

## Diagnose

Public checks go through Nginx:

```console
curl -fsS https://$DOMAIN/healthz
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
[BENCHMARK.md](../BENCHMARK.md). Increase only one of worker count, admission count,
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
