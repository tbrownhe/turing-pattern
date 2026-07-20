# Experience and performance budgets

P2.3 turns responsiveness and resource safety into repeatable measurements before
the browser-side simulation experiment. These are initial targets to validate, not
permission to raise production capacity.

| Measurement | Initial target |
| --- | ---: |
| Warm server-backed first preview | at most 2,000 ms |
| Server-backed control change to painted acknowledged frame, p95 | at most 500 ms |
| Browser frames waiting for decode | at most 1 |
| Retained Chromium heap growth over 30 minutes | at most 16 MiB |
| Public health and static UI latency during a render, p95 | at most 250 ms |
| Progress gap for a render lasting more than 5 seconds | at most 5,000 ms |
| Peak backend RSS during the maximum valid grid | below 70% of container limit |

## Browser study

The Chromium runner uses the real UI at a 390 x 844 viewport. It measures the first
session and then a second warm session, changes a real slider, waits until a painted
canvas reports the matching applied-control revision, runs a representative queued
render, and samples Chromium heap and frame diagnostics. Periodic control activity
keeps the 30-minute session intentionally active rather than defeating the server's
idle-session policy.

From a laptop with Node installed:

```console
cd frontend
npm ci
npx playwright install chromium
node scripts/experience-budget.mjs \
  --url https://turing.tobiasbrownheft.xyz \
  --duration 1800 \
  --sample-interval 5 \
  --activity-interval 300 \
  --control-trials 10 \
  --output ../soak-results.json
```

The runner prints progress once per minute. The output file starts as a running
checkpoint and is replaced with the complete report after a successful run, so an
external interruption still leaves its last observed state. In PowerShell, run the
same `node` command on one line or replace the backslash continuations with backticks.

Add `--cpu-throttle 4` as a reproducible constrained-laptop comparison, not as a
claim that desktop emulation represents a particular phone. Add `--enforce` only
after the team accepts the targets. Pair the automated trace with a named physical
device interaction pass; use remote browser instrumentation when the available
phone/computer combination supports it.

## Maximum-grid server memory

The memory probe requests a 2,048 x 2,048 PNG backed by the largest currently valid
1,024 x 1,024 numerical grid for 100 steps. Grid allocation, rather than a long
evolution, is the relevant worst-case memory condition. It samples private backend
metrics while also checking public health latency and the final PNG.

The probe is a client of the real backend; it does not start an API server itself.
Build and start the deployed backend, then stream the checked-in probe into that
exact running container:

```console
docker compose up -d --build backend
docker compose exec -T backend \
  python - \
  --url http://127.0.0.1:8000 \
  --metrics-url http://127.0.0.1:8000/metrics \
  < scripts/render_memory_probe.py
```

This avoids creating a second container with the `backend` service identity and
guarantees that the API and metrics belong to the deployed process being measured.
The default calculation assumes the current 2 GiB backend limit. Pass
`--memory-limit-mb` before the input redirection if production uses a different
limit.

## Short development validation

These Windows development-machine runs prove that the instruments work; they are
not the required OptiPlex or real-device baselines.

| Measurement | Short local result |
| --- | ---: |
| First session / warm session preview | 113 / 110 ms |
| Control-to-painted-frame p95, 3 samples | 196 ms |
| Maximum pending frames / frame regressions | 0 / 0 |
| Five-second retained heap growth | 50,888 bytes |
| Representative render duration / maximum progress gap | 2,387 / 269 ms |
| Public health / static UI p95 during render | 2.71 / 2.73 ms |
| Maximum-grid peak backend RSS | 128,532,480 bytes |
| Peak fraction of 2 GiB / maximum event-loop lag | 5.99% / 0.89 ms |

## Production and device results

### OptiPlex 5060 public-path Chromium soak

Recorded at 2026-07-20 01:06 UTC with headless Chromium on the development
workstation at a 390 x 844 viewport. Browser traffic reached the production
OptiPlex through Cloudflare, Traefik, and the same-origin Nginx proxy.

| Measurement | Production result |
| --- | ---: |
| First / warm preview | 337.62 / 306.30 ms |
| Control-to-painted-frame p95 / timeouts | 223.97 ms / 0 of 15 |
| Session duration / samples | 1,800 s / 360 |
| Maximum pending frames / frame regressions | 0 / 0 |
| Superseded frames dropped | 10 |
| Retained heap growth / DOM node growth | 515,608 bytes (0.49 MiB) / 0 |
| Representative render duration / progress updates | 6,858.49 ms / 21 |
| Maximum render progress gap | 566.99 ms |
| Public health / static UI p95 during render | 25.20 / 24.75 ms |
| All initial targets met | yes |

The deliberately bounded latest-frame path discarded ten superseded frames rather
than accumulating stale work. No frame IDs regressed, no sampled frame remained
queued, and retained heap growth used about 3% of the 16 MiB budget. This validates
the automated 30-minute public-path baseline; it does not replace a trace and
interaction pass on a named physical device.

### OptiPlex 5060 maximum-grid memory

Recorded on the production OptiPlex container with a 2 GiB backend limit. The
1,024 x 1,024 simulation is the largest numerical grid accepted by current config.

| Measurement | Production result |
| --- | ---: |
| Output / simulation grid | 2,048 x 2,048 / 1,024 x 1,024 |
| Development steps / duration | 100 / 4.457 s |
| Peak backend RSS | 122,363,904 bytes (116.7 MiB) |
| Peak fraction of container limit | 5.70% |
| Maximum event-loop lag | 0.66 ms |
| Health mean / p95 | 0.82 / 1.00 ms |
| Artifact size | 1,575,335 bytes |
| Below 70% memory budget | yes |

This validates the maximum-grid allocation budget with substantial recovery
headroom. It does not estimate the runtime of a 20,000-step render; runtime remains
bounded separately by the render timeout and should be calibrated from representative
jobs.

### OptiPlex 5060 configured-capacity load

Recorded against the production two-job configuration. A five-minute public probe
through Cloudflare validated sustained WebSocket work; Cloudflare rejected some
synthetic Python HTTP probes at the edge, so a focused 30-second private-network run
through the deployed Nginx frontend separately validated origin health and admission.
Cloudflare security was not weakened for the test.

| Measurement | Production result |
| --- | ---: |
| Sustained live sessions admitted / excess rejected | 2 / 1 |
| Public-run frames / disconnects | 5,946 / 0 |
| Approximate delivered frame rate | 9.9 FPS per session |
| Peak backend CPU | 132.04% (1.32 cores) |
| Peak backend container memory | 72.18 MiB (3.52% of 2 GiB) |
| Focused origin-run frames / disconnects | 594 / 0 |
| Saturated render responses | 503, 503 |
| Origin health mean / p95 / max | 1.33 / 2.83 / 12.29 ms |
| Origin probe errors | 0 |
| Session starts / finishes per run | 22 / 22 |
| Final active / waiting compute | 0 / 0 |
| Numerical failures | 0 |

The five-minute run added 5,948 backend frame observations and about 273 MiB of
encoded frame bytes. The focused origin run added the expected render rejections and
returned every sustained and disconnect-storm session slot. This validates the
conservative two-job limit; it is not evidence for raising concurrency.

### iPhone 15 Pro interaction pass

Recorded on an iPhone 15 Pro running iOS 26.5.2 and Chrome 150.0.7871.113 over
Wi-Fi. The hands-on pass lasted approximately 15 minutes.

| Measurement | Physical-device result |
| --- | ---: |
| Battery | 81% to 78% |
| Thermal observation | remained cool |
| Background/foreground | restarted cleanly |
| Exact live numerical state after restart | not preserved, as expected |
| High-resolution render | completed smoothly |
| Downloaded artifact | opened successfully in Downloads |
| JavaScript errors from `chrome://inspect` | 0 |
| Touch controls | no reproducible issue |

The physical run was shorter than the originally proposed 30 minutes and used a
high-end rather than mid-range phone. That scope is accepted because the separate
30-minute automated mobile-viewport run already supplied retained-heap, frame-queue,
and latency evidence; the iPhone pass supplied the missing real-device touch,
lifecycle, download, battery, and thermal evidence. Full remote heap instrumentation
for Chrome on iOS requires Safari on a Mac and was not available from the Windows
development machine. Longer and lower-end physical-device coverage moves to P3.4
artist testing rather than blocking this initial production budget.

All initial P2.3 budgets are accepted. Revisit them when daily operations data shows
capacity rejection, latency regression, or uncomfortable streaming bandwidth.

## P2.3 reproduction procedure

### 1. Configured-maximum correlated load

The 30-minute Chromium soak above validates the complete Cloudflare-to-origin path.
Run the configured-capacity probe on the private project network so Cloudflare bot
controls cannot turn an origin-capacity measurement into an edge-security test. Run
it during a quiet five-minute production window: it intentionally occupies both
configured compute slots, so real visitors may receive the normal busy response.

On the OptiPlex, capture the private counters before the run:

```console
docker compose exec -T backend python -c \
  'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8000/metrics").read().decode(), end="")' \
  > /tmp/turing-metrics-before.txt
```

Keep `docker stats turing-backend-1 turing-frontend-1` visible in a second OptiPlex
terminal. From the OptiPlex repository root, start a disposable client container on
the private network. Use `docker run`, not `docker compose run backend`: the latter
would duplicate the deployed backend's network aliases.

```console
docker run --rm \
  --network turing_default \
  -v "$PWD:/workspace:ro" \
  turing-backend:prod \
  python /workspace/scripts/load_test.py \
  --url http://frontend:8080 \
  --clients 2 \
  --excess 1 \
  --renders 2 \
  --duration 300 \
  --disconnect-storm 20 \
  --message-rate 2
```

Then capture and compare the private counters on the OptiPlex:

```console
docker compose exec -T backend python -c \
  'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8000/metrics").read().decode(), end="")' \
  > /tmp/turing-metrics-after.txt
diff -u /tmp/turing-metrics-before.txt /tmp/turing-metrics-after.txt
```

Record the load-test JSON, peak backend CPU/RSS observed in `docker stats`, and the
metric delta. Expected behavior is two admitted sessions, one prompt capacity
rejection, two `503` render responses while both slots are occupied, continued
frames, no probe errors, health p95 below 250 ms, no numerical failures, and active
and waiting compute returning to zero afterward.

### 2. Named physical-device trace

Record the device model, OS, browser/version, and network. Prefer 30 minutes with
several control changes and at least one pause/resume and background/foreground
cycle. Capture whatever memory/performance instrumentation the device supports,
note reconnects or visual failures, and confirm the UI remains usable.
