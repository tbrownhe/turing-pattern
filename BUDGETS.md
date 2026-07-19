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
npm run budget -- \
  --url https://turing.tobiasbrownheft.xyz \
  --duration 1800 \
  --sample-interval 5 \
  --activity-interval 300 \
  --control-trials 10
```

Add `--cpu-throttle 4` as a reproducible constrained-laptop comparison, not as a
claim that desktop emulation represents a particular phone. Add `--enforce` only
after the team accepts the targets. A real mid-range phone still needs a 30-minute
Chrome/Safari memory trace and a short usability pass; paste those device/browser
details and results below.

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

Pending:

- 30-minute Chromium run against the production OptiPlex.
- 30-minute trace and interaction check on a named mid-range phone or laptop.
- Maximum-grid memory probe on the production OptiPlex.
- Configured-maximum `load_test.py` run correlated with backend metrics.

P2.3 closes only after those results are recorded and the initial targets are either
accepted or revised with an explanation.
