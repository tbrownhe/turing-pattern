# Turing Pattern frontend

This package is the React/TypeScript interface for the Gray-Scott Pattern Lab. It
owns recipe editing, the live WebSocket client, bounded frame painting, render
planning, time studies, queued-job recovery, and responsive desktop/phone layout.
Numerical simulation and input authority remain in the backend.

## Develop

From `frontend/`:

```console
npm ci
npm run dev
```

Vite serves <http://localhost:5173> and proxies `/api` and `/ws` to the backend at
<http://127.0.0.1:8000>. Browser code deliberately uses same-origin relative URLs;
there is no `VITE_API_URL` to configure.

Useful commands:

```console
npm test
npm run lint -- --max-warnings=0
npm run build
npm run test:e2e
```

Run `python scripts/check.py` from the repository root for the complete backend,
frontend, audit, build, and browser gate.

## Important modules

- `src/App.tsx` owns Live Lab state, recipes, WebSocket lifecycle, and the sticky
  preview/control layout.
- `src/protocol.ts` defines the browser-side control vocabulary and hard bounds.
- `src/recipe.ts` defines versioned recipes and curated presets.
- `src/RenderStudio.tsx` owns physical-size planning, time studies, queue polling,
  cancellation, recovery, and artifact download.
- `e2e/app.e2e.ts` verifies a real live frame/control update and the phone viewport
  contract.
- `scripts/experience-budget.mjs` performs the long browser/production budget run.

## UI and protocol invariants

- Keep the simulation frame visible while phone controls scroll.
- Keep at most one decoded frame pending and never paint a frame ID regression.
- Pair every binary PNG with its preceding frame metadata before painting it.
- Revoke replaced object URLs and stop/reconnect cleanly after backgrounding.
- Recipe changes are undoable and serializable; display zoom/contrast are not recipe
  parameters.
- Importing Live Lab settings into Render Studio starts a new simulation. It never
  claims to reproduce the current transient frame.
- Production remains same-origin through Nginx. Do not add cross-origin API URLs
  without revisiting CORS, WebSocket origin checks, CSP, and deployment tests as one
  security-boundary change.

See the root [README](../README.md), [operations runbook](OPERATIONS.md), and
[experience budgets](BUDGETS.md) for the complete system contract.
