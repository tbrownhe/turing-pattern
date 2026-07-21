import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  testMatch: '**/*.e2e.ts',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: 'http://127.0.0.1:5173',
    trace: 'on-first-retry',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    {
      command:
        'uv run --with-requirements requirements-dev.lock python -m uvicorn app.api.main:app --host 127.0.0.1 --port 18001',
      cwd: '../backend',
      url: 'http://127.0.0.1:18001/healthz',
      reuseExistingServer: !process.env.CI,
      env: {
        TURING_ALLOWED_ORIGINS: 'http://127.0.0.1:5173',
        TURING_PREVIEW_SIZE: '64',
        TURING_STEPS_PER_FRAME: '1',
        TURING_FRAME_RATE: '10',
        TURING_RENDER_SIZE: '32',
        TURING_RENDER_STEPS: '100',
      },
    },
    {
      command:
        'npm run dev -- --host 127.0.0.1 --config vite.e2e.config.ts',
      cwd: '.',
      url: 'http://127.0.0.1:5173',
      reuseExistingServer: !process.env.CI,
    },
  ],
})
