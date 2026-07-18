// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import RenderStudio from './RenderStudio'
import { recipeForPreset } from './recipe'

const recipe = recipeForPreset('maze', 42)

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('high-resolution render planning', () => {
  it('resolves physical inputs through the server planner', async () => {
    const importSettings = vi.fn()
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        accepted: true,
        issues: [],
        quality_label: 'Studio',
        pixels_per_inch: 300,
        feature_scale: 1,
        scale_model_status: 'calibration-required',
        development_steps: 5000,
        output_width: 1800,
        output_height: 1800,
        simulation_width: 900,
        simulation_height: 900,
        simulation_pixels: 810000,
        bicubic_upsample: 2,
        estimated_seconds_low: 125,
        estimated_seconds_high: 220,
        estimated_memory_bytes: 90920000,
        resource_class: 'heavy',
        engine_version: '2.0.0',
      }),
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<RenderStudio recipe={recipe} liveSteps={5000} handoffVersion={1} onImportLiveSettings={importSettings} />)

    fireEvent.click(screen.getByRole('button', { name: 'Import current Live Lab settings' }))
    expect(importSettings).toHaveBeenCalledOnce()

    fireEvent.click(screen.getByRole('button', { name: 'Review render plan' }))

    expect(await screen.findByText('1,800 × 1,800 px')).toBeTruthy()
    const request = JSON.parse(fetchMock.mock.calls[0][1].body)
    expect(request).toEqual(expect.objectContaining({
      width: 6,
      height: 6,
      unit: 'in',
      quality: 'studio',
      feature_scale: 1,
      development_steps: 5000,
      controls: recipe.controls,
      seed: 42,
    }))
  })

  it('uses a selected time-study checkpoint as the development target', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        engine_version: '2.0.0',
        simulation_size: 256,
        seed: 42,
        checkpoints: [
          { steps: 1250, image_url: 'data:image/png;base64,AA==' },
          { steps: 2500, image_url: 'data:image/png;base64,AA==' },
          { steps: 3750, image_url: 'data:image/png;base64,AA==' },
          { steps: 5000, image_url: 'data:image/png;base64,AA==' },
        ],
      }),
    }))
    render(<RenderStudio recipe={recipe} liveSteps={5000} handoffVersion={1} onImportLiveSettings={vi.fn()} />)

    fireEvent.click(screen.getByRole('button', { name: 'Preview development stages' }))
    const checkpoint = await screen.findByRole('button', { name: /2,500 steps/ })
    fireEvent.click(checkpoint)

    await waitFor(() => {
      expect((screen.getByLabelText('Evolution steps') as HTMLInputElement).value).toBe('2500')
    })
  })
})
