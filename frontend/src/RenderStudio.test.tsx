// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import RenderStudio from './RenderStudio'
import { recipeForPreset } from './recipe'

const recipe = recipeForPreset('maze', 42)

afterEach(() => {
  cleanup()
  window.localStorage.clear()
  vi.unstubAllGlobals()
})

describe('high-resolution render planning', () => {
  it('restores validated render settings imported from a PNG', () => {
    render(
      <RenderStudio
        recipe={recipe}
        liveSteps={3200}
        handoffVersion={1}
        importedSettings={{
          width: 12,
          height: 18,
          unit: 'cm',
          quality: 'fine',
          featureScale: 0.5,
          developmentSteps: 3200,
          framing: 'fit',
        }}
        onImportLiveSettings={vi.fn()}
        onImportRecipeFile={vi.fn()}
      />,
    )

    expect((screen.getByLabelText('Width') as HTMLInputElement).value).toBe('12')
    expect((screen.getByLabelText('Height') as HTMLInputElement).value).toBe('18')
    expect((screen.getByLabelText('Units') as HTMLSelectElement).value).toBe('cm')
    expect((screen.getByLabelText('Evolution steps') as HTMLInputElement).value)
      .toBe('3200')
    expect(screen.getByRole('radio', { name: /Fine.*600/ })).toHaveProperty('checked', true)
    expect(screen.getByRole('radio', { name: /Fine.*0.5/ })).toHaveProperty('checked', true)
    expect((screen.getByDisplayValue('Fit without stretching') as HTMLSelectElement).value)
      .toBe('fit')
  })

  it('resolves physical inputs through the server planner', async () => {
    const importSettings = vi.fn()
    const importRecipeFile = vi.fn()
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
    render(<RenderStudio recipe={recipe} liveSteps={5000} handoffVersion={1} onImportLiveSettings={importSettings} onImportRecipeFile={importRecipeFile} />)

    fireEvent.click(screen.getByRole('button', { name: 'Import current Live Lab settings' }))
    expect(importSettings).toHaveBeenCalledOnce()
    fireEvent.click(screen.getByRole('button', { name: 'Load saved JSON/PNG' }))
    expect(importRecipeFile).toHaveBeenCalledOnce()

    fireEvent.click(screen.getByRole('button', { name: 'Review render plan' }))

    expect(await screen.findByText('1,800 × 1,800 px')).toBeTruthy()
    const request = JSON.parse(fetchMock.mock.calls[0][1].body)
    expect(request).toEqual(expect.objectContaining({
      recipe_name: 'Slow maze',
      recipe_preset: 'maze',
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
    render(<RenderStudio recipe={recipe} liveSteps={5000} handoffVersion={1} onImportLiveSettings={vi.fn()} onImportRecipeFile={vi.fn()} />)

    fireEvent.click(screen.getByRole('button', { name: 'Preview development stages' }))
    const checkpoint = await screen.findByRole('button', { name: /2,500 steps/ })
    fireEvent.click(checkpoint)

    await waitFor(() => {
      expect((screen.getByLabelText('Evolution steps') as HTMLInputElement).value).toBe('2500')
    })
  })

  it('queues an accepted plan and remembers the job for refresh recovery', async () => {
    const plan = {
      accepted: true,
      issues: [],
      quality_label: 'Draft',
      pixels_per_inch: 150,
      feature_scale: 1,
      scale_model_status: 'reference-validated',
      development_steps: 5000,
      output_width: 900,
      output_height: 900,
      simulation_width: 450,
      simulation_height: 450,
      simulation_pixels: 202500,
      bicubic_upsample: 2,
      estimated_seconds_low: 32,
      estimated_seconds_high: 56,
      estimated_memory_bytes: 71480000,
      resource_class: 'light',
      engine_version: '2.0.0',
    }
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => plan })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'job-123',
          state: 'queued',
          progress_steps: 0,
          requested_steps: 5000,
          progress_percent: 0,
          queue_position: 1,
          cancel_requested: false,
          error: null,
          artifact_available: false,
          artifact_url: null,
          expires_at: null,
          plan,
        }),
      })
    vi.stubGlobal('fetch', fetchMock)
    render(<RenderStudio recipe={recipe} liveSteps={5000} handoffVersion={1} onImportLiveSettings={vi.fn()} onImportRecipeFile={vi.fn()} />)

    fireEvent.click(screen.getByRole('button', { name: 'Review render plan' }))
    await screen.findByText('900 × 900 px')
    fireEvent.click(screen.getByRole('button', { name: 'Queue high-resolution render' }))

    expect(await screen.findByText('Queued · position 1')).toBeTruthy()
    expect(window.localStorage.getItem('turing-pattern.render-job.v1')).toBe('job-123')
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })
})
