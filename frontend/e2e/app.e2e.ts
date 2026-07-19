import { expect, test } from '@playwright/test'

test('a visitor receives a live frame and can tune a control', async ({ page }) => {
  await page.goto('/')

  await expect(
    page.getByRole('heading', { name: 'Gray-Scott Pattern Lab' }),
  ).toBeVisible()
  await expect(page.getByText('Live simulation running')).toBeVisible({
    timeout: 15_000,
  })
  await expect(page.getByLabel('Live simulation iteration')).not.toHaveText('Step 0', {
    timeout: 15_000,
  })
  await expect(page.getByText('Waiting for the first pattern…')).toHaveCount(0, {
    timeout: 15_000,
  })

  await page.getByText('Advanced chemistry').click()
  const feed = page.getByRole('slider', { name: /Feed.*left/ })
  await feed.fill('0.051')
  await expect(page.getByText('0.0510')).toBeVisible()

  const canvasHasVariation = await page
    .getByRole('img', { name: 'Live grayscale Turing pattern preview' })
    .evaluate((element) => {
      const canvas = element as HTMLCanvasElement
      const pixels = canvas
        .getContext('2d')
        ?.getImageData(0, 0, canvas.width, canvas.height).data
      if (!pixels?.length) return false
      let minimum = 255
      let maximum = 0
      for (let index = 0; index < pixels.length; index += 4) {
        minimum = Math.min(minimum, pixels[index])
        maximum = Math.max(maximum, pixels[index])
      }
      return maximum > minimum
    })
  expect(canvasHasVariation).toBe(true)

  await page.getByRole('button', { name: 'Set current as before' }).click()
  await page.getByRole('button', { name: 'View before' }).click()
  await expect(page.getByRole('img', { name: 'Saved before comparison' })).toBeVisible()
  await page.getByRole('button', { name: 'View current' }).click()
  await expect(
    page.getByRole('img', { name: 'Live grayscale Turing pattern preview' }),
  ).toBeVisible()

  await page.getByRole('button', { name: 'Step once' }).click()
  await expect(page.getByText('Simulation paused')).toBeVisible()

  await expect(page.getByRole('heading', { name: 'Plan a fresh realization' })).toBeVisible()
  await page.getByRole('button', { name: 'Import current Live Lab settings' }).click()
  await page.getByRole('spinbutton', { name: 'Width' }).fill('0.1')
  await page.getByRole('spinbutton', { name: 'Height' }).fill('0.1')
  await page.getByRole('radio', { name: /Draft/ }).check()
  await page.getByRole('spinbutton', { name: 'Evolution steps' }).fill('100')
  await page.getByRole('button', { name: 'Review render plan' }).click()
  await expect(page.getByText('15 × 15 px')).toBeVisible()
  await page.getByRole('button', { name: 'Queue high-resolution render' }).click()
  await expect(page.getByRole('link', { name: 'Download completed PNG' })).toBeVisible({
    timeout: 15_000,
  })
})

test('the preview remains fully visible while phone controls scroll', async ({ page }) => {
  const viewport = { width: 390, height: 844 }
  await page.setViewportSize(viewport)
  await page.goto('/')

  const previewPanel = page.locator('.preview-sticky')
  const previewFrame = page.locator('.preview-frame')
  await expect(previewPanel).toHaveCSS('position', 'sticky')
  await expect(previewPanel.getByRole('button')).toHaveCount(0)

  const pauseButton = page.locator('.simulation-controls').getByRole('button', {
    name: 'Pause',
  })
  await pauseButton.scrollIntoViewIfNeeded()
  await expect(pauseButton).toBeVisible()

  await page.getByText('Advanced chemistry').click()
  const lastControl = page.getByRole('slider', { name: /V diffusion.*bottom/ })
  await lastControl.scrollIntoViewIfNeeded()

  await expect(lastControl).toBeVisible()
  await expect(previewFrame).toBeInViewport()
  const previewBox = await previewFrame.boundingBox()
  expect(previewBox).not.toBeNull()
  expect(previewBox!.y).toBeGreaterThanOrEqual(0)
  expect(previewBox!.y + previewBox!.height).toBeLessThanOrEqual(viewport.height)
})
