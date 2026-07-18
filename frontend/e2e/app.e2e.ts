import { expect, test } from '@playwright/test'

test('a visitor receives a live frame and can tune a control', async ({ page }) => {
  await page.goto('/')

  await expect(
    page.getByRole('heading', { name: 'Gray-Scott Pattern Lab' }),
  ).toBeVisible()
  await expect(page.getByText('Live simulation running')).toBeVisible({
    timeout: 15_000,
  })
  await expect(page.getByText('Waiting for the first pattern…')).toHaveCount(0, {
    timeout: 15_000,
  })

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
})
