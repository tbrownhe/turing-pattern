import { describe, expect, it } from 'vitest'
import { recipeForPreset, type PatternRecipe } from './recipe'
import { extractTuringParams, importPngRecipe } from './pngRecipe'

const signature = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])
const encoder = new TextEncoder()

function concat(...parts: Uint8Array[]): Uint8Array {
  const output = new Uint8Array(parts.reduce((total, part) => total + part.length, 0))
  let offset = 0
  for (const part of parts) {
    output.set(part, offset)
    offset += part.length
  }
  return output
}

function chunk(type: string, data = new Uint8Array()): Uint8Array {
  const output = new Uint8Array(12 + data.length)
  new DataView(output.buffer).setUint32(0, data.length)
  output.set(encoder.encode(type), 4)
  output.set(data, 8)
  return output
}

function textData(keyword: string, text: string): Uint8Array {
  return concat(encoder.encode(keyword), new Uint8Array([0]), encoder.encode(text))
}

function internationalTextData(keyword: string, text: string): Uint8Array {
  return concat(
    encoder.encode(keyword),
    new Uint8Array([0, 0, 0, 0, 0]),
    encoder.encode(text),
  )
}

function png(...metadataChunks: Uint8Array[]): ArrayBuffer {
  return concat(
    signature,
    chunk('IHDR', new Uint8Array(13)),
    ...metadataChunks,
    chunk('IEND'),
  ).buffer as ArrayBuffer
}

describe('embedded PNG recipes', () => {
  it('imports the legacy high-resolution render envelope', () => {
    const maze = recipeForPreset('maze', 19)
    const metadata = JSON.stringify({
      render_version: 1,
      engine_version: '2.0.0',
      actual_steps: 5000,
      recipe: { controls: maze.controls, seed: maze.seed },
      plan: {
        physical_width: 6,
        physical_height: 6,
        unit: 'in',
        quality: 'studio',
        feature_scale: 1,
        development_steps: 5000,
        framing: 'crop',
        output_width: 1800,
        output_height: 1800,
      },
    })

    const imported = importPngRecipe(
      png(chunk('tEXt', textData('TuringParams', metadata))),
      'old-render.png',
    )

    expect(imported.recipe).toEqual(maze)
    expect(imported.renderSettings).toEqual({
      width: 6,
      height: 6,
      unit: 'in',
      quality: 'studio',
      featureScale: 1,
      developmentSteps: 5000,
      framing: 'crop',
    })
  })

  it('round-trips the complete recipe in a current international text chunk', () => {
    const recipe: PatternRecipe = {
      ...recipeForPreset('order-disorder', 42),
      name: 'Order–disorder sleeve study',
    }
    const metadata = JSON.stringify({
      render_version: 1,
      engine_version: '2.0.0',
      recipe,
    })

    expect(
      importPngRecipe(
        png(chunk('iTXt', internationalTextData('TuringParams', metadata))),
        'new-render.png',
      ).recipe,
    ).toEqual(recipe)
  })

  it('rejects missing, duplicate, malformed, and unsafe metadata', () => {
    expect(() => extractTuringParams(encoder.encode('not a png').buffer as ArrayBuffer))
      .toThrow(/not a PNG/)
    expect(() => importPngRecipe(png(), 'plain.png')).toThrow(/does not contain/)

    const valid = JSON.stringify({
      engine_version: '2.0.0',
      controls: recipeForPreset('coral').controls,
      seed: 7,
    })
    expect(() =>
      importPngRecipe(
        png(
          chunk('tEXt', textData('TuringParams', valid)),
          chunk('tEXt', textData('TuringParams', valid)),
        ),
        'duplicate.png',
      ),
    ).toThrow(/more than one/)

    expect(() =>
      importPngRecipe(
        png(chunk('tEXt', textData('TuringParams', '{broken'))),
        'broken.png',
      ),
    ).toThrow(/not valid JSON/)

    const unsafe = JSON.stringify({
      engine_version: '2.0.0',
      controls: { ...recipeForPreset('coral').controls, F1: 10 },
      seed: 7,
    })
    expect(() =>
      importPngRecipe(
        png(chunk('tEXt', textData('TuringParams', unsafe))),
        'unsafe.png',
      ),
    ).toThrow(/F1 must be finite/)
  })
})
