import {
  CURRENT_ENGINE_VERSION,
  PRESETS,
  RECIPE_VERSION,
  parseRecipe,
  type PatternRecipe,
  type RecipePreset,
} from './recipe'
import { CONTROL_KEYS, type Controls } from './protocol'

export const MAX_PNG_RECIPE_BYTES = 64 * 1024 * 1024
const MAX_METADATA_BYTES = 64 * 1024
const PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10] as const
const METADATA_KEY = 'TuringParams'

export interface ImportedRenderSettings {
  width: number
  height: number
  unit: 'in' | 'cm'
  quality: 'draft' | 'studio' | 'fine'
  featureScale: 0.5 | 1 | 2
  developmentSteps: number
  framing: 'crop' | 'fit' | 'extend'
}

export interface PngRecipeImport {
  recipe: PatternRecipe
  renderSettings: ImportedRenderSettings | null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function chunkType(bytes: Uint8Array, offset: number): string {
  return String.fromCharCode(
    bytes[offset],
    bytes[offset + 1],
    bytes[offset + 2],
    bytes[offset + 3],
  )
}

function findNull(bytes: Uint8Array, start: number, end: number): number {
  for (let index = start; index < end; index += 1) {
    if (bytes[index] === 0) return index
  }
  return -1
}

function decodeText(bytes: Uint8Array, encoding: 'latin1' | 'utf-8'): string {
  try {
    return new TextDecoder(encoding, { fatal: true }).decode(bytes)
  } catch {
    throw new Error('PNG recipe metadata contains invalid text.')
  }
}

function readTextChunk(
  bytes: Uint8Array,
  start: number,
  end: number,
): string | null {
  const separator = findNull(bytes, start, end)
  if (separator < 0) throw new Error('PNG contains a malformed text chunk.')
  const keyword = decodeText(bytes.subarray(start, separator), 'latin1')
  if (keyword !== METADATA_KEY) return null
  if (end - separator - 1 > MAX_METADATA_BYTES) {
    throw new Error('PNG recipe metadata is too large.')
  }
  return decodeText(bytes.subarray(separator + 1, end), 'latin1')
}

function readInternationalTextChunk(
  bytes: Uint8Array,
  start: number,
  end: number,
): string | null {
  const keywordEnd = findNull(bytes, start, end)
  if (keywordEnd < 0 || keywordEnd + 2 >= end) {
    throw new Error('PNG contains a malformed international text chunk.')
  }
  const keyword = decodeText(bytes.subarray(start, keywordEnd), 'latin1')
  if (keyword !== METADATA_KEY) return null

  const compressionFlag = bytes[keywordEnd + 1]
  const compressionMethod = bytes[keywordEnd + 2]
  if (compressionFlag !== 0 || compressionMethod !== 0) {
    throw new Error('Compressed PNG recipe metadata is not supported.')
  }

  const languageEnd = findNull(bytes, keywordEnd + 3, end)
  if (languageEnd < 0) {
    throw new Error('PNG contains a malformed international text chunk.')
  }
  const translatedKeywordEnd = findNull(bytes, languageEnd + 1, end)
  if (translatedKeywordEnd < 0) {
    throw new Error('PNG contains a malformed international text chunk.')
  }
  if (end - translatedKeywordEnd - 1 > MAX_METADATA_BYTES) {
    throw new Error('PNG recipe metadata is too large.')
  }
  return decodeText(bytes.subarray(translatedKeywordEnd + 1, end), 'utf-8')
}

export function extractTuringParams(png: ArrayBuffer): string {
  if (png.byteLength > MAX_PNG_RECIPE_BYTES) {
    throw new Error('PNG is too large to import (64 MB maximum).')
  }
  const bytes = new Uint8Array(png)
  if (
    bytes.length < PNG_SIGNATURE.length ||
    PNG_SIGNATURE.some((value, index) => bytes[index] !== value)
  ) {
    throw new Error('Selected file is not a PNG image.')
  }

  const view = new DataView(png)
  let offset: number = PNG_SIGNATURE.length
  let firstChunk = true
  let metadata: string | null = null
  let foundEnd = false

  while (offset + 12 <= bytes.length) {
    const length = view.getUint32(offset)
    const typeOffset = offset + 4
    const dataStart = offset + 8
    const dataEnd = dataStart + length
    const nextOffset = dataEnd + 4
    if (dataEnd < dataStart || nextOffset > bytes.length) {
      throw new Error('PNG contains a truncated chunk.')
    }

    const type = chunkType(bytes, typeOffset)
    if (firstChunk && type !== 'IHDR') {
      throw new Error('PNG is missing its initial image header.')
    }
    firstChunk = false

    let candidate: string | null = null
    if (type === 'tEXt') candidate = readTextChunk(bytes, dataStart, dataEnd)
    if (type === 'iTXt') {
      candidate = readInternationalTextChunk(bytes, dataStart, dataEnd)
    }
    if (candidate !== null) {
      if (metadata !== null) {
        throw new Error('PNG contains more than one embedded recipe.')
      }
      metadata = candidate
    }

    offset = nextOffset
    if (type === 'IEND') {
      foundEnd = true
      break
    }
  }

  if (!foundEnd) throw new Error('PNG is incomplete or missing its end marker.')
  if (metadata === null) throw new Error('PNG does not contain a TuringParams recipe.')
  return metadata
}

function matchingPreset(controls: Controls): RecipePreset {
  return (
    PRESETS.find((preset) =>
      CONTROL_KEYS.every((key) => preset.controls[key] === controls[key]),
    )?.id ?? 'custom'
  )
}

function fallbackRecipeName(filename: string): string {
  const cleaned = filename
    .replace(/\.png$/i, '')
    .replace(/[-_]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
  return (cleaned || 'Imported PNG recipe').slice(0, 80)
}

function legacyRecipe(
  value: Record<string, unknown>,
  filename: string,
): PatternRecipe {
  const nested = isRecord(value.recipe) ? value.recipe : value
  const controls = nested.controls
  const seed = nested.seed
  const engineVersion =
    typeof value.engine_version === 'string'
      ? value.engine_version
      : CURRENT_ENGINE_VERSION

  const preliminary = parseRecipe({
    recipe_version: RECIPE_VERSION,
    engine_version: engineVersion,
    name: fallbackRecipeName(filename),
    preset: 'custom',
    seed,
    controls,
  })
  const preset = matchingPreset(preliminary.controls)
  if (preset === 'custom') return preliminary
  const namedPreset = PRESETS.find((candidate) => candidate.id === preset)
  return parseRecipe({
    ...preliminary,
    name: namedPreset?.name ?? preliminary.name,
    preset,
  })
}

function renderSettings(value: Record<string, unknown>): ImportedRenderSettings | null {
  if (value.plan === undefined) return null
  if (!isRecord(value.plan)) throw new Error('PNG render settings must be an object.')
  const plan = value.plan
  const width = plan.physical_width
  const height = plan.physical_height
  const unit = plan.unit
  const quality = plan.quality
  const featureScale = plan.feature_scale
  const developmentSteps = plan.development_steps
  const framing = plan.framing

  if (typeof width !== 'number' || !Number.isFinite(width) || width <= 0 || width > 100) {
    throw new Error('PNG render width is invalid.')
  }
  if (
    typeof height !== 'number' ||
    !Number.isFinite(height) ||
    height <= 0 ||
    height > 100
  ) {
    throw new Error('PNG render height is invalid.')
  }
  if (unit !== 'in' && unit !== 'cm') throw new Error('PNG render unit is invalid.')
  if (quality !== 'draft' && quality !== 'studio' && quality !== 'fine') {
    throw new Error('PNG render quality is invalid.')
  }
  if (featureScale !== 0.5 && featureScale !== 1 && featureScale !== 2) {
    throw new Error('PNG render feature scale is invalid.')
  }
  if (
    typeof developmentSteps !== 'number' ||
    !Number.isInteger(developmentSteps) ||
    developmentSteps < 100 ||
    developmentSteps > 20_000
  ) {
    throw new Error('PNG render development steps are invalid.')
  }
  if (framing !== 'crop' && framing !== 'fit' && framing !== 'extend') {
    throw new Error('PNG render framing is invalid.')
  }
  if (value.actual_steps !== undefined && value.actual_steps !== developmentSteps) {
    throw new Error('PNG render step metadata is inconsistent.')
  }

  return {
    width,
    height,
    unit,
    quality,
    featureScale,
    developmentSteps,
    framing,
  }
}

export function importPngRecipe(png: ArrayBuffer, filename: string): PngRecipeImport {
  let value: unknown
  try {
    value = JSON.parse(extractTuringParams(png))
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new Error('PNG recipe metadata is not valid JSON.')
    }
    throw error
  }
  if (!isRecord(value)) throw new Error('PNG recipe metadata must be an object.')
  let recipe: PatternRecipe
  if (isRecord(value.recipe) && 'recipe_version' in value.recipe) {
    recipe = parseRecipe(value.recipe)
  } else {
    recipe = legacyRecipe(value, filename)
  }
  return { recipe, renderSettings: renderSettings(value) }
}
