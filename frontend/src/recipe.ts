import {
  CONTROL_KEYS,
  CONTROL_LIMITS,
  DEFAULT_CONTROLS,
  type Controls,
} from './protocol'

export const RECIPE_VERSION = 1 as const
export const CURRENT_ENGINE_VERSION = '2.0.0'
export const RECIPE_STORAGE_KEY = 'turing-pattern.recipe.v1'
export const RECIPE_QUERY_KEY = 'recipe'

export type PresetId =
  | 'mixed'
  | 'spots'
  | 'worms'
  | 'coral'
  | 'maze'
  | 'order-disorder'
export type RecipePreset = PresetId | 'custom'

export interface PatternPreset {
  id: PresetId
  name: string
  description: string
  controls: Controls
}

export interface PatternRecipe {
  recipe_version: typeof RECIPE_VERSION
  engine_version: string
  name: string
  preset: RecipePreset
  seed: number
  controls: Controls
}

export const PRESETS: readonly PatternPreset[] = [
  {
    id: 'mixed',
    name: 'Mixed terrain',
    description: 'A changing field that moves between spots, ribbons, and open cells.',
    controls: DEFAULT_CONTROLS,
  },
  {
    id: 'spots',
    name: 'Spotted stone',
    description: 'Dense islands with gentle variation from the left edge to the right.',
    controls: {
      F1: 0.032,
      F2: 0.038,
      K1: 0.061,
      K2: 0.064,
      Du1: 0.7,
      Du2: 0.7,
      Dv1: 0.25,
      Dv2: 0.25,
    },
  },
  {
    id: 'worms',
    name: 'Wandering worms',
    description: 'Long organic strands that curl and divide as the pattern develops.',
    controls: {
      F1: 0.046,
      F2: 0.054,
      K1: 0.059,
      K2: 0.063,
      Du1: 0.72,
      Du2: 0.68,
      Dv1: 0.25,
      Dv2: 0.25,
    },
  },
  {
    id: 'coral',
    name: 'Branching coral',
    description: 'Fine branching growth with a subtle top-to-bottom diffusion shift.',
    controls: {
      F1: 0.054,
      F2: 0.058,
      K1: 0.061,
      K2: 0.064,
      Du1: 0.74,
      Du2: 0.66,
      Dv1: 0.27,
      Dv2: 0.23,
    },
  },
  {
    id: 'maze',
    name: 'Slow maze',
    description: 'Broad connected corridors with a calm, stone-carved character.',
    controls: {
      F1: 0.04,
      F2: 0.04,
      K1: 0.062,
      K2: 0.062,
      Du1: 0.72,
      Du2: 0.72,
      Dv1: 0.25,
      Dv2: 0.25,
    },
  },
  {
    id: 'order-disorder',
    name: 'Order–disorder transition',
    description: 'The opposing reaction gradients featured in the project image shift from ordered structure into disorder.',
    controls: {
      F1: 0.067,
      F2: 0.043,
      K1: 0.059,
      K2: 0.076,
      Du1: 0.72,
      Du2: 0.72,
      Dv1: 0.25,
      Dv2: 0.25,
    },
  },
]

const presetIds = new Set<RecipePreset>([
  ...PRESETS.map((preset) => preset.id),
  'custom',
])

function cloneControls(controls: Controls): Controls {
  return { ...controls }
}

export function recipeForPreset(
  presetId: PresetId,
  seed = 0,
  engineVersion = CURRENT_ENGINE_VERSION,
): PatternRecipe {
  const preset = PRESETS.find((candidate) => candidate.id === presetId)
  if (!preset) throw new Error(`Unknown preset: ${presetId}`)
  return {
    recipe_version: RECIPE_VERSION,
    engine_version: engineVersion,
    name: preset.name,
    preset: preset.id,
    seed,
    controls: cloneControls(preset.controls),
  }
}

export const DEFAULT_RECIPE = recipeForPreset('mixed')

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function parseControls(value: unknown): Controls {
  if (!isRecord(value)) throw new Error('Recipe controls must be an object.')
  const keys = Object.keys(value)
  if (
    keys.length !== CONTROL_KEYS.length ||
    keys.some((key) => !CONTROL_KEYS.includes(key as (typeof CONTROL_KEYS)[number]))
  ) {
    throw new Error('Recipe controls must contain exactly the supported controls.')
  }

  return Object.fromEntries(
    CONTROL_KEYS.map((key) => {
      const control = value[key]
      const { min, max } = CONTROL_LIMITS[key]
      if (
        typeof control !== 'number' ||
        !Number.isFinite(control) ||
        control < min ||
        control > max
      ) {
        throw new Error(`${key} must be finite and between ${min} and ${max}.`)
      }
      return [key, control]
    }),
  ) as Controls
}

export function parseRecipe(value: unknown): PatternRecipe {
  if (!isRecord(value)) throw new Error('Recipe must be a JSON object.')
  const expectedKeys = [
    'recipe_version',
    'engine_version',
    'name',
    'preset',
    'seed',
    'controls',
  ]
  const keys = Object.keys(value)
  if (
    keys.length !== expectedKeys.length ||
    keys.some((key) => !expectedKeys.includes(key))
  ) {
    throw new Error('Recipe contains missing or unsupported fields.')
  }
  if (value.recipe_version !== RECIPE_VERSION) {
    throw new Error(`Only recipe version ${RECIPE_VERSION} is supported.`)
  }
  if (
    typeof value.engine_version !== 'string' ||
    value.engine_version.length < 1 ||
    value.engine_version.length > 32
  ) {
    throw new Error('Recipe engine_version must be a short, non-empty string.')
  }
  if (
    typeof value.name !== 'string' ||
    value.name.trim().length < 1 ||
    value.name.trim().length > 80
  ) {
    throw new Error('Recipe name must contain between 1 and 80 characters.')
  }
  if (typeof value.preset !== 'string' || !presetIds.has(value.preset as RecipePreset)) {
    throw new Error('Recipe preset is not recognized.')
  }
  if (
    typeof value.seed !== 'number' ||
    !Number.isInteger(value.seed) ||
    value.seed < 0 ||
    value.seed > 4_294_967_295
  ) {
    throw new Error('Recipe seed must be an unsigned 32-bit integer.')
  }

  const controls = parseControls(value.controls)
  let preset = value.preset as RecipePreset
  if (preset !== 'custom') {
    const currentPreset = PRESETS.find((candidate) => candidate.id === preset)
    if (
      currentPreset &&
      CONTROL_KEYS.some((key) => controls[key] !== currentPreset.controls[key])
    ) {
      preset = 'custom'
    }
  }

  return {
    recipe_version: RECIPE_VERSION,
    engine_version: value.engine_version,
    name: value.name.trim(),
    preset,
    seed: value.seed,
    controls,
  }
}

export function parseRecipeJson(raw: string): PatternRecipe {
  let value: unknown
  try {
    value = JSON.parse(raw)
  } catch {
    throw new Error('Recipe is not valid JSON.')
  }
  return parseRecipe(value)
}

export function serializeRecipe(recipe: PatternRecipe): string {
  return JSON.stringify(parseRecipe(recipe))
}

export function recipeFromSearch(search: string): PatternRecipe | null {
  const raw = new URLSearchParams(search).get(RECIPE_QUERY_KEY)
  return raw === null ? null : parseRecipeJson(raw)
}

export function recipeUrl(recipe: PatternRecipe, currentUrl: string): string {
  const url = new URL(currentUrl)
  url.searchParams.set(RECIPE_QUERY_KEY, serializeRecipe(recipe))
  return url.toString()
}

interface RecipeStorage {
  getItem(key: string): string | null
}

export interface LoadedRecipe {
  recipe: PatternRecipe
  source: 'url' | 'storage' | 'default'
  warning: string
}

export function loadInitialRecipe(
  search: string,
  storage?: RecipeStorage,
): LoadedRecipe {
  try {
    const recipe = recipeFromSearch(search)
    if (recipe) return { recipe, source: 'url', warning: '' }
  } catch (error) {
    return {
      recipe: { ...DEFAULT_RECIPE, controls: cloneControls(DEFAULT_RECIPE.controls) },
      source: 'default',
      warning: error instanceof Error ? `Shared recipe ignored: ${error.message}` : 'Shared recipe ignored.',
    }
  }

  const stored = storage?.getItem(RECIPE_STORAGE_KEY)
  if (stored !== null && stored !== undefined) {
    try {
      return { recipe: parseRecipeJson(stored), source: 'storage', warning: '' }
    } catch {
      return {
        recipe: { ...DEFAULT_RECIPE, controls: cloneControls(DEFAULT_RECIPE.controls) },
        source: 'default',
        warning: 'The saved recipe was invalid and has been reset.',
      }
    }
  }

  return {
    recipe: { ...DEFAULT_RECIPE, controls: cloneControls(DEFAULT_RECIPE.controls) },
    source: 'default',
    warning: '',
  }
}

export function recipeFilename(recipe: PatternRecipe): string {
  const slug = recipe.name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 48)
  return `${slug || 'turing-pattern'}-seed-${recipe.seed}.json`
}
