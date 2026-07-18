import { describe, expect, it } from 'vitest'
import {
  CURRENT_ENGINE_VERSION,
  RECIPE_QUERY_KEY,
  RECIPE_STORAGE_KEY,
  loadInitialRecipe,
  parseRecipeJson,
  recipeForPreset,
  recipeFromSearch,
  recipeUrl,
  serializeRecipe,
} from './recipe'

describe('pattern recipes', () => {
  it('round-trips every reproducibility field through JSON', () => {
    const recipe = recipeForPreset('coral', 8675309)

    expect(parseRecipeJson(serializeRecipe(recipe))).toEqual(recipe)
    expect(recipe.engine_version).toBe(CURRENT_ENGINE_VERSION)
  })

  it('keeps the slow maze preset on its stable uniform recipe', () => {
    expect(recipeForPreset('maze').controls).toEqual({
      F1: 0.04,
      F2: 0.04,
      K1: 0.062,
      K2: 0.062,
      Du1: 0.72,
      Du2: 0.72,
      Dv1: 0.25,
      Dv2: 0.25,
    })
  })

  it('preserves an old preset recipe but labels it custom after definitions change', () => {
    const staleMaze = {
      ...recipeForPreset('maze'),
      controls: { ...recipeForPreset('maze').controls, F1: 0.028 },
    }

    const restored = parseRecipeJson(JSON.stringify(staleMaze))

    expect(restored.preset).toBe('custom')
    expect(restored.controls.F1).toBe(0.028)
  })

  it('round-trips a recipe through a share URL', () => {
    const recipe = recipeForPreset('worms', 42)
    const url = new URL(recipeUrl(recipe, 'https://turing.example/lab?keep=this'))

    expect(url.searchParams.get('keep')).toBe('this')
    expect(recipeFromSearch(url.search)).toEqual(recipe)
    expect(url.searchParams.has(RECIPE_QUERY_KEY)).toBe(true)
  })

  it('prefers a shared recipe over a locally saved recipe', () => {
    const shared = recipeForPreset('spots', 10)
    const stored = recipeForPreset('maze', 20)
    const search = new URL(recipeUrl(shared, 'https://turing.example/')).search
    const storage = {
      getItem: (key: string) =>
        key === RECIPE_STORAGE_KEY ? serializeRecipe(stored) : null,
    }

    const loaded = loadInitialRecipe(search, storage)

    expect(loaded.source).toBe('url')
    expect(loaded.recipe).toEqual(shared)
  })

  it('rejects unknown fields, invalid seeds, and unsafe controls', () => {
    const recipe = recipeForPreset('mixed')

    expect(() =>
      parseRecipeJson(JSON.stringify({ ...recipe, surprise: true })),
    ).toThrow(/missing or unsupported fields/)
    expect(() =>
      parseRecipeJson(JSON.stringify({ ...recipe, seed: -1 })),
    ).toThrow(/unsigned 32-bit integer/)
    expect(() =>
      parseRecipeJson(
        JSON.stringify({
          ...recipe,
          controls: { ...recipe.controls, F1: 100 },
        }),
      ),
    ).toThrow(/F1 must be finite/)
  })

  it('falls back safely when a shared recipe is malformed', () => {
    const loaded = loadInitialRecipe(`?${RECIPE_QUERY_KEY}=not-json`)

    expect(loaded.source).toBe('default')
    expect(loaded.warning).toMatch(/Shared recipe ignored/)
  })
})
