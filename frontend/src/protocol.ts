export type ControlKey =
  | 'F1'
  | 'F2'
  | 'K1'
  | 'K2'
  | 'Du1'
  | 'Du2'
  | 'Dv1'
  | 'Dv2'

export type Controls = Record<ControlKey, number>

export const CONTROL_KEYS: readonly ControlKey[] = [
  'F1',
  'F2',
  'K1',
  'K2',
  'Du1',
  'Du2',
  'Dv1',
  'Dv2',
]

export const CONTROL_LIMITS: Record<
  ControlKey,
  { min: number; max: number }
> = {
  F1: { min: 0, max: 0.1 },
  F2: { min: 0, max: 0.1 },
  K1: { min: 0, max: 0.1 },
  K2: { min: 0, max: 0.1 },
  Du1: { min: 0, max: 1 },
  Du2: { min: 0, max: 1 },
  Dv1: { min: 0, max: 1 },
  Dv2: { min: 0, max: 1 },
}

export const DEFAULT_CONTROLS: Controls = {
  F1: 0.04,
  F2: 0.08,
  K1: 0.056,
  K2: 0.074,
  Du1: 0.7,
  Du2: 0.7,
  Dv1: 0.25,
  Dv2: 0.25,
}

export function updateControl(
  controls: Controls,
  key: ControlKey,
  value: number,
): Controls {
  return { ...controls, [key]: value }
}

export function controlsMessage(controls: Controls) {
  return { type: 'controls' as const, controls }
}
