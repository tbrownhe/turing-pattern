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
