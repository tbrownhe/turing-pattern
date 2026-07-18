// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { RECIPE_STORAGE_KEY, recipeForPreset, serializeRecipe } from './recipe'


class MockWebSocket {
  static readonly OPEN = 1
  static instances: MockWebSocket[] = []

  readyState = 0
  binaryType = ''
  sent: string[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onerror: (() => void) | null = null
  readonly url: string

  constructor(url: string) {
    this.url = url
    MockWebSocket.instances.push(this)
  }

  send(message: string) {
    this.sent.push(message)
  }

  open() {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.()
  }

  close(code = 1000) {
    this.readyState = 3
    this.onclose?.(new CloseEvent('close', { code }))
  }
}

function sendReady(websocket: MockWebSocket) {
  act(() => {
    websocket.open()
    websocket.onmessage?.(
      new MessageEvent('message', {
        data: JSON.stringify({
          type: 'ready',
          protocol_version: 1,
          engine_version: '2.0.0',
          preview_size: 256,
          frame_rate: 10,
        }),
      }),
    )
  })
}


describe('live controls', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    MockWebSocket.instances = []
    vi.stubGlobal('WebSocket', MockWebSocket)
    window.localStorage.clear()
    window.history.replaceState(null, '', '/')
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('sends the value that is displayed after a slider changes', async () => {
    render(<App />)
    const websocket = MockWebSocket.instances[0]
    act(() => websocket.open())
    fireEvent.click(screen.getByText('Advanced chemistry'))

    fireEvent.change(screen.getByRole('slider', { name: /Feed.*left/ }), {
      target: { value: '0.051' },
    })
    await act(() => vi.advanceTimersByTimeAsync(50))

    const message = JSON.parse(websocket.sent.at(-1) ?? '{}')
    expect(message).toEqual({
      type: 'controls',
      controls: expect.objectContaining({ F1: 0.051 }),
    })
    expect(screen.getByText('0.0510')).toBeTruthy()
  })

  it('closes the socket and discards queued controls on unmount', async () => {
    const { unmount } = render(<App />)
    const websocket = MockWebSocket.instances[0]
    act(() => websocket.open())
    fireEvent.click(screen.getByText('Advanced chemistry'))

    fireEvent.change(screen.getByRole('slider', { name: /Feed.*left/ }), {
      target: { value: '0.052' },
    })
    unmount()
    await act(() => vi.advanceTimersByTimeAsync(100))

    expect(websocket.readyState).toBe(3)
    expect(
      websocket.sent.some((raw) => JSON.parse(raw).type === 'controls'),
    ).toBe(false)
  })

  it('shows capacity rejection without entering a reconnect loop', async () => {
    render(<App />)
    const websocket = MockWebSocket.instances[0]
    act(() => websocket.open())

    act(() => {
      websocket.onmessage?.(
        new MessageEvent('message', {
          data: JSON.stringify({
            type: 'error',
            error: { code: 'server_busy', message: 'Try later.' },
          }),
        }),
      )
      websocket.close(1013)
    })
    await act(() => vi.advanceTimersByTimeAsync(20_000))

    expect(screen.getByText('Try later.')).toBeTruthy()
    expect(MockWebSocket.instances).toHaveLength(1)
  })

  it('restores a saved recipe before starting the socket', () => {
    const saved = recipeForPreset('maze', 12345)
    window.localStorage.setItem(RECIPE_STORAGE_KEY, serializeRecipe(saved))

    render(<App />)
    const websocket = MockWebSocket.instances[0]
    act(() => websocket.open())

    expect(JSON.parse(websocket.sent[0])).toEqual({
      type: 'start',
      protocol_version: 1,
      controls: saved.controls,
      seed: 12345,
    })
    expect((screen.getByLabelText('Name') as HTMLInputElement).value).toBe('Slow maze')
  })

  it('applies a preset as controls followed by a deterministic reset', () => {
    render(<App />)
    const websocket = MockWebSocket.instances[0]
    act(() => websocket.open())

    fireEvent.change(screen.getByLabelText('Pattern family'), {
      target: { value: 'coral' },
    })

    const messages = websocket.sent.map((raw) => JSON.parse(raw))
    expect(messages.at(-2)).toEqual({
      type: 'controls',
      controls: recipeForPreset('coral').controls,
    })
    expect(messages.at(-1)).toEqual({ type: 'reset', seed: 0 })
    expect((screen.getByLabelText('Name') as HTMLInputElement).value).toBe('Branching coral')
  })

  it('applies an exact seed and records it in the share URL', async () => {
    render(<App />)
    const websocket = MockWebSocket.instances[0]
    sendReady(websocket)

    fireEvent.change(screen.getByLabelText('Seed'), {
      target: { value: '424242' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))

    expect(JSON.parse(websocket.sent.at(-1) ?? '{}')).toEqual({
      type: 'reset',
      seed: 424242,
    })
    await act(() => vi.advanceTimersByTimeAsync(100))
    expect(decodeURIComponent(window.location.search)).toContain('424242')
  })

  it('pauses before requesting one numerical step', () => {
    render(<App />)
    const websocket = MockWebSocket.instances[0]
    sendReady(websocket)

    fireEvent.click(screen.getByRole('button', { name: 'Step once' }))

    const messages = websocket.sent.map((raw) => JSON.parse(raw))
    expect(messages.at(-2)).toEqual({ type: 'pause' })
    expect(messages.at(-1)).toEqual({ type: 'step' })
    expect(screen.getByText('Simulation paused')).toBeTruthy()
  })

  it('undoes and redoes complete recipes with deterministic restarts', () => {
    render(<App />)
    const websocket = MockWebSocket.instances[0]
    sendReady(websocket)

    fireEvent.change(screen.getByLabelText('Pattern family'), {
      target: { value: 'coral' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Undo recipe' }))

    let messages = websocket.sent.map((raw) => JSON.parse(raw))
    expect(messages.at(-2)).toEqual({
      type: 'controls',
      controls: recipeForPreset('mixed').controls,
    })
    expect(messages.at(-1)).toEqual({ type: 'reset', seed: 0 })

    fireEvent.click(screen.getByRole('button', { name: 'Redo recipe' }))
    messages = websocket.sent.map((raw) => JSON.parse(raw))
    expect(messages.at(-2)).toEqual({
      type: 'controls',
      controls: recipeForPreset('coral').controls,
    })
    expect(messages.at(-1)).toEqual({ type: 'reset', seed: 0 })
  })

  it('updates both endpoints when editing uniform chemistry', async () => {
    const maze = recipeForPreset('maze')
    window.localStorage.setItem(RECIPE_STORAGE_KEY, serializeRecipe(maze))
    render(<App />)
    const websocket = MockWebSocket.instances[0]
    sendReady(websocket)
    fireEvent.click(screen.getByText('Advanced chemistry'))

    fireEvent.change(screen.getByRole('slider', { name: 'Feed' }), {
      target: { value: '0.045' },
    })
    await act(() => vi.advanceTimersByTimeAsync(50))

    const message = JSON.parse(websocket.sent.at(-1) ?? '{}')
    expect(message.controls).toEqual(
      expect.objectContaining({ F1: 0.045, F2: 0.045 }),
    )
  })

  it('keeps preview zoom and contrast out of the simulation protocol', () => {
    render(<App />)
    const websocket = MockWebSocket.instances[0]
    sendReady(websocket)
    const sentBefore = websocket.sent.length

    fireEvent.change(screen.getByRole('slider', { name: 'Zoom' }), {
      target: { value: '2' },
    })
    fireEvent.change(screen.getByRole('slider', { name: 'Contrast' }), {
      target: { value: '1.8' },
    })

    expect(websocket.sent).toHaveLength(sentBefore)
  })
})
