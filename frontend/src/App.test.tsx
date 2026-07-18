// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'


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


describe('live controls', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    MockWebSocket.instances = []
    vi.stubGlobal('WebSocket', MockWebSocket)
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
})
