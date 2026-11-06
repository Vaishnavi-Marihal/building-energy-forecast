import { afterEach, describe, expect, it, vi } from 'vitest'
import { getJson } from './api'

// Replace the browser's fetch with a fake that returns a canned response.
function mockFetch(status: number, body: unknown, bodyIsJson = true) {
  const response = {
    ok: status >= 200 && status < 300,
    status,
    json: async () => {
      if (!bodyIsJson) throw new Error('not json')
      return body
    },
  }
  const fn = vi.fn().mockResolvedValue(response)
  vi.stubGlobal('fetch', fn)
  return fn
}

afterEach(() => vi.unstubAllGlobals())

describe('getJson', () => {
  it('returns the parsed body and asks for /api + the path', async () => {
    const fn = mockFetch(200, { status: 'ok' })
    const data = await getJson<{ status: string }>('/health')
    expect(data).toEqual({ status: 'ok' })
    expect(fn).toHaveBeenCalledWith('/api/health', { signal: undefined })
  })

  it('shows the API detail text when the server sends one', async () => {
    mockFetch(404, { detail: 'Unknown building' })
    await expect(getJson('/buildings/nope/forecast')).rejects.toThrow('Unknown building')
  })

  it('falls back to a default message when the body is not JSON', async () => {
    mockFetch(500, null, false)
    await expect(getJson('/health')).rejects.toThrow('Request failed (500)')
  })

  it('falls back to the default message when detail is not text', async () => {
    // FastAPI sends a list as detail for validation errors (HTTP 422)
    mockFetch(422, { detail: [{ msg: 'bad date' }] })
    await expect(getJson('/health')).rejects.toThrow('Request failed (422)')
  })
})
