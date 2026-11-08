// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import App from './App'

const health = {
  status: 'ok',
  n_buildings: 2,
  trained_through: '2017-05-31 23:00',
  calibration_month: '2017-06',
  forecast_dates: ['2017-07-01', '2017-12-31'],
}
const buildings = [
  { id: 'Bear_a', site_id: 'Bear' },
  { id: 'Fox_b', site_id: 'Fox' },
]
const summary = {
  evaluation: 'fixed origin',
  n_buildings: 2,
  median_relative_mae_vs_lag168: 0.67,
  pct_buildings_beating_lag168: 0.929,
  pooled_coverage_90: 0.9,
  median_mean_width_kwh: 40,
}

type Answer = { status: number; body: unknown }

// Replace the browser's fetch with a fake that answers by address, so no server is needed.
function fakeApi(answer: (path: string) => Answer) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      const { status, body } = answer(url.replace('/api', ''))
      return { ok: status >= 200 && status < 300, status, json: async () => body }
    }),
  )
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('App', () => {
  it('shows Loading… until the first answers arrive', () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {}))) // never answers
    render(<App />)
    expect(screen.getByText('Loading…')).toBeTruthy()
  })

  it('shows the error text from the API when the start-up calls fail', async () => {
    fakeApi(() => ({ status: 503, body: { detail: 'Database unavailable' } }))
    render(<App />)
    expect(await screen.findByText('Error: Database unavailable')).toBeTruthy()
    expect(screen.queryByText('Loading…')).toBeNull()
  })

  it('keeps the headline numbers when only the forecast and the scores fail', async () => {
    fakeApi((path) => {
      if (path.includes('/forecast')) {
        return { status: 404, body: { detail: 'No forecast for that date' } }
      }
      if (path === '/buildings/Bear_a/metrics') {
        return { status: 404, body: { detail: 'No scores for this building' } }
      }
      if (path === '/metrics/summary') return { status: 200, body: summary }
      if (path === '/buildings') return { status: 200, body: buildings }
      if (path === '/health') return { status: 200, body: health }
      return { status: 500, body: {} }
    })
    render(<App />)
    expect(await screen.findByText('Error: No forecast for that date')).toBeTruthy()
    expect(await screen.findByText('Error: No scores for this building')).toBeTruthy()
    // the page is still there: headline numbers and the building list
    expect(screen.getByText('0.67×')).toBeTruthy()
    expect(screen.getByText('92.9%')).toBeTruthy()
    expect(screen.getByText('Bear_a (Bear)')).toBeTruthy()
  })
})
