import { useEffect, useState } from 'react'
import ForecastChart from './ForecastChart'
import { getJson } from './api'
import type { Building, BuildingMetrics, Forecast, Health, Summary } from './api'

const DEFAULT_DATE = '2017-09-12'

const num = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined ? '—' : v.toFixed(d)
const pct = (v: number | null | undefined) =>
  v === null || v === undefined ? '—' : `${(v * 100).toFixed(1)}%`

function relativeText(r: number | null | undefined) {
  if (r === null || r === undefined) return '—'
  const diff = Math.abs(1 - r) * 100
  return r < 1
    ? `${diff.toFixed(0)}% lower error than last week's value`
    : `${diff.toFixed(0)}% higher error than last week's value`
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [summary, setSummary] = useState<Summary | null>(null)
  const [buildings, setBuildings] = useState<Building[]>([])
  const [buildingId, setBuildingId] = useState('')
  const [date, setDate] = useState('')
  const [forecast, setForecast] = useState<Forecast | null>(null)
  const [metrics, setMetrics] = useState<BuildingMetrics | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [forecastError, setForecastError] = useState<string | null>(null)
  const [metricsError, setMetricsError] = useState<string | null>(null)

  // one-time load: API status, building list, overall results
  useEffect(() => {
    const ctrl = new AbortController()
    Promise.all([
      getJson<Health>('/health', ctrl.signal),
      getJson<Building[]>('/buildings', ctrl.signal),
      getJson<Summary>('/metrics/summary', ctrl.signal),
    ])
      .then(([h, b, s]) => {
        setHealth(h)
        setBuildings(b)
        setSummary(s)
        if (b.length > 0) setBuildingId(b[0].id)
        const [first, last] = h.forecast_dates
        setDate(DEFAULT_DATE >= first && DEFAULT_DATE <= last ? DEFAULT_DATE : first)
      })
      .catch((e) => {
        if (e.name !== 'AbortError') setError(e.message)
      })
    return () => ctrl.abort()
  }, [])

  // forecast for the chosen building and date
  useEffect(() => {
    if (!buildingId || !date) return
    const ctrl = new AbortController()
    setLoading(true)
    setForecastError(null)
    getJson<Forecast>(
      `/buildings/${encodeURIComponent(buildingId)}/forecast?date=${date}`,
      ctrl.signal,
    )
      .then(setForecast)
      .catch((e) => {
        if (e.name !== 'AbortError') {
          setForecast(null)
          setForecastError(e.message)
        }
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false)
      })
    return () => ctrl.abort()
  }, [buildingId, date])

  // scores for the chosen building
  useEffect(() => {
    if (!buildingId) return
    const ctrl = new AbortController()
    setMetrics(null)
    setMetricsError(null)
    getJson<BuildingMetrics>(
      `/buildings/${encodeURIComponent(buildingId)}/metrics`,
      ctrl.signal,
    )
      .then(setMetrics)
      .catch((e) => {
        if (e.name !== 'AbortError') setMetricsError(e.message)
      })
    return () => ctrl.abort()
  }, [buildingId])

  if (error) return <p className="p-6 text-red-600">Error: {error}</p>
  if (!health || !summary) return <p className="p-6 text-gray-500">Loading…</p>

  return (
    <div className="mx-auto max-w-4xl p-6">
      <h1 className="text-2xl font-semibold">Building Energy Day-Ahead Forecast</h1>
      <p className="mt-1 text-sm text-gray-600">
        Each forecast is issued at 23:00 the day before, using only data available then.
        Replay of {health.forecast_dates[0]} to {health.forecast_dates[1]}; models
        trained through {health.trained_through.slice(0, 10)}.
      </p>

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="rounded border p-3">
          <div className="text-2xl font-semibold">
            {num(summary.median_relative_mae_vs_lag168, 2)}×
          </div>
          <div className="text-sm text-gray-600">
            median error vs same-hour-last-week (below 1 is better)
          </div>
        </div>
        <div className="rounded border p-3">
          <div className="text-2xl font-semibold">
            {pct(summary.pct_buildings_beating_lag168)}
          </div>
          <div className="text-sm text-gray-600">
            of {summary.n_buildings} buildings where the model beats it
          </div>
        </div>
        <div className="rounded border p-3">
          <div className="text-2xl font-semibold">{pct(summary.pooled_coverage_90)}</div>
          <div className="text-sm text-gray-600">
            of true values fall inside the 90% range
          </div>
        </div>
      </div>

      <div className="mt-6 flex flex-col gap-3 sm:flex-row">
        <select
          className="w-full rounded border p-2"
          value={buildingId}
          onChange={(e) => setBuildingId(e.target.value)}
        >
          {buildings.map((b) => (
            <option key={b.id} value={b.id}>
              {b.id} ({b.site_id})
            </option>
          ))}
        </select>
        <input
          type="date"
          className="rounded border p-2"
          value={date}
          min={health.forecast_dates[0]}
          max={health.forecast_dates[1]}
          onChange={(e) => setDate(e.target.value)}
        />
      </div>

      <div className="mt-4">
        {!date && <p className="text-gray-500">Pick a date.</p>}
        {loading && <p className="text-gray-500">Loading forecast…</p>}
        {forecastError && <p className="text-red-600">Error: {forecastError}</p>}
        {forecast && !forecastError && <ForecastChart hours={forecast.hours} />}
      </div>

      <h2 className="mt-6 text-lg font-semibold">This building, {metrics ? 'Jul–Dec 2017' : ''}</h2>
      {metricsError && <p className="text-red-600">Error: {metricsError}</p>}
      {metrics && (
        <table className="mt-2 w-full text-left text-sm">
          <tbody>
            <tr className="border-b">
              <td className="py-1">Average error, model</td>
              <td>{num(metrics.mae_model)} kWh</td>
            </tr>
            <tr className="border-b">
              <td className="py-1">Average error, same hour last week</td>
              <td>{num(metrics.mae_lag168)} kWh</td>
            </tr>
            <tr className="border-b">
              <td className="py-1">Average error, same hour yesterday</td>
              <td>{num(metrics.mae_lag24)} kWh</td>
            </tr>
            <tr className="border-b">
              <td className="py-1">Model vs last week</td>
              <td>{relativeText(metrics.relative_mae_vs_lag168)}</td>
            </tr>
            <tr className="border-b">
              <td className="py-1">True values inside the 90% range</td>
              <td>{pct(metrics.coverage_90)}</td>
            </tr>
            <tr className="border-b">
              <td className="py-1">Average width of the range</td>
              <td>{num(metrics.mean_width)} kWh</td>
            </tr>
          </tbody>
        </table>
      )}
      <p className="mt-3 text-xs text-gray-500">{summary.evaluation}</p>
    </div>
  )
}