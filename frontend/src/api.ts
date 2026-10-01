export type Building = { id: string; site_id: string }

export type Health = {
  status: string
  n_buildings: number
  trained_through: string
  calibration_month: string
  forecast_dates: [string, string]
}

export type Hour = {
  timestamp: string
  horizon: number
  forecast_kwh: number
  lower_kwh: number
  upper_kwh: number
  baseline_lag168_kwh: number | null
  baseline_lag24_kwh: number | null
  actual_kwh: number | null
}

export type Forecast = {
  building_id: string
  date: string
  issue_time: string
  interval_level: number
  hours: Hour[]
}

export type BuildingMetrics = {
  building_id: string
  evaluation: string
  n_hours: number
  mae_model: number | null
  mae_lag168: number | null
  mae_lag24: number | null
  relative_mae_vs_lag168: number | null
  coverage_90: number | null
  mean_width: number | null
}

export type Summary = {
  evaluation: string
  n_buildings: number
  median_relative_mae_vs_lag168: number
  pct_buildings_beating_lag168: number
  pooled_coverage_90: number
  median_mean_width_kwh: number
}

export async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`/api${path}`, { signal })
  if (!res.ok) {
    let message = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') message = body.detail
    } catch {
      // keep the default message
    }
    throw new Error(message)
  }
  return res.json() as Promise<T>
}