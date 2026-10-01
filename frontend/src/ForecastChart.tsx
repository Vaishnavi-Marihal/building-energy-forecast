import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { Hour } from './api'

type Row = {
  label: string
  forecast: number
  lower: number
  upper: number
  band: number
  baseline: number | null
  actual: number | null
}

type TipProps = { active?: boolean; payload?: { payload: Row }[] }

const fmt = (v: number | null) => (v === null ? '—' : v.toFixed(1))

function RowTooltip({ active, payload }: TipProps) {
  if (!active || !payload || payload.length === 0) return null
  const r = payload[0].payload
  return (
    <div className="rounded border bg-white p-2 text-sm shadow">
      <div className="font-medium">{r.label}</div>
      <div>Forecast: {fmt(r.forecast)} kWh</div>
      <div>
        90% range: {fmt(r.lower)} – {fmt(r.upper)}
      </div>
      <div>Same hour last week: {fmt(r.baseline)}</div>
      <div>Actual: {fmt(r.actual)}</div>
    </div>
  )
}

export default function ForecastChart({ hours }: { hours: Hour[] }) {
  const data: Row[] = hours.map((h) => ({
    label: h.timestamp.slice(11, 16),
    forecast: h.forecast_kwh,
    lower: h.lower_kwh,
    upper: h.upper_kwh,
    band: h.upper_kwh - h.lower_kwh,
    baseline: h.baseline_lag168_kwh,
    actual: h.actual_kwh,
  }))

  const values = data
    .flatMap((r) => [r.lower, r.upper, r.actual, r.baseline])
    .filter((v): v is number => v !== null)

  const lo = Math.floor(Math.min(...values) * 0.95)
  const hi = Math.ceil(Math.max(...values) * 1.05)

  return (
    <ResponsiveContainer width="100%" height={360}>
      <ComposedChart
        data={data}
        margin={{ top: 8, right: 16, bottom: 8, left: 8 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="label" interval={2} />
        <YAxis
          domain={[lo, hi]}
          allowDataOverflow
          unit=" kWh"
          width={80}
        />
        <Tooltip content={<RowTooltip />} />
        <Legend />
        <Area
          dataKey="lower"
          stackId="band"
          stroke="none"
          fill="none"
          legendType="none"
          isAnimationActive={false}
        />
        <Area
          dataKey="band"
          stackId="band"
          stroke="none"
          fill="#93c5fd"
          fillOpacity={0.4}
          name="90% range"
          isAnimationActive={false}
        />
        <Line
          dataKey="baseline"
          name="Same hour last week"
          stroke="#9ca3af"
          strokeDasharray="5 4"
          dot={false}
          isAnimationActive={false}
        />
        <Line
          dataKey="forecast"
          name="Model forecast"
          stroke="#2563eb"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          dataKey="actual"
          name="Actual"
          stroke="#111827"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}