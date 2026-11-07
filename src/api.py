import datetime as dt
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from features import build_features, FEATURES
from logging_setup import get_logger
from uncertainty import make_interval

ART = Path("artifacts")
# Local development default only. Anywhere else, set the DATABASE_URL environment variable.
DEFAULT_DATABASE_URL = "postgresql://postgres:devpass@localhost:5432/energy"
FIRST_DATE = dt.date(2017, 7, 1)     # first date after the training/calibration data
LAST_DATE = dt.date(2017, 12, 31)
WINDOW_DAYS = 7                      # history needed before the target day (same hour last week)

state = {}
log = get_logger()


def load_state(art=ART, database_url=None):
    meta = json.loads((Path(art) / "meta.json").read_text())
    metrics = json.loads((Path(art) / "metrics.json").read_text())
    url = database_url or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    pool = ConnectionPool(url, min_size=1, max_size=4, open=False)
    pool.open(wait=True, timeout=10)
    state.update(
        meta=meta, metrics=metrics, pool=pool,
        models={k: lgb.Booster(model_file=str(Path(art) / f"{k}.txt"))
                for k in ("point", "lo", "hi")},
    )


def close_state():
    pool = state.get("pool")
    if pool is not None:
        pool.close()
    state.clear()


@asynccontextmanager
async def lifespan(app):
    if not state:
        load_state()
    try:
        yield
    finally:
        close_state()


app = FastAPI(title="Building Energy Day-Ahead Forecast", version="0.2.0",
              lifespan=lifespan)


@app.middleware("http")
async def log_requests(request, call_next):
    """Write one JSON log line per request: method, path, status, milliseconds taken."""
    start = time.perf_counter()
    fields = {"method": request.method, "path": request.url.path}
    try:
        response = await call_next(request)
    except Exception:
        fields["ms"] = round((time.perf_counter() - start) * 1000, 1)
        log.exception("request_failed", extra={"fields": fields})
        raise
    fields["status"] = response.status_code
    fields["ms"] = round((time.perf_counter() - start) * 1000, 1)
    log.info("request", extra={"fields": fields})
    return response


class HourForecast(BaseModel):
    timestamp: dt.datetime
    horizon: int
    forecast_kwh: float
    lower_kwh: float
    upper_kwh: float
    baseline_lag168_kwh: Optional[float]
    baseline_lag24_kwh: Optional[float]
    actual_kwh: Optional[float]


class ForecastResponse(BaseModel):
    building_id: str
    date: dt.date
    issue_time: dt.datetime
    interval_level: float
    hours: List[HourForecast]


class BuildingMetrics(BaseModel):
    building_id: str
    evaluation: str
    n_hours: int
    mae_model: Optional[float]
    mae_lag168: Optional[float]
    mae_lag24: Optional[float]
    relative_mae_vs_lag168: Optional[float]
    coverage_90: Optional[float]
    mean_width: Optional[float]


def _opt(x):
    return None if pd.isna(x) else float(x)


def scale_for(conn, building_id):
    """The building's scale, or None if the building is not in the database."""
    row = conn.execute("SELECT scale FROM buildings WHERE building_id = %s",
                       (building_id,)).fetchone()
    return None if row is None else row[0]


def features_for_day(conn, building_id, scale, day):
    """Features for the 24 target hours of `day`, built from an 8-day window
    (7 days of history plus the target day) read from the database."""
    first = pd.Timestamp(day)
    last = first + pd.Timedelta(hours=23)
    start = first - pd.Timedelta(days=WINDOW_DAYS)
    rows = conn.execute(
        """SELECT ts, kwh FROM readings
           WHERE building_id = %s AND ts >= %s AND ts <= %s
           ORDER BY ts""",
        (building_id, start.to_pydatetime(), last.to_pydatetime())).fetchall()
    if len(rows) != (WINDOW_DAYS + 1) * 24:
        raise ValueError(f"incomplete data window: {len(rows)} rows")
    window = pd.DataFrame({building_id: pd.Series([r[1] for r in rows], dtype="float64")})
    window.index = pd.DatetimeIndex([r[0] for r in rows], name="timestamp")
    feat = build_features(window, pd.Series({building_id: scale}))
    return feat.drop(columns="building").loc[first: last]


@app.get("/health")
def health():
    m = state["meta"]
    return {"status": "ok", "n_buildings": len(m["buildings"]),
            "trained_through": m["train_end"],
            "calibration_month": m["calibration_month"],
            "forecast_dates": [str(FIRST_DATE), str(LAST_DATE)]}


@app.get("/buildings")
def buildings():
    with state["pool"].connection() as conn:
        rows = conn.execute("""SELECT building_id, site_id FROM buildings
                               ORDER BY site_id, building_id""").fetchall()
    return [{"id": b, "site_id": s} for b, s in rows]


def record_forecast(building_id, hours):
    """Remember what was served: one row per hour. The same building and hour is overwritten."""
    rows = [(building_id, h.timestamp, h.forecast_kwh, h.lower_kwh, h.upper_kwh) for h in hours]
    with state["pool"].connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO forecasts (building_id, ts, forecast_kwh, lower_kwh, upper_kwh)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (building_id, ts) DO UPDATE SET
                       forecast_kwh = EXCLUDED.forecast_kwh, lower_kwh = EXCLUDED.lower_kwh,
                       upper_kwh = EXCLUDED.upper_kwh, served_at = now()""",
                rows)


@app.get("/buildings/{building_id}/forecast", response_model=ForecastResponse)
def forecast(building_id: str, date: dt.date = Query(..., description="YYYY-MM-DD")):
    with state["pool"].connection() as conn:
        sc = scale_for(conn, building_id)
        if sc is None:
            raise HTTPException(status_code=404, detail=f"unknown building: {building_id}")
        if not (FIRST_DATE <= date <= LAST_DATE):
            raise HTTPException(status_code=422,
                                detail=f"date must be between {FIRST_DATE} and {LAST_DATE}")
        try:
            day = features_for_day(conn, building_id, sc, date)
        except ValueError:
            raise HTTPException(status_code=500, detail="incomplete data window")
    if len(day) != 24:
        raise HTTPException(status_code=500, detail="incomplete day in data")

    X = day[FEATURES]
    m = state["models"]
    pred = np.maximum(m["point"].predict(X), 0.0)
    lower, upper = make_interval(pred, m["lo"].predict(X), m["hi"].predict(X),
                                 state["meta"]["Q"])
    sc = float(sc)

    hours = [HourForecast(
        timestamp=ts.to_pydatetime(), horizon=int(hz),
        forecast_kwh=float(p * sc), lower_kwh=float(lo * sc), upper_kwh=float(up * sc),
        baseline_lag168_kwh=_opt(b168 * sc), baseline_lag24_kwh=_opt(b24 * sc),
        actual_kwh=_opt(y * sc),
    ) for ts, hz, p, lo, up, b168, b24, y in zip(
        day.index, day["horizon"], pred, lower, upper,
        day["lag168"], day["lag24"], day["y"])]

    try:
        record_forecast(building_id, hours)
    except Exception:  # keeping a record must never stop the forecast from being served
        log.exception("forecast_log_failed", extra={"fields": {"building": building_id}})

    start = pd.Timestamp(date)
    return ForecastResponse(
        building_id=building_id, date=date,
        issue_time=(start - pd.Timedelta(hours=1)).to_pydatetime(),
        interval_level=1 - state["meta"]["alpha"], hours=hours)


@app.get("/buildings/{building_id}/metrics", response_model=BuildingMetrics)
def building_metrics(building_id: str):
    with state["pool"].connection() as conn:
        known = scale_for(conn, building_id) is not None
    if not known:
        raise HTTPException(status_code=404, detail=f"unknown building: {building_id}")
    row = state["metrics"]["per_building"].get(building_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no scored hours for this building")
    return BuildingMetrics(building_id=building_id,
                           evaluation=state["metrics"]["evaluation"], **row)


@app.get("/metrics/summary")
def metrics_summary():
    return {"evaluation": state["metrics"]["evaluation"],
            **state["metrics"]["summary"]}
