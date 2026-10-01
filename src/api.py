import datetime as dt
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from features import build_features, FEATURES
from uncertainty import make_interval

ART = Path("artifacts")
DATA = Path("data/processed/education_clean.csv")
FIRST_DATE = dt.date(2017, 7, 1)     # first date after the training/calibration data
LAST_DATE = dt.date(2017, 12, 31)

state = {}


def load_state(art=ART, data=DATA):
    meta = json.loads((Path(art) / "meta.json").read_text())
    metrics = json.loads((Path(art) / "metrics.json").read_text())
    df = pd.read_csv(data, index_col="timestamp", parse_dates=True)
    scale = pd.Series(meta["scale"])[df.columns]
    feat = build_features(df, scale)
    state.update(
        meta=meta, metrics=metrics, scale=scale,
        feat={b: g.drop(columns="building") for b, g in feat.groupby("building")},
        models={k: lgb.Booster(model_file=str(Path(art) / f"{k}.txt"))
                for k in ("point", "lo", "hi")},
    )


@asynccontextmanager
async def lifespan(app):
    if not state:
        load_state()
    yield


app = FastAPI(title="Building Energy Day-Ahead Forecast", version="0.1.0",
              lifespan=lifespan)


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


@app.get("/health")
def health():
    m = state["meta"]
    return {"status": "ok", "n_buildings": len(m["buildings"]),
            "trained_through": m["train_end"],
            "calibration_month": m["calibration_month"],
            "forecast_dates": [str(FIRST_DATE), str(LAST_DATE)]}


@app.get("/buildings")
def buildings():
    return state["meta"]["buildings"]


@app.get("/buildings/{building_id}/forecast", response_model=ForecastResponse)
def forecast(building_id: str, date: dt.date = Query(..., description="YYYY-MM-DD")):
    if building_id not in state["feat"]:
        raise HTTPException(status_code=404, detail=f"unknown building: {building_id}")
    if not (FIRST_DATE <= date <= LAST_DATE):
        raise HTTPException(status_code=422,
                            detail=f"date must be between {FIRST_DATE} and {LAST_DATE}")
    start = pd.Timestamp(date)
    day = state["feat"][building_id].loc[start: start + pd.Timedelta(hours=23)]
    if len(day) != 24:
        raise HTTPException(status_code=500, detail="incomplete day in data")

    X = day[FEATURES]
    m = state["models"]
    pred = np.maximum(m["point"].predict(X), 0.0)
    lower, upper = make_interval(pred, m["lo"].predict(X), m["hi"].predict(X),
                                 state["meta"]["Q"])
    sc = float(state["scale"][building_id])

    hours = [HourForecast(
        timestamp=ts.to_pydatetime(), horizon=int(hz),
        forecast_kwh=float(p * sc), lower_kwh=float(lo * sc), upper_kwh=float(up * sc),
        baseline_lag168_kwh=_opt(b168 * sc), baseline_lag24_kwh=_opt(b24 * sc),
        actual_kwh=_opt(y * sc),
    ) for ts, hz, p, lo, up, b168, b24, y in zip(
        day.index, day["horizon"], pred, lower, upper,
        day["lag168"], day["lag24"], day["y"])]

    return ForecastResponse(
        building_id=building_id, date=date,
        issue_time=(start - pd.Timedelta(hours=1)).to_pydatetime(),
        interval_level=1 - state["meta"]["alpha"], hours=hours)


@app.get("/buildings/{building_id}/metrics", response_model=BuildingMetrics)
def building_metrics(building_id: str):
    if building_id not in state["feat"]:
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