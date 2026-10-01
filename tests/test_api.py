import sys
sys.path.insert(0, "src")

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api
from features import build_features, FEATURES


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("art")
    rng = np.random.default_rng(0)
    idx = pd.date_range("2017-05-01", "2017-12-31 23:00", freq="h")
    base = 10 + 5 * np.sin(2 * np.pi * idx.hour / 24)
    df = pd.DataFrame({"b1": base + rng.normal(0, 1, len(idx)),
                       "b2": 2 * base + rng.normal(0, 1, len(idx))}, index=idx)
    df.index.name = "timestamp"
    df.to_csv(tmp / "data.csv")
    scale = df.mean()
    feat = build_features(df, scale).dropna()
    for name in ("point", "lo", "hi"):
        m = lgb.LGBMRegressor(n_estimators=20, min_child_samples=20,
                              verbose=-1, random_state=0)
        m.fit(feat[FEATURES], feat["y"])
        m.booster_.save_model(str(tmp / f"{name}.txt"))
    meta = {"scale": {c: float(scale[c]) for c in df.columns},
            "Q": 0.1, "alpha": 0.1, "train_end": "2017-05-31 23:00",
            "calibration_month": "2017-06",
            "buildings": [{"id": "b1", "site_id": "S1"},
                          {"id": "b2", "site_id": "S2"}]}
    row = {"n_hours": 100, "mae_model": 1.0, "mae_lag168": 2.0, "mae_lag24": 2.5,
           "coverage_90": 0.9, "mean_width": 3.0, "relative_mae_vs_lag168": 0.5}
    metrics = {"evaluation": "toy",
               "summary": {"n_buildings": 1, "pooled_coverage_90": 0.9},
               "per_building": {"b1": row}}
    (tmp / "meta.json").write_text(json.dumps(meta))
    (tmp / "metrics.json").write_text(json.dumps(metrics))
    api.state.clear()
    api.load_state(art=tmp, data=tmp / "data.csv")
    yield TestClient(api.app)
    api.state.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_buildings(client):
    assert [b["id"] for b in client.get("/buildings").json()] == ["b1", "b2"]


def test_forecast_ok(client):
    r = client.get("/buildings/b1/forecast", params={"date": "2017-07-15"})
    assert r.status_code == 200
    body = r.json()
    h = body["hours"]
    assert len(h) == 24
    assert [x["horizon"] for x in h] == list(range(1, 25))
    assert h[0]["timestamp"].startswith("2017-07-15T00:00")
    assert body["issue_time"].startswith("2017-07-14T23:00")
    for x in h:
        assert x["lower_kwh"] <= x["forecast_kwh"] <= x["upper_kwh"]


def test_forecast_unknown_building(client):
    r = client.get("/buildings/nope/forecast", params={"date": "2017-07-15"})
    assert r.status_code == 404


@pytest.mark.parametrize("d", ["2017-06-30", "2018-01-01"])
def test_forecast_date_out_of_range(client, d):
    r = client.get("/buildings/b1/forecast", params={"date": d})
    assert r.status_code == 422


def test_forecast_bad_date(client):
    r = client.get("/buildings/b1/forecast", params={"date": "2017-13-45"})
    assert r.status_code == 422


def test_forecast_missing_date(client):
    assert client.get("/buildings/b1/forecast").status_code == 422


def test_metrics_ok(client):
    r = client.get("/buildings/b1/metrics")
    assert r.status_code == 200 and r.json()["relative_mae_vs_lag168"] == 0.5


def test_metrics_unknown(client):
    assert client.get("/buildings/nope/metrics").status_code == 404


def test_summary(client):
    r = client.get("/metrics/summary")
    assert r.status_code == 200 and r.json()["n_buildings"] == 1