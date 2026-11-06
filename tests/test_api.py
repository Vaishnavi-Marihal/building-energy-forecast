import sys
sys.path.insert(0, "src")

import datetime as dt
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.conninfo import conninfo_to_dict

import api
from features import build_features, FEATURES

# These tests need a PostgreSQL database. Point TEST_DATABASE_URL at an EMPTY
# test database (its name must end in "_test": the schema script drops tables).
TEST_DB = os.environ.get("TEST_DATABASE_URL")
SCHEMA = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"

pytestmark = pytest.mark.skipif(
    TEST_DB is None, reason="set TEST_DATABASE_URL to run the database tests")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    assert conninfo_to_dict(TEST_DB)["dbname"].endswith("_test"), \
        "refusing to run: TEST_DATABASE_URL must point at a database whose name ends in _test"
    tmp = tmp_path_factory.mktemp("art")
    rng = np.random.default_rng(0)
    idx = pd.date_range("2017-05-01", "2017-12-31 23:00", freq="h")
    base = 10 + 5 * np.sin(2 * np.pi * idx.hour / 24)
    df = pd.DataFrame({"b1": base + rng.normal(0, 1, len(idx)),
                       "b2": 2 * base + rng.normal(0, 1, len(idx))}, index=idx)
    df.index.name = "timestamp"
    df.loc["2017-07-10 03:00":"2017-07-10 05:00", "b1"] = np.nan   # masked hours -> NULL
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

    with psycopg.connect(TEST_DB) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA.read_text())
            cur.executemany(
                "INSERT INTO buildings (building_id, site_id, scale) VALUES (%s, %s, %s)",
                [(b["id"], b["site_id"], meta["scale"][b["id"]]) for b in meta["buildings"]])
            with cur.copy("COPY readings (building_id, ts, kwh) FROM STDIN") as copy:
                for col in df.columns:
                    for ts, v in zip(df.index.to_pydatetime(), df[col].to_numpy()):
                        copy.write_row((col, ts, None if pd.isna(v) else float(v)))

    api.state.clear()
    api.load_state(art=tmp, database_url=TEST_DB)
    yield {"client": TestClient(api.app), "df": df, "scale": scale}
    api.close_state()


@pytest.fixture
def client(env):
    return env["client"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_buildings(client):
    assert client.get("/buildings").json() == [{"id": "b1", "site_id": "S1"},
                                               {"id": "b2", "site_id": "S2"}]


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


@pytest.mark.parametrize("building", ["b1", "b2"])
@pytest.mark.parametrize("day", ["2017-07-01", "2017-07-10", "2017-07-17", "2017-09-15",
                                 "2017-12-31"])
def test_window_features_match_full_history(env, building, day):
    """Features from the 8-day database window must equal the features computed
    from the whole history. 2017-07-10 has masked (NULL) target hours and
    2017-07-17 has last-week values that are NULL."""
    full = build_features(env["df"], env["scale"])
    first = pd.Timestamp(day)
    expected = (full[full["building"] == building].drop(columns="building")
                .loc[first: first + pd.Timedelta(hours=23)])
    with api.state["pool"].connection() as conn:
        got = api.features_for_day(conn, building, float(env["scale"][building]),
                                   dt.date.fromisoformat(day))
    cols = FEATURES + ["y"]
    pd.testing.assert_frame_equal(expected[cols], got[cols], check_freq=False,
                                  check_names=False, atol=1e-12, rtol=0)


def test_window_with_missing_rows_is_rejected(env):
    # a day with no data in the table: the window comes back short
    with api.state["pool"].connection() as conn:
        with pytest.raises(ValueError):
            api.features_for_day(conn, "b1", 1.0, dt.date(2018, 3, 1))
