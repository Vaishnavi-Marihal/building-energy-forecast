import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from baselines import seasonal_naive


def _toy():
    idx = pd.date_range("2017-01-01", periods=400, freq="h")
    return pd.DataFrame({"b": np.arange(400, dtype=float)}, index=idx)


def test_lag168_is_same_hour_last_week():
    p = seasonal_naive(_toy(), 168)
    assert p["b"].iloc[:168].isna().all()
    assert p["b"].iloc[168] == 0.0
    assert p["b"].iloc[399] == 399 - 168


def test_lag24_is_same_hour_yesterday():
    p = seasonal_naive(_toy(), 24)
    assert p["b"].iloc[24] == 0.0
    assert p["b"].iloc[100] == 76.0


def test_baseline_sources_known_at_issue_time():
    issue = pd.Timestamp("2017-07-04 23:00")
    for lag in (24, 168):
        for h in range(24):
            target = pd.Timestamp("2017-07-05") + pd.Timedelta(hours=h)
            assert target - pd.Timedelta(hours=lag) <= issue