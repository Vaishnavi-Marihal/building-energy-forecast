import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from features import FEATURES, DEPLOYED_FEATURES, build_features


def test_is_holiday_is_one_for_all_24_hours_of_labor_day_and_zero_otherwise():
    idx = pd.date_range("2017-08-31", "2017-09-08 23:00", freq="h")
    df = pd.DataFrame({"b": np.arange(len(idx), dtype=float) + 1.0}, index=idx)
    feat = build_features(df, pd.Series({"b": 1.0}))
    by_day = feat.groupby(feat.index.normalize())["is_holiday"].agg(["min", "max"])
    for day, row in by_day.iterrows():
        expected = 1 if str(day.date()) == "2017-09-04" else 0   # Labor Day 2017
        assert row["min"] == row["max"] == expected, day


def test_research_features_stay_at_ten_and_deployed_adds_only_the_flag():
    assert len(FEATURES) == 10 and "is_holiday" not in FEATURES
    assert DEPLOYED_FEATURES == FEATURES + ["is_holiday"]
