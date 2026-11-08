import sys
sys.path.insert(0, "src")

import pandas as pd
import pytest

from experiment_weather_seeds import summarise


def make(rows):
    return pd.DataFrame(rows, columns=["seed", "month", "scenario", "nMAE"])


def test_change_is_measured_against_the_no_weather_model_of_the_same_seed():
    # seed 0: weather 1% better; seed 1: weather 1% worse
    res = make([
        (0, "2017-07", "none", 0.100), (0, "2017-07", "oracle", 0.099),
        (1, "2017-07", "none", 0.200), (1, "2017-07", "oracle", 0.202),
    ])
    per_seed, summary = summarise(res)
    assert list(per_seed.columns) == ["none", "oracle"]
    row = summary.loc["oracle"]
    assert row["change_lowest_pct"] == pytest.approx(-1.0)
    assert row["change_highest_pct"] == pytest.approx(1.0)
    assert row["change_median_pct"] == pytest.approx(0.0)
    assert summary.loc["none", "change_median_pct"] == pytest.approx(0.0)


def test_months_are_averaged_within_a_seed_before_comparing():
    res = make([
        (0, "2017-07", "none", 0.10), (0, "2017-07", "oracle", 0.10),
        (0, "2017-08", "none", 0.20), (0, "2017-08", "oracle", 0.18),
    ])
    per_seed, summary = summarise(res)
    assert per_seed.loc[0, "none"] == pytest.approx(0.15)
    assert per_seed.loc[0, "oracle"] == pytest.approx(0.14)
    assert summary.loc["oracle", "change_median_pct"] == pytest.approx(-100 / 15)
