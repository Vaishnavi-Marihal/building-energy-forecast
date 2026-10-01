import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from features import build_features, FEATURES


def _toy():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2017-01-01", periods=24 * 30, freq="h")
    df = pd.DataFrame(rng.random((len(idx), 3)) + 1, index=idx,
                      columns=["a", "b", "c"])
    return df, pd.Series(1.0, index=df.columns)


def _day(feat, day):
    d = feat[(feat.index >= day) & (feat.index < day + pd.Timedelta(days=1))]
    return (d.reset_index().sort_values(["building", "timestamp"])
            .reset_index(drop=True))


def test_features_ignore_everything_after_issue_time():
    df, scale = _toy()
    day = pd.Timestamp("2017-01-20")
    issue = day - pd.Timedelta(hours=1)          # 23:00 the day before
    tampered = df.copy()
    tampered.loc[tampered.index > issue] = 999.0
    a = _day(build_features(df, scale), day)
    b = _day(build_features(tampered, scale), day)
    assert_frame_equal(a[FEATURES], b[FEATURES])


def test_check_catches_a_leaky_feature():
    df, _ = _toy()
    day = pd.Timestamp("2017-01-20")
    issue = day - pd.Timedelta(hours=1)
    tampered = df.copy()
    tampered.loc[tampered.index > issue] = 999.0
    span = slice(day, day + pd.Timedelta(hours=23))
    assert not df.shift(1).loc[span].equals(tampered.shift(1).loc[span])