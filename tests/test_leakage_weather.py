import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from features import build_features, FEATURES, WEATHER


def _toy():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2017-01-01", periods=24 * 30, freq="h")
    df = pd.DataFrame(rng.random((len(idx), 3)) + 1, index=idx,
                      columns=["a", "b", "c"])
    temp = pd.DataFrame(rng.normal(10, 5, (len(idx), 3)), index=idx,
                        columns=df.columns)
    return df, temp, pd.Series(1.0, index=df.columns)


def _day(feat, day):
    d = feat[(feat.index >= day) & (feat.index < day + pd.Timedelta(days=1))]
    return (d.reset_index().sort_values(["building", "timestamp"])
            .reset_index(drop=True))


def test_load_after_issue_time_never_reaches_any_feature():
    df, temp, scale = _toy()
    day = pd.Timestamp("2017-01-20")
    issue = day - pd.Timedelta(hours=1)
    tampered = df.copy()
    tampered.loc[tampered.index > issue] = 999.0
    a = _day(build_features(df, scale, temp), day)
    b = _day(build_features(tampered, scale, temp), day)
    assert_frame_equal(a[FEATURES + WEATHER], b[FEATURES + WEATHER])


def test_load_features_do_not_depend_on_weather():
    df, temp, scale = _toy()
    day = pd.Timestamp("2017-01-20")
    t2 = temp.copy()
    t2.loc[t2.index >= day] += 50.0
    a = _day(build_features(df, scale, temp), day)
    b = _day(build_features(df, scale, t2), day)
    assert_frame_equal(a[FEATURES], b[FEATURES])
    assert not a[WEATHER].equals(b[WEATHER])