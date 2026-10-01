import pandas as pd

FEATURES = ["lag24", "lag48", "lag168", "dmean", "dmin", "dmax",
            "last", "horizon", "dow", "month"]
WEATHER = ["temp", "temp_dmean"]


def _daily_to_hourly(daily, index):
    return daily.reindex(index.normalize()).set_axis(index)


def build_features(df, scale, temp=None):
    """One row per (building, target hour). Load features use data up to the
    23:00 issue time. temp (optional, same index/columns as df, degrees C) is a
    weather input for the target hours; it uses target-day information by
    design, so the scenario that produced it must always be labelled."""
    assert (df.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
    assert df.index[0].hour == 0 and df.index[-1].hour == 23
    z = df / scale

    daily = z.resample("D")
    dmean = _daily_to_hourly(daily.mean().shift(1), z.index)
    dmin = _daily_to_hourly(daily.min().shift(1), z.index)
    dmax = _daily_to_hourly(daily.max().shift(1), z.index)

    at23 = z[z.index.hour == 23].copy()
    at23.index = at23.index.normalize()
    last = _daily_to_hourly(at23.shift(1), z.index)

    tdm = None
    if temp is not None:
        assert temp.index.equals(df.index)
        assert list(temp.columns) == list(df.columns)
        tdm = _daily_to_hourly(temp.resample("D").mean(), z.index)

    parts = []
    for b in z.columns:
        d = {
            "building": b,
            "y": z[b],
            "lag24": z[b].shift(24),
            "lag48": z[b].shift(48),
            "lag168": z[b].shift(168),
            "dmean": dmean[b], "dmin": dmin[b], "dmax": dmax[b],
            "last": last[b],
            "horizon": z.index.hour + 1,
            "dow": z.index.dayofweek,
            "month": z.index.month,
        }
        if temp is not None:
            d["temp"] = temp[b]
            d["temp_dmean"] = tdm[b]
        parts.append(pd.DataFrame(d, index=z.index))
    out = pd.concat(parts)
    out.index.name = "timestamp"
    return out