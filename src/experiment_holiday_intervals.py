"""Does the holiday flag also fix the prediction INTERVALS on holidays?

Same setup as train_final.py (data through May 2017, 5% and 95% quantile models,
conformal correction from June 2017, scored July-December 2017), trained with and
without the is_holiday feature, seeds 0-2. For holiday days and other days it
reports the share of hours inside the interval (target 0.90) and the median
interval width relative to the building's mean load (a wider interval covers more
but says less, so both are shown).

Run from the repo root:  python src/experiment_holiday_intervals.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiment_holiday import holiday_flag, SEEDS
from features import build_features, FEATURES
from uncertainty import conformal_correction, make_interval, coverage

ALPHA = 0.10
CAL_START = pd.Timestamp("2017-06-01")
TEST_START = pd.Timestamp("2017-07-01")
BASE = dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
            min_child_samples=100, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, n_jobs=-1, verbose=-1)


def interval_stats(rows):
    """rows: columns y, lag168, lag24, lower, upper (scaled units).
    Scores hours where the actual value and both baselines exist, like train_final.py.
    Returns (share of hours inside the interval, median interval width)."""
    ok = rows[rows["y"].notna() & rows["lag168"].notna() & rows["lag24"].notna()]
    return (coverage(ok["y"], ok["lower"], ok["upper"]),
            float(np.median(ok["upper"] - ok["lower"])))


def fit_intervals(train, cal, test, cols, seed):
    """Point model + two quantile models + conformal correction. Returns test with lower/upper."""
    p = {**BASE, "random_state": seed}
    point = lgb.LGBMRegressor(objective="l1", **p).fit(train[cols], train["y"])
    lo_m = lgb.LGBMRegressor(objective="quantile", alpha=ALPHA / 2, **p).fit(train[cols], train["y"])
    hi_m = lgb.LGBMRegressor(objective="quantile", alpha=1 - ALPHA / 2, **p).fit(train[cols], train["y"])
    lo_c, hi_c = lo_m.predict(cal[cols]), hi_m.predict(cal[cols])
    lo_c, hi_c = np.minimum(lo_c, hi_c), np.maximum(lo_c, hi_c)
    yc = cal["y"].to_numpy()
    q = conformal_correction(np.maximum(lo_c - yc, yc - hi_c), ALPHA)
    t = test.copy()
    pred = np.maximum(point.predict(t[cols]), 0.0)
    t["lower"], t["upper"] = make_interval(pred, lo_m.predict(t[cols]), hi_m.predict(t[cols]), q)
    return t


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/education_clean.csv"
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    warmup = df.index[0] + pd.Timedelta(days=8)
    scale = df.loc[:CAL_START - pd.Timedelta(hours=1)].mean()
    feat = build_features(df, scale)
    feat["is_holiday"] = holiday_flag(feat.index)
    train = feat[(feat.index >= warmup) & (feat.index < CAL_START)].dropna(subset=["y"])
    cal = feat[(feat.index >= CAL_START) & (feat.index < TEST_START)].dropna(subset=["y"])
    test = feat[feat.index >= TEST_START]
    print(f"holiday hours in the calibration month: {int(cal['is_holiday'].sum())}", flush=True)

    variants = {"without holiday flag": FEATURES, "with holiday flag": FEATURES + ["is_holiday"]}
    rows = []
    for seed in SEEDS:
        for name, cols in variants.items():
            t = fit_intervals(train, cal, test, cols, seed)
            for group, mask in (("holiday days", t["is_holiday"] == 1),
                                ("other days", t["is_holiday"] == 0)):
                cov, width = interval_stats(t[mask])
                rows.append({"variant": name, "group": group, "seed": seed,
                             "coverage": cov, "width": width})
            print(f"  seed {seed}, {name}: done", flush=True)

    res = pd.DataFrame(rows)
    table = (res.groupby(["group", "variant"])
             .agg(coverage=("coverage", "median"), cov_low=("coverage", "min"),
                  cov_high=("coverage", "max"), width=("width", "median"))
             .round(3).reset_index())
    print("\nShare of hours inside the interval (target 0.90) and median width (relative to mean load).")
    print("coverage is the median over seeds; cov_low / cov_high are the lowest and highest seed.")
    print(table.to_string(index=False))
    out = Path("docs") / "holiday_intervals.json"
    out.write_text(json.dumps({"seeds": SEEDS, "results": table.to_dict(orient="records")}, indent=2))
    print("\nsaved", out)


if __name__ == "__main__":
    main()
