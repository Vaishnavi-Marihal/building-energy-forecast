"""Does a US-federal-holiday flag help? Trained twice with the same setup as
train_final.py (data through May 2017, point model, seeds 0-2): once with the 10
usual features, once with an extra is_holiday feature. Scored on July-December 2017,
holiday days and other days separately, as the model's error relative to
"same hour last week" (below 1 = better), median over buildings.

Run from the repo root:  python src/experiment_holiday.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from backtest_unseen_sites import score_buildings, PARAMS, CAL_START, TEST_START
from features import build_features, FEATURES, holiday_flag

SEEDS = [0, 1, 2]
MIN_HOURS = {"holiday days": 48, "other days": 120}   # the holiday group has only a few days


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/education_clean.csv"
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    warmup = df.index[0] + pd.Timedelta(days=8)
    scale = df.loc[:CAL_START - pd.Timedelta(hours=1)].mean()    # training period only
    feat = build_features(df, scale)
    feat["is_holiday"] = holiday_flag(feat.index)
    train = feat[(feat.index >= warmup) & (feat.index < CAL_START)].dropna(subset=["y"])
    test = feat[feat.index >= TEST_START]
    n_days = test.loc[test["is_holiday"] == 1].index.normalize().nunique()
    print(f"training rows on a holiday: {int(train['is_holiday'].sum())} of {len(train)}; "
          f"holiday days in the test period: {n_days}", flush=True)

    variants = {"without holiday flag": FEATURES, "with holiday flag": FEATURES + ["is_holiday"]}
    rows = []
    for seed in SEEDS:
        for name, cols in variants.items():
            model = lgb.LGBMRegressor(**{**PARAMS, "random_state": seed}).fit(train[cols], train["y"])
            t = test.copy()
            t["pred"] = np.maximum(model.predict(t[cols]), 0.0)
            for group, mask in (("holiday days", t["is_holiday"] == 1),
                                ("other days", t["is_holiday"] == 0)):
                s = score_buildings(t[mask], min_hours=MIN_HOURS[group])
                rows.append({"variant": name, "group": group, "seed": seed,
                             "buildings": len(s), "median_ratio": float(s["ratio"].median())})
            print(f"  seed {seed}, {name}: done", flush=True)

    res = pd.DataFrame(rows)
    table = (res.groupby(["group", "variant"])["median_ratio"]
             .agg(median="median", lowest="min", highest="max").round(3).reset_index())
    print("\nMedian over buildings of (model error / last-week error); below 1 = better.")
    print("median / lowest / highest are taken over the seeds.")
    print(table.to_string(index=False))
    out = Path("docs") / "holiday_experiment.json"
    out.write_text(json.dumps({"seeds": SEEDS, "holiday_days_in_test": int(n_days),
                               "results": table.to_dict(orient="records")}, indent=2))
    print("\nsaved", out)


if __name__ == "__main__":
    main()
