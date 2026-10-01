import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from features import build_features, FEATURES
from uncertainty import conformal_correction, make_interval

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

ALPHA = 0.10
CAL_START = pd.Timestamp("2017-06-01")
TEST_START = pd.Timestamp("2017-07-01")

df = pd.read_csv("data/processed/education_clean.csv",
                 index_col="timestamp", parse_dates=True)
sel = pd.read_csv("data/processed/selected_buildings.csv", index_col="building_id")
warmup = df.index[0] + pd.Timedelta(days=8)

scale = df.loc[:CAL_START - pd.Timedelta(hours=1)].mean()   # training data only
feat = build_features(df, scale)
train = feat[(feat.index >= warmup) & (feat.index < CAL_START)].dropna(subset=["y"])
cal = feat[(feat.index >= CAL_START) & (feat.index < TEST_START)].dropna(subset=["y"])
test = feat[feat.index >= TEST_START].copy()

BASE = dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
            min_child_samples=100, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=0, n_jobs=-1, verbose=-1)
point = lgb.LGBMRegressor(objective="l1", **BASE).fit(train[FEATURES], train["y"])
lo_m = lgb.LGBMRegressor(objective="quantile", alpha=ALPHA / 2, **BASE
                         ).fit(train[FEATURES], train["y"])
hi_m = lgb.LGBMRegressor(objective="quantile", alpha=1 - ALPHA / 2, **BASE
                         ).fit(train[FEATURES], train["y"])

# conformal correction from the calibration month
lo_c = lo_m.predict(cal[FEATURES])
hi_c = hi_m.predict(cal[FEATURES])
lo_c, hi_c = np.minimum(lo_c, hi_c), np.maximum(lo_c, hi_c)
yc = cal["y"].to_numpy()
Q = conformal_correction(np.maximum(lo_c - yc, yc - hi_c), ALPHA)

point.booster_.save_model(str(ART / "point.txt"))
lo_m.booster_.save_model(str(ART / "lo.txt"))
hi_m.booster_.save_model(str(ART / "hi.txt"))

meta = {
    "scale": {b: float(scale[b]) for b in df.columns},
    "Q": float(Q),
    "alpha": ALPHA,
    "train_end": "2017-05-31 23:00",
    "calibration_month": "2017-06",
    "features": FEATURES,
    "buildings": [{"id": b, "site_id": str(sel.loc[b, "site_id"])}
                  for b in df.columns],
}
(ART / "meta.json").write_text(json.dumps(meta, allow_nan=False))

# score July-December 2017, per building (fixed origin: trained through May)
pred = np.maximum(point.predict(test[FEATURES]), 0.0)
lower, upper = make_interval(pred, lo_m.predict(test[FEATURES]),
                             hi_m.predict(test[FEATURES]), Q)
test["pred"], test["lower"], test["upper"] = pred, lower, upper

v = test[test["y"].notna() & test["lag168"].notna() & test["lag24"].notna()].copy()
sc = v["building"].map(scale).to_numpy()
v["ae_model"] = (v["y"] - v["pred"]).abs() * sc
v["ae_168"] = (v["y"] - v["lag168"]).abs() * sc
v["ae_24"] = (v["y"] - v["lag24"]).abs() * sc
v["inside"] = (v["y"] >= v["lower"]) & (v["y"] <= v["upper"])
v["width"] = (v["upper"] - v["lower"]) * sc

g = v.groupby("building").agg(
    n_hours=("y", "size"), mae_model=("ae_model", "mean"),
    mae_lag168=("ae_168", "mean"), mae_lag24=("ae_24", "mean"),
    coverage_90=("inside", "mean"), mean_width=("width", "mean"))
g["relative_mae_vs_lag168"] = g["mae_model"] / g["mae_lag168"]
g = g.replace([np.inf, -np.inf], np.nan).round(4)

per_building = {
    b: {k: (None if pd.isna(x) else (int(x) if k == "n_hours" else float(x)))
        for k, x in row.items()}
    for b, row in g.iterrows()}

big = g[g["n_hours"] >= 120]
summary = {
    "n_buildings": int(len(big)),
    "median_relative_mae_vs_lag168": round(float(big["relative_mae_vs_lag168"].median()), 4),
    "pct_buildings_beating_lag168": round(float((big["relative_mae_vs_lag168"] < 1).mean()), 4),
    "pooled_coverage_90": round(float(v["inside"].mean()), 4),
    "median_mean_width_kwh": round(float(big["mean_width"].median()), 4),
}
metrics = {
    "evaluation": ("Fixed origin: models trained through 2017-05-31, conformal "
                   "calibration on June 2017, scored on 2017-07-01 to 2017-12-31 "
                   "on hours where actual, last-week and yesterday values all exist."),
    "summary": summary,
    "per_building": per_building,
}
(ART / "metrics.json").write_text(json.dumps(metrics, allow_nan=False))

print("Q (scaled units):", round(Q, 4))
print("summary:", json.dumps(summary, indent=2))
print("saved:", sorted(p.name for p in ART.iterdir()))