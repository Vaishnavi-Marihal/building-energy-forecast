import numpy as np
import pandas as pd
import lightgbm as lgb

from features import build_features, FEATURES
from uncertainty import conformal_correction, coverage

df = pd.read_csv("data/processed/education_clean.csv",
                 index_col="timestamp", parse_dates=True)
warmup = df.index[0] + pd.Timedelta(days=8)
ALPHA = 0.10


def params(q):
    return dict(objective="quantile", alpha=q, n_estimators=400,
                learning_rate=0.05, num_leaves=63, min_child_samples=100,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                random_state=0, n_jobs=-1, verbose=-1)


rows, all_h, all_in = [], [], []
for m in pd.period_range("2017-07", "2017-12", freq="M"):
    t0, t1 = m.start_time, m.end_time
    c0 = (m - 1).start_time                    # calibration month = month before
    scale = df.loc[:c0 - pd.Timedelta(hours=1)].mean()   # training data only
    feat = build_features(df, scale)

    train = feat[(feat.index >= warmup) & (feat.index < c0)].dropna(subset=["y"])
    cal = feat[(feat.index >= c0) & (feat.index < t0)].dropna(subset=["y"])
    test = feat[(feat.index >= t0) & (feat.index <= t1)].dropna(subset=["y"])

    lo_m = lgb.LGBMRegressor(**params(ALPHA / 2)).fit(train[FEATURES], train["y"])
    hi_m = lgb.LGBMRegressor(**params(1 - ALPHA / 2)).fit(train[FEATURES], train["y"])

    def band(d):
        lo, hi = lo_m.predict(d[FEATURES]), hi_m.predict(d[FEATURES])
        return np.minimum(lo, hi), np.maximum(lo, hi)   # guard against crossing

    lo_c, hi_c = band(cal)
    yc = cal["y"].to_numpy()
    Q = conformal_correction(np.maximum(lo_c - yc, yc - hi_c), ALPHA)

    lo_t, hi_t = band(test)
    y = test["y"].to_numpy()
    inside = (y >= lo_t - Q) & (y <= hi_t + Q)
    per_b = pd.Series(inside, index=test["building"].to_numpy()).groupby(level=0).mean()

    all_h.append(test["horizon"].to_numpy())
    all_in.append(inside)
    rows.append({
        "month": str(m),
        "cal_rows": len(cal),
        "Q": round(Q, 4),
        "cov_raw": round(coverage(y, lo_t, hi_t), 3),
        "cov_calibrated": round(float(inside.mean()), 3),
        "width_raw": round(float(np.mean(hi_t - lo_t)), 3),
        "width_cal": round(float(np.mean(hi_t - lo_t + 2 * Q)), 3),
        "bldg_cov_p10": round(per_b.quantile(0.10), 3),
        "bldg_cov_p90": round(per_b.quantile(0.90), 3),
    })

print(pd.DataFrame(rows).to_string(index=False))

h, ins = np.concatenate(all_h), np.concatenate(all_in)
print("\npooled calibrated coverage by horizon block:")
for a, b in [(1, 6), (7, 12), (13, 18), (19, 24)]:
    sel = (h >= a) & (h <= b)
    print(f"  horizons {a}-{b}: {ins[sel].mean():.3f}")