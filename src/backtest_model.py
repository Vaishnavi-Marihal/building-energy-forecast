import pandas as pd
import lightgbm as lgb

from baselines import seasonal_naive
from features import build_features, FEATURES

df = pd.read_csv("data/processed/education_clean.csv",
                 index_col="timestamp", parse_dates=True)
p24 = seasonal_naive(df, 24)
p168 = seasonal_naive(df, 168)
warmup = df.index[0] + pd.Timedelta(days=8)

PARAMS = dict(objective="l1", n_estimators=400, learning_rate=0.05,
              num_leaves=63, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8,
              random_state=0, n_jobs=-1, verbose=-1)

rows = []
for m in pd.period_range("2017-07", "2017-12", freq="M"):
    t0, t1 = m.start_time, m.end_time
    scale = df.loc[:t0 - pd.Timedelta(hours=1)].mean()   # training data only
    feat = build_features(df, scale)

    train = feat[(feat.index >= warmup) & (feat.index < t0)].dropna(subset=["y"])
    test = feat[(feat.index >= t0) & (feat.index <= t1)]

    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(train[FEATURES], train["y"])

    pred = model.predict(test[FEATURES]) * test["building"].map(scale).to_numpy()
    pm = (test.assign(pred=pred).reset_index()
          .pivot(index="timestamp", columns="building", values="pred")[df.columns])

    a = df.loc[t0:t1]
    b24, b168, pmw = p24.loc[a.index], p168.loc[a.index], pm.loc[a.index]
    valid = a.notna() & b24.notna() & b168.notna() & pmw.notna()

    def mae(p):
        return (a - p).abs().where(valid).mean()

    m24, m168, mm = mae(b24), mae(b168), mae(pmw)
    ok = valid.sum() > 24 * 5

    rows.append({
        "month": str(m),
        "buildings": int(ok.sum()),
        "nMAE_lag24": round((m24 / scale)[ok].median(), 4),
        "nMAE_lag168": round((m168 / scale)[ok].median(), 4),
        "nMAE_model": round((mm / scale)[ok].median(), 4),
        "model_over_lag168": round((mm / m168)[ok].median(), 3),
        "pct_model_beats_lag168": round((mm < m168)[ok].mean(), 3),
        "model_over_lag24": round((mm / m24)[ok].median(), 3),
        "pct_model_beats_lag24": round((mm < m24)[ok].mean(), 3),
    })

print(pd.DataFrame(rows).to_string(index=False))
print("\nfeature importance (gain share, last fold):")
gain = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
print((gain / gain.sum()).round(3).sort_values(ascending=False))