import pandas as pd
import lightgbm as lgb

from baselines import seasonal_naive
from features import build_features, FEATURES, WEATHER
from weather import load_site_temperature, ar1_noise, per_building

df = pd.read_csv("data/processed/education_clean.csv",
                 index_col="timestamp", parse_dates=True)
sel = pd.read_csv("data/processed/selected_buildings.csv",
                  index_col="building_id")
building_site = sel.loc[df.columns, "site_id"]
site_t = load_site_temperature(df.index)[list(building_site.unique())]

scen = {"none": None, "oracle": per_building(site_t, building_site)}
for s in [1.0, 2.0, 3.0]:
    noise = ar1_noise(df.index, list(site_t.columns), sigma=s, phi=0.95, seed=0)
    scen[f"sim_{s:g}C"] = per_building(site_t + noise, building_site)

p168 = seasonal_naive(df, 168)
warmup = df.index[0] + pd.Timedelta(days=8)
PARAMS = dict(objective="l1", n_estimators=400, learning_rate=0.05,
              num_leaves=63, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8,
              random_state=0, n_jobs=-1, verbose=-1)

rows = []
for m in pd.period_range("2017-07", "2017-12", freq="M"):
    t0, t1 = m.start_time, m.end_time
    scale = df.loc[:t0 - pd.Timedelta(hours=1)].mean()    # training data only
    a = df.loc[t0:t1]
    b168 = p168.loc[a.index]
    for name, temp in scen.items():
        feat = build_features(df, scale, temp)
        cols = FEATURES if temp is None else FEATURES + WEATHER
        train = feat[(feat.index >= warmup) & (feat.index < t0)].dropna(subset=["y"])
        test = feat[(feat.index >= t0) & (feat.index <= t1)]

        model = lgb.LGBMRegressor(**PARAMS).fit(train[cols], train["y"])
        pred = model.predict(test[cols]) * test["building"].map(scale).to_numpy()
        pm = (test.assign(pred=pred).reset_index()
              .pivot(index="timestamp", columns="building",
                     values="pred")[df.columns])
        pmw = pm.loc[a.index]

        valid = a.notna() & b168.notna() & pmw.notna()
        mm = (a - pmw).abs().where(valid).mean()
        m168 = (a - b168).abs().where(valid).mean()
        ok = valid.sum() > 24 * 5
        rows.append({"month": str(m), "scenario": name,
                     "nMAE": (mm / scale)[ok].median(),
                     "ratio_vs_lag168": (mm / m168)[ok].median()})
    print("done", m, flush=True)

res = pd.DataFrame(rows)
order = list(scen)
t = res.pivot(index="month", columns="scenario", values="nMAE")[order]
print("\nmedian normalised MAE (lower is better):")
print(t.round(4).to_string())
mean = t.mean()
print("\nmean over months, and change vs no weather:")
print(pd.DataFrame({"nMAE": mean.round(4),
                    "vs_none_pct": ((mean / mean["none"] - 1) * 100).round(1)}))
r = res.pivot(index="month", columns="scenario",
              values="ratio_vs_lag168")[order]
print("\nmedian error ratio vs same-hour-last-week baseline:")
print(r.round(3).to_string())
