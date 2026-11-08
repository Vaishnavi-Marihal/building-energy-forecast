"""Does weather help? The same backtest as backtest_weather.py, repeated with three
random seeds (the seed changes the LightGBM randomness and the simulated forecast-error
noise). For each seed, every weather scenario is compared with the no-weather model of
the same seed; the table then shows the median and the range of that change over seeds.

Run from the repo root:  python src/experiment_weather_seeds.py
Timing check (1 seed, July only):  python src/experiment_weather_seeds.py --quick
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from baselines import seasonal_naive
from features import build_features, FEATURES, WEATHER
from weather import load_site_temperature, ar1_noise, per_building

SEEDS = [0, 1, 2]
PARAMS = dict(objective="l1", n_estimators=400, learning_rate=0.05,
              num_leaves=63, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8,
              random_state=0, n_jobs=-1, verbose=-1)


def summarise(res):
    """res: one row per (seed, month, scenario) with the median normalised MAE 'nMAE'.
    Returns (per_seed, summary): per_seed has one row per seed and one column per
    scenario (mean nMAE over months); summary gives, per scenario, the median nMAE over
    seeds and the change vs the no-weather model of the same seed, in % (median, lowest,
    highest over seeds)."""
    order = list(dict.fromkeys(res["scenario"]))
    per_seed = res.groupby(["seed", "scenario"])["nMAE"].mean().unstack("scenario")[order]
    change = (per_seed.div(per_seed["none"], axis=0) - 1) * 100
    summary = pd.DataFrame({"nMAE_median": per_seed.median(),
                            "change_median_pct": change.median(),
                            "change_lowest_pct": change.min(),
                            "change_highest_pct": change.max()})
    return per_seed, summary


def main(quick=False):
    df = pd.read_csv("data/processed/education_clean.csv",
                     index_col="timestamp", parse_dates=True)
    sel = pd.read_csv("data/processed/selected_buildings.csv", index_col="building_id")
    building_site = sel.loc[df.columns, "site_id"]
    site_t = load_site_temperature(df.index)[list(building_site.unique())]
    p168 = seasonal_naive(df, 168)
    warmup = df.index[0] + pd.Timedelta(days=8)

    seeds = SEEDS[:1] if quick else SEEDS
    months = list(pd.period_range("2017-07", "2017-12", freq="M"))
    if quick:
        months = months[:1]

    rows = []
    for seed in seeds:
        scen = {"none": None, "oracle": per_building(site_t, building_site)}
        for s in [1.0, 2.0, 3.0]:
            noise = ar1_noise(df.index, list(site_t.columns), sigma=s, phi=0.95, seed=seed)
            scen[f"sim_{s:g}C"] = per_building(site_t + noise, building_site)
        params = {**PARAMS, "random_state": seed}
        for m in months:
            t0, t1 = m.start_time, m.end_time
            scale = df.loc[:t0 - pd.Timedelta(hours=1)].mean()    # training data only
            a = df.loc[t0:t1]
            b168 = p168.loc[a.index]
            for name, temp in scen.items():
                feat = build_features(df, scale, temp)
                cols = FEATURES if temp is None else FEATURES + WEATHER
                train = feat[(feat.index >= warmup) & (feat.index < t0)].dropna(subset=["y"])
                test = feat[(feat.index >= t0) & (feat.index <= t1)]
                model = lgb.LGBMRegressor(**params).fit(train[cols], train["y"])
                pred = model.predict(test[cols]) * test["building"].map(scale).to_numpy()
                pm = (test.assign(pred=pred).reset_index()
                      .pivot(index="timestamp", columns="building",
                             values="pred")[df.columns])
                pmw = pm.loc[a.index]
                valid = a.notna() & b168.notna() & pmw.notna()
                mm = (a - pmw).abs().where(valid).mean()
                ok = valid.sum() > 24 * 5
                rows.append({"seed": seed, "month": str(m), "scenario": name,
                             "nMAE": float((mm / scale)[ok].median())})
            print(f"seed {seed}, {m}: done", flush=True)

    per_seed, summary = summarise(pd.DataFrame(rows))
    print("\nMean over months of the median normalised MAE, one row per seed:")
    print(per_seed.round(4).to_string())
    print("\nChange vs the no-weather model of the same seed (negative = weather helps):")
    print(summary.round(3).to_string())
    if quick:
        print("\n(quick run: nothing saved)")
        return
    out = Path("docs") / "weather_seeds.json"
    out.write_text(json.dumps({"seeds": seeds, "months": [str(m) for m in months],
                               "per_seed": per_seed.round(5).to_dict(orient="index"),
                               "summary": summary.round(4).to_dict(orient="index")},
                              indent=2))
    print("\nsaved", out)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
