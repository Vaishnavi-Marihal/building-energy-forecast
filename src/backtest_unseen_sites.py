"""Leave-one-site-out evaluation: how well does the model forecast a SITE it
never saw in training?

For each of the 14 sites: train on the other 13 sites (data through 2017-05-31),
then forecast the held-out site's buildings for 2017-07-01 to 2017-12-31 and
compare with "same hour last week". A control model trained on all 56 buildings
(the same setup as train_final.py) is scored on the same hours, so the only
difference between the two numbers is whether the site was in the training data.

Run from the repo root:  python src/backtest_unseen_sites.py
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import build_features, FEATURES

CAL_START = pd.Timestamp("2017-06-01")    # training data ends the hour before this
TEST_START = pd.Timestamp("2017-07-01")   # June is left out, as in train_final.py
MIN_HOURS = 120                           # fewer scored hours than this: building not scored
PARAMS = dict(objective="l1", n_estimators=400, learning_rate=0.05,
              num_leaves=63, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8,
              random_state=0, n_jobs=-1, verbose=-1)


def site_folds(buildings):
    """buildings: list of {"id": ..., "site_id": ...}.
    Yields (held_out_site, train_building_ids, test_building_ids). Every
    building of the held-out site is on the test side, none on the train side."""
    by_site = {}
    for b in buildings:
        by_site.setdefault(b["site_id"], []).append(b["id"])
    for site in sorted(by_site):
        test = by_site[site]
        train = [i for s, ids in by_site.items() if s != site for i in ids]
        yield site, train, test


def score_buildings(rows, min_hours=MIN_HOURS):
    """rows: columns building, y, lag168, lag24, pred (all in scaled units).
    Scores only hours where the actual value and both baselines exist.
    Returns one row per building with the model's error relative to the
    same-hour-last-week baseline (below 1 = the model is better)."""
    ok = rows[rows["y"].notna() & rows["lag168"].notna() & rows["lag24"].notna()]
    g = ok.assign(ae_model=(ok["y"] - ok["pred"]).abs(),
                  ae_168=(ok["y"] - ok["lag168"]).abs()).groupby("building")
    out = g.agg(n_hours=("y", "size"), mae_model=("ae_model", "mean"),
                mae_lag168=("ae_168", "mean"))
    out["ratio"] = out["mae_model"] / out["mae_lag168"]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out[out["n_hours"] >= min_hours]


def main():
    root = Path(__file__).resolve().parent.parent
    meta = json.loads((root / "deploy/model/meta.json").read_text())
    df = pd.read_csv("data/processed/education_clean.csv",
                     index_col="timestamp", parse_dates=True)
    warmup = df.index[0] + pd.Timedelta(days=8)
    scale = df.loc[:CAL_START - pd.Timedelta(hours=1)].mean()   # training period only
    feat = build_features(df, scale)
    train_all = feat[(feat.index >= warmup) & (feat.index < CAL_START)].dropna(subset=["y"])
    test_all = feat[feat.index >= TEST_START]

    def fit_predict(train_ids, test_ids):
        train = train_all[train_all["building"].isin(train_ids)]
        test = test_all[test_all["building"].isin(test_ids)].copy()
        model = lgb.LGBMRegressor(**PARAMS).fit(train[FEATURES], train["y"])
        test["pred"] = np.maximum(model.predict(test[FEATURES]), 0.0)
        return test

    ids = [b["id"] for b in meta["buildings"]]
    print("training the control model (all 56 buildings)...", flush=True)
    seen = score_buildings(fit_predict(ids, ids))

    rows, unseen_parts = [], []
    for site, train_ids, test_ids in site_folds(meta["buildings"]):
        assert not set(train_ids) & set(test_ids)
        unseen = score_buildings(fit_predict(train_ids, test_ids))
        unseen_parts.append(unseen)
        both = [b for b in test_ids if b in seen.index and b in unseen.index]
        rows.append({
            "site": site,
            "buildings_scored": len(both),
            "seen_ratio": round(float(seen.loc[both, "ratio"].median()), 3),
            "unseen_ratio": round(float(unseen.loc[both, "ratio"].median()), 3),
        })
        print(f"  held out {site}: done", flush=True)

    table = pd.DataFrame(rows)
    table["change"] = (table["unseen_ratio"] - table["seen_ratio"]).round(3)
    print("\nPer site (median over the site's buildings; ratio below 1 = beats last week)")
    print(table.to_string(index=False))

    unseen_all = pd.concat(unseen_parts)
    summary = {
        "n_sites": int(len(table)),
        "median_seen_ratio_by_site": round(float(table["seen_ratio"].median()), 3),
        "median_unseen_ratio_by_site": round(float(table["unseen_ratio"].median()), 3),
        "sites_beating_lag168_when_unseen": int((table["unseen_ratio"] < 1).sum()),
        "worst_site_unseen": table.loc[table["unseen_ratio"].idxmax(), "site"],
        "worst_site_unseen_ratio": round(float(table["unseen_ratio"].max()), 3),
        "median_seen_ratio_by_building": round(float(seen["ratio"].median()), 4),
        "median_unseen_ratio_by_building": round(float(unseen_all["ratio"].median()), 4),
        "buildings_beating_lag168_when_unseen": f"{int((unseen_all['ratio'] < 1).sum())} of {len(unseen_all)}",
    }
    print("\nSummary")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out = root / "docs" / "unseen_sites.json"
    out.write_text(json.dumps({"summary": summary, "per_site": rows}, indent=2))
    print("\nsaved", out.relative_to(root))


if __name__ == "__main__":
    main()
