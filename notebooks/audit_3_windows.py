import pandas as pd
from pathlib import Path

DATA = Path.home() / "Documents" / "building-data-genome-project-2" / "data"
meta = pd.read_csv(DATA / "metadata" / "metadata.csv")
elec = pd.read_csv(DATA / "meters" / "raw" / "electricity.csv",
                   index_col="timestamp", parse_dates=True)

info = meta[meta["building_id"].isin(elec.columns)].set_index("building_id")
ids = info.index[info["primaryspaceusage"].isin(
    ["Education", "Office", "Healthcare"])]
sub = elec[ids]

early = sub.loc["2016-01-01":"2017-06-30"]   # would be training data
late = sub.loc["2017-07-01":"2017-12-31"]    # would be test months


def longest_zero_run(s):
    z = (s == 0)
    grp = (z != z.shift()).cumsum()
    lengths = z.groupby(grp).sum()
    if lengths.max() == 0:
        return pd.Series({"run_len": 0, "run_start": pd.NaT})
    g = lengths.idxmax()
    start = s.index[(grp == g).values][0]
    return pd.Series({"run_len": int(lengths.max()), "run_start": start})


runs = sub.apply(longest_zero_run).T
runs["run_len"] = runs["run_len"].astype(int)
runs["run_start"] = pd.to_datetime(runs["run_start"])

q = pd.DataFrame({
    "category": info.loc[ids, "primaryspaceusage"],
    "site": info.loc[ids, "site_id"],
    "miss_early": early.isna().mean(),
    "miss_late": late.isna().mean(),
}).join(runs)

print("1) buildings per site (rows = site, columns = category):")
print(q.groupby(["site", "category"]).size().unstack(fill_value=0))

q["clean_windows"] = (q["miss_early"] <= 0.05) & (q["miss_late"] <= 0.05)
print("\n2) buildings with <=5% missing in BOTH windows:")
print(q.groupby("category")["clean_windows"].agg(["size", "sum"]))

print("\n3) longest zero run, bucketed (hours):")
buckets = pd.cut(q["run_len"], [-1, 0, 24, 72, 168, 720, 10**9],
                 labels=["0", "1-24", "25-72", "73-168", "169-720", ">720"])
print(pd.crosstab(q["category"], buckets))

long_run = q[q["run_len"] > 24]
at_start = long_run["run_start"] == elec.index[0]
print("\n4) long runs (>24h): how many start at the very first hour?")
print(pd.crosstab(long_run["category"], at_start,
                  colnames=["starts_at_first_hour"]))