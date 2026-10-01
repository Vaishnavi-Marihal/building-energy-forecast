import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path.home() / "Documents" / "building-data-genome-project-2" / "data"
meta = pd.read_csv(DATA / "metadata" / "metadata.csv")
elec = pd.read_csv(DATA / "meters" / "raw" / "electricity.csv",
                   index_col="timestamp", parse_dates=True)

info = meta[meta["building_id"].isin(elec.columns)].set_index("building_id")
ids = info.index[info["primaryspaceusage"].isin(["Education", "Office"])]
sub = elec[ids]


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
runs = runs.join(info[["primaryspaceusage", "site_id"]])
runs["long"] = runs["run_len"] > 168

print("1) buildings with a zero run > 168h, by site:")
by_site = runs.groupby("site_id").agg(n=("long", "size"),
                                      n_long=("long", "sum"))
by_site["share_long"] = (by_site["n_long"] / by_site["n"]).round(2)
print(by_site.sort_values("share_long", ascending=False))

print("\n2) start month of those long runs:")
print(runs[runs["long"]]["run_start"].dt.to_period("M")
      .value_counts().sort_index())

examples = runs[runs["long"]].sample(6, random_state=0)
fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
for ax, (bid, row) in zip(axes, examples.iterrows()):
    sub[bid].resample("D").mean().plot(ax=ax)
    ax.set_title(f"{bid} | {row['primaryspaceusage']} | site {row['site_id']}"
                 f" | longest zero run {row['run_len']}h", fontsize=9)
plt.tight_layout()
plt.savefig("notebooks/zero_runs_examples.png", dpi=100)
print("\nsaved notebooks/zero_runs_examples.png")