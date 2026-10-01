import pandas as pd
from pathlib import Path

DATA = Path.home() / "Documents" / "building-data-genome-project-2" / "data"

meta = pd.read_csv(DATA / "metadata" / "metadata.csv")
elec = pd.read_csv(DATA / "meters" / "raw" / "electricity.csv",
                   index_col="timestamp", parse_dates=True)

info = meta[meta["building_id"].isin(elec.columns)].set_index("building_id")


def longest_zero_run(s):
    z = (s == 0).astype(int)
    return z.groupby((z != z.shift()).cumsum()).cumsum().max()


q = pd.DataFrame({
    "missing_frac": elec.isna().mean(),
    "longest_zero_run_h": elec.apply(longest_zero_run),
    "first_valid": elec.apply(lambda s: s.first_valid_index()),
})
q = q.join(info[["primaryspaceusage", "site_id"]])

print("overall missing_frac and zero-run quantiles:")
print(q[["missing_frac", "longest_zero_run_h"]]
      .quantile([0.5, 0.75, 0.9, 0.95, 0.99]))

# Thresholds below are only for LOOKING, not decisions.
q["ok_5pct"] = q["missing_frac"] <= 0.05
q["no_long_zero"] = q["longest_zero_run_h"] <= 24
q["both"] = q["ok_5pct"] & q["no_long_zero"]

summary = q.groupby("primaryspaceusage").agg(
    n=("missing_frac", "size"),
    sites=("site_id", "nunique"),
    median_missing=("missing_frac", "median"),
    ok_5pct=("ok_5pct", "sum"),
    no_long_zero=("no_long_zero", "sum"),
    both=("both", "sum"),
).sort_values("n", ascending=False)

print("\nper category:")
print(summary)