import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path.home() / "Documents" / "building-data-genome-project-2" / "data"
meta = pd.read_csv(DATA / "metadata" / "metadata.csv")
elec = pd.read_csv(DATA / "meters" / "raw" / "electricity.csv",
                   index_col="timestamp", parse_dates=True)

info = meta[meta["building_id"].isin(elec.columns)].set_index("building_id")
ids = info.index[info["primaryspaceusage"].isin(["Education", "Office"])]
T = 168  # trial value, only for looking


def long_dead_mask(dead, T):
    """True for every hour inside a dead stretch longer than T hours."""
    n = len(dead)
    out = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if dead[i]:
            j = i
            while j < n and dead[j]:
                j += 1
            if j - i > T:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


# check by hand first: expect [0 1 1 1 1 1 0 0 0 0]
demo = np.array([0, 1, 1, 1, 1, 1, 0, 1, 1, 0], dtype=bool)
print("demo:", long_dead_mask(demo, 3).astype(int))

test_pos = np.asarray(elec.index >= "2017-07-01")
rows = []
for bid in ids:
    s = elec[bid]
    dead = ((s == 0) | s.isna()).to_numpy()
    bad = long_dead_mask(dead, T)
    rows.append({"building_id": bid,
                 "masked_total": bad.mean(),
                 "masked_test": bad[test_pos].mean()})

res = pd.DataFrame(rows).set_index("building_id").join(
    info[["primaryspaceusage", "site_id"]])

res["A_strict"] = res["masked_total"] == 0
res["B1_test_clean"] = res["masked_test"] <= 0.05
res["B2_test_clean_and_half_data"] = (res["B1_test_clean"]
                                      & (res["masked_total"] <= 0.5))

print("\nbuildings kept, by category:")
print(res.groupby("primaryspaceusage").agg(
    n=("A_strict", "size"),
    A_strict=("A_strict", "sum"),
    B1=("B1_test_clean", "sum"),
    B2=("B2_test_clean_and_half_data", "sum")))

print("\nsites represented in B2:")
print(res[res["B2_test_clean_and_half_data"]]
      .groupby("primaryspaceusage")["site_id"].nunique())

print("\nby site (both categories together):")
print(res.groupby("site_id").agg(
    n=("A_strict", "size"),
    A_strict=("A_strict", "sum"),
    B2=("B2_test_clean_and_half_data", "sum")))