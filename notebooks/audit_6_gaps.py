import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path.home() / "Documents" / "building-data-genome-project-2" / "data"
meta = pd.read_csv(DATA / "metadata" / "metadata.csv")
elec = pd.read_csv(DATA / "meters" / "raw" / "electricity.csv",
                   index_col="timestamp", parse_dates=True)

info = meta[meta["building_id"].isin(elec.columns)].set_index("building_id")
ids = info.index[info["primaryspaceusage"].isin(["Education", "Office"])]


def longest_run(mask):
    """Longest stretch of consecutive True values in a numpy bool array."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


# check by hand: expect 3, then 7
demo = pd.Series([0, 0, 0, np.nan, 0, 0, 0])
print("demo, zeros only      :", longest_run((demo == 0).to_numpy()))
print("demo, zeros or missing:",
      longest_run(((demo == 0) | demo.isna()).to_numpy()))

rows = []
for bid in ids:
    s = elec[bid]
    rows.append({
        "building_id": bid,
        "zeros_only": longest_run((s == 0).to_numpy()),
        "zeros_or_missing": longest_run(((s == 0) | s.isna()).to_numpy()),
    })
res = pd.DataFrame(rows).set_index("building_id").join(
    info[["primaryspaceusage"]])

table = []
for T in [24, 72, 168, 336, 720]:
    for cat in ["Education", "Office"]:
        r = res[res["primaryspaceusage"] == cat]
        table.append({"T_hours": T, "category": cat, "n": len(r),
                      "zeros_only": int((r["zeros_only"] > T).sum()),
                      "zeros_or_missing": int((r["zeros_or_missing"] > T).sum())})
print("\nbuildings with a stretch longer than T hours:")
print(pd.DataFrame(table).to_string(index=False))

newly = (res["zeros_only"] <= 168) & (res["zeros_or_missing"] > 168)
print("\ncalled fine by zeros_only at T=168 but flagged by zeros_or_missing:")
print(res[newly].groupby("primaryspaceusage").size())