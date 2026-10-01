import numpy as np
import pandas as pd
from pathlib import Path

from cleaning import long_dead_mask

DATA = Path.home() / "Documents" / "building-data-genome-project-2" / "data"
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

CATEGORY = "Education"
T = 168
MAX_MASKED_TEST = 0.05
MAX_MASKED_TOTAL = 0.50
PER_SITE = 4
SEED = 0

meta = pd.read_csv(DATA / "metadata" / "metadata.csv")
elec = pd.read_csv(DATA / "meters" / "raw" / "electricity.csv",
                   index_col="timestamp", parse_dates=True)

info = meta[meta["building_id"].isin(elec.columns)].set_index("building_id")
ids = info.index[info["primaryspaceusage"] == CATEGORY]
test_pos = np.asarray(elec.index >= "2017-07-01")

masks, rows = {}, []
for bid in ids:
    s = elec[bid]
    dead = ((s == 0) | s.isna()).to_numpy()
    bad = long_dead_mask(dead, T)
    masks[bid] = bad
    rows.append({"building_id": bid,
                 "masked_total": bad.mean(),
                 "masked_test": bad[test_pos].mean()})

res = pd.DataFrame(rows).set_index("building_id").join(info[["site_id"]])
res["keep"] = ((res["masked_test"] <= MAX_MASKED_TEST)
               & (res["masked_total"] <= MAX_MASKED_TOTAL))
kept = res[res["keep"]]

selected = (kept.sample(frac=1, random_state=SEED)
            .groupby("site_id").head(PER_SITE)
            .sort_values(["site_id"]))

cleaned = elec[selected.index].copy()
for bid in selected.index:
    cleaned.loc[masks[bid], bid] = np.nan

cleaned.to_csv(OUT / "education_clean.csv")
selected.to_csv(OUT / "selected_buildings.csv")

print("category:", CATEGORY)
print("buildings in category:", len(res))
print("kept by rules:", len(kept), "| selected:", len(selected))
print("\nselected per site:")
print(selected.groupby("site_id").size())
print("\ncleaned shape:", cleaned.shape)
print("share NaN overall:", round(cleaned.isna().mean().mean(), 4))
print("share NaN in Jul-Dec 2017:",
      round(cleaned.loc["2017-07-01":].isna().mean().mean(), 4))