import pandas as pd
from pathlib import Path

DATA = Path.home() / "Documents" / "building-data-genome-project-2" / "data"
meta = pd.read_csv(DATA / "metadata" / "metadata.csv")
elec = pd.read_csv(DATA / "meters" / "raw" / "electricity.csv",
                   index_col="timestamp", parse_dates=True)

info = meta[meta["building_id"].isin(elec.columns)].set_index("building_id")
ids = info.index[info["primaryspaceusage"].isin(["Education", "Office"])]
site_of = info.loc[ids, "site_id"]

dead = ((elec[ids] == 0) | elec[ids].isna()).astype(int)
# 1 if the building was dead for ALL 24 hours of that day
dead_day = dead.resample("D").min()
# share of each site's buildings that were dead that day
site_share = dead_day.T.groupby(site_of).mean().T


def ranges(flag):
    out, start, prev = [], None, None
    for day, v in flag.items():
        if v and start is None:
            start = day
        if (not v) and start is not None:
            out.append((start, prev))
            start = None
        prev = day
    if start is not None:
        out.append((start, prev))
    return [(a.date(), b.date(), (b - a).days + 1) for a, b in out]


print("date ranges (>=7 days) where >=50% of a site's buildings were dead:\n")
for site in sorted(site_share.columns):
    n = int((site_of == site).sum())
    r = [x for x in ranges(site_share[site] >= 0.5) if x[2] >= 7]
    print(f"{site} ({n} buildings):", r if r else "none")