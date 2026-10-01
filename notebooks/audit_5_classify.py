import pandas as pd
from pathlib import Path

DATA = Path.home() / "Documents" / "building-data-genome-project-2" / "data"
meta = pd.read_csv(DATA / "metadata" / "metadata.csv")
elec = pd.read_csv(DATA / "meters" / "raw" / "electricity.csv",
                   index_col="timestamp", parse_dates=True)

info = meta[meta["building_id"].isin(elec.columns)].set_index("building_id")
ids = info.index[info["primaryspaceusage"].isin(["Education", "Office"])]


def longest_true_run(mask):
    """Longest run of consecutive True values in a numpy bool array.
    Returns (length, start_position)."""
    best_len, best_start = 0, -1
    cur_len, cur_start = 0, -1
    for i, v in enumerate(mask):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    return best_len, best_start


rows = []
for bid in ids:
    s = elec[bid]
    valid_pos = s.notna().to_numpy().nonzero()[0]
    if len(valid_pos) == 0:
        continue
    first_i, last_i = valid_pos[0], valid_pos[-1]   # first/last real reading
    zlen, zstart = longest_true_run((s == 0).to_numpy())
    zend = zstart + zlen - 1
    if zlen <= 168:
        kind = "no long run"
    elif zstart == first_i and zend == last_i:
        kind = "all zeros"
    elif zstart == first_i:
        kind = "leading"     # zeros from the first reading: late start
    elif zend == last_i:
        kind = "trailing"    # zeros until the last reading: meter died
    else:
        kind = "interior"    # zeros in the middle: outage or closure
    rows.append({"building_id": bid, "zero_run_h": zlen, "kind": kind})

res = pd.DataFrame(rows).set_index("building_id").join(
    info[["primaryspaceusage", "site_id"]])

print("1) kind of longest zero run, by category:")
print(pd.crosstab(res["primaryspaceusage"], res["kind"]))

print("\n2) by site:")
print(pd.crosstab(res["site_id"], res["kind"]))

print("\n3) length (hours) of interior long runs:")
print(res.loc[res["kind"] == "interior", "zero_run_h"].describe())

print("\n4) raw values, Oct-Dec 2016, for two buildings from the plot:")
for b in ["Cockatoo_education_Latrice", "Rat_education_Romana"]:
    w = elec.loc["2016-10-01":"2016-12-31", b]
    print(b, "| missing hours in window:", int(w.isna().sum()))
    print(w.describe().round(3))