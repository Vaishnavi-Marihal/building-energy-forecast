import pandas as pd
from pathlib import Path

DATA = Path.home() / "Documents" / "building-data-genome-project-2" / "data"

meta = pd.read_csv(DATA / "metadata" / "metadata.csv")
elec = pd.read_csv(DATA / "meters" / "raw" / "electricity.csv",
                   index_col="timestamp", parse_dates=True)

print("metadata shape:", meta.shape)
print("electricity shape:", elec.shape)
print("time range:", elec.index.min(), "to", elec.index.max())

print("\nmetadata 'electricity' flag:")
print(meta["electricity"].value_counts(dropna=False))

has_elec = meta[meta["building_id"].isin(elec.columns)]
print("\nbuildings with a column in the electricity file:", len(has_elec))

print("\nby primaryspaceusage:")
print(has_elec["primaryspaceusage"].value_counts(dropna=False))