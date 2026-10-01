import pandas as pd
from pathlib import Path

DATA = Path.home() / "Documents" / "building-data-genome-project-2" / "data"
w = pd.read_csv(DATA / "weather" / "weather.csv", parse_dates=["timestamp"])
sel = pd.read_csv("data/processed/selected_buildings.csv")

print("columns:", w.columns.tolist())
print("shape:", w.shape)
print("time range:", w["timestamp"].min(), "to", w["timestamp"].max())
print("sites in weather:", w["site_id"].nunique())
print("selected sites missing from weather:",
      set(sel["site_id"]) - set(w["site_id"]))

ws = w[w["site_id"].isin(sel["site_id"].unique())]
print("\nshare missing by column (selected sites):")
print(ws.drop(columns=["timestamp", "site_id"]).isna().mean().round(3))

print("\nrows per site (full period would be 17544):")
print(ws.groupby("site_id").size())

print("\nduplicate (site, timestamp) rows:",
      ws.duplicated(["site_id", "timestamp"]).sum())