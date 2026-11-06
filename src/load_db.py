"""Load the demo data into PostgreSQL and check it against the model's own numbers.

Run from the repo root:  python src/load_db.py
Connection string comes from DATABASE_URL (defaults to the local Docker database).
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parent.parent
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:devpass@localhost:5432/energy")
TRAIN_END = "2017-06-01 00:00:00"      # scale = mean of everything before this


def main():
    meta = json.loads((ROOT / "deploy/model/meta.json").read_text())
    df = pd.read_csv(ROOT / "deploy/dataset/education_clean.csv",
                     index_col="timestamp", parse_dates=True)
    schema = (ROOT / "sql/schema.sql").read_text()
    timestamps = df.index.to_pydatetime()

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(schema)

            cur.executemany(
                "INSERT INTO buildings (building_id, site_id, scale) VALUES (%s, %s, %s)",
                [(b["id"], b["site_id"], meta["scale"][b["id"]])
                 for b in meta["buildings"]])

            # COPY is Postgres's bulk-load path: much faster than 982k INSERTs
            with cur.copy("COPY readings (building_id, ts, kwh) FROM STDIN") as copy:
                for building in df.columns:
                    for ts, value in zip(timestamps, df[building].to_numpy()):
                        copy.write_row((building, ts, None if pd.isna(value) else float(value)))

            n_b = cur.execute("SELECT COUNT(*) FROM buildings").fetchone()[0]
            n_r, n_real = cur.execute("SELECT COUNT(*), COUNT(kwh) FROM readings").fetchone()
            print(f"buildings: {n_b}")
            print(f"readings:  {n_r}  (real: {n_real}, masked/NULL: {n_r - n_real})")

            # Check: for every building, the SQL mean before June 2017 must equal
            # the scale stored in meta.json (the number the model was trained with).
            cur.execute("""
                SELECT b.building_id, AVG(r.kwh) AS sql_mean, b.scale
                FROM buildings b
                JOIN readings r ON r.building_id = b.building_id
                WHERE r.ts < %s
                GROUP BY b.building_id, b.scale
                ORDER BY b.building_id""", (TRAIN_END,))
            checked = cur.fetchall()

    diffs = [abs(sql_mean - scale) for _, sql_mean, scale in checked]
    bad = [(b, m, s) for (b, m, s), d in zip(checked, diffs) if d > 1e-9]
    print(f"\nbuildings checked: {len(checked)}")
    print(f"largest |SQL mean - scale|: {max(diffs):.3e}")
    if bad or len(checked) != n_b:
        print("MISMATCH:", bad[:5])
        sys.exit(1)
    print("all buildings match meta.json")


if __name__ == "__main__":
    main()