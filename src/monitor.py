"""Daily monitoring report: compare the forecasts the API served with what really happened.

For every building and day it measures the model's error against the error of
"same hour last week", and how often the actual value fell inside the interval.
Per day it reports the MEDIAN ratio over buildings (so one huge building cannot
dominate) and the share of hours inside the interval. A day is flagged when the
model is worse than last week's value, or the intervals cover too little.

Run from the repo root:  python src/monitor.py
Exit code 1 if any day is flagged, so a scheduler could raise an alert.
"""
import os
import statistics
import sys

import psycopg

DEFAULT_DATABASE_URL = "postgresql://postgres:devpass@localhost:5432/energy"
RATIO_LIMIT = 1.0       # flag a day if the model is worse than "same hour last week"
COVERAGE_LIMIT = 0.80   # intervals are meant to cover 90%; flag a day below 80%
MIN_HOURS = 12          # ignore a building-day with fewer scored hours (too noisy)

QUERY = """
    SELECT f.building_id,
           f.ts::date AS day,
           COUNT(*) AS n_scored,
           AVG(ABS(f.forecast_kwh - r.kwh)) AS mae_model,
           AVG(ABS(r.kwh - last_week.kwh)) AS mae_lag168,
           SUM((r.kwh BETWEEN f.lower_kwh AND f.upper_kwh)::int) AS n_covered
    FROM forecasts f
    JOIN readings r         ON r.building_id = f.building_id AND r.ts = f.ts
    JOIN readings last_week ON last_week.building_id = f.building_id
                           AND last_week.ts = f.ts - INTERVAL '168 hours'
    WHERE r.kwh IS NOT NULL AND last_week.kwh IS NOT NULL
    GROUP BY f.building_id, day
"""


def summarise(rows, min_hours=MIN_HOURS):
    """rows: (building_id, day, n_scored, mae_model, mae_lag168, n_covered) per building-day.
    Returns one dict per day: buildings, median_ratio, coverage, flag."""
    by_day = {}
    for building, day, n, mae_model, mae_lag, n_covered in rows:
        if n < min_hours or not mae_lag > 0:   # too few hours, or no baseline error to divide by
            continue
        d = by_day.setdefault(day, {"ratios": [], "hours": 0, "covered": 0})
        d["ratios"].append(float(mae_model) / float(mae_lag))
        d["hours"] += n
        d["covered"] += n_covered
    report = []
    for day in sorted(by_day):
        d = by_day[day]
        ratio = statistics.median(d["ratios"])
        coverage = d["covered"] / d["hours"]
        report.append({"day": day, "buildings": len(d["ratios"]),
                       "median_ratio": ratio, "coverage": coverage,
                       "flag": ratio > RATIO_LIMIT or coverage < COVERAGE_LIMIT})
    return report


def main():
    url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    with psycopg.connect(url) as conn:
        rows = conn.execute(QUERY).fetchall()
    report = summarise(rows)
    if not report:
        print("No scored forecasts yet: serve some forecasts first (the readings for those days must exist).")
        return 0
    print(f"day         buildings  median_ratio  coverage  verdict "
          f"(flag if ratio > {RATIO_LIMIT} or coverage < {COVERAGE_LIMIT})")
    for r in report:
        print(f"{r['day']}  {r['buildings']:>9}  {r['median_ratio']:>12.2f}  "
              f"{r['coverage']:>8.2f}  {'FLAG' if r['flag'] else 'ok'}")
    flagged = sum(r["flag"] for r in report)
    print(f"\n{len(report)} days checked, {flagged} flagged.")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
