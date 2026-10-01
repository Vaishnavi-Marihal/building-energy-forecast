import numpy as np
import pandas as pd
from baselines import seasonal_naive

df = pd.read_csv("data/processed/education_clean.csv",
                 index_col="timestamp", parse_dates=True)

p24 = seasonal_naive(df, 24)
p168 = seasonal_naive(df, 168)

rows = []
for m in pd.period_range("2017-07", "2017-12", freq="M"):
    test = df.loc[m.start_time:m.end_time]
    # scale: each building's mean load BEFORE the test month
    train_mean = df.loc[:m.start_time - pd.Timedelta(hours=1)].mean()

    a = test
    b24 = p24.loc[test.index]
    b168 = p168.loc[test.index]
    valid = a.notna() & b24.notna() & b168.notna()   # score both on same hours

    mae24 = (a - b24).abs().where(valid).mean()
    mae168 = (a - b168).abs().where(valid).mean()
    n = valid.sum()
    ok = n > 24 * 5                                  # need >= 5 scored days

    rows.append({
        "month": str(m),
        "buildings": int(ok.sum()),
        "scored_share": round(valid.to_numpy().mean(), 3),
        "median_nMAE_lag24": round((mae24 / train_mean)[ok].median(), 4),
        "median_nMAE_lag168": round((mae168 / train_mean)[ok].median(), 4),
        "median_ratio_24_over_168": round((mae24 / mae168)[ok].median(), 3),
        "pct_buildings_lag24_better": round(((mae24 < mae168)[ok]).mean(), 3),
    })

print(pd.DataFrame(rows).to_string(index=False))