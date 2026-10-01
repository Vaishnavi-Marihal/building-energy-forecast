import pandas as pd


def seasonal_naive(df: pd.DataFrame, lag_hours: int) -> pd.DataFrame:
    """Prediction for hour t = actual value at t - lag_hours."""
    idx = df.index.to_series().diff().dropna()
    assert (idx == pd.Timedelta("1h")).all(), "index must be regular hourly"
    return df.shift(lag_hours)