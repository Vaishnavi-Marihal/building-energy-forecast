import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path.home() / "Documents" / "building-data-genome-project-2" / "data"


def load_site_temperature(index):
    w = pd.read_csv(DATA / "weather" / "weather.csv", parse_dates=["timestamp"])
    t = w.pivot(index="timestamp", columns="site_id", values="airTemperature")
    t = t.reindex(index)                                   # missing hours -> NaN
    return t.interpolate(limit=3, limit_area="inside")     # fill gaps <= 3 hours


def ar1_noise(index, sites, sigma, phi=0.95, seed=0):
    """Forecast-error stand-in: AR(1) noise per site with std = sigma."""
    rng = np.random.default_rng(seed)
    n = len(index)
    eps = rng.normal(size=(n, len(sites)))
    out = np.zeros((n, len(sites)))
    out[0] = sigma * eps[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + np.sqrt(1 - phi ** 2) * sigma * eps[i]
    return pd.DataFrame(out, index=index, columns=sites)


def per_building(site_temp, building_site):
    """One column per building, taken from that building's site."""
    return pd.DataFrame({b: site_temp[s] for b, s in building_site.items()})