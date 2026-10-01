import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from weather import ar1_noise


def test_noise_has_requested_std_and_autocorrelation():
    idx = pd.date_range("2016-01-01", periods=20000, freq="h")
    e = ar1_noise(idx, ["a"], sigma=2.0, phi=0.9, seed=1)["a"].to_numpy()
    assert abs(e.std() - 2.0) < 0.2
    assert abs(np.corrcoef(e[:-1], e[1:])[0, 1] - 0.9) < 0.03