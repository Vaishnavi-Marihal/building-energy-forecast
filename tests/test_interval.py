import sys
sys.path.insert(0, "src")

import numpy as np
from uncertainty import make_interval


def test_interval_contains_point_and_is_nonnegative():
    pred = np.array([1.0, 5.0, 0.0])
    lo = np.array([2.0, 3.0, -1.0])
    hi = np.array([3.0, 4.0, 0.5])
    lower, upper = make_interval(pred, lo, hi, 0.1)
    assert (lower <= pred).all() and (pred <= upper).all()
    assert (lower >= 0).all()