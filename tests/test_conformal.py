import sys
sys.path.insert(0, "src")

import numpy as np
from uncertainty import conformal_correction, coverage


def test_correction_known_answer():
    # n=19, alpha=0.1 -> k = ceil(20 * 0.9) = 18 -> 18th smallest = 18
    assert conformal_correction(np.arange(1, 20), 0.1) == 18


def test_coverage_known_answer():
    y = [1, 2, 3, 4]
    lo = [0, 0, 0, 0]
    hi = [1, 2, 3, 3]
    assert coverage(y, lo, hi) == 0.75


def test_calibrated_interval_reaches_target_coverage():
    rng = np.random.default_rng(0)
    y_cal, y_test = rng.normal(size=2000), rng.normal(size=2000)
    lo, hi = -0.5, 0.5                      # deliberately too narrow
    scores = np.maximum(lo - y_cal, y_cal - hi)
    q = conformal_correction(scores, 0.1)
    c = coverage(y_test, lo - q, hi + q)
    assert abs(c - 0.9) < 0.03