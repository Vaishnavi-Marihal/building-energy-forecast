import numpy as np


def conformal_correction(scores, alpha=0.1):
    """Split-conformal correction: the ceil((n+1)(1-alpha))-th smallest score."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(np.ceil(round((n + 1) * (1 - alpha), 9)))
    if k > n:
        return np.inf          # not enough calibration data
    return float(s[k - 1])


def coverage(y, lo, hi):
    """Share of true values inside [lo, hi]."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))

def make_interval(pred, lo_raw, hi_raw, q):
    """Calibrated band in the model's scaled units: widen by q, make sure it
    contains the point forecast, and never go below zero load."""
    lo = np.minimum(lo_raw, hi_raw) - q
    hi = np.maximum(lo_raw, hi_raw) + q
    lower = np.maximum(np.minimum(lo, pred), 0.0)
    upper = np.maximum(hi, pred)
    return lower, upper