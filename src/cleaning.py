import numpy as np


def long_dead_mask(dead, T):
    """True for every hour inside a dead stretch longer than T hours."""
    n = len(dead)
    out = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if dead[i]:
            j = i
            while j < n and dead[j]:
                j += 1
            if j - i > T:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out