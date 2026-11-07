import sys
sys.path.insert(0, "src")

import pandas as pd

from experiment_holiday_intervals import interval_stats


def test_coverage_and_width_by_hand():
    # y = 1, 5, 9 against intervals [0,2], [4,6], [10,12]: the first two are inside, the third is not
    rows = pd.DataFrame({"y": [1.0, 5.0, 9.0], "lag168": [1.0] * 3, "lag24": [1.0] * 3,
                         "lower": [0.0, 4.0, 10.0], "upper": [2.0, 6.0, 12.0]})
    cov, width = interval_stats(rows)
    assert abs(cov - 2 / 3) < 1e-12
    assert width == 2.0


def test_hours_without_baselines_are_not_scored():
    # the second hour has no last-week value, so only the first hour counts (and it is inside)
    rows = pd.DataFrame({"y": [1.0, 50.0], "lag168": [1.0, float("nan")], "lag24": [1.0, 1.0],
                         "lower": [0.0, 0.0], "upper": [2.0, 2.0]})
    cov, _ = interval_stats(rows)
    assert cov == 1.0
