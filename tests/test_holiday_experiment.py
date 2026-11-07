import sys
sys.path.insert(0, "src")

import pandas as pd

from experiment_holiday import holiday_flag


def flags(start, hours):
    return list(holiday_flag(pd.date_range(start, periods=hours, freq="h")))


def test_labor_day_2017_is_flagged_for_exactly_its_24_hours():
    # Sun Sep 3, Mon Sep 4 (Labor Day), Tue Sep 5
    assert flags("2017-09-03", 72) == [0] * 24 + [1] * 24 + [0] * 24


def test_christmas_is_flagged_but_christmas_eve_is_not():
    assert flags("2017-12-24", 48) == [0] * 24 + [1] * 24


def test_observed_date_is_used_when_a_holiday_falls_on_a_weekend():
    # Veterans Day 2017 is Saturday Nov 11; it is observed on Friday Nov 10
    assert flags("2017-11-10", 48) == [1] * 24 + [0] * 24
