import sys
sys.path.insert(0, "src")

from monitor import summarise

DAY = "2017-09-19"


def test_median_over_buildings_and_pooled_coverage_by_hand():
    # (building, day, hours scored, model MAE, last-week MAE, hours inside interval)
    rows = [("b1", DAY, 24, 1.0, 2.0, 24),   # ratio 0.5
            ("b2", DAY, 24, 3.0, 2.0, 12),   # ratio 1.5 (one bad building)
            ("b3", DAY, 24, 1.0, 1.0, 24)]   # ratio 1.0
    (day,) = summarise(rows)
    assert day["buildings"] == 3
    assert day["median_ratio"] == 1.0                 # median of 0.5, 1.5, 1.0
    assert abs(day["coverage"] - 60 / 72) < 1e-12     # (24 + 12 + 24) of 72 hours
    assert day["flag"] is False                       # 1.0 is not above 1.0; 0.83 is not below 0.80


def test_bad_day_is_flagged_and_good_day_is_not():
    rows = [("b1", "2017-09-19", 2, 0.5, 1.0, 2),     # ratio 0.5, coverage 1.0
            ("b1", "2017-09-20", 2, 17.5, 12.5, 0)]   # ratio 1.4, coverage 0.0
    good, bad = summarise(rows, min_hours=1)
    assert (good["day"], good["flag"]) == ("2017-09-19", False)
    assert (bad["day"], bad["flag"]) == ("2017-09-20", True)
    assert abs(bad["median_ratio"] - 1.4) < 1e-12


def test_building_days_with_too_few_hours_are_ignored():
    assert summarise([("b1", DAY, 5, 1.0, 2.0, 5)]) == []   # default minimum is 12 hours


def test_zero_baseline_error_is_ignored_not_divided_by():
    assert summarise([("b1", DAY, 24, 1.0, 0.0, 24)]) == []
