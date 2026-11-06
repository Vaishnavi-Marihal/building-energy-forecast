import sys
sys.path.insert(0, "src")

import pandas as pd
import pytest

from backtest_unseen_sites import site_folds, score_buildings

BUILDINGS = [{"id": f"{s}_{i}", "site_id": s}
             for s in ["A", "B", "C"] for i in (1, 2)]


def test_one_fold_per_site():
    assert [site for site, _, _ in site_folds(BUILDINGS)] == ["A", "B", "C"]


def test_held_out_site_never_in_training():
    for site, train, test in site_folds(BUILDINGS):
        assert test == [f"{site}_1", f"{site}_2"]
        assert all(not b.startswith(site + "_") for b in train)
        assert not set(train) & set(test)
        assert len(train) == 4


def test_every_building_is_tested_exactly_once():
    tested = [b for _, _, test in site_folds(BUILDINGS) for b in test]
    assert sorted(tested) == sorted(b["id"] for b in BUILDINGS)


def test_score_buildings_by_hand():
    # y=10 both hours; last week said 8 and 12 (error 2 and 2 -> MAE 2);
    # model said 9 and 11 (error 1 and 1 -> MAE 1). Ratio = 1 / 2 = 0.5.
    rows = pd.DataFrame({"building": ["x", "x"], "y": [10.0, 10.0],
                         "lag168": [8.0, 12.0], "lag24": [9.0, 9.0],
                         "pred": [9.0, 11.0]})
    out = score_buildings(rows, min_hours=1)
    assert out.loc["x", "n_hours"] == 2
    assert out.loc["x", "mae_model"] == pytest.approx(1.0)
    assert out.loc["x", "mae_lag168"] == pytest.approx(2.0)
    assert out.loc["x", "ratio"] == pytest.approx(0.5)


def test_score_buildings_skips_hours_without_baselines():
    # the second hour has no last-week value, so only the first hour is scored
    rows = pd.DataFrame({"building": ["x", "x"], "y": [10.0, 99.0],
                         "lag168": [8.0, float("nan")], "lag24": [9.0, 9.0],
                         "pred": [9.0, 0.0]})
    out = score_buildings(rows, min_hours=1)
    assert out.loc["x", "n_hours"] == 1


def test_score_buildings_drops_buildings_with_too_few_hours():
    rows = pd.DataFrame({"building": ["x"], "y": [10.0], "lag168": [8.0],
                         "lag24": [9.0], "pred": [9.0]})
    assert score_buildings(rows, min_hours=2).empty
