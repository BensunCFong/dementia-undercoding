"""The scorer, on labels and scores small enough to check by hand."""

import numpy as np
import pytest

from dementia_undercoding.evaluate import report, score


def test_perfect_ranking_scores_one():
    result = score([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.8, 0.9])
    assert result["pr_auc"] == pytest.approx(1.0)


def test_constant_scores_equal_prevalence():
    # A model that cannot tell anyone apart scores exactly the positive rate.
    y = [0, 0, 0, 0, 1]
    result = score(y, [0.5] * 5)
    assert result["pr_auc"] == pytest.approx(0.2)
    assert result["prevalence"] == pytest.approx(0.2)
    assert result["lift"] == pytest.approx(1.0)


def test_random_scores_land_near_prevalence():
    rng = np.random.default_rng(0)
    y = rng.random(200_000) < 0.04
    result = score(y, rng.random(200_000))
    assert result["pr_auc"] == pytest.approx(0.04, abs=0.005)


def test_worked_example_by_hand():
    # Ranked by score: 1, 0, 1, 0. Precision at each positive: 1/1, 2/3.
    # Average precision = mean(1, 2/3) = 0.8333.
    result = score([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])
    assert result["pr_auc"] == pytest.approx(5 / 6)


@pytest.mark.parametrize(
    "y_true, y_score",
    [
        ([0, 0, 0], [0.1, 0.2, 0.3]),          # no positives
        ([0, 1, 2], [0.1, 0.2, 0.3]),          # not binary
        ([0, 1], [0.1, 0.2, 0.3]),             # length mismatch
        ([0, 1, 0], [0.1, float("nan"), 0.3]),  # NaN score
    ],
)
def test_bad_input_raises(y_true, y_score):
    with pytest.raises(ValueError):
        score(y_true, y_score)


def test_report_shows_baseline():
    line = report("arm 1", score([0, 0, 0, 1], [0.1, 0.2, 0.3, 0.9]))
    assert "chance 0.250" in line
    assert "PR-AUC 1.000" in line