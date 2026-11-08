"""The patient-level paired bootstrap, on invented labels and scores."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from dementia_undercoding.evaluate import compare, paired_bootstrap, score


def fake(n_patients=300, notes_each=3, seed=0):
    """Invented validation set: notes_each notes per patient, two arms of different skill."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_patients), notes_each)
    y = np.repeat(rng.random(n_patients) < 0.2, notes_each).astype(int)
    good = y + rng.normal(0, 0.6, len(y))
    weak = y + rng.normal(0, 1.5, len(y))
    return y, {"good": good, "weak": weak}, groups


def test_weights_match_copied_rows():
    # The trick _replicates relies on: weight 2 gives the same PR-AUC as the row appearing twice.
    y, scores, _ = fake()
    s = scores["good"]
    w = np.random.default_rng(1).integers(0, 3, len(y))
    keep = w > 0
    copied = average_precision_score(np.repeat(y, w), np.repeat(s, w))
    weighted = average_precision_score(y[keep], s[keep], sample_weight=w[keep])
    assert weighted == pytest.approx(copied)


def test_same_seed_same_answer_whatever_the_workers():
    y, scores, groups = fake()
    one = paired_bootstrap(y, scores, groups, n_boot=40, seed=3, workers=1, chunk=10)
    two = paired_bootstrap(y, scores, groups, n_boot=40, seed=3, workers=2, chunk=10)
    assert one.shape == (40, 2)
    assert np.array_equal(one.to_numpy(), two.to_numpy())


def test_an_arm_against_itself_differs_by_exactly_zero():
    y, scores, groups = fake()
    table = compare(y, {"a": scores["good"], "b": scores["good"].copy()}, groups, "a", n_boot=50)
    assert (table[["diff", "diff_low", "diff_high"]].to_numpy() == 0).all()


def test_point_estimate_is_the_plain_score():
    y, scores, groups = fake()
    table = compare(y, scores, groups, "good", n_boot=50).set_index("arm")
    assert table.loc["weak", "pr_auc"] == pytest.approx(score(y, scores["weak"])["pr_auc"])
    assert table.loc["weak", "diff"] < 0                     # weak arm is worse than the reference
    assert table.loc["weak", "ci_low"] <= table.loc["weak", "pr_auc"] <= table.loc["weak", "ci_high"]


def test_clear_difference_excludes_zero():
    y, scores, groups = fake(n_patients=1000)
    table = compare(y, scores, groups, "good", n_boot=200).set_index("arm")
    assert table.loc["weak", "diff_high"] < 0


def test_resampling_patients_widens_the_interval():
    # Five identical notes per patient carry the information of one note. Treating them as
    # independent (one group per note) would make the interval look narrower than it is.
    y1, scores1, _ = fake(n_patients=200, notes_each=1)
    y = np.repeat(y1, 5)
    scores = {name: np.repeat(s, 5) for name, s in scores1.items()}
    groups = np.repeat(np.arange(200), 5)
    by_patient = compare(y, scores, groups, "good", n_boot=200).set_index("arm")
    by_note = compare(y, scores, np.arange(len(y)), "good", n_boot=200).set_index("arm")
    width = lambda t: t.loc["weak", "ci_high"] - t.loc["weak", "ci_low"]  # noqa: E731
    assert width(by_patient) > 1.5 * width(by_note)
