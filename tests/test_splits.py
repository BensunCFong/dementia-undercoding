"""The validation split, on invented patients."""

import pandas as pd
import pytest

from dementia_undercoding.splits import assign_validation


@pytest.fixture
def notes():
    # 40 train patients with 1-3 admissions each, plus 5 test patients.
    rows = []
    hadm = 1000
    for subject in range(40):
        for _ in range(subject % 3 + 1):
            rows.append({"hadm_id": hadm, "subject_id": subject, "split": "train"})
            hadm += 1
    for subject in range(100, 105):
        rows.append({"hadm_id": hadm, "subject_id": subject, "split": "test"})
        hadm += 1
    return pd.DataFrame(rows)


def test_draws_ten_percent_of_train_patients(notes):
    out = assign_validation(notes, frac=0.1, seed=3)
    assert out.loc[out["split"] == "validation", "subject_id"].nunique() == 4


def test_test_rows_untouched(notes):
    out = assign_validation(notes, frac=0.1, seed=3)
    assert (out.loc[notes["split"] == "test", "split"] == "test").all()


def test_same_seed_same_patients_regardless_of_row_order(notes):
    a = assign_validation(notes, frac=0.1, seed=3)
    b = assign_validation(notes.sample(frac=1, random_state=0), frac=0.1, seed=3)
    val_a = set(a.loc[a["split"] == "validation", "subject_id"])
    val_b = set(b.loc[b["split"] == "validation", "subject_id"])
    assert val_a == val_b


def test_input_not_modified(notes):
    before = notes.copy()
    assign_validation(notes, frac=0.1, seed=3)
    pd.testing.assert_frame_equal(notes, before)