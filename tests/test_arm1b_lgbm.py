"""Arm 1b on invented notes where the label is visible in the words."""

import numpy as np
import pandas as pd

from dementia_undercoding.arm1_tfidf import PARAMS
from dementia_undercoding.arm1b_lgbm import fit_and_score

# n_jobs=1: pytest loads every test file into one process, and LightGBM and
# PyTorch each bring their own copy of OpenMP. Two copies both starting worker
# threads deadlock on macOS. Single-threaded LightGBM here avoids that; the
# real run (no torch in the process) uses all cores but two.
CFG = {
    "inner_frac": 0.1, "inner_seed": 1, "max_trees": 200, "patience": 20,
    "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20, "seed": 1,
    "n_jobs": 1,
}


def fake_notes(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    positive = rng.random(n) < 0.2
    filler = ["chest pain", "fall at home", "shortness of breath", "fever"]
    text = [
        f"{rng.choice(filler)} {'history of dementia' if p else 'no cognitive issues'}"
        for p in positive
    ]
    split = np.where(np.arange(n) < n * 0.8, "train", "validation")
    return pd.DataFrame({
        "hadm_id": np.arange(n), "subject_id": np.arange(n) // 2,   # two notes per patient
        "text": text, "dementia_core": positive, "split": split,
    })


def test_learns_an_obvious_signal():
    result = fit_and_score(fake_notes(), "dementia_core", CFG, {**PARAMS, "min_df": 1})
    assert result["val"]["pr_auc"] > 0.95
    assert 1 <= result["trees"] <= CFG["max_trees"]


def test_returns_numbers_and_one_score_per_validation_note():
    notes = fake_notes()
    result = fit_and_score(notes, "dementia_core", CFG, {**PARAMS, "min_df": 1})
    for part in ["val", "inner", "train"]:
        assert all(isinstance(v, int | float) for v in result[part].values())
    assert len(result["val_scores"]) == (notes["split"] == "validation").sum()


def test_inner_slice_is_whole_train_patients():
    # The inner slice is drawn by patient, so no patient is on both sides.
    from dementia_undercoding.splits import validation_patients

    train = fake_notes().query("split == 'train'")
    inner_ids = set(validation_patients(train, CFG["inner_frac"], CFG["inner_seed"]))
    inner = train[train["subject_id"].isin(inner_ids)]
    rest = train[~train["subject_id"].isin(inner_ids)]
    assert inner_ids and set(inner["subject_id"]).isdisjoint(rest["subject_id"])


def test_extra_notes_scored_by_the_same_model():
    notes = fake_notes()
    extra = notes[notes["split"] == "validation"].reset_index(drop=True)
    result = fit_and_score(notes, "dementia_core", CFG, {**PARAMS, "min_df": 1}, extra=extra)
    assert np.allclose(result["extra_scores"], result["val_scores"])