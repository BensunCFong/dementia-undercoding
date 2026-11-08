"""Arm 1 on invented notes where the label is visible in the words."""

import numpy as np
import pandas as pd

from dementia_undercoding.arm1_tfidf import PARAMS, fit_and_score


def fake_notes(n=400, seed=0):
    rng = np.random.default_rng(seed)
    positive = rng.random(n) < 0.2
    filler = ["chest pain", "fall at home", "shortness of breath", "fever"]
    text = [
        f"{rng.choice(filler)} {'history of dementia' if p else 'no cognitive issues'}"
        for p in positive
    ]
    split = np.where(np.arange(n) < n * 0.8, "train", "validation")
    return pd.DataFrame({"text": text, "dementia_core": positive, "split": split})


def test_learns_an_obvious_signal():
    result = fit_and_score(fake_notes(), "dementia_core", {**PARAMS, "min_df": 1})
    assert result["val"]["pr_auc"] > 0.95
    assert result["val"]["prevalence"] < 0.3


def test_returns_metrics_only():
    result = fit_and_score(fake_notes(), "dementia_core", {**PARAMS, "min_df": 1})
    values = [*result["val"].values(), *result["train"].values(), result["vocab_size"]]
    assert all(isinstance(v, int | float) for v in values)