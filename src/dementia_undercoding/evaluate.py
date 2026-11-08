"""Shared scoring for every arm: PR-AUC with the chance baseline beside it.

Minimal version. Bootstrap confidence intervals and the subgroup breakdown
come later and build on score().
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def _check(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape or y_true.ndim != 1:
        raise ValueError(f"y_true {y_true.shape} and y_score {y_score.shape} must be the same 1-D shape.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must be 0/1 (or False/True).")
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        raise ValueError("y_true needs both positives and negatives; PR-AUC is undefined otherwise.")
    if np.isnan(y_score).any():
        raise ValueError("y_score contains NaN.")
    return y_true.astype(int), y_score


def score(y_true, y_score) -> dict:
    """PR-AUC (average precision) and the prevalence it must beat.

    prevalence is the PR-AUC of a model that ranks at random, so it is the
    floor for this split. lift = pr_auc / prevalence.
    """
    y_true, y_score = _check(y_true, y_score)
    pr_auc = float(average_precision_score(y_true, y_score))
    prevalence = float(y_true.mean())
    return {
        "pr_auc": pr_auc,
        "prevalence": prevalence,
        "lift": pr_auc / prevalence,
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }


def report(name: str, result: dict) -> str:
    """One line: the score with its baseline, so neither is read alone."""
    return (
        f"{name}: PR-AUC {result['pr_auc']:.3f}  "
        f"(chance {result['prevalence']:.3f}, lift {result['lift']:.1f}x, "
        f"{result['n_pos']:,} positives of {result['n']:,})"
    )