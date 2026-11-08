"""Shared scoring for every arm: PR-AUC with the chance baseline beside it.

score() gives one number per arm. paired_bootstrap() and compare() add the
uncertainty: confidence intervals by patient, and the difference between arms
measured on the same resampled patients. The subgroup breakdown comes later
and builds on score().
"""

from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
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


def save_scores(hadm_ids, y_score, path: str | Path) -> None:
    """Write one row per note, hadm_id and score, so arms can later be compared note by note.

    Numbers only: no text and no labels (labels join back from notes.parquet).
    Kept under data/, never sent to MLflow.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "hadm_id": np.asarray(hadm_ids),
        "score": np.asarray(y_score, dtype=float),
    }).to_parquet(path, index=False)


# --- uncertainty: patient-level paired bootstrap -------------------------------------


def _replicates(y, S, patient, n_patients, seed, n) -> np.ndarray:
    """n bootstrap PR-AUCs for every column of S. Returns shape (n, columns).

    Each replicate draws n_patients patients WITH replacement. A patient drawn
    twice counts twice, so each note is weighted by how often its patient was
    drawn: the same result as copying the rows, without copying them. Notes
    whose patient was not drawn (weight 0) are left out.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n, S.shape[1]))
    for b in range(n):
        drawn = np.bincount(rng.integers(0, n_patients, n_patients), minlength=n_patients)
        w = drawn[patient]                      # each note's weight = times its patient was drawn
        keep = w > 0
        for j in range(S.shape[1]):             # every arm on the SAME patients: that is the pairing
            out[b, j] = average_precision_score(y[keep], S[keep, j], sample_weight=w[keep])
    return out


def paired_bootstrap(
    y_true, scores: dict[str, np.ndarray], groups, *,
    n_boot: int = 2000, seed: int = 0, workers: int = 1, chunk: int = 100,
) -> pd.DataFrame:
    """Bootstrap PR-AUCs, one row per replicate, one column per arm.

    groups: one patient id per note. Patients are resampled, not notes,
    because one patient's notes copy history forward and are not independent.
    Every arm is scored on the same resampled patients in each replicate, so
    column differences give a paired comparison.

    The work is split into fixed chunks, each with its own seed spawned from
    `seed`, so the result is the same whatever `workers` is.
    """
    names = list(scores)
    for name in names:
        _check(y_true, scores[name])
    y = np.asarray(y_true).astype(int)
    S = np.column_stack([np.asarray(scores[name], dtype=float) for name in names])
    _, patient = np.unique(np.asarray(groups), return_inverse=True)   # patient ids -> 0..P-1
    n_patients = int(patient.max()) + 1

    sizes = [chunk] * (n_boot // chunk) + ([n_boot % chunk] if n_boot % chunk else [])
    seeds = np.random.SeedSequence(seed).spawn(len(sizes))
    tasks = [(y, S, patient, n_patients, s, n) for s, n in zip(seeds, sizes, strict=True)]
    if workers == 1:
        parts = [_replicates(*t) for t in tasks]
    else:
        with Pool(workers) as pool:
            parts = pool.starmap(_replicates, tasks)
    return pd.DataFrame(np.vstack(parts), columns=names)


def compare(
    y_true, scores: dict[str, np.ndarray], groups, reference: str, *,
    ci: float = 0.95, n_boot: int = 2000, seed: int = 0, workers: int = 1,
) -> pd.DataFrame:
    """One row per arm: PR-AUC with its interval, and its difference from `reference`.

    diff = arm minus reference, so a negative diff means the reference scored
    higher. Intervals are bootstrap percentiles: the middle `ci` share of the
    replicates. A difference interval that excludes 0 is a difference the
    resampling does not explain away.
    """
    reps = paired_bootstrap(y_true, scores, groups, n_boot=n_boot, seed=seed, workers=workers)
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    rows = []
    for name in scores:
        diffs = reps[name] - reps[reference]
        rows.append({
            "arm": name,
            "pr_auc": score(y_true, scores[name])["pr_auc"],
            "ci_low": reps[name].quantile(lo),
            "ci_high": reps[name].quantile(hi),
            "diff": score(y_true, scores[name])["pr_auc"] - score(y_true, scores[reference])["pr_auc"],
            "diff_low": diffs.quantile(lo),
            "diff_high": diffs.quantile(hi),
        })
    return pd.DataFrame(rows)
