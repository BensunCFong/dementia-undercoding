"""Carve the validation split out of train, by patient.

notes.parquet carries split = 'train' or 'test'. Every model trains on train
minus validation and is scored on validation; test stays untouched until the
final comparison. The draw is by subject_id, so one patient's admissions all
land on the same side (notes copy history forward from earlier admissions).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cohort import load_config


def validation_patients(train: pd.DataFrame, frac: float, seed: int) -> np.ndarray:
    """Return the subject_ids drawn for validation from the train rows.

    Patients are listed in order of first appearance after sorting by hadm_id,
    which is the order the 2026-10-05 seed comparison saw (notes.parquet is
    written ORDER BY hadm_id). The sort is explicit so the draw does not depend
    on how the rows happened to be loaded.
    """
    patients = train.sort_values("hadm_id")["subject_id"].unique()
    rng = np.random.default_rng(seed)
    return rng.choice(patients, size=round(frac * len(patients)), replace=False)


def assign_validation(notes: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Return a copy of notes with the drawn train patients relabelled 'validation'."""
    out = notes.copy()
    is_train = out["split"] == "train"
    val_ids = validation_patients(out[is_train], frac, seed)
    out.loc[is_train & out["subject_id"].isin(val_ids), "split"] = "validation"

    sides = out.groupby("subject_id")["split"].nunique()
    if (sides > 1).any():
        raise ValueError(f"{int((sides > 1).sum())} patients appear in more than one split.")
    return out


def load_notes(
    config_path: str | Path = "config/config.yaml", columns: list[str] | None = None
) -> pd.DataFrame:
    """Read notes.parquet and apply the validation split from config.

    columns: read only these (hadm_id, subject_id and split are always added),
    so a summary can skip the text column entirely.
    """
    cfg = load_config(config_path)
    if columns is not None:
        columns = list(dict.fromkeys(["hadm_id", "subject_id", "split", *columns]))
    notes = pd.read_parquet(cfg["paths"]["notes"], columns=columns)
    val = cfg["split"]["validation"]
    return assign_validation(notes, val["frac"], val["seed"])