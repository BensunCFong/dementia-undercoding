"""Arm 1b: LightGBM on arm 1's TF-IDF features.

Same features as arm 1, so the two differ only in the model reading them:
logistic regression adds up words one at a time; boosted trees can combine
them ("dementia" means less when "mother" is also in the note).

The number of trees is set by early stopping on an INNER slice of train
patients, never on validation, so validation stays an untouched yardstick.
Every setting was fixed on 2026-11-08, before any result. Scores validation,
logs parameters and metrics to MLflow; no model, vocabulary or text is saved.

Run:  .venv/bin/python -m dementia_undercoding.arm1b_lgbm
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from .arm1_tfidf import PARAMS, build_pipeline
from .cohort import load_config
from .evaluate import report, save_scores, score
from .splits import load_notes, validation_patients


def fit_and_score(notes: pd.DataFrame, label: str, cfg: dict, params: dict = PARAMS) -> dict:
    """Fit on train minus an inner slice, stop early on the slice, score validation.

    cfg is the arm1b block of config.yaml. Returns metrics, plus val_scores:
    one probability per validation note, in row order.
    """
    train = notes[notes["split"] == "train"]
    val = notes[notes["split"] == "validation"]
    inner_ids = validation_patients(train, cfg["inner_frac"], cfg["inner_seed"])
    is_inner = train["subject_id"].isin(inner_ids)
    fit, inner = train[~is_inner], train[is_inner]

    # A fresh, unfitted copy of arm 1's vectorizer: same settings, learned on `fit` only.
    vectorizer = build_pipeline(params).named_steps["tfidf"]
    start = time.perf_counter()
    x_fit = vectorizer.fit_transform(fit["text"])
    x_inner = vectorizer.transform(inner["text"])
    x_val = vectorizer.transform(val["text"])

    model = lgb.LGBMClassifier(
        n_estimators=cfg["max_trees"],
        learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"],
        min_child_samples=cfg["min_child_samples"],
        metric="average_precision",          # LightGBM's name for PR-AUC
        random_state=cfg["seed"],
        n_jobs=cfg.get("n_jobs") or max(1, (os.cpu_count() or 2) - 2),
        verbose=-1,
    )
    model.fit(
        x_fit, fit[label],
        eval_set=[(x_inner, inner[label])],
        callbacks=[lgb.early_stopping(cfg["patience"], verbose=False)],
    )
    fit_seconds = time.perf_counter() - start

    val_scores = model.predict_proba(x_val)[:, 1]
    return {
        "val": score(val[label], val_scores),
        "inner": score(inner[label], model.predict_proba(x_inner)[:, 1]),
        "train": score(fit[label], model.predict_proba(x_fit)[:, 1]),
        "trees": int(model.best_iteration_),
        "n_features": int(x_fit.shape[1]),
        "fit_seconds": fit_seconds,
        "val_scores": val_scores,
    }


def run(config_path: str | Path = "config/config.yaml", label: str = "dementia_core") -> dict:
    import mlflow  # imported here, not at the top: CI installs only core packages

    cfg = load_config(config_path)
    bcfg = cfg["arm1b"]
    notes = load_notes(config_path, columns=[label, "text"])
    notes = notes[notes["split"].isin(["train", "validation"])]   # drop test first

    result = fit_and_score(notes, label, bcfg)
    val_ids = notes.loc[notes["split"] == "validation", "hadm_id"]
    save_scores(val_ids, result["val_scores"],
                Path(cfg["paths"]["scores"]) / f"arm1b-lgbm-{label}.parquet")

    print(report(f"arm 1b LightGBM, {label}, validation ", result["val"]))
    print(report(f"arm 1b LightGBM, {label}, inner slice", result["inner"]))
    print(report(f"arm 1b LightGBM, {label}, train      ", result["train"]))
    print(f"stopped at {result['trees']} trees; {result['fit_seconds'] / 60:.1f} min")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=f"arm1b-lgbm-{label}"):
        mlflow.log_params({**PARAMS, **bcfg, "arm": "1b", "label": label, "scored_on": "validation"})
        for part in ["val", "inner", "train"]:
            mlflow.log_metrics({f"{part}_{k}": v for k, v in result[part].items()})
        mlflow.log_metrics({k: result[k] for k in ["trees", "n_features", "fit_seconds"]})
    return result


if __name__ == "__main__":
    run()