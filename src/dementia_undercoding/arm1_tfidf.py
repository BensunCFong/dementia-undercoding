"""Arm 1: TF-IDF features + logistic regression.

Trains on train (validation already carved out), scores on validation, logs
parameters and metrics to MLflow. Test rows are dropped before anything else
runs, so no model ever sees them. Nothing derived from note text is logged or
saved: no model, no vocabulary, no predictions.

Run:  .venv/bin/python -m dementia_undercoding.arm1_tfidf
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .cohort import load_config
from .evaluate import report, save_scores, score
from .splits import load_notes

# Fixed, untuned settings. Tuning them on validation and then reporting
# validation would flatter the baseline; this is a first honest reference.
PARAMS = {
    "ngram_range": (1, 2),     # single words and adjacent pairs ("memory loss")
    "min_df": 5,               # ignore terms in fewer than 5 training notes
    "max_features": 200_000,   # keep the 200k most frequent terms that pass min_df
    "sublinear_tf": True,      # 1 + log(count): the 10th "confused" adds less than the 1st
    "C": 1.0,                  # inverse regularisation strength (sklearn default)
}


def build_pipeline(params: dict = PARAMS) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=params["ngram_range"],
            min_df=params["min_df"],
            max_features=params["max_features"],
            sublinear_tf=params["sublinear_tf"],
            lowercase=True,
            dtype=np.float32,
        )),
        ("lr", LogisticRegression(C=params["C"], solver="liblinear", max_iter=1000)),
    ])


def fit_and_score(notes: pd.DataFrame, label: str, params: dict = PARAMS) -> dict:
    """Fit on split == 'train', score on split == 'validation'.

    Returns metrics, plus val_scores: one probability per validation note, in row order.
    """
    train = notes[notes["split"] == "train"]
    val = notes[notes["split"] == "validation"]

    pipe = build_pipeline(params)
    start = time.perf_counter()
    pipe.fit(train["text"], train[label])
    fit_seconds = time.perf_counter() - start

    # Column 1 of predict_proba is P(label = 1): the score we rank by.
    val_scores = pipe.predict_proba(val["text"])[:, 1]
    val_result = score(val[label], val_scores)
    train_result = score(train[label], pipe.predict_proba(train["text"])[:, 1])
    return {
        "val": val_result,
        "train": train_result,
        "vocab_size": len(pipe.named_steps["tfidf"].vocabulary_),
        "fit_seconds": fit_seconds,
        "val_scores": val_scores,
    }


def run(config_path: str | Path = "config/config.yaml", label: str = "dementia_core") -> dict:
    import mlflow  # imported here, not at the top: CI installs only core packages

    cfg = load_config(config_path)
    notes = load_notes(config_path, columns=[label, "text"])
    notes = notes[notes["split"].isin(["train", "validation"])]   # drop test first

    result = fit_and_score(notes, label)
    val_ids = notes.loc[notes["split"] == "validation", "hadm_id"]
    save_scores(val_ids, result["val_scores"],
                Path(cfg["paths"]["scores"]) / f"arm1-tfidf-lr-{label}.parquet")
    
    print(report(f"arm 1 TF-IDF+LR, {label}, validation", result["val"]))
    print(report(f"arm 1 TF-IDF+LR, {label}, train     ", result["train"]))

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=f"arm1-tfidf-lr-{label}"):
        mlflow.log_params({**PARAMS, "arm": "1", "label": label, "scored_on": "validation"})
        mlflow.log_metrics({f"val_{k}": v for k, v in result["val"].items()})
        mlflow.log_metrics({f"train_{k}": v for k, v in result["train"].items()})
        mlflow.log_metrics({"vocab_size": result["vocab_size"], "fit_seconds": result["fit_seconds"]})
    return result


if __name__ == "__main__":
    run()