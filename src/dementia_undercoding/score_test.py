"""Score the test set, once, with the frozen arms (STATUS item 13; rules decided 2026-11-09).

Rules:
  1. Arms 1 and 1b refit on exactly what they used for validation (train minus validation,
     same settings and seeds). LoRA loads its saved adapters. Before any test score is
     written, each refit or reloaded model re-scores validation and must match its saved
     validation scores (within test_scoring.reproduce_tol), so test is scored by the model
     that was validated. Arm 0 has no fitting, so nothing to reproduce.
  2. Scores only: hadm_id + score files under data/. No metric is computed, printed or logged.
  3. Locked: refuses to run unless test_scoring.allowed is true in config.yaml.
  4. Scored once: an arm whose test score file exists is refused, not overwritten.

One arm per process: LightGBM and PyTorch each bring their own OpenMP, and two copies in
one process can deadlock on macOS.

Run (after unlocking):
  for arm in arm0 arm1 arm1b arm2-main arm2-main-head_tail; do
    HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -m dementia_undercoding.score_test $arm || break
  done
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .cohort import load_config
from .evaluate import save_scores
from .splits import load_notes

LABEL = "dementia_core"

# A scorer takes (dev notes, test notes, cfg) and returns {score file stem: (validation
# scores or None, test scores)}, each in the row order of the notes it was given.
Scorer = Callable[[pd.DataFrame, pd.DataFrame, dict], dict[str, tuple[np.ndarray | None, np.ndarray]]]


def check_unlocked(cfg: dict) -> None:
    if cfg.get("test_scoring", {}).get("allowed") is not True:
        raise PermissionError("test scoring is locked: set test_scoring.allowed to true in "
                              "config.yaml and commit that change first")


def scored_path(cfg: dict, stem: str, label: str = LABEL) -> Path:
    return Path(cfg["paths"]["scores"]) / "test" / f"{stem}-{label}.parquet"


def check_reproduced(cfg: dict, stem: str, val_ids, val_scores: np.ndarray, tol: float,
                     label: str = LABEL) -> float:
    """Largest gap between these validation scores and the saved ones. Fails above tol."""
    saved = pd.read_parquet(Path(cfg["paths"]["scores"]) / f"{stem}-{label}.parquet")
    now = pd.DataFrame({"hadm_id": np.asarray(val_ids), "now": np.asarray(val_scores, dtype=float)})
    both = now.merge(saved, on="hadm_id", how="outer", validate="one_to_one")
    if both[["now", "score"]].isna().any().any():
        raise ValueError(f"{stem}: validation notes differ from the saved validation scores")
    gap = float((both["now"] - both["score"]).abs().max())
    if gap > tol:
        raise ValueError(f"{stem}: validation scores differ from the saved ones by up to {gap:.2e} "
                         f"(allowed {tol:.0e}): not the validated model, nothing written")
    return gap


# --- the scorers ---------------------------------------------------------------------


def score_arm0(dev: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> dict:
    from .arm0_regex import first_512_tokens, score_all

    acfg = cfg["arm0"]
    texts = test["text"].tolist()
    full = score_all(texts, acfg["window"], acfg["uncertain_weight"])
    cut = score_all(first_512_tokens(texts, cfg["lora"]["model"], cfg["lora"]["max_len"]),
                    acfg["window"], acfg["uncertain_weight"])
    return {
        "arm0-regex": (None, full[:, 0]),
        "arm0_drugs-regex": (None, full[:, 0] + full[:, 1]),
        "arm0_first512-regex": (None, cut[:, 0]),
    }


def score_arm1(dev: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> dict:
    from .arm1_tfidf import fit_and_score

    result = fit_and_score(dev, LABEL, extra=test)
    return {"arm1-tfidf-lr": (result["val_scores"], result["extra_scores"])}


def score_arm1b(dev: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> dict:
    from .arm1b_lgbm import fit_and_score

    result = fit_and_score(dev, LABEL, cfg["arm1b"], extra=test)
    return {"arm1b-lgbm": (result["val_scores"], result["extra_scores"])}


def lora_scorer(name: str) -> Scorer:
    """Scorer for the saved LoRA adapter data/derived/arm2/<name> (main or main-head_tail)."""

    def scorer(dev: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> dict:
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer

        from .arm2_lora import load_base, predict

        lcfg = cfg["lora"]
        head = lcfg["head_tail"]["head"] if name.endswith("head_tail") else None
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(lcfg["model"])
        model = PeftModel.from_pretrained(load_base(lcfg["model"]), Path(lcfg["out_dir"]) / name)
        val = dev[dev["split"] == "validation"]
        val_scores = predict(model, tokenizer, val["text"].tolist(), device=device,
                             max_len=lcfg["max_len"], head=head)
        test_scores = predict(model, tokenizer, test["text"].tolist(), device=device,
                              max_len=lcfg["max_len"], head=head)
        return {f"arm2-lora-{name}": (val_scores, test_scores)}

    return scorer


ARMS: dict[str, tuple[list[str], Scorer]] = {
    "arm0": (["arm0-regex", "arm0_drugs-regex", "arm0_first512-regex"], score_arm0),
    "arm1": (["arm1-tfidf-lr"], score_arm1),
    "arm1b": (["arm1b-lgbm"], score_arm1b),
    "arm2-main": (["arm2-lora-main"], lora_scorer("main")),
    "arm2-main-head_tail": (["arm2-lora-main-head_tail"], lora_scorer("main-head_tail")),
}


def run_arm(arm: str, cfg: dict, notes: pd.DataFrame, arms: dict = ARMS) -> list[Path]:
    """Score one arm on test and write its files. Returns the files written.

    notes: every note (train, validation and test), as load_notes() gives them.
    """
    check_unlocked(cfg)
    stems, scorer = arms[arm]
    done = [s for s in stems if scored_path(cfg, s).exists()]
    if done:
        raise FileExistsError(f"{arm}: test already scored ({', '.join(done)}); test is scored once. "
                              f"Delete the file(s) under {scored_path(cfg, '').parent} by hand "
                              f"only if that is truly meant.")
    dev = notes[notes["split"].isin(["train", "validation"])]     # exactly what validation used
    test = notes[notes["split"] == "test"]
    val_ids = dev.loc[dev["split"] == "validation", "hadm_id"]

    start = time.perf_counter()
    out = scorer(dev, test, cfg)
    if sorted(out) != sorted(stems):
        raise ValueError(f"{arm}: scorer returned {sorted(out)}, expected {sorted(stems)}")
    tol = cfg["test_scoring"]["reproduce_tol"]
    for stem, (val_scores, _) in out.items():                       # check every model first
        if val_scores is not None:
            gap = check_reproduced(cfg, stem, val_ids, val_scores, tol)
            print(f"{stem}: validation reproduced (largest gap {gap:.1e})", flush=True)
    written = []
    for stem, (_, test_scores) in out.items():
        if len(test_scores) != len(test):
            raise ValueError(f"{stem}: {len(test_scores):,} scores for {len(test):,} test notes")
        path = scored_path(cfg, stem)
        save_scores(test["hadm_id"], test_scores, path)
        written.append(path)
    print(f"{arm}: {len(test):,} test notes scored in {(time.perf_counter() - start) / 60:.1f} min; "
          f"wrote {', '.join(p.name for p in written)}", flush=True)
    return written


def main(arm: str, config_path: str | Path = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    check_unlocked(cfg)                  # before reading any note
    run_arm(arm, cfg, load_notes(config_path, columns=[LABEL, "text"]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score test once with one frozen arm (scores only).")
    parser.add_argument("arm", choices=list(ARMS))
    main(parser.parse_args().arm)