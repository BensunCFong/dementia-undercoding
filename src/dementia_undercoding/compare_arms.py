"""Every arm on validation, with patient-level intervals and paired differences.

Reads only saved per-note scores (hadm_id, score) and the labels; no note text.
Prints aggregates only.

Run:  .venv/bin/python -m dementia_undercoding.compare_arms
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .cohort import load_config
from .evaluate import compare
from .splits import load_notes

# Display name -> score file stem, as each arm's run() writes it: <stem>-<label>.parquet
ARMS = {
    "arm 0 regex": "arm0-regex",
    "arm 0+drugs": "arm0_drugs-regex",
    "arm 0 first 512": "arm0_first512-regex",
    "arm 1 TF-IDF+LR": "arm1-tfidf-lr",
    "arm 1b LightGBM": "arm1b-lgbm",
    "arm 2 LoRA main": "arm2-lora-main",
    "arm 2 LoRA h+t": "arm2-lora-main-head_tail",
}


def load_scores(cfg: dict, val: pd.DataFrame, label: str) -> dict:
    """Each arm's scores lined up with val's rows. Fails if any note is missing or extra."""
    out = {}
    for name, stem in ARMS.items():
        s = pd.read_parquet(Path(cfg["paths"]["scores"]) / f"{stem}-{label}.parquet")
        merged = val[["hadm_id"]].merge(s, on="hadm_id", how="left", validate="one_to_one")
        if len(s) != len(val) or merged["score"].isna().any():
            raise ValueError(f"{name}: {len(s):,} scores for {len(val):,} validation notes, "
                             f"{int(merged['score'].isna().sum()):,} notes without a score")
        out[name] = merged["score"].to_numpy()
    return out


def run(config_path: str | Path = "config/config.yaml", label: str = "dementia_core") -> pd.DataFrame:
    cfg = load_config(config_path)
    ccfg = cfg["compare"]
    notes = load_notes(config_path, columns=[label])           # no text column read
    val = notes[notes["split"] == "validation"].reset_index(drop=True)
    scores = load_scores(cfg, val, label)

    workers = max(1, (os.cpu_count() or 2) - 2)
    table = compare(val[label], scores, val["subject_id"], ccfg["reference"],
                    ci=ccfg["ci"], n_boot=ccfg["n_boot"], seed=ccfg["seed"], workers=workers)

    pct = round(ccfg["ci"] * 100)
    print(f"validation: {len(val):,} notes, {val['subject_id'].nunique():,} patients, "
          f"{int(val[label].sum()):,} positives; "
          f"{ccfg['n_boot']:,} patient-level resamples, {pct}% intervals")
    print(f"diff = arm minus {ccfg['reference']}\n")
    print(f"{'arm':<18} {'PR-AUC':>7}  {f'{pct}% interval':<17} {'diff':>7}  {f'{pct}% interval':<17}")
    for r in table.itertuples():
        print(f"{r.arm:<18} {r.pr_auc:>7.3f}  [{r.ci_low:.3f}, {r.ci_high:.3f}]    "
              f"{r.diff:>+7.3f}  [{r.diff_low:+.3f}, {r.diff_high:+.3f}]")
    return table


if __name__ == "__main__":
    run()
