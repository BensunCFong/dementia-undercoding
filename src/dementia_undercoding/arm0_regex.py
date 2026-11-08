"""Arm 0: explicit mentions of dementia, with NegEx-style context rules.

No training. Every rule below was fixed on 2026-11-07, before any note was
read, from the ICD code definitions in config/codes.yaml and standard clinical
vocabulary. Nothing here is tuned on validation.

A mention is a term match. Each mention is classed by the words around it,
within the same clause (sentence ends and contrast words split clauses) and at
most `window` words away:

  negated       "no history of dementia", "dementia was ruled out"  -> not counted
  someone else  "mother had dementia"                                -> not counted
  uncertain     "concern for dementia", "r/o dementia"               -> counted at uncertain_weight
  certain       everything else                                      -> counted at 1

Score = sum of counted mentions. Two pre-declared variants:
  arm 0        diagnosis terms only (used at the validation stage, so the
               prescription validator stays independent)
  arm 0+drugs  diagnosis terms plus the four dementia drugs

Run:  .venv/bin/python -m dementia_undercoding.arm0_regex
"""

from __future__ import annotations

import os
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .cohort import load_config
from .evaluate import report, save_scores, score
from .splits import load_notes

# --- terms (mirror the dementia_core ICD codes) -------------------------------
DIAGNOSIS_TERMS = [
    r"dement(?:ia|ias|ed)",                    # any dementia: 290, 2941, 2942, F01-F03
    r"alzheimer(?:'s|’s|s)?",                  # 3310, G30
    r"lewy bod(?:y|ies)",                      # 33182, G3183
    r"pick(?:'s|’s|s)? disease",               # frontotemporal: 3311, G310
    r"major neurocognitive disorders?",        # DSM-5 name for dementia
]
DRUG_TERMS = [
    r"donepezil", r"aricept", r"rivastigmine", r"exelon",
    r"galantamine", r"razadyne", r"memantine", r"namenda", r"namzaric",
]

# --- context triggers (NegEx / ConText style) ----------------------------------
NEGATION_BEFORE = [
    r"no history of", r"no evidence of", r"no signs? of", r"negative for", r"free of",
    r"no", r"not", r"denies", r"denied", r"without",
]
NEGATION_AFTER = [r"(?:was |been |is )?ruled out", r"unlikely", r"not present", r"(?:is |was )?absent"]
# Standard NegEx pseudo-negations: they contain "no"/"not" but do not negate.
PSEUDO_NEGATION = [
    r"no change[sd]?", r"no worsening", r"no increase", r"no further", r"not only",
    r"not (?:been |yet )?ruled out", r"can ?not (?:be )?(?:excluded?|ruled out)", r"can ?not rule out",
]
OTHER_PERSON = [
    r"family history", r"fhx", r"mother", r"father", r"sister", r"brother", r"son",
    r"daughter", r"wife", r"husband", r"spouse", r"partner", r"relative", r"caregiver for",
]
# Words that bring the patient back between a relative and the term:
# "per daughter, patient has dementia" is about the patient.
PATIENT_WORDS = r"\b(?:patient|pt|she|he)\b"
UNCERTAIN_BEFORE = [
    r"concern(?:ing)? for", r"possible", r"possibly", r"probable", r"suspected", r"suspect",
    r"suspicion (?:of|for)", r"question of", r"questionable", r"r/o", r"rule out",
    r"evaluate for", r"eval(?:uation)? for", r"work ?up for", r"may have",
    r"can ?not (?:be )?(?:excluded?|ruled out)", r"can ?not rule out", r"not (?:been |yet )?ruled out",
]
UNCERTAIN_AFTER = [r"not (?:been |yet )?ruled out", r"(?:is |was )?(?:suspected|possible|questioned)"]

CLAUSE_BREAK = re.compile(
    r"[.;:\n]+|\?(?=\s)|\b(?:but|however|although|though|except|which)\b", re.IGNORECASE
)
WORD = re.compile(r"\S+")


def _compile(patterns: list[str]) -> re.Pattern:
    return re.compile(r"(?<![\w/])(?:" + "|".join(patterns) + r")(?![\w])", re.IGNORECASE)


DIAGNOSIS = _compile(DIAGNOSIS_TERMS)
DRUGS = _compile(DRUG_TERMS)
NEG_BEFORE = _compile(NEGATION_BEFORE)
NEG_AFTER = _compile(NEGATION_AFTER)
PSEUDO = _compile(PSEUDO_NEGATION)
OTHER = _compile(OTHER_PERSON)
UNCERTAIN = _compile(UNCERTAIN_BEFORE)
UNCERTAIN_POST = _compile(UNCERTAIN_AFTER)
QUESTION_MARK = re.compile(r"\?\s*$")             # "?dementia"


def _word_index(starts: list[int], pos: int) -> int:
    """Index of the word containing character position pos."""
    return int(np.searchsorted(starts, pos, side="right")) - 1


def classify(clause: str, term: re.Match, window: int) -> str:
    """'negated', 'other', 'uncertain' or 'certain' for one term match in one clause."""
    starts = [m.start() for m in WORD.finditer(clause)]
    t_first = _word_index(starts, term.start())
    t_last = _word_index(starts, term.end() - 1)
    pseudo = [(m.start(), m.end()) for m in PSEUDO.finditer(clause)]

    def in_pseudo(m: re.Match) -> bool:
        return any(a <= m.start() < b for a, b in pseudo)

    def before(pattern: re.Pattern, skip_pseudo: bool = False) -> list[re.Match]:
        hits = []
        for m in pattern.finditer(clause[:term.start()]):
            if skip_pseudo and in_pseudo(m):
                continue
            if 1 <= t_first - _word_index(starts, m.end() - 1) <= window:
                hits.append(m)
        return hits

    def after(pattern: re.Pattern, skip_pseudo: bool = False) -> list[re.Match]:
        return [m for m in pattern.finditer(clause, term.end())
                if not (skip_pseudo and in_pseudo(m))
                and 1 <= _word_index(starts, m.start()) - t_last <= window]

    if before(NEG_BEFORE, skip_pseudo=True) or after(NEG_AFTER, skip_pseudo=True):
        return "negated"
    for m in before(OTHER):
        if not re.search(PATIENT_WORDS, clause[m.end():term.start()], re.IGNORECASE):
            return "other"
    if before(UNCERTAIN) or after(UNCERTAIN_POST) or QUESTION_MARK.search(clause[:term.start()]):
        return "uncertain"
    return "certain"


def count_mentions(text: str, pattern: re.Pattern, window: int) -> dict[str, int]:
    """Counts of certain / uncertain / negated / other mentions of `pattern` in text."""
    counts = {"certain": 0, "uncertain": 0, "negated": 0, "other": 0}
    for clause in CLAUSE_BREAK.split(text):
        for term in pattern.finditer(clause):
            counts[classify(clause, term, window)] += 1
    return counts


def mention_score(counts: dict[str, int], uncertain_weight: float) -> float:
    return counts["certain"] + uncertain_weight * counts["uncertain"]


def score_note(text: str, window: int, uncertain_weight: float) -> tuple[float, float]:
    """(arm 0 score, drug-only score) for one note. arm 0+drugs = their sum."""
    dx = count_mentions(text, DIAGNOSIS, window)
    rx = count_mentions(text, DRUGS, window)
    return mention_score(dx, uncertain_weight), mention_score(rx, uncertain_weight)


def any_mention(y_true, y_score) -> dict:
    """Precision and recall of the simplest rule: flag a note if its score is above 0."""
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(y_score) > 0
    tp = int((flagged & y_true).sum())
    return {
        "flagged_share": float(flagged.mean()),
        "precision": tp / flagged.sum() if flagged.any() else 0.0,
        "recall": tp / y_true.sum(),
    }


def first_512_tokens(texts: list[str], model_name: str, max_len: int) -> list[str]:
    """Each text cut to exactly the window arm 2 reads: its first max_len tokens."""
    from transformers import AutoTokenizer  # only this step needs transformers

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    enc = tokenizer(texts, truncation=True, max_length=max_len, return_offsets_mapping=True)
    pairs = zip(texts, enc["offset_mapping"], strict=True)
    return [text[:max(end for _, end in offsets)] for text, offsets in pairs]


def _score_one(args):
    return score_note(*args)


def score_all(texts: list[str], window: int, uncertain_weight: float) -> np.ndarray:
    """Scores for every text, spread over CPU cores (leaving two free)."""
    workers = max(1, (os.cpu_count() or 2) - 2)
    with Pool(workers) as pool:
        return np.array(pool.map(_score_one, [(t, window, uncertain_weight) for t in texts], chunksize=500))


def run(config_path: str | Path = "config/config.yaml", label: str = "dementia_core") -> dict:
    import mlflow  # imported here, not at the top: CI installs only core packages

    cfg = load_config(config_path)
    acfg = cfg["arm0"]
    window, weight = acfg["window"], acfg["uncertain_weight"]

    # Only validation text is read; train needs nothing (no training) and test stays untouched.
    ids = load_notes(config_path, columns=[])
    val_ids = ids.loc[ids["split"] == "validation", "hadm_id"].tolist()
    val = pd.read_parquet(cfg["paths"]["notes"], columns=["hadm_id", label, "text"],
                          filters=[("hadm_id", "in", val_ids)])
    texts, y = val["text"].tolist(), val[label].to_numpy()

    start = time.perf_counter()
    full = score_all(texts, window, weight)
    cut = score_all(first_512_tokens(texts, cfg["lora"]["model"], cfg["lora"]["max_len"]), window, weight)
    seconds = time.perf_counter() - start

    variants = {
        "arm0": full[:, 0],
        "arm0_drugs": full[:, 0] + full[:, 1],
        "arm0_first512": cut[:, 0],
    }
    results = {}
    for name, s in variants.items():
        results[name] = {**score(y, s), **{f"any_{k}": v for k, v in any_mention(y, s).items()}}
        print(report(f"{name:<14} {label}, validation", results[name]))
        print(f"{'':<16}any mention: flags {results[name]['any_flagged_share']:.1%} of notes, "
              f"precision {results[name]['any_precision']:.3f}, recall {results[name]['any_recall']:.3f}")
        save_scores(val["hadm_id"], s, Path(cfg["paths"]["scores"]) / f"{name}-regex-{label}.parquet")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    for name, r in results.items():
        with mlflow.start_run(run_name=f"{name}-regex-{label}"):
            mlflow.log_params({"arm": "0", "variant": name, "label": label, "scored_on": "validation",
                               "window": window, "uncertain_weight": weight,
                               "n_diagnosis_terms": len(DIAGNOSIS_TERMS), "n_drug_terms": len(DRUG_TERMS)})
            mlflow.log_metrics({f"val_{k}": v for k, v in r.items()})
            mlflow.log_metric("seconds", seconds)
    return results


if __name__ == "__main__":
    run()