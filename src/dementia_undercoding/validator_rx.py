"""Validator 1: dementia drug orders as a second label, among uncoded notes.

Every rule here was fixed before any arm was scored on this label (2026-11-08 and 2026-11-16;
docs/explainers.md, "Validator 1"):

  pool    uncoded admissions (dementia_core = 0); MCI-coded ones (dementia_broad = 1) are
          dropped from every row except the sensitivity row
  label   1 if the same admission (hadm_id) has a dementia drug order: the drug string matches
          arm 0's drug names, or the order carries an ndc or gsn code seen on a name-matched order
  rows    main; hidden (positives with an explicit mention by arm 0 left out); clean (notes whose
          history text names a dementia drug or drug class left out); sensitivity (MCI kept)
  arms    every arm in compare_arms.ARMS plus age only, which is the reference
  bands   main row split by age, each band shown only with enough positive patients;
          PR-AUC and interval only, no differences (description, not tests)

Reads drug strings and codes from prescriptions, the history text only to look for drug names,
and the saved per-note scores. Prints counts and aggregates only; no note text.

Run:  .venv/bin/python -m dementia_undercoding.validator_rx
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .arm0_regex import DRUG_TERMS, DRUGS
from .cohort import load_config
from .compare_arms import AGE_ONLY, all_scores
from .evaluate import compare, score
from .splits import load_notes

LABEL = "dementia_core"              # the code label the pool is defined by, and the score files' label
HIDDEN_BY = "arm 0 regex"            # a positive is "explicit" if this arm's score is above 0
SELF_CHECK = "arm 0+drugs"           # reads drug names, so it partly checks itself here
ARM3 = ["arm 3 MedGemma 4B", "arm 3 Qwen 9B"]
ARM3_BAR = "arm 1 TF-IDF+LR"         # arm 3's code-label route: at least this arm's PR-AUC

# Clean row only: drug classes as a note might write them. Fixed 2026-11-16 (0 hits on validation,
# kept for test). Arm 0's drug names are not changed; this list only decides who leaves the clean row.
CLASS_TERMS = [
    r"(?:acetyl)?cholinesterase inhibitors?",
    r"a?ch-?eis?",                                # AChEI, ChEI, AChEIs
    r"nmda(?: receptor)? antagonists?",
]
NAMES_IN_TEXT = re.compile(r"(?<![\w/])(?:" + "|".join(DRUG_TERMS + CLASS_TERMS) + r")(?![\w])",
                           re.IGNORECASE)        # same word edges as arm 0's patterns
NO_CODE = {"", "0", "nan", "none"}               # code fields that carry no code


# --- the label ------------------------------------------------------------------------


def names_drug(drug) -> bool:
    """True if a prescriptions drug string names a dementia drug (arm 0's names)."""
    return isinstance(drug, str) and DRUGS.search(drug) is not None


def _codes(value) -> set[str]:
    """The codes in one ndc or gsn field. gsn can hold several, separated by spaces."""
    if not isinstance(value, str):
        return set()
    return {c for c in value.split() if c.lower() not in NO_CODE}


def code_sets(orders: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Every ndc and gsn code seen on an order whose drug string names a dementia drug."""
    named = orders[orders["drug"].map(names_drug)]
    ndcs = set().union(*named["ndc"].map(_codes)) if len(named) else set()
    gsns = set().union(*named["gsn"].map(_codes)) if len(named) else set()
    return ndcs, gsns


def is_drug_order(drug, ndc, gsn, ndcs: set[str], gsns: set[str]) -> bool:
    """A dementia drug order: the name matches, or the order shares an ndc or gsn code with one that does."""
    return names_drug(drug) or bool(_codes(ndc) & ndcs) or bool(_codes(gsn) & gsns)


def drug_label(cfg: dict, hadm_ids) -> pd.DataFrame:
    """One row per admission in hadm_ids: by_name and positive (name or code), from prescriptions.

    Two passes over prescriptions. The first finds the codes from every split's name-matched
    orders, so the code list is the same whatever split is scored. The second reads only these
    admissions' orders.
    """
    rx = Path(cfg["paths"]["mimic_hosp"]) / cfg["files"]["hosp"]["prescriptions"]
    src = f"read_csv('{rx}', types = {{'drug': 'VARCHAR', 'ndc': 'VARCHAR', 'gsn': 'VARCHAR'}})"
    con = duckdb.connect()
    combos = con.execute(f"SELECT DISTINCT drug, ndc, gsn FROM {src}").df()
    ndcs, gsns = code_sets(combos)
    con.register("pool", pd.DataFrame({"hadm_id": np.asarray(hadm_ids)}))
    orders = con.execute(f"SELECT DISTINCT r.hadm_id, r.drug, r.ndc, r.gsn FROM {src} r "
                         "JOIN pool USING (hadm_id)").df()
    orders["by_name"] = orders["drug"].map(names_drug)
    orders["positive"] = [is_drug_order(d, n, g, ndcs, gsns)
                          for d, n, g in zip(orders["drug"], orders["ndc"], orders["gsn"], strict=True)]
    per = orders.groupby("hadm_id")[["by_name", "positive"]].any()
    out = pd.DataFrame({"hadm_id": np.asarray(hadm_ids)}).merge(per, on="hadm_id", how="left")
    out[["by_name", "positive"]] = out[["by_name", "positive"]].eq(True)     # no orders: False
    print(f"label codes from name-matched orders, every split: {len(ndcs):,} ndc, {len(gsns):,} gsn")
    code_only = int((orders["positive"] & ~orders["by_name"]).sum())
    print(f"orders in the pool: {len(orders):,} distinct; dementia drug by name "
          f"{int(orders['by_name'].sum()):,}, by code only {code_only:,}")
    return out


# --- rows, bands and the top-k curve ----------------------------------------------------


def names_in_text(texts) -> np.ndarray:
    """True where the history text names a dementia drug or drug class, negated or not."""
    return np.array([NAMES_IN_TEXT.search(t) is not None for t in texts], dtype=bool)


def row_masks(y, mci, names_drug_text, explicit) -> dict[str, np.ndarray]:
    """Which pool notes each row keeps. All inputs are 0/1 arrays over the uncoded pool.

    hidden keeps every negative and only the positives without an explicit mention.
    """
    y, mci, names, explicit = (np.asarray(a, dtype=bool) for a in (y, mci, names_drug_text, explicit))
    main = ~mci
    return {
        "main": main,
        "hidden": main & ~(y & explicit),
        "clean": main & ~names,
        "sensitivity (MCI kept)": np.ones(len(y), dtype=bool),
    }


def band_masks(age, cut: float) -> dict[str, np.ndarray]:
    age = np.asarray(age, dtype=float)
    return {f"under {cut:g}": age < cut, f"{cut:g}+": age >= cut}


def positive_patients(y, groups) -> int:
    y = np.asarray(y, dtype=bool)
    return int(pd.Series(np.asarray(groups)[y]).nunique())


def topk_curve(y, s) -> pd.DataFrame:
    """Expected drug-positive notes among the top k notes, for every k.

    Notes with the same score are taken in random order, so inside a run of tied scores the
    count rises evenly (its expected value), not in an order that favours or hurts the arm.
    """
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    n = len(s)
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    block = np.repeat(np.arange(len(starts)), ends - starts)
    st, en = starts[block], ends[block]
    cum = np.r_[0.0, np.cumsum(y)]
    k = np.arange(1, n + 1)
    positives = cum[st] + (cum[en] - cum[st]) * (k - st) / (en - st)
    return pd.DataFrame({"k": k, "positives": positives, "rate": positives / k})


def arm3_decision(code_pr_auc: dict, bar: float, hidden: pd.DataFrame) -> dict[str, bool]:
    """STATUS item 2's rule: a model qualifies on code-label PR-AUC >= bar, or by beating age only
    on the hidden row with the whole difference interval above zero."""
    h = hidden.set_index("arm")
    return {m: bool(code_pr_auc[m] >= bar or h.loc[m, "diff_low"] > 0) for m in code_pr_auc}


def pick_arm3(code_pr_auc: dict, code_diff_ci: tuple[float, float], hidden_pr_auc: dict) -> str:
    """When both models qualify, only one goes to test: the higher code-label PR-AUC, unless the
    code-label difference interval includes 0; then the higher hidden-row PR-AUC."""
    if code_diff_ci[0] > 0 or code_diff_ci[1] < 0:
        return max(code_pr_auc, key=code_pr_auc.get)
    return max(hidden_pr_auc, key=hidden_pr_auc.get)


# --- printing -------------------------------------------------------------------------


def _name(arm: str) -> str:
    return arm + (" *" if arm == SELF_CHECK else "")


def print_table(table: pd.DataFrame, pct: int, diffs: bool = True) -> None:
    head = f"{'arm':<20} {'PR-AUC':>7}  {f'{pct}% interval':<17}"
    print(head + (f" {'diff':>7}  {f'{pct}% interval':<17}" if diffs else ""))
    for r in table.itertuples():
        line = f"{_name(r.arm):<20} {r.pr_auc:>7.3f}  [{r.ci_low:.3f}, {r.ci_high:.3f}]  "
        if diffs:
            line += f"  {r.diff:>+7.3f}  [{r.diff_low:+.3f}, {r.diff_high:+.3f}]"
        print(line)


# --- run ------------------------------------------------------------------------------


def run(config_path: str | Path = "config/config.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    vcfg, ccfg = cfg["validator_rx"], cfg["compare"]
    reference, pct = AGE_ONLY, round(ccfg["ci"] * 100)
    workers = max(1, (os.cpu_count() or 2) - 2)
    boot = {"ci": ccfg["ci"], "n_boot": ccfg["n_boot"], "seed": ccfg["seed"], "workers": workers}

    notes = load_notes(config_path, columns=["age_at_admit", LABEL, "dementia_broad", "delirium"])
    val = notes[notes["split"] == "validation"].reset_index(drop=True)
    scores = all_scores(cfg, val, LABEL)                      # every arm, lined up with val, checked
    code_pr_auc = {m: score(val[LABEL], scores[m])["pr_auc"] for m in [*ARM3, ARM3_BAR]}

    in_pool = (val[LABEL] == 0).to_numpy()
    pool = val[in_pool].reset_index(drop=True)
    pool_scores = {arm: s[in_pool] for arm, s in scores.items()}
    lab = drug_label(cfg, pool["hadm_id"])
    pool["y"] = lab["positive"].to_numpy()
    by_name = lab["by_name"].to_numpy()
    text = pd.read_parquet(cfg["paths"]["notes"], columns=["hadm_id", "text"],
                           filters=[("hadm_id", "in", pool["hadm_id"].tolist())])
    text = pool[["hadm_id"]].merge(text, on="hadm_id", how="left", validate="one_to_one")["text"]
    names = names_in_text(text.fillna(""))
    explicit = pool_scores[HIDDEN_BY] > 0
    mci = (pool["dementia_broad"] == 1).to_numpy()
    rows = row_masks(pool["y"], mci, names, explicit)

    # Label counts first, beside the 2026-11-08 counts (drug names only, before the codes were added).
    y, g = pool["y"].to_numpy(), pool["subject_id"].to_numpy()
    main = rows["main"]

    def count(mask) -> str:
        return f"{int(mask.sum()):,} notes, {positive_patients(mask, g):,} patients"

    print(f"\nvalidation, uncoded: {len(pool):,} notes; MCI-coded {int(mci.sum()):,}; "
          f"main pool {int(main.sum()):,}")
    print(f"  positives by name only, main: {count(by_name & main)}   (2026-11-08: 193 notes, 120 patients)")
    print(f"  positives by name or code, main: {count(y & main)}")
    print(f"    explicit (arm 0 score > 0): {count(y & main & explicit)}   (by name, 2026-11-08: 101 / 67)")
    print(f"    hidden (arm 0 score 0): {count(y & main & ~explicit)}   (by name, 2026-11-08: 92 / 58)")
    print(f"    delirium-coded: {count(y & main & (pool['delirium'] == 1).to_numpy())}   "
          f"(by name, 2026-11-08: 13 / 11; count only)")
    print(f"  history text names a drug or class, main: {int((names & main).sum()):,} notes "
          f"({int((names & main & y).sum()):,} positive)")

    tables, curves = [], []
    for row, m in rows.items():
        ym = y[m]
        print(f"\n=== {row}: {int(m.sum()):,} notes, {int(ym.sum()):,} positives "
              f"({positive_patients(ym, g[m]):,} patients), chance {ym.mean():.4f} ===")
        t = compare(ym, {a: s[m] for a, s in pool_scores.items()}, g[m], reference, **boot)
        print_table(t, pct)
        tables.append(t.assign(row=row, n=int(m.sum()), n_pos=int(ym.sum()),
                               pos_patients=positive_patients(ym, g[m])))
        curves += [topk_curve(ym, s[m]).assign(row=row, arm=a) for a, s in pool_scores.items()]
    print(f"\ndiff = arm minus {reference}; {ccfg['n_boot']:,} patient-level resamples. "
          f"* {SELF_CHECK} reads drug names, so it partly checks itself.")

    print(f"\n=== age bands, main row: PR-AUC and interval only; shown with >= {vcfg['min_patients']} "
          f"positive patients ===")
    for band, b in band_masks(pool["age_at_admit"], vcfg["age_cut"]).items():
        m = main & b
        n_pat = positive_patients(y[m], g[m])
        if n_pat < vcfg["min_patients"]:
            print(f"\n{band}: {n_pat} positive patients, below the rule; not shown")
            continue
        print(f"\n{band}: {int(m.sum()):,} notes, {int(y[m].sum()):,} positives ({n_pat} patients), "
              f"chance {y[m].mean():.4f}")
        t = compare(y[m], {a: s[m] for a, s in pool_scores.items()}, g[m], reference, **boot)
        t = t.drop(columns=["diff", "diff_low", "diff_high"])
        print_table(t, pct, diffs=False)
        tables.append(t.assign(row=f"main, {band}", n=int(m.sum()), n_pos=int(y[m].sum()),
                               pos_patients=n_pat))

    hidden = next(t for t in tables if t["row"].iat[0] == "hidden")
    ok = arm3_decision({m: code_pr_auc[m] for m in ARM3}, code_pr_auc[ARM3_BAR], hidden)
    print(f"\n=== arm 3 test rule (STATUS item 2): code-label PR-AUC >= {ARM3_BAR}'s "
          f"{code_pr_auc[ARM3_BAR]:.3f}, or hidden-row diff interval above 0 ===")
    h = hidden.set_index("arm")
    for m in ARM3:
        print(f"{m:<20} code-label {code_pr_auc[m]:.3f}; hidden diff {h.loc[m, 'diff']:+.3f} "
              f"[{h.loc[m, 'diff_low']:+.3f}, {h.loc[m, 'diff_high']:+.3f}] -> "
              f"{'qualifies' if ok[m] else 'does not qualify'}")
    if all(ok.values()):
        pair = compare(val[LABEL], {m: scores[m] for m in ARM3}, val["subject_id"], ARM3[0], **boot)
        d = pair.set_index("arm").loc[ARM3[1]]
        pick = pick_arm3({m: code_pr_auc[m] for m in ARM3}, (d["diff_low"], d["diff_high"]),
                         {m: h.loc[m, "pr_auc"] for m in ARM3})
        print(f"both qualify; code-label diff ({ARM3[1]} minus {ARM3[0]}) {d['diff']:+.3f} "
              f"[{d['diff_low']:+.3f}, {d['diff_high']:+.3f}] -> {pick} goes to test")

    out = Path(vcfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    table = pd.concat(tables, ignore_index=True)
    table.to_parquet(out / "validation-table.parquet", index=False)
    pd.concat(curves, ignore_index=True).to_parquet(out / "validation-topk.parquet", index=False)
    print(f"\nsaved {out / 'validation-table.parquet'} and {out / 'validation-topk.parquet'}")
    return table


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml")
