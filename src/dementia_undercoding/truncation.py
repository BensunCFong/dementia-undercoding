"""Which 512 tokens should arm 2 read? A cheap check with arm 0, before any GPU run.

Arm 2 reads only the first 512 tokens of each note. About 41% of notes are
longer, and the baseline text runs HPI, then PMH, then Social History, so PMH
(where dementia is usually listed) is what gets cut. Arm 0 lost 0.05 of
PR-AUC (0.622 -> 0.572) when cut to that window.

This module builds the two other windows the follow-up could use and scores
arm 0 on each:

  first 512    what arm 2 reads now
  PMH first    sections reordered PMH, HPI, Social History, then the first 512
  head + tail  first 128 tokens + last 382 (510 = 512 minus [CLS] and [SEP])

Arm 0 is not trained, so this only measures how many explicit mentions each
window keeps in view. It is evidence for choosing a window, not a prediction
of arm 2's score. Only validation is read; test stays untouched.

Run:  HF_HUB_OFFLINE=1 .venv/bin/python -m dementia_undercoding.truncation
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .text import BASELINE_SECTIONS, sections

PMH_FIRST_ORDER = ("Past Medical History", "History of Present Illness", "Social History")
HEAD, TAIL = 128, 382


def reorder(text: str, order: tuple[str, ...]) -> str:
    """Baseline text with its sections in `order`. Unchanged if no section is found.

    baseline_text() writes each section as "Heading:\\nbody" with the template
    spelling, so sections() splits it back apart cleanly.
    """
    parts = sections(text)
    found = [f"{name}:\n{parts[name]}" for name in order if parts.get(name)]
    return "\n\n".join(found) if found else text


def pmh_first(text: str) -> str:
    return reorder(text, PMH_FIRST_ORDER)


def head_tail(texts: list[str], tokenizer, head: int = HEAD, tail: int = TAIL) -> list[str]:
    """Each text cut to its first `head` and last `tail` tokens, joined by a line break.

    Notes that already fit are returned whole. The line break also ends a
    clause for arm 0, so no sentence is glued across the gap.
    """
    enc = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    out = []
    for text, offsets in zip(texts, enc["offset_mapping"], strict=True):
        if len(offsets) <= head + tail:
            out.append(text)
        else:
            out.append(text[:offsets[head - 1][1]] + "\n" + text[offsets[-tail][0]:])
    return out


def check(config_path: str | Path = "config/config.yaml", label: str = "dementia_core") -> dict:
    from transformers import AutoTokenizer  # only this step needs transformers

    from .arm0_regex import any_mention, first_512_tokens, score_all
    from .cohort import load_config
    from .evaluate import report, score
    from .splits import load_notes

    cfg = load_config(config_path)
    window, weight = cfg["arm0"]["window"], cfg["arm0"]["uncertain_weight"]
    model, max_len = cfg["lora"]["model"], cfg["lora"]["max_len"]

    ids = load_notes(config_path, columns=[])
    val_ids = ids.loc[ids["split"] == "validation", "hadm_id"].tolist()
    val = pd.read_parquet(cfg["paths"]["notes"], columns=["hadm_id", label, "text"],
                          filters=[("hadm_id", "in", val_ids)])
    texts, y = val["text"].tolist(), val[label].to_numpy()

    # Sanity check: putting the sections back in their usual order must give the note back.
    same = sum(reorder(t, BASELINE_SECTIONS) == t for t in texts)
    print(f"round trip: {same:,} of {len(texts):,} notes unchanged by split-and-rejoin")

    tokenizer = AutoTokenizer.from_pretrained(model)
    windows = {
        "full note (reference)": texts,
        "first 512": first_512_tokens(texts, model, max_len),
        "PMH first, first 512": first_512_tokens([pmh_first(t) for t in texts], model, max_len),
        "head 128 + tail 382": head_tail(texts, tokenizer),
    }
    results = {}
    for name, window_texts in windows.items():
        s = score_all(window_texts, window, weight)[:, 0]          # column 0: diagnosis terms only
        results[name] = {**score(y, s), **any_mention(y, s)}
        print(report(f"arm 0, {name:<22}", results[name]))
        print(f"{'':<8}any mention: recall {results[name]['recall']:.3f}, "
              f"precision {results[name]['precision']:.3f}")
    return results


if __name__ == "__main__":
    check()
