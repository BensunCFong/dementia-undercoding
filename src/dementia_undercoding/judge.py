"""The LLM judge (Week 3): reads whole discharge notes for flagged admissions and gives a verdict.

The arms read only the history sections; the judge reads the whole note (rubric in STATUS.md).
Nothing here prints or logs note text. Anything that holds note text is saved under data/ only.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .arm3_llm import LlamaServer
from .cohort import load_config
from .splits import load_notes
from .text import BASELINE_SECTIONS, CANONICAL, HEADING, HEADING_ANY_CASE, clean, sections


def whole_notes(con, discharge_path: str | Path, hadm_ids) -> pd.DataFrame:
    """Cleaned whole discharge notes for these admissions: columns hadm_id, text, sorted by hadm_id.

    The cohort table on con picks each admission's note (its note_id), the same note the
    arms read part of. Fails unless every admission asked for has exactly one note, so no
    flagged note is skipped without anyone noticing.
    """
    wanted = pd.DataFrame({"hadm_id": sorted(set(hadm_ids))})
    con.register("wanted", wanted)
    out = con.execute(f"""
        SELECT c.hadm_id, n.text
        FROM wanted w
        JOIN cohort c USING (hadm_id)
        JOIN read_csv('{discharge_path}') n USING (note_id)
        ORDER BY c.hadm_id
    """).df()
    con.unregister("wanted")
    if len(out) != len(wanted) or out["hadm_id"].duplicated().any():
        raise ValueError(f"{len(wanted):,} admissions asked for, {len(out):,} notes found")
    out["text"] = out["text"].map(clean)
    return out


def load_whole_notes(hadm_ids, config_path: str | Path = "config/config.yaml") -> pd.DataFrame:
    """whole_notes() on the real cohort and the raw discharge files named in config."""
    cfg = load_config(config_path)
    discharge = Path(cfg["paths"]["mimic_note"]) / cfg["files"]["note"]["discharge"]
    con = duckdb.connect(cfg["paths"]["duckdb"], read_only=True)
    try:
        return whole_notes(con, discharge, hadm_ids)
    finally:
        con.close()



# --- the rubric (STATUS.md, "Judge rubric", decided 2026-11-09) ------------------------

VERDICTS = ("Documented", "Possible", "Acute only", "None")

JUDGE_SYSTEM = (
    "You check a hospital discharge summary for one thing: does it document that the patient had "
    "chronic cognitive impairment before this admission, such as dementia of any type or a "
    "long-standing decline in memory or thinking? You do not diagnose. You decide only what the "
    "note documents, using the whole note.\n\n"
    "Verdicts:\n"
    "- Documented: the note states dementia (any type, including Alzheimer's disease) or clearly "
    "describes cognitive impairment that predates this admission, for example memory loss over "
    "months or years reported by family, or needing help with daily activities because of memory "
    "or thinking.\n"
    "- Possible: the note raises a cognitive concern, but it is unclear whether it predates this "
    "admission or how severe it is. Mild cognitive impairment goes here.\n"
    "- Acute only: the only cognitive problem described is acute, during this admission (delirium, "
    "encephalopathy, intoxication, sedation), and baseline cognition is described as normal or "
    "not described.\n"
    "- None: no cognitive impairment is documented, or it is denied, ruled out, or describes "
    "someone other than the patient.\n\n"
    "First give up to three passages from the note that your verdict rests on. Each passage is one "
    "sentence or one list line, exactly as it appears in the note. For None, give passages only if "
    "the note denies impairment or describes it in someone else. Then give the verdict."
)

# The judge does not retype its evidence: it picks passages, and llama-server allows only
# passages that are in the note (decided 2026-11-09 after the timing run, before the pilot).
# A sentence ends at . ! or ? followed by space and a capital, digit or opening bracket.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")
LIST_ITEM = re.compile(r"(?:[-*\u2022]|\d{1,2}[.)])\s")


def passages(note: str) -> list[str]:
    """The note cut into the passages the judge may cite: sentences and list lines.

    Lines are joined into blocks (a blank line, a heading or a list item starts a new one; a
    heading alone on its line is left out, since it holds no evidence), each block is split into
    sentences, and a line break inside a sentence becomes a space. Repeats are kept once, in
    order. Every passage is in the note word for word, spacing aside.
    """
    blocks: list[str] = []
    current: list[str] = []
    for raw in note.split("\n"):
        line = raw.strip()
        heading = HEADING_ANY_CASE.match(line)
        if (not line or LIST_ITEM.match(line) or heading) and current:
            blocks.append(" ".join(current))
            current = []
        if line:
            current.append(line)
        if heading and heading.group(0) == line:             # a bare heading: no evidence in it
            current = []
    if current:
        blocks.append(" ".join(current))

    out: list[str] = []
    for block in blocks:
        marker = LIST_ITEM.match(block)                       # "1. " is not a sentence of its own
        start = marker.end() if marker else 0
        sentences = SENTENCE_END.split(block[start:])
        sentences[0] = block[:start] + sentences[0]
        for sentence in sentences:
            if words(sentence) and sentence not in out:
                out.append(sentence)
    return out


def judge_schema(note_passages: list[str]) -> dict:
    """The answer shape for one note: up to 3 of its passages, then one of the four verdicts."""
    if note_passages:
        evidence = {"type": "array", "items": {"type": "string", "enum": note_passages}, "maxItems": 3}
    else:
        evidence = {"type": "array", "maxItems": 0}
    return {
        "type": "object",
        "properties": {"evidence": evidence, "verdict": {"type": "string", "enum": list(VERDICTS)}},
        "required": ["evidence", "verdict"],
        "additionalProperties": False,
    }


def judge_messages(note: str) -> list[dict]:
    """Chat messages for one note: the rubric, then the note. No examples."""
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"Discharge summary:\n{note}"},
    ]


def ask(url: str, messages: list[dict], max_tokens: int, schema: dict) -> tuple[dict | None, dict]:
    """(parsed answer or None, llama-server timings) for one note.

    None means no usable answer: the JSON was cut off by max_tokens, or llama-server
    refused the note as longer than its context (timings then hold {"rejected": True}).
    Either way it is counted, never guessed at.
    """
    body = {
        "messages": messages, "max_tokens": max_tokens, "temperature": 0,
        "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "schema": schema}},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 400:                     # prompt longer than the context
            return None, {"rejected": True}
        raise
    try:
        answer = json.loads(out["choices"][0]["message"]["content"])
    except json.JSONDecodeError:
        answer = None
    return answer, out.get("timings", {})     



# --- checking the evidence -----------------------------------------------------------
# Faithfulness tiers, most faithful first (decided 2026-11-09 on the validation timing run):
#   exact       - word for word; only spacing and line breaks differ
#   formatting  - the same words in the same order; only capitals or punctuation differ
#   joined      - several real passages glued together; every piece is in the note
#   edited      - at least half its words are one unbroken run of the note, the rest changed
#   not copied  - paraphrased or made up
#   empty       - no words at all
# The first three count as found. "Made-up evidence" = edited + not copied.

FOUND = ("exact", "formatting", "joined")
MADE_UP = ("edited", "not copied")
NO_HEADING = "(before any heading)"


def heading_pattern(note: str) -> re.Pattern:
    """The same choice baseline_text() makes: the template headings, or the capitals fallback
    when the template finds no history section (the older dictated format)."""
    parts = sections(note, HEADING)
    return HEADING if any(parts.get(name) for name in BASELINE_SECTIONS) else HEADING_ANY_CASE


def words(text: str) -> list[str]:
    """Lower-case letters-and-digits words: capitals and punctuation dropped."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _exact(note: str, quote: str) -> re.Match | None:
    """Word for word, spacing aside, and not starting or ending inside a longer word."""
    q = quote.strip()
    start = r"(?<![A-Za-z0-9])" if q[0].isalnum() else ""
    end = r"(?![A-Za-z0-9])" if q[-1].isalnum() else ""
    return re.search(start + r"\s+".join(map(re.escape, q.split())) + end, note)


def _same_words(note: str, text: str) -> re.Match | None:
    """The text's words in order in the note, anything but letters and digits between them,
    and not inside a longer word ("forget" does not match "forgetful")."""
    w = words(text)
    if not w:
        return None
    pattern = r"(?<![a-z0-9])" + r"[^a-z0-9]+".join(map(re.escape, w)) + r"(?![a-z0-9])"
    return re.search(pattern, note, re.IGNORECASE)


def locate(note: str, quote: str) -> tuple[str, list[int]]:
    """(tier, where each found piece starts in the note). Positions are empty unless found."""
    if not words(quote):
        return "empty", []
    if m := _exact(note, quote):
        return "exact", [m.start()]
    if m := _same_words(note, quote):
        return "formatting", [m.start()]
    pieces = [p for p in re.split(r"\.\.\.|…|(?<=[.;:])\s+", quote) if words(p)]
    if len(pieces) > 1:
        matches = [_same_words(note, p) for p in pieces]
        if all(matches):
            return "joined", [m.start() for m in matches]
    q, n = words(quote), words(note)
    run = SequenceMatcher(None, q, n, autojunk=False).find_longest_match(0, len(q), 0, len(n)).size
    return ("edited" if run >= len(q) / 2 else "not copied"), []


def section_at(note: str, position: int, heading: re.Pattern) -> str:
    """The section a position in the note falls in: the last heading that starts before it."""
    before = [h for h in heading.finditer(note) if h.start() <= position]
    if not before:
        return NO_HEADING
    name = before[-1].group(1)
    return CANONICAL.get(name.lower(), name)


def check_evidence(note: str, evidence: list[str]) -> dict:
    """Per note: each quote's tier and section(s), and whether any found piece is in the
    history sections the arms read. A joined quote's sections are listed with " + "."""
    heading = heading_pattern(note)
    tiers, where = [], []
    in_history = False
    for quote in evidence:
        tier, positions = locate(note, quote)
        found_in = [section_at(note, p, heading) for p in positions]
        in_history = in_history or any(s in BASELINE_SECTIONS for s in found_in)
        tiers.append(tier)
        where.append(" + ".join(found_in) if found_in else "(not found)")
    return {
        "n_quotes": len(tiers),
        "n_not_found": sum(t not in FOUND for t in tiers),
        "n_made_up": sum(t in MADE_UP for t in tiers),
        "in_history": in_history,
        "tiers": tiers,
        "sections": where,
    }



# --- the timing run (validation, development only) --------------------------------------

def flagged(val: pd.DataFrame, score: np.ndarray, label: str, depth: float) -> pd.Series:
    """hadm_ids of the uncoded notes among an arm's top (depth x coded count) notes.

    val must be sorted by hadm_id, so tied scores are broken the same way every run.
    """
    k = round(depth * int(val[label].sum()))
    top = val.assign(score=score).nlargest(k, "score", keep="first")
    return top.loc[top[label] == 0, "hadm_id"]


def run_timing(config_path: str | Path = "config/config.yaml", label: str = "dementia_core") -> pd.DataFrame:
    """The judge on a fixed random sample of validation gap-set notes. Prints counts only.

    The gap set here is provisional (both arms, one depth from config) because the bands
    are not fixed yet; this run measures speed and checks the machinery, nothing more.
    Per-note results, quotes included, are saved under data/ only.
    """
    cfg = load_config(config_path)
    jcfg, tcfg = cfg["judge"], cfg["judge"]["timing"]

    notes = load_notes(config_path, columns=[label])                     # no text column read
    val = notes[notes["split"] == "validation"].sort_values("hadm_id").reset_index(drop=True)
    pool: set[int] = set()
    for stem in jcfg["arms"].values():
        s = pd.read_parquet(Path(cfg["paths"]["scores"]) / f"{stem}-{label}.parquet")
        score = val[["hadm_id"]].merge(s, on="hadm_id", how="left", validate="one_to_one")["score"]
        if score.isna().any():
            raise ValueError(f"{stem}: {int(score.isna().sum()):,} validation notes without a score")
        pool |= set(flagged(val, score.to_numpy(), label, tcfg["depth"]).tolist())

    rng = np.random.default_rng(tcfg["seed"])
    ids = sorted(rng.choice(sorted(pool), size=min(tcfg["n"], len(pool)), replace=False).tolist())
    texts = load_whole_notes(ids, config_path)

    rows = []
    model_path = cfg["arm3"]["models"][jcfg["model"]]["path"]
    with LlamaServer(model_path, port=jcfg["port"], ctx=jcfg["ctx"]) as server:
        for hadm_id, note in zip(texts["hadm_id"], texts["text"], strict=True):
            cut = len(note) > jcfg["max_note_chars"]
            note = note[: jcfg["max_note_chars"]]
            t = time.perf_counter()
            note_passages = passages(note)
            answer, timings = ask(server.url, judge_messages(note), jcfg["max_tokens"],
                                  judge_schema(note_passages))
            seconds = time.perf_counter() - t
            evidence = answer["evidence"] if answer else []
            rows.append({"hadm_id": hadm_id, "verdict": answer["verdict"] if answer else None,
                         "evidence": evidence, **check_evidence(note, evidence),
                         "seconds": seconds, "chars": len(note), "cut": cut,
                         "n_passages": len(note_passages),
                         "rejected": timings.get("rejected", False)})
    out = pd.DataFrame(rows)
    path = Path(cfg["paths"]["judge"]) / f"timing-validation-{label}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)

    sec = out["seconds"]
    documented = out["verdict"] == "Documented"
    print(f"gap-set pool (both arms, top {tcfg['depth']}x coded count, uncoded): {len(pool):,} notes; "
          f"judged a random {len(out):,}")
    print(f"seconds per note: median {sec.median():.1f}, p90 {sec.quantile(0.9):.1f}; "
          f"about {sec.mean() * 1000 / 3600:.1f} h per 1,000 notes; whole pool about "
          f"{sec.mean() * len(pool) / 3600:.1f} h")
    print("verdicts:", out["verdict"].fillna("(no valid answer)").value_counts().to_dict())
    print_tiers(out["tiers"])
    print(f"Documented with no found quote in the history sections: "
          f"{int((documented & ~out['in_history']).sum()):,} of {int(documented.sum()):,}")
    print(f"notes cut to {jcfg['max_note_chars']:,} characters: {int(out['cut'].sum()):,}; "
          f"refused as too long for the context: {int(out['rejected'].sum()):,}; "
          f"longest note {int(out['chars'].max()):,} characters")
    return out

# --- re-checking saved answers (counts only) -------------------------------------------


def print_tiers(tiers: pd.Series) -> None:
    """Quote counts per faithfulness tier, from a column of per-note tier lists."""
    counts = Counter(t for note_tiers in tiers for t in note_tiers)
    total = sum(counts.values())
    made_up = sum(counts[t] for t in MADE_UP)
    print(f"quotes: {total:,}; made up (edited + not copied): {made_up:,}")
    print("  " + ", ".join(f"{t} {counts[t]:,}" for t in (*FOUND, *MADE_UP, "empty")))


def run_recheck(config_path: str | Path = "config/config.yaml", label: str = "dementia_core") -> pd.DataFrame:
    """The timing run's saved answers, checked again with the current rules. Counts only.

    Saves nothing new: the judge is not asked again, only its quotes are re-checked.
    """
    cfg = load_config(config_path)
    jcfg = cfg["judge"]
    runs = pd.read_parquet(Path(cfg["paths"]["judge"]) / f"timing-validation-{label}.parquet")
    texts = load_whole_notes(runs["hadm_id"].tolist(), config_path).set_index("hadm_id")["text"]
    checks = pd.DataFrame([
        check_evidence(texts[h][: jcfg["max_note_chars"]], list(ev))
        for h, ev in zip(runs["hadm_id"], runs["evidence"], strict=True)
    ])
    documented = (runs["verdict"] == "Documented").to_numpy()
    print_tiers(checks["tiers"])
    print(f"Documented with no found quote in the history sections: "
          f"{int((documented & ~checks['in_history'].to_numpy()).sum()):,} of {int(documented.sum()):,}")
    return checks



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="The LLM judge (Week 3).")
    parser.add_argument("--timing", action="store_true", help="timing run on validation gap-set notes")
    parser.add_argument("--recheck", action="store_true", help="re-check the timing run's quotes (counts)")
    args = parser.parse_args()
    if args.timing:
        run_timing()
    elif args.recheck:
        run_recheck()
    else:
        parser.print_help()