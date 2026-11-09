"""The LLM judge (Week 3): reads whole discharge notes for flagged admissions and gives a verdict.

The arms read only the history sections; the judge reads the whole note (rubric in STATUS.md).
Nothing here prints or logs note text. Anything that holds note text is saved under data/ only.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterator
from difflib import SequenceMatcher
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .arm0_regex import DIAGNOSIS, DRUGS, count_mentions
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
    "note documents, using the whole note. Only the patient's own cognition counts: dementia or "
    "memory problems in a spouse, parent or anyone else is never evidence about the patient.\n\n"
    "Decide in this order and stop at the first verdict that fits:\n"
    "1. Documented: the note states dementia for the patient (any type, including Alzheimer's "
    "disease), anywhere in the note and even in a list with other conditions, or clearly "
    "describes cognitive impairment that predates this admission, for example memory loss over "
    "months or years reported by family, or needing help with daily activities because of memory "
    "or thinking. A cognitive problem during this admission does not change this verdict.\n"
    "2. Possible: the note raises a cognitive concern, but it is unclear whether it predates this "
    "admission or how severe it is. Mild cognitive impairment goes here. A history of memory "
    "problems, or any hint that a cognitive problem was present before this admission, goes here, "
    "as does a patient called a poor historian with no other reason given (such as sedation, "
    "intubation, language or hearing).\n"
    "3. Acute only: the note describes an acute cognitive problem during this admission and treats "
    "it as a problem: it names delirium or encephalopathy, or describes an assessment or treatment "
    "aimed at the confusion (not only at the illness causing it). There must be such a problem.\n"
    "4. None: anything else. No cognitive impairment is documented, or it is denied, ruled out, or "
    "describes someone other than the patient. Normal cognition (alert, oriented, clear and "
    "coherent) with no cognitive problem is None. Words such as \"confused\", \"lethargic\" or "
    "\"altered\" in passing, in a very sick patient whose note does not treat the confusion itself "
    "as a problem, are None.\n\n"
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



# --- gap sets and judging a set of notes ---------------------------------------------------


def flagged(val: pd.DataFrame, score: np.ndarray, label: str, depth: float) -> pd.Series:
    """hadm_ids of the uncoded notes among an arm's top (depth x coded count) notes.

    val must be sorted by hadm_id, so tied scores are broken the same way every run.
    """
    k = round(depth * int(val[label].sum()))
    top = val.assign(score=score).nlargest(k, "score", keep="first")
    return top.loc[top[label] == 0, "hadm_id"]


def validation_set(config_path: str | Path, label: str) -> pd.DataFrame:
    """Validation notes, labels only (no text), sorted by hadm_id."""
    notes = load_notes(config_path, columns=[label])
    return notes[notes["split"] == "validation"].sort_values("hadm_id").reset_index(drop=True)


def gap_pool(cfg: dict, val: pd.DataFrame, label: str, depth: float) -> set[int]:
    """Uncoded notes in either gap-set arm's top (depth x coded count): each note once."""
    pool: set[int] = set()
    for stem in cfg["judge"]["arms"].values():
        s = pd.read_parquet(Path(cfg["paths"]["scores"]) / f"{stem}-{label}.parquet")
        score = val[["hadm_id"]].merge(s, on="hadm_id", how="left", validate="one_to_one")["score"]
        if score.isna().any():
            raise ValueError(f"{stem}: {int(score.isna().sum()):,} validation notes without a score")
        pool |= set(flagged(val, score.to_numpy(), label, depth).tolist())
    return pool


def judge_notes(cfg: dict, texts: pd.DataFrame) -> pd.DataFrame:
    """The judge on each note (columns hadm_id, text), one at a time. One row per note."""
    return pd.DataFrame(list(judge_rows(cfg, texts)))


def judge_rows(cfg: dict, texts: pd.DataFrame) -> Iterator[dict]:
    """The judge on each note (columns hadm_id, text), one at a time: yields one row per note
    as soon as it is judged, so a long run can save as it goes. One server for the whole run."""
    jcfg = cfg["judge"]
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
            picked = answer["evidence"] if answer else []
            evidence = list(dict.fromkeys(picked))     # llama-server ignores uniqueItems: drop repeats
            yield {"hadm_id": hadm_id, "verdict": answer["verdict"] if answer else None,
                   "evidence": evidence, "n_repeats": len(picked) - len(evidence),
                   **check_evidence(note, evidence),
                   "seconds": seconds, "chars": len(note), "cut": cut,
                   "n_passages": len(note_passages),
                   "rejected": timings.get("rejected", False)}

# --- the arm 0 cross-check (STATUS item 12, decided 2026-11-09 before test was touched) ----


def certain_mentions(note: str, window: int, pattern: re.Pattern = DIAGNOSIS) -> int:
    """Certain arm 0 mentions in a note: not negated, not someone else's, not uncertain.

    pattern: arm 0's diagnosis terms (the cross-check) or its drug terms (the drug-only count).
    """
    return count_mentions(note, pattern, window)["certain"]


def _certain_one(args: tuple[str, int]) -> tuple[int, int]:
    note, window = args
    return certain_mentions(note, window), certain_mentions(note, window, DRUGS)


def cross_check(judged: pd.DataFrame, texts: pd.DataFrame, window: int, max_chars: int) -> pd.DataFrame:
    """Each judged note with its certain arm 0 mentions, and whether it is flagged.

    judged: hadm_id, verdict (the judge's answers). texts: hadm_id, text (the whole notes).
    Arm 0 reads the text the judge saw: the note cut at max_chars. Flagged = at least one
    certain diagnosis mention and the judge did not say Documented (no valid answer counts
    as not). Drug only = a certain drug mention, no certain diagnosis mention and not
    Documented: counted, never read. Returns hadm_id, verdict, n_certain, n_certain_drug,
    flagged, drug_only; no note text.
    """
    df = judged[["hadm_id", "verdict"]].merge(texts[["hadm_id", "text"]], on="hadm_id",
                                              how="left", validate="one_to_one")
    if df["text"].isna().any():
        raise ValueError(f"{int(df['text'].isna().sum()):,} judged notes without text")
    workers = max(1, (os.cpu_count() or 2) - 2)
    with Pool(workers) as pool:
        counts = pool.map(_certain_one, [(t[:max_chars], window) for t in df["text"]], chunksize=50)
    df["n_certain"] = [dx for dx, _ in counts]
    df["n_certain_drug"] = [rx for _, rx in counts]
    not_documented = df["verdict"] != "Documented"
    df["flagged"] = (df["n_certain"] > 0) & not_documented
    df["drug_only"] = (df["n_certain"] == 0) & (df["n_certain_drug"] > 0) & not_documented
    return df.drop(columns="text")


def draw_cross_check_reads(checks: dict[str, pd.DataFrame], cap: int, seed: int) -> pd.DataFrame:
    """The flagged notes I read: all of them, or a random `cap` from the combined list.

    checks: set name -> cross_check() output, e.g. {"pool": ..., "sample": ...}. Returns
    read_id, hadm_id, set in a random order (so the reading order hides the set). The
    judge's verdict is left out, so the reads are blind to it.
    """
    flags = pd.concat([c.loc[c["flagged"], ["hadm_id"]].assign(set=name) for name, c in checks.items()])
    if flags["hadm_id"].duplicated().any():
        raise ValueError("a note is in more than one set")
    flags = flags.sort_values("hadm_id").reset_index(drop=True)      # fixed order before the draw
    rng = np.random.default_rng(seed)
    n = min(cap, len(flags))
    picked = flags.iloc[rng.choice(len(flags), size=n, replace=False)].reset_index(drop=True)
    picked.insert(0, "read_id", [f"X{i + 1:02d}" for i in range(n)])
    return picked


# --- the test run (STATUS item 11, decided 2026-11-09 before test was touched) -----------
# Draw first (pool, bands, the random sample, my 60 reads), then judge. The judge run saves
# as it goes, resumes where it stopped, and prints progress only: no verdict is shown.


def test_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["judge"]) / "test"


def test_set(config_path: str | Path, label: str) -> pd.DataFrame:
    """Test notes, labels only (no text), sorted by hadm_id."""
    notes = load_notes(config_path, columns=[label])
    return notes[notes["split"] == "test"].sort_values("hadm_id").reset_index(drop=True)


def band_names(depths: list[float]) -> list[str]:
    """[1, 2, 4] -> ["1x", "1-2x", "2-4x"]."""
    names, low = [], None
    for d in depths:
        names.append(f"{d:g}x" if low is None else f"{low:g}-{d:g}x")
        low = d
    return names


def arm_bands(notes: pd.DataFrame, score: np.ndarray, label: str, depths: list[float]) -> pd.Series:
    """Each uncoded note's band for one arm (index hadm_id): the shallowest depth whose top
    (depth x coded count) holds it, named as band_names(); missing = below the last depth."""
    band = pd.Series(index=notes.loc[notes[label] == 0, "hadm_id"], dtype=object)
    for depth, name in reversed(list(zip(depths, band_names(depths), strict=True))):
        band[flagged(notes, score, label, depth).to_numpy()] = name   # shallower bands overwrite
    return band


def bands_table(notes: pd.DataFrame, scores: dict[str, np.ndarray], label: str,
                depths: list[float]) -> pd.DataFrame:
    """Uncoded notes with each arm's band and the note's band = the shallower of the two."""
    order = band_names(depths)
    table = pd.DataFrame({arm: arm_bands(notes, s, label, depths) for arm, s in scores.items()})
    rank = table.apply(lambda col: col.map({b: i for i, b in enumerate(order)}))
    table["band"] = rank.min(axis=1).map(dict(enumerate(order)))       # NaN only if no arm has it
    return table.rename_axis("hadm_id").reset_index()


def draw_test_reads(bands: pd.DataFrame, quotas: dict[str, int], seed: int) -> pd.DataFrame:
    """My blind reads: quotas[band] notes drawn at random within each band (all of them if the
    band is short; no topping up), shuffled. Returns read_id, hadm_id, band."""
    rng = np.random.default_rng(seed)
    picked = []
    for band, k in quotas.items():
        ids = np.sort(bands.loc[bands["band"] == band, "hadm_id"].to_numpy())
        for h in rng.choice(ids, size=min(k, len(ids)), replace=False):
            picked.append((int(h), band))
    order = rng.permutation(len(picked))                     # reading order hides the band
    return pd.DataFrame([{"read_id": f"T{i + 1:02d}", "hadm_id": picked[j][0], "band": picked[j][1]}
                         for i, j in enumerate(order)])


def test_scores(cfg: dict, notes: pd.DataFrame, label: str) -> dict[str, np.ndarray]:
    """Each gap-set arm's test scores, in the row order of notes (from score_test.py)."""
    out = {}
    for arm, stem in cfg["judge"]["arms"].items():
        s = pd.read_parquet(Path(cfg["paths"]["scores"]) / "test" / f"{stem}-{label}.parquet")
        score = notes[["hadm_id"]].merge(s, on="hadm_id", how="left", validate="one_to_one")["score"]
        if score.isna().any():
            raise ValueError(f"{stem}: {int(score.isna().sum()):,} test notes without a score")
        out[arm] = score.to_numpy()
    return out


def run_test_draw(config_path: str | Path = "config/config.yaml", label: str = "dementia_core") -> None:
    """Bands, the random sample below the last band, my 60 reads, and the judging order.

    Run once, before the judge: refuses if the draw exists. Prints counts only.
    """
    cfg = load_config(config_path)
    tcfg = cfg["judge"]["test"]
    out = test_dir(cfg)
    if (out / "reads.csv").exists():
        raise FileExistsError(f"{out / 'reads.csv'} exists: the test draw is done. "
                              f"Delete it on purpose to redraw.")

    notes = test_set(config_path, label)
    bands = bands_table(notes, test_scores(cfg, notes, label), label, tcfg["depths"])
    below = np.sort(bands.loc[bands["band"].isna(), "hadm_id"].to_numpy())
    sample = np.random.default_rng(tcfg["sample_seed"]).choice(below, size=min(tcfg["sample_n"], len(below)),
                                                               replace=False)
    bands["in_sample"] = bands["hadm_id"].isin(sample)
    reads = draw_test_reads(bands, tcfg["reads"], tcfg["reads_seed"])

    to_judge = np.sort(bands.loc[bands["band"].notna() | bands["in_sample"], "hadm_id"].to_numpy())
    to_judge = np.random.default_rng(tcfg["order_seed"]).permutation(to_judge)   # a stop leaves a fair mix

    texts = load_whole_notes(reads["hadm_id"].tolist(), config_path).set_index("hadm_id")["text"]
    out.mkdir(parents=True, exist_ok=True)
    bands.to_parquet(out / "bands.parquet", index=False)
    pd.DataFrame({"hadm_id": to_judge}).to_parquet(out / "to-judge.parquet", index=False)
    reads.to_parquet(out / "reads-key.parquet", index=False)
    with open(out / "reads-notes.txt", "w") as f:
        for r in reads.itertuples():
            note = texts[r.hadm_id][: cfg["judge"]["max_note_chars"]]
            f.write(f"{'=' * 70}\n{r.read_id}\n{'=' * 70}\n\n{note}\n\n")
    reads[["read_id"]].assign(verdict="", comment="").to_csv(out / "reads.csv", index=False)

    coded = int(notes[label].sum())
    print(f"test: {len(notes):,} notes, {coded:,} coded; bands at {tcfg['depths']} x coded count")
    print("pool (uncoded, either arm), by the shallower band:",
          bands["band"].value_counts().reindex(band_names(tcfg["depths"]), fill_value=0).to_dict())
    print(f"random sample: {len(sample):,} of {len(below):,} uncoded notes below the last band")
    print("my reads by band:",
          reads["band"].value_counts().reindex(list(tcfg["reads"]), fill_value=0).to_dict())
    print(f"to judge: {len(to_judge):,} notes; at {tcfg['seconds_per_note']} s a note about "
          f"{len(to_judge) * tcfg['seconds_per_note'] / 3600:.0f} h")
    print(f"read:  {out / 'reads-notes.txt'}\nfill:  {out / 'reads.csv'}  (verdict: {', '.join(VERDICTS)})")


def judged_so_far(parts: Path) -> set[int]:
    if not parts.exists():
        return set()
    return {int(h) for f in sorted(parts.glob("part-*.parquet"))
            for h in pd.read_parquet(f, columns=["hadm_id"])["hadm_id"]}


def run_test_judge(config_path: str | Path = "config/config.yaml") -> None:
    """The frozen judge on every note in to-judge.parquet, in that order. Saves a part file
    every save_every notes; run again after a stop and it carries on with the notes not yet
    judged. Prints progress only: no verdict, no quote."""
    cfg = load_config(config_path)
    tcfg = cfg["judge"]["test"]
    out = test_dir(cfg)
    parts = out / "parts"
    order = pd.read_parquet(out / "to-judge.parquet")["hadm_id"].tolist()
    done = judged_so_far(parts)
    left = [h for h in order if h not in done]
    print(f"{len(order):,} notes to judge; {len(done):,} done before; {len(left):,} left", flush=True)
    if not left:
        return
    texts = load_whole_notes(left, config_path).set_index("hadm_id").loc[left].reset_index()

    parts.mkdir(parents=True, exist_ok=True)
    k = len(list(parts.glob("part-*.parquet")))
    batch, n, failed, seconds = [], 0, 0, []
    start = time.perf_counter()
    for row in judge_rows(cfg, texts):
        batch.append(row)
        n += 1
        failed += row["verdict"] is None
        seconds.append(row["seconds"])
        if len(batch) == tcfg["save_every"] or n == len(left):
            pd.DataFrame(batch).to_parquet(parts / f"part-{k:05d}.parquet", index=False)
            k, batch = k + 1, []
            elapsed = time.perf_counter() - start
            print(f"{len(done) + n:,}/{len(order):,} judged; {np.median(seconds):.1f} s a note (median); "
                  f"about {elapsed / n * (len(left) - n) / 3600:.1f} h left; "
                  f"no valid answer so far: {failed:,}", flush=True)
    print("finished: answers saved under data/, not shown", flush=True)

    
# --- the timing run (validation, development only) --------------------------------------


def run_timing(config_path: str | Path = "config/config.yaml", label: str = "dementia_core") -> pd.DataFrame:
    """The judge on a fixed random sample of validation gap-set notes. Prints counts only.

    The gap set here is provisional (both arms, one depth from config) because the bands
    are not fixed yet; this run measures speed and checks the machinery, nothing more.
    Per-note results, quotes included, are saved under data/ only.
    """
    cfg = load_config(config_path)
    jcfg, tcfg = cfg["judge"], cfg["judge"]["timing"]
    pool = gap_pool(cfg, validation_set(config_path, label), label, tcfg["depth"])

    rng = np.random.default_rng(tcfg["seed"])
    ids = sorted(rng.choice(sorted(pool), size=min(tcfg["n"], len(pool)), replace=False).tolist())
    out = judge_notes(cfg, load_whole_notes(ids, config_path))
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


# --- the pilot (validation, development only) ---------------------------------------------
# Draw: notes from two depths, timing-run notes left out (seen with a verdict, so not blind).
# I read pilot-notes.txt and fill in pilot-reads.csv before looking at anything else; the
# judge's answers go to a separate file; compare prints counts only.


def pilot_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["judge"]) / "pilot"


def run_pilot_draw(config_path: str | Path = "config/config.yaml", label: str = "dementia_core") -> None:
    """Draw the pilot notes and write the reading file and the blank reads sheet."""
    cfg = load_config(config_path)
    jcfg, pcfg = cfg["judge"], cfg["judge"]["pilot"]
    pdir = pilot_dir(cfg)
    reads = pdir / "pilot-reads.csv"
    if reads.exists():
        raise FileExistsError(f"{reads} exists: the pilot is already drawn. Delete it on purpose to redraw.")

    val = validation_set(config_path, label)
    seen = set(pd.read_parquet(Path(cfg["paths"]["judge"]) / f"timing-validation-{label}.parquet")["hadm_id"])
    shallow_depth, deep_depth = pcfg["depths"]
    shallow_all = gap_pool(cfg, val, label, shallow_depth)
    shallow = sorted(shallow_all - seen)
    deep = sorted(gap_pool(cfg, val, label, deep_depth) - shallow_all - seen)

    rng = np.random.default_rng(pcfg["seed"])
    k = pcfg["n_per_stratum"]
    picked = ([(h, f"top {shallow_depth}x") for h in rng.choice(shallow, k, replace=False)]
              + [(h, f"{shallow_depth}x-{deep_depth}x") for h in rng.choice(deep, k, replace=False)])
    order = rng.permutation(len(picked))                    # reading order hides the depth
    key = pd.DataFrame([{"pilot_id": f"P{i + 1:02d}", "hadm_id": int(picked[j][0]), "stratum": picked[j][1]}
                        for i, j in enumerate(order)])

    texts = load_whole_notes(key["hadm_id"].tolist(), config_path).set_index("hadm_id")["text"]
    pdir.mkdir(parents=True, exist_ok=True)
    key.to_parquet(pdir / "pilot-key.parquet", index=False)
    with open(pdir / "pilot-notes.txt", "w") as f:
        for r in key.itertuples():
            note = texts[r.hadm_id][: jcfg["max_note_chars"]]
            f.write(f"{'=' * 70}\n{r.pilot_id}\n{'=' * 70}\n\n{note}\n\n")
    key[["pilot_id"]].assign(verdict="", comment="").to_csv(reads, index=False)

    print(f"pilot: {k} notes from the top {shallow_depth}x pool ({len(shallow):,} left after taking out "
          f"the timing run's notes) and {k} from {shallow_depth}x-{deep_depth}x ({len(deep):,})")
    print(f"read:  {pdir / 'pilot-notes.txt'}")
    print(f"fill:  {reads}  (verdict: {', '.join(VERDICTS)})")


def run_pilot_judge(config_path: str | Path = "config/config.yaml") -> None:
    """The judge on the pilot notes. Prints timing only, so it can run while I read."""
    cfg = load_config(config_path)
    pdir = pilot_dir(cfg)
    key = pd.read_parquet(pdir / "pilot-key.parquet")
    out = judge_notes(cfg, load_whole_notes(key["hadm_id"].tolist(), config_path))
    out.to_parquet(pdir / "pilot-judge.parquet", index=False)
    print(f"judged {len(out):,} pilot notes, {out['seconds'].median():.1f} s median; "
          f"answers saved, not shown")


def read_pilot_reads(path: str | Path) -> pd.DataFrame:
    """pilot_id and my verdict, spelled as in VERDICTS. Fails on any blank or unknown verdict."""
    reads = pd.read_csv(path, dtype=str, keep_default_na=False)
    canon = {v.lower(): v for v in VERDICTS}
    mine = reads["verdict"].str.strip().str.lower().map(canon)
    if mine.isna().any():
        bad = ", ".join(reads.loc[mine.isna(), "pilot_id"])
        raise ValueError(f"no valid verdict ({', '.join(VERDICTS)}) for: {bad}")
    return pd.DataFrame({"pilot_id": reads["pilot_id"], "mine": mine})


def run_pilot_compare(config_path: str | Path = "config/config.yaml", reads: str = "pilot-reads.csv",
                      answers: str = "pilot-judge.parquet") -> pd.DataFrame:
    """My reads against the judge. Prints counts only. Development: not the reported agreement.

    reads and answers are file names in the pilot folder, so one judge run can be compared
    with my original and my revised reads, and two judge runs with the same reads.
    """
    cfg = load_config(config_path)
    pdir = pilot_dir(cfg)
    key = pd.read_parquet(pdir / "pilot-key.parquet")
    judge = pd.read_parquet(pdir / answers)[["hadm_id", "verdict"]]
    print(f"reads: {reads}; judge answers: {answers}")
    df = (key.merge(read_pilot_reads(pdir / reads), on="pilot_id", validate="one_to_one")
             .merge(judge.rename(columns={"verdict": "judge"}), on="hadm_id", validate="one_to_one"))
    df["judge"] = df["judge"].fillna("(no answer)")

    cols = [*VERDICTS, *(["(no answer)"] if (df["judge"] == "(no answer)").any() else [])]
    table = pd.crosstab(df["mine"], df["judge"]).reindex(index=list(VERDICTS), columns=cols, fill_value=0)
    me_doc, judge_doc = df["mine"] == "Documented", df["judge"] == "Documented"
    print(f"pilot: {len(df):,} notes; rows = my read, columns = the judge\n")
    print(table.to_string(), "\n")
    print(f"same verdict: {int((df['mine'] == df['judge']).sum()):,} of {len(df):,}; "
          f"kappa, all four verdicts: {cohen_kappa_score(df['mine'], df['judge']):.2f}")
    print(f"Documented vs the rest: kappa {cohen_kappa_score(me_doc, judge_doc):.2f}; the judge found "
          f"{int((me_doc & judge_doc).sum())} of my {int(me_doc.sum())} Documented and called "
          f"{int((~me_doc & judge_doc).sum())} of my {int((~me_doc).sum())} others Documented")
    for stratum, part in df.groupby("stratum"):
        print(f"  {stratum}: same verdict {int((part['mine'] == part['judge']).sum())} of {len(part)}")
    print(f"Possible: {int((df['mine'] == 'Possible').sum())} by me, "
          f"{int((df['judge'] == 'Possible').sum())} by the judge")
    return df

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

    commands = {
        "timing": run_timing,
        "recheck": run_recheck,
        "pilot-draw": run_pilot_draw,
        "pilot-judge": run_pilot_judge,
        "pilot-compare": run_pilot_compare,
        "test-draw": run_test_draw,
        "test-judge": run_test_judge,
    }
    parser = argparse.ArgumentParser(description="The LLM judge (Week 3).")
    parser.add_argument("command", choices=list(commands), help="which step to run")
    parser.add_argument("--reads", default="pilot-reads.csv", help="pilot-compare: my reads file")
    parser.add_argument("--answers", default="pilot-judge.parquet", help="pilot-compare: judge answers file")
    args = parser.parse_args()
    if args.command == "pilot-compare":
        run_pilot_compare(reads=args.reads, answers=args.answers)
    else:
        commands[args.command]()