"""The LLM judge, on invented notes."""

import io
import json
import urllib.error
import urllib.request

import duckdb
import numpy as np
import pandas as pd
import pytest

from dementia_undercoding.judge import (
    VERDICTS,
    arm_bands,
    ask,
    band_names,
    bands_table,
    certain_mentions,
    check_evidence,
    cross_check,
    draw_cross_check_reads,
    draw_test_reads,
    flagged,
    judge_messages,
    judge_schema,
    judged_so_far,
    locate,
    passages,
    read_pilot_reads,
    whole_notes,
)

# Invented notes. The judge must get the whole note, hospital course included.
FAKE_DISCHARGE = '''note_id,hadm_id,text
n1,101,"History of Present Illness:
Per daughter, forgetful ___.

Brief Hospital Course:
Baseline dementia per family."
n2,102,"Past Medical History:
Hypertension"
'''


@pytest.fixture
def con(tmp_path):
    discharge = tmp_path / "discharge.csv"
    discharge.write_text(FAKE_DISCHARGE)
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE cohort AS
        SELECT * FROM (VALUES (101, 'n1'), (102, 'n2')) AS t(hadm_id, note_id)
    """)
    return con, discharge


def test_whole_note_cleaned_and_sorted(con):
    con, discharge = con
    out = whole_notes(con, discharge, [102, 101, 101])   # out of order, repeated
    assert out["hadm_id"].tolist() == [101, 102]
    text = out.loc[out["hadm_id"] == 101, "text"].item()
    assert "Baseline dementia per family" in text         # hospital course kept
    assert "___" not in text                              # placeholder cleaned away


def test_missing_admission_fails(con):
    con, discharge = con
    with pytest.raises(ValueError, match="1 notes found"):
        whole_notes(con, discharge, [101, 999])



def test_schema_allows_only_the_notes_passages():
    schema = judge_schema(["Forgetful at home.", "Lives alone."])
    assert list(schema["properties"]) == ["evidence", "verdict"]          # evidence first
    assert schema["properties"]["evidence"]["items"]["enum"] == ["Forgetful at home.", "Lives alone."]
    assert schema["properties"]["verdict"]["enum"] == list(VERDICTS)
    assert judge_schema([])["properties"]["evidence"]["maxItems"] == 0      # nothing to cite


PASSAGE_NOTE = """History of Present Illness:
Today she says she
is at home and that it is , she describes
her week clearly. Per son, she is more
settled. Per son, she is more
settled.

Past Medical History:
- Hypertension
- Memory loss
1. Atrial fibrillation"""


def test_passages_are_sentences_and_list_lines():
    assert passages(PASSAGE_NOTE) == [                                      # bare headings left out
        "Today she says she is at home and that it is , she describes her week clearly.",
        "Per son, she is more settled.",                                     # repeat kept once
        "- Hypertension",
        "- Memory loss",
        "1. Atrial fibrillation",
    ]


def test_every_passage_is_found_exactly():
    for p in passages(PASSAGE_NOTE):
        assert locate(PASSAGE_NOTE, p)[0] == "exact"

def test_messages_are_rubric_then_note():
    msgs = judge_messages("Invented note.")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"].endswith("Invented note.")


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_server(content):
    body = {"choices": [{"message": {"content": content}}], "timings": {"predicted_n": 5}}
    return lambda req, timeout: FakeResponse(json.dumps(body).encode())


def test_ask_parses_answer(monkeypatch):
    answer = '{"evidence": ["forgetful"], "verdict": "Documented"}'
    monkeypatch.setattr(urllib.request, "urlopen", fake_server(answer))
    parsed, timings = ask("http://x", judge_messages("n"), 10, judge_schema([]))
    assert parsed == {"evidence": ["forgetful"], "verdict": "Documented"}
    assert timings == {"predicted_n": 5}


def test_ask_cut_off_answer_is_none(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", fake_server('{"evidence": ["forgetf'))
    parsed, _ = ask("http://x", judge_messages("n"), 10, judge_schema([]))
    assert parsed is None        


def test_ask_too_long_is_none_and_flagged(monkeypatch):
    def refuse(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    parsed, timings = ask("http://x", judge_messages("n"), 10, judge_schema([]))
    assert parsed is None and timings == {"rejected": True}


def test_flagged_takes_uncoded_from_the_top():
    val = pd.DataFrame({"hadm_id": [1, 2, 3, 4, 5, 6], "dementia_core": [1, 0, 1, 0, 0, 0]})
    score = np.array([0.9, 0.8, 0.1, 0.7, 0.6, 0.2])
    # 2 coded notes; depth 2 -> top 4 by score: 1, 2, 4, 5; uncoded among them: 2, 4, 5
    assert flagged(val, score, "dementia_core", 2).tolist() == [2, 4, 5]
    # depth 1 -> top 2: 1, 2; uncoded: 2
    assert flagged(val, score, "dementia_core", 1).tolist() == [2]

NOTE = """Chief Complaint:
Fall

History of Present Illness:
Per daughter, increasingly
forgetful at home.

Past Medical History:
- Hypertension
- Memory loss

Brief Hospital Course:
At baseline per family she is oriented to self only."""


@pytest.mark.parametrize("quote, tier", [
    ("Per daughter, increasingly forgetful at home.", "exact"),           # across a line break
    ("per Daughter, increasingly forgetful", "formatting"),               # capitals
    ("Hypertension, Memory loss", "formatting"),                          # list dashes dropped
    ("increasingly forgetful... oriented to self only", "joined"),
    ("Per daughter, increasingly forgetful at home and at work.", "edited"),
    ("The patient has dementia.", "not copied"),
    ("forget", "not copied"),                                             # not inside "forgetful"
    ("  ", "empty"),
])
def test_tiers(quote, tier):
    assert locate(NOTE, quote)[0] == tier


def test_sections_and_history():
    out = check_evidence(NOTE, ["At baseline per family she is oriented to self only."])
    assert out["sections"] == ["Brief Hospital Course"]
    assert not out["in_history"]
    out = check_evidence(NOTE, ["Hypertension, Memory loss"])
    assert out["sections"] == ["Past Medical History"]
    assert out["in_history"]


def test_joined_quote_lists_every_section():
    out = check_evidence(NOTE, ["increasingly forgetful... oriented to self only"])
    assert out["sections"] == ["History of Present Illness + Brief Hospital Course"]
    assert out["in_history"]                                 # one piece is in the history
    assert out["n_not_found"] == 0


def test_counts():
    out = check_evidence(NOTE, ["Per daughter,", "The patient has dementia.", "Per daughter, increasingly "
                                "forgetful at home and at work.", " "])
    assert out["n_quotes"] == 4
    assert out["n_not_found"] == 3                           # not copied, edited, empty
    assert out["n_made_up"] == 2                             # not copied, edited


def test_quote_before_any_heading():
    out = check_evidence("Name: ___\nUnit No: ___\n\n" + NOTE, ["Unit No:"])
    assert out["sections"] == ["(before any heading)"]


def test_capitals_format_uses_fallback_headings():
    old = "HISTORY OF PRESENT ILLNESS:\nWife reports memory loss for years.\n\nHOSPITAL COURSE:\nStable."
    out = check_evidence(old, ["Wife reports memory loss for years.", "Stable."])
    assert out["sections"] == ["History of Present Illness", "HOSPITAL COURSE"]
    assert out["in_history"]



def test_read_pilot_reads_normalises_and_checks(tmp_path):
    path = tmp_path / "reads.csv"
    path.write_text("pilot_id,verdict,comment\nP01, documented ,\nP02,Acute only,odd one\nP03,NONE,\n")
    assert read_pilot_reads(path)["mine"].tolist() == ["Documented", "Acute only", "None"]
    path.write_text("pilot_id,verdict,comment\nP01,Documented,\nP02,,\nP03,maybe,\n")
    with pytest.raises(ValueError, match="P02, P03"):
        read_pilot_reads(path)

# --- the arm 0 cross-check (invented sentences) ------------------------------------------


@pytest.mark.parametrize("note, expected", [
    ("Past Medical History: dementia, hypertension.", 1),
    ("Alzheimer's disease. Baseline dementia per daughter.", 2),
    ("No history of dementia.", 0),
    ("Mother had dementia.", 0),
    ("Concern for dementia given recent forgetfulness.", 0),
    ("Hypertension, atrial fibrillation.", 0),
])
def test_certain_mentions(note, expected):
    assert certain_mentions(note, window=6) == expected


def test_cross_check_flags_certain_mention_without_documented():
    judged = pd.DataFrame({"hadm_id": [1, 2, 3, 4, 5],
                           "verdict": ["Possible", "Documented", "None", None, "Acute only"]})
    texts = pd.DataFrame({"hadm_id": [5, 4, 3, 2, 1], "text": [
        "Concern for dementia.",                  # 5: uncertain only, not flagged
        "PMH: dementia.",                          # 4: no valid answer counts as not Documented
        "PMH: hypertension.",                      # 3: no mention
        "PMH: dementia.",                          # 2: the judge said Documented
        "PMH: dementia.",                          # 1: flagged
    ]})
    out = cross_check(judged, texts, window=6, max_chars=1000)
    assert list(out.columns) == ["hadm_id", "verdict", "n_certain", "n_certain_drug", "flagged", "drug_only"]
    assert out.set_index("hadm_id")["flagged"].to_dict() == {1: True, 2: False, 3: False, 4: True, 5: False}


def test_cross_check_counts_drug_only_notes_but_never_flags_them():
    judged = pd.DataFrame({"hadm_id": [1, 2, 3, 4, 5],
                           "verdict": ["None", "Documented", "Possible", "None", "None"]})
    texts = pd.DataFrame({"hadm_id": [1, 2, 3, 4, 5], "text": [
        "Home medications: Aricept 10 mg daily.",  # 1: drug only, counted
        "Home medications: donepezil.",            # 2: the judge said Documented
        "PMH: dementia. Meds: memantine.",         # 3: a diagnosis mention: flagged, not drug only
        "Not on donepezil.",                       # 4: negated drug mention
        "Meds: lisinopril.",                       # 5: no mention
    ]})
    out = cross_check(judged, texts, window=6, max_chars=1000).set_index("hadm_id")
    assert out["drug_only"].to_dict() == {1: True, 2: False, 3: False, 4: False, 5: False}
    assert out["flagged"].to_dict() == {1: False, 2: False, 3: True, 4: False, 5: False}


def test_cross_check_reads_only_what_the_judge_saw():
    judged = pd.DataFrame({"hadm_id": [1], "verdict": ["None"]})
    texts = pd.DataFrame({"hadm_id": [1], "text": ["x" * 50 + " PMH: dementia."]})
    assert not cross_check(judged, texts, window=6, max_chars=50)["flagged"].iloc[0]
    assert cross_check(judged, texts, window=6, max_chars=100)["flagged"].iloc[0]


def test_cross_check_fails_on_missing_text():
    judged = pd.DataFrame({"hadm_id": [1, 2], "verdict": ["None", "None"]})
    texts = pd.DataFrame({"hadm_id": [1], "text": ["PMH: dementia."]})
    with pytest.raises(ValueError, match="without text"):
        cross_check(judged, texts, window=6, max_chars=100)


def _checks(pool_ids, sample_ids):
    def one(ids):
        return pd.DataFrame({"hadm_id": ids, "flagged": [True] * len(ids)})
    return {"pool": one(pool_ids), "sample": one(sample_ids)}


def test_draw_takes_all_when_under_the_cap():
    reads = draw_cross_check_reads(_checks([3, 1], [2]), cap=50, seed=12)
    assert list(reads.columns) == ["read_id", "hadm_id", "set"]
    assert sorted(reads["hadm_id"]) == [1, 2, 3]
    assert list(reads["read_id"]) == ["X01", "X02", "X03"]
    assert reads.set_index("hadm_id")["set"].to_dict() == {1: "pool", 2: "sample", 3: "pool"}


def test_draw_caps_and_repeats_with_the_seed():
    checks = _checks(list(range(100)), list(range(100, 130)))
    first = draw_cross_check_reads(checks, cap=50, seed=12)
    assert len(first) == 50 and first["hadm_id"].is_unique
    assert first.equals(draw_cross_check_reads(checks, cap=50, seed=12))


def test_draw_skips_unflagged_and_rejects_overlap():
    checks = {"pool": pd.DataFrame({"hadm_id": [1, 2], "flagged": [True, False]}),
              "sample": pd.DataFrame({"hadm_id": [3], "flagged": [True]})}
    assert sorted(draw_cross_check_reads(checks, cap=50, seed=12)["hadm_id"]) == [1, 3]
    with pytest.raises(ValueError, match="more than one set"):
        draw_cross_check_reads(_checks([1], [1]), cap=50, seed=12)


# --- the test run: bands and the reads draw (invented numbers, no text) -------------------


def _ranked(n_coded=2, n=20):
    """n notes sorted by hadm_id; the first n_coded are coded."""
    return pd.DataFrame({"hadm_id": np.arange(1, n + 1),
                         "dementia_core": [1] * n_coded + [0] * (n - n_coded)})


def test_band_names():
    assert band_names([1, 2, 4]) == ["1x", "1-2x", "2-4x"]
    assert band_names([1, 1.5]) == ["1x", "1-1.5x"]


def test_arm_bands_cut_at_multiples_of_the_coded_count():
    notes = _ranked()                                      # 2 coded: top 2 / 4 / 8
    score = -notes["hadm_id"].to_numpy().astype(float)     # rank = hadm_id order
    band = arm_bands(notes, score, "dementia_core", [1, 2, 4])
    # top 2 are the coded notes 1-2, so 1x holds no uncoded note; 3-4 are 1-2x; 5-8 are 2-4x
    assert band.loc[3] == "1-2x" and band.loc[4] == "1-2x"
    assert list(band.loc[5:8]) == ["2-4x"] * 4
    assert band.loc[9:].isna().all()
    assert 1 not in band.index                             # coded notes are never in the table


def test_bands_table_takes_the_shallower_band():
    notes = _ranked(n_coded=1, n=10)                       # 1 coded: top 1 / 2 / 4
    a = -notes["hadm_id"].to_numpy().astype(float)         # arm A ranks 1, 2, 3, ...
    b = notes["hadm_id"].to_numpy().astype(float)          # arm B ranks 10, 9, 8, ...
    table = bands_table(notes, {"A": a, "B": b}, "dementia_core", [1, 2, 4]).set_index("hadm_id")
    assert table.loc[2, "A"] == "1-2x" and pd.isna(table.loc[2, "B"]) and table.loc[2, "band"] == "1-2x"
    assert table.loc[10, "B"] == "1x" and table.loc[10, "band"] == "1x"
    assert table.loc[9, "B"] == "1-2x" and table.loc[9, "band"] == "1-2x"
    assert pd.isna(table.loc[6, "band"])                   # inside neither arm's 4x


def test_draw_test_reads_by_band_without_topping_up():
    bands = pd.DataFrame({"hadm_id": range(1, 101),
                          "band": ["1x"] * 50 + ["1-2x"] * 40 + ["2-4x"] * 5 + [None] * 5})
    reads = draw_test_reads(bands, {"1x": 35, "1-2x": 15, "2-4x": 10}, seed=59)
    assert reads["band"].value_counts().to_dict() == {"1x": 35, "1-2x": 15, "2-4x": 5}   # short band: all 5
    assert reads["hadm_id"].is_unique and list(reads.columns) == ["read_id", "hadm_id", "band"]
    assert reads["read_id"].iloc[0] == "T01"
    assert reads.equals(draw_test_reads(bands, {"1x": 35, "1-2x": 15, "2-4x": 10}, seed=59))
    assert reads["band"].iloc[:35].nunique() > 1           # shuffled: not all 1x first


def test_judged_so_far_reads_part_files(tmp_path):
    assert judged_so_far(tmp_path / "parts") == set()
    (tmp_path / "parts").mkdir()
    pd.DataFrame({"hadm_id": [3, 1]}).to_parquet(tmp_path / "parts" / "part-00000.parquet")
    pd.DataFrame({"hadm_id": [7]}).to_parquet(tmp_path / "parts" / "part-00001.parquet")
    assert judged_so_far(tmp_path / "parts") == {1, 3, 7}