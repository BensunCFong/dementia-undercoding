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
    ask,
    check_evidence,
    flagged,
    judge_messages,
    judge_schema,
    locate,
    passages,
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

