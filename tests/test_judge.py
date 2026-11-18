"""The LLM judge, on invented notes."""

import hashlib
import io
import json
import urllib.error
import urllib.request

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from dementia_undercoding.judge import (
    JUDGE_SYSTEM,
    JUDGE_SYSTEM_V6,
    VERDICTS,
    agreement,
    agreement_rows,
    arm_bands,
    ask,
    band_names,
    band_patient_weights,
    bands_table,
    certain_mentions,
    check_evidence,
    check_output_unlocked,
    cross_check,
    draw_cross_check_reads,
    draw_sample_documented,
    draw_test_reads,
    flagged,
    judge_messages,
    judge_schema,
    judge_text,
    judged_so_far,
    judged_test,
    load_test_answers,
    locate,
    passages,
    read_my_reads,
    trim,
    whole_notes,
    worst_case,
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


# JUDGE_SYSTEM as frozen at ef7d6a2
V4_SHA256 = "2cd23486819fce01d5f8cd815fbd5cc9b3e2be3e94f5c79f3c7778757fa18150"


def test_v4_rubric_is_frozen_and_stays_the_default():
    assert hashlib.sha256(JUDGE_SYSTEM.encode()).hexdigest() == V4_SHA256
    assert judge_messages("Invented note.")[0]["content"] == JUDGE_SYSTEM
    v6 = judge_messages("Invented note.", "v6")
    assert v6[0]["content"] == JUDGE_SYSTEM_V6
    assert v6[1]["content"] == judge_messages("Invented note.")[1]["content"]    # same note message


# Invented, in the template's shape: a header block, every kind of section trim() keeps or
# drops, sub-headings inside sections, an empty section and a repeated one.
TRIM_NOTE = """Name: Unit No: Sex: F
Chief Complaint:
Fall
History of Present Illness:
Per son, more forgetful over the past year.
Past Medical History:
PMH:
- Hypertension
Social History:
Family History:
Mother had dementia.
Physical Exam:
ADMISSION EXAM:
Neuro: oriented to self only.
Pertinent Results:
Head CT to evaluate for dementia.
Brief Hospital Course:
Treated for a urinary infection.
Medications on Admission:
Donepezil 10 mg daily
Physical Exam:
DISCHARGE EXAM:
Neuro: oriented to self only.
Discharge Condition:
Mental Status: Confused - sometimes.
Discharge Instructions:
You were admitted after a fall."""


def test_trim_keeps_the_chosen_sections_whole_in_note_order():
    assert trim(TRIM_NOTE) == """Chief Complaint:
Fall

History of Present Illness:
Per son, more forgetful over the past year.

Past Medical History:
PMH:
- Hypertension

Physical Exam:
ADMISSION EXAM:
Neuro: oriented to self only.

Brief Hospital Course:
Treated for a urinary infection.

Physical Exam:
DISCHARGE EXAM:
Neuro: oriented to self only.

Discharge Condition:
Mental Status: Confused - sometimes."""


def test_trim_leaves_a_note_without_kept_headings_whole():
    note = "HISTORY OF PRESENT ILLNESS:\nAn older dictated note.\nFamily History:\nNoncontributory."
    assert trim(note) == note                    # a dropped heading alone does not start a trim


def test_every_passage_of_the_trim_is_in_the_whole_note():
    for p in passages(trim(TRIM_NOTE)):
        assert locate(TRIM_NOTE, p)[0] == "exact"


def test_judge_text_trims_for_v6_then_cuts():
    assert judge_text(TRIM_NOTE, "v4", 60_000) == (TRIM_NOTE, False)
    assert judge_text(TRIM_NOTE, "v6", 60_000) == (trim(TRIM_NOTE), False)
    assert judge_text(TRIM_NOTE, "v6", 20) == (trim(TRIM_NOTE)[:20], True)
    with pytest.raises(KeyError):
        judge_text(TRIM_NOTE, "v5", 60_000)
        
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



def test_read_my_reads_normalises_and_checks(tmp_path):
    path = tmp_path / "reads.csv"
    path.write_text("pilot_id,verdict,comment\nP01, documented ,\nP02,Acute only,odd one\nP03,NONE,\n")
    assert read_my_reads(path, "pilot_id")["mine"].tolist() == ["Documented", "Acute only", "None"]
    path.write_text("read_id,verdict,comment\nT01,Documented,\nT02,,\nT03,maybe,\n")
    with pytest.raises(ValueError, match="T02, T03"):
        read_my_reads(path, "read_id")

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



# --- the test agreement compare (invented verdicts) ---------------------------------------


def test_agreement_matches_sklearn_and_counts():
    rng = np.random.default_rng(0)
    a, b, w = rng.random(200) < 0.3, rng.random(200) < 0.4, rng.integers(0, 3, 200)
    assert agreement(a, b, w)["kappa"] == pytest.approx(cohen_kappa_score(a, b, sample_weight=w))
    assert agreement(a, b)["kappa"] == pytest.approx(cohen_kappa_score(a, b))
    got = agreement([True, True, False, False, False], [True, False, True, False, False])
    assert got["sensitivity"] == 0.5 and got["specificity"] == pytest.approx(2 / 3)


def test_agreement_is_nan_when_it_cannot_be_computed():
    got = agreement([False, False], [False, False])            # nobody says Documented
    assert np.isnan(got["kappa"]) and np.isnan(got["sensitivity"]) and got["specificity"] == 1.0


def test_worst_case_counts_each_failure_as_a_disagreement():
    df = pd.DataFrame({"mine": ["Documented", "None", "Documented", "None"],
                       "judge": [None, None, "Documented", "None"]})
    got = worst_case(df)                    # a miss, a false Documented, a hit, a correct None
    assert got["sensitivity"] == 0.5 and got["specificity"] == 0.5


def test_band_patient_weights_keep_bands_and_patients_together():
    band = np.array(["1x"] * 4 + ["2-4x"] * 3)
    patient = np.array([1, 1, 2, 3, 3, 4, 5])     # patient 1 has two notes; patient 3 is in both bands
    rng = np.random.default_rng(5)
    for _ in range(50):
        w = band_patient_weights(band, patient, rng)
        assert w[0] == w[1]                        # one patient's notes are drawn together
        assert w[0] + w[2] + w[3] == 3             # 1x: 3 patients drawn, 3 draws
        assert w[4:].sum() == 3                    # 2-4x: 3 patients drawn, 3 draws
    assert np.array_equal(band_patient_weights(band, patient, np.random.default_rng(1)),
                          band_patient_weights(band, patient, np.random.default_rng(1)))


def test_load_test_answers_reads_parts_and_rejects_repeats(tmp_path):
    parts = tmp_path / "parts"
    parts.mkdir()
    pd.DataFrame({"hadm_id": [3, 1], "verdict": ["None", None], "rejected": [False, True],
                  "seconds": [1.0, 2.0]}).to_parquet(parts / "part-00000.parquet")
    pd.DataFrame({"hadm_id": [7], "verdict": ["Documented"], "rejected": [False],
                  "seconds": [1.0]}).to_parquet(parts / "part-00001.parquet")
    got = load_test_answers(parts)
    assert got["hadm_id"].tolist() == [3, 1, 7] and list(got.columns) == ["hadm_id", "verdict", "rejected"]
    assert got["verdict"].isna().tolist() == [False, True, False]
    pd.DataFrame({"hadm_id": [1], "verdict": ["None"], "rejected": [False],
                  "seconds": [1.0]}).to_parquet(parts / "part-00002.parquet")
    with pytest.raises(ValueError, match="judged twice"):
        load_test_answers(parts)


def test_agreement_rows_keeps_no_answer_and_fails_on_gaps():
    key = pd.DataFrame({"read_id": ["T01", "T02", "T03"], "hadm_id": [11, 12, 13],
                        "band": ["1x", "1x", "2-4x"]})
    mine = pd.DataFrame({"read_id": ["T02", "T01", "T03"], "mine": ["None", "Documented", "Possible"]})
    patients = pd.DataFrame({"hadm_id": [11, 12, 13, 14], "subject_id": [1, 1, 2, 3]})
    answers = pd.DataFrame({"hadm_id": [13, 11, 12, 99], "verdict": ["None", "Documented", None, "None"],
                            "rejected": [False, False, True, False]})
    df = agreement_rows(key, mine, answers, patients).set_index("read_id")
    assert df.loc["T01", ["mine", "judge", "subject_id"]].tolist() == ["Documented", "Documented", 1]
    assert pd.isna(df.loc["T02", "judge"]) and df.loc["T02", "rejected"]
    with pytest.raises(ValueError, match="never judged: T03"):
        agreement_rows(key, mine, answers[answers["hadm_id"] != 13], patients)
    with pytest.raises(ValueError, match="different read IDs"):
        agreement_rows(key, mine.iloc[:2], answers, patients)


def test_judge_output_is_locked_until_allowed():
    for tcfg in ({}, {"output_allowed": False}, {"output_allowed": "true"}):
        with pytest.raises(PermissionError, match="locked"):
            check_output_unlocked(tcfg)
    check_output_unlocked({"output_allowed": True})


def test_sample_documented_takes_only_documented_sample_notes():
    answers = pd.DataFrame({"hadm_id": [1, 2, 3, 4, 5, 6],
                            "verdict": ["Documented", "Possible", None, "Documented", "Documented", "None"]})
    sample = [1, 2, 3, 5, 6]                       # 4 is Documented but in the pool, not the sample
    got = draw_sample_documented(answers, sample, seed=7)
    assert list(got.columns) == ["read_id", "hadm_id"]
    assert sorted(got["hadm_id"]) == [1, 5]
    assert list(got["read_id"]) == ["S01", "S02"]
    assert got.equals(draw_sample_documented(answers.iloc[::-1], sample, seed=7))   # row order: no effect
    assert draw_sample_documented(answers, [2, 3, 6], seed=7).empty


def test_judged_test_fails_when_answers_and_to_judge_differ(tmp_path):
    (tmp_path / "parts").mkdir()
    pd.DataFrame({"hadm_id": [1, 2], "band": ["1x", None], "in_sample": [False, True]}).to_parquet(
        tmp_path / "bands.parquet")
    pd.DataFrame({"hadm_id": [1, 2]}).to_parquet(tmp_path / "to-judge.parquet")
    pd.DataFrame({"hadm_id": [1], "verdict": ["None"], "rejected": [False]}).to_parquet(
        tmp_path / "parts" / "part-00000.parquet")
    with pytest.raises(ValueError, match="1 missing, 0 extra"):
        judged_test(tmp_path)
    pd.DataFrame({"hadm_id": [2], "verdict": [None], "rejected": [False]}).to_parquet(
        tmp_path / "parts" / "part-00001.parquet")
    bands, answers = judged_test(tmp_path)
    assert sorted(answers["hadm_id"]) == [1, 2]