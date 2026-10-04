"""The modelling dataset, built from a tiny fake cohort and fake notes."""

import duckdb
import pytest

from dementia_undercoding.dataset import build_notes

# Invented notes. 1 and 2 have baseline sections; 3 has none; 4 is in an
# excluded band. Quoted because each note spans several lines.
FAKE_DISCHARGE = '''note_id,hadm_id,text
n1,101,"History of Present Illness:
Per daughter, forgetful.

Brief Hospital Course:
Disoriented overnight."
n2,102,"Past Medical History:
Hypertension ___"
n3,103,"Brief Hospital Course:
Stable."
n4,104,"History of Present Illness:
Fall at home."
'''


@pytest.fixture
def notes(tmp_path):
    discharge = tmp_path / "discharge.csv"
    discharge.write_text(FAKE_DISCHARGE)
    out = tmp_path / "notes.parquet"

    con = duckdb.connect()
    con.execute("""
        CREATE TABLE cohort AS
        SELECT hadm_id, hadm_id - 100 AS subject_id, note_id, split,
               70 AS age_at_admit, 'F' AS gender, 'Medicare' AS insurance,
               'WHITE' AS race, 'MARRIED' AS marital_status, 'English' AS language,
               'URGENT' AS admission_type, 1 AS n_codes,
               dementia_core, dementia_core AS dementia_broad, false AS delirium
        FROM (VALUES
            (101, 'n1', 'train',   true),
            (102, 'n2', 'test',    false),
            (103, 'n3', 'train',   false),
            (104, 'n4', 'exclude', false)
        ) AS t(hadm_id, note_id, split, dementia_core)
    """)
    build_notes(con, discharge, out)
    rows = con.execute(f"SELECT hadm_id, split, text FROM read_parquet('{out}') ORDER BY hadm_id").fetchall()
    return {row[0]: row[1:] for row in rows}


def test_keeps_train_and_test_with_baseline_text(notes):
    # 103 has no baseline section; 104 is in an excluded band.
    assert sorted(notes) == [101, 102]


def test_text_is_cleaned_baseline_only(notes):
    split, text = notes[101]
    assert split == "train"
    assert "forgetful" in text
    assert "Disoriented" not in text  # hospital course left out
    assert notes[102][1] == "Past Medical History:\nHypertension"  # placeholder gone