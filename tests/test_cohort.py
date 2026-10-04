"""The cohort build and its checks, run on tiny fake MIMIC-shaped files."""

import duckdb
import pytest

from dementia_undercoding.codes import load_code_sets
from dementia_undercoding.cohort import (
    assert_bands_present,
    assert_codes_clean,
    assert_prevalence,
    assert_unique_admissions,
    build_cohort,
    build_dx,
    load_prefixes,
    load_split_bands,
)

# subject_id, hadm_id, seq_num, icd_code, icd_version - same columns as MIMIC.
FAKE_DIAGNOSES = """subject_id,hadm_id,seq_num,icd_code,icd_version
1,101,1,F0390,10
1,101,2,I10,10
2,102,1,G3184,10
3,103,1,2930,9
3,103,2,3310,9
4,104,1,4019,9
5,105,1,F03,9
"""

FAKE_PATIENTS = """subject_id,gender,anchor_age,anchor_year,anchor_year_group
1,F,80,2150,2011 - 2013
2,M,70,2160,2017 - 2019
3,F,75,2170,2014 - 2016
4,M,17,2155,2011 - 2013
5,F,60,2180,2023 - 2025
"""

FAKE_ADMISSIONS = """subject_id,hadm_id,admittime,admission_type,insurance,language,marital_status,race
1,101,2153-05-01 10:00:00,URGENT,Medicare,English,WIDOWED,WHITE
2,102,2160-01-01 10:00:00,URGENT,Medicare,English,MARRIED,WHITE
3,103,2170-01-01 10:00:00,URGENT,Medicare,English,MARRIED,WHITE
4,104,2155-01-01 10:00:00,URGENT,Medicaid,English,SINGLE,WHITE
5,105,2180-01-01 10:00:00,URGENT,Private,English,SINGLE,WHITE
1,106,2154-01-01 10:00:00,URGENT,Medicare,English,WIDOWED,WHITE
2,107,2161-01-01 10:00:00,URGENT,Medicare,English,MARRIED,WHITE
"""

# 106 deliberately has no discharge summary. Text is placeholder, not clinical.
FAKE_DISCHARGE = """note_id,subject_id,hadm_id,note_type,note_seq,text
1-DS-1,1,101,DS,1,aaaa
2-DS-1,2,102,DS,1,aaaa
3-DS-1,3,103,DS,1,aaaa
4-DS-1,4,104,DS,1,aaaa
5-DS-1,5,105,DS,1,aaaa
2-DS-2,2,107,DS,2,aaaa
"""

SPLIT = {
    "train": ["2011 - 2013"],
    "test": ["2017 - 2019"],
    "exclude": {"2014 - 2016": "held out"},
}


@pytest.fixture
def dx(tmp_path):
    path = tmp_path / "diagnoses_icd.csv"
    path.write_text(FAKE_DIAGNOSES)
    con = duckdb.connect()
    load_prefixes(con, load_code_sets("config/codes.yaml"))
    build_dx(con, path)
    rows = con.execute("SELECT * FROM dx ORDER BY hadm_id").fetchall()
    return {row[0]: row[1:] for row in rows}


def test_one_row_per_admission(dx):
    # 103 has a core code, which matches both core and broad prefixes.
    # Fan-out would show up here as extra rows or an inflated count.
    assert sorted(dx) == [101, 102, 103, 104, 105]
    assert dx[103][0] == 2


def test_flags(dx):
    # (n_codes, dementia_core, dementia_broad, delirium)
    assert dx[101] == (2, True, True, False)    # vascular dementia + hypertension
    assert dx[102] == (1, False, True, False)   # MCI: broad only
    assert dx[103] == (2, True, True, True)     # delirium + Alzheimer's
    assert dx[104] == (1, False, False, False)  # hypertension only
    assert dx[105] == (1, False, False, False)  # F03 under ICD-9 is not dementia


@pytest.fixture
def fake_dir(tmp_path):
    files = {
        "diagnoses_icd.csv": FAKE_DIAGNOSES,
        "patients.csv": FAKE_PATIENTS,
        "admissions.csv": FAKE_ADMISSIONS,
        "discharge.csv": FAKE_DISCHARGE,
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return tmp_path


@pytest.fixture
def cohort_con(fake_dir):
    con = duckdb.connect()
    load_prefixes(con, load_code_sets("config/codes.yaml"))
    load_split_bands(con, SPLIT)
    build_dx(con, fake_dir / "diagnoses_icd.csv")
    build_cohort(
        con,
        fake_dir / "admissions.csv",
        fake_dir / "patients.csv",
        fake_dir / "discharge.csv",
        min_age=18,
    )
    return con


@pytest.fixture
def cohort(cohort_con):
    rows = cohort_con.execute("""
        SELECT hadm_id, split, age_at_admit, n_codes, dementia_core
        FROM cohort ORDER BY hadm_id
    """).fetchall()
    return {row[0]: row[1:] for row in rows}


def test_cohort_keeps_adults_with_a_note(cohort):
    # 104 is 17. 106 has no discharge summary.
    assert sorted(cohort) == [101, 102, 103, 105, 107]


def test_split_from_band(cohort):
    assert cohort[101][0] == "train"
    assert cohort[102][0] == "test"
    assert cohort[103][0] == "exclude"
    # 2023 - 2025 is not in SPLIT: NULL, not quietly bucketed. Piece 4 asserts on this.
    assert cohort[105][0] is None


def test_age_at_admission(cohort):
    # anchor_age 80 in 2150, admitted 2153.
    assert cohort[101][1] == 83


def test_admission_without_codes(cohort):
    # 107 has a note but no diagnosis rows at all: zero codes, not NULL.
    assert cohort[107][2:] == (0, False)


# --- the checks: each one must pass on good input and fail on bad ---


def test_bands_unconfigured_band_fails(fake_dir):
    # FAKE_PATIENTS has 2023 - 2025, which SPLIT does not mention.
    with pytest.raises(ValueError, match="2023 - 2025"):
        assert_bands_present(duckdb.connect(), fake_dir / "patients.csv", SPLIT)


def test_bands_missing_from_data_fails(fake_dir):
    split = {**SPLIT, "test": ["2017 - 2019", "2023 - 2025", "2099 - 2101"]}
    with pytest.raises(ValueError, match="2099 - 2101"):
        assert_bands_present(duckdb.connect(), fake_dir / "patients.csv", split)


def test_bands_all_accounted_for_passes(fake_dir):
    split = {**SPLIT, "test": ["2017 - 2019", "2023 - 2025"]}
    assert_bands_present(duckdb.connect(), fake_dir / "patients.csv", split)


def test_codes_clean_passes(fake_dir):
    assert_codes_clean(duckdb.connect(), fake_dir / "diagnoses_icd.csv")


@pytest.mark.parametrize("bad_code", ["331.0", " 3310", "g309"])
def test_codes_dirty_fails(tmp_path, bad_code):
    path = tmp_path / "diagnoses_icd.csv"
    path.write_text(f"subject_id,hadm_id,seq_num,icd_code,icd_version\n1,101,1,{bad_code},9\n")
    with pytest.raises(ValueError, match="1 ICD codes"):
        assert_codes_clean(duckdb.connect(), path)


def test_unique_admissions(cohort_con):
    assert_unique_admissions(cohort_con)
    cohort_con.execute("INSERT INTO cohort SELECT * FROM cohort WHERE hadm_id = 101")
    with pytest.raises(ValueError, match="1 duplicate"):
        assert_unique_admissions(cohort_con)


def test_prevalence(cohort_con):
    # 65+ in the fake cohort: train is 101 (core), so 100%; test is 102 and
    # 107 (neither core), so 0%. A wide range passes; a narrow one fails.
    assert_prevalence(cohort_con, 65, [0.0, 1.0])
    with pytest.raises(ValueError, match="test split is 0.00%"):
        assert_prevalence(cohort_con, 65, [0.05, 1.0])