"""The admission-grain diagnosis table, built from a tiny fake diagnoses file."""

import duckdb
import pytest

from dementia_undercoding.codes import load_code_sets
from dementia_undercoding.cohort import build_dx, load_prefixes

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