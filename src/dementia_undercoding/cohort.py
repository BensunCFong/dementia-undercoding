"""Build the admission-level cohort in DuckDB.

Everything stays local. DuckDB reads the gzipped MIMIC CSVs in place, so no
data is copied, moved, or sent anywhere. This matters: the DUA prohibits
sending MIMIC data through third-party APIs or online services.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import yaml

from .codes import load_code_sets, prefixes_for, sql_predicate


def load_config(path: str | Path = "config/config.yaml") -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build(config_path: str | Path = "config/config.yaml", codes_path: str | Path = "config/codes.yaml"):
    cfg = load_config(config_path)
    code_sets = load_code_sets(codes_path)
    broad = cfg["label"]["set"] == "broad"

    dem = sql_predicate(prefixes_for(code_sets, "dementia_core", broad=broad))
    dlm = sql_predicate(prefixes_for(code_sets, "delirium"))

    hosp = Path(cfg["paths"]["mimic_hosp"])
    note = Path(cfg["paths"]["mimic_note"])
    train = cfg["split"]["train"]
    test = cfg["split"]["test"]

    con = duckdb.connect(cfg["paths"]["duckdb"])
    con.execute(f"""
        CREATE OR REPLACE TABLE cohort AS
        WITH dx AS (
            SELECT
                hadm_id,
                MAX(CASE WHEN {dem} THEN 1 ELSE 0 END) AS label_dementia,
                MAX(CASE WHEN {dlm} THEN 1 ELSE 0 END) AS code_delirium,
                COUNT(*) AS n_codes
            FROM read_csv_auto('{hosp}/diagnoses_icd.csv.gz')
            GROUP BY hadm_id
        ),
        ds AS (
            -- one discharge summary per admission, lowest note_seq wins
            SELECT hadm_id, text
            FROM (
                SELECT hadm_id, text,
                       ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY note_seq) AS rn
                FROM read_csv_auto('{note}/discharge.csv.gz')
            ) WHERE rn = 1
        )
        SELECT
            a.hadm_id,
            a.subject_id,
            p.anchor_year_group,
            p.anchor_age,
            p.gender,
            a.insurance,
            a.race,
            a.marital_status,
            a.language,
            a.admission_type,
            COALESCE(dx.label_dementia, 0) AS label,
            COALESCE(dx.code_delirium, 0)  AS code_delirium,
            COALESCE(dx.n_codes, 0)        AS n_codes,
            LENGTH(ds.text)                AS n_chars,
            CASE
                WHEN p.anchor_year_group IN ({",".join(f"'{v}'" for v in train)}) THEN 'train'
                WHEN p.anchor_year_group IN ({",".join(f"'{v}'" for v in test)})  THEN 'test'
                ELSE 'excluded'
            END AS split,
            ds.text
        FROM read_csv_auto('{hosp}/admissions.csv.gz') a
        JOIN read_csv_auto('{hosp}/patients.csv.gz') p USING (subject_id)
        JOIN ds USING (hadm_id)
        LEFT JOIN dx USING (hadm_id)
        WHERE p.anchor_age >= {cfg["cohort"]["min_age"]}
    """)
    return con


def summarise(con) -> None:
    """Sanity checks to run before trusting anything downstream."""
    print(con.execute("""
        SELECT split,
               COUNT(*) AS n,
               SUM(label) AS positives,
               ROUND(100.0 * AVG(label), 2) AS pct_positive,
               ROUND(AVG(n_chars)) AS mean_chars
        FROM cohort GROUP BY split ORDER BY split
    """).df())
    # If pct_positive is 0.00 anywhere, the code matching is broken.
    # Expect roughly 5-9% on the core set.
