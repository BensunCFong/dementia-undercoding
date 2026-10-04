"""Build the admission-level cohort in DuckDB.

Everything stays local. DuckDB reads the gzipped MIMIC CSVs in place, so no
data is copied, moved, or sent anywhere. This matters: the DUA prohibits
sending MIMIC data through third-party APIs or online services.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .codes import prefix_rows


def load_config(path: str | Path = "config/config.yaml") -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_prefixes(con, code_sets: dict) -> None:
    """Load prefix_rows into a table the SQL joins against."""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE code_prefixes (
            label VARCHAR, icd_version INTEGER, prefix VARCHAR
        )
    """)
    con.executemany("INSERT INTO code_prefixes VALUES (?, ?, ?)", prefix_rows(code_sets))


def build_dx(con, diagnoses_path: str | Path) -> None:
    """One row per admission: its code count and which label sets its codes hit.

    Counting happens before the prefix join, and the join is grouped back to
    one row per admission before the two meet, so nothing can fan out.
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE dx AS
        WITH codes AS (
            -- icd_code forced to text: type sniffing could read an all-digit
            -- ICD-9 sample as a number and drop leading zeros ('0389' -> 389).
            SELECT hadm_id, icd_version, icd_code
            FROM read_csv('{diagnoses_path}',
                          types = {{'icd_code': 'VARCHAR', 'icd_version': 'INTEGER'}})
        ),
        per_admission AS (
            SELECT hadm_id, COUNT(*) AS n_codes
            FROM codes
            GROUP BY hadm_id
        ),
        hits AS (
            -- Every core code also matches broad, so one diagnosis can join to
            -- several prefix rows. bool_or collapses that back per admission.
            SELECT c.hadm_id,
                   bool_or(p.label = 'dementia_core')  AS dementia_core,
                   bool_or(p.label = 'dementia_broad') AS dementia_broad,
                   bool_or(p.label = 'delirium')       AS delirium
            FROM codes c
            JOIN code_prefixes p
              ON c.icd_version = p.icd_version
             AND starts_with(c.icd_code, p.prefix)
            GROUP BY c.hadm_id
        )
        SELECT a.hadm_id,
               a.n_codes,
               COALESCE(h.dementia_core, false)  AS dementia_core,
               COALESCE(h.dementia_broad, false) AS dementia_broad,
               COALESCE(h.delirium, false)       AS delirium
        FROM per_admission a
        LEFT JOIN hits h USING (hadm_id)
    """)


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