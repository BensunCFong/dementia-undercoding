"""Build the admission-level cohort in DuckDB.

Everything stays local. DuckDB reads the gzipped MIMIC CSVs in place, so no
data is copied, moved, or sent anywhere. This matters: the DUA prohibits
sending MIMIC data through third-party APIs or online services.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import yaml

from .codes import load_code_sets, prefix_rows


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


def load_split_bands(con, split_cfg: dict) -> None:
    """Load the config's band-to-split mapping into a table the SQL joins against.

    A band missing from the config gets no row here, so it comes out of the
    cohort join with split = NULL rather than silently landing in a bucket.
    """
    rows = [(band, "train") for band in split_cfg["train"]]
    rows += [(band, "test") for band in split_cfg["test"]]
    rows += [(band, "exclude") for band in split_cfg["exclude"]]
    con.execute("CREATE OR REPLACE TEMP TABLE split_bands (band VARCHAR, split VARCHAR)")
    con.executemany("INSERT INTO split_bands VALUES (?, ?)", rows)


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


def build_cohort(
    con,
    admissions_path: str | Path,
    patients_path: str | Path,
    discharge_path: str | Path | None,
    min_age: int,
) -> None:
    """One row per adult admission with a discharge summary: who, when, which split, which codes.

    Needs dx, split_bands and code_prefixes to exist already. With no
    discharge_path (the demo ships no notes) every admission is kept and the
    note columns are NULL.
    """
    if discharge_path is None:
        note_cols = "NULL::VARCHAR AS note_id, NULL::BIGINT AS n_chars"
        note_join = ""
    else:
        note_cols = "n.note_id, LENGTH(n.text) AS n_chars"
        note_join = f"JOIN read_csv('{discharge_path}') n USING (hadm_id)"

    con.execute(f"""
        CREATE OR REPLACE TABLE cohort AS
        SELECT
            a.hadm_id,
            a.subject_id,
            p.anchor_year_group,
            s.split,
            -- anchor_age is age in anchor_year, not at this admission.
            p.anchor_age + (YEAR(a.admittime) - p.anchor_year) AS age_at_admit,
            p.gender,
            a.insurance,
            a.race,
            a.marital_status,
            a.language,
            a.admission_type,
            COALESCE(dx.n_codes, 0)            AS n_codes,
            COALESCE(dx.dementia_core, false)  AS dementia_core,
            COALESCE(dx.dementia_broad, false) AS dementia_broad,
            COALESCE(dx.delirium, false)       AS delirium,
            {note_cols}
        FROM read_csv('{admissions_path}') a
        JOIN read_csv('{patients_path}') p USING (subject_id)
        {note_join}
        LEFT JOIN dx USING (hadm_id)
        LEFT JOIN split_bands s ON p.anchor_year_group = s.band
        WHERE p.anchor_age >= ?
    """, [min_age])


def assert_bands_present(con, patients_path: str | Path, split_cfg: dict) -> None:
    """Every configured band exists in the data, and every band in the data is configured.

    Checked against patients, not the cohort, so a band that loses all its
    admissions downstream still has to be accounted for in config.
    """
    configured = set(split_cfg["train"]) | set(split_cfg["test"]) | set(split_cfg["exclude"])
    observed = {
        row[0]
        for row in con.execute(
            f"SELECT DISTINCT anchor_year_group FROM read_csv('{patients_path}')"
        ).fetchall()
    }
    not_in_data = configured - observed
    not_in_config = observed - configured
    if not_in_data:
        raise ValueError(f"Bands in config but not in patients: {sorted(not_in_data)}")
    if not_in_config:
        raise ValueError(f"Bands in patients but not in config: {sorted(not_in_config)}")


def assert_codes_clean(con, diagnoses_path: str | Path) -> None:
    """Every ICD code is bare uppercase letters and digits: no dots, no padding.

    build_dx matches raw codes without normalising them in SQL. That is only
    safe while this holds, so it is checked on every build rather than trusted.
    """
    (n_dirty,) = con.execute(f"""
        SELECT COUNT(*)
        FROM read_csv('{diagnoses_path}', types = {{'icd_code': 'VARCHAR'}})
        WHERE NOT regexp_full_match(icd_code, '[0-9A-Z]+')
    """).fetchone()
    if n_dirty:
        raise ValueError(f"{n_dirty} ICD codes contain dots, spaces or lowercase")


def assert_unique_admissions(con) -> None:
    """The cohort has exactly one row per admission."""
    (n_dupes,) = con.execute("SELECT COUNT(*) - COUNT(DISTINCT hadm_id) FROM cohort").fetchone()
    if n_dupes:
        raise ValueError(f"{n_dupes} duplicate hadm_id rows in cohort")


def assert_prevalence(con, min_age: int, core_range: list[float]) -> None:
    """Coded core dementia among older admissions sits in a sane range, in every split.

    A guard on the ICD matching, not a hypothesis. Restricted to min_age and
    over because younger admissions dilute the rate toward zero.
    """
    low, high = core_range
    rows = con.execute("""
        SELECT split, AVG(dementia_core::INT)
        FROM cohort
        WHERE age_at_admit >= ?
        GROUP BY split
        ORDER BY split
    """, [min_age]).fetchall()
    for split, rate in rows:
        if not low <= rate <= high:
            raise ValueError(
                f"Core dementia among {min_age}+ in {split} split is {rate:.2%}, "
                f"outside {low:.0%}-{high:.0%}"
            )


def build(config_path: str | Path = "config/config.yaml", codes_path: str | Path = "config/codes.yaml"):
    """Run every step in order and return the open connection.

    Raises ValueError, and stops, if any check fails.
    """
    cfg = load_config(config_path)
    files = cfg["files"]
    hosp = Path(cfg["paths"]["mimic_hosp"])
    note = cfg["paths"]["mimic_note"]
    discharge = Path(note) / files["note"]["discharge"] if note else None

    Path(cfg["paths"]["duckdb"]).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(cfg["paths"]["duckdb"])

    # Checks on the inputs, before anything is built from them.
    assert_bands_present(con, hosp / files["hosp"]["patients"], cfg["split"])
    assert_codes_clean(con, hosp / files["hosp"]["diagnoses_icd"])

    load_prefixes(con, load_code_sets(codes_path))
    load_split_bands(con, cfg["split"])
    build_dx(con, hosp / files["hosp"]["diagnoses_icd"])
    build_cohort(
        con,
        hosp / files["hosp"]["admissions"],
        hosp / files["hosp"]["patients"],
        discharge,
        cfg["cohort"]["min_age"],
    )

    # Checks on the output.
    assert_unique_admissions(con)
    check = cfg["label"]["prevalence_check"]
    if check:
        assert_prevalence(con, check["min_age"], check["core_range"])
    return con


def summarise(con) -> None:
    """Sanity checks to run before trusting anything downstream."""
    print(con.execute("""
        SELECT split,
               COUNT(*) AS n,
               ROUND(100.0 * AVG(dementia_core::INT), 2)  AS pct_core,
               ROUND(100.0 * AVG(dementia_broad::INT), 2) AS pct_broad,
               ROUND(100.0 * AVG(delirium::INT), 2)       AS pct_delirium,
               ROUND(AVG(n_chars))                        AS mean_chars
        FROM cohort GROUP BY split ORDER BY split
    """).df().to_string())
    # If pct_core is 0.00 anywhere, the code matching is broken.
    # A NULL split means a band is missing from config.yaml.