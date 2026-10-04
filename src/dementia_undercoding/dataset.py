"""Turn the cohort into the modelling dataset: one row per note, baseline text only.

Reads the cohort table built by cohort.build() and the raw discharge notes,
keeps train and test (exclude stays out), cleans each note and cuts it down to
its baseline sections. Writes one Parquet file under data/derived/. Nothing
is sampled here: every train and test note is kept, and any sampling for a
particular model happens at training time.
"""

from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import duckdb
import pandas as pd

from .cohort import load_config
from .text import baseline_text, clean


def _baseline(text: str) -> str:
    return baseline_text(clean(text))


def build_notes(con, discharge_path: str | Path, out_path: str | Path, workers: int | None = None) -> None:
    """Write train and test notes with non-empty baseline text to Parquet.

    Needs the cohort table on con. The text cleaning is plain Python, which
    runs on one core per process, so it is spread over a pool of worker
    processes (default: one per CPU core). Notes with no baseline text are
    dropped here; summarise_notes() counts them.
    """
    raw = con.execute(f"""
        SELECT c.hadm_id, n.text
        FROM cohort c
        JOIN read_csv('{discharge_path}') n USING (note_id)
        WHERE c.split IN ('train', 'test')
    """).fetchall()
    hadm_ids = [hadm_id for hadm_id, _ in raw]
    texts = [text for _, text in raw]

    with Pool(workers) as pool:
        baselines = pool.map(_baseline, texts, chunksize=1000)

    baseline_df = pd.DataFrame({"hadm_id": hadm_ids, "text": baselines})
    con.register("baseline_df", baseline_df)
    con.execute(f"""
        COPY (
            SELECT
                c.hadm_id,
                c.subject_id,
                c.split,
                c.age_at_admit,
                c.gender,
                c.insurance,
                c.race,
                c.marital_status,
                c.language,
                c.admission_type,
                c.n_codes,
                c.dementia_core,
                c.dementia_broad,
                c.delirium,
                b.text
            FROM cohort c
            JOIN baseline_df b USING (hadm_id)
            WHERE b.text <> ''
            ORDER BY hadm_id
        ) TO '{out_path}' (FORMAT parquet)
    """)


def summarise_notes(con, out_path: str | Path) -> None:
    """Per split: cohort admissions, notes kept, notes dropped, positive rates."""
    print(con.execute(f"""
        SELECT c.split,
               COUNT(*)                                    AS cohort,
               COUNT(n.hadm_id)                            AS kept,
               COUNT(*) - COUNT(n.hadm_id)                 AS dropped_no_baseline,
               ROUND(100.0 * AVG(n.dementia_core::INT), 2) AS pct_core,
               ROUND(AVG(LENGTH(n.text)))                  AS mean_chars
        FROM cohort c
        LEFT JOIN read_parquet('{out_path}') n USING (hadm_id)
        WHERE c.split IN ('train', 'test')
        GROUP BY c.split
        ORDER BY c.split
    """).df().to_string())


def build(config_path: str | Path = "config/config.yaml"):
    """Build data/derived/notes.parquet from the cohort and print the summary."""
    cfg = load_config(config_path)
    note = cfg["paths"]["mimic_note"]
    if not note:
        raise ValueError("This config has no notes (mimic_note is null), so there is no dataset to build.")
    discharge = Path(note) / cfg["files"]["note"]["discharge"]
    out = cfg["paths"]["notes"]

    con = duckdb.connect(cfg["paths"]["duckdb"], read_only=True)
    build_notes(con, discharge, out)
    summarise_notes(con, out)