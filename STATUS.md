# Project status

As of 2026-11-07, late evening: repo private on GitHub, CI green, 101 tests (94 + 1 skipped in CI,
which installs only core packages).

## Results so far (validation only; test untouched)

| Arm | How it reads the note | PR-AUC | Notes |
|---|---|---|---|
| Chance | | 0.036 | 820 positives of 22,692 |
| 0, regex | Explicit mentions, NegEx-style context | 0.622 | any mention: flags 4.1%, precision 0.689, recall 0.784 |
| 0+drugs | Same, plus the four dementia drugs | 0.623 | drugs add almost nothing |
| 0, first 512 tokens | Arm 2's window only | 0.572 | about 5% of coded notes have their only explicit mention past token 512 |
| 1, TF-IDF + logistic regression | Each word separately | 0.730 | train 0.822 |
| 1b, LightGBM | Words in combination | 0.792 | inner slice 0.769, train 0.973; 180 trees |
| 2, LoRA on Bio_ClinicalBERT | Words in context | running | |

Differences between arms are not yet tested: the paired bootstrap (by patient) is next. About
290 validation notes mention dementia (not negated, not a relative) but carry no dementia code:
an early sign for the project question, not yet validated.

## Done

- **Cohort** (`make cohort`): 331,731 admissions with a discharge summary. Split by patient
  `anchor_year_group`: train 2008-2013, test 2017-2019, 2014-2016 and 2020-2022 excluded.
  Build asserts bands, clean ICD codes, one row per admission and 65+ prevalence.
- **Modelling dataset** (`make dataset`): `data/derived/notes.parquet`, baseline sections only.
  Train 225,597 notes, test 41,266. Core dementia about 4.7% on test.
- **Known limitations, documented:** about 65% note coverage; 4.1% of test notes have no history
  section and are dropped.
- **Validation split** (`splits.load_notes()`): 10% of train patients, grouped by `subject_id`,
  `np.random.default_rng(3)`, chosen after comparing seeds 1-4. Patients are listed in `hadm_id`
  order before the draw, the order the seed comparison saw, so row order cannot change it.
  Train 76,124 patients / 202,905 notes; validation 8,458 / 22,692 (core 3.61%); test untouched.
- **Scorer** (`evaluate.py`): PR-AUC with prevalence and lift beside it; `save_scores()` writes
  per-note scores (hadm_id, score) under `data/derived/scores/` for paired comparison.
- **Arm 0, regex** (`arm0_regex.py`): every rule fixed 2026-11-07 before any note was read.
  Diagnosis terms mirror the core ICD codes; no abbreviations, MCI, delirium or implicit
  language. Negated and someone-else mentions excluded, uncertain ones count 0.5; same clause,
  at most 6 words. Three pre-declared variants. 34 invented-sentence tests. Design in
  `docs/explainers.md`.
- **Arm 1, TF-IDF + logistic regression** (`arm1_tfidf.py`): fixed, untuned settings.
- **Arm 1b, LightGBM** (`arm1b_lgbm.py`): arm 1's features; early stopping on an inner 10% slice
  of train patients, never on validation; library defaults otherwise.
- **Arm 2 set up** (`arm2_lora.py`): Bio_ClinicalBERT + LoRA, rank 8, alpha 16, query and value,
  dropout 0.1; 295,681 trainable weights (0.27%). Batch 16 at 512 tokens: about 0.85 s per step.
  fp16 gave no speed-up on this GPU, so training is full precision. Two passes over all train
  notes. Pre-declared fallback: rank 12, alpha 24, only if validation PR-AUC < 0.76 (set before
  arm 1b existed; deliberately left unchanged).
- **Token lengths** (Bio_ClinicalBERT tokenizer, test not counted): median 451 train / 460
  validation; 40.9% / 42.6% of notes exceed 512 and are truncated. Baseline text runs HPI, then
  PMH, then Social History, so truncation cuts PMH first.
- **MPS:** PyTorch sees the M1 Max GPU. **MLflow** on SQLite under `data/` (MLflow 3.x refuses the
  old file store); metrics and parameters only, no models.
- **CI fixed 2026-11-07:** mlflow is imported inside `run()` functions and pyarrow is a declared
  core dependency. In the test process, LightGBM and PyTorch are both single-threaded: each brings
  its own OpenMP, and two copies starting threads in one process deadlock on macOS.

## Decided, not yet in code

- **ICD-10 in train:** train is about 19% ICD-10 (43,298 of 227,130 admissions), because a
  train patient's later admissions follow them into train. Patients never cross splits, so there
  is no leakage. Decision: keep and document; docs corrected 2026-11-07. Still to do:
  - add `icd_version` per admission to `cohort.py` and `dataset.py`
  - report results by era

- **Delta tables via PySpark dropped** (2026-11-08): only a Databricks-skills demonstration;
  DuckDB and Parquet already do the job. Removed from config, PLAN, README and dependencies.

## In progress

Stage 3, piece 3: every arm on validation, then a fair comparison.

1. Running (started 2026-11-07 ~21:15): arm 2 main run, about 25,400 steps, done ~03:30-04:00.
   Log in `data/derived/logs/arm2-main.log`. A watcher starts the rank-12 fallback on its own if
   validation PR-AUC < 0.76 (done ~10:00-10:30). Schedule in `docs/explainers.md` ("Run plan").
2. Next, in order:
   - choose the truncation follow-up (PMH first, head and tail, or chunking) **before** reading
     arm 2's score; arm 0's 0.622 -> 0.572 drop at 512 tokens is the evidence
   - re-run arm 1 to save its per-note scores (the only arm without them)
   - patient-level bootstrap CIs and paired comparisons in `evaluate.py`: arm 2 vs each arm
   - optional GPU slots Sunday 12:00-18:00 and 21:00-03:00; GPU free by Monday

Deferred: subgroup breakdown by age band, sex, insurance, race, ICD era.

## Next

- Open decision: keep arm 3 (prompted local LLM) or prune it, on its own merits
- Test set scored once, for every arm, after all choices are final

## Before going public

- Scan tracked files for employer names or personal details
- Recreate the GitHub repo so commits from before the 2026-11-06 history rewrite are gone from GitHub
- Then make it public, ideally after the first paired comparison
