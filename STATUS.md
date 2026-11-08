# Project status

As of 2026-11-08, evening: repo private on GitHub, 116 tests (106 + 1 skipped in CI, which
installs only core packages).

## Results so far (validation only; test untouched)

Intervals: 95%, from 2,000 resamples of patients (`python -m dementia_undercoding.compare_arms`).
Difference = arm minus arm 2 LoRA main, with every arm scored on the same resampled patients.

| Arm | How it reads the note | PR-AUC [95%] | Minus LoRA main [95%] | Notes |
|---|---|---|---|---|
| Chance | | 0.036 | | 820 positives of 22,692 notes, 8,458 patients |
| 0, regex | Explicit mentions, NegEx-style context | 0.622 [0.574, 0.664] | -0.146 [-0.176, -0.119] | any mention: flags 4.1%, precision 0.689, recall 0.784 |
| 0+drugs | Same, plus the four dementia drugs | 0.623 [0.576, 0.666] | -0.145 [-0.177, -0.116] | drugs add almost nothing |
| 0, first 512 tokens | Arm 2's window only | 0.572 [0.523, 0.614] | -0.196 [-0.223, -0.172] | about 5% of coded notes have their only explicit mention past token 512 |
| 1, TF-IDF + logistic regression | Each word separately | 0.730 [0.688, 0.766] | -0.038 [-0.063, -0.014] | train 0.822 |
| 1b, LightGBM | Words in combination | 0.792 [0.757, 0.824] | +0.023 [+0.002, +0.045] | inner slice 0.769, train 0.973; 180 trees |
| 2, LoRA main (first 512 tokens) | Words in context | 0.768 [0.734, 0.800] | | train sample 0.730 (below validation: no overfitting, likely underfit); 6.15 h |
| 2, LoRA head + tail | First 128 + last 382 tokens | 0.761 [0.723, 0.796] | -0.007 [-0.029, +0.012] | train sample 0.734; indistinguishable from main |

Reading: LoRA clearly beats the regex (PLAN §6's hypothesis, on validation) and TF-IDF + LR.
LightGBM probably ranks slightly better than LoRA, but the interval only just clears zero and it
is one of several comparisons; it would likely not survive a multiplicity correction. Head + tail
matched LoRA main: extra context from the end of the note did not help, so truncation does not
explain LightGBM's lead.

Labels are ICD codes, and the project is about undercoding: a note that describes dementia but
carries no code counts as a false positive here. About 290 validation notes mention dementia (not
negated, not a relative) but carry no dementia code: an early sign for the project question, not
yet validated. So PR-AUC against codes is not the whole test of which arm is most useful.

**Truncation check** (`truncation.py`, arm 0 on validation; measures explicit mentions kept, not
LoRA's score): full note 0.622, first 512 tokens 0.572, PMH first then 512 0.610, head 128 + tail
382 **0.621**. Head + tail keeps nearly every explicit mention. Chosen for the follow-up run, which
matched LoRA main (above).

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
- **Arm 2 main run** (2026-11-07/08): 25,364 steps, 6.15 h, finished cleanly. The fallback watcher
  read 0.768 >= 0.76 and stood down, as designed.
- **Paired bootstrap** (`evaluate.paired_bootstrap`, `evaluate.compare`, `compare_arms.py`):
  resamples patients, not notes; every arm scored on the same resampled patients; 95% percentile
  intervals from 2,000 resamples, seed fixed in config before any interval was seen. Each note is
  weighted by how often its patient was drawn (same result as copying rows). Same answer on any
  number of cores. Reads saved scores and labels only.
- **Truncation check** (`truncation.py`): arm 0 on three 512-token windows; results above.
- **Arm 2 head + tail** (`arm2_lora.py --window head_tail`): token-level, first 128 + last 382,
  `[CLS]`/`[SEP]`/padding built by hand (transformers 5.18 dropped the helper). Notes that fit in
  512 tokens get exactly the main run's input (tested). Its own run name, adapter folder and score
  file. Chosen 2026-11-08 **after** the main score and the bootstrap were seen; recorded in config.

## Decided, not yet in code

- **Validator 1, dementia drug orders** (2026-11-08): design fixed before any arm is scored on
  it; full rules in `docs/explainers.md`. Validation: 193 drug-positive uncoded notes (120
  patients) in the main pool, 92 of them with no explicit mention (the hidden row). Age-only arm
  as reference. Next: choose the Week 3 gap-set arm, then build `validator_rx.py`.

- **ICD-10 in train:** train is about 19% ICD-10 (43,298 of 227,130 admissions), because a
  train patient's later admissions follow them into train. Patients never cross splits, so there
  is no leakage. Decision: keep and document; docs corrected 2026-11-07. Still to do:
  - add `icd_version` per admission to `cohort.py` and `dataset.py`
  - report results by era

- **Delta tables via PySpark dropped** (2026-11-08): only a Databricks-skills demonstration;
  DuckDB and Parquet already do the job. Removed from config, PLAN, README and dependencies.

## In progress

Stage 3, piece 3: every arm on validation, then a fair comparison.

1. **Arm 2 head + tail finished** 2026-11-08 (about 6 h). Stop rule, written before the result:
   chunking only if head + tail beat LoRA main with its difference interval entirely above zero.
   It did not (-0.007 [-0.029, +0.012]), so **no chunking; LoRA tuning stops here.** Both LoRA
   runs are reported.
2. Next: arm 3 timing (`arm3_llm --n 200`, MedGemma 1.5 4B then Qwen3.5-9B), then full validation
   or a sample; then the Week 3 gap-set arm (decided before any validator score); then
   `validator_rx.py` and the age-only arm; `icd_version` and subgroups.

Deferred: subgroup breakdown by age band, sex, insurance, race, ICD era.

## Next

- Open decision: keep arm 3 (prompted local LLM) or prune it, on its own merits
- Test set scored once, for every arm, after all choices are final

## Before going public

- Scan tracked files for employer names or personal details
- Recreate the GitHub repo so commits from before the 2026-11-06 history rewrite are gone from GitHub
- Then make it public (the first paired comparison is done, 2026-11-08)
