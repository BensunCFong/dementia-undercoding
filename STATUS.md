# Project status

As of 2026-11-07: repo private on GitHub, 61 tests.

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
  Train 76,124 patients / 202,905 notes; validation 8,458 / 22,692 (core 3.61%, chance PR-AUC
  about 0.036); test untouched.
- **MPS:** PyTorch sees the M1 Max GPU (4096x4096 matmul about 3.8x faster than CPU).
- **Arm 1, TF-IDF + logistic regression** (`arm1_tfidf.py`): fixed, untuned settings. Validation
  PR-AUC 0.730 (chance 0.036, lift 20x, 820 positives); train 0.822.
- **Arm 2 set up** (`arm2_lora.py`): Bio_ClinicalBERT + LoRA, rank 8, alpha 16, query and value,
  dropout 0.1; 295,681 trainable weights (0.27%). Batch 16 at 512 tokens: 0.84 s per step,
  about 3 h per pass. fp16 gave no speed-up on this GPU, so training is full precision.
  Pre-declared fallback: rank 12, alpha 24, runs only if validation PR-AUC < 0.76.
- **Token lengths** (Bio_ClinicalBERT tokenizer, test not counted): median 451 train / 460
  validation; 40.9% / 42.6% of notes exceed 512 and are truncated. Baseline text runs HPI, then
  PMH, then Social History, so truncation cuts PMH first, where dementia is most often listed.
  Arm 1 reads full text; arm 2 the first 512 tokens. Follow-up to choose before reading arm 2:
  PMH first, head and tail, or chunking.
- **Per-note scores** saved under `data/derived/scores/` (hadm_id, score) for paired comparison.
- **MLflow** moved to SQLite under `data/` (MLflow 3.x refuses the old file store).

## Decided, not yet in code

- **ICD-10 in train:** train is about 19% ICD-10 (43,298 of 227,130 admissions), because a
  train patient's later admissions follow them into train. Patients never cross splits, so there
  is no leakage. Decision: keep and document. PLAN §5, README, the config exclusion comment and
  the "ICD-10 codes more dementia" claim were corrected 2026-11-07. Still to do:
  - add `icd_version` per admission to `cohort.py` and `dataset.py`
  - report results by era

## In progress

Stage 3, piece 3: a first LoRA run against a baseline, scored on validation only.

1. Done: MPS check, validation split, minimal scorer, arm 1, LoRA set-up and timing.
2. Running (started 2026-11-07 ~21:40): arm 2 main run, all 202,905 train notes, two passes,
   about 25,400 steps, done ~03:50. Log in `data/derived/logs/arm2-main.log`. A watcher starts
   the rank-12 fallback on its own if validation PR-AUC < 0.76 (done ~10:15). Full schedule and
   optional slots in `docs/explainers.md` ("Run plan").
3. Next: re-run arm 1 to save its per-note scores; bootstrap CIs, including a paired arm 1 vs
   arm 2 comparison; real token counts.

Deferred from piece 2 (`evaluate.py`): bootstrap confidence intervals; subgroup breakdown by
age band, sex, insurance, race, ICD era.

## Next

- ML dependencies are installed (torch 2.14, transformers 5.18, peft 0.21, mlflow 3.15).
- Arm 0: regex with negation. Arm 1 LightGBM variant
- Open decision: keep arm 3 (prompted local LLM) or prune it

## Before going public

- Recreate the GitHub repo so commits from before the 2026-11-06 history rewrite are gone from GitHub
