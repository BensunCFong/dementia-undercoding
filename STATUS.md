# Project status

As of 2026-11-07: repo private on GitHub, CI green, 54 tests.

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

## Decided, not yet in code

- **ICD-10 in train:** train is about 19% ICD-10 (43,298 of 227,130 admissions), because a
  train patient's later admissions follow them into train. Patients never cross splits, so there
  is no leakage. Decision: keep and document. Still to do:
  - add `icd_version` per admission to `cohort.py` and `dataset.py`
  - correct PLAN §5 ("train ICD-9, test ICD-10")
  - correct the 2014-2016 exclusion comment in config
  - drop or redo the "ICD-10 years code more dementia (9.8% vs 7.8%)" claim, which compared
    splits
  - report results by era

## In progress

Stage 3, piece 3: a first LoRA run against a baseline, scored on validation only.

1. Done: MPS check, validation split, minimal scorer (`evaluate.score()` / `report()`: PR-AUC
   with prevalence and lift beside it).
2. Next: arm 1 (TF-IDF + logistic regression) on validation, logged to MLflow.
3. Then: LoRA lesson, time a 512-token batch on MPS, LoRA fine-tune of Bio_ClinicalBERT.

Deferred from piece 2 (`evaluate.py`): bootstrap confidence intervals; subgroup breakdown by
age band, sex, insurance, race, ICD era.

## Next

- ML dependencies are installed (torch 2.14, transformers 5.18, peft 0.21, mlflow 3.15).
- Arm 0: regex with negation. Arm 1: TF-IDF + logistic regression, LightGBM
- Arm 2: LoRA on Bio_ClinicalBERT. Time a batch before setting `sample.n`
- Open decision: keep arm 3 (prompted local LLM) or prune it

## Before going public

- Recreate the GitHub repo so commits from before the 2026-11-06 history rewrite are gone from GitHub
