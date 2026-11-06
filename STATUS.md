# Project status

Last commit: `460292f` (2026-11-06). Repo private on GitHub, CI green, 41 tests.

## Done

- **Cohort** (`make cohort`): 331,731 admissions with a discharge summary. Split by patient
  `anchor_year_group`: train 2008-2013, test 2017-2019, 2014-2016 and 2020-2022 excluded.
  Build asserts bands, clean ICD codes, one row per admission and 65+ prevalence.
- **Modelling dataset** (`make dataset`): `data/derived/notes.parquet`, baseline sections only.
  Train 225,597 notes, test 41,266. Core dementia about 4.7% on test.
- **Known limitations, documented:** about 65% note coverage; 4.1% of test notes have no history
  section and are dropped.

## Decided, not yet in code

- **Validation split:** 10% of train patients, grouped by `subject_id`, plain random draw,
  seed 3. Chosen after comparing seeds 1-4. In `config/config.yaml` but uncommitted, and the
  indentation is off: `frac` and `seed` need two more spaces so they sit under `validation:`.
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

Stage 3, piece 2: shared evaluation module (`src/dementia_undercoding/evaluate.py`).

1. PR-AUC (average precision): lesson done, not yet written. Print the prevalence baseline
   (about 0.047 on test) next to every score.
2. Bootstrap confidence intervals: next.
3. Subgroup breakdown: age band, sex, insurance, race, ICD era.

## Next

- ML dependencies are installed (torch 2.14, transformers 5.18, peft 0.21, mlflow 3.15).
  Whether MPS is available hasn't been checked yet.
- Arm 0: regex with negation. Arm 1: TF-IDF + logistic regression, LightGBM
- Arm 2: LoRA on Bio_ClinicalBERT. Time a batch before setting `sample.n`
- Open decision: keep arm 3 (prompted local LLM) or prune it

## Before going public

- Recreate the GitHub repo so commits from before the 2026-11-06 history rewrite are gone from GitHub
