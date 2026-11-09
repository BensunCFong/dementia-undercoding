# Project plan: dementia under-coding in MIMIC-IV

Working document. Supersedes the earlier project brief.

---

## 1. The question

How often is cognitive impairment documented in clinical notes but absent from the diagnosis
codes, and which patients does that happen to?

Dementia is under-coded in structured EHR data. Patients get described as confused, disoriented,
or with family-reported memory loss, and no diagnosis code ever appears. The disagreements
between model and codes are the finding, not the error.

**Personal angle:** I published on neural stem cell dysfunction in the 3xTg-AD model. I know the
biology of this disease. I do not know how it gets recorded in hospitals.

## 2. Why this project exists

I have trained models in AI-300 certification labs (training orchestration, hyperparameter
tuning, MLflow tracking, model comparison across jobs, fine-tuning methods) and deployed
pretrained models at scale in production. What I have not done is choose a question, build the
dataset myself, and decide independently whether the result means anything.

Guided labs on curated data with a known answer are not the same as owning a modelling question
end to end. That is the gap, and four weeks closes it.

Secondary purpose: find out whether I enjoy this work enough to move into it. Permission to
conclude "no" is part of the design.

Personal infrastructure only: my own hardware, my own accounts and my own time, so ownership of
the work is never in question.

## 3. Data and constraints

MIMIC-IV v3.1 (hosp module) and MIMIC-IV-Note v2.2, via PhysioNet. Credentialing completed
2026-09-04; CITI approval was the gating step, not the download.

**Two separate DUAs required:** MIMIC-IV and MIMIC-IV-Note. People routinely sign one and wonder
why the notes are missing. Confirm CITI training has not lapsed.

**Note coverage is ~65%, not 100%.** Only 331,731 of 546,028 admissions in v3.1 have a discharge
summary, and the shortfall is consistent across bands (61.6%-67.5%), so it is a property of
MIMIC-IV-Note rather than a version-mismatch artefact.

This changes what the result is a statement about: not "how often is cognitive impairment
under-coded in MIMIC-IV admissions" but "...in admissions **that have a discharge summary**."
The missing third is unlikely to be random — short stays, same-day procedures and observation
admissions are both the least likely to generate a summary and the least likely to carry a
chronic diagnosis code. State the qualifier explicitly; do not let a reader find it.

**A further 4.1% of test admissions have no history section.** Their discharge summaries skip
History of Present Illness and Past Medical History entirely, so the baseline-section rule leaves
nothing to read (train loses 0.7%). They over-represent nursing-facility transfers, walk-ins, and
discharges to hospice or psychiatric care, and carry dementia codes at 6.3% vs 4.7% (12.5% vs
9.7% at 65+), so dropping them removes about 5% of test dementia cases. Not explained by age.
A read of 5 sampled notes (2026-10-04) found no pointer to a separate note ("see H&P",
"transferred from"). They simply use a different, problem-based format that starts at Brief
Hospital Course (SUMMARY, ACUTE ISSUES, CHRONIC ISSUES, TRANSITIONAL ISSUES) with no history block.
Five notes is a small sample; the cause is unknown. Same treatment as note coverage: state it,
do not bury it. Possible stage 4 sensitivity check: run the trained model on the CHRONIC ISSUES
text of the roughly 21% of these notes that have one, and see whether the gap moves.

**`2020 - 2022` is excluded on data coverage.** MIMIC-IV v3.1 extends to a fifth
`anchor_year_group` band that MIMIC-IV-Note v2.2 predates: exactly 1 of its 42,491 admissions has
a discharge summary. There is no text for any arm to read. The COVID confound would also justify
exclusion, but the coverage fact settles it without needing that argument.

**Hard constraint:** the DUA prohibits redistribution and prohibits sending the data through
third-party APIs or online services, and recommends locally deployed models. Consequences:

- Every LLM in this project runs locally or on infrastructure I control. No OpenAI, no Bedrock,
  no hosted inference, no Databricks free tier.
- MLflow logs metrics and parameters only; trained models stay under `data/`. Never note text, never sample
  predictions containing clinical content.
- No data in the repo, ever. CI fails the build if a data file is committed.

## 4. Compute decisions, and why

**Everything local on the Mac Studio (M1 Max, 32GB unified).**

| Stage | Where | Notes |
|-------|-------|-------|
| Cohort build (DuckDB) | Mac | Reads gzipped CSVs in place, nothing copied |
| Regex, TF-IDF, LightGBM | Mac | Trivial at this scale |
| LoRA fine-tune (110M encoder) | Mac, MPS, full precision | About 3 h per pass; fp16 gave no speed-up on this GPU |
| Prompted arm (4B and 9B, 4-bit) | Mac, llama.cpp | About 4 GB of memory for the 4B; one note at a time keeps the GPU about 94% busy |

**Rejected: AWS.** My platform experience is Azure and Databricks, and AWS would add a third
platform without adding anything this project needs. Credits stay unspent.

**Rejected: standard Lightsail.** No GPU, and burstable CPU throttles on long DuckDB builds.

**Rejected: paid personal Azure subscription.** Fresh pay-as-you-go accounts typically start
with zero GPU quota for the NC/ND families; a quota request could eat a week. Azure Databricks
also bills DBUs on top of VMs. And the platform is not the gap: I already have Azure
experience (DP-203, AI-300, Synapse, ADLS Gen2). LoRA on a Mac is the same LoRA.

**Planned, not used: free Azure ML workspace for MLflow tracking.** The idea was to train
locally and log runs remotely. In practice MLflow runs on a local SQLite store under `data/`, with
metrics and parameters only, and no model is registered anywhere. Nothing about the science
needed a remote tracker.

**Dropped 2026-11-08: Delta Lake and PySpark locally.** The original plan used `delta-spark` in
local mode for real Delta format with ACID semantics and time travel, plus a PySpark stage in
note preprocessing. It was there to demonstrate Databricks skills, not because the science
needed it: DuckDB and Parquet already do every job it would have done. Deferred 2026-10-03,
dropped 2026-11-08 before any Spark code was written.

**Accepted gap: Unity Catalog.** Databricks-managed, no local equivalent, not worth putting
MIMIC on a hosted service to chase.

## 5. Design

Weak labels from ICD codes, four arms, temporal split, then investigation of disagreements.

**Arms:**

| # | Method | Purpose |
|---|--------|---------|
| 0 | Regex over explicit mentions | Serious competitor, not a strawman |
| 1 | TF-IDF + logistic regression; LightGBM on same features | Classical baseline |
| 2 | LoRA fine-tune of a clinical encoder (HuggingFace PEFT) | The capability being built |
| 3 | Few-shot prompted local LLM: MedGemma 1.5 4B and Qwen3.5-9B | Cost and quality comparison |

**Split:** temporal, on `patients.anchor_year_group`. Train 2008-2013, test 2017-2019.
2014-2016 held out entirely because it straddles the ICD-9 to ICD-10 transition. The anchor
band belongs to the patient, not the admission, so train patients bring their later admissions
with them: train is about 19% ICD-10 (43,298 of 227,130 admissions), while test is all but one
admission ICD-10. Patients never cross splits, so this is not leakage, but "train is ICD-9, test
is ICD-10" is not accurate. Validation is 10% of train patients (seed 3), drawn by patient.

**Both coding systems kept.** MIMIC-IV spans 2008-2019 and the US switched to ICD-10 in October
2015, so ICD-10-only would delete the training years. The mapping is already written and tested,
so keeping both is free. The transition is also an asset: "this EHR contains a coding-system
change partway through, here is how I detected and handled it" is exactly the
reverse-engineering that working with real EHR data demands.

**Label sets:** `core` and `broad` (adds mild cognitive impairment). Running both is the
sensitivity analysis. If the coding gap moves a lot between them, that says something about
where clinicians place the diagnostic boundary.

**Metric:** PR-AUC. Positive class runs about 4% across all adults (8-10% among admissions aged
65+), so accuracy is meaningless.

**Sampling:** none. Every train and test note is kept. After timing a batch (2026-11-07), arm 2
trains on all 202,905 train notes, two passes, overnight. The test set is never sampled.

**Tokens:** truncate to 512 by default. Dementia history almost always appears early, in HPI and
past medical history. Whether the tail carries signal was measured, not assumed: a second LoRA
run read the first 128 plus the last 382 tokens and matched the main run (2026-11-08), so by a
stop rule written before that result, chunking was not run.

## 6. The hypothesis worth testing

Regex will catch the explicit mentions, because discharge summaries often just say "dementia".
The fine-tuned model only earns its place on implicit language:

> *oriented times one* · *unable to provide history* · *per daughter, increasingly forgetful at
> home* · *poor historian*

If arm 2 beats arm 0, it beats it there. Characterising that set of notes is the most
interesting page of the writeup.

## 7. Week by week

### Week 0 — scaffold — BUILT

Sign both DUAs. Repo skeleton with `pyproject.toml`, `src/`, `tests/`, `Dockerfile`, and a
`.github/workflows/ci.yml` declaring ruff, pytest, and a separate job that fails if a data file
is committed. 17 tests pass locally in a clean venv with no data present.

Done before any modelling. Retrofitting tests is how these projects die.

**Two corrections to the original entry, which claimed more than was true:**

- *"~1 hour"* was wrong. Week 0 grew Docker, CI, a pre-commit hook and this document — all of it
  in the column I already know how to do. Infrastructure feels like progress because it is
  unambiguous. It is not the skill this project exists to build.
- *"GitHub Actions running ruff and pytest"* had **never executed** when this entry was first
  written: there was no git remote, so GitHub had never seen the repo. **Resolved 2026-10-03:**
  the repo now has a private GitHub remote, and both CI jobs (lint plus tests, and `no-data`)
  have run and passed on every push since. "Tests pass on a clean machine" is now verified, not
  claimed.

The repo stayed private until commit authorship was settled (every commit now carries a GitHub
noreply address) and went public on 2026-11-08, recreated so no pre-rewrite history remains.
Pushing early and privately was the right call: a committed data file was still a cheap fix
then, and would have been a public history rewrite after.

### Week 1 — cohort and baselines

- Build the cohort in DuckDB. Join discharge summaries to admissions to diagnosis codes.
- ~~Write the cohort out as Delta tables via local PySpark.~~ Dropped 2026-11-08 (§4).
- Note preprocessing: strip de-identification placeholders, collapse whitespace, keep section
  headers (they carry signal).
- Arm 0: regex baseline.
- Arm 1: TF-IDF + logistic regression, and LightGBM.
- Log everything to MLflow from the first run, not retrofitted at the end.

**First number to watch:** class balance from `make cohort`. The original expectation was 5-9%
positive on core. The first full build (2026-10-03) gave 3.9-4.7% by split across all adults.
That is not a matching failure: over half the cohort is under 65, and among admissions aged 65+
the rate is 7.8-9.8%. The 5-9% figure was never sourced; it fits older inpatients, not all
adults. The cohort keeps all adults on purpose, since early-onset dementia is real. The build
now checks the 65+ rate per split against a 5-15% sanity range. Test codes more dementia than
train at 65+ (9.8% vs 7.8%), but the splits differ in era, coding system and follow-up at once,
so this is not evidence that ICD-10 itself codes more dementia. Reporting results by ICD version
(an `icd_version` column, still to add) will separate them.

### Week 2 — the fine-tune

- Arm 2: LoRA on `Bio_ClinicalBERT` via HuggingFace PEFT. `Clinical-Longformer` (4096 tokens) was
  the fallback if truncation lost too much. Not needed: reading the end of the note did not help
  (§5, Tokens).
- Arm 3: two local models, few-shot, via llama.cpp. Chosen 2026-11-08, before any arm 3 score:
  - **MedGemma 1.5 4B** (Google): small and medically trained.
  - **Qwen3.5-9B** (Alibaba): larger and general, the common choice for a model this size on a
    32 GB Mac.

  One medical, one general, both run locally at 4-bit (Q4_K_M), so the pair also asks whether
  medical training beats a bigger general model on these notes. The score is P(Yes) on the first
  answer token, not generated text; the four examples are train notes drawn by a fixed rule. Not
  the basis for the choice, but consistent with it: a UCLA preprint from October 2026 (Lee et
  al., MedJev, arXiv 2610.36006) uses both models as baselines, reads binary answers the same
  way, and runs locally on Apple silicon with MIMIC-IV (details in `docs/explainers.md`).
- Compare the arms with a paired bootstrap over patients (`compare_arms.py`); MLflow keeps each
  run's metrics. Record the winner (not "register": that uploads the model, which stays under
  `data/`).

**STOP POINT.** If life intervenes here, four arms plus a temporal split plus a result is
publishable. Everything after is upside.

### Week 3 — the actual science

Take every admission where the model says yes and the codes say no. That set is the finding.

1. **LLM-as-judge.** A local model reads the note and adjudicates whether cognitive impairment is
   genuinely documented. **Validate the judge against 100 I read myself and report agreement.**
   An unvalidated judge is the standard failure in this pattern, and the writeup says so.
2. **Custom metric.** Coding gap rate: share of code-negative admissions with confirmed
   documented impairment.
3. **Subgroup analysis.** By sex, age band, insurance, race/ethnicity. If under-coding
   concentrates in particular groups, that is a fairness finding and the most interesting
   possible result. I have done equity analysis before (PCO and U15 equity work).
4. **Delirium contrast.** Separate chronic documented impairment from acute confusion, the likely
   dominant false positive. Contrast codes are already in `config/codes.yaml`.

Prior chart review experience is relevant here: the 2011 Ottawa Patch Test Clinic retrospective
review that became the *Dermatitis* paper was this same activity.

### Week 4 — ship

Dockerfile, README with results and no note text, model card, writeup. Post the writeup.

## 8. Risks and honest caveats

- **The gap may be small.** Then the finding is that MIMIC coding is better than expected and the
  model is mostly generating false positives. Write that up too. A project that reports the
  unglamorous answer is more credible than one that finds what it went looking for.
- **Regex may win.** Also a real result, and arguably the more useful one.
- **Delirium contamination.** "Oriented times one" on post-op day two is delirium, not dementia.
  Where that line sits determines the finding, and it is a judgment call to make explicitly and
  document, not to bury.
- **Spark is gratuitous at this scale.** DuckDB handles 330k documents comfortably, which is
  why the PySpark and Delta stage was dropped (§4).
- **Every decision has to be defensible.** What makes a project like this survive twenty minutes
  of questioning is being able to explain its choices: why PR-AUC, what LoRA rank and why, what
  broke, what was abandoned.

## 9. House style

No Oxford comma. Hyphens, never em or en dashes. Plain-spoken prose, first person. Specifics do
the work, not marketing language.
