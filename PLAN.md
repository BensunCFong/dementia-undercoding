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

MIMIC-IV (hosp module) and MIMIC-IV-Note, via PhysioNet. Already credentialed.

**Two separate DUAs required:** MIMIC-IV and MIMIC-IV-Note. People routinely sign one and wonder
why the notes are missing. Confirm CITI training has not lapsed.

**Hard constraint:** the DUA prohibits redistribution and prohibits sending the data through
third-party APIs or online services, and recommends locally deployed models. Consequences:

- Every LLM in this project runs locally or on infrastructure I control. No OpenAI, no Bedrock,
  no hosted inference, no Databricks free tier.
- MLflow logs metrics, parameters and model artifacts only. Never note text, never sample
  predictions containing clinical content.
- No data in the repo, ever. CI fails the build if a data file is committed.

## 4. Compute decisions, and why

**Everything local on the Mac Studio (M1 Max, 32GB unified).**

| Stage | Where | Notes |
|-------|-------|-------|
| Cohort build (DuckDB) | Mac | Reads gzipped CSVs in place, nothing copied |
| Regex, TF-IDF, LightGBM | Mac | Trivial at this scale |
| LoRA fine-tune (110M encoder) | Mac, MPS, fp16 | Overnight run, no quantization needed |
| Prompted arm (7-8B, 4-bit) | Mac, MLX or llama.cpp | Fits comfortably in unified memory |

**Rejected: AWS.** My platform experience is Azure and Databricks, and AWS would add a third
platform without adding anything this project needs. Credits stay unspent.

**Rejected: standard Lightsail.** No GPU, and burstable CPU throttles on long DuckDB builds.

**Rejected: paid personal Azure subscription.** Fresh pay-as-you-go accounts typically start
with zero GPU quota for the NC/ND families; a quota request could eat a week. Azure Databricks
also bills DBUs on top of VMs. And the platform is not the gap: I already have Azure
experience (DP-203, AI-300, Synapse, ADLS Gen2). LoRA on a Mac is the same LoRA.

**Taken: free Azure ML workspace for MLflow tracking.** Costs nothing (no compute, no storage
used). Train locally, log runs remotely. Converts "MLflow, certification labs" into "MLflow
tracking on Azure ML, four experiment arms, model registered."

**Taken: Delta Lake and PySpark locally.** `delta-spark` in local mode gives real Delta format
with ACID semantics and time travel, plus a genuine PySpark stage in note preprocessing. Three
of the four core Databricks capabilities, on hardware I own.

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
| 3 | Few-shot prompted local instruct model | Cost and quality comparison |

**Split:** temporal, on `patients.anchor_year_group`. Train 2008-2013, test 2017-2019.
2014-2016 held out entirely because it straddles the ICD-9 to ICD-10 transition.

**Both coding systems kept.** MIMIC-IV spans 2008-2019 and the US switched to ICD-10 in October
2015, so ICD-10-only would delete the training years. The mapping is already written and tested,
so keeping both is free. The transition is also an asset: "this EHR contains a coding-system
change partway through, here is how I detected and handled it" is exactly the
reverse-engineering that working with real EHR data demands.

**Label sets:** `core` and `broad` (adds mild cognitive impairment). Running both is the
sensitivity analysis. If the coding gap moves a lot between them, that says something about
where clinicians place the diagnostic boundary.

**Metric:** PR-AUC. Positive class runs 5-9%, so accuracy is meaningless.

**Sampling:** 50,000 documents, stratified on label and split. Ample for this question and keeps
the fine-tune to an overnight run.

**Tokens:** truncate to 512 by default. Dementia history almost always appears early, in HPI and
past medical history. Chunking with stride 128 is run as a comparison to test whether the tail
carries signal — a measured decision, not an assumed one.

## 6. The hypothesis worth testing

Regex will catch the explicit mentions, because discharge summaries often just say "dementia".
The fine-tuned model only earns its place on implicit language:

> *oriented times one* · *unable to provide history* · *per daughter, increasingly forgetful at
> home* · *poor historian*

If arm 2 beats arm 0, it beats it there. Characterising that set of notes is the most
interesting page of the writeup.

## 7. Week by week

### Week 0 — scaffold (~1 hour) — BUILT

Sign both DUAs. Repo skeleton with `pyproject.toml`, `src/`, `tests/`, `Dockerfile`, GitHub
Actions running ruff and pytest, plus a separate CI job that fails if a data file is committed.
17 tests pass on a clean machine with no data present.

Done before any modelling. Retrofitting tests is how these projects die.

### Week 1 — cohort and baselines

- Build the cohort in DuckDB. Join discharge summaries to admissions to diagnosis codes.
- Write the cohort out as Delta tables via local PySpark.
- Note preprocessing: strip de-identification placeholders, collapse whitespace, keep section
  headers (they carry signal).
- Arm 0: regex baseline.
- Arm 1: TF-IDF + logistic regression, and LightGBM.
- Log everything to MLflow from the first run, not retrofitted at the end.

**First number to watch:** class balance from `make cohort`. Expect 5-9% positive on core. If it
prints 0.00, the ICD matching is broken — which is what `tests/test_codes.py` exists to prevent.

### Week 2 — the fine-tune

- Arm 2: LoRA on `Bio_ClinicalBERT` via HuggingFace PEFT. `Clinical-Longformer` (4096 tokens) as
  fallback if truncation loses too much.
- Arm 3: local instruct model, few-shot, via MLX or llama.cpp.
- Compare all four arms in MLflow. Register the winner.

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
   possible result. I have prior form here (PCO and U15 equity work).
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
- **Spark is gratuitous at this scale.** DuckDB handles 330k documents comfortably. PySpark is
  used for the note stage and for the Delta write, which is a legitimate demonstration, but be
  ready to say plainly that it was not a necessity. Claiming otherwise would not survive
  scrutiny.
- **Every decision has to be defensible.** What makes a project like this survive twenty minutes
  of questioning is being able to explain its choices: why PR-AUC, what LoRA rank and why, what
  broke, what was abandoned.

## 9. House style

No Oxford comma. Hyphens, never em or en dashes. Plain-spoken prose, first person. Specifics do
the work, not marketing language.
