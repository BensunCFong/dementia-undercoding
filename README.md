# Dementia under-coding in MIMIC-IV

**How often is cognitive impairment documented in clinical notes but absent from the diagnosis
codes, and which patients does that happen to?**

Dementia is under-coded in structured EHR data. Patients get described as confused, disoriented,
or with family-reported memory loss, and no diagnosis code ever appears. This project treats the
disagreements between model and codes as the finding rather than as error.

Status: **week 0** — scaffold and cohort logic. No results yet.

---

## Data access

Requires PhysioNet credentialing and two separate signed DUAs:

- **MIMIC-IV** — hosp module: `admissions`, `patients`, `diagnoses_icd`, `prescriptions`
- **MIMIC-IV-Note** — `discharge`

Place the extracted files under `data/` and point `config/config.yaml` at them.

### No data in this repository, ever

The DUA prohibits redistribution and prohibits sending the data through third-party APIs or
online services. Consequences, applied throughout:

- Every model runs locally or on infrastructure I control. No hosted inference.
- MLflow logs metrics, parameters and model artifacts only. Never note text, never sample
  predictions containing clinical content.
- CI fails the build if a data file is committed. See `.github/workflows/ci.yml`.

---

## Getting started

Do these in order. Steps 1 to 4 involve no data and no code that touches data.

**1. Sign the data use agreements.** On PhysioNet, sign **both**:

- MIMIC-IV
- MIMIC-IV-Note

These are two separate signatures. Signing only the first is a common mistake and leaves you
without any notes. While you are there, confirm your CITI training has not lapsed — credentialing
expires on a cycle and a lapsed certificate blocks download.

**2. Get the code.**

```bash
git clone <this-repo> && cd dementia-undercoding
python3 -m venv .venv && source .venv/bin/activate
make install
```

**3. Run the tests before any data exists.**

```bash
make test     # 17 tests, all pass on a clean machine with no data present
```

This is deliberate. The test suite validates ICD normalisation and matching against fixtures, not
against MIMIC, so a green run here means the labelling logic is sound before a single patient
record is downloaded.

**4. Push and confirm CI is green.** Two jobs run: lint plus tests, and a separate job that fails
the build if any data file has been committed.

**5. Only now, download the data.** Place it under `data/` and point `config/config.yaml` at it.
Nothing in `data/` is ever committed — see `data/.gitignore`.

**6. Build the cohort.**

```bash
make cohort   # writes data/cohort.duckdb and prints class balance
```

See *First number to watch* below before trusting the output.

---

## Design

Weak labels from ICD codes, four arms, temporal split, then investigation of the disagreements.

| Arm | Method | Purpose |
| --- | --- | --- |
| 0 | Regex over explicit mentions | Serious competitor, not a strawman |
| 1 | TF-IDF + logistic regression; LightGBM | Classical baseline |
| 2 | LoRA fine-tune of a clinical encoder | The capability being built |
| 3 | Few-shot prompted local instruct model | Cost and quality comparison |

**Split.** Temporal, on `patients.anchor_year_group`. Train 2008-2013, test 2017-2019.
2014-2016 is held out entirely because it straddles the ICD-9 to ICD-10 transition.

**Both coding systems kept.** MIMIC-IV spans 2008-2019 and the US moved to ICD-10 in October
2015, so an ICD-10-only cohort would delete the training years. Both mappings live in
`config/codes.yaml` and are driven from the same file in Python and SQL.

**Label sets.** `core` and `broad` (adds mild cognitive impairment). Running both is the
sensitivity analysis: if the gap moves a lot between them, that says something about where
clinicians place the diagnostic boundary.

**Metric.** PR-AUC. The positive class runs around 5-9%, so accuracy is meaningless here.

**Sampling.** 50,000 documents, stratified on label and split.

**Tokens.** Truncate to 512 by default — dementia history almost always appears early, in HPI and
past medical history. Chunking with stride 128 is run as a comparison to test whether the tail
carries signal. A measured decision, not an assumed one.

---

## Pre-registered decision: separating dementia from delirium

Recorded before any results are seen.

MIMIC-IV is heavily ICU, and delirium is endemic there. "Confused", "disoriented", "oriented
times one" are produced by both conditions. Without a stated rule, a gap set that is mostly
delirium would look like a finding and would not be one.

**Target definition.** Chronic *baseline* cognitive impairment, not acute confusion. Dementia is
a pre-admission state; delirium is an acute deviation from it.

**Primary approach — section restriction.** The model sees baseline-describing sections only:
Past Medical History, History of Present Illness, Social History. Brief Hospital Course and daily
assessments are excluded, because that is where delirium lives. Section parsing in MIMIC-IV-Note
is inconsistent, so the fraction of notes successfully parsed is reported as a result.

**Reporting — stratification, not exclusion.** The gap is reported separately for
delirium-coded and non-delirium-coded admissions. Delirium-coded admissions are *not* dropped:
dementia is the single largest risk factor for delirium, so excluding them would remove exactly
the patients most likely to have undiagnosed dementia and would bias the study toward null.

**Validator 1 — medication proxy.** Cholinesterase inhibitors (donepezil, rivastigmine,
galantamine) and memantine are close to dementia-specific and sit in `prescriptions`. Not used as
a label, since they capture treated dementia only. Used to check the gap set: elevated rates in
model-positive/code-negative admissions is orthogonal evidence the gap is real.

**Validator 2 — future-coded confirmation.** MIMIC-IV has multiple admissions per patient. A
patient who is note-positive and code-negative in 2016 and then carries a dementia code in 2018
was almost certainly under-coded on the earlier admission. A within-patient prospective check
using data already in hand.

---

## The hypothesis worth testing

Regex will catch the explicit mentions, because discharge summaries often just say "dementia".
The fine-tuned model only earns its place on implicit language:

> *oriented times one* · *unable to provide history* · *per daughter, increasingly forgetful at
> home* · *poor historian*

If arm 2 beats arm 0, it beats it there. Characterising that set of notes is the point of the
whole exercise.

---

## Stack

Local, on an M1 Max with 32GB unified memory.

- **DuckDB** — cohort build, reads the gzipped CSVs in place
- **PySpark + delta-spark** — note preprocessing, cohort written as Delta tables
- **scikit-learn, LightGBM** — arms 0 and 1
- **PyTorch, HuggingFace Transformers, PEFT** — arm 2, LoRA on MPS in fp16
- **MLX or llama.cpp** — arm 3, 7-8B instruct model at 4-bit
- **MLflow** — all four arms tracked, local file backend or a personal Azure ML workspace

---

## Layout

```
config/codes.yaml     ICD code sets, core and broad, with prefix-matching notes
config/config.yaml    paths, cohort filters, split, sampling, tokens, MLflow, Delta
src/.../codes.py      normalisation and matching. The riskiest code in the repo.
src/.../cohort.py     DuckDB cohort build
src/.../text.py       de-identification cleanup and chunking
tests/                17 tests, mostly guarding the code-matching logic
```

---

## Running it

```bash
make install
make test
make cohort
make lint
```

### First number to watch

`make cohort` prints class balance by split. Expect **5-9% positive** on the core set.

If it prints `0.00`, the ICD matching is broken. That is the failure mode `tests/test_codes.py`
exists to prevent: MIMIC stores ICD codes without decimal points and sometimes space-padded, so
matching a literal `331.0` silently labels nothing, the model trains on an all-negative target,
and everything downstream looks broken for reasons that are hard to trace.

---

## What would count as a null result

Any of these, and they get published as readily as a positive finding:

- The coding gap is small, and MIMIC coding is better than expected.
- Regex wins, and the fine-tuned model adds nothing.
- The gap set is mostly delirium once the validators are applied.
