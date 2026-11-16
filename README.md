# Dementia under-coding in MIMIC-IV

**How often is cognitive impairment documented in a hospital note but absent from the diagnosis
codes, and which patients does that happen to?**

Dementia is under-coded in structured hospital data. A discharge summary can describe a patient
as forgetful at home, unable to give a history, or living with dementia outright, and the
admission still carries no dementia code. Anyone who counts dementia from codes alone misses
those patients.

This project trains models to predict the dementia code from the note, then looks at where a
model says yes and the codes say no. Those disagreements are treated as the finding, not as
error. A separate judge, checked against notes I read myself, decides how many of them are real.

## Why I am doing this

One of my doctoral research projects was to characterize neural stem cell dysfunction and
transcriptomic alterations in the 3xTg-AD mouse model of Alzheimer's disease. I know the biology
of this disease. I did not know how it gets recorded once a patient is in hospital, and that
record is what many large studies of dementia are built on. If the codes miss people, and miss
some groups more than others, every count that starts from codes inherits the gap. This project
follows the same disease from the mechanism to the record.

It is also deliberate practice. I had trained models in guided labs and deployed pretrained ones
in production, but had not chosen a question, built the dataset and decided for myself whether
the result means anything. Everything here runs on my own hardware, in my own time.

## Where it stands (2026-11-15)

| Stage | State |
| --- | --- |
| Cohort and modelling dataset | Done |
| Every arm scored on validation, compared with a paired patient bootstrap | Done, results below |
| Week 3 rules: gap sets, judge, my reads, the one test | Written before test was touched |
| Test scored by every frozen arm | Done 2026-11-09, **no metric computed or seen** |
| Judge run on 9,140 test notes | Done 2026-11-11, **no verdict seen** |
| My 60 blind reads of test notes | Done and frozen 2026-11-15 |
| Agreement compare, judge against my reads | Built, **locked** |
| Drug-order validator, age-only arm, subgroups | Rules partly written, code not built |
| Test results | Not computed. They are computed once, after every choice above is fixed |

No test result exists yet, by design. The order matters more than the speed: every rule that
could bend a result is written down, with its date, before the number it governs is seen.
`STATUS.md` is that record; this page is the summary.

## Results so far: validation only

Labels are ICD codes. 22,692 validation notes from 8,458 patients, 820 coded for dementia
(3.6%). Intervals are 95%, from 2,000 resamples of patients; the difference column scores both
arms on the same resampled patients.

| Arm | How it reads the note | PR-AUC [95%] | Minus LoRA main [95%] |
| --- | --- | --- | --- |
| Chance | | 0.036 | |
| 0, regex | Explicit mentions, with negation and "someone else" handling | 0.622 [0.574, 0.664] | -0.146 [-0.176, -0.119] |
| 1, TF-IDF + logistic regression | Each word separately | 0.730 [0.688, 0.766] | -0.038 [-0.063, -0.014] |
| 1b, LightGBM | Words in combination | 0.792 [0.757, 0.824] | +0.023 [+0.002, +0.045] |
| 2, LoRA on Bio_ClinicalBERT, first 512 tokens | Words in context | 0.768 [0.734, 0.800] | |
| 2, LoRA, first 128 + last 382 tokens | Words in context | 0.761 [0.723, 0.796] | -0.007 [-0.029, +0.012] |
| 3, MedGemma 1.5 4B, few-shot | Prompted, no training | 0.447 [0.399, 0.491] | |
| 3, Qwen3.5-9B, few-shot | Prompted, no training | 0.657 [0.610, 0.700] | |

What this says:

- The fine-tuned model clearly beats the regex and TF-IDF + logistic regression.
- LightGBM probably ranks slightly better than LoRA. The interval only just clears zero and it is
  one of several comparisons, so I do not lean on it.
- Reading the end of the note as well as the start did not help, so truncation does not explain
  LightGBM's lead. The stop rule for more LoRA tuning was written before that result, and it
  said stop.
- Neither prompted model reached arm 1.

What it does not say: which arm is most useful. The labels are the codes, and the project is
about the codes being wrong. A note that describes dementia and carries no code counts against a
model here. About 290 validation notes mention dementia plainly and are uncoded. That is why the
real test is Week 3 and not this table.

## Design

**Data.** MIMIC-IV discharge summaries, one per admission, joined to the admission's ICD codes.
331,731 admissions have a summary. The modelling dataset keeps 225,597 train notes and 41,266
test notes.

**Label.** A dementia code on the admission (`core` set, ICD-9 and ICD-10, in
`config/codes.yaml`). A `broad` set that adds mild cognitive impairment is built into the cohort
as a planned sensitivity analysis; every result so far uses `core`.

**Split.** By time, on the patient's `anchor_year_group`: train 2008-2013, test 2017-2019.
2014-2016 is left out because it straddles the US move from ICD-9 to ICD-10. The band belongs to
the patient, so a train patient's later admissions stay in train: train is about 19% ICD-10, test
almost all ICD-10. No patient is in two splits. Validation is 10% of train patients.

**What the models read.** Only the sections that describe the patient before admission: History
of Present Illness, Past Medical History, Social History. See the delirium rule below for why.

**Metric.** PR-AUC, with prevalence beside it. About 4% of notes are positive, so accuracy would
be meaningless.

**The arms.**

| Arm | Method | Why it is here |
| --- | --- | --- |
| 0 | Regex over explicit mentions, rules fixed before any note was read | A serious competitor, not a strawman |
| 1, 1b | TF-IDF + logistic regression; LightGBM on the same features | Classical baselines |
| 2 | LoRA fine-tune of Bio_ClinicalBERT, rank 8, about 0.3% of weights trained | The capability being built |
| 3 | Few-shot prompted local LLMs | What no training at all buys |

No sampling: arm 2 trains on every train note.

## The hypothesis

Regex catches the explicit mentions, because a summary often just says "dementia". A fine-tuned
model earns its place only on language that implies impairment without naming it: a patient
oriented to self only, a history taken from the daughter because the patient cannot give one, a
family report of growing forgetfulness.

On validation, LoRA did beat the regex against the codes. But the codes cannot show whether its
extra flags are hidden dementia or noise, and LightGBM, which only counts words, scored about as
well. So the question became sharper for Week 3: among uncoded notes, does the model that reads
words in context find more **hidden** dementia (no explicit term) than the one that counts them?

## Week 3: the gap sets, the judge and my reads

All of this was written before the test set was touched (2026-11-09).

**Gap sets.** Two, side by side: LightGBM (best against the codes) and LoRA (reads in context).
Each arm's test notes are ranked by score and cut into depth bands at 1x, 2x and 4x the number of
coded notes (1,924 on test). The gap set in a band is its uncoded notes. There is no single
cutoff, because hidden cases are likely to score lower than explicit ones.

| | Notes |
| --- | --- |
| Uncoded test notes inside either arm's 4x | 8,140 (619 in 1x, 2,533 in 1-2x, 4,988 in 2-4x) |
| Random sample of uncoded notes below 4x | 1,000 of 31,202 |
| Judged in total | 9,140 |

The random sample is there so the gap rate is an estimate for the whole test set, not only a
floor from the notes the models liked.

**The judge.** A local LLM (Qwen3.5-9B through llama.cpp) reads the **whole** discharge note and
answers one question: does this note document chronic, baseline cognitive impairment? It gives
one of four verdicts: Documented, Possible, Acute only, None. It does not diagnose. Its evidence
is up to three passages picked from the note itself; llama-server allows no other text, and code
checks every passage is really in the note. The prompt was developed on 30 validation notes and
frozen (version 4); any later change counts as a new judge. On test the run printed progress
only.

**My reads.** A judge nobody checked is the usual failure of this method. I read 60 test notes
blind: 35 from the 1x band, 15 from 1-2x, 10 from 2-4x, shown by ID only, with no arm, band or
judge output. They were finished before any judge answer was opened, and a read-only copy was
fixed by its SHA-256 in a commit before the comparing code existed.

**The trust rule.** Cohen's kappa on Documented against the rest, judge against my reads. The
judge is trusted if kappa is at least 0.6. If it is not, its verdicts are not used: the gap rate
comes from my 60 reads alone and the test below is reported as not run. Sensitivity, specificity
and the full 4x4 table are reported either way, with intervals from patients redrawn within
bands.

**The one test.** At 2x depth: notes only LoRA flags against notes only LightGBM flags, share
confirmed as hidden dementia, with a patient-level interval. The idea is supported only if the
interval sits entirely above zero. The groups may be small, so a null is likely even if the idea
is true, and would be reported as that. Everything else is a descriptive table.

**Two more checks, both needing my reads.** Notes where the frozen regex finds a plainly stated
dementia but the judge did not say Documented (how often does the judge miss the obvious?), and
every note in the random sample that the judge calls Documented (is the estimate below 4x real?).

**Still to do before any test result.** Build the drug-order validator and the age-only
reference arm, decide whether a prompted model goes to test under its rule, add the ICD era to
the cohort and write the subgroup rules. The agreement compare is locked in config until then,
because its table is an early look at the gap rate and must not steer those choices.

## Pre-registered: separating dementia from delirium

Recorded before any results were seen.

MIMIC-IV is heavily ICU, and delirium is common there. Confusion and disorientation are produced
by both conditions. Without a stated rule, a gap set that is mostly delirium would look like a
finding and would not be one.

**Target.** Chronic *baseline* cognitive impairment, not acute confusion. Dementia is a state the
patient arrives with; delirium is an acute change from it.

**Section restriction.** The arms see baseline-describing sections only. Brief Hospital Course
and daily assessments are left out, because that is where delirium is written down. Across all
331,793 discharge notes, 98.8% yield some baseline text: History of Present Illness is non-empty
in 98.1% and Past Medical History in 95.9%. Social History is non-empty in only 3.0%, because
MIMIC-IV-Note de-identifies it to a bare placeholder, so in practice the arms read HPI and Past
Medical History. The judge is the reference, not an arm, so it reads the whole note.

**Stratify, do not exclude.** The gap is reported separately for delirium-coded and other
admissions. Delirium-coded admissions are not dropped: dementia is the largest risk factor for
delirium, so dropping them would remove exactly the patients most likely to have undiagnosed
dementia.

**Validator 1, dementia drug orders.** Cholinesterase inhibitors and memantine are close to
dementia-specific and sit in `prescriptions`. They are never a training label. Among uncoded
admissions, each arm will be scored on how well it ranks the ones with a dementia-drug order:
evidence from outside the codes. Rules were fixed on 2026-11-08 before any arm was scored on it
(`docs/explainers.md`), including a row for drug-positive notes with no explicit mention and an
age-only arm as the floor. The code is not built yet.

**Validator 2, a later code.** A patient flagged and uncoded on one admission who carries a
dementia code on a later one was probably under-coded the first time. Planned; no rules written
yet.

## Known limits

- **Admissions with no history section.** 4.1% of test admissions (1,757) have a summary with no
  history section, so no arm can read them and they are left out. They are not a random 4%: they
  come more often from nursing-facility transfers, and carry dementia codes at 6.3% against 4.7%
  for the rest of test. Results describe admissions with a documented history.
- **Note coverage.** About 65% of admissions have a discharge summary at all.
- **One training run.** LoRA was trained once. The bootstrap covers luck in which patients were
  drawn, not luck in training.
- **The judge is a model.** It is trusted only as far as my 60 reads support, and its verdicts
  shifted with small prompt changes during development. That is why it was frozen.
- **Drug orders mean treated dementia.** A low rate on the validator is weak evidence against an
  arm, and donepezil is also used off-label for mild cognitive impairment.
- **One reader.** The blind reads are mine alone; there is no second reader.

## What would count as a null result

Any of these, reported as readily as a positive finding:

- The coding gap is small, and MIMIC coding is better than expected.
- The fine-tuned model finds no more hidden dementia than LightGBM, or than the regex.
- The gap set is mostly delirium once the validators are applied.
- The judge fails its agreement rule, and the gap rate rests on 60 notes.

---

## Data access

Requires PhysioNet credentialing and two separately signed data use agreements:

- **MIMIC-IV**, hosp module: `admissions`, `patients`, `diagnoses_icd`, `prescriptions`
- **MIMIC-IV-Note**: `discharge`

### No data in this repository, ever

The agreement prohibits redistribution and prohibits sending the data through third-party APIs
or online services. So:

- Every model runs locally, including the judge. No hosted inference.
- MLflow logs metrics and parameters only. Trained models, per-note scores, judge answers and my
  reads stay under `data/`, which is never committed.
- Commands print counts, never note text. Tests use invented notes only.
- CI and a pre-commit hook fail if a data file is committed.

## Getting started

Steps 1 to 4 involve no data.

**1. Sign both agreements on PhysioNet**: MIMIC-IV and MIMIC-IV-Note. They are separate
signatures, and signing only the first leaves you without notes. Check your CITI training has
not lapsed.

**2. Get the code.**

```bash
git clone <this-repo> && cd dementia-undercoding
python3.11 -m venv .venv && source .venv/bin/activate
make install                          # core + dev packages
pip install -e ".[ml]"                # PyTorch, PEFT, MLflow: needed to run the arms
git config core.hooksPath .githooks   # pre-commit hook that refuses data files
```

Arm 3 and the judge also need `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp)
on the path and the GGUF model files named in `config/config.yaml`.

**3. Run the tests before any data exists.**

```bash
make test
```

Every test passes on a clean machine with no data present.

**4. Download the data**, place it under `data/` and point `config/config.yaml` at it.

**5. Build the cohort and the modelling dataset.**

```bash
make cohort    # writes data/derived/cohort.duckdb and prints class balance by split
make dataset   # writes data/derived/notes.parquet: baseline sections only
```

`make cohort` fails on its own if the dementia rate among admissions aged 65+ leaves 5-15% in any
split. A rate near zero means the ICD matching is broken: MIMIC stores codes without decimal
points, so matching a literal `331.0` silently labels nothing. `tests/test_codes.py` exists to
prevent that.

**6. Run the arms on validation and compare them.**

```bash
python -m dementia_undercoding.arm0_regex     # likewise arm1_tfidf, arm1b_lgbm, arm2_lora
python -m dementia_undercoding.arm3_llm --model qwen3.5-9b    # or medgemma-1.5-4b
python -m dementia_undercoding.compare_arms   # paired bootstrap over every saved score
```

**7. Test, once.** Each of these is locked or refuses to repeat itself, on purpose.

```bash
python -m dementia_undercoding.score_test arm1b     # also arm0, arm1, arm2-main, arm2-main-head_tail
python -m dementia_undercoding.judge test-draw      # bands, random sample, the blind reads
python -m dementia_undercoding.judge test-judge     # about 32 h of GPU time here; resumes after a stop
python -m dementia_undercoding.judge test-agreement # my reads against the judge
python -m dementia_undercoding.judge test-cross-check # arm 0 cross-check, and the flagged notes I read
python -m dementia_undercoding.judge test-sample-reads # the sample notes the judge called Documented
```

- `score_test` runs only once `test_scoring.allowed` is set in its own commit, writes scores and
  no metric, and refuses to overwrite a score file.
- `test-draw` refuses to redraw.
- `test-agreement`, `test-cross-check` and `test-sample-reads` run only once
  `judge.test.output_allowed` is set in its own commit; the last two refuse to redraw.
## Stack

Local, on an M1 Max with 32 GB of unified memory.

- **DuckDB**: cohort build, reads the gzipped CSVs in place
- **scikit-learn, LightGBM**: arms 1 and 1b
- **PyTorch, HuggingFace Transformers, PEFT**: arm 2, LoRA on MPS at full precision
- **llama.cpp** (`llama-server`, bound to 127.0.0.1): arm 3 and the judge, 4-bit models
- **MLflow**: metrics and parameters, local SQLite under `data/`

## Layout

```
STATUS.md                  every decision and result, dated, in the order it was made
PLAN.md                    the original plan
docs/explainers.md         plain-language notes on the methods and the choices
config/codes.yaml          ICD code sets, core and broad
config/config.yaml         paths, split, every arm's fixed settings, seeds, the two locks
src/.../codes.py           ICD normalisation and matching
src/.../cohort.py          DuckDB cohort build
src/.../dataset.py         modelling dataset: baseline sections of each discharge summary
src/.../text.py            de-identification cleanup, section parsing
src/.../splits.py          validation split, drawn by patient
src/.../arm0_regex.py      arm 0, explicit mentions with NegEx-style context
src/.../arm1_tfidf.py      arm 1, TF-IDF + logistic regression
src/.../arm1b_lgbm.py      arm 1b, LightGBM on arm 1's features
src/.../arm2_lora.py       arm 2, LoRA on Bio_ClinicalBERT
src/.../arm3_llm.py        arm 3, prompted local LLM via llama.cpp
src/.../evaluate.py        PR-AUC, saved scores, paired patient bootstrap
src/.../compare_arms.py    every arm against LoRA main, same resampled patients
src/.../truncation.py      how much explicit signal each 512-token window keeps
src/.../score_test.py      test scoring: locked, scores only, once
src/.../judge.py           the judge, the test draw, the arm 0 cross-check, the agreement compare
tests/                     no data needed; invented notes only
```

## License

The code is MIT-licensed (`LICENSE`). The license covers this repository's code and documents
only. MIMIC-IV data is not included and stays under its PhysioNet data use agreement.
