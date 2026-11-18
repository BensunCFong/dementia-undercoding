# Project status

As of 2026-11-18: repo public on GitHub (MIT), 196 tests locally (CI installs only core
packages, so it collects fewer). Test opened and scored, the test judge run finished, and my 60
blind reads are done and frozen. Every choice before test results was written (items 14-19),
then the judge-output lock was opened: judge v4 failed its agreement rule (kappa 0.44, item
20), so its verdicts are not used. No arm's test metric has been looked at. Next: judge v6, a
second attempt reported after v4's failure (item 20).

## Results so far (validation only; no test arm result seen)

Intervals: 95%, from 2,000 resamples of patients (`python -m dementia_undercoding.compare_arms`).
Difference = arm minus arm 2 LoRA main, with every arm scored on the same resampled patients.

| Arm | How it reads the note | PR-AUC [95%] | Minus LoRA main [95%] | Notes |
|---|---|---|---|---|
| Chance | | 0.036 | | 820 positives of 22,692 notes, 8,458 patients |
| Age only | Age at admission, no text | 0.148 [0.125, 0.178] | -0.620 [-0.651, -0.581] | floor beyond chance (added 2026-11-18); the other arms reproduced exactly when it was added |
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

**Drug-order validator** (`python -m dementia_undercoding.validator_rx`, run 2026-11-18; rules
in `docs/explainers.md`, fixed before this run). Among uncoded validation notes, positive = a
dementia-drug order in the same admission. Difference = arm minus the age-only arm, on the same
2,000 resampled patients. Every count reproduced the 2026-11-08 count exactly; the product-code
rule added no positives on validation.

| Arm | Main [95%] | Minus age only [95%] | Hidden [95%] | Minus age only [95%] |
|---|---|---|---|---|
| Chance | 0.009 (193 positives, 120 patients) | | 0.004 (92 positives, 58 patients) | |
| Age only | 0.036 [0.028, 0.048] | | 0.014 [0.009, 0.021] | |
| 0, regex | 0.201 [0.142, 0.277] | +0.164 [+0.108, +0.237] | 0.004 [0.003, 0.006] | -0.010 [-0.015, -0.006] |
| 0+drugs (partly checks itself) | 0.217 [0.155, 0.296] | +0.180 [+0.123, +0.256] | 0.004 [0.003, 0.006] | -0.010 [-0.015, -0.006] |
| 0, first 512 tokens | 0.179 [0.126, 0.249] | +0.143 [+0.091, +0.209] | 0.004 [0.003, 0.006] | -0.010 [-0.015, -0.006] |
| 1, TF-IDF + LR | 0.194 [0.140, 0.271] | +0.158 [+0.107, +0.230] | 0.014 [0.009, 0.021] | -0.000 [-0.005, +0.005] |
| 1b, LightGBM | 0.208 [0.150, 0.287] | +0.171 [+0.117, +0.247] | 0.013 [0.009, 0.020] | -0.001 [-0.006, +0.005] |
| 2, LoRA main | 0.193 [0.138, 0.261] | +0.157 [+0.103, +0.222] | 0.013 [0.009, 0.020] | -0.001 [-0.006, +0.006] |
| 2, LoRA head + tail | 0.234 [0.170, 0.318] | +0.198 [+0.138, +0.276] | 0.015 [0.010, 0.023] | +0.001 [-0.005, +0.008] |
| 3, MedGemma 1.5 4B | 0.093 [0.061, 0.140] | +0.057 [+0.026, +0.099] | 0.007 [0.005, 0.014] | -0.007 [-0.012, +0.001] |
| 3, Qwen3.5-9B | 0.249 [0.179, 0.338] | +0.212 [+0.144, +0.300] | 0.010 [0.006, 0.017] | -0.004 [-0.010, +0.003] |

Reading, against the rules written before the run:
- Main row: every arm ranks treated, uncoded admissions far above age alone. The intervals
  overlap heavily, so this row does not rank the arms against each other.
- Hidden row, the implicit-language test (PLAN §6): no arm beats age only. Once a treated
  patient's history text does not name dementia, no arm ranks them above their age. This is no
  support for the idea from this yardstick, not a disproof: 58 patients, treated dementia only,
  and the trained arms were taught that treated, uncoded train notes are negatives.
- Clean row (14 notes naming a drug left out, 183 positives): scores barely move, so the arms
  were not leaning on drug names. Arm 0 and 0+drugs become identical, as they should.
- Sensitivity row (MCI kept, 200 positives): no change worth noting.
- Age bands (description only): within 85+, age only falls to chance (0.039 vs 0.041); every
  arm keeps its signal in both bands (under 85: 63 patients; 85+: 61), so the main row is not
  age in disguise. Delirium-coded positives: 13 notes, 11 patients, a count only.
- Saved: `data/derived/validator/validation-table.parquet` (every row and band) and
  `validation-topk.parquet` (the top-k curves, for the Week 4 figure).

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
- **Arm 3 set up** (`arm3_llm.py`, 2026-11-08): MedGemma 1.5 4B and Qwen3.5-9B in llama.cpp on
  127.0.0.1, 4-bit, few-shot (2 + 2 train notes drawn by rule), scored by P(Yes) on the first
  answer token. Timed on 200 validation notes: no failures, no notes cut. Choice and context in
  `docs/explainers.md`.
- **ICD-10 in train:** train is about 19% ICD-10 (43,298 of 227,130 admissions), because a
  train patient's later admissions follow them into train. Patients never cross splits, so there
  is no leakage. Decision: keep and document; docs corrected 2026-11-07.
  - Built 2026-11-18 as a lookup, not a column: `cohort.icd_versions(diagnoses_path, hadm_ids)`.
    `notes.parquet` is not rebuilt for it, because the test scores, the judge's bands and my
    blind reads are all tied to that file. Rule, fixed after a count of versions per admission
    (codes and labels only): the highest version among the admission's codes; no codes, no
    version. Notes, ICD-10 / ICD-9 / no codes: train minus validation 37,704 / 165,107 / 94 (5
    of the ICD-10 mix both versions); validation 4,204 / 18,480 / 8; test 41,256 / 1 / 9.
    `n_codes` matches `diagnoses_icd` for every note.
  - Test is all ICD-10, so the era cannot split any test result. It can only be described
    within train and validation, where version still moves with age and follow-up (a train
    patient's ICD-10 admissions are their later ones). How it is reported: item 19.
- **Delta tables via PySpark dropped** (2026-11-08): only a Databricks-skills demonstration;
  DuckDB and Parquet already do the job. Removed from config, PLAN, README and dependencies.

## In progress

Stage 3 done on validation (every arm, fair comparison). Week 3 on test: judge v4 failed its
agreement rule (item 20). Next: judge v6, developed on the 90 notes I have already read, then
run on test and checked against fresh blind reads.

1. **Arm 2 head + tail finished** 2026-11-08 (about 6 h). Stop rule, written before the result:
   chunking only if head + tail beat LoRA main with its difference interval entirely above zero.
   It did not (-0.007 [-0.029, +0.012]), so **no chunking; LoRA tuning stops here.** Both LoRA
   runs are reported.
2. **Arm 3 test rule, written 2026-11-08 evening before the full validation results were seen**
   (MedGemma 1.5 4B and Qwen3.5-9B running overnight, about 16 h):
   - A model goes to test only if, on validation, its code-label PR-AUC is at least arm 1's
     (0.730), **or** it beats the age-only arm on the drug validator's hidden row with the
     difference interval entirely above zero.
   - If both models qualify, only the better one goes to test (by code-label PR-AUC; by the hidden
     row if their difference interval includes zero).
   - Arm 3's validation results are reported whatever the outcome; if it does not go to test, the
     write-up says so and gives this rule.
   - Arms 0, 1, 1b and both LoRA runs are frozen and may be scored on test without waiting for arm
     3, once the Week 3 rules (arm, split, cutoff) are written.
3. **Arm 3 full validation finished** 2026-11-09: MedGemma 1.5 4B 0.447 [0.399, 0.491], Qwen3.5-9B
   0.657 [0.610, 0.700]. Qwen beats arm 0 (+0.035 [+0.012, +0.058]) and trails arm 1 (-0.073
   [-0.099, -0.046]). Neither clears the 0.730 route of the test rule. **Decided 2026-11-18 by
   that rule: arm 3 does not go to test.** Neither beats age only on the drug validator's hidden
   row (MedGemma -0.007 [-0.012, +0.001], Qwen -0.004 [-0.010, +0.003]). Their validation
   results are reported, Qwen's main-row 0.249 included, with the rule.
4. **Week 3 gap-set arms, decided 2026-11-09 after the code-label results and before any validator
   score:** two gap sets side by side, LightGBM (best on code labels) and LoRA main (reads words
   in context, so its extra flags are the test of the implicit-language idea). Arm 3 is not a
   gap-set arm. Judge: Qwen3.5-9B, decided 2026-11-09 after arm 3's validation results (larger
   general model; MedGemma weaker on every measure). Code-label PR-AUC is not evidence of judging
   skill; the judge is trusted only if it meets an agreement rule against 100 notes read blind by
   me, to be written before it judges anything.
5. **Week 3 split, decided 2026-11-09:** develop on validation, report on test. On validation:
   the cutoff rule, the judge rubric and prompt (tried on a handful of gap-set notes) and the
   judge agreement rule; nothing measured there is reported as the result. Then everything is
   frozen and run once on test (2017-2019): both gap sets, the judge, my 100 blind reads and the
   gap rate by subgroup. Any look at validation gap sets is description only, not an era result.
   Needs a test-scoring path (LightGBM refit on train, LoRA from its saved adapter).
6. **Week 3 cutoff, decided 2026-11-09:** no single line. Each arm's notes are ranked and cut
   into depth bands measured in multiples of the split's coded count (validation 820; test about
   4.7%), the same bands for both arms. The gap set in each band is its uncoded notes. Reported
   per band: the share the judge confirms as documented impairment, and how much of that is
   hidden (no explicit term). A random sample from below the last band is also judged, so the gap
   rate becomes an estimate, not only a floor. My 100 blind reads are spread across the bands.
   Band edges and sample size are fixed after timing the judge with its rubric on validation. The
   drug validator plays no part in setting the bands. Reason: hidden cases likely score lower than
   explicit ones, so a single line at the coded count would mostly miss them.
7. **Week 3 reporting, decided 2026-11-09:** the judge reads each flagged note once (both arms
   pooled). My 100 reads are blind to arm, band and judge verdict. One test only: at one depth
   (fixed when the bands are set), LoRA-only vs LightGBM-only notes, share confirmed as hidden
   dementia, patient-level bootstrap interval; the idea is supported only if the interval is
   entirely above zero. The groups may be small, so a null is likely even if the idea is true,
   and is reported as such. Everything else (yield per band, overlap counts) is a descriptive
   table, no further tests; two or three bands at most. Any lens chosen after the results is
   labelled so and reported as exploratory, never in place of this test.
   LoRA main was trained once; a second seed was considered and declined on 2026-11-09, before
   test. The bootstrap covers luck in which patients were drawn, not luck in training.
8. **Judge rubric, decided 2026-11-09 before the judge has read any note:**
   - Question: does the note document chronic baseline cognitive impairment (README target)?
     The judge does not diagnose. "Hidden" stays defined by arm 0 (no explicit term in the
     history sections), not by the judge.
   - Verdicts: Documented / Possible (includes MCI) / Acute only / None. Primary gap rate uses
     Documented; sensitivity version uses Documented + Possible; Acute only is counted as the
     delirium check.
   - Input: the whole discharge note, not only the history sections the arms read. The judge is
     the reference standard, not another arm. My reads use the same text.
   - Output: up to 3 sentences quoted word for word, then the verdict, in a fixed JSON shape
     forced by llama.cpp. Code checks every quote is in the note (made-up quotes counted and
     reported) and finds each quote's section, so confirmations from outside the history are
     reported as such. Output saved under `data/` only. Thinking mode off.
   - Agreement: Cohen's kappa, Documented vs the rest, judge against my blind reads; sensitivity,
     specificity and the 4x4 table also reported. The judge is trusted if the point estimate is
     at least 0.6; the interval is reported.
   - If it fails on test: judge verdicts are not used, the gap rate comes from my 100 reads alone,
     and the LoRA-vs-LightGBM test is reported as not run, with this rule as the reason.
   - Pilot: about 30 validation gap-set notes read blind by me and by the judge, to adjust the
     prompt before freezing. Development only, never reported as the result.
9. **Judge built and timed, 2026-11-09** (`judge.py`; development on validation, not a result).
   50 notes from a provisional gap set (uncoded notes in either arm's top 2x coded count, 1,311
   notes): 12.7 s per note median, about 3.7 h per 1,000 notes; no note cut or refused.
   **Evidence by constrained decoding, decided after the first timing run and before the
   pilot:** the judge does not retype its evidence; it picks up to 3 passages (the note's own
   sentences and list lines, bare headings left out) and llama-server allows only those. The
   first run, with free quoting, had 1 edited quote in 140 (a dropped "it" next to a
   de-identified gap) and 11 passages joined together; with picked passages, 148 of 148 were
   exact. Every quote is still checked in code and given a tier (exact / formatting / joined /
   edited / not copied); any non-exact quote would mean a bug in the splitting.
   The verdict mix changed between the two runs (Possible 19 to 22, Acute only 17 to 14,
   Documented 14 to 12, None 0 to 2: at least 5 of 50 notes), from the evidence format alone.
   Verdicts are sensitive to prompt and format, so the prompt is frozen after the pilot and any
   later change counts as a new judge.
10. **Judge pilot and freeze, 2026-11-09** (development on validation; not the reported agreement).
    30 notes read blind by me (15 from the top 2x pool, 15 from 2x-4x, timing-run notes left
    out), judged by five prompt versions. After seeing the judge's evidence I revised 5 of my 30
    reads (a home dementia drug, Aricept, I had missed; two baseline descriptions; two acute
    cases); both my blind and revised reads are kept. Against my revised reads:
    v1 13/30 (Documented line kappa 1.00), v2 17/30, v3 26/30 (0.84), v4 25/30 (1.00),
    v5 22/30 (1.00). Against my blind reads, v4: 22/30, Documented line 0.71.
    - v3 found the main error: the judge read routine normal-cognition lines ("A&Ox3",
      "clear and coherent") as grounds for Acute only, because of the rubric's wording.
    - v4 (my idea): decide in order, Documented first. Fixed a plainly stated dementia that v3
      under-called (listed with a psychiatric diagnosis).
    - v5 (my idea, weigh Acute only against Possible using "returned to baseline") was tested
      against a rule written before it ran (Documented 4/4 with none false, all 13 None kept,
      Acute only better than 0/3). It failed the third condition and was rejected.
    - **Frozen: v4** (ef7d6a2), plus dropping repeated passages in code (llama-server
      ignores uniqueItems; repeats were common). Any later prompt change is a new judge.
    - Known weaknesses: v4 never used Acute only in the pilot (my 3 revised Acute only reads
      all went to Possible; blind, 3 of my 4 went to Possible and 1 to None); a history of
      memory loss and an unexplained "poor historian" were read as None; verdicts shift with
      any prompt change.
    - Consequences: Acute only and None are reported together as "no chronic impairment
      documented"; the delirium contrast uses the delirium codes, as the README pre-registered;
      the sensitivity version (Documented + Possible) is an upper bound, since it takes in some
      acute cases. An automatic cross-check is added on test: notes where arm 0 finds an
      explicit, non-negated mention but the judge does not say Documented.
    - **Test reads changed from 100 to 60, before test was touched:** stratified toward notes
      likely to be Documented (aiming for about 20 or more), with weights set alongside the band
      edges; kappa reported with sensitivity and specificity. The 30 pilot reads are development
      only and never pooled with the test reads.
11. **Week 3 test design, decided 2026-11-09 before test was touched:**
    - Bands: each arm's test notes ranked and cut at 1x, 2x and 4x the test coded count (about
      1,940 notes), giving bands 1x / 1-2x / 2-4x. Every uncoded note either arm puts inside 4x is
      judged, about 8,000 notes and about 30 h of GPU time (frozen judge v4, about 12.7 s and 7 GB
      per note).
    - The one test (item 7) is at 2x: the 1x and 1-2x bands together. On validation the one-arm
      groups there were 389 (LoRA only) and 355 (LightGBM only), against 60 and 43 at 1x; the
      pilot found all its Documented notes inside 2x and none in 2x-4x.
    - Random sample below 4x: 1,000 uncoded notes, judged (about 4 h), so the gap rate is an
      estimate and not only a floor. I read every sample note the judge calls Documented, to
      confirm it; these reads are separate from the 60 agreement reads.
    - Agreement reads: 60, stratified 35 from 1x, 15 from 1-2x, 10 from 2-4x, drawn at random
      within each band with a fixed seed, shuffled, shown by ID only and read blind before I see
      any judge output. Aim about 20 Documented; reported as they fall, with no topping up.
      Kappa (Documented vs the rest), sensitivity and specificity reported, with the band mix
      stated.
    - Arm scoring on test is a separate step: about 1 h for arms 0, 1, 1b, 2 and age-only, plus
      about 29 h for arm 3 only if it qualifies under its test rule.
12. **Arm 0 cross-check, decided 2026-11-09 before test was touched** (the check promised in
    item 10):
    - Arm 0's frozen rules (2026-11-07: terms, negation, window) run on the text the judge saw:
      the whole discharge note, cut at the judge's 60,000 characters. Not the history sections
      arm 0 was scored on, so that a mention outside them, seen by the judge, is still checked.
    - A note is flagged when it has at least one certain mention (not negated, not someone else,
      not uncertain) and the judge did not say Documented. Uncertain-only mentions ("concern
      for dementia") are not flagged: they fit Possible.
    - Run on both judged sets: the 4x pool and the random 1,000 below 4x. Counts reported per
      set.
    - I read the flagged notes, blind to the judge's verdict: all of them, or a random 50 (fixed
      seed) from the combined list if there are more. Each read is tagged with its set and
      reported as it falls, separate from the 60 agreement reads and the sample's Documented
      reads.
    - A flag never changes a judge verdict; the gap rate stays the judge's. The reads measure
      how often the judge misses a plainly stated dementia.
    - Limit: these reads are not fully blind, since every flagged note is one the judge did not
      call Documented. They check the judge's misses; the 60 agreement reads remain the blind
      measure.
    - Diagnosis terms only, for the flags and the reads. A note can be Documented through a drug
      name alone (pilot P06, home Aricept), which this check cannot see. So a count is added,
      per set and never read: notes with a certain mention of one of arm 0's drug names, no
      certain diagnosis mention, and no Documented verdict. It sizes how far the reads may
      understate the judge's misses, and it never changes a verdict.
13. **Test-scoring path, decided 2026-11-09 before test was touched** (`score_test.py`):
    - Arms 1 and 1b refit on exactly what they used for validation (train minus validation,
      same settings and seeds), so test is scored by the validated models and LightGBM vs
      LoRA stays fair (LoRA's adapters saw train minus validation). LoRA loads its saved
      adapters. Each refit or reloaded model must first reproduce its saved validation scores
      (within 1e-5), or nothing is written.
    - Scores only: no metric is printed or logged when test is scored. Results are computed in
      a separate step, after the remaining choices (validator, arm 3, subgroups) are made.
    - Locked: refuses to run until `test_scoring.allowed` is set to true in its own commit,
      which records when test was opened.
    - Scored once: an existing test score file is refused, never overwritten.
    - Route to the judge: STATUS item 2 already allows scoring arms 0, 1, 1b and both LoRA runs on
      test now. The rest of this list is CPU work on validation and is done while the judge runs;
      the judge's test run prints no verdicts, and no test result is looked at until then.
    - Test judge run (`judge.py test-draw`, `test-judge`): bands per arm at 1x / 2x / 4x of the
      test coded count; a note the two arms place differently takes the shallower band (decided
      2026-11-09). Random 1,000 from uncoded notes inside neither arm's 4x. My 60 reads drawn
      35 / 15 / 10 by band, random within band, shuffled, no topping up if a band is short.
      The judge works through a shuffled order, saves every 25 notes, resumes after a stop and
      prints progress only. Seeds fixed in config before test was touched.
    - 2026-11-09: test opened (2117c67). The first scoring run was stopped by hand after arm 0, to
      move it to another window; arm 0's test files were deleted and the loop restarted. Arm 0 is
      a fixed regex (same scores on rerun) and no metric was computed or seen.
    - 2026-11-09: test scored, all five arms (0, 1, 1b, LoRA main, LoRA head + tail), 41,266
      notes. Arms 1, 1b and both LoRA runs reproduced their saved validation scores with a gap of
      exactly 0.0.
    - 2026-11-09: test draw (`test-draw`). 1,924 coded notes. Pool of uncoded notes inside either
      arm's 4x: 8,140 (619 in 1x, 2,533 in 1-2x, 4,988 in 2-4x). Random sample: 1,000 of the
      31,202 uncoded notes below 4x. My reads filled 35 / 15 / 10. To judge: 9,140 notes.
    - 2026-11-09 to 2026-11-11: test judge run (`test-judge`), finished the evening of
      2026-11-11. The Mac's weekday 22:00 sleep schedule paused it for about 20 h (resumed
      cleanly). It printed progress only; no verdict has been looked at.
    - Pushed fd435e8..5d9da78 on 2026-11-09; CI result not yet checked.
14. **My 60 blind agreement reads, done 2026-11-15:** finished before any judge output was opened
    and frozen before any compare code existed: a read-only copy,
    `data/derived/judge/test/reads-blind.csv`, SHA-256
    `cf2889f5914978985e11321e33609b05a2888eed19b608b1754ced720afa2f7d`. Agreement is
    reported against these reads. Any later revision goes in `reads.csv` and is reported
    beside them, never in their place (as in the pilot).
15. **Agreement compare** (`judge.py test-agreement`), decided 2026-11-15 before any judge answer
    was opened. Built now; locked (`judge.test.output_allowed`) and run only after the
    validator, age-only arm, arm 3 decision, `icd_version` and subgroup rules are written,
    with the subgroup rules covering both a judge pass and a fail. Reason: its 4x4 table is an
    early look at the gap rate, so it must not steer those choices.
    - Documented vs the rest, on `reads-blind.csv`; the judge is trusted if kappa is at least
      0.6 (point estimate, item 8).
    - Interval: 2,000 redraws of patients within each band, so the 35 / 15 / 10 mix is kept;
      95% percentile; seed 20261115. The same redraws give sensitivity and specificity
      intervals. Redraws where a figure cannot be computed are counted and printed.
    - A read note with no valid judge answer is left out of the main figures (the pass rule
      uses them) and listed with its read ID and cause. Beside them, a worst case counts each
      failure as a disagreement with my read. Counting a failure as "not Documented" was
      rejected: it scores a failure as agreement whenever my read is not Documented.
    - Per band: counts only (notes, no answer, same verdict, Documented by me / the judge /
      both), labelled description. No kappa per band: 10 notes in 2-4x is too few.
16. **One lock for all judge output on test, decided 2026-11-16:** `judge.test.output_allowed`
    shuts the agreement compare, the cross-check and the sample's Documented draw alike, and
    is opened in one commit once the choices above are written. Reason: running the other two
    also opens judge answers (the sample's Documented count is the gap-rate estimate below
    4x), the same reason the compare was locked.
17. **Cross-check and sample reads, built 2026-11-16, locked, not run:**
    - `judge.py test-cross-check`: arm 0's cross-check (item 12) on the pool and the sample,
      counts per set, then up to 50 flagged notes drawn for my reads, blind to the verdict.
    - `judge.py test-sample-reads`: every sample note the judge called Documented,
      shuffled (seed 83), for me to confirm (item 11).
    - Each writes a key, the notes and a blank reads sheet under `data/`, and refuses to
      redraw. The two read lists stay separate, as items 11 and 12 say: the sample reads
      confirm the judge's positives and are not meant to be blind to its verdict.
18. **Remaining choices, done 2026-11-18:** `validator_rx.py` and the age-only arm, run on
    validation (results at the top); the arm 3 decision (item 3: does not go to test);
    `icd_version` as a lookup (2026-11-18, "ICD-10 in train" under Done).
19. **Subgroup rules, decided 2026-11-18 before any judge output was opened.** Taken one at a
    time; the counts behind them are codes, ages and recorded values on validation only.
    - **What is split.** The missed share leads: of the notes coded for dementia or confirmed
      by the judge, the share with no code. The gap rate (PLAN's metric: the share of uncoded
      notes the judge confirms) is shown beside it. Reason: by group, the gap rate mostly
      tracks how common dementia is (age above all), not how well it is coded. The missed
      share takes every code as correct, since coded notes were never judged; the write-up
      says so. Both are description only.
    - **Verdicts.** Both versions by group: Documented (primary, item 8) and Documented +
      Possible beside it. Each version gets a figure only if its own cases pass the size rule.
    - **If the judge fails** (kappa below 0.6): no missed share or gap rate by group. My 60
      reads all sit inside the 4x pool and are too few to split. The coded rate by group is
      shown instead, labelled as describing coding, not under-coding, with the statement that
      the subgroup analysis was not run and this rule as the reason.
    - **Groups**, fixed now and carried unchanged to test:
      - Age at admission: under 75 / 75-84 / 85+. Chosen from coded dementia patients on
        validation by 5-year age: under 65 holds 17 coded patients in 11,840 notes, too few
        for a band; two equal halves would cut at 84. Coded patients on validation: 80 / 176 /
        250 (a patient can sit in two bands).
      - Sex as recorded in MIMIC (M / F). Sex and age overlap (more women at 85+); nothing is
        adjusted for it.
      - Insurance: Medicare, Medicaid, Private. Other, missing and No charge are pooled and
        shown as a count only. Insurance largely tracks age: Medicare covers 65 and over, and
        holds 407 of the 454 coded patients on validation.
      - Race and ethnicity as the hospital recorded it (one field, 33 values), grouped by
        prefix into named groups: White; Black / African American; Hispanic / Latino; Asian;
        American Indian / Alaska Native; Native Hawaiian / Pacific Islander; Multiple race /
        ethnicity; Unknown (unknown, unable to obtain, declined). SOUTH AMERICAN goes to
        Hispanic / Latino and PORTUGUESE to White, following the US federal categories. The
        recorded value OTHER is shown as recorded "Other", in quotes. My decision: every group
        is named and none is pooled to reach a size, so small groups appear as counts rather
        than disappear into a merged row. Asian and Pacific Islander stay separate for the
        same reason.
    - **Size.** A group gets a figure only with at least 30 real patients behind the missed
      share: coded plus judge-confirmed uncoded, each patient counted once, never scaled up by
      the sample's weight. Below that, counts only. On a judge fail, the same rule applies to
      coded patients for the coded rate.
    - **Small counts.** Every count from 1 to 10 in the subgroup tables is shown as "fewer
      than 11" (0 shows as 0), because the repo is public.
    - **How shown.** Estimate and 95% interval per group, under an all-patients row; no
      differences between groups and no tests (item 7). Missed share and gap rate are counted
      in notes, the size rule in patients. Below 4x, each sample note stands for 31,202 /
      1,000 notes.
    - **Interval.** Patients redrawn within each part (the 1,924 coded notes, the 8,140-note
      pool, the 1,000-note sample), 2,000 redraws, 95% percentile, with a new seed committed
      to config before the run. A patient whose notes fall in two parts is handled separately
      within each.
    - **No valid answer.** A judged note without one counts neither as confirmed nor in its
      part's uncoded total; the number left out is listed per part. Beside the main figures, a
      worst case counts every failure as confirmed (Documented, and Documented + Possible for
      the wider version). Same treatment as the agreement compare.
    - **ICD era is not a subgroup.** Test is all ICD-10 (41,256 notes; 1 ICD-9, 9 with no
      codes), and the judge ran only on test, so there is no gap figure by era anywhere.
      Reported instead, as coding only: within train minus validation, ICD-10 admissions are
      coded for dementia more often than ICD-9 ones, 5.4% against 3.5% (2,046 of 37,704
      against 5,859 of 165,107), and 9.5% against 7.4% at 65 and over. Confounded: a train
      patient's ICD-10 admissions are their later ones, further into follow-up and older.
20. **Judge v4 fails on test, 2026-11-18.** The lock was opened in its own commit (adf524f) and
    the agreement compare run on `reads-blind.csv`. Every read note had a valid judge answer; 60
    patients, none read twice.
    - Kappa, Documented against the rest: **0.44 [0.12, 0.71], below 0.6, so the judge is not
      trusted.** Sensitivity 0.42 (5 of my 12 Documented; the other 7 it called Possible),
      specificity 0.96 (46 of 48).
    - My read (rows) against the judge, in the order Documented / Possible / Acute only / None:
      Documented 5 / 7 / 0 / 0; Possible 0 / 17 / 1 / 4; Acute only 1 / 9 / 2 / 4; None
      1 / 2 / 0 / 7. Same verdict on 31 of 60.
    - Per band, Documented by me / the judge / both (description): 1x 11 / 6 / 5; 1-2x
      1 / 1 / 0; 2-4x 0 / 0 / 0.
    - By the rules written before (items 8 and 19): v4's verdicts are not used, the one test
      (item 7) is reported as not run, and by group only the coded rate is shown. This result
      is reported first and in full, whatever a later judge does. The cross-check and the
      sample reads (item 17) have not been run.

    **What went wrong** (checked after the result; patterns only, no note text):
    - No bug in passage splitting. In the 4 misses whose note names dementia, that line was
      among the passages the judge could pick, and it never picked it. These were long notes
      (158-286 passages); in 4 hits it did pick the line (75-159 passages).
    - In the other misses the judge saw the decline and said Possible. v4's wording overlapped:
      "memory loss over months" under Documented, "history of memory problems" under Possible.
    - Confusion still present at discharge was passed over. The discharge "Mental Status:
      Confused" line is in 24 of the 60 notes, and the judge picked it in 4. Three of the 4
      Possible reads it called None have confusion lasting to discharge; in one it picked that
      line and still said None.
    - Reading the side by side, I found 3 of my blind reads I would now read differently (a
      memory-loss line in the history I passed over; two Acute only reads). `reads-blind.csv`
      is unchanged (item 14).

    **Judge v6, decided 2026-11-18 after the v4 result.** A new judge, chosen after a failure on
    test was seen, so it is reported as a second attempt, after v4's result. My 60 test reads
    have now been seen beside v4's answers: from here they are development notes and cannot
    measure a new judge.
    - **Input, trim B (frozen).** The judge reads only Chief Complaint, History of Present
      Illness, Past Medical History, Social History, Physical Exam, Brief Hospital Course,
      Discharge Disposition, Discharge Diagnosis and Discharge Condition: whole sections,
      repeats kept; a note with none of these headings stays whole. Dropped: results and labs,
      medications, Family History, procedures, instructions. No keyword rescue of dropped
      lines. This replaces item 8's whole-note input for v6; I still read whole notes.
      On the validation 4x pool (3,395 notes): passages per note, median 140 to 72 (90th
      percentile 211 to 115); 53% of the text kept. Notes losing every certain dementia
      mention (arm 0's rules): 66 of 436. 56 of these are Family History, a relative that arm
      0's clause rule takes for the patient ("Mother: dementia"), so about 10 (2.6%) are the
      patient's own, mostly in results (the reason given for a scan). Side effect: the
      cross-check (item 12, whole note) will flag family-history lines.
    - **Rubric, decided one point at a time (my decisions):**
      1. A drug name alone is not evidence, for the judge or my reads; drug-only notes are
         counted separately. This keeps the judge independent of the drug validator, and
         reverses the pilot's reading of home Aricept (item 10).
      2. Dementia named with a word of conclusion (likely, probable, presumed, known, history
         of) is Documented; with an open question (possible, concern for, rule out, may have,
         suspected) it is Possible. Arm 0 counts "probable" as uncertain, so the two differ
         there.
      3. Documented: impairment from before admission, either a stated impaired cognitive
         baseline (non-verbal alone does not count) or decline in memory or thinking over
         about a month or more, not put down to an acute cause. A bare label (memory loss,
         forgetful) or MCI is Possible.
      4. Acute only: confusion this admission with an acute cause (infection, sepsis, hypoxia,
         a drug, metabolic, trauma, a new stroke, bleed or mass) or a named delirium, and no
         sign it lasts to discharge. Possible: no clear cause or timing; put down only to a
         long-standing condition (psychiatric illness, old strokes) without point 3's baseline
         or span; confusion still present at discharge ("Confused - sometimes", reduced
         orientation) with no earlier baseline; delirium not resolved. If the patient died,
         the cause decides.
      5. Physical decline is not evidence unless the note ties it to memory or thinking. An
         unexplained "poor historian" is Possible. Anyone else's dementia is never evidence.
      The prompt points to the sections to read (the discharge mental status line among
      them), with no keyword list.
    - **Bake-off, three local 4-bit models:** Qwen3.5-9B (v4's model), Gemma 4 12B (Google's
      own QAT Q4_0 file) and Granite 4.2 8B (IBM's own Q4_K_M file; thinking off). Same trim
      and prompt; v4's prompt kept frozen beside. Run on the 90 notes I have read (30
      validation pilot, 60 test). A model passes only if it beats v4's Documented-line
      agreement on the same notes; the rule for picking among passers is written before any
      run. These figures flatter every model, since the same notes shaped the rubric, and are
      never reported as agreement.
    - **Then:** the winner judges the test pool and sample (9,140 notes) and is checked against
      20-30 fresh blind test reads, none of them among my 60. Their count and draw, and v6's
      trust rule, are fixed before that run.
    - **Phase 2, later, with its own rules written first:** the arms on wider input (adding
      Chronic and Transitional Issues; never Acute Issues or Discharge Diagnosis, which would
      leak the label) and LoRA on chunks of about 4 x 512 tokens. Phase 1 results are kept
      beside, not replaced.
    - **Open, to decide before v6 touches test:** whether v6's verdicts may feed the gap rate,
      the one test (item 7) and the subgroups (items 8 and 19 were written for v4); how a gap
      rate comes from my 60 reads alone (drawn 35 / 15 / 10 by band); whether the cross-check
      and sample reads run for v4. I chose not to rule now that v6 is the last version.

## Next

- Judge v6 (item 20): trim and prompt in code, v4 kept frozen beside; the bake-off rule
  written; then the bake-off on the 90 notes I have read
- Before v6 touches test: the open choices in item 20, the fresh reads' count and draw, and
  v6's trust rule
- Test results computed once, after all choices are final (scores already written, item 13)
- Week 4: README results with no note text, model card, write-up, Dockerfile

## Public since 2026-11-08

- Tracked files and full history scanned (no data, secrets, personal paths or scrubbed terms;
  every commit uses the GitHub noreply email).
- GitHub repo deleted and recreated, so commits from before the 2026-11-06 history rewrite are
  gone from GitHub; pushed fresh as public with an MIT license (code only, not MIMIC data).
