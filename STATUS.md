# Project status

As of 2026-11-08, evening: repo public on GitHub (MIT), 125 tests locally (CI collects 116 and
skips 1, since it installs only core packages).

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
- **Arm 3 set up** (`arm3_llm.py`, 2026-11-08): MedGemma 1.5 4B and Qwen3.5-9B in llama.cpp on
  127.0.0.1, 4-bit, few-shot (2 + 2 train notes drawn by rule), scored by P(Yes) on the first
  answer token. Timed on 200 validation notes: no failures, no notes cut. Choice and context in
  `docs/explainers.md`.

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
   [-0.099, -0.046]). Neither clears the 0.730 route of the test rule; the hidden-row route is
   still open until the validator is built.
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
    - Known weaknesses: v4 never used Acute only in the pilot (my Acute only notes went to
      Possible); a history of memory loss and an unexplained "poor historian" were read as
      None; verdicts shift with any prompt change.
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
13. Next: test-scoring path (arms 1 and 1b refit on train, LoRA from its saved adapters);
    `validator_rx.py` and the age-only arm; then the arm 3 decision from the hidden row;
    `icd_version` and subgroups. The cross-check's command (run on the judge's test answers,
    the reads draw) is wired in with the test judge run.
    
Deferred: subgroup breakdown by age band, sex, insurance, race, ICD era.

## Next

- Arm 3 goes to test only under the rule in "In progress" above
- Test set scored once, for every arm, after all choices are final (no test-scoring path exists
  yet: arms 1 and 1b refit on train, LoRA loads its saved adapters)

## Public since 2026-11-08

- Tracked files and full history scanned (no data, secrets, personal paths or scrubbed terms;
  every commit uses the GitHub noreply email).
- GitHub repo deleted and recreated, so commits from before the 2026-11-06 history rewrite are
  gone from GitHub; pushed fresh as public with an MIT license (code only, not MIMIC data).
