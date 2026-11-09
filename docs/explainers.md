# Explainers

Plain-language explanations of the evaluation and modelling choices, written while building
them. Each section answers a question someone could reasonably ask about this project.

## What PR-AUC measures, and what "820 positives" means

The model sorts every validation note from most to least likely to be dementia-coded.
**PR-AUC** (average precision) summarises how close to the top of that list the coded notes
land. A model that sorts at random scores about the positive rate, so every score is printed
next to that chance level.

On validation, 22,692 notes all have text; **820 are positives** (the admission carries a
dementia code) and 21,872 are negatives. The score is made from where those 820 land, so they
set how precise the score is. The share (3.6%) sets the bar to beat.

A poll is the right picture: 1,000 respondents give about the same accuracy whether the country
has 5 million people or 300 million. The count earns the precision, not the share. As a rough
guide (the bootstrap will give the real answer):

| Positives | Rough swing on a score near 0.73 |
|---|---|
| 50 | about ±0.12 |
| 820 (validation) | about ±0.03 |
| 8,000 | about ±0.01 |

## Two jobs, in order: test the model, then use it

1. **Test the model.** The 820 coded notes are the only ones where the answer is known to be
   yes. Does the model put them near the top? That is the PR-AUC.
2. **Use the model.** Once trusted, look at the *uncoded* notes it scores highest. Those are
   the candidates for "documented in the note but not coded", which is the project's question.

Job 2 is only defensible if job 1 holds up.

## Reading a score: lift is the fold change, the CI is the error bar

| In the lab | Here |
|---|---|
| Fold change vs control | **Lift**: PR-AUC divided by chance (arm 1: 0.730 / 0.036 = 20x) |
| Error bars / 95% CI | **Bootstrap CI**, resampling patients (`compare_arms.py`) |
| p < 0.05 | Whether the CI excludes chance, or another arm |

Arm 1 beating chance is a formality. The comparisons where significance matters are between
arms, and since every arm is scored on the same 820 notes, the right test pairs them, like a
paired t-test rather than two separate error bars. The bootstrap does this by patient: each
resample draws patients, and every arm is scored on that same draw.

## Train vs validation: the train score is a health check, not a result

The train score is a student graded on the exact questions they practised. It says nothing
about new notes. What it is good for is the **gap**:

- train near-perfect, validation much lower: the model memorised its training notes (worry)
- train a little higher (arm 1: 0.822 vs 0.730): normal

The validation split is drawn by patient, so no validation patient's notes were seen in
training.

## Why a perfect score is not reachable here

The labels are diagnosis codes, and the premise of the project is that codes miss real
dementia. So some of the model's "errors" are the cases being looked for:

- a note that clearly describes dementia but has no code counts as a false alarm, and may be
  an under-coded patient
- a coded admission whose history sections never mention it counts as a miss, though there was
  nothing in the text to find

The ceiling sits below 1.0 by an unknown amount, which is why arms are compared with each
other on the same notes and the same imperfect labels.

## Arm 1 vs arm 2: same scorer, smarter reading

Both arms end in a logistic regression. What differs is what it reads.

| | Arm 1 | Arm 2 |
|---|---|---|
| What the note becomes | 200k word and word-pair counts (TF-IDF) | 768 numbers from BERT, read in context |
| What turns that into a score | Logistic regression | Logistic regression (the "head") |

In arm 1, "dementia" adds the same weight in "history of dementia" and "dementia was ruled
out". BERT represents the word differently in each.

## Bidirectional vs left-to-right models

BERT reads in both directions at once: every word sees every other word, before and after it.
In "dementia was considered but ruled out", the meaning arrives after the key word, and BERT
sees it. Text generators (GPT, Claude) read left to right because they write one word at a
time and the words to the right do not exist yet. Each design fits its job: reading fixed text
to classify it, or producing new text.

LoRA does not change this. Bidirectionality is the shape of the model, not something stored in
the weights LoRA freezes or trains. The 512-token limit is a separate, 2018-era design choice.

## Fine-tuning, and what LoRA freezes: the photo-editing picture

**Fine-tuning** adjusts a pretrained model's weights with labelled examples, in a four-move
loop: predict, measure the error (loss), work out which way to nudge each weight (gradient),
nudge a little (learning rate). Full fine-tuning nudges all 110 million weights.

**LoRA** is non-destructive editing. The original photo (BERT's weights) is never painted on.
Instead, adjustment layers are stacked on top:

| Photo editing | LoRA |
|---|---|
| The original photo, never painted on | **W**, BERT's frozen weights |
| One adjustment ("warm the colours") | **A**, the pattern being added |
| That layer's mask (strength per region) | **B**, how strongly each output takes the pattern |
| Number of adjustment layers | **Rank r** |
| Opacity of the whole stack | **alpha / r** |

So: output = W (frozen) + (alpha / r) x B x A. B starts at zero, so every mask starts blank and
the model begins as exactly the pretrained BERT. Only A, B and the new head are trained, about
295,000 numbers out of 110 million (about 0.3%).

Why so few are enough: a 768 x 768 block holds 589,824 numbers, but B x A at rank 8 is built
from 12,288 and can only push in 8 independent directions. The bet, tested in the LoRA paper,
is that steering a model that already knows clinical language needs a few broad corrections,
not a repaint.

On this hardware, full fine-tuning would fit in memory. LoRA's benefits here are less
overfitting with about 7,900 training positives, faster training and a small saved adapter, and
it is the method that scales to models too large to fine-tune in full.

## Choosing rank and alpha

- **Rank**: more layers allow a richer edit but also more memorising. Ranks of 4 to 16 are
  usual for a model this size, and small ranks often do as well as large ones.
- **Alpha**: the add-on is scaled by alpha / r, so adding layers keeps the overall edit about
  the same strength. A higher alpha acts much like a higher learning rate for the add-on.

Decided 2026-11-07, before any LoRA result:

| | Main run | Pre-declared fallback |
|---|---|---|
| Rank | 8 | 12 |
| Alpha | 16 | 24 (alpha = 2r, same opacity per layer) |
| Targets | query, value | query, value |
| Dropout | 0.1 | 0.1 |
| Runs if | always | main validation PR-AUC < 0.76 (arm 1 + about 0.03) |

The fallback changes one variable, rank, by a modest step. Settings may be chosen on
validation as long as the final number comes from the untouched test set and the arms are
treated alike.

## Why Bio_ClinicalBERT

Standard BERT (110M weights, 12 layers) given extra pretraining on biomedical literature and
then about 2 million MIMIC-III clinical notes (Alsentzer et al., 2019). It reads clinical
shorthand ("h/o", "s/p", "MMSE 18/30"), reads in both directions, fits the hardware and is
widely used, so the choice is easy to defend. Like mm9 after mm10 arrived, it is old, but its
quirks are well known.

Known weaknesses: the 512-token limit (about a third of notes are truncated), its age (newer
long-context clinical encoders exist), and a general-English vocabulary that splits drug names
into fragments.

**Overlap with this dataset.** MIMIC-III and MIMIC-IV come from the same hospital and overlap
in 2008-2012, so the model has likely read notes from some train and validation patients during
pretraining. That is exposure to text, not labels: pretraining never saw a dementia code. It may
flatter validation slightly; the test set (2017-2019) falls outside the overlap.

## How this LoRA evaluation compares with common practice

| Common practice | Here |
|---|---|
| Settings fixed from convention, or tuned on validation, then the final number reported once on an untouched test set | Same. Settings fixed before any result; test scored once, at the very end |
| 2 to 4 passes (epochs) for BERT-style fine-tuning | 2 passes over all 202,905 train notes |
| A strong simple baseline alongside the fine-tune | Arm 1, TF-IDF + logistic regression |
| Sweeps over rank and target modules (for example rank 4, 8, 16) | Narrower by design: one pre-declared fallback, rank 12, run only if the main run scores below 0.76 |
| Results averaged over several random seeds (often 3 to 5), because fine-tuning outcomes vary with the seed | **One seed per configuration.** A known gap: the bootstrap measures how much the score moves with a different sample of patients, not how much it moves with a different training run. A second seed of the main run is a candidate for an optional slot |
| Paired comparison of models on the same evaluation set | Paired bootstrap over patients: every arm minus LoRA main, with a 95% CI |
| Long clinical notes: truncation compared against chunking or other input choices | Truncation at 512 tokens. About 41% of notes are longer, and the section order (HPI, then PMH) means PMH is cut first. One follow-up, first 128 + last 382 tokens, scored the same as the main run (-0.007 [-0.029, +0.012]), so chunking was not run |

## What ran on the weekend of 2026-11-07

Each LoRA run is about 6 hours of training (two passes) plus about 20 minutes of scoring. The
GPU holds one run at a time (about 15 GB). The plan was written before any run; this is what
happened.

| Run | Rule, written in advance | Outcome |
|---|---|---|
| Main: rank 8, alpha 16, two passes | Always | 0.768 on validation, 6.15 h |
| Fallback: rank 12, alpha 24 | Only if main scored below 0.76 | Not run: 0.768 cleared the bar |
| Optional slot | Chosen 2026-11-08 from a truncation check of arm 0, after the main score and bootstrap were seen | Head + tail (first 128 + last 382 tokens): 0.761, the same as main |
| Chunking | Only if head + tail beat main with the interval above zero | Not run |

The other candidates for the optional slot (core vs broad label, a second seed of the main run)
were not used.

## Arm 3: which two LLMs, and why

Chosen 2026-11-08, before any arm 3 score. Both run in llama.cpp on this Mac, at 4-bit (Q4_K_M),
so no note leaves the machine.

| | MedGemma 1.5 4B | Qwen3.5-9B |
|---|---|---|
| Maker | Google | Alibaba |
| Kind | Medically trained, small | General, larger |
| Why | A purpose-built medical model that fits easily | The common choice for the best model a 32 GB Mac runs well |
| Time for all of validation (from 200 notes) | About 4.9 h | About 10.8 h |

Together they ask a question of their own: does medical training beat a bigger general model on
these notes?

**How a note is scored.** The model sees an instruction, four real train notes with their answers
(drawn by a fixed rule: two coded, two uncoded with no explicit mention), then the note, and is
asked whether it documents dementia. Nothing is generated: the score is P(Yes) / (P(Yes) + P(No))
for the first answer token, so every note gets a number to rank on, like the other arms.

**Published context, found after the choice.** A UCLA preprint (Lee et al., "MedJev: Evaluating
Zero-Shot System-One Decision Making in Medicine", arXiv 2610.36006, October 2026, not yet peer
reviewed) builds its model on Qwen3.5-9B and uses both models as baselines. It reads binary
answers the same way, from the first answer token, and runs locally on Apple silicon with
MIMIC-IV. On medical exam questions Qwen3.5-9B scored above MedGemma 1.5 4B on every benchmark
they report (MedQA 68.6% vs 47.7%). The match is reassuring, not proof:

- Their tasks are exam questions and structured-EHR prediction (mortality, readmission), not
  reading notes for dementia; MedGemma was not run on their MIMIC tasks.
- They ran 8-bit models in MLX, not 4-bit in llama.cpp, so numbers do not carry across.
- Their own model was trained on MIMIC-IV-Note, which very likely includes this project's test
  notes, so it could never be an arm here.

## Arm 0: the regex baseline, designed to be a serious competitor

PLAN §6 frames the project's hypothesis as LoRA vs regex: a regex catches explicit mentions,
and the fine-tuned model has to earn its place on implicit language ("poor historian",
"oriented x1", "per daughter, increasingly forgetful"). So the regex is built to be as strong as
an explicit-mention rule can honestly be. Every rule below was fixed on 2026-11-07, before any
note was read, and nothing is tuned on validation.

**Terms mirror the `dementia_core` ICD codes.** Diagnosis names: dementia/demented,
Alzheimer('s), Lewy body/bodies, Pick's disease, major neurocognitive disorder. Left out on
purpose: abbreviations (AD can mean Alzheimer's, "as directed", atopic dermatitis or advance
directive), mild cognitive impairment (the broad label), delirium (the contrast condition), and
implicit language (the territory LoRA has to win).

**Two pre-declared variants.** *Arm 0* uses diagnosis terms only. *Arm 0+drugs* adds the four
dementia drugs (donepezil, rivastigmine, galantamine, memantine, with brand names). The study
uses prescriptions of those drugs as a validator, so arm 0 is the version used at the validation
stage: a regex that finds candidates by drug name and is then "validated" by drug prescriptions
would be partly checking itself.

**Context, NegEx/ConText style.** Each mention is classed by trigger words within the same
clause and at most 6 words away. Sentence ends, "?" followed by a space, and contrast words
(but, however, although, though, except, which) end a clause.

| Class | Example | Counted as |
|---|---|---|
| Negated | "No history of dementia", "dementia was ruled out" | 0 |
| Someone else | "Mother had dementia" (but "per daughter, patient has dementia" is the patient) | 0 |
| Uncertain | "Concern for dementia", "?dementia", "cannot exclude dementia" | 0.5 |
| Certain | Everything else | 1 |

Standard pseudo-negations ("no change in her dementia", "not ruled out") do not negate.

**Score** = sum of counted mentions, so a note naming dementia in both HPI and PMH ranks above
one naming it once. A yes/no score would give PR-AUC a single point to rank on and undersell the
regex. The 0.5 weight is a judgement call made in advance. Also reported: precision and recall of
the simplest rule, "flag any note with a score above 0".

**Text.** The same baseline sections as the other arms, read in full. A third pre-declared run,
*arm 0 first 512*, reads only the first 512 tokens: exactly arm 2's window. The gap between
the full and first-512 runs measures how much explicit signal truncation cuts off.

**Tests.** 34 invented sentences with known answers pin down every rule above
(`tests/test_arm0_regex.py`). Rule-based negation is good but not perfect (published NegEx
precision is in the 80-90% range); its known weaknesses are part of what the comparison shows.

## Validator 1: dementia drug orders, fixed before any arm is scored on it

Every arm is scored against ICD codes, and the project is about codes that are missing. So a
second yardstick is needed, one the arms never read and the codes do not define. The four
dementia drugs (donepezil, rivastigmine, galantamine, memantine, with brand names) are close to
dementia-specific and sit in `prescriptions`. Every rule below was fixed on 2026-11-08 from counts
of labels, ages and drug orders on validation. No arm's scores had been looked at.

**The label.** Among uncoded admissions (`dementia_core` = 0), positive = at least one order for a
dementia drug during the same admission (`hadm_id`). A dementia code on a later admission is
validator 2's job, kept separate. Drug names are matched as text in `prescriptions.drug`; on
2026-11-08 every matching string was checked by eye and all were genuine (generic, brand, patch,
strength, one typo).

**The score.** Each arm's PR-AUC on this label, with the same patient-level paired bootstrap as the
code label, plus a curve of the drug rate among the top *k* uncoded notes for every *k*. No
cutoff and no bins, so nothing is chosen after the results are seen.

**Who is in the pool.**

| Rule | Why | Validation count |
|---|---|---|
| Uncoded admissions only | The question is whether arms find dementia the codes missed | 21,872 notes; 200 with an order (0.9%) |
| MCI-coded admissions dropped (main); kept (sensitivity) | Every arm defines MCI as not dementia; donepezil is used off-label for MCI | 48 MCI-coded, 7 with an order |
| Main pool | | **21,824 notes; 193 positives, 120 patients; chance 0.009** |

For scale: 288 of the 820 coded notes (35.1%) had an order. About a third of coded dementia is
treated in hospital, so even a perfect arm will not see most real dementia through this label.

**Rows of the validator table.**

1. **Main:** the pool above.
2. **Hidden:** positives are the 92 drug-positive notes (58 patients) whose history text has no
   explicit dementia mention by arm 0's frozen rules. The 101 explicit drug-positive notes are
   left out, so mentioning the word earns nothing. Arm 0 sits at chance here by construction. This
   row is the implicit-language test of PLAN §6. "Hidden" means absent from the history sections
   the arms read, not from the whole record.
3. **Clean:** the main row without the 11 notes that name a dementia drug in their text, so no
   arm is credited for reading the drug name.
4. **Sensitivity:** MCI-coded admissions kept in.

**Arms.** Arm 0 (all three variants; 0+drugs marked "partly checks itself"), arm 1, arm 1b, both
LoRA runs, arm 3 if it scores all of validation, and an **age-only arm** (score = age at
admission). Age-only is the reference: drug orders rise steeply with age (54% of the pool is
under 65 and almost none of them have an order), so an arm must beat age, not chance, before its
validator score means anything. Age-only is also added to the code-label table as a floor.

**Age bands, description only.** Under 85 and 85+, chosen from the age distribution of the
positives (tertile cuts 82 and 87; 85 is close to their median), giving about 66 and 62
patients. A band, or any other split, is reported only with at least 30 positive patients; the
rule was set before the counts were seen. The same bands carry to test unchanged.

**Delirium.** 13 positives (11 patients) are delirium-coded: below the rule, so reported as a
count, not a split.

**Validation, then test once.** Everything is built and run on validation and reported as
development. Nothing changes after those results. Test is scored once, with every rule above
unchanged, alongside the code-label results. Validator numbers are compared between arms within a
split, never across splits: prescribing changed between 2008-2013 and 2017-2019, and MIMIC's
shifted dates allow only approximate matching to real-world drug history.

**How to read it, written before any result.**

- A high score is evidence that an arm finds real, uncoded dementia. A low score is weak evidence
  against: the label sees treated dementia only.
- The main row favours explicit mentions (a treated patient usually has a named diagnosis), so a
  strong arm 0 there is expected, not a win for regex. The hidden row is the test that matters
  for LoRA and arm 3.
- The trained arms (1, 1b, LoRA) were taught that 1,578 treated, uncoded train notes are
  negatives (0.81% of uncoded train notes, against 7,905 coded positives). The validator is
  tilted against them; arms 0 and 3 never saw a label.
- Advanced dementia is often untreated (no orders at all in the 95+ group), and quiet patients
  are documented less than agitated ones. Neither is visible here.
- Uncoded MCI treated with donepezil cannot be separated out without reading notes.
