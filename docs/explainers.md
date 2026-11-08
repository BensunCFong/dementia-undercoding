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
| Error bars / 95% CI | **Bootstrap CI** (still to build) |
| p < 0.05 | Whether the CI excludes chance, or another arm |

Arm 1 beating chance is a formality. The comparison where significance matters is LoRA vs
arm 1, and since both are scored on the same 820 notes, the right test pairs them note by note,
like a paired t-test rather than two separate error bars.

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
