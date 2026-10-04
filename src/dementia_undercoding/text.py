"""Discharge summary preprocessing.

Deliberately light. De-identification placeholders carry no signal, so they
go. Everything else stays: header structure and section names are
informative for this task and stripping them is a decision that should be
measured, not assumed.

MIMIC-IV-Note writes every placeholder as three underscores, "___". Every one
of the 331,793 discharge notes contains them. The MIMIC-III style,
[** Name **], appears in only 8 notes and is kept as a fallback. Longer
underscore runs (50-60 long) are horizontal rules in the note template and
go with them.
"""

from __future__ import annotations

import re

DEID = re.compile(r"_{3,}|\[\*\*.*?\*\*\]")
WS = re.compile(r"[ \t]+")
LINE_EDGE = re.compile(r" *\n *")
BLANK = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    text = DEID.sub(" ", text)
    text = WS.sub(" ", text)
    text = LINE_EDGE.sub("\n", text)
    text = BLANK.sub("\n\n", text)
    return text.strip()


def chunk(text: str, tokenizer, max_len: int = 512, stride: int = 128) -> list[str]:
    """Overlapping windows for encoders with a 512-token limit.

    Discharge summaries run well past 512 tokens, so the model sees windows
    and predictions are pooled at the document level. The stride matters:
    a mention split across a boundary is a mention lost.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    step = max_len - stride
    return [
        tokenizer.decode(ids[i:i + max_len])
        for i in range(0, max(len(ids), 1), step)
    ]


# Top-level headings of the MIMIC-IV discharge template. Each appears at the
# start of a line in 88-100% of the 331,793 notes (counted 2026-10-03). Matched
# case-sensitively and only from this list, so sub-headings inside a section
# ("PMH:", "In the ED, initial vitals were:", "HEENT:") do not end it.
TEMPLATE_HEADINGS = (
    "Chief Complaint",
    "Major Surgical or Invasive Procedure",
    "History of Present Illness",
    "Past Medical History",
    "Social History",
    "Family History",
    "Physical Exam",
    "Pertinent Results",
    "Brief Hospital Course",
    "Medications on Admission",
    "Discharge Medications",
    "Discharge Disposition",
    "Discharge Diagnosis",
    "Discharge Condition",
    "Discharge Instructions",
    "Followup Instructions",
)
HEADING = re.compile(r"^(" + "|".join(map(re.escape, TEMPLATE_HEADINGS)) + r"):", re.MULTILINE)

# The pre-registered delirium rule: the models read only the sections that
# describe the patient before this admission. Brief Hospital Course and the
# discharge sections, where acute confusion is documented, are left out.
# Social History is de-identified down to a bare placeholder in 97% of notes,
# so in practice the models mostly read HPI and Past Medical History.
BASELINE_SECTIONS = ("History of Present Illness", "Past Medical History", "Social History")


def sections(text: str) -> dict[str, str]:
    """Split a cleaned note into template sections: {heading: body}.

    A section runs from its heading to the next template heading. If a heading
    appears twice, the first one wins.
    """
    found: dict[str, str] = {}
    matches = list(HEADING.finditer(text))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        found.setdefault(match.group(1), text[match.end():end].strip())
    return found


def baseline_text(text: str) -> str:
    """Only the baseline sections, headings kept. Empty string if none were found."""
    parts = sections(text)
    return "\n\n".join(f"{name}:\n{parts[name]}" for name in BASELINE_SECTIONS if parts.get(name))