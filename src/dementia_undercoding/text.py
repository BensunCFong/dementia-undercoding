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
