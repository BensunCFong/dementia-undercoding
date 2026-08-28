"""Discharge summary preprocessing.

Deliberately light. De-identification placeholders in MIMIC look like
[** Name **] and carry no signal, so they go. Everything else stays: header
structure and section names are informative for this task and stripping
them is a decision that should be measured, not assumed.
"""

from __future__ import annotations

import re

DEID = re.compile(r"\[\*\*.*?\*\*\]")
WS = re.compile(r"[ \t]+")
BLANK = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    text = DEID.sub(" ", text)
    text = WS.sub(" ", text)
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
