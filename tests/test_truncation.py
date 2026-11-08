import re

from dementia_undercoding.text import BASELINE_SECTIONS, baseline_text
from dementia_undercoding.truncation import head_tail, pmh_first, reorder

# Invented note in the MIMIC-IV discharge template. Not real patient text.
NOTE = """History of Present Illness:
Per daughter, increasingly forgetful at home.

Past Medical History:
Dementia, hypertension

Social History:
Lives with daughter

Brief Hospital Course:
Disoriented overnight.
"""


class WordTokenizer:
    """Stand-in for a Hugging Face tokenizer: one token per word, with character offsets."""

    def __call__(self, texts, add_special_tokens, return_offsets_mapping, truncation):
        return {"offset_mapping": [[(m.start(), m.end()) for m in re.finditer(r"\S+", t)] for t in texts]}


def test_round_trip_keeps_baseline_text():
    text = baseline_text(NOTE)
    assert reorder(text, BASELINE_SECTIONS) == text


def test_pmh_first_moves_pmh_to_the_front():
    text = pmh_first(baseline_text(NOTE))
    assert text.startswith("Past Medical History:\nDementia, hypertension")
    assert text.index("History of Present Illness") < text.index("Social History")


def test_pmh_first_without_pmh_keeps_the_rest():
    text = baseline_text(NOTE.replace("Past Medical History:\nDementia, hypertension\n\n", ""))
    assert pmh_first(text) == text


def test_pmh_first_leaves_text_without_headings_alone():
    assert pmh_first("no headings here") == "no headings here"


def test_head_tail_keeps_short_texts_whole():
    assert head_tail(["one two three"], WordTokenizer(), head=2, tail=2) == ["one two three"]


def test_head_tail_cuts_the_middle():
    out = head_tail(["a b c d e f g"], WordTokenizer(), head=2, tail=3)
    assert out == ["a b\ne f g"]
