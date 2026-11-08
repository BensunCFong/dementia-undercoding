"""Arm 3's prompt and scoring, on invented notes. No model or server needed."""

import math

import numpy as np
import pandas as pd
import pytest

from dementia_undercoding.arm3_llm import QUESTION, build_messages, pick_examples, yes_probability


def fake_train(seed=0):
    """Invented train notes: some coded, some not, some long, one uncoded note that mentions dementia."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(40):
        coded = i % 4 == 0
        text = "Past Medical History:\n" + ("memory loss, on donepezil" if coded else "hypertension")
        rows.append({"hadm_id": 1000 + i, "text": text, "dementia_core": int(coded)})
    rows.append({"hadm_id": 2000, "text": "Past Medical History:\ndementia", "dementia_core": 0})
    rows.append({"hadm_id": 2001, "text": "x" * 5000, "dementia_core": 1})
    return pd.DataFrame(rows).sample(frac=1, random_state=int(rng.integers(100)))


def pick(df, **kw):
    args = {"n_pos": 2, "n_neg": 2, "max_chars": 1600, "seed": 1, "window": 6, "uncertain_weight": 0.5}
    return pick_examples(df, "dementia_core", **{**args, **kw})


def test_examples_follow_the_rule():
    ex = pick(fake_train())
    assert [positive for _, positive in ex] == [True, False, False, True]
    assert all(len(text) <= 1600 for text, _ in ex)                     # the long note is never drawn
    assert all("dementia" not in text for text, positive in ex if not positive)


def test_examples_do_not_depend_on_row_order():
    assert pick(fake_train(seed=0)) == pick(fake_train(seed=5))


def test_uncoded_note_with_a_dementia_mention_is_never_a_negative():
    df = fake_train()
    for seed in range(20):
        assert all("dementia" not in t for t, positive in pick(df, seed=seed) if not positive)


def test_messages_put_the_note_last():
    ex = [("note A", True), ("note B", False)]
    msgs = build_messages(ex, "the note")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[2]["content"] == "Yes" and msgs[4]["content"] == "No"
    assert msgs[-1]["content"] == f"Note:\nthe note\n\n{QUESTION}"


def test_prefix_is_the_same_for_every_note():
    ex = [("note A", True), ("note B", False)]
    assert build_messages(ex, "one")[:-1] == build_messages(ex, "two")[:-1]


def test_prefill_starts_the_answer():
    msgs = build_messages([("note A", True)], "the note", prefill="<think></think>")
    assert msgs[-1] == {"role": "assistant", "content": "<think></think>"}


def lp(p):
    return math.log(p)


def test_yes_probability_renormalises_over_yes_and_no():
    top = [{"token": "No", "logprob": lp(0.6)}, {"token": "", "logprob": lp(0.2)},
           {"token": "Yes", "logprob": lp(0.2)}]
    assert yes_probability(top) == pytest.approx(0.25)


def test_yes_probability_merges_spellings():
    top = [{"token": "Yes", "logprob": lp(0.3)}, {"token": " yes", "logprob": lp(0.1)},
           {"token": "No", "logprob": lp(0.4)}]
    assert yes_probability(top) == pytest.approx(0.5)


def test_yes_probability_is_nan_without_an_answer():
    assert math.isnan(yes_probability([{"token": "The", "logprob": lp(0.9)}]))