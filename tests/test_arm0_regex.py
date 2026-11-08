"""Arm 0 on invented sentences with known answers. No real notes."""

import pytest

from dementia_undercoding.arm0_regex import (
    DIAGNOSIS,
    DRUGS,
    any_mention,
    count_mentions,
    mention_score,
    score_note,
)

WINDOW, WEIGHT = 6, 0.5


def kinds(text, pattern=DIAGNOSIS):
    """The non-zero mention counts for text, e.g. {'certain': 1}."""
    return {k: v for k, v in count_mentions(text, pattern, WINDOW).items() if v}


@pytest.mark.parametrize(
    "text, expected",
    [
        # plain mentions
        ("Dementia, HTN, CKD.", {"certain": 1}),
        ("History of Alzheimer's disease.", {"certain": 1}),
        ("Dementia with Lewy bodies.", {"certain": 2}),
        ("Frontotemporal dementia (Pick's disease).", {"certain": 2}),
        ("Major neurocognitive disorder.", {"certain": 1}),
        ("Patient is demented at baseline.", {"certain": 1}),
        # negation, before and after
        ("No dementia.", {"negated": 1}),
        ("No known history of Alzheimer's.", {"negated": 1}),
        ("Denies dementia.", {"negated": 1}),
        ("Dementia was ruled out.", {"negated": 1}),
        # scope: sentence ends, contrast words, the 6-word limit
        ("No fever or chills. Has dementia.", {"certain": 1}),
        ("No delirium, but has dementia.", {"certain": 1}),
        ("Any falls? Has dementia.", {"certain": 1}),
        ("Patient denies chest pain, shortness of breath, nausea, vomiting, "
         "and has a long history of dementia.", {"certain": 1}),
        # pseudo-negations that must not negate
        ("No change in her dementia.", {"certain": 1}),
        # someone else
        ("Mother had dementia.", {"other": 1}),
        ("Lives with her husband, who has dementia.", {"other": 1}),
        ("Family history of Alzheimer's.", {"other": 1}),
        # ...but a relative as informant is about the patient
        ("Per daughter, patient has dementia.", {"certain": 1}),
        ("Daughter reports she has worsening dementia.", {"certain": 1}),
        # uncertain
        ("Concern for dementia.", {"uncertain": 1}),
        ("?dementia", {"uncertain": 1}),
        ("r/o dementia", {"uncertain": 1}),
        ("Cannot exclude dementia.", {"uncertain": 1}),
        ("Dementia not ruled out.", {"uncertain": 1}),
        ("Possible early Alzheimer's.", {"uncertain": 1}),
        # nothing to find: implicit language and look-alikes are deliberately not terms
        ("Poor historian, oriented x1, forgetful per daughter.", {}),
        ("Mild cognitive impairment.", {}),
        ("Acute delirium.", {}),
        ("AD, HTN.", {}),
    ],
)
def test_classification(text, expected):
    assert kinds(text) == expected


def test_drugs_only_in_drug_pattern():
    assert kinds("On donepezil and memantine.", DRUGS) == {"certain": 2}
    assert kinds("Not on Aricept.", DRUGS) == {"negated": 1}
    assert kinds("On donepezil.") == {}                     # arm 0 itself ignores drugs


def test_score_weights_uncertain_by_half():
    counts = count_mentions("History of dementia. Concern for Alzheimer's. No dementia.", DIAGNOSIS, WINDOW)
    assert mention_score(counts, WEIGHT) == pytest.approx(1.5)


def test_score_note_splits_diagnosis_and_drugs():
    assert score_note("Alzheimer's dementia on donepezil.", WINDOW, WEIGHT) == (2.0, 1.0)


def test_any_mention():
    result = any_mention([1, 0, 1, 0], [2.0, 0.5, 0.0, 0.0])
    assert result == {"flagged_share": 0.5, "precision": 0.5, "recall": 0.5}