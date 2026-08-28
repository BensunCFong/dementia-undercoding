"""Tests for the thing most likely to silently break the whole project."""

import pytest

from dementia_undercoding.codes import (
    load_code_sets,
    matches,
    normalise,
    prefixes_for,
    sql_predicate,
)


@pytest.fixture(scope="module")
def sets():
    return load_code_sets("config/codes.yaml")


@pytest.fixture(scope="module")
def core(sets):
    return prefixes_for(sets, "dementia_core")


@pytest.fixture(scope="module")
def broad(sets):
    return prefixes_for(sets, "dementia_core", broad=True)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("331.0", "3310"),
        ("  3310  ", "3310"),
        ("g30 9", "G309"),
        ("G30.9", "G309"),
        (None, ""),
        ("", ""),
    ],
)
def test_normalise(raw, expected):
    assert normalise(raw) == expected


def test_dotted_codes_still_match(core):
    # The bug this whole module exists to prevent.
    assert matches("331.0", 9, core)
    assert matches("G30.9", 10, core)


def test_space_padded_codes_match(core):
    assert matches(" 3310 ", 9, core)
    assert matches("G309  ", 10, core)


def test_version_is_respected(core):
    # F03 is ICD-10 dementia. As an ICD-9 code it is a bacterial infection
    # code that does not exist in this form; either way it must not match.
    assert matches("F03", 10, core)
    assert not matches("F03", 9, core)


def test_true_positives(core):
    for code, ver in [("29040", 9), ("2941", 9), ("3310", 9), ("33182", 9),
                      ("F0150", 10), ("F028", 10), ("F039", 10), ("G3009", 10),
                      ("G3109", 10), ("G3183", 10)]:
        assert matches(code, ver, core), f"{code} (ICD-{ver}) should match"


def test_true_negatives(core):
    # Delirium, depression, stroke, Parkinson's: none are dementia.
    for code, ver in [("2930", 9), ("311", 9), ("43491", 9), ("3320", 9),
                      ("F050", 10), ("F329", 10), ("I635", 10), ("G20", 10)]:
        assert not matches(code, ver, core), f"{code} (ICD-{ver}) should not match"


def test_mci_only_in_broad(core, broad):
    # The sensitivity analysis depends on this boundary being exact.
    assert not matches("33183", 9, core)
    assert not matches("G3184", 10, core)
    assert matches("33183", 9, broad)
    assert matches("G3184", 10, broad)


def test_prefix_does_not_overreach(core):
    # "290" must not swallow 2900-adjacent codes from other chapters,
    # and "G30" must not match G300-lookalikes that are really G3-something.
    assert not matches("2809", 9, core)
    assert not matches("G40", 10, core)


def test_sql_predicate_shape(core):
    pred = sql_predicate(core)
    assert "icd_version = 9" in pred
    assert "icd_version = 10" in pred
    assert "'3310%'" in pred
    assert "TRIM" in pred and "REPLACE" in pred
