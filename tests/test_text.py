from dementia_undercoding.text import clean


def test_deid_placeholders_removed():
    # MIMIC-IV-Note: every placeholder is three underscores.
    assert clean("Name: ___ Unit No: ___") == "Name: Unit No:"
    assert clean("Admitted ___ with confusion.") == "Admitted with confusion."


def test_template_rules_removed():
    assert clean("Discharge Instructions:\n" + "_" * 57 + "\nTake meds.") == (
        "Discharge Instructions:\n\nTake meds."
    )


def test_mimic3_placeholders_still_removed():
    out = clean("Patient [**Known lastname 123**] admitted with confusion.")
    assert "[**" not in out
    assert "confusion" in out


def test_whitespace_collapsed():
    assert clean("a     b\n\n\n\n\nc") == "a b\n\nc"


def test_clinical_content_preserved():
    # Section headers carry signal for this task. They stay.
    src = "HISTORY OF PRESENT ILLNESS:\nOriented times one, poor historian."
    out = clean(src)
    assert "HISTORY OF PRESENT ILLNESS" in out
    assert "poor historian" in out
