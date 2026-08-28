from dementia_undercoding.text import clean


def test_deid_placeholders_removed():
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
