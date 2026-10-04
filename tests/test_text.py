from dementia_undercoding.text import baseline_text, clean, sections

# Invented note in the MIMIC-IV discharge template. Not real patient text.
NOTE = """Chief Complaint:
confusion

History of Present Illness:
Per daughter, increasingly forgetful at home.
In the ED, initial vitals were: stable.

Past Medical History:
Hypertension

Social History:
___

Family History:
Noncontributory

Brief Hospital Course:
Disoriented overnight, improved by day 3.
"""


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


def test_sections_split_on_template_headings():
    parts = sections(clean(NOTE))
    assert parts["Past Medical History"] == "Hypertension"
    # A sub-heading inside a section is not a template heading, so it does not end it.
    assert "In the ED, initial vitals were: stable." in parts["History of Present Illness"]


def test_baseline_keeps_only_baseline_sections():
    out = baseline_text(clean(NOTE))
    assert out.startswith("History of Present Illness:")
    assert "forgetful" in out
    assert "Hypertension" in out
    # Hospital course is where acute confusion lives: the delirium rule keeps it out.
    assert "Disoriented" not in out
    assert "Noncontributory" not in out
    # Social History was only a placeholder, so it is empty after cleaning and dropped.
    assert "Social History" not in out


def test_no_baseline_sections_gives_empty_string():
    assert baseline_text("Brief Hospital Course:\nStable.") == ""