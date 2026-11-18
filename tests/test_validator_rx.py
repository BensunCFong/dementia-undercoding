"""The drug validator's rules, on invented drug strings, codes and sentences."""

import numpy as np
import pandas as pd
import pytest
import yaml

from dementia_undercoding.compare_arms import AGE_ONLY, ARMS
from dementia_undercoding.splits import load_notes
from dementia_undercoding.validator_rx import (
    arm3_decision,
    band_masks,
    code_sets,
    is_drug_order,
    names_drug,
    names_in_text,
    pick_arm3,
    positive_patients,
    row_masks,
    run,
    topk_curve,
)


def test_drug_names_match_every_form_and_nothing_close():
    assert names_drug("Donepezil 5mg ODT")
    assert names_drug("Exelon Patch")
    assert names_drug("*NF* Namenda XR")
    assert not names_drug("Pyridostigmine")
    assert not names_drug("Amantadine")
    assert not names_drug(None)


def test_codes_come_only_from_name_matched_orders():
    orders = pd.DataFrame({
        "drug": ["Donepezil", "Memantine", "Aspirin", None],
        "ndc": ["111", "0", "999", "555"],
        "gsn": ["21 22", None, "77", "88"],
    })
    ndcs, gsns = code_sets(orders)
    assert ndcs == {"111"}                       # "0" is no code; aspirin's code is not taken
    assert gsns == {"21", "22"}                  # a gsn field can hold several codes


def test_an_order_counts_by_name_or_by_a_shared_code():
    ndcs, gsns = {"111"}, {"21", "22"}
    assert is_drug_order("Aricept", None, None, ndcs, gsns)
    assert is_drug_order("gala", "111", None, ndcs, gsns)          # truncated name, same ndc
    assert is_drug_order("rivas", None, "40 22", ndcs, gsns)       # one of its gsn codes is shared
    assert not is_drug_order("Aspirin", "999", "77", ndcs, gsns)
    assert not is_drug_order(None, "0", "", ndcs, gsns)


def test_text_filter_finds_drug_names_and_classes():
    found = names_in_text([
        "Home meds include donepezil.",
        "Was on a cholinesterase inhibitor.",
        "Started an AChEI last year.",
        "NMDA receptor antagonist stopped.",
        "No aricept.",                               # negated still counts: the name is in the text
        "Pyridostigmine for myasthenia.",
        "Has a headache and chest pain.",
    ])
    assert found.tolist() == [True, True, True, True, True, False, False]


def test_rows_keep_the_right_notes():
    y =        [1, 1, 0, 0, 1, 0]
    mci =      [0, 0, 0, 0, 0, 1]
    names =    [0, 1, 1, 0, 0, 0]
    explicit = [1, 0, 1, 0, 0, 0]
    rows = row_masks(y, mci, names, explicit)
    assert rows["main"].tolist() == [True, True, True, True, True, False]
    # hidden drops the explicit positive only; an explicit negative stays
    assert rows["hidden"].tolist() == [False, True, True, True, True, False]
    # clean drops every note naming a drug, positive or negative
    assert rows["clean"].tolist() == [True, False, False, True, True, False]
    assert rows["sensitivity (MCI kept)"].all()


def test_bands_split_at_the_cut():
    bands = band_masks([60, 84.9, 85, 99], 85)
    assert bands["under 85"].tolist() == [True, True, False, False]
    assert bands["85+"].tolist() == [False, False, True, True]


def test_positive_patients_counts_patients_not_notes():
    assert positive_patients([1, 1, 1, 0], ["a", "a", "b", "c"]) == 2


def test_topk_curve_counts_tied_scores_evenly():
    curve = topk_curve([1, 0, 1, 0], [0.9, 0.5, 0.5, 0.1])
    # top 1 is the positive; the next two are tied, one positive between them
    assert curve["positives"].tolist() == pytest.approx([1.0, 1.5, 2.0, 2.0])
    assert curve["rate"].iloc[-1] == pytest.approx(0.5)


def test_topk_curve_with_no_ties_is_a_plain_count():
    curve = topk_curve([0, 1, 1], [0.1, 0.3, 0.2])
    assert curve["positives"].tolist() == [1, 2, 2]


def test_arm3_qualifies_by_either_route_only():
    hidden = pd.DataFrame({"arm": ["a", "b", "c"], "diff_low": [0.01, -0.02, -0.01]})
    ok = arm3_decision({"a": 0.60, "b": 0.74, "c": 0.729}, bar=0.73, hidden=hidden)
    assert ok == {"a": True, "b": True, "c": False}


def test_when_both_qualify_code_label_decides_unless_too_close():
    code = {"a": 0.70, "b": 0.66}
    hidden = {"a": 0.05, "b": 0.09}
    assert pick_arm3(code, (-0.06, -0.01), hidden) == "a"     # interval excludes 0: code label decides
    assert pick_arm3(code, (-0.06, +0.01), hidden) == "b"     # includes 0: the hidden row decides


def fake_project(tmp_path, n_patients=600, seed=0):
    """Invented notes, prescriptions and arm scores laid out as the config expects."""
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n_patients), 2)
    n = len(pid)
    age = np.repeat(rng.integers(50, 100, n_patients), 2).astype(float)
    core = (rng.random(n) < 0.1).astype(int)
    treated = (rng.random(n) < np.where(age >= 75, 0.25, 0.02)) & (core == 0)
    notes = pd.DataFrame({
        "hadm_id": np.arange(n) + 1000, "subject_id": pid, "split": "train", "age_at_admit": age,
        "dementia_core": core, "dementia_broad": core | (rng.random(n) < 0.03).astype(int),
        "delirium": (rng.random(n) < 0.1).astype(int),
        "text": np.where(rng.random(n) < 0.05, "Home meds: Aricept.", "Chest pain, admitted."),
    })
    notes.to_parquet(tmp_path / "notes.parquet", index=False)
    hosp = tmp_path / "hosp"
    hosp.mkdir()
    drugs = np.where(treated, rng.choice(["Donepezil 5mg", "gala"], n), "Aspirin")
    rx = pd.DataFrame({"hadm_id": notes["hadm_id"], "drug": drugs,
                       "ndc": np.where(drugs == "Aspirin", "999", "111"),
                       "gsn": np.where(drugs == "Aspirin", "77", "21 22")})
    other_split = pd.DataFrame({"hadm_id": [1], "drug": ["Galantamine"], "ndc": ["111"], "gsn": ["21"]})
    rx = pd.concat([rx, other_split])
    rx.iloc[: n // 2].to_csv(hosp / "prescriptions-000.csv.gz", index=False)
    rx.iloc[n // 2:].to_csv(hosp / "prescriptions-001.csv.gz", index=False)
    scores = tmp_path / "scores"
    scores.mkdir()
    cfg = {
        "paths": {"notes": str(tmp_path / "notes.parquet"), "scores": str(scores), "mimic_hosp": str(hosp)},
        "files": {"hosp": {"prescriptions": "prescriptions-*.csv.gz"}},
        "split": {"validation": {"frac": 0.5, "seed": 3}},
        "compare": {"n_boot": 20, "seed": 1, "ci": 0.95},
        "validator_rx": {"age_cut": 85, "min_patients": 2, "out_dir": str(tmp_path / "validator")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    val = load_notes(path, columns=[])
    val = val["split"].to_numpy() == "validation"            # score files hold validation notes only
    for stem in ARMS.values():
        s = treated + rng.normal(0, 1, n)
        pd.DataFrame({"hadm_id": notes["hadm_id"][val], "score": s[val]}).to_parquet(
            scores / f"{stem}-dementia_core.parquet", index=False)
    return path


def test_run_end_to_end_on_invented_data(tmp_path, capsys):
    table = run(fake_project(tmp_path))
    out = capsys.readouterr().out
    assert "by code only" in out
    assert {"main", "hidden", "clean", "sensitivity (MCI kept)"} <= set(table["row"])
    assert set(table["arm"]) == {*ARMS, AGE_ONLY}
    main = table[(table["row"] == "main") & (table["arm"] == AGE_ONLY)].iloc[0]
    assert main["diff"] == 0                       # age only is the reference
    bands = table[table["row"].str.startswith("main, ")]
    assert bands["diff"].isna().all()              # bands carry no differences
    curves = pd.read_parquet(tmp_path / "validator" / "validation-topk.parquet")
    assert curves.groupby(["row", "arm"])["k"].max().nunique() == 4      # one curve length per row
