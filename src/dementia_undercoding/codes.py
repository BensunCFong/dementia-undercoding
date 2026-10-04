"""ICD code normalisation and prefix matching for weak labelling.

The single most dangerous bug in this project lives here. MIMIC stores ICD
codes without decimal points, sometimes space-padded, and mixes ICD-9 and
ICD-10 in the same column with an `icd_version` flag. Matching on "331.0"
silently labels nothing, the model trains on an all-negative target, and
everything downstream looks broken for reasons that are hard to trace.

So: normalise first, match by prefix, and test it.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def normalise(code: str | None) -> str:
    """Strip whitespace, drop dots, uppercase.

    '331.0 ' -> '3310'; ' g30.9' -> 'G309'; None -> ''.
    """
    if code is None:
        return ""
    return str(code).strip().replace(".", "").replace(" ", "").upper()


def load_code_sets(path: str | Path = "config/codes.yaml") -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def prefixes_for(code_sets: dict, name: str, broad: bool = False) -> dict[int, tuple[str, ...]]:
    """Return {9: (...), 10: (...)} prefixes for a named code set.

    `broad=True` folds dementia_broad_extra into the dementia_core set.
    """
    base = code_sets[name]
    icd9 = list(base.get("icd9", []))
    icd10 = list(base.get("icd10", []))
    if broad and name == "dementia_core":
        extra = code_sets.get("dementia_broad_extra", {})
        icd9 += list(extra.get("icd9", []))
        icd10 += list(extra.get("icd10", []))
    return {9: tuple(icd9), 10: tuple(icd10)}


def prefix_rows(code_sets: dict) -> list[tuple[str, int, str]]:
    """(label, icd_version, prefix) for every label column the cohort carries.

    Loaded into DuckDB as a lookup table, so the prefix rule lives here only.
    """
    label_sets = {
        "dementia_core": prefixes_for(code_sets, "dementia_core"),
        "dementia_broad": prefixes_for(code_sets, "dementia_core", broad=True),
        "delirium": prefixes_for(code_sets, "delirium"),
    }
    rows = []
    for label, by_version in label_sets.items():
        for version, prefixes in by_version.items():
            for prefix in prefixes:
                rows.append((label, version, normalise(prefix)))
    return rows


def matches(code: str | None, version: int | str | None, prefixes: dict[int, tuple[str, ...]]) -> bool:
    """True if a single (code, icd_version) row falls in the code set."""
    try:
        ver = int(version)
    except (TypeError, ValueError):
        return False
    if ver not in prefixes:
        return False
    norm = normalise(code)
    if not norm:
        return False
    return norm.startswith(prefixes[ver])
    