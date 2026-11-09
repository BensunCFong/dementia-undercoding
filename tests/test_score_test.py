"""Scoring test once: the lock, scored-once, and the validation check, with fake scorers."""

import numpy as np
import pandas as pd
import pytest

from dementia_undercoding.evaluate import save_scores
from dementia_undercoding.score_test import check_unlocked, run_arm, scored_path


def notes():
    return pd.DataFrame({
        "hadm_id": [1, 2, 3, 4, 5, 6],
        "split": ["train", "train", "validation", "validation", "test", "test"],
        "text": ["a", "b", "c", "d", "e", "f"],
    })


def config(tmp_path, allowed=True):
    return {"paths": {"scores": str(tmp_path)},
            "test_scoring": {"allowed": allowed, "reproduce_tol": 1e-5}}


def fake_arm(val_scores):
    def scorer(dev, test, cfg):
        return {"fake": (val_scores, np.linspace(0, 1, len(test)))}
    return {"fake": (["fake"], scorer)}


def saved_validation(tmp_path):
    save_scores([3, 4], [0.2, 0.7], tmp_path / "fake-dementia_core.parquet")


def test_locked_by_default(tmp_path):
    with pytest.raises(PermissionError, match="locked"):
        check_unlocked(config(tmp_path, allowed=False))
    with pytest.raises(PermissionError):
        check_unlocked({})
    with pytest.raises(PermissionError):
        run_arm("fake", config(tmp_path, allowed=False), notes(), fake_arm(None))


def test_writes_test_scores_once(tmp_path):
    cfg = config(tmp_path)
    saved_validation(tmp_path)
    [path] = run_arm("fake", cfg, notes(), fake_arm(np.array([0.2, 0.7])))
    assert path == scored_path(cfg, "fake")
    out = pd.read_parquet(path)
    assert list(out.columns) == ["hadm_id", "score"] and list(out["hadm_id"]) == [5, 6]
    with pytest.raises(FileExistsError, match="scored once"):
        run_arm("fake", cfg, notes(), fake_arm(np.array([0.2, 0.7])))
    assert pd.read_parquet(path).equals(out)


def test_not_the_validated_model_writes_nothing(tmp_path):
    cfg = config(tmp_path)
    saved_validation(tmp_path)
    with pytest.raises(ValueError, match="not the validated model"):
        run_arm("fake", cfg, notes(), fake_arm(np.array([0.2, 0.71])))
    assert not scored_path(cfg, "fake").exists()


def test_validation_notes_must_match(tmp_path):
    cfg = config(tmp_path)
    save_scores([3, 99], [0.2, 0.7], tmp_path / "fake-dementia_core.parquet")
    with pytest.raises(ValueError, match="differ from the saved"):
        run_arm("fake", cfg, notes(), fake_arm(np.array([0.2, 0.7])))


def test_no_fitting_means_no_check(tmp_path):
    cfg = config(tmp_path)                        # no saved validation file at all
    [path] = run_arm("fake", cfg, notes(), fake_arm(None))
    assert path.exists()