import numpy as np
import pandas as pd
import pytest

import config
from src import data_utils as du
from src import feature_extractor as fe
from tests.synthetic import make_hand


def synthetic_rows(n_per_class=40, users=("u1", "u2"), sessions_per_user=1, seed=0):
    """SYNTHETIC rows (unit tests only)."""
    rng = np.random.default_rng(seed)
    rows, t = [], 0.0
    for u in users:
        for s in range(sessions_per_user):
            for cls in config.GESTURE_CLASSES:
                for _ in range(n_per_class):
                    t += 0.03
                    r = du.make_row(fe.extract_features(make_hand(cls, rng)), cls, u, f"{u}_s{s}", "Right")
                    r["timestamp"] = t
                    rows.append(r)
    return rows


def test_append_load_roundtrip(tmp_path):
    p = tmp_path / "d.csv"
    du.append_samples(synthetic_rows(5), p)
    du.append_samples(synthetic_rows(5, seed=1), p)
    df = du.load_dataset(p)
    assert len(df) == 2 * 2 * 5 * 6 and list(df.columns) == du.DATA_COLUMNS
    assert du.class_counts(p)["FIST"] == 20


def test_missing_dataset_message(tmp_path):
    with pytest.raises(du.DatasetError, match="collect_data"):
        du.load_dataset(tmp_path / "nope.csv")
    assert du.class_counts(tmp_path / "nope.csv") == {}


def test_schema_mismatch_rejected(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(du.DatasetError, match="schema"):
        du.append_samples(synthetic_rows(1), p)


def test_validation_flags_missing_class_and_low_counts():
    df = pd.DataFrame(synthetic_rows(40))
    rep = du.validate_dataset(df[df.label != "FIST"], min_per_class=30)
    assert any("FIST" in e for e in rep["errors"])
    rep = du.validate_dataset(df.groupby("label").head(5), min_per_class=30)
    assert any("Fewer than" in e for e in rep["errors"])


def test_validation_drops_nan_and_duplicates():
    df = pd.DataFrame(synthetic_rows(40))
    df.loc[0, fe.FEATURE_NAMES[0]] = np.nan
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    rep = du.validate_dataset(df, min_per_class=30)
    assert rep["info"]["dropped_invalid"] == 1 and rep["info"]["dropped_duplicates"] == 1
    assert not rep["errors"]


def test_split_by_user_has_no_group_leakage():
    df = pd.DataFrame(synthetic_rows(40, users=("a", "b", "c")))
    sp = du.split_dataset(df, seed=1)
    assert sp["strategy"] == "group_by_user"
    tr_users, te_users = set(df.iloc[sp["train_idx"]].user_id), set(df.iloc[sp["test_idx"]].user_id)
    assert not tr_users & te_users
    assert set(df.iloc[sp["test_idx"]].label) == set(config.GESTURE_CLASSES)


def test_split_by_session_for_single_user():
    df = pd.DataFrame(synthetic_rows(40, users=("a",), sessions_per_user=3))
    sp = du.split_dataset(df)
    assert sp["strategy"] == "group_by_session" and sp["preliminary"]


def test_blocked_time_split_for_single_session_has_gap_and_no_overlap():
    df = pd.DataFrame(synthetic_rows(40, users=("a",)))
    sp = du.split_dataset(df, gap=5)
    assert sp["strategy"] == "blocked_time" and sp["preliminary"]
    assert not set(sp["train_idx"]) & set(sp["test_idx"])
    for cls in config.GESTURE_CLASSES:
        tr = df.iloc[sp["train_idx"]]; te = df.iloc[sp["test_idx"]]
        assert tr[tr.label == cls].timestamp.max() < te[te.label == cls].timestamp.min()


def test_fingerprint_changes_with_data():
    a = pd.DataFrame(synthetic_rows(5))
    b = a.copy(); b.loc[0, fe.FEATURE_NAMES[0]] += 1
    assert du.dataset_fingerprint(a) != du.dataset_fingerprint(b)
    assert du.dataset_fingerprint(a) == du.dataset_fingerprint(a.copy())
