"""End-to-end plumbing tests on SYNTHETIC data: verifies code paths, NOT real-world accuracy."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

import config
import train_model
from src import data_utils as du
from src import feature_extractor as fe
from src.command_controller import CommandController
from src.gesture_classifier import GestureClassifier, ModelNotAvailableError, StabilityFilter
from src.hand_detector import HandResult
from src.pipeline import GesturePipeline
from tests.synthetic import make_hand
from tests.test_data_utils import synthetic_rows


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("train")
    data = d / "data.csv"
    du.append_samples(synthetic_rows(60, users=("a", "b", "c")), data)
    out = train_model.run(data, d / "m.joblib", d / "res", d / "labels.json", compare_svm=True, quiet=True)
    return d, out


def test_training_writes_all_artifacts(trained):
    d, out = trained
    for f in ("metrics.json", "confusion_matrix.png", "confusion_matrix.csv",
              "classification_report.txt", "model_comparison.json", "class_distribution.png"):
        assert (d / "res" / f).exists(), f
    assert (d / "m.joblib").exists() and json.loads((d / "labels.json").read_text())
    assert out["split_strategy"] == "group_by_user" and out["dataset_fingerprint"]
    assert 0 <= out["accuracy"] <= 1


def test_model_save_load_and_schema_consistency(trained):
    d, _ = trained
    clf = GestureClassifier.load(d / "m.joblib")
    assert clf.feature_names == fe.FEATURE_NAMES
    rng = np.random.default_rng(5)
    label, conf, probs = clf.predict(fe.extract_features(make_hand("FIST", rng)))
    assert label in config.GESTURE_CLASSES and 0 < conf <= 1 and abs(sum(probs.values()) - 1) < 1e-6


def test_schema_mismatch_model_rejected(trained, tmp_path):
    d, _ = trained
    b = joblib.load(d / "m.joblib"); b["feature_names"] = b["feature_names"][:-1]
    joblib.dump(b, tmp_path / "bad.joblib")
    with pytest.raises(ModelNotAvailableError):
        GestureClassifier.load(tmp_path / "bad.joblib")


def test_missing_and_corrupt_model_messages(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="train_model"):
        GestureClassifier.load(tmp_path / "none.joblib")
    (tmp_path / "x.joblib").write_text("garbage")
    with pytest.raises(ModelNotAvailableError):
        GestureClassifier.load(tmp_path / "x.joblib")


def test_training_refuses_insufficient_data(tmp_path):
    p = tmp_path / "d.csv"
    du.append_samples(synthetic_rows(5), p)
    with pytest.raises(du.DatasetError):
        train_model.run(p, tmp_path / "m.joblib", tmp_path / "r", tmp_path / "l.json", quiet=True)
    assert not (tmp_path / "m.joblib").exists()


def test_stability_filter():
    f = StabilityFilter(consistent_frames=3, confidence_threshold=0.7)
    assert f.update("A", 0.9) is None and f.update("A", 0.9) is None
    assert f.update("A", 0.9) == "A"
    assert f.update("A", 0.5) is None
    assert f.update("A", 0.9) is None
    assert f.update(None, 0) is None
    assert f.update("B", 0.9) is None


def test_single_frame_never_triggers():
    f = StabilityFilter(consistent_frames=1 + 5, confidence_threshold=0.5)
    assert f.update("PEACE", 1.0) is None


def test_pipeline_step_end_to_end_with_synthetic_landmarks(trained):
    d, _ = trained
    clf = GestureClassifier.load(d / "m.joblib")
    s = config.Settings(consistent_frames=4, confidence_threshold=0.5, cooldown_s=1.0)
    ctrl = CommandController(s)
    pipe = GesturePipeline(s, None, clf, ctrl)
    ctrl.activate()
    rng = np.random.default_rng(9)
    cmds = []
    for i in range(40):
        lm = make_hand("PEACE", rng, noise=0.001, rotate=False)
        r = pipe.step(HandResult(lm, "Right", 0.9, 1, [lm], 0), now=i * 0.05)
        if r.command:
            cmds.append(r.command.action)
    assert cmds.count(config.NEXT_SLIDE) == 1


def test_pipeline_no_hand_and_invalid_landmarks(trained):
    d, _ = trained
    pipe = GesturePipeline(config.Settings(), None, GestureClassifier.load(d / "m.joblib"))
    r = pipe.step(HandResult(), 0.0)
    assert not r.hand_detected and "No hand" in r.message
    bad = np.full((21, 3), np.nan)
    r = pipe.step(HandResult(bad, "Right", 0.9, 1, [bad], 0), 0.1)
    assert not r.hand_detected and "Unusable" in r.message


def test_pipeline_without_model_gives_instruction():
    pipe = GesturePipeline(config.Settings(), None, None)
    lm = make_hand("FIST", np.random.default_rng(0))
    r = pipe.step(HandResult(lm, "Right", 0.9, 1, [lm], 0), 0.0)
    assert "train_model" in r.message and r.command is None


def test_pipeline_swipe_executes_when_active():
    s = config.Settings()
    ctrl = CommandController(s)
    pipe = GesturePipeline(s, None, None, ctrl)
    ctrl.activate()
    cmds = []
    for i in range(14):
        lm = make_hand("OPEN_PALM", np.random.default_rng(0), noise=0, rotate=False)
        lm[:, 0] += (0.3 + 0.4 * i / 13) - lm[:, 0].mean()
        r = pipe.step(HandResult(lm, "Right", 0.9, 1, [lm], 0), now=i * 0.03)
        if r.command:
            cmds.append(r.command.action)
    assert cmds == [config.NEXT_SLIDE]
