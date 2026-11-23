import numpy as np
import pytest

from src import feature_extractor as fe
from tests.synthetic import make_hand


def hand(pose="OPEN_PALM", seed=0, **kw):
    return make_hand(pose, np.random.default_rng(seed), **kw)


def test_feature_vector_matches_schema():
    v = fe.extract_features(hand())
    assert v.shape == (fe.NUM_FEATURES,) and len(fe.FEATURE_NAMES) == fe.NUM_FEATURES
    assert len(set(fe.FEATURE_NAMES)) == fe.NUM_FEATURES
    assert np.all(np.isfinite(v))


def test_translation_and_scale_invariance():
    lm = hand(noise=0)
    base = fe.extract_features(lm)
    moved = lm.copy(); moved[:, :2] = moved[:, :2] * 0.6 + 0.13
    moved[:, 2] = lm[:, 2] * 0.6
    np.testing.assert_allclose(fe.extract_features(moved), base, atol=1e-9)


def test_inplane_rotation_invariance_when_enabled():
    lm = hand(noise=0, rotate=False)
    a = 0.5
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rot = lm.copy(); rot[:, :2] = (lm[:, :2] - 0.5) @ R.T + 0.5
    np.testing.assert_allclose(fe.extract_features(rot), fe.extract_features(lm), atol=1e-6)


def test_left_hand_mirroring():
    lm = hand(noise=0)
    mirrored = lm.copy(); mirrored[:, 0] = 1.0 - mirrored[:, 0]
    np.testing.assert_allclose(fe.extract_features(mirrored, "Left"), fe.extract_features(lm, "Right"), atol=1e-9)


def test_different_poses_give_different_features():
    f = {p: fe.extract_features(hand(p, noise=0)) for p in ("OPEN_PALM", "FIST", "PEACE")}
    assert np.linalg.norm(f["OPEN_PALM"] - f["FIST"]) > 0.5
    assert np.linalg.norm(f["PEACE"] - f["FIST"]) > 0.3


@pytest.mark.parametrize("bad", [None, [], np.zeros((20, 3)), np.zeros((21, 4)), "abc", np.zeros((21, 3))])
def test_invalid_landmarks_raise(bad):
    with pytest.raises(fe.InvalidLandmarksError):
        fe.extract_features(bad)


def test_nan_landmarks_raise():
    lm = hand(); lm[3, 0] = np.nan
    with pytest.raises(fe.InvalidLandmarksError):
        fe.extract_features(lm)


def test_2d_input_is_accepted():
    assert fe.extract_features(hand()[:, :2]).shape == (fe.NUM_FEATURES,)
