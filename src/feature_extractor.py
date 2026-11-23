"""Landmark feature extraction shared by collection, training, evaluation and live inference.

Input: 21 MediaPipe hand landmarks as an array of shape (21, 3) (x, y normalised to the image,
z relative depth). Output: a fixed-length 1-D float vector described by ``FEATURE_NAMES``.

Normalisation (reduces sensitivity to position and scale):
  1. Left hands are mirrored (x -> -x) so both hands share one canonical shape.
  2. Landmarks are translated so the wrist is the origin   -> position invariance.
  3. Divided by the wrist->middle-finger-MCP distance        -> scale invariance.
  4. (optional) rotated in the image plane so wrist->middle-MCP points "up".

Limitations (documented honestly in the README):
  * Only IN-PLANE rotation is handled. Out-of-plane rotation (palm turned away, hand tilted
    toward/away from the camera) changes the 2-D projection and degrades accuracy.
  * z from MediaPipe is a relative, noisy estimate; it is used with a small weight (raw but scaled).
  * Left/right handling relies on MediaPipe's handedness label; a wrong label mirrors the shape.
  * Scale uses the palm-length projected into 2-D, which shrinks when the hand tilts.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

import config

NUM_LANDMARKS = 21
WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 0, 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}
PALM_POINTS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


class InvalidLandmarksError(ValueError):
    """Raised when landmarks are missing, malformed or contain NaN/inf."""


def _build_feature_names() -> list[str]:
    names = []
    for i in range(1, NUM_LANDMARKS):
        names += [f"rel_x{i}", f"rel_y{i}", f"rel_z{i}"]
    for f in FINGERS:
        names += [f"{f}_angle_a", f"{f}_angle_b"]
    for f in FINGERS:
        names.append(f"{f}_tip_wrist_dist")
    for f in FINGERS:
        names.append(f"{f}_tip_palm_dist")
    for f in FINGERS:
        names.append(f"{f}_extension_ratio")
    names += ["thumb_index_tip_dist", "index_middle_tip_dist", "palm_spread"]
    return names


FEATURE_NAMES: list[str] = _build_feature_names()
NUM_FEATURES = len(FEATURE_NAMES)


def validate_landmarks(landmarks) -> np.ndarray:
    """Return landmarks as float array (21, 3) or raise InvalidLandmarksError."""
    if landmarks is None:
        raise InvalidLandmarksError("landmarks is None")
    try:
        arr = np.asarray(landmarks, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidLandmarksError(f"landmarks not numeric: {exc}") from exc
    if arr.ndim != 2 or arr.shape[0] != NUM_LANDMARKS or arr.shape[1] not in (2, 3):
        raise InvalidLandmarksError(f"expected shape (21, 2|3), got {arr.shape}")
    if arr.shape[1] == 2:
        arr = np.hstack([arr, np.zeros((NUM_LANDMARKS, 1))])
    if not np.all(np.isfinite(arr)):
        raise InvalidLandmarksError("landmarks contain NaN or inf")
    return arr


def normalize_landmarks(landmarks, handedness: Optional[str] = None,
                        rotate: bool = True) -> np.ndarray:
    """Mirror left hands, centre on wrist, scale by palm length, optionally align rotation."""
    pts = validate_landmarks(landmarks).copy()
    if handedness is not None and handedness.lower().startswith("left"):
        pts[:, 0] = -pts[:, 0]
    pts -= pts[WRIST]
    scale = float(np.linalg.norm(pts[MIDDLE_MCP, :2]))
    if scale < 1e-6:
        raise InvalidLandmarksError("degenerate hand: wrist and middle-MCP coincide")
    pts /= scale
    if rotate:
        vx, vy = pts[MIDDLE_MCP, 0], pts[MIDDLE_MCP, 1]
        angle = -math.atan2(vx, -vy)
        c, s = math.cos(angle), math.sin(angle)
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        pts[:, 0] = c * x - s * y
        pts[:, 1] = s * x + c * y
    return pts


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle (radians) at b between segments b->a and b->c, using x,y,z."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def extract_features(landmarks, handedness: Optional[str] = None,
                     rotate: bool = True) -> np.ndarray:
    """Convert 21 landmarks to the model feature vector (length NUM_FEATURES).

    Raises InvalidLandmarksError for unusable input. Callers on the live path should catch it
    and treat the frame as "no usable hand".
    """
    p = normalize_landmarks(landmarks, handedness, rotate)
    feats: list[float] = []
    for i in range(1, NUM_LANDMARKS):
        feats += [p[i, 0], p[i, 1], p[i, 2] * 0.5]
    palm_center = p[PALM_POINTS].mean(axis=0)
    for name, (mcp_or_cmc, a, b, tip) in FINGERS.items():
        base = WRIST if name != "thumb" else THUMB_CMC
        feats.append(_angle(p[base], p[mcp_or_cmc], p[a]) if name != "thumb"
                     else _angle(p[WRIST], p[THUMB_CMC], p[THUMB_MCP]))
        feats.append(_angle(p[mcp_or_cmc], p[a], p[b]) if name != "thumb"
                     else _angle(p[THUMB_MCP], p[THUMB_IP], p[THUMB_TIP]))
    for _, (_, _, _, tip) in FINGERS.items():
        feats.append(float(np.linalg.norm(p[tip] - p[WRIST])))
    for _, (_, _, _, tip) in FINGERS.items():
        feats.append(float(np.linalg.norm(p[tip] - palm_center)))
    for name, (mcp, pip_, dip, tip) in FINGERS.items():
        ref = p[pip_] if name != "thumb" else p[THUMB_MCP]
        feats.append(float(np.linalg.norm(p[tip] - p[WRIST]) /
                           max(np.linalg.norm(ref - p[WRIST]), 1e-6)))
    feats.append(float(np.linalg.norm(p[THUMB_TIP] - p[INDEX_TIP])))
    feats.append(float(np.linalg.norm(p[INDEX_TIP] - p[MIDDLE_TIP])))
    feats.append(float(np.linalg.norm(p[INDEX_TIP] - p[PINKY_TIP])))
    out = np.asarray(feats, dtype=np.float64)
    if out.shape[0] != NUM_FEATURES or not np.all(np.isfinite(out)):
        raise InvalidLandmarksError("feature extraction produced invalid output")
    return out


def hand_center(landmarks) -> tuple[float, float]:
    """Palm centre in image-normalised coordinates (used by swipe detection)."""
    arr = validate_landmarks(landmarks)
    c = arr[PALM_POINTS, :2].mean(axis=0)
    return float(c[0]), float(c[1])


def features_to_dict(vec: np.ndarray) -> dict:
    return dict(zip(FEATURE_NAMES, map(float, vec)))
