"""MediaPipe Hand Landmarker wrapper (Tasks API, IMAGE mode, CPU).

MediaPipe supplies the PRETRAINED palm detector + 21-landmark regressor. We do not train it.
Hand selection policy (documented): when several hands are visible, the hand with the largest
landmark bounding box (normally the one closest to the camera) is used for classification and
control; every detected hand is still drawn, the chosen one in a highlight colour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
    (18, 19), (19, 20), (0, 17),
]


class HandModelMissingError(RuntimeError):
    """The hand_landmarker.task asset is not available."""


@dataclass
class HandResult:
    landmarks: Optional[np.ndarray] = None
    handedness: Optional[str] = None
    score: float = 0.0
    num_hands: int = 0
    all_hands: list = field(default_factory=list)
    selected_index: int = -1

    @property
    def detected(self) -> bool:
        return self.landmarks is not None


def _bbox_area(lm: np.ndarray) -> float:
    return float((lm[:, 0].max() - lm[:, 0].min()) * (lm[:, 1].max() - lm[:, 1].min()))


class HandDetector:
    def __init__(self, model_path: Path = config.HAND_MODEL_PATH, max_hands: int = 1,
                 min_detection_conf: float = 0.5, min_tracking_conf: float = 0.5,
                 max_detection_width: int = 480):
        model_path = Path(model_path)
        if not model_path.exists():
            raise HandModelMissingError(
                f"MediaPipe hand model not found at {model_path}.\n"
                "Run:  python download_model.py   (needs internet once), or download\n"
                f"{config.HAND_MODEL_URL}\nand save it as models/hand_landmarker.task")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        self.max_detection_width = max_detection_width
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_conf,
            min_hand_presence_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> HandResult:
        """Detect hands in a BGR frame. The caller should pass the mirrored (selfie) frame."""
        if frame_bgr is None or frame_bgr.size == 0:
            return HandResult()
        h, w = frame_bgr.shape[:2]
        detection_frame = frame_bgr
        if w > self.max_detection_width:
            scale = self.max_detection_width / w
            detection_frame = cv2.resize(
                frame_bgr,
                (self.max_detection_width, int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect(mp_image)
        if not res.hand_landmarks:
            return HandResult()
        hands, labels, scores = [], [], []
        for i, lms in enumerate(res.hand_landmarks):
            hands.append(np.array([[l.x, l.y, l.z] for l in lms], dtype=np.float64))
            cat = res.handedness[i][0] if res.handedness and len(res.handedness) > i else None
            labels.append(cat.category_name if cat else None)
            scores.append(float(cat.score) if cat else 0.0)
        idx = int(np.argmax([_bbox_area(h) for h in hands]))
        return HandResult(hands[idx], labels[idx], scores[idx], len(hands), hands, idx)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


def draw_hands(frame: np.ndarray, result: HandResult) -> np.ndarray:
    """Draw landmarks + connections on ``frame`` in place and return it."""
    h, w = frame.shape[:2]
    for i, lm in enumerate(result.all_hands):
        selected = i == result.selected_index
        line_c = (0, 220, 120) if selected else (120, 120, 120)
        pts = [(int(x * w), int(y * h)) for x, y, _ in lm]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], line_c, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 120, 255) if selected else (90, 90, 90), -1, cv2.LINE_AA)
    return frame
