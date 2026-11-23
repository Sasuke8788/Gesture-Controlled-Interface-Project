"""Random Forest gesture model wrapper + temporal stability filter.

NOTE ON CONFIDENCE: the value returned is the Random Forest's predicted class probability
(fraction of trees voting for the class). It is a useful ranking score but NOT a calibrated
probability that the prediction is correct, especially for poses unlike the training data.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

import config
from src.feature_extractor import FEATURE_NAMES


class ModelNotAvailableError(RuntimeError):
    pass


class GestureClassifier:
    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.classes: list[str] = list(bundle["classes"])
        self.feature_names: list[str] = list(bundle["feature_names"])
        self.meta: dict = bundle.get("meta", {})
        if self.feature_names != FEATURE_NAMES:
            raise ModelNotAvailableError(
                "The saved model was trained with a different feature schema than the "
                "current code. Re-collect/convert data and re-run train_model.py.")
        if int(bundle.get("schema_version", -1)) != config.FEATURE_SCHEMA_VERSION:
            raise ModelNotAvailableError("Feature schema version mismatch; retrain the model.")

    @classmethod
    def load(cls, path: Path = config.MODEL_PATH) -> "GestureClassifier":
        path = Path(path)
        if not path.exists():
            raise ModelNotAvailableError(
                f"No trained model at {path}. Collect data (python collect_data.py) and "
                "train it (python train_model.py) first.")
        try:
            bundle = joblib.load(path)
        except Exception as exc:
            raise ModelNotAvailableError(f"Could not load model ({exc}). Retrain it.") from exc
        return cls(bundle)

    def predict(self, features: np.ndarray) -> tuple[str, float, dict]:
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        if x.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {x.shape[1]}")
        proba = self.model.predict_proba(x)[0]
        i = int(np.argmax(proba))
        return str(self.model.classes_[i]), float(proba[i]), {
            str(c): float(p) for c, p in zip(self.model.classes_, proba)}


class StabilityFilter:
    """Turns noisy per-frame predictions into a *stable* gesture.

    A gesture is stable once the last ``consistent_frames`` frames all predicted it with
    confidence >= threshold. Any hand-less frame, low-confidence frame, or different label
    breaks the streak, so a single frame can never produce a stable gesture.
    """

    def __init__(self, consistent_frames: int = 6, confidence_threshold: float = 0.7):
        self.consistent_frames = max(1, int(consistent_frames))
        self.threshold = float(confidence_threshold)
        self._recent: deque = deque(maxlen=self.consistent_frames)

    def configure(self, consistent_frames: int, confidence_threshold: float) -> None:
        self.consistent_frames = max(1, int(consistent_frames))
        self.threshold = float(confidence_threshold)
        self._recent = deque(self._recent, maxlen=self.consistent_frames)

    def reset(self) -> None:
        self._recent.clear()

    def update(self, label: Optional[str], confidence: float) -> Optional[str]:
        """Feed one frame (label None = no hand). Returns the stable label or None."""
        if label is None or confidence < self.threshold:
            self._recent.append(None)
            return None
        self._recent.append(label)
        if len(self._recent) == self.consistent_frames and len(set(self._recent)) == 1:
            return label
        return None
