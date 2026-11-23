"""Central configuration: paths, gesture classes, command mapping, tunable settings."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

DATASET_PATH = DATA_DIR / "gesture_samples.csv"
MODEL_PATH = MODELS_DIR / "gesture_rf.joblib"
LABELS_PATH = MODELS_DIR / "class_labels.json"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
METRICS_PATH = RESULTS_DIR / "metrics.json"
RELIABILITY_LOG_PATH = RESULTS_DIR / "reliability_log.csv"

GESTURE_CLASSES = ["OPEN_PALM", "FIST", "THUMBS_UP", "PEACE", "POINTING", "NO_GESTURE"]
NO_GESTURE = "NO_GESTURE"
GESTURE_HELP = {
    "OPEN_PALM": "All five fingers extended and spread, palm facing the camera.",
    "FIST": "All fingers curled into a closed fist.",
    "THUMBS_UP": "Fist with only the thumb extended upward.",
    "PEACE": "Index and middle fingers extended in a V, others curled.",
    "POINTING": "Only the index finger extended, others curled.",
    "NO_GESTURE": "Hand visible but NOT one of the above: relaxed hand, half-open "
                  "fingers, transitions between gestures, random poses.",
}

SWIPE_LEFT = "SWIPE_LEFT"
SWIPE_RIGHT = "SWIPE_RIGHT"

ACTIVATE = "ACTIVATE"
DEACTIVATE = "DEACTIVATE"
NEXT_SLIDE = "NEXT_SLIDE"
PREV_SLIDE = "PREV_SLIDE"
CONFIRM = "CONFIRM_TEST"

GESTURE_TO_ACTION = {
    "OPEN_PALM": ACTIVATE,
    "FIST": DEACTIVATE,
    "PEACE": NEXT_SLIDE,
    "POINTING": PREV_SLIDE,
    "THUMBS_UP": CONFIRM,
    SWIPE_RIGHT: NEXT_SLIDE,
    SWIPE_LEFT: PREV_SLIDE,
}

DEFAULT_SHORTCUTS = {NEXT_SLIDE: "right", PREV_SLIDE: "left"}
ALLOWED_KEYS = ["right", "left", "page_down", "page_up", "down", "up", "space"]

FEATURE_SCHEMA_VERSION = 1


@dataclass
class Settings:
    """Runtime-tunable settings (editable on the dashboard)."""
    confidence_threshold: float = 0.70
    consistent_frames: int = 6
    cooldown_s: float = 1.2
    release_s: float = 0.5
    activation_hold_s: float = 1.0
    max_static_speed: float = 0.6
    swipe_enabled: bool = True
    swipe_min_displacement: float = 0.22
    swipe_max_duration_s: float = 0.7
    swipe_min_duration_s: float = 0.08
    swipe_max_vertical_ratio: float = 0.6
    swipe_cooldown_s: float = 1.0
    real_mode: bool = False
    next_key: str = "right"
    prev_key: str = "left"
    target_window_title: str = ""
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    min_detection_conf: float = 0.5
    min_tracking_conf: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)
