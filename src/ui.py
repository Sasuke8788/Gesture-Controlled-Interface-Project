"""Streamlit helpers: shared session state, cached asset loading, styling."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

import config
from src.command_controller import CommandController
from src.gesture_classifier import GestureClassifier, ModelNotAvailableError
from src.hand_detector import HandDetector, HandModelMissingError
from src.pipeline import GesturePipeline
from src.webcam_manager import WebcamManager

CSS = """
<style>
.block-container {padding-top: 1.4rem;}
.card {background:#161b26;border:1px solid #2a3142;border-radius:10px;padding:12px 16px;margin-bottom:10px;}
.card .k {color:#8b94a7;font-size:0.78rem;text-transform:uppercase;letter-spacing:.06em;}
.card .v {font-size:1.5rem;font-weight:650;}
.ok {color:#34d399;} .bad {color:#f87171;} .warn {color:#fbbf24;}
.flash {background:#14532d;border:1px solid #22c55e;border-radius:10px;padding:10px 14px;font-weight:600;}
</style>
"""


def setup(title: str) -> None:
    st.set_page_config(page_title=title, page_icon="🖐️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def card(label: str, value: str, cls: str = "") -> str:
    return f'<div class="card"><div class="k">{label}</div><div class="v {cls}">{value}</div></div>'


def get_settings() -> config.Settings:
    if "settings" not in st.session_state:
        st.session_state.settings = config.Settings()
    return st.session_state.settings


def get_controller() -> CommandController:
    if "controller" not in st.session_state:
        st.session_state.controller = CommandController(get_settings())
    return st.session_state.controller


def model_mtime() -> float:
    return config.MODEL_PATH.stat().st_mtime if config.MODEL_PATH.exists() else 0.0


@st.cache_resource(show_spinner=False)
def _load_classifier(mtime: float):
    return GestureClassifier.load()


@st.cache_resource(show_spinner=False)
def _load_detector(det_conf: float, track_conf: float, asset_mtime: float):
    return HandDetector(min_detection_conf=det_conf, min_tracking_conf=track_conf)


def load_assets(settings: config.Settings):
    """Returns (detector|None, classifier|None, errors: list[str], warnings: list[str])."""
    errors, warnings, det, clf = [], [], None, None
    try:
        asset_m = config.HAND_MODEL_PATH.stat().st_mtime if config.HAND_MODEL_PATH.exists() else 0.0
        det = _load_detector(settings.min_detection_conf, settings.min_tracking_conf, asset_m)
    except HandModelMissingError as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"Could not start MediaPipe: {exc}")
    try:
        clf = _load_classifier(model_mtime())
    except ModelNotAvailableError as exc:
        _load_classifier.clear()
        warnings.append(str(exc))
    return det, clf, errors, warnings


def get_pipeline(settings, detector, classifier) -> GesturePipeline:
    key = (id(detector), id(classifier))
    pipe = st.session_state.get("pipeline")
    if pipe is None or st.session_state.get("pipeline_key") != key:
        pipe = GesturePipeline(settings, detector, classifier, get_controller())
        st.session_state.pipeline, st.session_state.pipeline_key = pipe, key
    pipe.apply_settings()
    return pipe


def get_camera(settings: config.Settings) -> WebcamManager:
    cam = st.session_state.get("cam")
    want = (settings.camera_index, settings.frame_width, settings.frame_height)
    if cam is None or st.session_state.get("cam_key") != want:
        if cam is not None:
            cam.release()
        cam = WebcamManager(*want)
        st.session_state.cam, st.session_state.cam_key = cam, want
    return cam


def release_camera() -> None:
    """Stop streaming and free the webcam. Called when leaving the live page and on Stop."""
    st.session_state.running = False
    cam = st.session_state.get("cam")
    if cam is not None:
        cam.release()


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
