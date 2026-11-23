"""Glue: frame -> hand -> features -> classifier -> stability -> swipe -> command controller."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import config
from src import feature_extractor as fe
from src.command_controller import CommandController, CommandRecord
from src.dynamic_gestures import SwipeDetector
from src.gesture_classifier import GestureClassifier, StabilityFilter
from src.hand_detector import HandDetector, HandResult, draw_hands


@dataclass
class FrameResult:
    frame: Optional[np.ndarray] = None
    hand_detected: bool = False
    num_hands: int = 0
    raw_label: Optional[str] = None
    confidence: float = 0.0
    stable_gesture: Optional[str] = None
    swipe: Optional[str] = None
    command: Optional[CommandRecord] = None
    proc_ms: float = 0.0
    fps: float = 0.0
    message: str = ""


class GesturePipeline:
    def __init__(self, settings: config.Settings, detector: Optional[HandDetector],
                 classifier: Optional[GestureClassifier],
                 controller: Optional[CommandController] = None):
        self.settings = settings
        self.detector = detector
        self.classifier = classifier
        self.controller = controller or CommandController(settings)
        self.controller.settings = settings
        self.stability = StabilityFilter(settings.consistent_frames, settings.confidence_threshold)
        self.swipe = SwipeDetector.from_settings(settings)
        self._last_t: Optional[float] = None
        self._fps = 0.0

    def apply_settings(self) -> None:
        s = self.settings
        self.stability.configure(s.consistent_frames, s.confidence_threshold)
        keep_cd = self.swipe.in_cooldown_until
        self.swipe = SwipeDetector.from_settings(s)
        self.swipe._cooldown_until = keep_cd

    def process(self, frame_bgr: np.ndarray, now: Optional[float] = None) -> FrameResult:
        """Process one MIRRORED BGR frame (annotated in place)."""
        t0 = time.perf_counter()
        hand = self.detector.detect(frame_bgr) if self.detector else HandResult()
        res = self.step(hand, now if now is not None else time.monotonic())
        draw_hands(frame_bgr, hand)
        res.frame = frame_bgr
        res.proc_ms = (time.perf_counter() - t0) * 1000.0
        return res

    def step(self, hand: HandResult, now: float) -> FrameResult:
        """Logic after detection. Separated so it can be unit-tested with synthetic landmarks."""
        if self._last_t is not None and now > self._last_t:
            inst = 1.0 / (now - self._last_t)
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
        self._last_t = now
        r = FrameResult(hand_detected=hand.detected, num_hands=hand.num_hands, fps=self._fps)
        ctrl = self.controller

        if not hand.detected:
            self.stability.update(None, 0.0)
            self.swipe.update(now, None)
            ctrl.handle_no_hand(now)
            r.message = "No hand detected"
            return r

        try:
            feats = fe.extract_features(hand.landmarks, hand.handedness)
            center = fe.hand_center(hand.landmarks)
        except fe.InvalidLandmarksError as exc:
            self.stability.update(None, 0.0)
            self.swipe.update(now, None)
            ctrl.handle_no_hand(now)
            r.hand_detected = False
            r.message = f"Unusable landmarks: {exc}"
            return r

        swipe = self.swipe.update(now, center)
        r.swipe = swipe
        if swipe:
            r.command = ctrl.handle_swipe(swipe, now) or r.command

        if self.classifier is None:
            r.message = "No trained model: run collect_data.py then train_model.py"
            return r
        label, conf, _ = self.classifier.predict(feats)
        r.raw_label, r.confidence = label, conf
        stable = self.stability.update(label, conf)
        r.stable_gesture = stable
        cmd = ctrl.handle_static(stable, now, self.swipe.last_speed)
        r.command = cmd or r.command
        return r


def draw_overlay(frame: np.ndarray, res: FrameResult, controller: CommandController) -> None:
    """Minimal HUD for OpenCV-window scripts."""
    text = f"{res.raw_label or '-'} {res.confidence:.2f} | stable: {res.stable_gesture or '-'}"
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    state = "ACTIVE" if controller.active else "INACTIVE"
    mode = "REAL" if controller.real_mode else "SIMULATED"
    cv2.putText(frame, f"{state} | {mode} | {res.fps:.0f} FPS", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if controller.active else (0, 0, 255), 2)
