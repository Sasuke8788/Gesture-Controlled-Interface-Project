"""Webcam lifecycle: open, read with failure tracking, always release."""
from __future__ import annotations

import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np


class WebcamError(RuntimeError):
    pass


class WebcamManager:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 max_consecutive_failures: int = 30, idle_timeout_s: Optional[float] = None):
        """idle_timeout_s: if set, a watchdog thread releases the camera when no frame has been
        read for that long (e.g. the Streamlit script was closed with the browser tab)."""
        self.index, self.width, self.height = index, width, height
        self.idle_timeout_s = idle_timeout_s
        self._lock = threading.Lock()
        self._last_touch = time.monotonic()
        self._watchdog: Optional[threading.Thread] = None
        self.max_failures = max_consecutive_failures
        self._cap: Optional[cv2.VideoCapture] = None
        self._failures = 0

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(self) -> "WebcamManager":
        if self.is_open:
            return self
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(self.index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if backend == cv2.CAP_DSHOW:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self._cap, self._failures = cap, 0
                self._last_touch = time.monotonic()
                self._start_watchdog()
                return self
            cap.release()
        raise WebcamError(
            f"Could not open camera index {self.index}. Close other apps using the camera "
            "(Teams/Zoom/Camera app), check Windows Settings > Privacy > Camera, or try a "
            "different camera index.")

    def read(self) -> np.ndarray:
        """Return one BGR frame. Raises WebcamError when the camera stops delivering frames."""
        with self._lock:
            cap = self._cap
        if cap is None or not cap.isOpened():
            raise WebcamError("Camera is not open.")
        self._last_touch = time.monotonic()
        ok, frame = cap.read()
        if not ok or frame is None:
            self._failures += 1
            if self._failures >= self.max_failures:
                self.release()
                raise WebcamError("Camera stopped delivering frames (disconnected or in use).")
            time.sleep(0.01)
            return None
        self._failures = 0
        return frame

    def read_latest(self) -> np.ndarray:
        with self._lock:
            cap = self._cap
        if cap is None or not cap.isOpened():
            raise WebcamError("Camera is not open.")
        self._last_touch = time.monotonic()
        latest = None
        for _ in range(2):
            if not cap.grab():
                break
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                latest = frame
        if latest is None:
            return self.read()
        self._failures = 0
        return latest

    def release(self) -> None:
        with self._lock:
            cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()

    def _start_watchdog(self) -> None:
        if not self.idle_timeout_s or (self._watchdog and self._watchdog.is_alive()):
            return

        def run():
            while self._cap is not None:
                time.sleep(0.5)
                if self._cap is not None and time.monotonic() - self._last_touch > self.idle_timeout_s:
                    self.release()
        self._watchdog = threading.Thread(target=run, daemon=True)
        self._watchdog.start()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.release()


def probe_cameras(max_index: int = 4) -> list[int]:
    """Return camera indices that open successfully (call only when no camera is in use)."""
    found = []
    for i in range(max_index):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found
