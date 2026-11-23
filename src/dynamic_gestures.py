"""Rule-based temporal swipe detection (NOT a trained ML model).

Uses a sliding window of (timestamp, x, y) palm-centre samples in image-normalised coordinates
of the MIRRORED (selfie) frame, so moving your hand to your right increases x => SWIPE_RIGHT.
A swipe needs: enough horizontal displacement, a plausible duration, mostly-monotonic motion,
low vertical drift, continuous tracking, and it is followed by a cooldown. Losing the hand
clears the window.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import config


class SwipeDetector:
    def __init__(self, min_displacement: float = 0.22, max_duration_s: float = 0.7,
                 min_duration_s: float = 0.08, max_vertical_ratio: float = 0.6,
                 cooldown_s: float = 1.0, max_gap_s: float = 0.25,
                 min_monotonic_fraction: float = 0.7, window_s: float = 1.0):
        self.min_displacement = min_displacement
        self.max_duration_s = max_duration_s
        self.min_duration_s = min_duration_s
        self.max_vertical_ratio = max_vertical_ratio
        self.cooldown_s = cooldown_s
        self.max_gap_s = max_gap_s
        self.min_monotonic_fraction = min_monotonic_fraction
        self.window_s = window_s
        self._buf: deque = deque()
        self._cooldown_until = -1e9
        self.last_speed = 0.0

    @classmethod
    def from_settings(cls, s: config.Settings) -> "SwipeDetector":
        return cls(s.swipe_min_displacement, s.swipe_max_duration_s, s.swipe_min_duration_s,
                   s.swipe_max_vertical_ratio, s.swipe_cooldown_s)

    def reset(self) -> None:
        self._buf.clear()
        self.last_speed = 0.0

    @property
    def in_cooldown_until(self) -> float:
        return self._cooldown_until

    def update(self, t: float, position: Optional[tuple[float, float]]) -> Optional[str]:
        """Feed one frame. ``position`` None means tracking lost. Returns a swipe or None."""
        if position is None:
            self.reset()
            return None
        if self._buf and t - self._buf[-1][0] > self.max_gap_s:
            self._buf.clear()
        if self._buf and t <= self._buf[-1][0]:
            return None
        self._buf.append((t, position[0], position[1]))
        while self._buf and t - self._buf[0][0] > self.window_s:
            self._buf.popleft()
        self._update_speed()
        if t < self._cooldown_until or len(self._buf) < 4:
            return None
        return self._evaluate(t)

    def _update_speed(self) -> None:
        recent = [s for s in self._buf if self._buf[-1][0] - s[0] <= 0.3]
        if len(recent) >= 2 and recent[-1][0] > recent[0][0]:
            self.last_speed = abs(recent[-1][1] - recent[0][1]) / (recent[-1][0] - recent[0][0])
        else:
            self.last_speed = 0.0

    def _evaluate(self, now: float) -> Optional[str]:
        samples = list(self._buf)
        samples = [s for s in samples if now - s[0] <= self.max_duration_s]
        if len(samples) < 4:
            return None
        duration = samples[-1][0] - samples[0][0]
        if duration < self.min_duration_s:
            return None
        xs = [s[1] for s in samples]
        ys = [s[2] for s in samples]
        dx, dy = xs[-1] - xs[0], ys[-1] - ys[0]
        if abs(dx) < self.min_displacement:
            return None
        if abs(dy) > self.max_vertical_ratio * abs(dx):
            return None
        sign = 1 if dx > 0 else -1
        steps = [(xs[i + 1] - xs[i]) * sign for i in range(len(xs) - 1)]
        moving = sum(1 for st in steps if st > 0)
        if moving / len(steps) < self.min_monotonic_fraction:
            return None
        self._cooldown_until = now + self.cooldown_s
        self._buf.clear()
        return config.SWIPE_RIGHT if sign > 0 else config.SWIPE_LEFT
