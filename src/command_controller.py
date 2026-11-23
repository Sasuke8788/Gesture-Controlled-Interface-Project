"""Gesture -> command execution with safety interlocks. ALL OS-specific code lives here.

State model
-----------
* ``active``   : gesture-control mode. OPEN_PALM (held) or the UI button turns it on, FIST or
                 the UI button turns it off. While inactive, slide commands are ignored.
* ``real_mode``: False (default) = SAFE SIMULATION, commands are only logged. True = real
                 keyboard events are sent. Only the UI / explicit flag can turn this on;
                 gestures never can.
* ``estop``    : emergency-stop latch. Clears active AND real_mode, and blocks gesture
                 activation until ``reset_estop()`` is called from the UI.

Anti-repeat logic: a gesture that has fired must be ABSENT (different gesture, no hand, or
unstable) for ``release_s`` before it can fire again, and slide commands share a global
cooldown. Static commands are blocked while the hand moves fast (``max_static_speed``) and
during the swipe cooldown, which resolves PEACE/POINTING vs SWIPE conflicts.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Optional, Protocol

import config
from config import (ACTIVATE, CONFIRM, DEACTIVATE, NEXT_SLIDE, PREV_SLIDE)

SLIDE_ACTIONS = {NEXT_SLIDE, PREV_SLIDE}


class KeySender(Protocol):
    def send(self, key: str) -> None: ...


class PynputKeySender:
    """Sends a key press via pynput (lazy import; Windows/macOS/Linux)."""

    def __init__(self):
        from pynput.keyboard import Controller, Key
        self._kb, self._Key = Controller(), Key

    def send(self, key: str) -> None:
        k = getattr(self._Key, key, None) or key
        self._kb.press(k)
        self._kb.release(k)


def focus_window_by_title(title_fragment: str) -> bool:
    """Best-effort: bring a top-level Windows window whose title contains the text to front.

    Returns True if a window was found and focus was requested. No-op (False) off Windows.
    Windows may refuse focus changes for background processes; the README explains the fallback
    (click the slideshow window yourself before activating real mode).
    """
    if not sys.platform.startswith("win") or not title_fragment:
        return False
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def enum_cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if title_fragment.lower() in buf.value.lower():
                found.append(hwnd)
        return True

    user32.EnumWindows(enum_cb, 0)
    if not found:
        return False
    hwnd = found[0]
    if user32.GetForegroundWindow() == hwnd:
        return True
    user32.keybd_event(0x12, 0, 0, 0)
    user32.keybd_event(0x12, 0, 2, 0)
    user32.ShowWindow(hwnd, 9)
    return bool(user32.SetForegroundWindow(hwnd))


@dataclass
class CommandRecord:
    timestamp: str
    source: str
    action: str
    mode: str
    sent: bool
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class CommandController:
    def __init__(self, settings: Optional[config.Settings] = None,
                 key_sender: Optional[KeySender] = None,
                 focus_fn: Callable[[str], bool] = focus_window_by_title,
                 max_history: int = 200):
        self.settings = settings or config.Settings()
        self._sender = key_sender
        self._focus_fn = focus_fn
        self.max_history = max_history
        self.active = False
        self.estop = False
        self.history: list[CommandRecord] = []
        self.last_action: Optional[CommandRecord] = None
        self.last_flash_time = 0.0
        self._last_fired: Optional[str] = None
        self._absent_since: Optional[float] = None
        self._cooldown_until = -1e9
        self._palm_since: Optional[float] = None
        self._suppress_static_until = -1e9

    @property
    def real_mode(self) -> bool:
        return self.settings.real_mode

    def set_real_mode(self, enabled: bool) -> None:
        if enabled and self.estop:
            self._log("UI", "SET_REAL_MODE", "STATE", False, "blocked: emergency stop latched")
            return
        self.settings.real_mode = bool(enabled)
        self._log("UI", "REAL_MODE_ON" if enabled else "REAL_MODE_OFF", "STATE", False)

    def activate(self, source: str = "UI") -> bool:
        if self.estop:
            self._log(source, ACTIVATE, "STATE", False, "blocked: emergency stop latched")
            return False
        if not self.active:
            self.active = True
            self._log(source, ACTIVATE, "STATE", False)
        return True

    def deactivate(self, source: str = "UI") -> None:
        if self.active:
            self.active = False
            self._log(source, DEACTIVATE, "STATE", False)
        self._reset_gesture_state()

    def emergency_stop(self) -> None:
        self.active = False
        self.settings.real_mode = False
        self.estop = True
        self._reset_gesture_state()
        self._log("UI", "EMERGENCY_STOP", "STATE", False, "control disabled, real mode off")

    def reset_estop(self) -> None:
        self.estop = False
        self._log("UI", "ESTOP_RESET", "STATE", False, "gesture activation allowed again")

    def handle_no_hand(self, now: float) -> None:
        self._palm_since = None
        self._mark_absent(now)

    def handle_static(self, gesture: Optional[str], now: float,
                      hand_speed: float = 0.0) -> Optional[CommandRecord]:
        """``gesture`` is a STABLE static gesture (or None). Returns a record if one fired."""
        if gesture is None or gesture == config.NO_GESTURE:
            self._palm_since = None
            self._mark_absent(now)
            return None
        if gesture != self._last_fired:
            self._mark_absent(now)
        else:
            self._absent_since = None

        if gesture == "OPEN_PALM":
            if self.active or self.estop:
                self._palm_since = None
                return None
            if self._palm_since is None:
                self._palm_since = now
            held = now - self._palm_since
            if held >= self.settings.activation_hold_s and hand_speed <= self.settings.max_static_speed:
                self._palm_since = None
                self.activate("OPEN_PALM")
                return self.history[-1] if self.history else None
            return None
        self._palm_since = None
        if gesture == "FIST":
            if self.active:
                self.deactivate("FIST")
                self._last_fired = "FIST"
                return self.history[-1]
            return None

        if not self.active or self.estop:
            return None
        action = config.GESTURE_TO_ACTION.get(gesture)
        if action is None:
            return None
        if hand_speed > self.settings.max_static_speed or now < self._suppress_static_until:
            return None
        if not self._can_fire(gesture, now):
            return None
        rec = self._execute(gesture, action, now)
        self._last_fired = gesture
        self._absent_since = None
        return rec

    def handle_swipe(self, swipe: Optional[str], now: float) -> Optional[CommandRecord]:
        if swipe is None or not self.settings.swipe_enabled:
            return None
        self._suppress_static_until = now + self.settings.swipe_cooldown_s
        if not self.active or self.estop:
            return None
        if now < self._cooldown_until:
            return None
        action = config.GESTURE_TO_ACTION[swipe]
        rec = self._execute(swipe, action, now)
        self._last_fired = swipe
        return rec

    def _mark_absent(self, now: float) -> None:
        if self._absent_since is None:
            self._absent_since = now
        if self._last_fired is not None and now - self._absent_since >= self.settings.release_s:
            self._last_fired = None

    def _can_fire(self, gesture: str, now: float) -> bool:
        if now < self._cooldown_until:
            return False
        if self._last_fired == gesture:
            return False
        return True

    def _reset_gesture_state(self) -> None:
        self._palm_since = None
        self._absent_since = None

    def _execute(self, source: str, action: str, now: float) -> CommandRecord:
        s = self.settings
        if action == CONFIRM:
            rec = self._log(source, action, "SIMULATED", False, "safe test action (no key sent)")
            self._cooldown_until = now + s.cooldown_s
            return rec
        key = s.next_key if action == NEXT_SLIDE else s.prev_key
        self._cooldown_until = now + s.cooldown_s
        if not s.real_mode:
            return self._log(source, action, "SIMULATED", False, f"would press '{key}'")
        try:
            if self._sender is None:
                self._sender = PynputKeySender()
            if s.target_window_title:
                self._focus_fn(s.target_window_title)
            self._sender.send(key)
            return self._log(source, action, "REAL", True, f"pressed '{key}'")
        except Exception as exc:
            s.real_mode = False
            return self._log(source, action, "SIMULATED", False,
                             f"key send FAILED ({exc}); real mode turned off")

    def _log(self, source: str, action: str, mode: str, sent: bool, note: str = "") -> CommandRecord:
        rec = CommandRecord(datetime.now().strftime("%H:%M:%S.%f")[:-3], source, action,
                            mode, sent, note)
        self.history.append(rec)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        self.last_action = rec
        self.last_flash_time = time.monotonic()
        return rec

    def executed_actions(self) -> list[CommandRecord]:
        """Slide/confirm/activate-type records (excludes UI-only state messages)."""
        return [r for r in self.history if r.source != "UI"]
