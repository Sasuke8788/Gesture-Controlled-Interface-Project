import pytest

import config
from src.command_controller import CommandController


class FakeSender:
    def __init__(self, fail=False):
        self.keys, self.fail = [], fail

    def send(self, key):
        if self.fail:
            raise RuntimeError("boom")
        self.keys.append(key)


def make(real=False, **kw):
    s = config.Settings(real_mode=real, cooldown_s=1.0, release_s=0.5, activation_hold_s=1.0, **kw)
    snd = FakeSender()
    return CommandController(s, snd, focus_fn=lambda t: True), snd


def hold(c, gesture, t0, dur, step=0.05, speed=0.0):
    last = None
    t = t0
    while t <= t0 + dur:
        last = c.handle_static(gesture, t, speed) or last
        t += step
    return last


def test_inactive_ignores_slide_commands():
    c, snd = make(real=True)
    assert hold(c, "PEACE", 0, 2) is None and snd.keys == []


def test_open_palm_requires_hold_then_activates():
    c, _ = make()
    hold(c, "OPEN_PALM", 0, 0.5)
    assert not c.active
    hold(c, "OPEN_PALM", 0.6, 1.0)
    assert c.active


def test_open_palm_blocked_while_hand_moving():
    c, _ = make()
    hold(c, "OPEN_PALM", 0, 3, speed=2.0)
    assert not c.active


def test_simulated_mode_never_sends_keys():
    c, snd = make(real=False)
    c.activate()
    rec = hold(c, "PEACE", 0, 0.3)
    assert rec.action == config.NEXT_SLIDE and rec.mode == "SIMULATED" and not rec.sent
    assert snd.keys == []


def test_real_mode_sends_once_while_held():
    c, snd = make(real=True)
    c.activate()
    hold(c, "PEACE", 0, 5.0)
    assert snd.keys == ["right"]


def test_retrigger_after_release_and_cooldown():
    c, snd = make(real=True)
    c.activate()
    hold(c, "PEACE", 0, 0.3)
    c.handle_static(None, 0.4); c.handle_static(None, 1.2)
    hold(c, "PEACE", 1.3, 0.3)
    assert snd.keys == ["right", "right"]


def test_flicker_does_not_retrigger_held_gesture():
    c, snd = make(real=True)
    c.activate()
    hold(c, "PEACE", 0, 0.2)
    t = 0.3
    for _ in range(40):
        c.handle_static(None, t); t += 0.05
        c.handle_static("PEACE", t); t += 0.05
    assert snd.keys == ["right"]


def test_cooldown_blocks_different_gesture():
    c, snd = make(real=True)
    c.activate()
    hold(c, "PEACE", 0, 0.1)
    hold(c, "POINTING", 0.2, 0.3)
    assert snd.keys == ["right"]
    hold(c, "POINTING", 1.5, 0.3)
    assert snd.keys == ["right", "left"]


def test_fist_deactivates():
    c, _ = make()
    c.activate()
    c.handle_static("FIST", 0.0)
    assert not c.active


def test_confirm_never_sends_key():
    c, snd = make(real=True)
    c.activate()
    rec = hold(c, "THUMBS_UP", 0, 0.2)
    assert rec.action == config.CONFIRM and snd.keys == []


def test_emergency_stop_latches_and_disables_real_mode():
    c, snd = make(real=True)
    c.activate()
    c.emergency_stop()
    assert not c.active and not c.real_mode and c.estop
    hold(c, "OPEN_PALM", 0, 3)
    assert not c.active
    c.set_real_mode(True)
    assert not c.real_mode
    c.reset_estop()
    assert c.activate()


def test_swipe_executes_when_active_and_suppresses_static():
    c, snd = make(real=True)
    c.activate()
    rec = c.handle_swipe(config.SWIPE_RIGHT, 0.0)
    assert rec.action == config.NEXT_SLIDE and snd.keys == ["right"]
    hold(c, "POINTING", 0.1, 0.3)
    assert snd.keys == ["right"]


def test_swipe_ignored_when_inactive():
    c, snd = make(real=True)
    assert c.handle_swipe(config.SWIPE_LEFT, 0.0) is None and snd.keys == []


def test_fast_hand_blocks_static_command():
    c, snd = make(real=True)
    c.activate()
    hold(c, "PEACE", 0, 1.0, speed=2.0)
    assert snd.keys == []


def test_sender_failure_falls_back_to_safe_mode():
    s = config.Settings(real_mode=True)
    c = CommandController(s, FakeSender(fail=True), focus_fn=lambda t: True)
    c.activate()
    rec = hold(c, "PEACE", 0, 0.2)
    assert not rec.sent and not c.real_mode and "FAILED" in rec.note


def test_custom_keys_and_history_timestamps():
    c, snd = make(real=True, next_key="page_down")
    c.activate()
    hold(c, "PEACE", 0, 0.2)
    assert snd.keys == ["page_down"]
    assert all(r.timestamp for r in c.history)


def test_no_hand_resets_activation_hold():
    c, _ = make()
    hold(c, "OPEN_PALM", 0, 0.8)
    c.handle_no_hand(0.9)
    hold(c, "OPEN_PALM", 1.0, 0.5)
    assert not c.active
