from src.dynamic_gestures import SwipeDetector
import config


def feed(det, xs, y=0.5, dt=0.03, t0=0.0):
    out = None
    for i, x in enumerate(xs):
        r = det.update(t0 + i * dt, (x, y))
        out = r or out
    return out


def line(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def test_swipe_right_and_left():
    assert feed(SwipeDetector(), line(0.3, 0.7, 12)) == config.SWIPE_RIGHT
    assert feed(SwipeDetector(), line(0.7, 0.3, 12)) == config.SWIPE_LEFT


def test_small_movement_and_jitter_ignored():
    d = SwipeDetector()
    assert feed(d, [0.5, 0.51, 0.5, 0.49, 0.5, 0.51, 0.5, 0.52] * 3) is None


def test_too_slow_is_ignored():
    assert feed(SwipeDetector(), line(0.3, 0.7, 40), dt=0.05) is None


def test_mostly_vertical_motion_ignored():
    d = SwipeDetector()
    out = None
    for i in range(12):
        out = d.update(i * 0.03, (0.3 + 0.4 * i / 11 * 0.5, 0.2 + 0.7 * i / 11)) or out
    assert out is None


def test_single_frame_jump_is_not_a_swipe():
    d = SwipeDetector()
    assert d.update(0.0, (0.2, 0.5)) is None
    assert d.update(0.03, (0.8, 0.5)) is None


def test_cooldown_blocks_second_swipe():
    d = SwipeDetector(cooldown_s=1.0)
    assert feed(d, line(0.3, 0.7, 12)) == config.SWIPE_RIGHT
    assert feed(d, line(0.3, 0.7, 12), t0=0.5) is None
    assert feed(d, line(0.3, 0.7, 12), t0=2.0) == config.SWIPE_RIGHT


def test_tracking_loss_resets_window():
    d = SwipeDetector()
    feed(d, line(0.3, 0.5, 6))
    d.update(0.2, None)
    assert feed(d, line(0.5, 0.7, 6), t0=0.25) is None


def test_gap_in_tracking_breaks_continuity():
    d = SwipeDetector()
    feed(d, line(0.3, 0.5, 6), dt=0.03)
    assert feed(d, line(0.5, 0.7, 6), t0=1.0) is None


def test_zigzag_not_monotonic():
    d = SwipeDetector()
    xs = [0.3, 0.45, 0.3, 0.5, 0.32, 0.55, 0.35, 0.6, 0.38, 0.7]
    assert feed(d, xs) is None
