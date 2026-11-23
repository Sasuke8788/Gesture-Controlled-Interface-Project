"""Live reliability trials -> results/reliability_log.csv (REAL observations only).

    python test_reliability.py --user alice --distance 1m --lighting bright --mode simulate
Keys in the window:
  1 OPEN_PALM  2 FIST  3 PEACE  4 POINTING  5 THUMBS_UP  6 SWIPE_RIGHT  7 SWIPE_LEFT
  i  idle trial (move naturally, do NOT try to issue commands; any command = false activation)
  q  quit
Each command trial: 2 s countdown, then a window (--trial-seconds) in which you perform the
intended gesture once. Idle trials run for --idle-seconds with the controller ACTIVE (worst case).
Use --mode real only when a slideshow window is focused; the default simulates (log only).
"""
import argparse
import time

import cv2
import mediapipe
import pandas as pd

import config
from src import metrics as mt
from src.command_controller import CommandController
from src.gesture_classifier import GestureClassifier, ModelNotAvailableError
from src.hand_detector import HandDetector, HandModelMissingError
from src.pipeline import GesturePipeline, draw_overlay
from src.webcam_manager import WebcamError, WebcamManager

KEYMAP = {"1": "OPEN_PALM", "2": "FIST", "3": "PEACE", "4": "POINTING", "5": "THUMBS_UP",
          "6": config.SWIPE_RIGHT, "7": config.SWIPE_LEFT}


def append_log(row: dict, path=config.RELIABILITY_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row]).reindex(columns=mt.LOG_COLUMNS)
    df.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)


def summarize_trial(ctrl, intended, trial_type, t_start, t_raw, t_stable, recognized, fps_l, proc_l,
                    duration, meta) -> dict:
    executed = [r for r in ctrl.history if r.source != "UI"]
    actions = [r.action for r in executed]
    intended_action = config.GESTURE_TO_ACTION.get(intended) if intended else None
    if trial_type == "command":
        n_correct = sum(a == intended_action for a in actions)
        n_wrong = len(actions) - n_correct
        outcome = mt.classify_outcome(n_correct, n_wrong)
    else:
        n_correct, n_wrong, outcome = 0, len(actions), "false_activation" if actions else "clean"
    latency = (t_stable - t_raw) * 1000 if (t_raw is not None and t_stable is not None) else None
    return {**meta, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "trial_type": trial_type,
            "intended_gesture": intended, "intended_action": intended_action,
            "duration_s": round(duration, 2), "recognized_correctly": bool(recognized),
            "latency_ms": None if latency is None else round(latency, 1),
            "time_to_command_s": None if t_stable is None else round(t_stable - t_start, 2),
            "executed_actions": ";".join(actions), "n_executed": len(actions),
            "n_correct": n_correct, "n_wrong": n_wrong, "outcome": outcome,
            "mean_fps": round(sum(fps_l) / len(fps_l), 1) if fps_l else None,
            "mean_proc_ms": round(sum(proc_l) / len(proc_l), 1) if proc_l else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="user1")
    ap.add_argument("--session", default=time.strftime("rel_%Y%m%d_%H%M%S"))
    ap.add_argument("--distance", default="unspecified")
    ap.add_argument("--lighting", default="unspecified")
    ap.add_argument("--orientation", default="upright")
    ap.add_argument("--speed", default="normal")
    ap.add_argument("--background", default="unspecified")
    ap.add_argument("--scenario", default="normal_use",
                    help="e.g. natural_movement, similar_gesture_switch, long_hold, tracking_loss, "
                         "out_of_frame, poor_lighting")
    ap.add_argument("--mode", choices=["simulate", "real"], default="simulate")
    ap.add_argument("--trial-seconds", type=float, default=4.0)
    ap.add_argument("--idle-seconds", type=float, default=30.0)
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    try:
        clf = GestureClassifier.load()
        detector = HandDetector()
    except (ModelNotAvailableError, HandModelMissingError) as exc:
        print(exc)
        return 1
    settings = config.Settings(camera_index=a.camera, real_mode=(a.mode == "real"))
    ctrl = CommandController(settings)
    pipe = GesturePipeline(settings, detector, clf, ctrl)
    meta = {"user_id": a.user, "session_id": a.session, "mode": a.mode.upper(), "distance": a.distance,
            "lighting": a.lighting, "orientation": a.orientation, "speed": a.speed,
            "background": a.background, "scenario": a.scenario}
    cam = WebcamManager(a.camera)
    try:
        cam.open()
    except WebcamError as exc:
        print(exc)
        detector.close()
        return 1
    print(__doc__)
    state, trial = "wait", None
    win = "Reliability trials (q quit)"
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            res = pipe.process(frame, now)
            draw_overlay(frame, res, ctrl)
            msg = "Press 1-7 for a command trial, i for idle trial, q to quit"
            if state == "countdown":
                left = trial["t_go"] - now
                msg = f"GET READY: {trial['intended'] or 'IDLE'} in {left:.0f}s"
                if left <= 0:
                    if trial["type"] == "command" and trial["intended"] == "OPEN_PALM":
                        ctrl.deactivate("UI")
                    else:
                        ctrl.activate("UI")
                    ctrl.history.clear()
                    pipe.stability.reset(); pipe.swipe.reset()
                    trial.update(t_start=now, t_raw=None, t_stable=None, recognized=False, fps=[], proc=[])
                    state = "run"
            elif state == "run":
                dur = a.idle_seconds if trial["type"] == "idle" else a.trial_seconds
                msg = f"DO IT NOW: {trial['intended'] or 'move naturally'} ({dur-(now-trial['t_start']):.0f}s)"
                it = trial["intended"]
                if trial["type"] == "command":
                    if it in (config.SWIPE_LEFT, config.SWIPE_RIGHT):
                        if res.swipe == it and not trial["recognized"]:
                            trial["recognized"], trial["t_stable"] = True, now
                    else:
                        if res.raw_label == it and trial["t_raw"] is None:
                            trial["t_raw"] = now
                        if res.stable_gesture == it and not trial["recognized"]:
                            trial["recognized"], trial["t_stable"] = True, now
                    if res.stable_gesture and trial.get("first_stable") is None:
                        trial["first_stable"] = res.stable_gesture
                trial["fps"].append(res.fps)
                trial["proc"].append(res.proc_ms)
                if now - trial["t_start"] >= dur:
                    row = summarize_trial(ctrl, trial["intended"], trial["type"], trial["t_start"],
                                          trial["t_raw"], trial["t_stable"], trial["recognized"],
                                          trial["fps"], trial["proc"], now - trial["t_start"], meta)
                    row["first_stable_gesture"] = trial.get("first_stable")
                    append_log(row)
                    print(f"{row['trial_type']:7s} {str(row['intended_gesture']):12s} -> {row['outcome']}"
                          f" executed={row['executed_actions'] or '-'}")
                    state, trial = "wait", None
            cv2.putText(frame, msg, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if state == "wait" and chr(k) in KEYMAP:
                trial = {"type": "command", "intended": KEYMAP[chr(k)], "t_go": time.monotonic() + 2}
                state = "countdown"
            elif state == "wait" and k == ord("i"):
                trial = {"type": "idle", "intended": None, "t_go": time.monotonic() + 2}
                state = "countdown"
    except WebcamError as exc:
        print(exc)
    finally:
        ctrl.deactivate("UI")
        cam.release()
        detector.close()
        cv2.destroyAllWindows()
    print(f"Log: {config.RELIABILITY_LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
