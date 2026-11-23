"""Reliability metric definitions computed from the observation log (results/reliability_log.csv).

Definitions (command trials = trials where the user INTENDS one command):
  recognition_rate     = trials whose intended gesture was recognised / command trials
                         (static: first stable prediction == intended; swipe: swipe detected).
  command_success_rate = trials where the intended action executed exactly once AND no other
                         action executed / command trials.  NOT the same as classifier accuracy.
  missed_command_rate  = trials where no action executed at all / command trials.
  incorrect_command_rate = trials where any non-intended action executed / command trials.
  repeated_command_rate  = trials where the intended action executed more than once
                           (held-gesture re-triggering) and nothing wrong executed / command trials.
  false_activations_per_min = actions executed during IDLE trials (user not trying to issue a
                         command) / total idle minutes.
  latency_ms           = time from the first raw correct prediction to the stable (executed)
                         gesture, i.e. the system's smoothing delay, excluding human reaction time.
  fps                  = mean processing frames per second during trials.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOG_COLUMNS = [
    "timestamp", "user_id", "session_id", "trial_type", "scenario", "intended_gesture",
    "intended_action", "mode", "distance", "lighting", "orientation", "speed", "background",
    "duration_s", "first_stable_gesture", "recognized_correctly", "latency_ms",
    "time_to_command_s", "executed_actions", "n_executed", "n_correct", "n_wrong",
    "outcome", "mean_fps", "mean_proc_ms", "notes",
]
CONDITION_COLUMNS = ["user_id", "distance", "lighting", "orientation", "speed", "background",
                     "mode", "scenario"]


def classify_outcome(n_correct: int, n_wrong: int) -> str:
    if n_wrong > 0:
        return "incorrect"
    if n_correct > 1:
        return "repeated"
    if n_correct == 0:
        return "missed"
    return "success"


def _rate(num: float, den: float):
    return float(num) / den if den else None


def reliability_summary(df: pd.DataFrame) -> dict:
    """Aggregate metrics. Missing values are None (never invented)."""
    if df is None or df.empty:
        return {"n_trials": 0}
    cmd = df[df["trial_type"] == "command"]
    idle = df[df["trial_type"] == "idle"]
    n = len(cmd)
    out = {"n_trials": int(len(df)), "n_command_trials": int(n), "n_idle_trials": int(len(idle))}
    out["recognition_rate"] = _rate(cmd["recognized_correctly"].astype(bool).sum(), n)
    for key, outcome in (("command_success_rate", "success"), ("missed_command_rate", "missed"),
                         ("incorrect_command_rate", "incorrect"),
                         ("repeated_command_rate", "repeated")):
        out[key] = _rate((cmd["outcome"] == outcome).sum(), n)
    idle_min = idle["duration_s"].sum() / 60.0 if len(idle) else 0.0
    out["idle_minutes"] = float(idle_min)
    out["false_activations"] = int(idle["n_executed"].sum()) if len(idle) else 0
    out["false_activations_per_min"] = _rate(out["false_activations"], idle_min)
    lat = pd.to_numeric(cmd["latency_ms"], errors="coerce").dropna()
    out["latency_ms_mean"] = float(lat.mean()) if len(lat) else None
    out["latency_ms_median"] = float(lat.median()) if len(lat) else None
    out["latency_ms_p95"] = float(np.percentile(lat, 95)) if len(lat) else None
    fps = pd.to_numeric(df["mean_fps"], errors="coerce").dropna()
    out["fps_mean"] = float(fps.mean()) if len(fps) else None
    return out


def breakdown(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Command-trial success/recognition rates split by a condition column."""
    cmd = df[df["trial_type"] == "command"] if df is not None and not df.empty else df
    if cmd is None or cmd.empty or column not in cmd.columns:
        return pd.DataFrame()
    rows = []
    for val, g in cmd.groupby(column, dropna=False):
        rows.append({column: val, "trials": len(g),
                     "recognition_rate": g["recognized_correctly"].astype(bool).mean(),
                     "command_success_rate": (g["outcome"] == "success").mean(),
                     "missed_rate": (g["outcome"] == "missed").mean(),
                     "incorrect_rate": (g["outcome"] == "incorrect").mean(),
                     "repeated_rate": (g["outcome"] == "repeated").mean()})
    return pd.DataFrame(rows)


def idle_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    idle = df[df["trial_type"] == "idle"] if df is not None and not df.empty else df
    if idle is None or idle.empty:
        return pd.DataFrame()
    rows = []
    for val, g in idle.groupby("scenario", dropna=False):
        mins = g["duration_s"].sum() / 60
        rows.append({"scenario": val, "trials": len(g), "minutes": round(mins, 2),
                     "false_activations": int(g["n_executed"].sum()),
                     "per_minute": g["n_executed"].sum() / mins if mins else None})
    return pd.DataFrame(rows)
