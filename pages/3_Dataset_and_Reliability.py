import subprocess
import sys

import pandas as pd
import streamlit as st

import config
from src import data_utils as du
from src import metrics as mt
from src import ui

ui.setup("Dataset and Reliability")
ui.release_camera()
st.title("Dataset and Reliability")
s = ui.get_settings()

st.header("1 · Data collection")
counts = du.class_counts()
if counts:
    st.bar_chart(pd.Series(counts, name="samples"))
else:
    st.info("No dataset yet.")
st.markdown("""
**How to collect** (opens an OpenCV window; the camera is free because it was released):
1. Keys `1`–`6` choose the gesture, `SPACE` starts/stops recording, `Q` quits.
2. For each gesture record ~150+ samples. Move the hand slightly, change distance and tilt a little.
3. `NO_GESTURE` = relaxed or half-open hand and transition poses (the hand must still be visible).
4. Use a different **user id** per person and a new **session id** per sitting; held-out evaluation
   needs more than one user or session.
""")
c1, c2 = st.columns(2)
user = c1.text_input("User id", "user1")
sess = c2.text_input("Session id", pd.Timestamp.now().strftime("sess_%Y%m%d_%H%M"))
if st.button("Launch collection window"):
    flags = subprocess.CREATE_NEW_CONSOLE if sys.platform.startswith("win") else 0
    subprocess.Popen([sys.executable, "collect_data.py", "--user", user, "--session", sess],
                     cwd=config.PROJECT_ROOT, creationflags=flags)
    st.success("Launched. Finish recording in that window, then reload this page.")

st.header("2 · Safety and smoothing settings")
a, b, c = st.columns(3)
s.confidence_threshold = a.slider("Confidence threshold", 0.3, 0.99, s.confidence_threshold, 0.01)
s.consistent_frames = b.slider("Consistent frames required", 1, 20, s.consistent_frames)
s.cooldown_s = c.slider("Command cooldown (s)", 0.2, 4.0, s.cooldown_s, 0.1)
a, b, c = st.columns(3)
s.release_s = a.slider("Release time before re-trigger (s)", 0.1, 2.0, s.release_s, 0.1)
s.activation_hold_s = b.slider("OPEN_PALM hold to activate (s)", 0.3, 3.0, s.activation_hold_s, 0.1)
s.swipe_enabled = c.checkbox("Enable swipe gestures", s.swipe_enabled)
a, b, c = st.columns(3)
s.swipe_min_displacement = a.slider("Swipe min displacement (frame widths)", 0.1, 0.6, s.swipe_min_displacement, 0.01)
s.next_key = b.selectbox("Next-slide key", config.ALLOWED_KEYS, config.ALLOWED_KEYS.index(s.next_key))
s.prev_key = c.selectbox("Previous-slide key", config.ALLOWED_KEYS, config.ALLOWED_KEYS.index(s.prev_key))
s.target_window_title = st.text_input("Target window title contains (optional, Windows)", s.target_window_title,
                                      help="e.g. 'PowerPoint' or 'Slide Show'. Empty = keys go to the focused window.")
st.table(pd.DataFrame([{"Gesture": g, "Action": a_} for g, a_ in config.GESTURE_TO_ACTION.items()]))

st.header("3 · Reliability results")
st.markdown("Record observations with `python test_reliability.py` (see README for the protocol). "
            "Only logged, measured trials appear here.")
if not config.RELIABILITY_LOG_PATH.exists():
    st.info("No reliability log yet.")
else:
    log = pd.read_csv(config.RELIABILITY_LOG_PATH)
    sm = mt.reliability_summary(log)
    fmt = lambda v, p=True: "n/a" if v is None else (f"{v*100:.1f}%" if p else f"{v:.1f}")
    r1 = st.columns(4)
    r1[0].metric("Command trials", sm["n_command_trials"])
    r1[1].metric("Recognition rate", fmt(sm.get("recognition_rate")))
    r1[2].metric("Command success rate", fmt(sm.get("command_success_rate")))
    r1[3].metric("Missed-command rate", fmt(sm.get("missed_command_rate")))
    r2 = st.columns(4)
    r2[0].metric("Incorrect-command rate", fmt(sm.get("incorrect_command_rate")))
    r2[1].metric("Repeated-command rate", fmt(sm.get("repeated_command_rate")))
    r2[2].metric("False activations / min", fmt(sm.get("false_activations_per_min"), False))
    r2[3].metric("Latency median (ms)", fmt(sm.get("latency_ms_median"), False))
    st.caption(f"Idle minutes observed: {sm['idle_minutes']:.1f} · mean FPS: {fmt(sm.get('fps_mean'), False)}. "
               "Command success ≠ classifier accuracy: it requires the intended action to execute exactly once.")
    col = st.selectbox("Break down by condition", [c for c in mt.CONDITION_COLUMNS if c in log.columns])
    bd = mt.breakdown(log, col)
    st.dataframe(bd, hide_index=True, use_container_width=True) if not bd.empty else st.caption("No command trials.")
    ib = mt.idle_breakdown(log)
    if not ib.empty:
        st.subheader("Accidental-trigger (idle) trials by scenario")
        st.dataframe(ib, hide_index=True, use_container_width=True)
    with st.expander("Raw observations"):
        st.dataframe(log, use_container_width=True)
    st.download_button("Download reliability_log.csv", log.to_csv(index=False), "reliability_log.csv", "text/csv")
