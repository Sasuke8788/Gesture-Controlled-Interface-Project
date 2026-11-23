import base64
import time

import cv2
import mediapipe
import pandas as pd
import streamlit as st

import config
from src import ui
from src.webcam_manager import WebcamError

ui.setup("Live Gesture Detection")
st.title("Live Gesture Detection")
settings = ui.get_settings()
ctrl = ui.get_controller()
st.session_state.setdefault("running", False)

detector, classifier, errors, warnings = ui.load_assets(settings)
for e in errors:
    st.error(e)
for w in warnings:
    st.warning(w)

b1, b2, b3, b4, b5 = st.columns(5)
if b1.button("▶ Start camera", disabled=detector is None, use_container_width=True):
    st.session_state.running = True
if b2.button("⏹ Stop camera", use_container_width=True):
    ui.release_camera()
if b3.button("✅ Activate control", use_container_width=True):
    ctrl.activate("UI")
if b4.button("⏸ Disable control", use_container_width=True):
    ctrl.deactivate("UI")
if b5.button("🛑 EMERGENCY STOP", type="primary", use_container_width=True):
    ctrl.emergency_stop()

t1, t2 = st.columns([2, 3])
real = t1.toggle("Send REAL keyboard commands", value=ctrl.real_mode, disabled=ctrl.estop,
                 help="OFF = safe simulation (commands are only logged).")
if real != ctrl.real_mode:
    ctrl.set_real_mode(real)
if ctrl.estop:
    t2.error("Emergency stop is latched. Gestures cannot activate control.")
    if t2.button("Reset emergency stop"):
        ctrl.reset_estop()
elif ctrl.real_mode:
    t2.warning("REAL mode: keys go to the focused window. Focus your slideshow window "
               "(or set a target window title on page 3).")
else:
    t2.success("Safe simulation mode: no keyboard events are sent.")

left, right = st.columns([3, 2])
video_ph = left.empty()
status_ph = right.empty()
flash_ph = right.empty()
hist_ph = right.empty()
with right.expander("Gesture → command mapping", expanded=False):
    st.table(pd.DataFrame([{"Gesture": g, "Action": a} for g, a in config.GESTURE_TO_ACTION.items()]))


def render_status(res):
    state = "ACTIVE" if ctrl.active else "DISABLED"
    hand = f"{res.num_hands} hand(s)" if res.hand_detected else "no hand"
    html = (ui.card("Gesture control", state, "ok" if ctrl.active else "bad") +
            ui.card("Raw prediction", f"{res.raw_label or '–'} ({res.confidence:.2f})") +
            ui.card("Stable gesture", res.stable_gesture or "–") +
            ui.card("Hand tracking", hand, "ok" if res.hand_detected else "warn") +
            ui.card("Speed", f"{res.fps:.0f} FPS · {res.proc_ms:.0f} ms/frame"))
    if res.message:
        html += f'<div class="card"><div class="k">Note</div><div class="warn">{res.message}</div></div>'
    status_ph.markdown(html, unsafe_allow_html=True)
    last = ctrl.last_action
    if last and time.monotonic() - ctrl.last_flash_time < 1.5:
        flash_ph.markdown(f'<div class="flash">EXECUTED: {last.action} ({last.mode}) – {last.source}</div>',
                          unsafe_allow_html=True)
    else:
        flash_ph.empty()
    rows = [r.to_dict() for r in reversed(ctrl.history[-8:])]
    hist_ph.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True) if rows \
        else hist_ph.caption("No commands yet.")


def render_frame(frame_bgr, jpeg_quality: int = 55) -> None:
    h, w = frame_bgr.shape[:2]
    if w > 640:
        scale = 640 / w
        frame_bgr = cv2.resize(frame_bgr, (640, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame_bgr,
                                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
                                 cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    if not ok:
        return
    jpg_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    video_ph.markdown(
        f'<img src="data:image/jpeg;base64,{jpg_b64}" '
        'style="width:100%;border-radius:8px;display:block;" />',
        unsafe_allow_html=True,
    )


@st.fragment(run_every=0.033)
def live_camera_fragment():
    if not st.session_state.running or detector is None:
        video_ph.info("Camera is stopped. Press **Start camera** (allow camera access if Windows asks). "
                      "Control is DISABLED until you press Activate or hold OPEN_PALM.")
        return

    pipe = ui.get_pipeline(settings, detector, classifier)
    cam = ui.get_camera(settings)
    if not cam.is_open:
        try:
            cam.open()
        except WebcamError as exc:
            st.error(str(exc))
            ui.release_camera()
            return

    try:
        frame = cam.read_latest()
        if frame is None:
            return
        frame = cv2.flip(frame, 1)
        res = pipe.process(frame)
        now = time.monotonic()
        last_render = st.session_state.get("live_last_render", 0.0)
        if now - last_render >= 0.083:
            render_frame(frame)
            st.session_state.live_last_render = now
        if res.command:
            render_status(res)
        else:
            n = st.session_state.get("live_status_frame", 0) + 1
            st.session_state.live_status_frame = n
            if n % 3 == 0:
                render_status(res)
    except WebcamError as exc:
        st.error(f"Camera error: {exc}")
        ui.release_camera()


live_camera_fragment()
