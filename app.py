"""Home page. Run from the project root:  streamlit run app.py"""
import mediapipe
import streamlit as st

import config
from src import data_utils as du
from src import ui

ui.setup("Gesture-Controlled Interface")
ui.release_camera()
st.title("🖐️ Gesture-Controlled Presentation Interface")
st.caption("PS 15 · MediaPipe hand landmarks → feature extraction → Random Forest → safe command controller")

counts = du.class_counts()
c1, c2, c3, c4 = st.columns(4)
checks = [
    ("MediaPipe hand model", config.HAND_MODEL_PATH.exists(), "run `python download_model.py`"),
    ("Dataset", bool(counts) and sum(counts.values()) > 0, "run `python collect_data.py`"),
    ("Trained model", config.MODEL_PATH.exists(), "run `python train_model.py`"),
    ("Evaluation results", config.METRICS_PATH.exists(), "created by train_model.py"),
]
for col, (name, ok, hint) in zip((c1, c2, c3, c4), checks):
    col.markdown(ui.card(name, "Ready" if ok else "Missing", "ok" if ok else "bad"), unsafe_allow_html=True)
    if not ok:
        col.caption(hint)

st.subheader("Workflow")
st.markdown("""
1. **Download** the hand model once: `python download_model.py`
2. **Collect** samples for all 6 classes: `python collect_data.py --user <name>` (or from page 3)
3. **Train**: `python train_model.py` → see **Model Evaluation**
4. **Run live** on **Live Detection** (starts in *safe simulation*; real keys need explicit opt-in)
5. **Test reliability**: `python test_reliability.py` → see **Dataset and Reliability**
""")
st.info("Pages are in the sidebar. The camera is released whenever you leave the Live Detection page.")
