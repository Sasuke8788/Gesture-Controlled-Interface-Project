import pandas as pd
import streamlit as st

import config
from src import ui

ui.setup("Model Evaluation")
ui.release_camera()
st.title("Model Evaluation")
m = ui.read_json(config.METRICS_PATH)
if not m:
    st.info("No evaluation results yet. Collect data (`python collect_data.py`), then run "
            "`python train_model.py`. Nothing is shown until real results exist.")
    st.stop()

if m.get("preliminary"):
    st.warning(f"PRELIMINARY: {m.get('metrics_note')}")
else:
    st.success(m.get("metrics_note", ""))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset size (clean)", m["rows_clean"])
c2.metric("Train / test", f"{m['n_train']} / {m['n_test']}")
c3.metric("Users / sessions", f"{m['n_users']} / {m['n_sessions']}")
c4.metric("Accuracy (test)", f"{m['accuracy']:.3f}")
c1, c2, c3 = st.columns(3)
c1.metric("Macro precision", f"{m['macro_precision']:.3f}")
c2.metric("Macro recall", f"{m['macro_recall']:.3f}")
c3.metric("Macro F1", f"{m['macro_f1']:.3f}")

st.subheader("Methodology")
st.markdown(f"""
- **Split strategy:** `{m['split_strategy']}` — {m['split_notes']}
- Random Forest, {m['n_estimators']} trees, `class_weight=balanced`, seed {m['seed']}, {m['n_features']} features
  (schema v{m['feature_schema_version']}).
- Dataset fingerprint: `{m['dataset_fingerprint']}` · trained at {m['trained_at']}
- Dropped before training: {m['dropped_invalid']} invalid rows, {m['dropped_duplicates']} duplicates.
- Deployed model refit on all data: **{m['deployed_model_refit_on_all_data']}**
- Why not a random row split? Adjacent webcam frames are near-identical, so random splitting leaks
  almost-duplicates into the test set and inflates accuracy.
- Random Forest probabilities are vote fractions, not calibrated probabilities of correctness.
""")

left, right = st.columns(2)
with left:
    st.subheader("Samples per class")
    st.bar_chart(pd.Series(m["class_counts"], name="samples"))
    st.caption("Test samples per class: " + ", ".join(f"{k}={v}" for k, v in m["test_class_counts"].items()))
with right:
    st.subheader("Confusion matrix")
    png = config.RESULTS_DIR / "confusion_matrix.png"
    if png.exists():
        st.image(str(png), use_container_width=True)

st.subheader("Per-class precision / recall / F1")
st.dataframe(pd.DataFrame(m["per_class"]).T.rename_axis("class"), use_container_width=True)
rpt = config.RESULTS_DIR / "classification_report.txt"
if rpt.exists():
    with st.expander("Classification report (text)"):
        st.code(rpt.read_text())
if m.get("model_comparison"):
    st.subheader("Model comparison (same split)")
    st.dataframe(pd.DataFrame(m["model_comparison"]).T, use_container_width=True)
ext = sorted(config.RESULTS_DIR.glob("eval_*.json"))
if ext:
    st.subheader("External / held-out evaluations")
    for p in ext:
        e = ui.read_json(p)
        st.write(f"**{e['name']}** — n={e['n_samples']}, accuracy {e['accuracy']:.3f}, macro-F1 {e['macro_f1']:.3f}")
