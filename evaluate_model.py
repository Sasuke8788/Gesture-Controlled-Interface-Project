"""Evaluate the saved model on a labelled CSV (ideally a NEW user/session never used in training).

    python evaluate_model.py --data data/other_person.csv --name other_person
Collect that file with:  python collect_data.py --user bob   (then copy/move the rows to a separate CSV)
or filter by user:       python evaluate_model.py --holdout-user bob   (only valid if bob was NOT in training)
"""
import argparse
import json
import sys

import config
from src import data_utils as du
from src import evaluation as ev
from src.feature_extractor import FEATURE_NAMES
from src.gesture_classifier import GestureClassifier, ModelNotAvailableError


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATASET_PATH))
    ap.add_argument("--holdout-user", default=None, help="evaluate only rows of this user_id")
    ap.add_argument("--name", default="external")
    a = ap.parse_args()
    try:
        clf = GestureClassifier.load()
        df = du.load_dataset(a.data)
    except (ModelNotAvailableError, du.DatasetError) as exc:
        print(exc)
        return 1
    rep = du.validate_dataset(df, config.GESTURE_CLASSES, min_per_class=1)
    df = rep["clean"]
    if a.holdout_user:
        df = df[df["user_id"].astype(str) == a.holdout_user]
    if df.empty:
        print("No rows to evaluate after filtering.")
        return 1
    y_pred = clf.model.predict(df[FEATURE_NAMES].to_numpy(dtype=float))
    labels = [c for c in config.GESTURE_CLASSES if c in set(df["label"])]
    m = ev.compute_metrics(df["label"].to_numpy(), y_pred, labels)
    out_dir = config.RESULTS_DIR
    ev.save_confusion_plot(m["confusion_matrix"], labels, out_dir / f"confusion_{a.name}.png",
                           f"Confusion matrix: {a.name} (n={len(df)})")
    saved = {k: v for k, v in m.items() if k != "report_text"} | {
        "name": a.name, "n_samples": int(len(df)), "dataset_fingerprint": du.dataset_fingerprint(df),
        "note": "Valid as held-out only if these rows were not in the training data."}
    (out_dir / f"eval_{a.name}.json").write_text(json.dumps(saved, indent=2))
    print(m["report_text"])
    print(f"Accuracy {m['accuracy']:.3f} on {len(df)} samples -> results/eval_{a.name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
