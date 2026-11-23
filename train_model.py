"""Train + evaluate the Random Forest gesture classifier.

    python train_model.py                      # train on data/gesture_samples.csv
    python train_model.py --compare-svm        # also train an SVM for comparison
Outputs: models/gesture_rf.joblib, models/class_labels.json, results/metrics.json,
         results/confusion_matrix.(png|csv), results/classification_report.txt, ...
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import config
from src import data_utils as du
from src import evaluation as ev
from src.feature_extractor import FEATURE_NAMES


def run(data_path=config.DATASET_PATH, model_path=config.MODEL_PATH, results_dir=config.RESULTS_DIR,
        labels_path=config.LABELS_PATH, seed=42, test_size=0.3, min_per_class=30,
        n_estimators=200, compare_svm=False, refit_all=False, quiet=False) -> dict:
    log = (lambda *a: None) if quiet else print
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df = du.load_dataset(data_path)
    rep = du.validate_dataset(df, config.GESTURE_CLASSES, min_per_class)
    for w in rep["warnings"]:
        log("WARNING:", w)
    if rep["errors"]:
        raise du.DatasetError("Dataset not ready:\n  - " + "\n  - ".join(rep["errors"]))
    df = rep["clean"]
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["label"].to_numpy()
    split = du.split_dataset(df, seed=seed, test_size=test_size)
    tr, te = split["train_idx"], split["test_idx"]
    if len(te) == 0 or len(tr) == 0:
        raise du.DatasetError("Split produced an empty train or test set; collect more data.")
    labels = [c for c in config.GESTURE_CLASSES if c in set(y)]
    log(f"Split: {split['strategy']} | train={len(tr)} test={len(te)} | {split['notes']}")

    rf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced",
                                random_state=seed, n_jobs=-1)
    rf.fit(X[tr], y[tr])
    metrics = ev.compute_metrics(y[te], rf.predict(X[te]), labels)

    comparison = {"RandomForest": {k: metrics[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}}
    if compare_svm:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", random_state=seed))
        svm.fit(X[tr], y[tr])
        m2 = ev.compute_metrics(y[te], svm.predict(X[te]), labels)
        comparison["SVM_RBF"] = {k: m2[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}

    final = rf
    if refit_all:
        final = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced",
                                       random_state=seed, n_jobs=-1).fit(X, y)

    info = rep["info"]
    out = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_fingerprint": du.dataset_fingerprint(df),
        "dataset_path": str(data_path),
        "rows_clean": info["rows_clean"], "rows_raw": info["rows_raw"],
        "dropped_invalid": info["dropped_invalid"], "dropped_duplicates": info["dropped_duplicates"],
        "class_counts": info["class_counts"], "n_users": info["n_users"], "n_sessions": info["n_sessions"],
        "split_strategy": split["strategy"], "split_notes": split["notes"],
        "preliminary": bool(split["preliminary"]), "n_train": int(len(tr)), "n_test": int(len(te)),
        "test_class_counts": {c: int((y[te] == c).sum()) for c in labels},
        "seed": seed, "test_size": test_size, "n_estimators": n_estimators,
        "deployed_model_refit_on_all_data": bool(refit_all),
        "feature_schema_version": config.FEATURE_SCHEMA_VERSION, "n_features": len(FEATURE_NAMES),
        "metrics_note": ("PRELIMINARY: single user/session; metrics are optimistic."
                         if split["preliminary"] else "Held-out groups; still limited by dataset size."),
        "model_comparison": comparison if compare_svm else None,
        **{k: v for k, v in metrics.items() if k != "report_text"},
    }
    (results_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    (results_dir / "classification_report.txt").write_text(metrics["report_text"])
    np.savetxt(results_dir / "confusion_matrix.csv", np.array(metrics["confusion_matrix"], dtype=int),
               fmt="%d", delimiter=",", header=",".join(labels), comments="")
    ev.save_confusion_plot(metrics["confusion_matrix"], labels, results_dir / "confusion_matrix.png",
                           f"Confusion matrix ({split['strategy']}, n_test={len(te)})")
    ev.save_class_distribution_plot(info["class_counts"], results_dir / "class_distribution.png")
    if compare_svm:
        (results_dir / "model_comparison.json").write_text(json.dumps(comparison, indent=2))
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "classes": [str(c) for c in final.classes_],
                 "feature_names": FEATURE_NAMES, "schema_version": config.FEATURE_SCHEMA_VERSION,
                 "meta": {"dataset_fingerprint": out["dataset_fingerprint"], "trained_at": out["trained_at"],
                          "split_strategy": split["strategy"]}}, model_path)
    Path(labels_path).write_text(json.dumps([str(c) for c in final.classes_]))
    log(metrics["report_text"])
    log(f"Accuracy {metrics['accuracy']:.3f} | {out['metrics_note']}")
    log(f"Saved model -> {model_path}; results -> {results_dir}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATASET_PATH))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--min-per-class", type=int, default=30)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--compare-svm", action="store_true")
    ap.add_argument("--refit-all", action="store_true",
                    help="after evaluating, refit the deployed model on ALL data (metrics then describe the train-only model)")
    a = ap.parse_args()
    try:
        run(a.data, seed=a.seed, test_size=a.test_size, min_per_class=a.min_per_class,
            n_estimators=a.n_estimators, compare_svm=a.compare_svm, refit_all=a.refit_all)
    except du.DatasetError as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
