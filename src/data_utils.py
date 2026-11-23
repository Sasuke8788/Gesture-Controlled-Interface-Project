"""Dataset I/O, validation, versioning and leakage-aware splitting."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import config
from src.feature_extractor import FEATURE_NAMES

META_COLUMNS = ["timestamp", "user_id", "session_id", "label", "handedness"]
DATA_COLUMNS = META_COLUMNS + FEATURE_NAMES


class DatasetError(RuntimeError):
    pass


def append_samples(rows: Iterable[dict], path: Path = config.DATASET_PATH) -> int:
    """Append labelled feature rows (dicts with META_COLUMNS + feature columns). Returns count."""
    path = Path(path)
    rows = list(rows)
    if not rows:
        return 0
    df = pd.DataFrame(rows)[DATA_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if header != DATA_COLUMNS:
            raise DatasetError(
                f"{path} was written with a different feature schema. Move it aside and "
                "re-collect, or retrain with matching code.")
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
    return len(df)


def make_row(features: np.ndarray, label: str, user_id: str, session_id: str,
             handedness: Optional[str]) -> dict:
    row = {"timestamp": time.time(), "user_id": user_id, "session_id": session_id,
           "label": label, "handedness": handedness or ""}
    row.update(dict(zip(FEATURE_NAMES, map(float, features))))
    return row


def load_dataset(path: Path = config.DATASET_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"No dataset at {path}. Run:  python collect_data.py")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DatasetError(f"Could not read {path}: {exc}") from exc
    if df.empty:
        raise DatasetError(f"{path} is empty. Collect samples first.")
    return df


def class_counts(path: Path = config.DATASET_PATH) -> dict:
    """Samples per class for display; zero for classes not yet collected. {} if no dataset."""
    try:
        df = load_dataset(path)
    except DatasetError:
        return {}
    vc = df["label"].value_counts().to_dict()
    return {c: int(vc.get(c, 0)) for c in config.GESTURE_CLASSES} | {
        k: int(v) for k, v in vc.items() if k not in config.GESTURE_CLASSES}


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Short content hash identifying the exact data an evaluation used."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df[DATA_COLUMNS], index=False).values.tobytes())
    return h.hexdigest()[:12]


def validate_dataset(df: pd.DataFrame, required_classes=None, min_per_class: int = 30) -> dict:
    """Return {'errors': [...], 'warnings': [...], 'clean': DataFrame, 'info': {...}}."""
    required = list(required_classes or config.GESTURE_CLASSES)
    errors, warnings = [], []
    missing_cols = [c for c in DATA_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}")
        return {"errors": errors, "warnings": warnings, "clean": df, "info": {}}
    df = df[DATA_COLUMNS].copy()
    n_raw = len(df)
    unknown = sorted(set(df["label"].dropna()) - set(required))
    if unknown:
        errors.append(f"Unknown labels in dataset: {unknown}")
    feat = df[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(feat.to_numpy(dtype=float)).all(axis=1) | df["label"].isna().to_numpy()
    n_bad = int(bad.sum())
    if n_bad:
        warnings.append(f"Dropped {n_bad} rows with missing/NaN/inf values or labels.")
    df = df[~bad]
    df[FEATURE_NAMES] = feat[~bad]
    n_dup = int(df.duplicated(subset=FEATURE_NAMES + ["label"]).sum())
    if n_dup:
        warnings.append(f"Dropped {n_dup} exact duplicate rows.")
        df = df.drop_duplicates(subset=FEATURE_NAMES + ["label"])
    counts = df["label"].value_counts().to_dict()
    absent = [c for c in required if counts.get(c, 0) == 0]
    if absent:
        errors.append(f"No samples for classes: {absent}")
    low = {c: counts.get(c, 0) for c in required if 0 < counts.get(c, 0) < min_per_class}
    if low:
        errors.append(f"Fewer than {min_per_class} samples for: {low}")
    if counts and max(counts.values()) > 3 * max(min(counts.values()), 1):
        warnings.append("Class imbalance > 3:1; class_weight='balanced' is used in training.")
    info = {"rows_raw": n_raw, "rows_clean": len(df), "dropped_invalid": n_bad,
            "dropped_duplicates": n_dup, "class_counts": {k: int(v) for k, v in counts.items()},
            "n_users": int(df["user_id"].nunique()), "n_sessions": int(df["session_id"].nunique())}
    return {"errors": errors, "warnings": warnings, "clean": df.reset_index(drop=True), "info": info}


def split_dataset(df: pd.DataFrame, seed: int = 42, test_size: float = 0.3,
                  gap: int = 10) -> dict:
    """Leakage-aware train/test split.

    Adjacent webcam frames are near-duplicates. A random row split puts almost identical
    frames into both train and test, so the test score measures memorisation, not
    generalisation, and is inflated. Strategies, in order of preference:
      1. 'group_by_user'    - whole users held out (needs >= 2 users).
      2. 'group_by_session' - whole recording sessions held out (needs >= 2 sessions).
      3. 'blocked_time'     - within each (session, class) the first part of the recording
                              trains, a ``gap`` of samples is discarded, the last part tests.
                              Reduces (but does not remove) leakage; results are PRELIMINARY.
    Returns dict(train_idx, test_idx, strategy, preliminary, notes).
    """
    from sklearn.model_selection import GroupShuffleSplit
    classes = sorted(df["label"].unique())
    for col, name in (("user_id", "group_by_user"), ("session_id", "group_by_session")):
        groups = df[col].astype(str)
        if groups.nunique() >= 2:
            for k in range(200):
                gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + k)
                tr, te = next(gss.split(df, df["label"], groups))
                if set(df.iloc[tr]["label"]) == set(classes) and set(df.iloc[te]["label"]) == set(classes):
                    return {"train_idx": tr, "test_idx": te, "strategy": name,
                            "preliminary": name == "group_by_session" and df["user_id"].nunique() < 2,
                            "notes": f"Held-out {col}s: {sorted(groups.iloc[te].unique())}"}
    tr_idx, te_idx = [], []
    order = df.sort_values("timestamp")
    for (_, _), g in order.groupby([order["session_id"].astype(str), order["label"]]):
        idx = g.index.to_numpy()
        n_train = int(len(idx) * (1 - test_size))
        tr_idx += list(idx[:n_train])
        te_idx += list(idx[min(n_train + gap, len(idx)):])
    pos = {ix: i for i, ix in enumerate(df.index)}
    return {"train_idx": np.array([pos[i] for i in tr_idx]),
            "test_idx": np.array([pos[i] for i in te_idx]),
            "strategy": "blocked_time", "preliminary": True,
            "notes": "Only one user/session available: blocked chronological split per class "
                     f"with a {gap}-sample gap. Optimistic: same person, same session."}
