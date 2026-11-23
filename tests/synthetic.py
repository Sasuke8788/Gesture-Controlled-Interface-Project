"""SYNTHETIC hand-landmark generator used ONLY by unit tests.

These are procedurally generated skeletons, NOT webcam data. They verify code paths (feature
schema, training/inference plumbing) and must never be used to claim real-world accuracy.
"""
import numpy as np

BASES = {
    "index": ((0.45, 0.62), (-0.10, -1.0)), "middle": ((0.50, 0.60), (0.0, -1.0)),
    "ring": ((0.55, 0.62), (0.10, -1.0)), "pinky": ((0.60, 0.66), (0.22, -1.0)),
}
LEN = (0.07, 0.045, 0.04)
CURL_ANGLES = (0.0, 1.3, 2.4)


def _rot(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _chain(mcp, d, curl, sign):
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    pts, cur = [], np.asarray(mcp, float)
    for ln, ang in zip(LEN, CURL_ANGLES):
        cur = cur + _rot(d, sign * ang * curl) * ln
        pts.append(cur.copy())
    return pts


def make_hand(pose, rng, noise=0.004, rotate=True):
    curls = {"thumb": 1.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0}
    ext = {"OPEN_PALM": ["thumb", "index", "middle", "ring", "pinky"], "FIST": [],
           "THUMBS_UP": ["thumb"], "PEACE": ["index", "middle"], "POINTING": ["index"]}
    if pose == "NO_GESTURE":
        curls = {k: rng.uniform(0.25, 0.75) for k in curls}
    else:
        for f in ext[pose]:
            curls[f] = 0.0
    wrist = np.array([0.5, 0.8])
    lm = np.zeros((21, 2)); lm[0] = wrist
    cmc, mcp = np.array([0.45, 0.76]), np.array([0.40, 0.70])
    lm[1], lm[2] = cmc, mcp
    tdir = np.array([-0.5, -1.0]) if pose != "THUMBS_UP" else np.array([-0.05, -1.0])
    t = _chain(mcp, tdir, curls["thumb"] * 0.8, +1)
    lm[3], lm[4] = t[1], t[2]
    for i, f in enumerate(["index", "middle", "ring", "pinky"]):
        base, d = BASES[f]
        lm[5 + 4 * i] = base
        pip, dip, tip = _chain(base, d, curls[f], -1 if f in ("index", "middle") else -1)
        lm[6 + 4 * i], lm[7 + 4 * i], lm[8 + 4 * i] = pip, dip, tip
    if pose == "PEACE":
        lm[8] += np.array([-0.03, 0]); lm[12] += np.array([0.03, 0])
    z = rng.normal(0, 0.01, (21, 1))
    lm = lm + rng.normal(0, noise, lm.shape)
    if rotate:
        a = rng.uniform(-0.25, 0.25)
        c = lm.mean(0)
        lm = (lm - c) @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T + c
    scale = rng.uniform(0.7, 1.3)
    lm = (lm - 0.5) * scale + 0.5 + rng.uniform(-0.1, 0.1, 2)
    return np.hstack([lm, z])
