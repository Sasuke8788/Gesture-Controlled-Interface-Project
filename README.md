# Gesture-Controlled Interface (PS 15)

Control a slide presentation with hand gestures using a webcam, MediaPipe hand landmarks, a
**custom-trained Random Forest** classifier, rule-based swipe detection, and a safe command
controller, with a Streamlit dashboard.

> **Honesty note.** MediaPipe provides the *pretrained* hand detector/landmark model (not trained here).
> The **Random Forest** is the custom, trainable static-gesture classifier. **SWIPE_LEFT/RIGHT use
> programmed temporal rules**, not a trained model. All accuracy / reliability numbers shown by the app come
> from files your own runs create; nothing is hard-coded.

## 1. Problem statement and objectives
PS 15: detect hand landmarks in real time, map gestures to commands, trigger actions (slide changes,
volume), test reliability across users, handle unintended gestures, and demonstrate the working system.
Objectives: real-time landmarks → trained classifier → validated gestures → real keyboard commands, with
quantitative evaluation and accidental-trigger protection.

## 2. Motivation and applications
Touch-free slide control for lectures and talks, sterile environments, accessibility, kiosks.

## 3. Technologies
Python 3.11 (3.12 also resolves), MediaPipe Tasks `HandLandmarker`, OpenCV, NumPy, pandas,
scikit-learn, matplotlib, Streamlit, pytest, pynput (keyboard events). Versions are pinned in
`requirements.txt` (mediapipe 0.10.21 forces `numpy<2` and `protobuf<5`, so `pyarrow` is pinned to 18.1.0).

## 4. Architecture and data flow
```
webcam (OpenCV) -> mirror -> HandDetector (MediaPipe, 21 landmarks)
   -> feature_extractor (88 features) -> GestureClassifier (Random Forest) -> StabilityFilter
   -> CommandController  <- SwipeDetector (wrist/palm trajectory)  
   -> keyboard (pynput)  |  log only (safe simulation)
```
| Module | Role |
|---|---|
| `src/hand_detector.py` | MediaPipe wrapper, hand-selection policy, drawing |
| `src/feature_extractor.py` | the ONE feature function used everywhere |
| `src/gesture_classifier.py` | model load/predict, `StabilityFilter` |
| `src/dynamic_gestures.py` | rule-based swipes |
| `src/command_controller.py` | all keyboard/OS code, safety interlocks |
| `src/pipeline.py` | glue used by dashboard and scripts |
| `src/data_utils.py`, `evaluation.py`, `metrics.py` | dataset, metrics, reliability metrics |
| `app.py`, `pages/` | Streamlit UI (home, Live, Evaluation, Dataset & Reliability) |

## 5. MediaPipe landmark detection
`HandLandmarker` (Tasks API, CPU, IMAGE mode) returns 21 landmarks per hand and handedness. The frame is
mirrored first (selfie view). **Multiple hands policy:** the hand with the largest landmark bounding box
(normally the closest) is used; all hands are drawn, the selected one highlighted. No hand / invalid
landmarks → the frame counts as "no hand" and resets all temporal state.

## 6. Feature extraction and normalisation (88 features)
Wrist-relative coordinates (60), two joint angles per finger (10), tip–wrist distances (5),
tip–palm-centre distances (5), extension ratios (5), three inter-fingertip distances (3).
Normalisation: left hands mirrored → wrist at origin (position invariance) → divide by wrist–middle-MCP
length (scale invariance) → in-plane rotation aligned so the palm axis points up.
**Limitations:** out-of-plane rotation (palm turned away/tilted) changes the 2-D projection and hurts
accuracy; palm length shrinks when tilted; MediaPipe `z` is noisy (down-weighted); a wrong handedness
label mirrors the hand; the mirroring rule assumes the selfie-view handedness MediaPipe reports.
Collect data at the angles you will demo.

## 7. Random Forest classification
200 trees, `class_weight="balanced"`, fixed seed. Classes: OPEN_PALM, FIST, THUMBS_UP, PEACE, POINTING,
NO_GESTURE. The displayed "confidence" is the fraction of trees voting for the class: a ranking score,
**not** a calibrated probability of being correct. `NO_GESTURE` is a real trained class (relaxed /
half-open / transition hand poses with the hand visible); frames with no hand are never treated as a gesture.

## 8. Dataset collection and labelling
`python collect_data.py --user alice --session s1` opens a window: keys `1`–`6` select a gesture, `SPACE`
records, `Q` quits. Rows go to `data/gesture_samples.csv` with `timestamp, user_id, session_id, label,
handedness` + features. Counts per class are shown in the window and on page 3; you can append more later.
Collect ≥150 samples/class (minimum enforced at training: 30), vary distance/tilt slightly, use a different
`--user` per person and a new `--session` per sitting.

## 9. Training and evaluation
`python train_model.py` validates columns/labels/NaN/duplicates/class balance, splits, trains, writes
`models/gesture_rf.joblib`, `models/class_labels.json`, and `results/` (`metrics.json`, confusion matrix
PNG/CSV, classification report, class distribution, dataset fingerprint). Optional: `--compare-svm`,
`--refit-all`.
**Leakage:** consecutive webcam frames are nearly identical, so a random row split puts near-duplicates in
train and test and inflates accuracy. Split preference: held-out **users** → held-out **sessions** →
(fallback, flagged *PRELIMINARY*) blocked chronological split per class with a gap. With one user and one
session the score is optimistic; the app labels it so.
`python evaluate_model.py --data other.csv --name other_person` evaluates the saved model on a separate
labelled CSV (valid as held-out only if those rows were not used in training).

## 10. Static vs dynamic gestures
Static gestures: Random Forest per frame + `StabilityFilter` (N consecutive agreeing, confident frames).
Dynamic gestures: `SwipeDetector` over a sliding window of palm-centre positions + timestamps: minimum
displacement, duration window, mostly-monotonic motion, low vertical drift, continuity (gap resets),
cooldown, reset on tracking loss. Direction is in the mirrored view (your right = SWIPE_RIGHT).
Conflicts (e.g. PEACE vs SWIPE_RIGHT both = next slide): static commands are blocked while the hand moves
faster than `max_static_speed` and during the swipe cooldown, and all slide commands share one cooldown.

## 11. Command execution and safeguards
| Gesture | Action |
|---|---|
| OPEN_PALM (held ~1 s, hand still) | activate gesture control |
| FIST | disable gesture control |
| PEACE / SWIPE_RIGHT | next slide (default key `Right`) |
| POINTING / SWIPE_LEFT | previous slide (default key `Left`) |
| THUMBS_UP | confirm: logs only, never sends a key |

* Starts in **safe simulation** (log only). Real keys require the "Send REAL keyboard commands" toggle (or `--mode real`); gestures can never enable real mode.
* Control must be **active**; slide commands are ignored otherwise.
* Stable gesture (N frames ≥ threshold), shared cooldown, and a gesture must be **absent for `release_s`** before it can fire again (held gesture = one command; flicker does not re-trigger).
* **Emergency stop** latches: control off, real mode off, gestures cannot re-activate until "Reset emergency stop".
* Key-send failure drops back to simulation. Camera is released on Stop, on leaving the page, and by a watchdog if the browser tab closes.
* **Windows focus:** keys go to the focused window. Click your slideshow (PowerPoint "Slide Show", PDF viewer, Google Slides full screen) before enabling real mode, or set "Target window title contains" on page 3 (best-effort focus via Win32; Windows may refuse). Apps running as Administrator ignore keys from a non-admin process. Never run collection/evaluation with real mode on (those scripts never send keys).

## 12. Reliability metrics and test procedures
`python test_reliability.py --user alice --distance 1m --lighting bright --orientation upright --speed normal --background plain --mode simulate`
Keys: `1–7` run one command trial (OPEN_PALM, FIST, PEACE, POINTING, THUMBS_UP, SWIPE_RIGHT, SWIPE_LEFT); `i` runs an
idle trial (move naturally; any command = false activation); `q` quits. Rows append to `results/reliability_log.csv`.
Definitions (see `src/metrics.py`):
* **Classifier accuracy / precision / recall / F1** – held-out static classification (training output).
* **Recognition rate** – intended gesture recognised (stable prediction or detected swipe) / command trials.
* **Command success rate** – intended action executed *exactly once* and nothing else / command trials. *Not* classifier accuracy.
* **Missed / incorrect / repeated rates** – no command / wrong command / intended command executed more than once.
* **False activations per minute** – actions executed in idle trials / idle minutes (controller ACTIVE = worst case).
* **Latency (ms)** – first correct raw prediction → stable gesture (system delay, excludes human reaction time; not recorded for swipes).
* **FPS** – mean processing frame rate.
Protocol (≥3 users if available): per user, per condition (distance 0.5/1/2 m, bright/dim light, 2 backgrounds, slow/normal/fast, tilt),
do ≥10 trials per command. Accidental-trigger scenarios (use `--scenario`): `natural_movement`, `similar_gesture_switch`,
`long_hold` (hold a gesture in a command trial), `tracking_loss`, `out_of_frame`, `poor_lighting`; run idle trials of ≥30 s.
One person only? Vary `--user` tags honestly (do **not** invent users); report the results as single-user.
Page 3 shows summaries, breakdowns by condition, and a CSV download.

## 13. Installation and execution (Windows, PowerShell, from the project folder)
```powershell
cd path\to\gesture_controlled_interface
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_model.py                # saves models\hand_landmarker.task (internet needed once)
python collect_data.py --user alice --session s1
python train_model.py
python evaluate_model.py --data data\gesture_samples.csv --name all_data   # (optional) NOT held-out on its own data
python test_reliability.py --user alice --mode simulate
python -m pytest -q                      # unit tests
streamlit run app.py                     # start the dashboard (browser opens)
# stop: press Ctrl+C in the terminal (or Stop camera in the app); deactivate leaves the venv
```
If `download_model.py` fails, open the URL printed in `config.HAND_MODEL_URL`
(`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`) and save the file as `models\hand_landmarker.task`.
After installing and downloading, everything runs offline. Webcam preview is embedded in the Streamlit page via a single OpenCV capture loop
(no browser camera permission needed; the *local Python process* opens the camera, so Windows Privacy > Camera must allow desktop apps).

## 14. Known limitations
* Not verified on a physical webcam by the author of this code: live landmarking, FPS, and real keystrokes need your manual test.
* Single-person data gives optimistic metrics; generalisation to new users/lighting needs more data.
* 2-D features, no out-of-plane rotation robustness; one hand controls at a time; swipe thresholds are hand-tuned.
* Streamlit frame loop is ~15–25 FPS depending on the machine (JPEG over websocket); the OpenCV scripts are faster.
* Future: temporal model (LSTM/1D-CNN) for dynamic gestures, per-user calibration, data augmentation, volume control.

## 15. Troubleshooting
| Symptom | Fix |
|---|---|
| `ImportError: DLL load failed while importing _framework_bindings` from MediaPipe on Windows | Use the project `.venv` and run the updated camera scripts from the project root. They load MediaPipe before pandas, which avoids a native DLL initialization conflict. Verify with `.\.venv\Scripts\Activate.ps1` followed by `python -c "import mediapipe; print(mediapipe.__version__)"`. |
| Streamlit shows `missing ScriptRunContext` warnings | Do not run `python app.py`; start the dashboard with `streamlit run app.py`. |
| "MediaPipe hand model not found" | `python download_model.py` |
| "No trained model" | collect data, `python train_model.py` |
| "Could not open camera" | close Teams/Zoom/Camera app; Settings > Privacy > Camera; change camera index (page 3 / `--camera 1`) |
| Slides don't move in real mode | focus the slideshow window; set target window title; run the app and PowerPoint at the same privilege level |
| Commands never fire | control not active (press Activate / hold OPEN_PALM); lower confidence threshold or consistent frames; check "stable gesture" panel |
| Double triggers | raise cooldown / release time |
| Port in use | `streamlit run app.py --server.port 8502` |
| Import errors after pip changes | recreate the venv; keep the pinned versions |
