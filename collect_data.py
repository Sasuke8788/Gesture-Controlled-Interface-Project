"""Labelled landmark-feature collection with an OpenCV window.

Usage (from project root):
    python collect_data.py --user alice --session s1
Keys:  1-6 choose gesture | SPACE start/stop recording | Q quit
Only frames with a detected, valid hand are saved. No keyboard commands are ever sent here.
"""
import argparse
import time
from collections import Counter

import cv2
import mediapipe

import config
from src.hand_detector import HandDetector, HandModelMissingError, draw_hands
from src.webcam_manager import WebcamError, WebcamManager
from src import data_utils as du
from src import feature_extractor as fe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="user1", help="person id (use a different id per person)")
    ap.add_argument("--session", default=time.strftime("sess_%Y%m%d_%H%M%S"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--every", type=int, default=2, help="save every Nth detected frame")
    ap.add_argument("--target", type=int, default=150, help="samples per class goal (display only)")
    args = ap.parse_args()

    try:
        detector = HandDetector()
    except HandModelMissingError as exc:
        print(exc)
        return 1
    existing = du.class_counts()
    session_counts = Counter()
    classes = config.GESTURE_CLASSES
    current, recording, buffer, frame_i = 0, False, [], 0
    cam = WebcamManager(args.camera)
    try:
        cam.open()
    except WebcamError as exc:
        print(exc)
        detector.close()
        return 1
    print(__doc__)
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)
            hand = detector.detect(frame)
            draw_hands(frame, hand)
            label = classes[current]
            if recording and hand.detected:
                frame_i += 1
                if frame_i % args.every == 0:
                    try:
                        feats = fe.extract_features(hand.landmarks, hand.handedness)
                        buffer.append(du.make_row(feats, label, args.user, args.session, hand.handedness))
                        session_counts[label] += 1
                    except fe.InvalidLandmarksError:
                        pass
            y = 22
            cv2.putText(frame, f"[{current+1}] {label}  {'REC' if recording else 'paused'}  "
                        f"hand={'yes' if hand.detected else 'NO'}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if recording else (255, 255, 0), 2)
            for i, c in enumerate(classes):
                tot = existing.get(c, 0) + session_counts[c]
                cv2.putText(frame, f"{i+1} {c}: {tot} (+{session_counts[c]})", (10, 48 + 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if tot >= args.target else (255, 255, 255), 1)
            cv2.putText(frame, GESTURE_HELP_SHORT(label), (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.imshow("Collect gesture data (Q quit, SPACE rec)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                recording = not recording
                frame_i = 0
                if not recording and buffer:
                    du.append_samples(buffer)
                    buffer = []
            if ord("1") <= key <= ord(str(len(classes))):
                current, recording = key - ord("1"), False
                if buffer:
                    du.append_samples(buffer)
                    buffer = []
            if cv2.getWindowProperty("Collect gesture data (Q quit, SPACE rec)", cv2.WND_PROP_VISIBLE) < 1:
                break
    except WebcamError as exc:
        print(exc)
    finally:
        if buffer:
            du.append_samples(buffer)
        cam.release()
        detector.close()
        cv2.destroyAllWindows()
    print(f"Saved this session: {dict(session_counts)} -> {config.DATASET_PATH}")
    return 0


def GESTURE_HELP_SHORT(label: str) -> str:
    return config.GESTURE_HELP[label][:95]


if __name__ == "__main__":
    raise SystemExit(main())
