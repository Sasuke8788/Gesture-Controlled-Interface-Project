"""Download the pretrained MediaPipe hand_landmarker.task asset into models/ (needs internet once)."""
import sys
import urllib.request

import config

def main() -> int:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if config.HAND_MODEL_PATH.exists():
        print(f"Already present: {config.HAND_MODEL_PATH}")
        return 0
    print(f"Downloading {config.HAND_MODEL_URL}")
    try:
        urllib.request.urlretrieve(config.HAND_MODEL_URL, config.HAND_MODEL_PATH)
    except Exception as exc:
        print(f"Download failed: {exc}\nOpen the URL in a browser and save the file as\n"
              f"{config.HAND_MODEL_PATH}")
        return 1
    print(f"Saved {config.HAND_MODEL_PATH} ({config.HAND_MODEL_PATH.stat().st_size/1e6:.1f} MB)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
