"""Download MediaPipe Hand Landmarker task model into model/."""

from __future__ import annotations

import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dest = root / "model" / "hand_landmarker.task"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        print(f"Model already present: {dest}")
        return
    print(f"Downloading to {dest} ...")
    urllib.request.urlretrieve(MODEL_URL, dest)
    print("Done.")


if __name__ == "__main__":
    main()
