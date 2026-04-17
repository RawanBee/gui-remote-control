"""MediaPipe Tasks Hand Landmarker wrapper."""

from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    def __init__(self, model_path: str | Path, num_hands: int = 1) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {path.resolve()}. "
                "Run: python scripts/download_model.py"
            )

        base_options = python.BaseOptions(model_asset_path=str(path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, rgb_frame: np.ndarray, timestamp_ms: int):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self) -> None:
        self._landmarker.close()
