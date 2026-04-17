"""Webcam capture only."""

from __future__ import annotations

import cv2
import numpy as np


class Camera:
    def __init__(self, index: int = 0) -> None:
        self._index = index
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._index)
        return bool(self._cap and self._cap.isOpened())

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._cap:
            return False, None
        ok, frame = self._cap.read()
        return ok, frame

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
