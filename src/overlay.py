"""Draw landmarks, status, and debug HUD on frames."""

from __future__ import annotations

import cv2
import numpy as np

# Same topology as MediaPipe legacy Hands drawing (21 landmarks).
_HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def draw_hand_landmarks(
    frame_bgr: np.ndarray,
    landmarks,
    color: tuple[int, int, int] = (0, 255, 0),
    radius: int = 4,
) -> None:
    if not landmarks:
        return
    h, w = frame_bgr.shape[:2]

    pts: list[tuple[int, int]] = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))

    for a, b in _HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame_bgr, pts[a], pts[b], color, 2, lineType=cv2.LINE_AA)

    for x, y in pts:
        cv2.circle(frame_bgr, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


def draw_status(
    frame_bgr: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (12, 28),
) -> None:
    x0, y0 = origin
    for i, text in enumerate(lines):
        y = y0 + i * 26
        cv2.putText(
            frame_bgr,
            text,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            text,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            1,
            lineType=cv2.LINE_AA,
        )
