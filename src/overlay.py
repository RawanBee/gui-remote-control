"""Draw landmarks, status, and debug HUD on frames."""

from __future__ import annotations

import cv2

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


def _tip_px(landmarks, tip_idx: int, w: int, h: int) -> tuple[int, int]:
    lm = landmarks[tip_idx]
    return int(float(lm.x) * w), int(float(lm.y) * h)


def draw_hand_interaction_feedback(
    frame_bgr: np.ndarray,
    landmarks,
    *,
    now_mono: float,
    index_left_lock: bool,
    index_left_hold_ready: bool,
    middle_right_lock: bool,
    middle_right_hold_ready: bool,
    left_button_down: bool,
    feedback_flash_tip: str | None,
    feedback_flash_until_mono: float,
) -> None:
    """
    Preview-only HUD at fingertips:
    - Left intent: yellow outer / orange inner ring on index = cursor frozen; red thick ring = press (drag).
    - Right intent: cyan outer / magenta inner on middle = frozen for right click.
    - Green burst on index or middle when a click just fired (short flash).
    """
    if not landmarks:
        return
    h, w = frame_bgr.shape[:2]

    # Left lock / aim (index tip — matches anchored cursor)
    if index_left_lock:
        cx, cy = _tip_px(landmarks, 8, w, h)
        cv2.circle(frame_bgr, (cx, cy), 20, (0, 255, 255), 2, lineType=cv2.LINE_AA)  # yellow
        if index_left_hold_ready:
            cv2.circle(frame_bgr, (cx, cy), 11, (0, 165, 255), 2, lineType=cv2.LINE_AA)  # orange

    # Right lock (middle tip — thumb–middle pinch)
    if middle_right_lock:
        mx, my = _tip_px(landmarks, 12, w, h)
        cv2.circle(frame_bgr, (mx, my), 20, (255, 255, 0), 2, lineType=cv2.LINE_AA)  # cyan in BGR
        if middle_right_hold_ready:
            cv2.circle(frame_bgr, (mx, my), 11, (255, 0, 255), 2, lineType=cv2.LINE_AA)  # magenta

    # Physical press (left mouse down while dragging)
    if left_button_down:
        cx, cy = _tip_px(landmarks, 8, w, h)
        cv2.circle(frame_bgr, (cx, cy), 24, (0, 0, 255), 3, lineType=cv2.LINE_AA)  # red ring

    # Click / release flash
    if (
        feedback_flash_tip
        and now_mono < feedback_flash_until_mono
        and feedback_flash_tip in ("index", "middle")
    ):
        idx = 8 if feedback_flash_tip == "index" else 12
        fx, fy = _tip_px(landmarks, idx, w, h)
        cv2.circle(frame_bgr, (fx, fy), 28, (0, 255, 0), 3, lineType=cv2.LINE_AA)


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
