"""Gestures: pointer, pinch click/drag, two-finger scroll, thumb–middle right click."""

from __future__ import annotations

import math
from dataclasses import dataclass


def thumb_index_distance(landmarks) -> float:
    """Euclidean distance between thumb tip (4) and index tip (8) in normalized image space."""
    t = landmarks[4]
    i = landmarks[8]
    return math.hypot(t.x - i.x, t.y - i.y)


def thumb_middle_distance(landmarks) -> float:
    """Distance between thumb tip (4) and middle tip (12)."""
    t = landmarks[4]
    m = landmarks[12]
    return math.hypot(t.x - m.x, t.y - m.y)


def finger_extended_up(landmarks, tip_idx: int, pip_idx: int, margin: float = 0.0) -> bool:
    """Tip above PIP in image (smaller y), roughly finger pointing up in frame."""
    return float(landmarks[tip_idx].y) + margin < float(landmarks[pip_idx].y)


def finger_curled_down(landmarks, tip_idx: int, pip_idx: int, margin: float = 0.008) -> bool:
    """Tip below PIP (larger y) — finger curled / not raised for scroll vs point."""
    return float(landmarks[tip_idx].y) > float(landmarks[pip_idx].y) + margin


def index_pointer_extended(landmarks, margin: float = 0.0) -> bool:
    return finger_extended_up(landmarks, 8, 6, margin)


def scroll_gesture_active(
    landmarks,
    pinch_dist: float,
    pinch_threshold: float,
    pinch_margin: float,
    *,
    min_middle_tip_above_index_y: float = 0.0,
) -> bool:
    """
    Dedicated scroll mode: index + middle extended, ring + pinky down, thumb not pinching index.

    Optional ``min_middle_tip_above_index_y`` (>0) adds an extra V-shape filter on top.
    """
    thr = max(1e-6, float(pinch_threshold))
    if pinch_dist < thr * float(pinch_margin):
        return False
    if not finger_extended_up(landmarks, 8, 6):
        return False
    if not finger_extended_up(landmarks, 12, 10):
        return False
    if not finger_curled_down(landmarks, 16, 14):
        return False
    if not finger_curled_down(landmarks, 20, 18):
        return False
    gap = float(min_middle_tip_above_index_y)
    if gap > 0.0 and not (float(landmarks[12].y) < float(landmarks[8].y) - gap):
        return False
    return True


class PinchApproachFreeze:
    """
    Loose enter (fingers coming together) / wider exit hysteresis.
    While active, cursor should stay anchored until release or drag takes over.
    """

    def __init__(self) -> None:
        self._active = False

    def reset(self) -> None:
        self._active = False

    def update(
        self,
        dist: float,
        *,
        enter_max: float,
        exit_max: float,
        drag_active: bool,
    ) -> tuple[bool, bool, bool]:
        """Returns ``(active, just_started, just_ended)``."""
        if drag_active:
            ended = self._active
            self._active = False
            return False, False, ended

        started = False
        ended = False
        if not self._active:
            if dist < enter_max:
                self._active = True
                started = True
        else:
            if dist > exit_max:
                self._active = False
                ended = True
        return self._active, started, ended


@dataclass
class PinchDragFrameResult:
    pinch_active: bool
    pinch_hold_ready: bool
    drag_active: bool
    drag_just_started: bool
    drag_just_ended: bool
    click_on_release: bool


class PinchDragClickProcessor:
    """
    Left: release pinch after ``click_hold_ms``–``drag_start_ms`` → click;
    hold past ``drag_start_ms`` → drag until release.
    """

    def __init__(self) -> None:
        self._was_pinching = False
        self._pinch_t0: float | None = None
        self._dragging = False

    def reset(self) -> None:
        self._was_pinching = False
        self._pinch_t0 = None
        self._dragging = False

    def update(
        self,
        distance: float,
        *,
        threshold: float,
        pinch_open_scale: float,
        click_hold_ms: float,
        click_hold_slop_ms: float,
        drag_start_ms: float,
        click_cooldown_ms: float,
        now: float,
        last_click_time: float,
    ) -> PinchDragFrameResult:
        thr_close = max(1e-6, float(threshold))
        open_scale = max(1.02, float(pinch_open_scale))
        thr_open = thr_close * open_scale
        if not self._was_pinching:
            pinch_active = distance < thr_close
        else:
            pinch_active = distance < thr_open

        click_hold = max(0.0, float(click_hold_ms))
        slop = max(0.0, float(click_hold_slop_ms))
        click_hold_eff = max(0.0, click_hold - slop)
        drag_start = max(float(drag_start_ms), click_hold + 1.0, click_hold_eff + 25.0)
        cooldown_s = max(0.0, float(click_cooldown_ms)) / 1000.0

        drag_just_started = False
        drag_just_ended = False
        click_on_release = False

        if pinch_active and not self._was_pinching:
            self._pinch_t0 = now

        if not pinch_active and self._was_pinching:
            t0 = self._pinch_t0 if self._pinch_t0 is not None else now
            held_ms = (now - t0) * 1000.0
            if self._dragging:
                self._dragging = False
                drag_just_ended = True
            elif held_ms >= click_hold_eff and held_ms < drag_start:
                cooldown_ok = last_click_time <= 0.0 or (now - last_click_time) >= cooldown_s
                if cooldown_ok:
                    click_on_release = True
            self._pinch_t0 = None

        pinch_hold_ready = False
        if pinch_active and self._pinch_t0 is not None:
            held_ms = (now - self._pinch_t0) * 1000.0
            pinch_hold_ready = held_ms >= click_hold_eff
            if not self._dragging and held_ms >= drag_start:
                self._dragging = True
                drag_just_started = True

        self._was_pinching = pinch_active
        return PinchDragFrameResult(
            pinch_active=pinch_active,
            pinch_hold_ready=pinch_hold_ready,
            drag_active=self._dragging,
            drag_just_started=drag_just_started,
            drag_just_ended=drag_just_ended,
            click_on_release=click_on_release,
        )


@dataclass
class SimpleReleaseClickResult:
    pinch_active: bool
    pinch_hold_ready: bool
    click_on_release: bool


class PinchReleaseClickOnly:
    """Thumb–middle (or any pair) pinch: click on release after min hold; no drag."""

    def __init__(self) -> None:
        self._was_pinching = False
        self._pinch_t0: float | None = None

    def reset(self) -> None:
        self._was_pinching = False
        self._pinch_t0 = None

    def update(
        self,
        distance: float,
        *,
        threshold: float,
        pinch_open_scale: float,
        hold_ms: float,
        cooldown_ms: float,
        now: float,
        last_click_time: float,
    ) -> SimpleReleaseClickResult:
        thr_close = max(1e-6, float(threshold))
        thr_open = thr_close * max(1.02, float(pinch_open_scale))
        if not self._was_pinching:
            pinch_active = distance < thr_close
        else:
            pinch_active = distance < thr_open

        hold = max(0.0, float(hold_ms))
        cooldown_s = max(0.0, float(cooldown_ms)) / 1000.0
        click_on_release = False

        if pinch_active and not self._was_pinching:
            self._pinch_t0 = now

        pinch_hold_ready = False
        if pinch_active and self._pinch_t0 is not None:
            held_ms = (now - self._pinch_t0) * 1000.0
            pinch_hold_ready = held_ms >= hold

        if not pinch_active and self._was_pinching:
            t0 = self._pinch_t0 if self._pinch_t0 is not None else now
            held_ms = (now - t0) * 1000.0
            if held_ms >= hold:
                ok = last_click_time <= 0.0 or (now - last_click_time) >= cooldown_s
                if ok:
                    click_on_release = True
            self._pinch_t0 = None

        self._was_pinching = pinch_active
        return SimpleReleaseClickResult(
            pinch_active=pinch_active,
            pinch_hold_ready=pinch_hold_ready,
            click_on_release=click_on_release,
        )


class TwoFingerScrollTracker:
    """Maps vertical motion of a reference point (normalized y) to scroll clicks."""

    def __init__(self) -> None:
        self._last_y: float | None = None
        self._dy_smooth = 0.0

    def reset(self) -> None:
        self._last_y = None
        self._dy_smooth = 0.0

    def scroll_delta(
        self,
        mid_y_norm: float,
        *,
        sensitivity: float,
        invert_y: bool,
        base_scale: float,
        max_clicks_per_frame: int,
        deadzone_y: float,
        motion_smoothing: float,
    ) -> int:
        if self._last_y is None:
            self._last_y = float(mid_y_norm)
            return 0
        dy = float(mid_y_norm) - float(self._last_y)
        self._last_y = float(mid_y_norm)
        if invert_y:
            dy = -dy
        dz = max(0.0, float(deadzone_y))
        if abs(dy) < dz:
            dy = 0.0
        sm = max(0.0, min(0.95, float(motion_smoothing)))
        self._dy_smooth = (1.0 - sm) * dy + sm * self._dy_smooth
        scale = max(1.0, float(base_scale)) * max(0.0, float(sensitivity))
        clicks = int(round(self._dy_smooth * scale))
        cap = int(max_clicks_per_frame)
        if cap > 0:
            clicks = max(-cap, min(cap, clicks))
        return clicks
