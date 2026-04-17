"""Gesture recognition: pinch click (thumb tip + index tip)."""

from __future__ import annotations

import math
from dataclasses import dataclass


def thumb_index_distance(landmarks) -> float:
    """Euclidean distance between thumb tip (4) and index tip (8) in normalized image space."""
    t = landmarks[4]
    i = landmarks[8]
    dx = t.x - i.x
    dy = t.y - i.y
    return math.hypot(dx, dy)


@dataclass
class PinchClickResult:
    pinched: bool
    pinch_hold_ready: bool
    should_click: bool


class PinchClickDetector:
    """
    Pinch must stay below threshold for click_hold_ms, then one click fires.
    No repeat until fingers separate (distance >= threshold) again.
    Cooldown also limits rapid re-clicks after a full cycle.
    """

    def __init__(self) -> None:
        self._pinch_since: float | None = None
        self._latched: bool = False

    def reset(self) -> None:
        self._pinch_since = None
        self._latched = False

    def update(
        self,
        distance: float,
        *,
        threshold: float,
        hold_ms: float,
        cooldown_ms: float,
        now: float,
        last_click_time: float,
    ) -> PinchClickResult:
        thr = max(1e-6, float(threshold))
        hold_s = max(0.0, float(hold_ms)) / 1000.0
        cooldown_s = max(0.0, float(cooldown_ms)) / 1000.0

        if distance >= thr:
            self._pinch_since = None
            self._latched = False
            return PinchClickResult(False, False, False)

        if self._pinch_since is None:
            self._pinch_since = now

        held = now - self._pinch_since
        pinch_hold_ready = held >= hold_s

        if self._latched:
            return PinchClickResult(True, pinch_hold_ready, False)

        if last_click_time > 0.0 and (now - last_click_time) < cooldown_s:
            return PinchClickResult(True, pinch_hold_ready, False)

        if held < hold_s:
            return PinchClickResult(True, pinch_hold_ready, False)

        self._latched = True
        return PinchClickResult(True, pinch_hold_ready, True)
