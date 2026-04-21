"""Shared runtime flags and timestamps for gesture phases."""

from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass
class AppState:
    control_enabled: bool = False
    is_dragging: bool = False
    scroll_mode: bool = False
    last_hand_seen: bool = False
    last_click_time: float = 0.0
    last_right_click_time: float = 0.0
    pinch_start_time: float | None = None
    last_cursor_x: float | None = None
    last_cursor_y: float | None = None
    locked_cursor_x: float | None = None
    locked_cursor_y: float | None = None
    last_sent_cursor_x: float | None = None
    last_sent_cursor_y: float | None = None
    last_hand_y: float | None = None
    last_gesture: str = "idle"
    # Preview flash after click: "index" | "middle" | None until monotonic time
    feedback_flash_tip: str | None = None
    feedback_flash_until_mono: float = 0.0

    def monotonic_ms(self) -> int:
        return int(time.monotonic() * 1000.0)
