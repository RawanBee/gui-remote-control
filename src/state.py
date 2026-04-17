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
    pinch_start_time: float | None = None
    last_cursor_x: float | None = None
    last_cursor_y: float | None = None
    last_hand_y: float | None = None
    last_gesture: str = "idle"

    def monotonic_ms(self) -> int:
        return int(time.monotonic() * 1000.0)
