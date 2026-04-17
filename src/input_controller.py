"""OS pointer movement via PyAutoGUI (Phase 2+)."""

from __future__ import annotations

import pyautogui

_initialized = False


def _ensure_configured() -> None:
    global _initialized
    if _initialized:
        return
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
    _initialized = True


def screen_size() -> tuple[int, int]:
    _ensure_configured()
    w, h = pyautogui.size()
    return int(w), int(h)


def move_pointer(x: int, y: int) -> None:
    _ensure_configured()
    pyautogui.moveTo(int(x), int(y), duration=0)


def click_left() -> None:
    _ensure_configured()
    pyautogui.click(button="left")
