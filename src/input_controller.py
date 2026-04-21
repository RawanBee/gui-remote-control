"""OS pointer movement via PyAutoGUI (Phase 2+)."""

from __future__ import annotations

import time

import pyautogui

_initialized = False
_screen_cache: tuple[int, int] | None = None
_screen_cache_mono: float = 0.0
_SCREEN_CACHE_TTL_S = 0.5


def _ensure_configured() -> None:
    global _initialized
    if _initialized:
        return
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
    _initialized = True


def screen_size() -> tuple[int, int]:
    """Cached briefly — ``pyautogui.size()`` can be surprisingly costly each frame."""
    global _screen_cache, _screen_cache_mono
    _ensure_configured()
    now = time.monotonic()
    if _screen_cache is None or (now - _screen_cache_mono) > _SCREEN_CACHE_TTL_S:
        w, h = pyautogui.size()
        _screen_cache = (int(w), int(h))
        _screen_cache_mono = now
    return _screen_cache


def invalidate_screen_size_cache() -> None:
    global _screen_cache
    _screen_cache = None


def move_pointer(x: int, y: int) -> None:
    _ensure_configured()
    pyautogui.moveTo(int(x), int(y), duration=0)


def click_left() -> None:
    _ensure_configured()
    pyautogui.click(button="left")


def click_left_at(x: int, y: int) -> None:
    """Click after a same-frame move — avoids using stale coordinates."""
    _ensure_configured()
    pyautogui.click(int(x), int(y), button="left", duration=0)


def click_right() -> None:
    _ensure_configured()
    pyautogui.click(button="right")


def click_right_at(x: int, y: int) -> None:
    _ensure_configured()
    pyautogui.click(int(x), int(y), button="right", duration=0)


def mouse_down_left() -> None:
    _ensure_configured()
    pyautogui.mouseDown(button="left")


def mouse_up_left() -> None:
    _ensure_configured()
    pyautogui.mouseUp(button="left")


def scroll_vertical(clicks: int) -> None:
    _ensure_configured()
    if clicks:
        pyautogui.scroll(int(clicks))
