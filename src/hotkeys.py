"""Keyboard shortcuts for the OpenCV preview window."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KeyAction:
    toggle_control: bool = False
    disable_control: bool = False
    quit_app: bool = False
    start_calibration: bool = False
    confirm_calibration: bool = False


def interpret_preview_key(key: int) -> KeyAction:
    if key == ord("q"):
        return KeyAction(quit_app=True)
    if key == ord(" "):
        return KeyAction(toggle_control=True)
    if key == 27:  # ESC
        return KeyAction(disable_control=True)
    if key in (ord("c"), ord("C")):
        return KeyAction(start_calibration=True)
    if key in (10, 13):  # Enter / Return
        return KeyAction(confirm_calibration=True)
    return KeyAction()
