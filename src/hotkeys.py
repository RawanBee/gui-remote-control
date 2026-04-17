"""Keyboard shortcuts for the OpenCV preview window."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KeyAction:
    toggle_control: bool = False
    disable_control: bool = False
    quit_app: bool = False


def interpret_preview_key(key: int) -> KeyAction:
    if key == ord("q"):
        return KeyAction(quit_app=True)
    if key == ord(" "):
        return KeyAction(toggle_control=True)
    if key == 27:  # ESC
        return KeyAction(disable_control=True)
    return KeyAction()
