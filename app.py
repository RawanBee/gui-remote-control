#!/usr/bin/env python3
"""Phase 3: cursor from index tip + pinch click (debounced)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

from src.camera import Camera
from src.cursor_mapper import index_tip_to_screen
from src.gesture_engine import PinchClickDetector, thumb_index_distance
from src.hand_tracker import HandTracker
from src.hotkeys import interpret_preview_key
from src import input_controller
from src.overlay import draw_hand_landmarks, draw_status
from src.smoothing import smooth_pointer
from src.state import AppState


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    root = Path(__file__).resolve().parent
    cfg_path = root / "config.json"
    if not cfg_path.is_file():
        print(f"Missing {cfg_path}", file=sys.stderr)
        return 1

    cfg = _load_config(cfg_path)
    model_path = root / "model" / "hand_landmarker.task"

    cam = Camera(index=int(cfg.get("camera_index", 0)))
    if not cam.open():
        print("Could not open webcam. Check camera_index in config.json.", file=sys.stderr)
        return 1

    try:
        tracker = HandTracker(
            model_path=model_path,
            num_hands=int(cfg.get("num_hands", 1)),
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        cam.release()
        return 1

    state = AppState()
    pinch_clicks = PinchClickDetector()
    window = "Gesture Mouse (Phase 3)"
    prev_mono = time.monotonic()
    fps_smooth = 0.0

    pointer_smooth = float(cfg.get("pointer_smoothing", 0.35))
    dead_zone = float(cfg.get("dead_zone_fraction", 0.08))
    mirror_x = bool(cfg.get("mirror_camera_x", True))
    pinch_thr = float(cfg.get("pinch_distance_threshold", 0.05))
    click_hold_ms = float(cfg.get("click_hold_ms", 220))
    click_cooldown_ms = float(cfg.get("click_cooldown_ms", 350))

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("Frame grab failed.", file=sys.stderr)
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = tracker.process(frame_rgb, state.monotonic_ms())

            hand_detected = bool(result.hand_landmarks)
            state.last_hand_seen = hand_detected

            if hand_detected:
                draw_hand_landmarks(frame, result.hand_landmarks[0])

            now_mono = time.monotonic()
            screen_w, screen_h = input_controller.screen_size()

            pinch_line = ""
            if hand_detected and state.control_enabled:
                lms = result.hand_landmarks[0]
                tip = lms[8]
                sx, sy = index_tip_to_screen(
                    tip.x,
                    tip.y,
                    screen_width=screen_w,
                    screen_height=screen_h,
                    dead_zone_fraction=dead_zone,
                    mirror_camera_x=mirror_x,
                )
                fx, fy = smooth_pointer(
                    float(sx),
                    float(sy),
                    state.last_cursor_x,
                    state.last_cursor_y,
                    pointer_smooth,
                )
                state.last_cursor_x = fx
                state.last_cursor_y = fy
                input_controller.move_pointer(int(round(fx)), int(round(fy)))

                dist = thumb_index_distance(lms)
                pinch_res = pinch_clicks.update(
                    dist,
                    threshold=pinch_thr,
                    hold_ms=click_hold_ms,
                    cooldown_ms=click_cooldown_ms,
                    now=now_mono,
                    last_click_time=state.last_click_time,
                )
                if pinch_res.pinched:
                    state.last_gesture = "pinch"
                    if pinch_res.pinch_hold_ready:
                        pinch_line = "pinch: stable (release to reset)"
                    else:
                        pinch_line = "pinch: hold…"
                else:
                    state.last_gesture = "move"

                if pinch_res.should_click:
                    input_controller.click_left()
                    state.last_click_time = now_mono
            else:
                pinch_clicks.reset()
                state.last_cursor_x = None
                state.last_cursor_y = None
                if not hand_detected:
                    state.last_gesture = "idle"
                elif not state.control_enabled:
                    state.last_gesture = "idle"

            dt = now_mono - prev_mono
            prev_mono = now_mono
            if dt > 0.0:
                inst_fps = 1.0 / dt
                fps_smooth = inst_fps if fps_smooth == 0.0 else (0.9 * fps_smooth + 0.1 * inst_fps)

            pinch_hint = pinch_line
            if not pinch_hint and hand_detected and state.control_enabled:
                pinch_hint = f"click: pinch thumb+index, hold ~{int(click_hold_ms)} ms"

            status_lines = [
                "hand detected" if hand_detected else "no hand",
                f"fps: {fps_smooth:4.1f}",
                f"control: {'ON' if state.control_enabled else 'OFF'} (space toggles, esc off)",
                f"gesture: {state.last_gesture}",
            ]
            if pinch_hint:
                status_lines.append(pinch_hint)
            status_lines.append("q quit")
            draw_status(frame, status_lines)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            action = interpret_preview_key(key)
            if action.quit_app:
                break
            if action.toggle_control:
                state.control_enabled = not state.control_enabled
                if not state.control_enabled:
                    state.last_cursor_x = None
                    state.last_cursor_y = None
                    pinch_clicks.reset()
            if action.disable_control:
                state.control_enabled = False
                state.last_cursor_x = None
                state.last_cursor_y = None
                pinch_clicks.reset()
    finally:
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
