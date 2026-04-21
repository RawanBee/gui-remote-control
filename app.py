#!/usr/bin/env python3
"""Hand-tracked pointer: freeze-on-intent (left + right), press/click feedback rings."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2

from src.camera import Camera
from src.cursor_mapper import index_tip_to_screen
from src.gesture_engine import (
    PinchApproachFreeze,
    PinchDragClickProcessor,
    PinchReleaseClickOnly,
    TwoFingerScrollTracker,
    index_pointer_extended,
    scroll_gesture_active,
    thumb_index_distance,
    thumb_middle_distance,
)
from src.hand_tracker import HandTracker
from src.hotkeys import interpret_preview_key
from src import input_controller
from src.overlay import draw_hand_interaction_feedback, draw_hand_landmarks, draw_status
from src.smoothing import smooth_pointer
from src.state import AppState


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_end_drag(state: AppState) -> None:
    if state.is_dragging:
        input_controller.mouse_up_left()
        state.is_dragging = False


def _clear_pointer_lock(state: AppState, *freezes: PinchApproachFreeze) -> None:
    for fz in freezes:
        fz.reset()
    state.locked_cursor_x = None
    state.locked_cursor_y = None


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
    pinch_drag = PinchDragClickProcessor()
    pinch_freeze = PinchApproachFreeze()
    right_pinch_freeze = PinchApproachFreeze()
    right_click = PinchReleaseClickOnly()
    scroll_track = TwoFingerScrollTracker()
    window = "Gesture Mouse"
    prev_mono = time.monotonic()
    fps_smooth = 0.0
    prev_scroll_active = False

    pointer_smooth = float(cfg.get("pointer_smoothing", 0.14))
    pointer_smooth_drag = float(cfg.get("pointer_smoothing_drag", 0.12))
    dead_zone = float(cfg.get("dead_zone_fraction", 0.08))
    mirror_x = bool(cfg.get("mirror_camera_x", True))
    require_index_ext = bool(cfg.get("require_index_extended_for_pointer", True))
    pinch_thr = float(cfg.get("pinch_distance_threshold", 0.055))
    pinch_release_scale = float(cfg.get("pinch_release_scale", 1.52))
    freeze_enter_scale = float(cfg.get("pinch_freeze_enter_scale", 1.48))
    freeze_exit_scale = float(cfg.get("pinch_freeze_exit_scale", 1.12))
    right_freeze_scale = float(cfg.get("right_pinch_freeze_scale", 1.32))
    right_hold_ms = float(cfg.get("right_click_hold_ms", 120))
    right_cooldown_ms = float(cfg.get("right_click_cooldown_ms", 300))
    # Left pinch uses the same minimum hold and cooldown as right unless overridden.
    click_hold_ms = float(cfg.get("click_hold_ms", right_hold_ms))
    click_hold_slop_ms = float(cfg.get("click_hold_slop_ms", 0.0))
    click_cooldown_ms = float(cfg.get("click_cooldown_ms", right_cooldown_ms))
    drag_start_ms = float(cfg.get("drag_start_ms", 420))
    right_index_clear = float(cfg.get("right_pinch_index_clear_scale", 1.12))
    pixel_deadzone = float(cfg.get("cursor_pixel_deadzone", 2.0))
    left_pinch_lock_cursor = bool(cfg.get("left_pinch_lock_cursor", False))
    flash_seconds = float(cfg.get("feedback_flash_seconds", 0.22))
    scroll_sens = float(cfg.get("scroll_sensitivity", 0.55))
    scroll_base = float(cfg.get("scroll_base_scale", 220))
    scroll_cap = int(cfg.get("scroll_max_clicks_per_frame", 2))
    scroll_dz = float(cfg.get("scroll_deadzone_y", 0.0011))
    scroll_smooth = float(cfg.get("scroll_motion_smoothing", 0.78))
    scroll_invert = bool(cfg.get("scroll_invert_y", True))
    scroll_pinch_margin = float(cfg.get("scroll_pinch_margin", 1.15))
    scroll_middle_above = float(cfg.get("scroll_min_middle_tip_above_index_y", 0.0))

    fe_enter = pinch_thr * freeze_enter_scale
    fe_exit = fe_enter * freeze_exit_scale
    fe_enter_r = fe_enter * right_freeze_scale
    fe_exit_r = fe_exit * right_freeze_scale

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

            pr = None
            rr = None
            freeze_on = False
            freeze_started = False
            right_freeze_on = False
            right_freeze_started = False
            use_freeze = False

            now_mono = time.monotonic()
            if now_mono >= state.feedback_flash_until_mono:
                state.feedback_flash_tip = None

            screen_w, screen_h = input_controller.screen_size()
            hint_extra = ""

            if hand_detected and state.control_enabled:
                lms = result.hand_landmarks[0]
                dist_l = thumb_index_distance(lms)
                dist_r = thumb_middle_distance(lms)
                scroll_active = scroll_gesture_active(
                    lms,
                    dist_l,
                    pinch_thr,
                    scroll_pinch_margin,
                    min_middle_tip_above_index_y=scroll_middle_above,
                )

                if scroll_active and not prev_scroll_active:
                    _safe_end_drag(state)
                    pinch_drag.reset()
                    right_click.reset()
                    _clear_pointer_lock(state, pinch_freeze, right_pinch_freeze)

                if not scroll_active:
                    rr = None
                    right_freeze_on = False
                    pr = pinch_drag.update(
                        dist_l,
                        threshold=pinch_thr,
                        pinch_open_scale=pinch_release_scale,
                        click_hold_ms=click_hold_ms,
                        click_hold_slop_ms=click_hold_slop_ms,
                        drag_start_ms=drag_start_ms,
                        click_cooldown_ms=click_cooldown_ms,
                        now=now_mono,
                        last_click_time=state.last_click_time,
                    )

                    state.is_dragging = pr.drag_active

                    if pr.pinch_active:
                        right_pinch_freeze.reset()
                        right_freeze_on = False

                    if pr.drag_active:
                        _clear_pointer_lock(state, pinch_freeze, right_pinch_freeze)
                        freeze_on = False
                    else:
                        freeze_on, freeze_started, _ = pinch_freeze.update(
                            dist_l,
                            enter_max=fe_enter,
                            exit_max=fe_exit,
                            drag_active=pr.drag_active,
                        )

                    if pr.click_on_release or pr.drag_just_ended:
                        _clear_pointer_lock(state, pinch_freeze, right_pinch_freeze)
                        freeze_on = False
                        right_freeze_on = False

                    drag_follow_hint = pr.drag_active
                    use_freeze_hint = bool(
                        not scroll_active
                        and freeze_on
                        and not drag_follow_hint
                        and left_pinch_lock_cursor
                    )

                    if pr.pinch_active:
                        state.last_gesture = "pinch" if not pr.drag_active else "drag"
                        if pr.drag_active:
                            hint_extra = "drag: release thumb–index pinch (red = press)"
                        elif use_freeze_hint and pr.pinch_hold_ready:
                            hint_extra = "left: locked — release to click (orange ring)"
                        elif use_freeze_hint:
                            hint_extra = "left: locked — aim (yellow ring)"
                        else:
                            hint_extra = "left: pinch…"
                    else:
                        dist_for_right = (
                            dist_r if dist_l > pinch_thr * right_index_clear else 99.0
                        )
                        rr = right_click.update(
                            dist_for_right,
                            threshold=pinch_thr,
                            pinch_open_scale=pinch_release_scale,
                            hold_ms=right_hold_ms,
                            cooldown_ms=right_cooldown_ms,
                            now=now_mono,
                            last_click_time=state.last_right_click_time,
                        )
                        right_freeze_on, right_freeze_started, _ = right_pinch_freeze.update(
                            dist_for_right,
                            enter_max=fe_enter_r,
                            exit_max=fe_exit_r,
                            drag_active=False,
                        )
                        if rr.pinch_active:
                            state.last_gesture = "right pinch"
                            hint_extra = "right: locked on middle — release (cyan/magenta)"
                        else:
                            state.last_gesture = "move"
                            hint_extra = (
                                f"L: pinch {int(click_hold_ms)}–{int(drag_start_ms)} ms | "
                                f"R: thumb–middle | scroll: 2 up, ring/pinky down"
                            )
                else:
                    state.scroll_mode = True
                    state.last_gesture = "scroll"
                    mid_y = 0.5 * (float(lms[8].y) + float(lms[12].y))
                    clicks = scroll_track.scroll_delta(
                        mid_y,
                        sensitivity=scroll_sens,
                        invert_y=scroll_invert,
                        base_scale=scroll_base,
                        max_clicks_per_frame=scroll_cap,
                        deadzone_y=scroll_dz,
                        motion_smoothing=scroll_smooth,
                    )
                    input_controller.scroll_vertical(clicks)
                    hint_extra = "scroll: index+middle up, ring+pinky down, move vertically"

                open_rough = pinch_thr * max(1.25, pinch_release_scale) * 1.02
                in_left_pinch_rough = dist_l < open_rough
                thumb_index_tracked = pr is not None and (pr.pinch_active or pr.drag_active)
                pointer_ok = (
                    (not require_index_ext)
                    or scroll_active
                    or index_pointer_extended(lms)
                    or in_left_pinch_rough
                    or thumb_index_tracked
                )

                drag_follow = bool(pr and pr.drag_active)
                use_freeze = bool(
                    not scroll_active
                    and pr is not None
                    and not drag_follow
                    and (
                        right_freeze_on
                        or (freeze_on and left_pinch_lock_cursor)
                    )
                )

                if pointer_ok:
                    tip = lms[8]
                    sx, sy = index_tip_to_screen(
                        tip.x,
                        tip.y,
                        screen_width=screen_w,
                        screen_height=screen_h,
                        dead_zone_fraction=dead_zone,
                        mirror_camera_x=mirror_x,
                    )
                    sm = (
                        pointer_smooth_drag
                        if (pr is not None and pr.drag_active)
                        else pointer_smooth
                    )
                    raw_fx, raw_fy = smooth_pointer(
                        float(sx),
                        float(sy),
                        state.last_cursor_x,
                        state.last_cursor_y,
                        sm,
                    )

                    apply_deadzone = (
                        not scroll_active
                        and not use_freeze
                        and not drag_follow
                        and pixel_deadzone > 0.0
                        and state.last_sent_cursor_x is not None
                        and state.last_sent_cursor_y is not None
                    )
                    if apply_deadzone:
                        d = math.hypot(
                            raw_fx - state.last_sent_cursor_x,
                            raw_fy - state.last_sent_cursor_y,
                        )
                        if d < pixel_deadzone:
                            raw_fx = state.last_sent_cursor_x
                            raw_fy = state.last_sent_cursor_y

                    if use_freeze:
                        lock_start = (
                            freeze_started
                            or right_freeze_started
                            or state.locked_cursor_x is None
                        )
                        if lock_start:
                            state.locked_cursor_x = raw_fx
                            state.locked_cursor_y = raw_fy
                        fx = float(state.locked_cursor_x)
                        fy = float(state.locked_cursor_y)
                    else:
                        fx, fy = raw_fx, raw_fy
                        state.locked_cursor_x = None
                        state.locked_cursor_y = None

                    state.last_cursor_x = fx
                    state.last_cursor_y = fy
                    input_controller.move_pointer(int(round(fx)), int(round(fy)))
                    state.last_sent_cursor_x = fx
                    state.last_sent_cursor_y = fy

                    draw_hand_interaction_feedback(
                        frame,
                        lms,
                        now_mono=now_mono,
                        index_left_lock=bool(
                            not scroll_active
                            and freeze_on
                            and not drag_follow
                            and left_pinch_lock_cursor
                        ),
                        index_left_hold_ready=bool(pr and pr.pinch_hold_ready),
                        middle_right_lock=bool(
                            not scroll_active and right_freeze_on and not drag_follow
                        ),
                        middle_right_hold_ready=bool(
                            rr and rr.pinch_hold_ready and right_freeze_on
                        ),
                        left_button_down=bool(pr and pr.drag_active),
                        feedback_flash_tip=state.feedback_flash_tip,
                        feedback_flash_until_mono=state.feedback_flash_until_mono,
                    )

                if hand_detected and state.control_enabled and not scroll_active and pr is not None:
                    cx = state.last_sent_cursor_x
                    cy = state.last_sent_cursor_y
                    if pr.drag_just_ended:
                        input_controller.mouse_up_left()
                    if pr.drag_just_started:
                        input_controller.mouse_down_left()
                    if pr.click_on_release:
                        if cx is not None and cy is not None:
                            input_controller.click_left_at(
                                int(round(cx)), int(round(cy))
                            )
                        else:
                            input_controller.click_left()
                        state.last_click_time = now_mono
                        state.feedback_flash_tip = "index"
                        state.feedback_flash_until_mono = now_mono + flash_seconds

                if (
                    hand_detected
                    and state.control_enabled
                    and not scroll_active
                    and rr is not None
                    and rr.click_on_release
                ):
                    cx = state.last_sent_cursor_x
                    cy = state.last_sent_cursor_y
                    if cx is not None and cy is not None:
                        input_controller.click_right_at(int(round(cx)), int(round(cy)))
                    else:
                        input_controller.click_right()
                    state.last_right_click_time = now_mono
                    state.feedback_flash_tip = "middle"
                    state.feedback_flash_until_mono = now_mono + flash_seconds
                    _clear_pointer_lock(state, pinch_freeze, right_pinch_freeze)

                if scroll_active:
                    state.scroll_mode = True
                else:
                    state.scroll_mode = False
                    if prev_scroll_active:
                        scroll_track.reset()

                prev_scroll_active = scroll_active
            else:
                prev_scroll_active = False
                state.scroll_mode = False
                scroll_track.reset()
                pinch_drag.reset()
                right_click.reset()
                _clear_pointer_lock(state, pinch_freeze, right_pinch_freeze)
                _safe_end_drag(state)
                state.last_cursor_x = None
                state.last_cursor_y = None
                state.last_sent_cursor_x = None
                state.last_sent_cursor_y = None
                state.last_gesture = "idle"
                state.feedback_flash_tip = None

            dt = now_mono - prev_mono
            prev_mono = now_mono
            if dt > 0.0:
                inst_fps = 1.0 / dt
                fps_smooth = inst_fps if fps_smooth == 0.0 else (0.9 * fps_smooth + 0.1 * inst_fps)

            status_lines = [
                "hand detected" if hand_detected else "no hand",
                f"fps: {fps_smooth:4.1f}",
                f"control: {'ON' if state.control_enabled else 'OFF'} (space toggles, esc off)",
                f"gesture: {state.last_gesture}",
            ]
            if hint_extra and (hand_detected and state.control_enabled):
                status_lines.append(hint_extra)
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
                    prev_scroll_active = False
                    scroll_track.reset()
                    pinch_drag.reset()
                    right_click.reset()
                    _clear_pointer_lock(state, pinch_freeze, right_pinch_freeze)
                    _safe_end_drag(state)
                    state.last_cursor_x = None
                    state.last_cursor_y = None
                    state.last_sent_cursor_x = None
                    state.last_sent_cursor_y = None
                    state.feedback_flash_tip = None
            if action.disable_control:
                state.control_enabled = False
                prev_scroll_active = False
                scroll_track.reset()
                pinch_drag.reset()
                right_click.reset()
                _clear_pointer_lock(state, pinch_freeze, right_pinch_freeze)
                _safe_end_drag(state)
                state.last_cursor_x = None
                state.last_cursor_y = None
                state.last_sent_cursor_x = None
                state.last_sent_cursor_y = None
                state.feedback_flash_tip = None
    finally:
        _safe_end_drag(state)
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
