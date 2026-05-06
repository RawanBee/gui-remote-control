#!/usr/bin/env python3
"""Hand-tracked pointer: freeze-on-intent (left + right), press/click feedback rings."""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera
from src.cursor_mapper import index_tip_to_screen
from src.gesture_engine import (
    PinchApproachFreeze,
    PinchDragClickProcessor,
    PoseHoldClickOnly,
    TwoFingerScrollTracker,
    hand_scale,
    index_pointer_extended,
    right_click_pose_active,
    scroll_gesture_active,
    thumb_index_distance,
)
from src.hand_tracker import HandTracker
from src.hotkeys import interpret_preview_key
from src import input_controller
from src.overlay import draw_hand_interaction_feedback, draw_hand_landmarks
from src.smoothing import smooth_pointer
from src.state import AppState


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_config_path(root: Path, argv: list[str]) -> Path:
    """Resolve config path from CLI (`--config PATH`) or default to config.json."""
    cfg_path = root / "config.json"
    if "--config" not in argv:
        return cfg_path
    i = argv.index("--config")
    if i + 1 >= len(argv):
        raise ValueError("--config requires a file path")
    raw = Path(argv[i + 1]).expanduser()
    return raw if raw.is_absolute() else (root / raw)


def _wrap_text_to_width(
    text: str,
    max_width_px: int,
    *,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    thickness: int = 1,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        w, _ = cv2.getTextSize(trial, font_face, font_scale, thickness)[0]
        if w <= max_width_px:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


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
    try:
        cfg_path = _resolve_config_path(root, sys.argv[1:])
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
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
    right_click = PoseHoldClickOnly()
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
    right_hold_ms = float(cfg.get("right_click_hold_ms", 120))
    right_cooldown_ms = float(cfg.get("right_click_cooldown_ms", 300))
    # Left pinch uses the same minimum hold and cooldown as right unless overridden.
    click_hold_ms = float(cfg.get("click_hold_ms", right_hold_ms))
    click_hold_slop_ms = float(cfg.get("click_hold_slop_ms", 0.0))
    click_cooldown_ms = float(cfg.get("click_cooldown_ms", right_cooldown_ms))
    drag_start_ms = float(cfg.get("drag_start_ms", 420))
    right_click_pose_hold_ms = float(cfg.get("right_click_pose_hold_ms", right_hold_ms))
    right_click_pose_motion_deadzone = float(
        cfg.get(
            "right_click_index_motion_deadzone",
            cfg.get("right_click_pose_motion_deadzone", 0.0016),
        )
    )
    click_stability_pixels = float(cfg.get("click_stability_pixels", 6.0))
    pixel_deadzone = float(cfg.get("cursor_pixel_deadzone", 2.0))
    left_pinch_lock_cursor = bool(cfg.get("left_pinch_lock_cursor", False))
    flash_seconds = float(cfg.get("feedback_flash_seconds", 0.22))
    scroll_sens = float(cfg.get("scroll_sensitivity", 0.55))
    scroll_base = float(cfg.get("scroll_base_scale", 220))
    scroll_cap = int(cfg.get("scroll_max_clicks_per_frame", 2))
    scroll_dz = float(cfg.get("scroll_deadzone_y", 0.0011))
    scroll_mode_motion_deadzone = float(cfg.get("scroll_mode_motion_deadzone", 0.0022))
    scroll_smooth = float(cfg.get("scroll_motion_smoothing", 0.78))
    scroll_invert = bool(cfg.get("scroll_invert_y", True))
    scroll_pinch_margin = float(cfg.get("scroll_pinch_margin", 1.15))
    scroll_middle_above = float(cfg.get("scroll_min_middle_tip_above_index_y", 0.0))
    calibration_enabled = bool(cfg.get("calibration_enabled", True))
    calibration_frames = int(cfg.get("calibration_frames", 90))
    calibration_min_open_multiplier = float(cfg.get("calibration_min_open_multiplier", 1.25))
    adaptive_thresholds = bool(cfg.get("adaptive_thresholds", True))
    adaptive_min_scale = float(cfg.get("adaptive_min_scale", 0.75))
    adaptive_max_scale = float(cfg.get("adaptive_max_scale", 1.35))
    preview_width = int(cfg.get("preview_width", 360))
    preview_height = int(cfg.get("preview_height", 220))
    preview_width_percent = float(cfg.get("preview_width_percent", 0.0))
    preview_size_percent = float(cfg.get("preview_size_percent", 0.0))
    preview_margin = int(cfg.get("preview_margin", 20))
    preview_anchor = str(cfg.get("preview_anchor", "bottom-left")).strip().lower()
    preview_always_on_top = bool(cfg.get("preview_always_on_top", True))
    preview_mirror = bool(cfg.get("preview_mirror", mirror_x))
    hud_height = int(cfg.get("hud_height", 120))
    hud_gap = int(cfg.get("hud_gap", 6))

    fe_enter = pinch_thr * freeze_enter_scale
    fe_exit = fe_enter * freeze_exit_scale
    calibrated_pinch_thr = pinch_thr
    ref_hand_scale: float | None = None
    calib_hand_scales: list[float] = []
    calib_open_dists: list[float] = []
    calib_state = "ready" if not calibration_enabled else "awaiting_start"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    init_w, init_h = max(180, preview_width), max(120, preview_height)
    if preview_width_percent > 0.0:
        sw0, _ = input_controller.screen_size()
        p = max(0.05, min(0.95, preview_width_percent))
        init_w = max(180, int(round(sw0 * p)))
    elif preview_size_percent > 0.0:
        sw0, sh0 = input_controller.screen_size()
        p = max(0.05, min(0.95, preview_size_percent))
        init_w = max(180, int(round(sw0 * p)))
        init_h = max(120, int(round(sh0 * p)))
    cv2.resizeWindow(window, init_w, init_h + max(90, hud_height) + max(0, hud_gap))
    try:
        if preview_always_on_top:
            cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass

    try:
        outer_x = 0
        outer_y = 0
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
            onboarding_line = ""
            calibration_progress: float | None = None

            if calibration_enabled and calib_state != "ready":
                state.control_enabled = False
                state.last_gesture = "calibrating"
                needed = max(15, calibration_frames)
                if calib_state == "awaiting_start":
                    onboarding_line = "calibration: press C to start"
                elif calib_state == "collecting":
                    if hand_detected:
                        lms0 = result.hand_landmarks[0]
                        hs0 = hand_scale(lms0)
                        dist0 = thumb_index_distance(lms0)
                        if hs0 > 1e-6:
                            calib_hand_scales.append(float(hs0))
                        if dist0 > pinch_thr * max(1.05, calibration_min_open_multiplier):
                            calib_open_dists.append(float(dist0))
                    calibration_progress = min(1.0, len(calib_hand_scales) / float(needed))
                    if len(calib_hand_scales) >= needed:
                        ref_hand_scale = float(statistics.median(calib_hand_scales))
                        if calib_open_dists:
                            open_med = float(statistics.median(calib_open_dists))
                            calibrated_pinch_thr = max(0.02, min(0.10, open_med * 0.40))
                        else:
                            calibrated_pinch_thr = pinch_thr
                        calib_state = "confirm"
                        calibration_progress = 1.0
                        onboarding_line = (
                            f"calibration done: thr={calibrated_pinch_thr:.3f} (Enter confirm / C redo)"
                        )
                    else:
                        if hand_detected:
                            onboarding_line = (
                                f"calibrating... {int(round(100*calibration_progress))}% keep hand open/steady"
                            )
                        else:
                            onboarding_line = "calibrating... show your hand to camera"
                elif calib_state == "confirm":
                    calibration_progress = 1.0
                    onboarding_line = (
                        f"calibration ready: thr={calibrated_pinch_thr:.3f} (Enter confirm / C redo)"
                    )

            if hand_detected and state.control_enabled and calib_state == "ready":
                lms = result.hand_landmarks[0]
                dist_l = thumb_index_distance(lms)
                hs = hand_scale(lms)

                pinch_thr_live = calibrated_pinch_thr
                if adaptive_thresholds and ref_hand_scale and hs > 1e-6:
                    ratio = float(hs) / float(ref_hand_scale)
                    ratio = max(adaptive_min_scale, min(adaptive_max_scale, ratio))
                    pinch_thr_live = calibrated_pinch_thr * ratio
                fe_enter_live = pinch_thr_live * freeze_enter_scale
                fe_exit_live = fe_enter_live * freeze_exit_scale
                pose_right = right_click_pose_active(lms)
                mid_y = 0.5 * (float(lms[8].y) + float(lms[12].y))
                index_y = float(lms[8].y)
                dy_for_mode = (
                    0.0 if state.last_hand_y is None else index_y - float(state.last_hand_y)
                )
                abs_dy_for_mode = abs(dy_for_mode)
                right_pose_stationary = abs(dy_for_mode) <= right_click_pose_motion_deadzone
                state.last_hand_y = index_y
                scroll_pose_active = scroll_gesture_active(
                    lms,
                    dist_l,
                    pinch_thr_live,
                    scroll_pinch_margin,
                    min_middle_tip_above_index_y=scroll_middle_above,
                )
                # The same 2-finger hand shape can mean either right-click or scroll.
                # Disambiguate by motion: stationary pose => right click, moving pose => scroll.
                scroll_active = bool(
                    scroll_pose_active and abs_dy_for_mode >= scroll_mode_motion_deadzone
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
                        threshold=pinch_thr_live,
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
                            enter_max=fe_enter_live,
                            exit_max=fe_exit_live,
                            drag_active=pr.drag_active,
                        )

                    if pr.pinch_just_ended or pr.drag_just_ended:
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
                        rr = right_click.update(
                            pose_right and right_pose_stationary,
                            hold_ms=right_click_pose_hold_ms,
                            cooldown_ms=right_cooldown_ms,
                            now=now_mono,
                            last_click_time=state.last_right_click_time,
                        )
                        right_freeze_on = bool(rr.pose_active)
                        right_freeze_started = bool(rr.pose_just_started)
                        if rr.pose_active:
                            state.last_gesture = "right index-hold"
                            hint_extra = "right: hold index finger steady"
                        else:
                            state.last_gesture = "move"
                            hint_extra = (
                                f"L: hold {int(click_hold_ms)} ms for click, {int(drag_start_ms)} ms drag | "
                                "R: steady index hold | scroll: 2 up + vertical motion"
                            )
                else:
                    state.scroll_mode = True
                    state.last_gesture = "scroll"
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

                open_rough = pinch_thr_live * max(1.25, pinch_release_scale) * 1.02
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
                        or pr.pinch_active
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
                            rr and rr.pose_hold_ready and right_freeze_on
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
                    if pr.click_fired:
                        stable = True
                        if (
                            cx is not None
                            and cy is not None
                            and state.last_cursor_x is not None
                            and state.last_cursor_y is not None
                        ):
                            stable = (
                                math.hypot(
                                    float(state.last_cursor_x) - float(cx),
                                    float(state.last_cursor_y) - float(cy),
                                )
                                <= click_stability_pixels
                            )
                        if not stable:
                            pr.click_fired = False
                    if pr.click_fired:
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
                    and rr.click_fired
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
                state.last_hand_y = None
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

            target_w = max(180, preview_width)
            target_h = max(120, preview_height)
            h_frame, w_frame = frame.shape[:2]
            if preview_width_percent > 0.0:
                p = max(0.05, min(0.95, preview_width_percent))
                target_w = max(180, int(round(screen_w * p)))
                target_h = max(120, int(round(target_w * (h_frame / max(1, w_frame)))))
                cv2.resizeWindow(window, target_w, target_h)
            elif preview_size_percent > 0.0:
                p = max(0.05, min(0.95, preview_size_percent))
                target_w = max(180, int(round(screen_w * p)))
                target_h = max(120, int(round(screen_h * p)))
                cv2.resizeWindow(window, target_w, target_h)
            frame_for_preview = cv2.flip(frame, 1) if preview_mirror else frame
            display_frame = cv2.resize(
                frame_for_preview, (target_w, target_h), interpolation=cv2.INTER_AREA
            )

            min_hud_h = max(90, hud_height)
            if calibration_enabled and calib_state != "ready":
                help_line = onboarding_line or "calibration pending"
                key_line = "c start/restart | enter confirm | q quit"
            else:
                help_line = (
                    hint_extra
                    if hint_extra and (hand_detected and state.control_enabled)
                    else "move:index | left:pinch hold | right:index hold | scroll:2-finger move"
                )
                key_line = "space toggle | esc off | q quit"
            ui_lines = [
                f"{'hand' if hand_detected else 'no hand'} | fps {fps_smooth:4.1f} | control {'ON' if state.control_enabled else 'OFF'}",
                f"gesture: {state.last_gesture}",
                help_line,
                key_line,
            ]

            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            line_step = 22
            pad_x = 10
            pad_y = 20
            max_text_w = max(60, target_w - (2 * pad_x))
            wrapped_lines: list[str] = []
            for line in ui_lines:
                wrapped_lines.extend(
                    _wrap_text_to_width(
                        line,
                        max_text_w,
                        font_face=font_face,
                        font_scale=font_scale,
                        thickness=thickness,
                    )
                )
            target_hud_h = max(min_hud_h, pad_y + line_step * len(wrapped_lines) + 8)
            hud_frame = np.full((target_hud_h, target_w, 3), 20, dtype=np.uint8)

            y = pad_y
            for text in wrapped_lines:
                cv2.putText(
                    hud_frame,
                    text,
                    (pad_x, y),
                    font_face,
                    font_scale,
                    (240, 240, 240),
                    thickness,
                    lineType=cv2.LINE_AA,
                )
                y += line_step

            if calibration_progress is not None:
                bar_x = pad_x
                bar_y = max(pad_y, target_hud_h - 16)
                bar_w = max(80, target_w - (2 * pad_x))
                bar_h = 8
                cv2.rectangle(
                    hud_frame,
                    (bar_x, bar_y),
                    (bar_x + bar_w, bar_y + bar_h),
                    (80, 80, 80),
                    1,
                    lineType=cv2.LINE_AA,
                )
                fill_w = int(round((bar_w - 2) * max(0.0, min(1.0, calibration_progress))))
                if fill_w > 0:
                    cv2.rectangle(
                        hud_frame,
                        (bar_x + 1, bar_y + 1),
                        (bar_x + 1 + fill_w, bar_y + bar_h - 1),
                        (80, 210, 120),
                        -1,
                        lineType=cv2.LINE_AA,
                    )

            gap_px = max(0, hud_gap)
            if gap_px > 0:
                spacer = np.full((gap_px, target_w, 3), 12, dtype=np.uint8)
                combined = np.vstack((display_frame, spacer, hud_frame))
            else:
                combined = np.vstack((display_frame, hud_frame))

            win_w = int(combined.shape[1])
            win_h = int(combined.shape[0])
            cv2.resizeWindow(window, win_w, win_h)
            cv2.imshow(window, combined)
            rect_x = rect_y = 0
            try:
                rect_x, rect_y, w, h = cv2.getWindowImageRect(window)
                if w > 0 and h > 0:
                    win_w, win_h = int(w), int(h)
            except cv2.error:
                pass

            if preview_anchor == "bottom-left":
                px = max(0, preview_margin)
                py = max(0, screen_h - win_h - preview_margin)
            elif preview_anchor == "top-left":
                px = max(0, preview_margin)
                py = max(0, preview_margin)
            elif preview_anchor == "bottom-right":
                px = max(0, screen_w - win_w - preview_margin)
                py = max(0, screen_h - win_h - preview_margin)
            elif preview_anchor == "top-right":
                px = max(0, screen_w - win_w - preview_margin)
                py = max(0, preview_margin)
            else:
                px = max(0, preview_margin)
                py = max(0, screen_h - win_h - preview_margin)

            # Correct for platform window chrome/title-bar offsets by nudging from
            # the observed image-rect position instead of assuming outer==inner.
            dx = int(px) - int(rect_x)
            dy = int(py) - int(rect_y)
            outer_x = int(max(0, outer_x + dx))
            outer_y = int(max(0, outer_y + dy))
            cv2.moveWindow(window, outer_x, outer_y)
            try:
                if preview_always_on_top:
                    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
            except cv2.error:
                pass
            key = cv2.waitKey(1) & 0xFF
            action = interpret_preview_key(key)
            if action.quit_app:
                break
            if action.start_calibration and calibration_enabled:
                calib_state = "collecting"
                calib_hand_scales.clear()
                calib_open_dists.clear()
                state.control_enabled = False
                prev_scroll_active = False
                state.last_hand_y = None
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
            if action.confirm_calibration and calibration_enabled and calib_state == "confirm":
                calib_state = "ready"
                state.last_gesture = "idle"
            if action.toggle_control:
                if calibration_enabled and calib_state != "ready":
                    continue
                state.control_enabled = not state.control_enabled
                if not state.control_enabled:
                    prev_scroll_active = False
                    state.last_hand_y = None
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
                state.last_hand_y = None
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
