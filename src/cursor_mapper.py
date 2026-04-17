"""Map normalized hand coordinates to screen pixels (dead zone + optional mirror)."""

from __future__ import annotations


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def apply_dead_zone(
    norm_x: float,
    norm_y: float,
    dead_zone_fraction: float,
) -> tuple[float, float]:
    """Map [0,1] camera space into [0,1] pointer space using an inner rectangle."""
    dz = max(0.0, min(0.49, float(dead_zone_fraction)))
    span = 1.0 - 2.0 * dz
    if span <= 1e-6:
        return 0.5, 0.5
    u = _clamp01((norm_x - dz) / span)
    v = _clamp01((norm_y - dz) / span)
    return u, v


def map_to_screen(
    u: float,
    v: float,
    screen_width: int,
    screen_height: int,
) -> tuple[int, int]:
    """Map [0,1] pointer space to integer pixel coordinates (top-left origin)."""
    sw = max(int(screen_width), 1)
    sh = max(int(screen_height), 1)
    x = int(round(u * float(sw - 1)))
    y = int(round(v * float(sh - 1)))
    return x, y


def index_tip_to_screen(
    tip_x: float,
    tip_y: float,
    *,
    screen_width: int,
    screen_height: int,
    dead_zone_fraction: float,
    mirror_camera_x: bool,
) -> tuple[int, int]:
    xn = (1.0 - tip_x) if mirror_camera_x else tip_x
    u, v = apply_dead_zone(xn, tip_y, dead_zone_fraction)
    return map_to_screen(u, v, screen_width, screen_height)
