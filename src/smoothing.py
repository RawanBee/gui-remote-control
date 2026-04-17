"""Low-pass filter for pointer coordinates."""


def smooth_pointer(
    x: float,
    y: float,
    prev_x: float | None,
    prev_y: float | None,
    pointer_smoothing: float,
) -> tuple[float, float]:
    """
    Exponential blend toward the previous sample.

    pointer_smoothing: weight on the *previous* sample in [0, 1).
    0 = no smoothing (raw), higher = smoother / more lag.
    """
    w_prev = max(0.0, min(0.95, float(pointer_smoothing)))
    w_new = 1.0 - w_prev
    if prev_x is None or prev_y is None:
        return x, y
    return w_new * x + w_prev * prev_x, w_new * y + w_prev * prev_y
