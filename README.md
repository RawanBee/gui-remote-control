# Gesture Mouse

Webcam → **MediaPipe Hand Landmarker** → on-screen preview with **gesture-driven** mouse control (macOS + Windows) via **PyAutoGUI**. No extra hardware—just a hand in the camera frame.

## What it does

| Gesture | Action |
|--------|--------|
| Index tip | Move cursor (mapped + smoothed; optional dead zone) |
| Thumb–index pinch | **Left** click on hold (no release required) or **drag** if held longer |
| Index finger up (held steady) | **Right** click |
| Index + middle up, ring + pinky down (move vertically) | **Scroll** (vertical) |
| HUD / preview | Landmarks, status, pinch “lock” / hold / drag rings |

Hotkeys: **Space** toggles control, **Esc** turns control off, **q** quits.

Tuning lives in **`config.json`** (thresholds, smoothing, scroll gain, optional `left_pinch_lock_cursor`, etc.).

Preview window placement is also configurable (`preview_width`, `preview_height`, `preview_width_percent`, `preview_size_percent`, `preview_margin`, `preview_anchor`, `preview_always_on_top`). Status text can be shown in a separate attached HUD window using `hud_height` and `hud_gap`.

## Setup

1. Python **3.11 or 3.12** recommended.
2. Venv + dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the MediaPipe task model:

```bash
python scripts/download_model.py
```

This writes `model/hand_landmarker.task`.

## Run

```bash
python app.py
```

The first `mediapipe` import may build matplotlib’s font cache once; later starts are faster.

### Demo mode

Use a safer pre-tuned profile for live demos:

```bash
bash scripts/run_demo.sh
```

Or run directly:

```bash
python app.py --config config.demo.json
```

Demo HUD always shows a gesture cheat-sheet so viewers can follow along.

If calibration is enabled, onboarding is explicit:
- Press `c` to start calibration
- Hold an open, steady hand until progress reaches 100%
- Press `Enter` to confirm (or `c` to recalibrate)

## Permissions (OS)

- **macOS**: Camera + **Accessibility** (required for PyAutoGUI to move the pointer).
- **Windows**: Allow camera access for desktop apps.

## Layout

| Path | Role |
|------|------|
| `app.py` | Main loop: camera, gestures, pointer, HUD |
| `config.json` | Thresholds and tuning |
| `src/gesture_engine.py` | Pinch / scroll / timing state machines |
| `src/input_controller.py` | Pointer, clicks, drag, scroll |
| `src/cursor_mapper.py` | Normalized tip → screen (dead zone, mirror) |
| `src/hand_tracker.py` | MediaPipe wrapper |
| `src/overlay.py` | Landmarks + interaction rings |
| `src/state.py` | Shared runtime state |

## License

This repository is **proprietary**. See [`LICENSE`](LICENSE). **No use, copying, modification, distribution, or sublicensing is allowed without prior written permission from the copyright holder.**

## Packaging (later)

Build with **PyInstaller on each OS separately** once gesture defaults are stable.
