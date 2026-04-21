# Gesture Mouse

Webcam → **MediaPipe Hand Landmarker** → on-screen preview with **gesture-driven** mouse control (macOS + Windows) via **PyAutoGUI**. No extra hardware—just a hand in the camera frame.

## What it does

| Gesture | Action |
|--------|--------|
| Index tip | Move cursor (mapped + smoothed; optional dead zone) |
| Thumb–index pinch | **Left** click on release (hold timing aligns with right by default) or **drag** if held longer |
| Thumb–middle pinch (index not pinching index) | **Right** click on release |
| Index + middle up, ring + pinky down | **Scroll** (vertical) |
| HUD / preview | Landmarks, status, pinch “lock” / hold / drag rings |

Hotkeys: **Space** toggles control, **Esc** turns control off, **q** quits.

Tuning lives in **`config.json`** (thresholds, smoothing, scroll gain, optional `left_pinch_lock_cursor`, etc.).

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
