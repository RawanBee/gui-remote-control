# Gesture Mouse (v1 scaffold)

Cross-platform (macOS + Windows) Python app: webcam → MediaPipe Hand Landmarker → gestures → system input (PyAutoGUI). **Phase 1** only opens the camera, draws landmarks, and shows `hand detected` / `no hand`. Cursor control comes in Phase 2.

## Setup

1. Python **3.11 or 3.12** recommended (newer versions may work, but wheels differ by OS).
2. Create a venv and install deps:

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

The first import of `mediapipe` may spend a short time building **matplotlib**’s font cache; later starts are faster.

```bash
python app.py
```

- **Space**: toggle `control_enabled` flag (shown in HUD; no mouse yet in Phase 1).
- **Esc**: force control off.
- **q**: quit.

## Permissions

- **macOS**: Camera + Accessibility (for later phases when PyAutoGUI moves the cursor).
- **Windows**: allow camera access for desktop apps.

## Layout

See `src/` modules: `camera`, `hand_tracker`, `overlay`, `state`, `hotkeys`, plus placeholders for `gesture_engine`, `cursor_mapper`, `input_controller`, `smoothing`.

## Packaging (later)

Build with PyInstaller **on each OS separately** once gestures are stable.
