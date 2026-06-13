# Gesture-Controlled Volume

Control your macOS master volume with hand gestures in front of the webcam.
A [MediaPipe](https://github.com/google/mediapipe) hand tracker measures the
**pinch distance** between your thumb and index finger and maps it to the
system volume in real time:

- Fingers **together** → volume down
- Fingers **apart** → volume up

The pinch distance is the Euclidean distance between the fingertips,
`d = sqrt((x2-x1)^2 + (y2-y1)^2)`, normalized by hand size so it stays stable as
your hand moves toward or away from the camera, then mapped to `0–100` with
`np.interp`.

## Requirements

- macOS (volume is set via `osascript`)
- Python 3.12 (pinned in `.python-version`; MediaPipe has no 3.13/3.14 wheels yet)
- [uv](https://docs.astral.sh/uv/)

## Setup

Install dependencies and download the hand-landmark model (~7.8 MB, not checked
into git):

```bash
uv sync
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Run

```bash
uv run main.py
```

A mirrored webcam window opens with a live volume bar and FPS readout. Hold up
one hand and pinch to adjust the volume. Press **`q`** or **ESC** to quit.

> **First run — camera permission:** macOS will ask the app running this script
> (your terminal, iTerm, VS Code, etc.) for camera access. Approve it under
> **System Settings → Privacy & Security → Camera**, then re-run.

## Implementation notes

This uses MediaPipe's modern **Tasks API** (`HandLandmarker`), not the legacy
`mp.solutions.hands` API — recent MediaPipe builds ship only the Tasks API. The
model file is loaded from `hand_landmarker.task` in the project root.

## Tuning

If the full 0–100% range is hard to reach on your camera, adjust
`PINCH_RATIO_MIN` / `PINCH_RATIO_MAX` near the top of `main.py`. Increase
`SMOOTHING` for steadier (but laggier) response.
