"""Gesture-controlled master volume.

Track one hand with MediaPipe's HandLandmarker (Tasks API), measure the
thumb-to-index pinch distance, and map it to the macOS system volume in real
time. Pinch fingers together to lower the volume, spread them apart to raise it.

Press 'q' or ESC to quit.
"""

import math
import os
import subprocess
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

# --- Config -----------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# Hand-landmark indices (MediaPipe hand model).
WRIST, INDEX_MCP, THUMB_TIP, INDEX_TIP = 0, 5, 4, 8

# The pinch distance is normalized by hand size (wrist -> index knuckle) so the
# mapping stays stable as the hand moves toward or away from the camera. These
# ratio bounds were tuned on a built-in webcam; nudge them if the full 0-100
# range feels hard to reach for your camera/hand.
PINCH_RATIO_MIN = 0.30   # fingers together  -> 0% volume
PINCH_RATIO_MAX = 1.60   # fingers apart     -> 100% volume
SMOOTHING = 0.5          # EMA factor; higher = smoother but laggier
SET_INTERVAL = 0.05      # min seconds between osascript volume calls


def set_system_volume(pct: float) -> None:
    """Set the macOS output volume (0-100)."""
    pct = int(max(0, min(100, pct)))
    subprocess.run(
        ["osascript", "-e", f"set volume output volume {pct}"],
        check=False,
    )


def make_landmarker() -> HandLandmarker:
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"Model not found at {MODEL_PATH}. Download it with:\n"
            "  curl -L -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return HandLandmarker.create_from_options(options)


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam (index 0).")

    landmarker = make_landmarker()

    volume = 0.0          # smoothed volume we display/apply
    last_sent = -1        # last integer volume pushed to the system
    last_send_time = 0.0
    prev_time = time.time()
    start = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror for natural interaction
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Tasks VIDEO mode needs a monotonically increasing timestamp (ms).
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            hand_found = bool(result.hand_landmarks)
            if hand_found:
                lm = result.hand_landmarks[0]
                pts = [(int(p.x * w), int(p.y * h)) for p in lm]
                thumb_tip = pts[THUMB_TIP]
                index_tip = pts[INDEX_TIP]
                wrist = pts[WRIST]
                index_mcp = pts[INDEX_MCP]

                # Euclidean pinch distance: d = sqrt((x2-x1)^2 + (y2-y1)^2)
                pinch = math.hypot(
                    index_tip[0] - thumb_tip[0],
                    index_tip[1] - thumb_tip[1],
                )
                # Reference length for scale-invariance.
                ref = math.hypot(
                    index_mcp[0] - wrist[0],
                    index_mcp[1] - wrist[1],
                )
                ratio = pinch / ref if ref > 0 else 0.0

                target = float(
                    np.interp(ratio, [PINCH_RATIO_MIN, PINCH_RATIO_MAX], [0, 100])
                )
                volume = SMOOTHING * volume + (1 - SMOOTHING) * target

                # Throttle system calls: only on a real change, not every frame.
                now = time.time()
                rounded = int(round(volume))
                if rounded != last_sent and (now - last_send_time) >= SET_INTERVAL:
                    set_system_volume(rounded)
                    last_sent = rounded
                    last_send_time = now

                # Draw the hand skeleton (simple connections).
                draw_hand(frame, pts)

                # Visual feedback on the pinch.
                cx = (thumb_tip[0] + index_tip[0]) // 2
                cy = (thumb_tip[1] + index_tip[1]) // 2
                cv2.circle(frame, thumb_tip, 12, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, index_tip, 12, (0, 255, 0), cv2.FILLED)
                cv2.line(frame, thumb_tip, index_tip, (0, 255, 0), 3)
                pinch_color = (0, 0, 255) if volume <= 2 else (255, 0, 255)
                cv2.circle(frame, (cx, cy), 8, pinch_color, cv2.FILLED)

            draw_hud(frame, volume, hand_found)

            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now
            cv2.putText(frame, f"FPS: {int(fps)}", (w - 140, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Gesture Volume Control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


# Minimal hand connections (the Tasks API has no built-in drawing util here).
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm base
]


def draw_hand(frame, pts) -> None:
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for p in pts:
        cv2.circle(frame, p, 3, (0, 200, 255), cv2.FILLED)


def draw_hud(frame, volume: float, hand_found: bool) -> None:
    h, w = frame.shape[:2]
    bar_x, bar_top, bar_bottom = 50, 100, 400
    bar_h = bar_bottom - bar_top
    fill = int(bar_bottom - (volume / 100) * bar_h)
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 35, bar_bottom),
                  (200, 200, 200), 2)
    cv2.rectangle(frame, (bar_x, fill), (bar_x + 35, bar_bottom),
                  (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, f"{int(round(volume))}%", (bar_x - 5, bar_bottom + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if not hand_found:
        cv2.putText(frame, "No hand detected", (w // 2 - 130, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, "Pinch to control volume  |  'q' to quit",
                (50, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)


if __name__ == "__main__":
    main()
