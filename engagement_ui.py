"""engagement_ui.py — live webcam engagement detector (8 states).

Opens the laptop camera, extracts face landmarks, and decides which of the
8 engagement states the user is in — using landmark signals + rules
(MediaPipe FaceLandmarker), not a CNN, so it works with no trained model
and no NPU. It targets a steady 45 fps display.

States
------
    focused, confused, distracted, neutral, curious, thinking, bored, sleepy

Run it
-----
    python engagement_ui.py                  # live, 45 fps target
    python engagement_ui.py --max-frames 300  # CI / headless
    python engagement_ui.py --no-face         # hide the landmark overlay

Keys while the window is open:
    [space] pause   [s] save a frame   [q / ESC] quit

If the laptop has no camera, the program says so and exits cleanly.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from engagement_detector import EngagementDetector, STATES

# Per-state display colour (BGR).
STATE_COLOR = {
    "focused":    (0, 120, 255),
    "confused":   (220, 140, 0),
    "distracted": (0, 0, 200),
    "neutral":    (180, 180, 180),
    "curious":    (90, 200, 150),
    "thinking":   (230, 120, 60),
    "bored":      (140, 140, 140),
    "sleepy":     (230, 90, 60),
}


class FpsMeter:
    """Rolling-FPS estimator using the last ``n`` frame timestamps."""

    def __init__(self, n: int = 40) -> None:
        self._stamps: list[float] = []
        self._n = n

    def tick(self) -> float:
        now = time.time()
        self._stamps.append(now)
        if len(self._stamps) > self._n:
            self._stamps.pop(0)
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 1e-6 else 0.0


def _draw_mesh(frame: np.ndarray, pts: np.ndarray) -> None:
    """Draw the 478 landmark points (normalised xy) as a light dot mesh."""
    h, w = frame.shape[:2]
    for p in pts:
        x, y = int(p[0] * w), int(p[1] * h)
        cv2.circle(frame, (x, y), 1, (0, 120, 255), -1)


def _draw(frame: np.ndarray, state: str, conf: float, fps: float,
          target_fps: int, paused: bool, evidence: dict) -> None:
    h, w = frame.shape[:2]
    col = STATE_COLOR.get(state, (0, 255, 0))
    # Top status band.
    cv2.rectangle(frame, (0, 0), (w, 46), (0, 0, 0), -1)
    label = f"{state.upper()}  {conf*100:.0f}%"
    cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 2)
    info = f"fps {fps:5.1f} (target {target_fps})   [SPACE]pause [S]shot [Q]quit"
    if paused:
        info += "   PAUSED"
    cv2.putText(frame, info, (w - 470, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    # Evidence line (the primitives that drove the decision).
    ev = (f"closed {evidence.get('closed_frac',0):.0%}  away {evidence.get('away_frac',0):.0%}  "
          f"down {evidence.get('down_frac',0):.0%}  up {evidence.get('up_frac',0):.0%}  "
          f"yawn {'yes' if evidence.get('yawn') else 'no'}")
    cv2.putText(frame, ev, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-fps", type=int, default=45)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Quit after N frames (0 = run until closed)")
    ap.add_argument("--no-face", action="store_true",
                    help="Hide the landmark face mesh in the overlay")
    ap.add_argument("--window", type=float, default=10.0,
                    help="Rolling window (seconds) for state rules")
    args = ap.parse_args()

    det = EngagementDetector(window_seconds=args.window)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the laptop camera. Check that no other "
                         "app is using it, and that you have camera permission.")

    fps = FpsMeter()
    paused = False
    frame_idx = 0
    frame_period = 1.0 / args.target_fps if args.target_fps > 0 else 0.0

    print(f"[engagement] open (target_fps={args.target_fps}, window={args.window}s). "
          f"States: {', '.join(STATES)}")
    print("[engagement] SPACE=pause  S=snapshot  Q/ESC=quit")
    while True:
        loop_start = time.time()
        ok, frame = cap.read()
        if not ok:
            print("[engagement] frame read failed; retrying.")
            time.sleep(0.1)
            continue

        res = None if paused else det.update(frame)
        state = res.state if res else "neutral"
        conf = res.confidence if res else 0.0
        ev = res.evidence if res else {}

        if not args.no_face:
            mesh = det.last_landmarks()
            if mesh is not None:
                _draw_mesh(frame, mesh)

        fps_val = fps.tick()
        _draw(frame, state, conf, fps_val, args.target_fps, paused, ev)
        cv2.imshow("Engagement detector (8 states)", frame)

        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            print(f"[engagement] reached --max-frames={args.max_frames}; exiting.")
            break
        if frame_period:
            elapsed = time.time() - loop_start
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == ord("p") or key == ord(" "):
            paused = not paused
        if key == ord("s"):
            out = f"engagement_snapshot_{int(time.time())}.png"
            cv2.imwrite(out, frame)
            print(f"[engagement] saved {out}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
