"""mood_ui.py — live camera app: detects all 8 moods in one pass.

Uses FusedDetector (landmark rules + NPU CNN) so every one of the 8
states is scored live:
    focused, confused, distracted, neutral, curious, thinking, bored, sleepy

The window shows the current state banner, 8 confidence bars, an FPS
readout (target 45), and the NPU device in use.

Run it
-----
    python mood_ui.py                # live, NPU, 45 fps target
    python mood_ui.py --no-cnn       # rules only (no NPU)
    python mood_ui.py --max-frames 300

Keys: [space] pause   [s] snapshot   [q / ESC] quit

Headless / replay modes (testing and evidence capture, no GUI needed):
    python mood_ui.py --headless --max-frames 30 --log-file run.jsonl
    python mood_ui.py --source clip.mp4 --headless              # replay a video
    python mood_ui.py --source datasets/engagement/aligned/test/neutral --headless
"""

from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np

from fused_detector import FusedDetector, all_states

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

BAR_W = 160
BAR_H = 10
PANEL_W = 240
PANEL_H = 190


class FpsMeter:
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


def _draw_panel(frame: np.ndarray, state: str, conf: float,
                scores: dict, fps: float, target_fps: int,
                device: str, latency_ms: float, face_found: bool) -> None:
    h, w = frame.shape[:2]
    x0 = w - PANEL_W - 12
    y0 = 12
    # Panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + PANEL_W, y0 + PANEL_H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + PANEL_W, y0 + PANEL_H), (80, 80, 80), 1)

    col = STATE_COLOR.get(state, (0, 255, 0))
    banner = f"{state.upper()}" if face_found else "NO FACE"
    cv2.putText(frame, banner, (x0 + 10, y0 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    cv2.putText(frame, f"{conf*100:.0f}%  fps {fps:4.1f} (tg {target_fps})",
                (x0 + 10, y0 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 255, 0), 1)
    cv2.putText(frame, f"device {device} | {latency_ms:.0f} ms",
                (x0 + 10, y0 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # 8 bars
    yb = y0 + 78
    for i, s in enumerate(all_states()):
        v = float(scores.get(s, 0.0))
        bx, by = x0 + 90, yb + i * 16
        cv2.putText(frame, s, (x0 + 8, by + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (230, 230, 230), 1)
        cv2.rectangle(frame, (bx, by - 4), (bx + BAR_W, by + 4), (60, 60, 60), -1)
        cv2.rectangle(frame, (bx, by - 4),
                      (int(bx + BAR_W * min(v, 1.0)), by + 4), STATE_COLOR[s], -1)


def _open_source(src: str):
    """Frame source: live camera (default), a video file, or an image dir."""
    if not src:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise SystemExit("Could not open the laptop camera.")
        return cap, False
    if os.path.isdir(src):
        files = sorted(
            os.path.join(src, f) for f in os.listdir(src)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
        if not files:
            raise SystemExit(f"No images found in {src}")
        return DirSource(files), True
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {src}")
    return cap, True


class DirSource:
    """Replays a directory of images in sorted order, looping forever."""

    def __init__(self, files: list):
        self._files = files
        self._i = 0

    def read(self):
        img = cv2.imread(self._files[self._i])
        self._i = (self._i + 1) % len(self._files)
        return (img is not None), (img if img is not None else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-fps", type=int, default=45)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-cnn", action="store_true",
                    help="Skip the NPU CNN (rules only)")
    ap.add_argument("--device", default="NPU", help="NPU / CPU / GPU for the CNN")
    ap.add_argument("--source", default="",
                    help="Video file or directory of images to replay; "
                         "empty = live camera")
    ap.add_argument("--headless", action="store_true",
                    help="No GUI window (run detector + log only)")
    ap.add_argument("--log-file", default="",
                    help="Write one JSON object per logged frame (state, all 8 "
                         "scores, fps, device, latency) to this path")
    args = ap.parse_args()

    det = FusedDetector(use_cnn=not args.no_cnn, device=args.device)
    cap, replay = _open_source(args.source)
    if not replay and not args.source:
        # Ask the webcam for the target rate; hardware-delivered (many
        # webcams deliver 30 fps regardless).
        cap.set(cv2.CAP_PROP_FPS, float(args.target_fps))
    log_fh = open(args.log_file, "a", encoding="utf-8") if args.log_file else None

    fps = FpsMeter()
    paused = False
    idx = 0
    period = 1.0 / args.target_fps if args.target_fps > 0 else 0.0

    print(f"[mood] open (device={det.device or 'rules-only'}, "
          f"target_fps={args.target_fps}, source={args.source or 'live camera'})")
    if not args.headless:
        print("[mood] SPACE=pause  S=snapshot  Q/ESC=quit")

    frame = None
    while True:
        t_loop = time.time()
        ok, new_frame = cap.read()
        if not ok:
            if isinstance(cap, DirSource):
                continue  # looped around; keep replaying
            if replay:
                print("[mood] end of source; exiting.")
                break
            time.sleep(0.1)
            continue
        frame = new_frame

        res = None if paused else det.update(frame)
        if res is not None:
            _fps = fps.tick()
            if not args.headless:
                _draw_panel(frame, res.state, res.confidence, res.scores,
                            _fps, args.target_fps,
                            det.device or "rules", res.latency_ms, res.face_found)
            if log_fh is not None:
                rec = {"idx": idx, "state": res.state, "conf": round(res.confidence, 3),
                       "scores": {k: round(v, 3) for k, v in res.scores.items()},
                       "cnn_scores": {k: round(v, 3) for k, v in res.cnn_scores.items()},
                       "rules_scores": {k: round(v, 3) for k, v in res.rules_scores.items()},
                       "device": res.device, "fps": round(_fps, 2),
                       "latency_ms": round(res.latency_ms, 1), "face_found": res.face_found}
                log_fh.write(json.dumps(rec) + "\n")
            if idx % 45 == 0:
                print(f"[mood] f{idx} {res.state} {res.confidence:.2f} "
                      f"fps={_fps:.1f} dev={res.device} lat={res.latency_ms:.0f}ms")
        if not args.headless:
            cv2.imshow("Mood detector (8 states, NPU)", frame)

        idx += 1
        if args.max_frames and idx >= args.max_frames:
            print(f"[mood] reached --max-frames={args.max_frames}; exiting.")
            break
        if period:
            left = period - (time.time() - t_loop)
            if left > 0:
                time.sleep(left)
        if not args.headless:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("p"), ord(" ")):
                paused = not paused
            if key == ord("s"):
                out = f"mood_snapshot_{int(time.time())}.png"
                cv2.imwrite(out, frame)
                print(f"[mood] saved {out}")
        elif args.max_frames:
            # In headless replay mode without --max-frames there is no way to
            # stop (no keyboard), so require one.
            pass

    if hasattr(cap, "release"):
        cap.release()
    if log_fh is not None:
        log_fh.close()
        print(f"[mood] evidence log: {args.log_file}")
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
