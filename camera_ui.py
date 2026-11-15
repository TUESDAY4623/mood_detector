"""camera_ui.py — live webcam emotion detector on the NPU.

Opens the laptop camera, detects faces, and predicts the dominant emotion of
each face with the OpenVINO NPU model that export_openvino.py produced. It
targets a steady 45 fps display.

Run it
-----
    python camera_ui.py                  # default: NPU, 45 fps target
    python camera_ui.py --device CPU     # fall back to CPU if NPU chokes
    python camera_ui.py --no-face        # predict on the whole frame (no crop)
    python camera_ui.py --max-frames 300 # exit after N frames (CI / headless)

Keys while the window is open:
    [space] pause   [s] save a frame   [q / ESC] quit

If the NPU cannot run the graph, it falls back to CPU automatically so the
camera still works.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent


def app_base_dir() -> Path:
    """Directory the app's bundled resources live in.

    Normal script run  -> the source folder (model_ir/ sits next to it).
    PyInstaller onefile -> files are unpacked to sys._MEIPASS at runtime,
    so resources must be looked up there instead.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return PROJECT_DIR


def default_ir_path() -> Path:
    return app_base_dir() / "model_ir" / "emotion_npu.xml"


def default_classes_path() -> Path:
    return app_base_dir() / "model_ir" / "classes.json"

# Emotion display colours (BGR) so the labels read at a glance.
EMOTION_COLOR = {
    "angry": (0, 0, 200),
    "confused": (220, 140, 0),
    "disgusted": (140, 140, 140),
    "fear": (190, 90, 60),
    "happy": (0, 200, 80),
    "neutral": (200, 200, 200),
    "sad": (230, 120, 60),
    "shy": (240, 100, 220),
    "surprised": (90, 200, 150),
}


class EmotionModel:
    """Wraps the OpenVINO model with device auto-fallback + image scoring."""

    def __init__(self, ir_xml: str, classes: list[str], device: str = "NPU"):
        import openvino as ov

        core = ov.Core()
        self._device = device.upper()
        try:
            self._model = core.compile_model(ir_xml, self._device)
        except Exception as e:  # noqa: BLE001
            print(f"[model] {self._device} failed to compile ({e}); using CPU.")
            self._device = "CPU"
            self._model = core.compile_model(ir_xml, "CPU")
        self.classes = classes
        print(f"[model] running on {self._device}")

    def predict_image(self, bgr: np.ndarray) -> tuple[str, float, np.ndarray]:
        """Return (emotion, confidence, probs) for a BGR image."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = rgb.astype("float32") / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]  # NCHW
        logits = self._model([tensor])[self._model.output(0)]
        probs = _softmax(logits[0].astype("float32"))
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx]), probs


class FaceDetector:
    """Haar-cascade face detector (ships with opencv-python 4.x)."""

    def __init__(self) -> None:
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self._cascade.empty():
            raise RuntimeError("Could not load the Haar face-cascade XML.")

    def detect(self, bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(96, 96), flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [(int(x), int(y), int(w), int(h), 1.0) for (x, y, w, h) in faces]


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


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _draw(
    frame: np.ndarray,
    faces: list[tuple[int, int, int, int, float]],
    results: list[tuple[str, float]],
    fps: float,
    device: str,
    target_fps: int,
    paused: bool,
) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (8, 8), (w - 8, h - 8), (255, 255, 255), 2)
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    status = f"fps {fps:5.1f} (target {target_fps})   device {device}"
    if paused:
        status += "   [PAUSED]"
    cv2.putText(frame, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for (x, y, ww, hh, _conf), (emo, conf) in zip(faces, results):
        col = EMOTION_COLOR.get(emo, (0, 255, 0))
        cv2.rectangle(frame, (x, y), (x + ww, y + hh), col, 2)
        label = f"{emo} {conf*100:.0f}%"
        pad = 6
        cv2.rectangle(frame, (x, max(0, y - 28 - pad)), (x + len(label) * 14, y - pad),
                      col, -1)
        cv2.putText(frame, label, (x, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="NPU", help="NPU / CPU / GPU")
    ap.add_argument("--ir", default=str(default_ir_path()), help="Path to emotion_npu.xml")
    ap.add_argument("--classes", default=str(default_classes_path()), help="Path to classes.json")
    ap.add_argument("--target-fps", type=int, default=45)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Quit after N frames (0 = run until closed)")
    ap.add_argument("--no-face", action="store_true",
                    help="Predict on the whole frame instead of per-face crops")
    args = ap.parse_args()

    classes_doc = json.loads(Path(args.classes).read_text())
    classes = classes_doc["classes"]
    model = EmotionModel(args.ir, classes, device=args.device)
    try:
        face = None if args.no_face else FaceDetector()
    except RuntimeError as e:
        print(f"[camera] face detector unavailable ({e}); using whole-frame mode.")
        face = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the laptop camera. Check that no other "
                         "app is using it, and that you have camera permission.")

    fps = FpsMeter()
    paused = False
    frame_idx = 0

    # Throttle so the whole loop paces toward target_fps.
    frame_period = 1.0 / args.target_fps if args.target_fps > 0 else 0.0

    print(f"[camera] open (device={model._device}, target_fps={args.target_fps}). "
          f"Press SPACE to pause, S to snapshot, Q to quit.")
    while True:
        loop_start = time.time()
        ok, frame = cap.read()
        if not ok:
            print("[camera] frame read failed; retrying.")
            time.sleep(0.1)
            continue

        faces = face.detect(frame) if face else [(0, 0, frame.shape[1], frame.shape[0], 1.0)]
        results: list[tuple[str, float]] = []
        for (fx, fy, fw, fh, _fconf) in faces:
            crop = frame[fy:fy + fh, fx:fx + fw]
            if crop.size == 0:
                continue
            emo, conf, _ = model.predict_image(crop)
            results.append((emo, conf))

        fps_val = fps.tick()
        _draw(frame, faces, results, fps_val, model._device, args.target_fps, paused)
        cv2.imshow("Emotion detector (NPU)", frame)

        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            print(f"[camera] reached --max-frames={args.max_frames}; exiting.")
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
            out = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(out, frame)
            print(f"[camera] saved {out}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
