"""predict.py — score a photo (or video) with the NPU emotion model.

Predicts the dominant emotion of a face in an image (or every N frames of a
video) using the OpenVINO model from export_openvino.py.

Usage
-----
    python predict.py photo.jpg
    python predict.py photo.jpg --device CPU
    python predict.py video.mp4 --every 30      # sample every 30 frames
    python predict.py photo.jpg --save annotated.jpg
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_IR = PROJECT_DIR / "model_ir" / "emotion_npu.xml"
DEFAULT_CLASSES = PROJECT_DIR / "model_ir" / "classes.json"


def load_model(ir_xml: str, classes: list[str], device: str = "NPU"):
    import openvino as ov

    core = ov.Core()
    try:
        return core.compile_model(ir_xml, device.upper()), device.upper()
    except Exception:
        print(f"[predict] {device} unavailable; falling back to CPU.")
        return core.compile_model(ir_xml, "CPU"), "CPU"


def score(model, classes: list[str], bgr: np.ndarray) -> tuple[str, float, np.ndarray]:
    rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (224, 224),
                     interpolation=cv2.INTER_AREA)
    tensor = (rgb.astype("float32") / 255.0).transpose(2, 0, 1)[None, ...]
    logits = model([tensor])[model.output(0)]
    z = logits[0].astype("float32")
    z = z - z.max()
    probs = np.exp(z) / np.exp(z).sum()
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx]), probs


def find_face(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(96, 96))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return int(x), int(y), int(w), int(h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="Image or video file to predict on")
    ap.add_argument("--ir", default=str(DEFAULT_IR))
    ap.add_argument("--classes", default=str(DEFAULT_CLASSES))
    ap.add_argument("--device", default="NPU", help="NPU / CPU / GPU")
    ap.add_argument("--every", type=int, default=0,
                    help="For videos: run on every Nth frame (0 = first frame only)")
    ap.add_argument("--no-face", action="store_true",
                    help="Score the whole frame instead of the detected face")
    ap.add_argument("--save", default="", help="Save the annotated image here")
    args = ap.parse_args()

    src = Path(args.path)
    if not src.exists():
        raise SystemExit(f"File not found: {src}")
    classes = json.loads(Path(args.classes).read_text())["classes"]
    model, dev = load_model(args.ir, classes, args.device)
    print(f"[predict] device={dev}  file={src.name}")

    is_video = src.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {src} as image or video.")

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if is_video and args.every and n and n % args.every != 0:
            continue
        region = frame
        box = None
        if not args.no_face:
            box = find_face(frame)
            if box is not None:
                x, y, w, h = box
                region = frame[y:y + h, x:x + w]
        t0 = time.perf_counter()
        emo, conf, probs = score(model, classes, region)
        ms = (time.perf_counter() - t0) * 1000
        tag = f"face crop {box}" if box is not None else "whole frame"
        print(f"  frame {n:4d} ({tag}): {emo} {conf*100:.1f}%  [{ms:5.1f} ms]")

        if args.save:
            out = frame.copy()
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(out, f"{emo} {conf*100:.0f}%", (x, max(18, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(out, f"{emo} {conf*100:.0f}%", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(args.save, out)
            print(f"  saved annotated image -> {args.save}")
        if is_video:
            n += 1
        else:
            break
    cap.release()


if __name__ == "__main__":
    main()
