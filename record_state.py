"""record_state.py — record self-labelled engagement-state clips from your
own webcam and turn them into CNN training frames.

The 8 engagement states are the full set used by the rest of the project:
    focused, confused, distracted, neutral, curious, thinking, bored, sleepy

Three of them (curious, thinking, confused) have no public dataset, so the
only way to CNN-train them is to record yourself in that state and label it.
This tool does that in two subcommands:

    1. record  — sit in front of the camera, perform/hold a state for a few
                seconds, and save the session (frames + face presence +
                timing) under recordings/<state>/<session_id>/.
    2. ingest  — take every recorded session and split it into the
                train/validation/test tree the pipeline expects:
                datasets/engagement/aligned_rec/{train,validation,test}/<state>/

The split is *session-level* (70/15/15): a session's frames always go to a
single split, so the same person's face cannot leak between train and test.
Re-running ingest is safe (deterministic seed, idempotent writes).

Quick start
-----------
    python record_state.py record --state thinking --seconds 20
    python record_state.py record --state curious  --seconds 15
    python record_state.py record --state confused --seconds 15
    python record_state.py ingest
    python pipeline.py          # train + export the full 8-state CNN

Recorded sessions are kept (they are the raw labels); ingest only *copies*
selected frames into the aligned_rec tree. ``--face-filter`` keeps only
frames where the face landmarker actually found a face (slower, cleaner).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
REC_ROOT = PROJECT_DIR / "datasets" / "engagement" / "recordings"
REC_OUT = PROJECT_DIR / "datasets" / "engagement" / "aligned_rec"

STATES_8 = ["focused", "confused", "distracted", "neutral",
            "curious", "thinking", "bored", "sleepy"]

# Session-level split fractions (train / validation / test), mirroring the
# DAiSEE prep so self-recorded data is honest the same way it is there.
# NOTE: the RNG here is deliberately seeded *weak* random — it only makes
# the train/validation/test split reproducible, it has no security role.
SPLITS = [("train", 0.70), ("validation", 0.15), ("test", 0.15)]


# --------------------------------------------------------------------------
# recording (live camera)
# --------------------------------------------------------------------------
def _open_camera() -> "cv2.VideoCapture":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the laptop camera. Free up any other "
                         "camera app and try again.")
    return cap


def _face_ok(frame: np.ndarray, checker) -> bool:
    """True when the face landmarker finds a face (or when filtering is off)."""
    if checker is None:
        return True
    try:
        sig = checker._compute(frame)
        return bool(sig.face_found)
    except Exception:
        return True  # a detector hiccup should not silently drop a frame


def record_session(state: str, seconds: int, target_fps: int,
                   face_filter: bool, out_root: Path) -> Path:
    """Record one live session into out_root/<state>/<session_id>/."""
    from engagement_detector import EngagementDetector

    cap = _open_camera()
    checker = EngagementDetector() if face_filter else None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sess_dir = out_root / state / f"{ts}_s{seconds}"
    sess_dir.mkdir(parents=True, exist_ok=True)

    period = 1.0 / target_fps
    total = seconds * target_fps
    n_faces = 0
    t_start = time.time()
    print(f"[record] state={state} ~{seconds}s @ {target_fps} fps "
          f"-> {sess_dir.relative_to(PROJECT_DIR)}")
    print("[record] hold the state for the whole session; "
          "Q stops early, ESC quits")

    idx = 0
    while idx < total:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        if _face_ok(frame, checker):
            n_faces += 1
        path = sess_dir / f"f_{idx:05d}.jpg"
        if not cv2.imwrite(str(path), frame):
            raise SystemExit(f"[record] could not write {path}")
        idx += 1
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print(f"[record] stopped early at frame {idx}/{total}")
            break
        left = period - (time.time() - t0)
        if left > 0:
            time.sleep(left)

    cap.release()
    meta = {
        "state": state,
        "requested_seconds": seconds,
        "captured_frames": idx,
        "target_fps": target_fps,
        "face_filtered": face_filter,
        "face_found_frames": n_faces,
        "started": ts,
        "wall_seconds": round(time.time() - t_start, 2),
    }
    (sess_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    ratio = (n_faces / idx) if idx else 0.0
    print(f"[record] saved {idx} frames; face present in {ratio:.0%} of them")
    if idx and ratio < 0.5:
        print("[record] warning: face was rarely detected — "
              "consider re-recording closer to the camera")
    return sess_dir


# --------------------------------------------------------------------------
# ingesting recorded sessions into the training tree
# --------------------------------------------------------------------------
def list_sessions(out_root: Path) -> list:
    """Every recorded session dir (…/<state>/<session_id>/)."""
    out: list[Path] = []
    if not out_root.is_dir():
        return out
    for state_dir in sorted(out_root.iterdir()):
        if not state_dir.is_dir() or state_dir.name not in STATES_8:
            continue
        for sess in sorted(state_dir.iterdir()):
            if sess.is_dir() and (sess / "meta.json").is_file():
                out.append(sess)
    return out


def session_split_map(sessions: list, seed: int) -> dict:
    """Deterministic session->split assignment (70/15/15), seeded."""
    rng = random.Random(seed)
    pool = [s.name for s in sessions]
    rng.shuffle(pool)
    n = len(pool)
    cuts = {"train": int(round(n * SPLITS[0][1])),
            "validation": int(round(n * SPLITS[1][1]))}
    assign = {"train": pool[:cuts["train"]],
              "validation": pool[cuts["train"]:cuts["train"] + cuts["validation"]],
              "test": pool[cuts["train"] + cuts["validation"]]}
    out: dict[str, str] = {}
    for s in pool:
        for split, members in assign.items():
            if s in members:
                out[s] = split
    return out


def ingest(out_root: Path, dest_root: Path, seed: int,
           frame_step: int, face_filter: bool) -> dict:
    sessions = list_sessions(out_root)
    if not sessions:
        raise SystemExit(f"[ingest] no recorded sessions under {out_root}\n"
                         f"Record some first:  python record_state.py record --state curious --seconds 20")
    split_of = session_split_map(sessions, seed)

    checker = None
    if face_filter:
        from engagement_detector import EngagementDetector
        checker = EngagementDetector()

    counts: dict[str, dict] = {"train": {}, "validation": {}, "test": {}}
    face_skipped = 0
    for sess in sessions:
        meta = json.loads((sess / "meta.json").read_text())
        state = meta["state"]
        split = split_of[sess.name]
        frames = sorted(sess.glob("f_*.jpg"))
        written = 0
        for f in frames[::frame_step]:
            if checker is not None and not _face_ok_cv(f, checker):
                face_skipped += 1
                continue
            d = dest_root / split / state
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, d / f.name)
            written += 1
        counts[split][state] = counts[split].get(state, 0) + written

    manifest = {
        "recordings_root": str(out_root),
        "sessions": len(sessions),
        "seed": seed,
        "frame_step": frame_step,
        "face_filter": face_filter,
        "face_skipped": face_skipped,
        "split_of_session": split_of,
        "counts_by_split": counts,
    }
    dest_root.mkdir(parents=True, exist_ok=True)
    (dest_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _face_ok_cv(frame_path: Path, checker) -> bool:
    img = cv2.imread(str(frame_path))
    if img is None:
        return False
    return _face_ok(img, checker)


def print_ingest_summary(manifest: dict) -> None:
    print(f"[ingest] {manifest['sessions']} sessions -> "
          f"{manifest['recordings_root']} (seed={manifest['seed']}, "
          f"step={manifest['frame_step']})")
    for split in ("train", "validation", "test"):
        by_state = manifest["counts_by_split"].get(split, {})
        total = sum(by_state.values())
        print(f"  {split}: {total} frames | "
              + (", ".join(f"{k}={v}" for k, v in sorted(by_state.items()))
                 or "(none)"))
    if manifest["face_skipped"]:
        print(f"  [warn] {manifest['face_skipped']} frames had no detectable "
              "face and were skipped")
    empty = [s for s in ("validation", "test")
             if not any(manifest["counts_by_split"].get(s, {}).values())]
    if empty:
        print(f"  [warn] empty splits: {', '.join(empty)} — record more "
              "sessions so validation/test are not empty")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rc = sub.add_parser("record",
                        help="record one live camera session for a state")
    rc.add_argument("--state", required=True, choices=STATES_8)
    rc.add_argument("--seconds", type=int, default=20)
    rc.add_argument("--fps", type=int, default=15,
                    help="frames per second to capture (default 15; "
                         "higher = more frames but bigger files)")
    rc.add_argument("--face-filter", action="store_true",
                    help="check face presence per frame while recording")
    rc.add_argument("--out", default=str(REC_ROOT), help="recordings root")

    ig = sub.add_parser("ingest",
                        help="split recorded sessions into the aligned_rec tree")
    ig.add_argument("--seed", type=int, default=42)
    ig.add_argument("--frame-step", type=int, default=3,
                    help="keep 1 of every N frames (default 3)")
    ig.add_argument("--face-filter", action="store_true",
                    help="drop frames with no detected face (slower)")
    ig.add_argument("--out", default=str(REC_OUT),
                    help="aligned_rec root")
    ig.add_argument("--recordings", default=str(REC_ROOT),
                    help="recordings root to read from")

    args = ap.parse_args()
    if args.cmd == "record":
        record_session(args.state, args.seconds, args.fps,
                       args.face_filter, Path(args.out))
    else:
        m = ingest(Path(args.recordings), Path(args.out),
                   args.seed, args.frame_step, args.face_filter)
        print_ingest_summary(m)


if __name__ == "__main__":
    main()
