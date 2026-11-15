"""prepare_daisee.py — convert raw DAiSEE data into the engagement training tree.

DAiSEE (classroom engagement dataset, 112 subjects, 10-min videos each)
ships as per-subject MP4 videos plus an annotation CSV with, per timestamp,
an ``engagement_level`` (1 very engaged … 5 not present) and an ``emotion``.

This script turns that into ImageFolder class directories compatible with
``train.py``:

    datasets/engagement/aligned_daisee/{train,validation,test}/<state>/*.jpg

Mapping (documented, honest)
----------------------------
engagement_level -> state
    1 (very engaged)    -> focused
    2 (moderately)      -> neutral
    3 (low)             -> distracted
    4 (not at all)      -> bored
    5 (not present)     -> SKIPPED (subject was away; not a usable sample)

The 3 states DAiSEE cannot provide (curious / thinking / confused) are left
as empty pending folders — no public data exists for them; see
DATASET_INDEX.md for the 8-state feasibility table.

Design notes
------------
* Videos are split at the *subject* level (a subject's frames never land in
  two splits) so validation is honest.
* No ffmpeg needed: frames come out of OpenCV's VideoCapture.
* Local files only: the CSV and videos must already sit on disk (after your
  Kaggle download); nothing is fetched over the network.
* Subject ids are sanitised to digits only, so a hostile CSV value cannot
  steer file writes anywhere unexpected.
* The split RNG is seeded and used for data shuffling only (not anything
  cryptographic), which is exactly what reproducibility wants.

Usage
-----
    python prepare_daisee.py \
        --csv  datasets/engagement/daisee_input/DAiSEE_annotation.csv \
        --videos-dir datasets/engagement/daisee_input \
        --frame-step 10 --max-frames-per-subject 500
    add  --face-filter   to keep only frames where a face is detected
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import cv2

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_DIR / "datasets" / "engagement" / "aligned_daisee"

# engagement_level (DAiSEE scale) -> our state; level 5 means the subject
# was not present and is skipped entirely.
LEVEL_TO_STATE = {
    1: "focused",
    2: "neutral",
    3: "distracted",
    4: "bored",
}
SKIPPED_LEVELS = {5}

# Video-level split fractions (train / validation / test).
SPLITS = [("train", 0.70), ("validation", 0.15), ("test", 0.15)]


def _sanitize_subject(raw: str) -> str | None:
    """Accept only plain digits (DAiSEE subjects are 001..112)."""
    s = str(raw).strip().zfill(3)
    if not re.fullmatch(r"\d+", s):
        return None
    return s


def load_annotations(csv_path: Path) -> dict:
    """Return {subject: [(start_sec, end_sec, state), ...]} from the CSV.

    The official DAiSEE CSV columns are:
        subject, timestamp, emotion, engagement_level
    (``timestamp`` in seconds, annotations are segment-start times). A
    segment runs until the next annotation for that subject. The optional
    activity-extended CSV is also accepted (extra columns are ignored).
    """
    segments: dict[str, list] = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            subj = _sanitize_subject(row.get("subject", ""))
            if subj is None:
                continue
            try:
                t = float(row.get("timestamp"))
                lvl = int(float(row.get("engagement_level")))
            except (TypeError, ValueError):
                continue
            if lvl in SKIPPED_LEVELS:
                continue
            state = LEVEL_TO_STATE.get(lvl)
            if state is None:
                continue
            segments.setdefault(subj, []).append((t, None, state))  # end set later
    # Fill end times: each segment runs until the next one (or +30 s cap).
    for subj, segs in segments.items():
        segs.sort(key=lambda s: s[0])
        for i, (t, _e, state) in enumerate(segs):
            end = segs[i + 1][0] if i + 1 < len(segs) else t + 30.0
            segs[i] = (t, end, state)
    return segments


def state_at(segments: list, t: float) -> str | None:
    """State whose segment contains time ``t`` (seconds)."""
    for start, end, s in segments:
        if start <= t < end:
            return s
    return None


def find_videos(videos_dir: Path) -> dict:
    """Map subject -> video path for every .mp4/.mov/.avi in the dir tree."""
    found: dict[str, Path] = {}
    for p in sorted(videos_dir.rglob("*")):
        if p.suffix.lower() in (".mp4", ".mov", ".avi"):
            subj = _sanitize_subject(p.stem)
            if subj is not None:
                found[subj] = p
    return found


def make_split(subjects: list, seed: int) -> dict:
    """Deterministic subject-level split (train/val/test at 70/15/15)."""
    rng = random.Random(seed)
    pool = sorted(subjects)
    rng.shuffle(pool)
    n = len(pool)
    cuts = {
        "train": int(round(n * SPLITS[0][1])),
        "validation": int(round(n * SPLITS[1][1])),
    }
    assign = {"train": pool[:cuts["train"]],
              "validation": pool[cuts["train"]:cuts["train"] + cuts["validation"]],
              "test": pool[cuts["train"] + cuts["validation"]]}
    out: dict[str, str] = {}
    for subj in pool:
        for split, members in assign.items():
            if subj in members:
                out[subj] = split
    return out


def extract_subject(
    video: Path, segments: list, split: str, out_root: Path,
    frame_step: int, max_frames: int, face_filter: bool,
) -> tuple[int, int]:
    """Sample frames, classify each by the active engagement segment, save.

    Returns (written, face_skipped).
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  [warn] could not open video {video.name}; subject skipped")
        return 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    det = None
    if face_filter:
        from engagement_detector import EngagementDetector
        det = EngagementDetector()

    written, skipped = 0, 0
    idx = 0
    while written < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if (idx - 1) % frame_step != 0:
            continue
        t = (idx - 1) / fps
        state = state_at(segments, t)
        if state is None:
            continue  # no active engagement segment here (e.g. not-present)
        if det is not None:
            sig = det._compute(frame)
            if not sig.face_found:
                skipped += 1
                continue
        out_dir = out_root / split / state
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{video.stem}_{idx:06d}.jpg"
        cv2.imwrite(str(out_dir / name), frame)
        written += 1
    cap.release()
    return written, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="DAiSEE annotation CSV path")
    ap.add_argument("--videos-dir", required=True,
                    help="Directory (or parent) containing subject videos")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--frame-step", type=int, default=10,
                    help="Keep 1 of every N frames (10 ~= 3 fps at 30 fps)")
    ap.add_argument("--max-frames-per-subject", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--face-filter", action="store_true",
                    help="Discard frames with no detected face (slower, cleaner)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    videos_dir = Path(args.videos_dir)
    out_root = Path(args.out)
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}\n"
                         f"(download it from the DAiSEE distribution; see DATASET_INDEX.md)")
    if not videos_dir.is_dir():
        raise SystemExit(f"Videos dir not found: {videos_dir}")

    print(f"[prepare] parsing {csv_path.name} ...")
    segments = load_annotations(csv_path)
    n_ann = sum(len(v) for v in segments.values())
    print(f"[prepare] {len(segments)} subjects, {n_ann} usable segments "
          f"(not-present rows skipped)")

    videos = find_videos(videos_dir)
    missing = set(segments) - set(videos)
    if missing:
        print(f"[prepare] [warn] {len(missing)} subjects have no video file: "
              f"{sorted(missing)[:8]}{'...' if len(missing) > 8 else ''}")
    subjects = sorted(set(segments) & set(videos))
    print(f"[prepare] {len(subjects)} subjects ready "
          f"(frame_step={args.frame_step}, cap={args.max_frames_per_subject}/subject, "
          f"face_filter={args.face_filter})")

    split_of = make_split(subjects, args.seed)
    counts: dict[str, int] = {}
    t0 = time.time()
    for i, subj in enumerate(subjects, 1):
        segs = segments[subj]
        split = split_of.get(subj, "test")
        written, skipped = extract_subject(
            videos[subj], segs, split, out_root,
            args.frame_step, args.max_frames_per_subject, args.face_filter)
        counts[subj] = written
        if i % 10 == 0 or i == len(subjects):
            elapsed = time.time() - t0
            rate = elapsed / max(i, 1)
            print(f"[prepare] {i}/{len(subjects)} subjects | "
                  f"{sum(counts.values())} frames written | "
                  f"{rate:.1f}s/subject | ~{rate * (len(subjects) - i):.0f}s remaining")

    # Per-split, per-state counts for the log.
    summary = {}
    for split in ["train", "validation", "test"]:
        sdir = out_root / split
        summary[split] = {
            d.name: len(list(d.glob("*.jpg")))
            for d in sdir.iterdir() if d.is_dir()
        } if sdir.is_dir() else {}
    log = {
        "csv": str(csv_path),
        "frame_step": args.frame_step,
        "max_frames_per_subject": args.max_frames_per_subject,
        "face_filter": args.face_filter,
        "seed": args.seed,
        "subjects": len(subjects),
        "missing_videos": sorted(missing),
        "counts_by_split": summary,
    }
    (out_root / "mapping.json").write_text(json.dumps(log, indent=2))
    print(f"[prepare] done in {time.time() - t0:.0f}s -> {out_root}")
    for split, by_state in summary.items():
        total = sum(by_state.values())
        print(f"  {split}: {total} frames | " +
              ", ".join(f"{k}={v}" for k, v in sorted(by_state.items())))


if __name__ == "__main__":
    main()
