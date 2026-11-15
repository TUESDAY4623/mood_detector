"""pipeline.py — one-command engagement training pipeline.

Stages (each skipped cleanly if its input is missing, so the pipeline
works today with the 3-class dataset and later with the full 8-state one):

    1. prepare   prepare_daisee.py   (only if DAiSEE raw data exists)
    2. combine   merges yawn/eye frames + DAiSEE frames into combined/
    3. train     train.py            (resumable; safe to re-run)
    4. export    export_openvino.py  (NPU IR + classes.json)
    5. verify    NPU smoke inference + dataset consistency check

Stages are the project's own modules, called in-process with their
CLI arguments as data (argparse lists). No shell string is ever built,
so a value coming from a file or flag cannot inject anything.

Run it
-----
    python pipeline.py                       # all applicable stages
    python pipeline.py --stages train,export,verify
    python pipeline.py --dataset datasets/engagement/combined --epochs 20
    python pipeline.py --report               # feasibility report only

If a stage fails, the pipeline stops there: fix it and re-run — the
resumable checkpoint makes the train stage pick up where it left off.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DAISEE_IN = PROJECT_DIR / "datasets" / "engagement" / "daisee_input"
DAISEE_OUT = PROJECT_DIR / "datasets" / "engagement" / "aligned_daisee"
ALIGNED = PROJECT_DIR / "datasets" / "engagement" / "aligned"
REC_OUT = PROJECT_DIR / "datasets" / "engagement" / "aligned_rec"
COMBINED = PROJECT_DIR / "datasets" / "engagement" / "combined"

STATES_8 = ["focused", "confused", "distracted", "neutral",
            "curious", "thinking", "bored", "sleepy"]

# Stage name -> (module to call, label printed on screen).
STAGE_MODULES = {
    "prepare": None,   # handled separately: conditional on raw data presence
    "combine": None,
    "train": "train",
    "export": "export_openvino",
    "verify": None,
}


def _run_stage(module_name: str, argv: list, label: str) -> None:
    """Call a stage module's main() with CLI arguments, in process.

    ``sys.argv`` is patched around the call so the stage's argparse sees
    exactly what its CLI would; a failure stops the pipeline with a
    clear message.
    """
    print(f"\n[pipeline] === {label} ===")
    shown = " ".join(str(a) for a in argv)
    print(f"[pipeline] {module_name}.py {shown}")
    mod = importlib.import_module(module_name)
    old_argv = sys.argv
    sys.argv = [f"{module_name}.py", *map(str, argv)]
    try:
        mod.main()
    except SystemExit as e:
        if e.code in (0, None):
            return
        raise SystemExit(f"[pipeline] {label} failed: {e}\n"
                         f"Fix that stage and re-run; later stages would be stale.")
    finally:
        sys.argv = old_argv


def _have_daisee_input() -> bool:
    csvs = list(DAISEE_IN.glob("*.csv")) if DAISEE_IN.is_dir() else []
    vids = list(DAISEE_IN.rglob("*.mp4")) + list(DAISEE_IN.rglob("*.mov"))
    return bool(csvs) and bool(vids)


def _have_yawn_input() -> bool:
    return (PROJECT_DIR / "datasets" / "engagement" / "raw" / "train").is_dir()


def stage_prepare() -> None:
    if not _have_daisee_input():
        print("[pipeline] prepare skipped: no DAiSEE CSV+videos in "
              f"{DAISEE_IN} (drop them there after the Kaggle download)")
        return
    csv = next(DAISEE_IN.glob("*.csv"))
    _run_stage("prepare_daisee", ["--csv", csv, "--videos-dir", DAISEE_IN],
               "prepare (DAiSEE)")


def _merge_in(frames_root: Path, combined: Path, label: str) -> int:
    """Copy frames from a prepared split-tree into the combined tree.

    Mirrors the DAiSEE merge; returns the number of files merged. A missing
    or empty source is a no-op (returns 0), so the pipeline still works
    before the optional data has been produced.
    """
    if not frames_root.is_dir():
        return 0
    merged = 0
    for split in ["train", "validation", "test"]:
        split_dir = frames_root / split
        if not split_dir.is_dir():
            continue
        for state_dir in split_dir.iterdir():
            if not state_dir.is_dir():
                continue
            dest = combined / split / state_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in state_dir.glob("*.jpg"):
                shutil.copy2(f, dest / f.name)
                merged += 1
    if merged:
        print(f"[pipeline] {label} frames merged ({merged} files)")
    return merged


def stage_combine() -> None:
    """Merge the yawn/eye aligned tree + DAiSEE + self-recorded into combined/."""
    if not _have_yawn_input():
        print("[pipeline] no yawn/eye raw data at datasets/engagement/raw; "
              "nothing to combine yet")
        return
    src = ALIGNED
    if not src.is_dir():
        raise SystemExit(f"[pipeline] {src} missing (arrange the raw data first)")
    print(f"[pipeline] combining {src} -> {COMBINED}")
    shutil.rmtree(COMBINED, ignore_errors=True)
    shutil.copytree(src, COMBINED)
    _merge_in(DAISEE_OUT, COMBINED, "DAiSEE")
    _merge_in(REC_OUT, COMBINED, "self-recorded")
    for split in ["train", "validation", "test"]:
        total = {d.name: len(list(d.glob("*.jpg")))
                 for d in (COMBINED / split).iterdir() if d.is_dir()}
        print(f"  {split}: " + ", ".join(f"{k}={v}" for k, v in sorted(total.items())))


def stage_verify() -> None:
    """NPU smoke inference + dataset/IR consistency check."""
    import openvino as ov
    ir = PROJECT_DIR / "model_ir" / "emotion_npu.xml"
    if not ir.is_file():
        raise SystemExit(f"[pipeline] {ir} missing; run the export stage first")
    classes_doc = json.loads((PROJECT_DIR / "model_ir" / "classes.json").read_text())
    classes = classes_doc["classes"]
    core = ov.Core()
    dev = "NPU" if "NPU" in core.available_devices else "CPU"
    model = core.compile_model(str(ir), dev)
    # A constant input is enough to prove the graph runs and its shape.
    dummy = np.zeros((1, 3, 224, 224), dtype="float32")
    out = model([dummy])[model.output(0)]
    if out.shape[1] != len(classes):
        raise SystemExit(f"[pipeline] IR outputs {out.shape[1]} classes but "
                         f"classes.json lists {len(classes)}")
    print(f"[pipeline] verify: IR runs on {dev}, {len(classes)} classes = {classes}")
    for base in [ALIGNED, COMBINED]:
        if not base.is_dir():
            continue
        for split in ["train", "validation", "test"]:
            for d in (base / split).iterdir():
                if d.is_dir() and not list(d.glob("*.jpg")):
                    print(f"[pipeline] [warn] empty class folder {d}")
    print("[pipeline] verify OK")


def report_8_states() -> None:
    """Honest feasibility table for the 8 states from the data we have."""
    print("\n[pipeline] 8-state feasibility (based on current data)")
    counts: dict[str, int] = {}
    for base in [ALIGNED, DAISEE_OUT, REC_OUT, COMBINED]:
        if not base.is_dir():
            continue
        for split in ["train", "validation", "test"]:
            for d in (base / split).glob("*"):
                if d.is_dir():
                    counts[d.name] = counts.get(d.name, 0) + len(list(d.glob("*.jpg")))
    # Self-recorded data (from record_state.py) closes the public gaps.
    rec_counts: dict[str, int] = {}
    for split in ["train", "validation", "test"]:
        for d in (REC_OUT / split).glob("*"):
            if d.is_dir():
                rec_counts[d.name] = rec_counts.get(d.name, 0) \
                    + len(list(d.glob("*.jpg")))

    def state_status(s: str) -> str:
        if counts.get(s):
            return "OK"
        if rec_counts.get(s):
            return "self-recorded data present"
        return "needs DAiSEE or self-recorded data" \
            if s in ("focused", "distracted") else \
            "public gap - record your own samples (record_state.py)"

    for s in STATES_8:
        n = counts.get(s, 0)
        mark = "OK " if n else " .."
        print(f"  [{mark}] {s:12s} {n:>8d} frames  {state_status(s)}")
    print("\n  Note: the landmark+rules detector (engagement_detector.py) "
          "covers all 8 states TODAY with no training; the CNN learns only "
          "the states that have frames (above).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="prepare,combine,train,export,verify",
                    help="Comma-separated stage list "
                         "(prepare|combine|train|export|verify)")
    ap.add_argument("--report", action="store_true",
                    help="Print the 8-state feasibility report and stop")
    ap.add_argument("--dataset", default="",
                    help="train.py --dataset override "
                         "(default: combined/ if it exists, else aligned/)")
    ap.add_argument("--epochs", default="", help="train.py --epochs override")
    ap.add_argument("--skip-train", action="store_true",
                    help="Never train in this run (use the existing checkpoint)")
    args = ap.parse_args()

    if args.report:
        report_8_states()
        return

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    known = set(STAGE_MODULES)
    for s in stages:
        if s not in known:
            raise SystemExit(f"Unknown stage: {s}")

    if "prepare" in stages:
        stage_prepare()

    if "combine" in stages:
        stage_combine()

    if "train" in stages:
        if args.skip_train:
            print("[pipeline] train skipped (--skip-train)")
        else:
            dataset = Path(args.dataset)
            if not dataset:
                dataset = COMBINED if COMBINED.is_dir() else ALIGNED
            stage_argv = ["--dataset", dataset]
            if args.epochs:
                stage_argv += ["--epochs", args.epochs]
            _run_stage("train", stage_argv, "train")

    if "export" in stages:
        _run_stage("export_openvino", [], "export (NPU IR)")

    if "verify" in stages:
        stage_verify()

    print("\n[pipeline] all requested stages done.")
    print("  Live camera:  python engagement_ui.py")
    print("  Image check:  python predict.py <photo.jpg>")


if __name__ == "__main__":
    main()
