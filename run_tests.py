"""run_tests.py — end-to-end test for the 8-state mood system (plan step F4).

What it runs
------------
1. python script  : `.venv/Scripts/python mood_ui.py --headless`
2. packaged exe   : `dist/MoodDetector.exe --headless`
against three sources:
   * live camera (when available)
   * the test image folders of the yawn dataset (bored / sleepy / neutral)

Writes JSONL evidence logs under ``_evidence/`` and a summary table to
stdout. Face-present images prove the detector sees a real face; the
logs record state, all-8 scores (fused + CNN + rules), fps, device and
latency for every logged frame.

Usage
-----
    python run_tests.py            # full run (camera + datasets)
    python run_tests.py --no-cam   # datasets only
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV_PY = HERE / ".venv" / "Scripts" / "python.exe"
EXE = HERE / "dist" / "MoodDetector.exe"
EVIDENCE = HERE / "_evidence"

TEST_DIRS = {
    "bored": "datasets/engagement/aligned/test/bored",
    "sleepy": "datasets/engagement/aligned/test/sleepy",
    "neutral": "datasets/engagement/aligned/test/neutral",
}


def _cmd(exe: bool, source: str, frames: int, tag: str) -> list:
    runner = [str(EXE)] if exe else [str(VENV_PY), str(HERE / "mood_ui.py")]
    return runner + ["--headless", "--source", source,
                    "--max-frames", str(frames),
                    "--log-file", str(EVIDENCE / f"{tag}.jsonl")]


def _summarize(log: Path) -> dict:
    recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    if not recs:
        return {"n": 0}
    fps = [r["fps"] for r in recs[5:] if r.get("fps", 0) > 0]
    lat = [r["latency_ms"] for r in recs]
    top: dict[str, int] = {}
    for r in recs:
        top[r["state"]] = top.get(r["state"], 0) + 1
    return {
        "n": len(recs),
        "face_found_frames": sum(1 for r in recs if r["face_found"]),
        "fps_median": round(statistics.median(fps), 1) if fps else 0,
        "latency_ms_median": round(statistics.median(lat), 1),
        "device": recs[-1]["device"],
        "states_seen": dict(sorted(top.items(), key=lambda kv: -kv[1])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-cam", action="store_true",
                    help="Skip the live-camera case (use datasets only)")
    ap.add_argument("--cam-frames", type=int, default=240,
                    help="Live-camera frames to capture (~5.3s at 45 fps target)")
    ap.add_argument("--dir-frames", type=int, default=45,
                    help="Frames to replay from each dataset test folder")
    args = ap.parse_args()

    EVIDENCE.mkdir(exist_ok=True)
    results: list[tuple[str, str, dict]] = []

    def run(tag: str, source: str, frames: int) -> None:
        if VENV_PY.exists():
            cmd = _cmd(False, source, frames, f"py_{tag}")
            print(f"== python  [{tag}] {' '.join(os.path.basename(c) for c in cmd[1:])}")
            r = subprocess.run(cmd, cwd=str(HERE),
                               capture_output=True, text=True, timeout=600)
            summary = _summarize(EVIDENCE / f"py_{tag}.jsonl") if r.returncode == 0 \
                else {"error": (r.stderr or r.stdout).strip()[-400:]}
            print(f"   exit={r.returncode} -> {summary}")
            results.append(("python", tag, summary))
        else:
            print(f"-- python runner {VENV_PY.name} missing; skipped")
        if EXE.exists():
            cmd = _cmd(True, source, frames, f"exe_{tag}")
            print(f"== exe     [{tag}] {EXE.name} --headless ...")
            r = subprocess.run(cmd, cwd=str(HERE),
                               capture_output=True, text=True, timeout=600)
            summary = _summarize(EVIDENCE / f"exe_{tag}.jsonl") if r.returncode == 0 \
                else {"error": (r.stderr or r.stdout).strip()[-400:]}
            print(f"   exit={r.returncode} -> {summary}")
            results.append(("exe", tag, summary))
        else:
            print(f"-- {EXE.name} missing; run the PyInstaller build first")

    if not args.no_cam:
        run("cam", "", args.cam_frames)
    for name, rel in TEST_DIRS.items():
        p = HERE / rel
        if p.exists():
            run(name, rel, args.dir_frames)
        else:
            print(f"-- dataset folder {rel} not present; skipped")

    print("\n=== SUMMARY ===")
    print(f"{'runner':<7} {'case':<8} {'frames':>6} {'face':>5} {'fps':>6} "
          f"{'lat(ms)':>8} {'dev':<5} states")
    for runner_name, tag, s in results:
        if "n" not in s:
            print(f"{runner_name:<7} {tag:<8} ERROR: {str(s.get('error', '?'))[:80]}")
            continue
        print(f"{runner_name:<7} {tag:<8} {s['n']:>6} "
              f"{s.get('face_found_frames', 0):>5} "
              f"{s.get('fps_median', 0):>6} "
              f"{s.get('latency_ms_median', 0):>8} "
              f"{s.get('device', '?'):<5} "
              f"{s.get('states_seen', {})}")

    ok = all("n" in s for _, _, s in results)
    print(f"\nEVIDENCE: {EVIDENCE.name}/")
    sys.exit(0 if ok and results else 1)


if __name__ == "__main__":
    main()
