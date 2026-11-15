"""fused_detector.py — one-pass, all-8-state emotion/mood detector.

Fuses the two engines that already exist in this project:

* the **landmark + rules** engine (``engagement_detector.EngagementDetector``)
  which can score all 8 states from face geometry over a short window; and
* the **NPU CNN** (OpenVINO IR from ``export_openvino.py``) which learns
  whatever states it was trained on (currently bored / neutral / sleepy).

One call to ``FusedDetector.update(frame)`` returns a confidence for
*every* one of the 8 states, so the camera app can draw 8 bars and a
banner in a single pass. The NPU path is used when the IR model is
available; when it is not (e.g. on a machine without OpenVINO), the
detector still works on rules alone, so nothing is left behind.

States (the full 8):
    focused, confused, distracted, neutral, curious, thinking, bored, sleepy

Which engine owns which state
------------------------------
States the CNN was trained on are *fused*: 0.5 * rules + 0.5 * cnn.
States the CNN has no data for are *rule-only*. This keeps the detector
fully live for all 8 today, and automatically better for the CNN's own
classes as more data is added and it is retrained/re-exported.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from engagement_detector import EngagementDetector, STATES

# NPU CNN input size and per-state fusion weight.
_CNN_SIZE = 224
_CNN_WEIGHT = 0.5   # remainder goes to the rules engine

# Frozen-resource base: when run from a PyInstaller onefile, model +
# landmarker are unpacked next to the exe (see the build spec), so look
# there first, then fall back to the source tree.
def _base_dirs() -> list:
    dirs = []
    meipass = getattr(__import__("sys"), "_MEIPASS", None)
    if meipass:
        dirs.append(os.path.abspath(meipass))
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in dirs:
        dirs.append(here)
    return dirs


def _find_rel(rel: str) -> Optional[str]:
    """Locate a resource shipped next to the app or in the source tree."""
    for d in _base_dirs():
        p = os.path.join(d, rel)
        if os.path.exists(p):
            return p
    return None


@dataclass
class MoodResult:
    state: str
    confidence: float
    scores: dict                 # all 8 states -> fused confidence 0..1
    cnn_scores: dict             # all 8 states -> raw CNN prob (0 if not trained)
    rules_scores: dict           # all 8 states -> raw rules score
    face_found: bool
    device: str                  # "NPU" | "CPU" | "none"
    latency_ms: float
    evidence: dict = field(default_factory=dict)


class NpuCnn:
    """Wraps the OpenVINO IR so it yields per-state probabilities.

    States the model was not trained on simply never appear in its
    output, so their probability is 0.0 by construction.
    """

    def __init__(self, device: str = "NPU"):
        import openvino as ov

        ir = _find_rel(os.path.join("model_ir", "emotion_npu.xml"))
        cj = _find_rel(os.path.join("model_ir", "classes.json"))
        if not ir or not cj:
            raise FileNotFoundError("model_ir/emotion_npu.xml + classes.json not found")
        import json

        classes_doc = json.load(open(cj))
        self.classes: list[str] = list(classes_doc["classes"])
        core = ov.Core()
        self.device = device.upper()
        try:
            self.model = core.compile_model(ir, self.device)
        except Exception:
            print(f"[fused] {self.device} compile failed; falling back to CPU.")
            self.device = "CPU"
            self.model = core.compile_model(ir, "CPU")

    def probs(self, bgr: np.ndarray) -> dict:
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (_CNN_SIZE, _CNN_SIZE), interpolation=cv2.INTER_AREA)
        t = rgb.astype("float32") / 255.0
        t = np.transpose(t, (2, 0, 1))[None, ...]
        out = self.model([t])[self.model.output(0)][0].astype("float32")
        out = out - out.max()
        p = np.exp(out)
        p = p / p.sum()
        return {c: float(p[i]) for i, c in enumerate(self.classes)}


class FusedDetector:
    """All-8-state detector = landmark rules + NPU CNN in one pass."""

    def __init__(self, use_cnn: bool = True, device: str = "NPU",
                 window_seconds: float = 10.0):
        self.rules = EngagementDetector(window_seconds=window_seconds)
        self.device = "none"
        self.cnn: Optional[NpuCnn] = None
        if use_cnn:
            try:
                self.cnn = NpuCnn(device=device)
                self.device = self.cnn.device
            except Exception as e:  # noqa: BLE001
                print(f"[fused] CNN unavailable ({e}); rules only.")
        self._states = STATES

    def update(self, bgr: np.ndarray) -> MoodResult:
        t0 = time.perf_counter()
        res = self.rules.update(bgr)
        rules_scores = res.evidence.get("all_scores", {})
        # Normalise so every one of the 8 states is present.
        rules_scores = {s: float(rules_scores.get(s, 0.0)) for s in self._states}
        face_found = bool(res.evidence.get("face_in_window",
                                           res.signals.face_found))

        cnn_scores = {s: 0.0 for s in self._states}
        if self.cnn is not None and face_found:
            raw = self.cnn.probs(bgr)
            # Only trust the CNN when a face is actually visible; otherwise
            # the network is looking at background.
            cnn_scores = {s: float(raw.get(s, 0.0)) for s in self._states}

        # Fuse: CNN carries its trained states, rules carry the rest.
        fused: dict[str, float] = {}
        for s in self._states:
            if s in (self.cnn.classes if self.cnn else []) and face_found:
                fused[s] = min(1.0, _CNN_WEIGHT * cnn_scores[s]
                               + (1 - _CNN_WEIGHT) * rules_scores[s])
            else:
                fused[s] = rules_scores[s]

        # Same hard overrides the rules engine uses, so a sustained closed
        # eye / yawn / look-away always wins regardless of the CNN.
        ev = res.evidence
        if ev.get("closed_frac", 0) > 0.6:
            fused["sleepy"] = max(fused["sleepy"], 0.9)
        if ev.get("yawn"):
            fused["bored"] = max(fused["bored"], 0.9)
        if ev.get("away_frac", 0) > 0.6:
            fused["distracted"] = max(fused["distracted"], 0.85)

        if not face_found:
            state, conf = "neutral", 0.0
        else:
            state = max(fused, key=lambda k: fused[k])
            top = sorted(fused.values(), reverse=True)
            conf = max(0.0, min(1.0, (top[0] - top[1]) + 0.2))

        return MoodResult(
            state=state,
            confidence=conf,
            scores=fused,
            cnn_scores=cnn_scores,
            rules_scores=rules_scores,
            face_found=face_found,
            device=self.device,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            evidence=ev,
        )


def all_states() -> list:
    return list(STATES)


if __name__ == "__main__":
    import cv2
    import glob
    det = FusedDetector()
    print(f"[fused] states={len(det._states)} device={det.device}")
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "datasets", "engagement", "aligned", "test")
    for p in glob.glob(os.path.join(sample_dir, "*", "*.jpg"))[:1]:
        img = cv2.imread(p)
        r = det.update(img)
        print("sample:", p, "->", r.state, round(r.confidence, 2),
              f"(rules {round(max(r.rules_scores.values()),2)} / "
              f"cnn {round(max(r.cnn_scores.values()),2)})")
