"""engagement_detector.py — landmark-based recognition of 8 engagement states.

The 8 states (Option C hybrid, landmark+rules stage) are:
    focused, confused, distracted, neutral, curious, thinking, bored, sleepy

Why landmarks + rules (and not just a CNN on frames):
    "focused", "thinking" and "curious" look nearly identical in a single
    photo; "bored" (yawning frequently) is a *pattern over time*; "distracted"
    depends on head/gaze direction. So this module extracts 478 face
    landmarks per frame with MediaPipe FaceLandmarker and turns them into
    *primitive signals* (eye openness, mouth openness/yawn, head pose,
    gaze), then applies rules over a rolling window to pick a state.

Signals
-------
    EAR  (Eye Aspect Ratio)     mean of left/right eyes; tracked relative to
                                the user's own open-eye baseline.
    MAR  (Mouth Aspect Ratio)   mouth open-ness; a yawn = MAR > ~0.62 held
                                for ~2 s.
    head pose (yaw / pitch)     from MediaPipe's 4x4 face transform.
    iris position               which way the eyes point (iris landmarks).

Grounding honesty (read this before trusting outputs):
    - sleepy / bored / distracted / neutral / focused are well-supported by
      the geometry above.
    - thinking (eyes open, head tilted, gaze up, calm) and
      curious (gaze up, small head motion) are *approximations* built from
      the same primitives; they will overlap with focused/neutral until the
      hybrid CNN stage refines them with real labeled data.
    - confused (head tilt + open eyes) is the weakest; treat its
      confidence as indicative, not definitive.

Per-frame API:
    det = EngagementDetector()
    result = det.update(bgr_frame)
    result.state       # one of the 8 states
    result.confidence  # 0..1 how strongly the primitives support it
    result.evidence    # dict of the primitives that fired this frame

For the later hybrid CNN stage, ``result.feature_vector`` is the numeric
feature row you would feed a small model.
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

try:  # mediapipe 1.2.x layout: attribute access works, plain import may not
    import mediapipe.tasks as _mp_tasks
    _VISION = _mp_tasks.vision
    _HAS_MEDIAPIPE = True
except Exception:  # pragma: no cover - mediapipe missing -> rules degrade
    _HAS_MEDIAPIPE = False
    _VISION = None

# Default location of the face-landmarker model file (assets/ next to this file).
_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
DEFAULT_LANDMARKER = os.path.join(_ASSETS, "face_landmarker.task")

STATES = ["focused", "confused", "distracted", "neutral", "curious",
          "thinking", "bored", "sleepy"]

# MediaPipe face-mesh indices (0-based, 478-point layout) used by the EAR
# drowsify formulation: 6 boundary points per eye.
_EYE_L = (33, 133, 159, 153, 145, 144)
_EYE_R = (33, 7, 163, 144, 145, 159)
# Mouth: upper lip centre, lower lip centre, left/right corners.
_MOUTH_UP, _MOUTH_LOW = 13, 14
_MOUTH_L, _MOUTH_R = 61, 291
# Iris centre (478-point builds append iris points after the 468 base).
_IRIS_L, _IRIS_R = 473, 474


def _d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _ear(pts: np.ndarray, idx: tuple) -> float:
    p = (pts[idx[i]] for i in range(6))
    p1, p2, p3, p4, p5, p6 = list(p)
    vert = _d(p2, p6) + _d(p3, p5)
    horiz = 3.0 * _d(p1, p4)
    return vert / horiz if horiz > 1e-6 else 0.0


def _pose_from_transform(m: np.ndarray) -> tuple:
    """Extract (yaw, pitch, roll) in radians from a 4x4 face transform."""
    c = m[:3]
    m11, m12 = c[0][0], c[1][0]
    m21, m22, m23 = c[1][0], c[2][0], c[2][1]
    m32, m33 = c[2][1], c[2][2]
    pitch = math.atan2(-m23, math.hypot(m21, m22))
    yaw = math.atan2(m12, m11)
    roll = math.atan2(m32, m33)
    return float(yaw), float(pitch), float(roll)


@dataclass
class FrameSignals:
    """Primitive signals computed for one frame."""
    face_found: bool = False
    ear: float = 0.0
    ear_rel: float = 1.0        # EAR / personal open-eye baseline
    mar: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    eye_closed: bool = False
    yawn: bool = False
    head_away: bool = False      # |yaw| large
    looking_down: bool = False   # pitch large
    looking_up: bool = False
    iris_lateral: float = 0.0   # -1 left, 0 centred, +1 right
    iris_up: bool = False
    features: list = field(default_factory=list)


@dataclass
class DetectionResult:
    state: str
    confidence: float
    evidence: dict
    signals: FrameSignals
    feature_vector: list


class EngagementDetector:
    """Wraps MediaPipe FaceLandmarker + rolling-window rules.

    ``model_path`` defaults to the project's ``assets/face_landmarker.task``.
    The open-eye EAR baseline is learned automatically from the first frames
    where the eye is open (EAR >= 0.21).
    """

    def __init__(self, model_path: str = DEFAULT_LANDMARKER,
                 window_seconds: float = 10.0,
                 yaw_away_deg: float = 25.0,
                 pitch_down_deg: float = 15.0,
                 pitch_up_deg: float = 8.0,
                 roll_tilt_deg: float = 12.0,
                 yawn_mar: float = 0.62,
                 yawn_seconds: float = 2.0):
        if not _HAS_MEDIAPIPE:
            raise RuntimeError(
                "mediapipe is not importable in this environment. "
                "Run:  .venv/Scripts/pip install mediapipe")
        opts = _VISION.FaceLandmarkerOptions(
            base_options=_mp_tasks.BaseOptions(model_asset_path=model_path),
            num_faces=1,
            min_face_detection_confidence=0.4,
            output_facial_transformation_matrixes=True,
        )
        self._lm = _VISION.FaceLandmarker.create_from_options(opts)
        self._win = window_seconds
        self._yaw_away = math.radians(yaw_away_deg)
        self._p_down = math.radians(pitch_down_deg)
        self._p_up = math.radians(-pitch_up_deg)
        self._roll_tilt = math.radians(roll_tilt_deg)
        self._yawn_mar = yawn_mar
        self._yawn_seconds = yawn_seconds
        self._baseline_ear: Optional[float] = None
        self._mar_open_samples: list = []
        self._hist: deque = deque(maxlen=max(1, int(window_seconds * 10)))
        self._mar_hold = 0.0
        self._last_t = time.perf_counter()
        self._last_pts: Optional[np.ndarray] = None

    # -- calibration -------------------------------------------------------
    def calibrate(self) -> float:
        """Return the current open-eye EAR baseline (0.30 until measured).

        The baseline fills in automatically from the first frames where the
        user's eye is clearly open; call this after a few seconds of
        ``update()`` calls if you want an explicit value.
        """
        return self._baseline_ear or 0.30

    def _update_baseline(self, ear: float) -> None:
        if ear >= 0.21:
            if self._baseline_ear is None:
                self._baseline_ear = ear
            else:
                self._baseline_ear = 0.98 * self._baseline_ear + 0.02 * ear

    # -- overlay helpers ----------------------------------------------------
    def last_landmarks(self) -> Optional[np.ndarray]:
        """Normalised (x, y) landmark points from the most recent update.

        ``None`` when no face was found in the last frame. Shape
        ``(N, 2)`` with values in ``[0, 1]`` — multiply by frame width/
        height to draw on a BGR frame.
        """
        return self._last_pts

    # -- per-frame primitives --------------------------------------------
    def _compute(self, bgr: np.ndarray) -> FrameSignals:
        s = FrameSignals()
        if not _HAS_MEDIAPIPE:
            return s
        import mediapipe as mp
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self._lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            return s
        pts = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]])
        self._last_pts = pts[:, :2]  # normalised x/y, for overlay drawing
        s.face_found = True
        # eyes
        s.ear = 0.5 * (_ear(pts, _EYE_L) + _ear(pts, _EYE_R))
        self._update_baseline(s.ear)
        self._baseline_ear = self._baseline_ear or 0.30
        s.ear_rel = s.ear / self._baseline_ear
        s.eye_closed = s.ear_rel < 0.55
        # mouth
        s.mar = _d(pts[_MOUTH_UP], pts[_MOUTH_LOW]) / max(
            _d(pts[_MOUTH_L], pts[_MOUTH_R]), 1e-6)
        s.yawn = s.mar > self._yawn_mar
        # head pose (enable the transformation matrices in the options)
        if res.facial_transformation_matrixes:
            m = np.array(res.facial_transformation_matrixes[0])
            s.yaw, s.pitch, s.roll = _pose_from_transform(m)
        s.head_away = abs(s.yaw) > self._yaw_away
        s.looking_down = s.pitch > self._p_down
        s.looking_up = s.pitch < self._p_up
        # iris (only in 478-point builds)
        if len(pts) > 477:
            cl = pts[_IRIS_L] - pts[133]
            cr = pts[_IRIS_R] - pts[33]
            lat = cl[0] + cr[0]
            s.iris_lateral = float(np.clip(lat / 2.0, -1, 1))
            s.iris_up = bool(cl[1] < 0 or cr[1] < 0)
        s.features = [s.ear_rel, s.mar, s.yaw, s.pitch, s.roll,
                      1.0 if s.eye_closed else 0.0,
                      1.0 if s.yawn else 0.0,
                      1.0 if s.head_away else 0.0,
                      1.0 if s.looking_down else 0.0,
                      s.iris_lateral]
        return s

    def update(self, bgr: np.ndarray) -> DetectionResult:
        """Feed one BGR frame; return the current 8-state decision.

        Timing is measured from wall-clock so callers need not thread a
        timer through; a very short pause (e.g. dropped frame) will not
        skew the window much.
        """
        now = time.perf_counter()
        dt = max(0.001, now - self._last_t)
        self._last_t = now

        s = self._compute(bgr)
        # Track yawning episode duration.
        if s.yawn:
            self._mar_hold += dt
            s.yawn = self._mar_hold >= self._yawn_seconds
        else:
            self._mar_hold = 0.0
        # Remember open-mouth mean for relative reference.
        if s.face_found and s.mar < 0.3:
            self._mar_open_samples.append(s.mar)
            if len(self._mar_open_samples) > 200:
                self._mar_open_samples.pop(0)

        self._hist.append((now, s))

        # -- windowed aggregation (last ``_win`` seconds) -----------------
        recent = [e for e in self._hist if now - e[0] <= self._win]
        found = [e for e in recent if e[1].face_found]
        if not found:
            # No face in the whole window -> we genuinely cannot decide.
            any_sig = recent[0][1] if recent else FrameSignals()
            return DetectionResult("neutral", 0.0,
                                   {"reason": "no face in window"},
                                   any_sig, any_sig.features or [])
        sigs = [e[1] for e in found]
        closed_frac = sum(1 for x in sigs if x.eye_closed) / len(sigs)
        away_frac = sum(1 for x in sigs if x.head_away) / len(sigs)
        down_frac = sum(1 for x in sigs if x.looking_down) / len(sigs)
        up_frac = sum(1 for x in sigs if x.looking_up) / len(sigs)
        tilt_frac = sum(1 for x in sigs if abs(x.roll) > self._roll_tilt) / len(sigs)
        yawn_now = bool(s.yawn)
        calm = float(np.mean([1.0 - min(abs(x.yaw), 1.0) for x in sigs]))

        # Candidate scores per state (heuristic; confidence derived below).
        scores: dict[str, float] = {
            "sleepy":     closed_frac * 0.7 + (0.3 if closed_frac > 0.5 else 0.0),
            "bored":      (0.8 if (s.yawn or yawn_now) else 0.0) + 0.2 * closed_frac,
            "distracted": away_frac,
            "focused":    down_frac * 0.6 + calm * 0.4 - 0.3 * closed_frac,
            "thinking":   up_frac * 0.5 + calm * 0.3 - 0.4 * away_frac,
            "curious":    up_frac * 0.6 - 0.4 * away_frac - 0.3 * closed_frac,
            "confused":   tilt_frac * 0.6 - 0.3 * away_frac,
            "neutral":    calm * 0.5,
        }
        # Hard overrides: closed eyes / confirmed yawn / sustained turn-away
        # dominate everything else.
        if closed_frac > 0.6:
            scores["sleepy"] = max(scores["sleepy"], 0.9)
        if s.yawn or yawn_now:
            scores["bored"] = max(scores["bored"], 0.9)
        if away_frac > 0.6:
            scores["distracted"] = max(scores["distracted"], 0.85)

        state = max(scores, key=lambda k: scores[k])
        vals = sorted(scores.values(), reverse=True)
        conf = max(0.0, min(1.0, (vals[0] - vals[1]) + 0.3)) if len(vals) > 1 else 0.3
        ev = {
            "closed_frac": round(closed_frac, 2),
            "away_frac": round(away_frac, 2),
            "down_frac": round(down_frac, 2),
            "up_frac": round(up_frac, 2),
            "tilt_frac": round(tilt_frac, 2),
            "yawn": bool(s.yawn or yawn_now),
            "face_in_window": True,
            "scores": {k: round(v, 2) for k, v in scores.items() if v > 0.01},
            # Complete, un-truncated table for the fused CNN layer: every one
            # of the 8 states, clamped to [0,1] and rounded to 3 places.
            "all_scores": {k: round(max(0.0, min(1.0, v)), 3)
                           for k, v in scores.items()},
        }
        return DetectionResult(state, conf, ev, s, s.features)


if __name__ == "__main__":
    import glob
    from pathlib import Path
    det = EngagementDetector()
    sample_dir = Path(__file__).parent / "dataset" / "dataset1_clean" / "train"
    sample = next(iter(sorted(glob.glob(str(sample_dir / "neutral" / "*.jpg")))), None)
    if sample:
        img = cv2.imread(sample)
        r = det.update(img)
        print("sample:", sample)
        print("state:", r.state, "| conf:", round(r.confidence, 2))
        print("evidence:", r.evidence)
