"""mood_dashboard.py — cute, smooth 8-mood camera dashboard (tkinter + NPU).

Fast path: a worker thread grabs camera frames at full camera rate and runs
the fused detector (landmark rules + NPU CNN) on every frame; the UI thread
only renders. Nothing blocks the camera, so the mood updates in the least
time possible.

The window is a pastel "desk" with active skeuomorphic widgets:
  * a glossy video card (camera feed + blinking LIVE led),
  * a spring-loaded confidence dial (gauge style) for the top mood,
  * 8 pastel meter cards with glass bars that glide (no jumps) and a little
    face per state; the winning card lifts and glows,
  * pill buttons (pause / snapshot / quit) with hover-lift and press-shadow.

Run it
-----
    python mood_dashboard.py                    # live camera + GUI
    python mood_dashboard.py --selftest         # GUI auto-exits after ~8s,
                                                #   saves a window screenshot
    python mood_dashboard.py --headless --source datasets/engagement/aligned/test/bored
    python mood_dashboard.py --headless --max-frames 30 --log-file run.jsonl

Keys: [space] pause   [s] snapshot   [q] quit
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from fused_detector import FusedDetector, all_states

STATES = all_states()

# ---------------- pastel palette (RGB hex for tkinter) ----------------
PALETTE = {
    "focused":    "#7FB06F",   # leafy green
    "confused":   "#E8A87C",   # peach
    "distracted": "#E88FA8",   # rose
    "neutral":    "#A9A9BC",   # misty grey
    "curious":    "#5FC9B8",   # aqua mint
    "thinking":   "#9A8FE0",   # lavender
    "bored":      "#C9B384",   # sand
    "sleepy":     "#8FA8E0",   # periwinkle
}
INK       = "#3A3A48"
CARD_BG   = "#FFFFFF"
CARD_LINE = "#F0E4DA"
LED_ON    = "#FF5B5B"
LED_OFF   = "#D8C7BE"
PILL_BG   = "#FFF3EA"
PILL_EDGE = "#E8D5C5"

# per-state face glyphs: (left-eye, right-eye, mouth)
FACES = {
    "focused":    ("dot", "dot", "smile"),
    "confused":   ("dot", "dot", "zig"),
    "distracted": ("left", "left", "flat"),
    "neutral":    ("dot", "dot", "flat"),
    "curious":    ("big", "big", "o"),
    "thinking":   ("dot", "dot", "o"),
    "bored":      ("arc", "arc", "flat"),
    "sleepy":     ("arc", "arc", "smile"),
}

# Compact face-mesh connection sets for the demo overlay. Indices are the
# standard 468-point FaceMesh layout (the 478-point build appends the two
# irises at 473/474, handled separately).
FACE_MESH = {
    "jaw":        [(10, 338), (338, 287), (287, 296), (296, 295), (295, 304),
                   (304, 454), (454, 144), (144, 145), (145, 153), (153, 154),
                   (154, 155), (155, 133), (133, 173), (173, 157), (157, 158),
                   (158, 159), (159, 160), (160, 161), (161, 246), (246, 93),
                   (93, 134), (134, 91), (91, 140), (140, 141), (141, 62),
                   (62, 38), (38, 39), (39, 40), (40, 41), (41, 120), (120, 10)],
    "left_brow":  [(70, 63), (63, 105), (105, 66), (66, 107)],
    "right_brow": [(334, 296), (296, 346), (346, 319), (319, 317)],
    "left_eye":   [(159, 145), (145, 144), (144, 153), (153, 158),
                   (158, 157), (157, 159)],
    "right_eye":  [(386, 362), (362, 397), (397, 288), (288, 354),
                   (354, 355), (355, 386)],
    "nose":       [(1, 62), (62, 10), (10, 338), (338, 25), (25, 454),
                   (454, 2), (2, 1), (1, 326)],
    "lips":       [(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                   (17, 314), (314, 405), (405, 321), (321, 41), (41, 98),
                   (98, 86), (86, 17)],
}
MESH_COLORS = {
    "jaw":        "#35D0BA",
    "left_brow":  "#FFC94D",
    "right_brow": "#FFC94D",
    "left_eye":   "#FF8FA3",
    "right_eye":  "#FF8FA3",
    "nose":       "#B39DDB",
    "lips":       "#FF8FA3",
}
IRIS_COLOR = "#FF8C42"

# fixed geometry (also used by the render path — keep in sync)
W, H = 980, 620
FEED_X, FEED_Y, FEED_W, FEED_H = 34, 132, 380, 220
DIAL_CX, DIAL_CY, DIAL_R = 760, 190, 96
METER_X, METER_Y, METER_W, METER_H, METER_GAP = 540, 330, 200, 62, 12
PILL_Y = H - 44


class DirLoop:
    """Replays an image folder, looping forever."""

    def __init__(self, files: list):
        self._files = files
        self._i = 0

    def read(self):
        img = cv2.imread(self._files[self._i])
        self._i = (self._i + 1) % len(self._files)
        return img is not None, img


def _hex_bgr(hexstr: str) -> tuple:
    """'#RRGGBB' -> (B, G, R) int tuple for OpenCV drawing."""
    r = int(hexstr[1:3], 16)
    g = int(hexstr[3:5], 16)
    b = int(hexstr[5:7], 16)
    return (b, g, r)


# Pre-compute the BGR colours once (they never change).
_MESH_BGR = {k: _hex_bgr(v) for k, v in MESH_COLORS.items()}
_IRIS_BGR = _hex_bgr(IRIS_COLOR)


def draw_mesh(bgr: np.ndarray, landmarks) -> np.ndarray:
    """Bake the colour-coded landmark mesh onto a BGR frame (in place).

    ``landmarks`` is the detector's normalised (N, 2) point array for the
    most recent frame (``None`` when no face was found). Draws the compact
    FACE_MESH connection sets + the two iris dots — enough to show how the
    478-point face model tracks the face, without cluttering the feed with
    all ~2 700 mesh triangles.
    """
    if landmarks is None:
        return bgr
    h, w = bgr.shape[:2]
    pts = landmarks * np.array([w, h], dtype=np.float32)
    for part, pairs in FACE_MESH.items():
        col = _MESH_BGR.get(part, (255, 255, 255))
        for a, b in pairs:
            if b >= len(pts):
                continue
            pa = (int(pts[a][0]), int(pts[a][1]))
            pb = (int(pts[b][0]), int(pts[b][1]))
            cv2.line(bgr, pa, pb, col, 1, cv2.LINE_AA)
    # iris dots (473/474 in the 478-point build)
    for i in (473, 474):
        if i < len(pts):
            c = (int(pts[i][0]), int(pts[i][1]))
            cv2.circle(bgr, c, 4, _IRIS_BGR, -1, cv2.LINE_AA)
    return bgr


class DetectorWorker(threading.Thread):
    """Grabs frames at full camera rate and runs the fused detector.

    Results are published on a small drop-oldest queue so the UI thread can
    never stall inference (and vice versa).
    """

    def __init__(self, use_cnn: bool = True, device: str = "NPU",
                 source: str = "", mesh: bool = True):
        super().__init__(daemon=True)
        self.det = FusedDetector(use_cnn=use_cnn, device=device)
        self.source = source
        self.mesh = mesh
        self.paused = False
        self.out_q: "queue.Queue" = queue.Queue(maxsize=3)
        self.stop_ev = threading.Event()
        self.fps = 0.0
        self._stamps: list = []
        self._fps_n = 0
        self._fps_t0 = time.time()

    def _open(self):
        if not self.source:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("could not open the laptop camera")
            cap.set(cv2.CAP_PROP_FPS, 45.0)   # ask for the target rate
            return cap
        if os.path.isdir(self.source):
            files = sorted(os.path.join(self.source, f)
                           for f in os.listdir(self.source)
                           if f.lower().endswith((".jpg", ".jpeg", ".png")))
            if not files:
                raise RuntimeError(f"no images in {self.source}")
            return DirLoop(files)
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"could not open {self.source}")
        return cap

    def _fps_tick(self) -> float:
        now = time.time()
        self._stamps.append(now)
        self._fps_n += 1
        if now - self._fps_t0 >= 1.0:
            span = now - self._stamps[0]
            if span > 1e-3:
                self.fps = (self._fps_n - 1) / span
            self._stamps.clear()
            self._fps_n = 0
            self._fps_t0 = now
        return self.fps

    def run(self) -> None:
        cap = self._open()
        while not self.stop_ev.is_set():
            ok, frame = cap.read()
            if not ok:
                if isinstance(cap, DirLoop):
                    continue                # wrapped around; keep replaying
                time.sleep(0.05)
                continue
            res = None if self.paused else self.det.update(frame)
            if self.mesh:
                draw_mesh(frame, self.det.rules.last_landmarks())
            try:
                self.out_q.put_nowait((frame, res, self._fps_tick()))
            except queue.Full:
                pass                        # UI fell behind; drop oldest
        if hasattr(cap, "release"):
            cap.release()

    @property
    def device(self) -> str:
        return self.det.device or "rules"


class Dashboard:
    """The tkinter skeuomorphic dashboard bound to a DetectorWorker."""

    def __init__(self, root, worker: DetectorWorker, snapshot_dir: str,
                 selftest_seconds: int = 0):
        import tkinter as tk
        self.tk = tk
        self.root = root
        self.worker = worker
        self.snapshot_dir = snapshot_dir
        self._paused = False
        self._display_scores = {s: 0.0 for s in STATES}
        self._last_state = "neutral"
        self._led_on = False
        self._new_item = None
        self._last_frame = None
        self._photo_ref = None
        self._selftest_end = (time.time() + selftest_seconds
                              if selftest_seconds else 0)

        self.root.title("Mood Dashboard — 8 moods, NPU")
        self.root.configure(bg="#FFF7F0")
        self.root.resizable(False, False)

        self.cv = tk.Canvas(root, width=W, height=H,
                            bg="#FFF7F0", highlightthickness=0)
        self.cv.pack()

        self._build_background()
        self._build_title()
        self._build_video_card()
        self._build_dial()
        self._build_meters()
        self._build_pills()

        self.root.bind("<space>", lambda e: self._toggle_pause())
        self.root.bind("s", lambda e: self._snapshot())
        self.root.bind("q", lambda e: self._quit())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        self.root.after(33, self._poll)     # ~30 Hz render cadence

    # ---------------- static scene --------------------------------------
    def _build_background(self) -> None:
        cv = self.cv
        # soft cream -> blush vertical gradient
        for i in range(40):
            t = i / 39
            r = int(0xFF + (0xF9 - 0xFF) * t)
            g = int(0xF7 + (0xE8 - 0xF7) * t)
            b = int(0xF0 + (0xE8 - 0xF0) * t)
            cv.create_line(0, i * 16, W, i * 16, fill=f"#{r:02X}{g:02X}{b:02X}")

    def _build_title(self) -> None:
        cv = self.cv
        cv.create_text(W // 2, 22, text="MY  MOOD  DESK",
                      font=("Segoe UI", 15, "bold"), fill=INK)
        cv.create_text(W // 2, 42, text="8 moods · live · NPU",
                      font=("Segoe UI", 9), fill="#9A8F86")

    def _card(self, x, y, w, h, r=16, small=False):
        """A raised white card with a soft offset shadow (skeuomorph)."""
        cv = self.cv
        shadow = cv.create_rectangle(x + 3, y + 4, x + w + 3, y + h + 4,
                                     fill="#EADBD0", outline="")
        body = cv.create_rectangle(x, y, x + w, y + h, fill=CARD_BG,
                                   outline=CARD_LINE,
                                   width=1 if small else 2)
        return {"shadow": shadow, "body": body}

    def _build_video_card(self) -> None:
        cv = self.cv
        vx, vy, vw, vh = 24, 64, 400, 300
        self._card(vx, vy, vw, vh, r=18)
        cv.create_rectangle(vx + 2, vy + 2, vx + vw - 2, vy + 54,
                            fill="#3A3A48", outline="")
        cv.create_text(vx + 14, vy + 28, anchor="w", text="WEBCAM",
                       font=("Consolas", 10, "bold"), fill="#F5EFE8")
        self.led_id = cv.create_oval(vx + vw - 46, vy + 18,
                                     vx + vw - 30, vy + 34,
                                     fill=LED_ON, outline="#B03A3A")
        cv.create_text(vx + vw - 54, vy + 26, anchor="e", text="LIVE",
                       font=("Segoe UI", 9, "bold"), fill="#FFD9D9")
        # Dark background rect (visible before the first frame / on error),
        # then the actual image item on top of it.
        cv.create_rectangle(FEED_X, FEED_Y,
                            FEED_X + FEED_W, FEED_Y + FEED_H,
                            fill="#22222B", outline="")
        self.feed_id = cv.create_image(FEED_X, FEED_Y, anchor="nw", image="")
        # glossy sheen across the top of the feed
        cv.create_polygon(FEED_X, FEED_Y, FEED_X + FEED_W, FEED_Y,
                          FEED_X, FEED_Y + 46,
                          fill="#FFFFFF", stipple="gray12")
        # state sticker (the big cute face, re-painted every result)
        sx, sy = vx + vw - 70, vy + vh - 70
        self.sticker = (sx, sy)
        self._draw_state_face(sx, sy, 42, "neutral", "sface")
        self.fps_chip = cv.create_text(vx + 20, vy + vh + 22, anchor="w",
                                       text="— fps",
                                       font=("Consolas", 10, "bold"),
                                       fill=INK)
        self.dev_chip = cv.create_text(vx + 120, vy + vh + 22, anchor="w",
                                       text=f"{self.worker.device}",
                                       font=("Consolas", 10),
                                       fill="#7A7A8C")
        cv.create_text(vx + vw // 2, vy + vh + 40,
                       text="big face = current mood · bars = all 8",
                       font=("Segoe UI", 9), fill="#8A829A")

    def _build_dial(self) -> None:
        cv = self.cv
        self._card(DIAL_CX - DIAL_R - 26, DIAL_CY - DIAL_R - 44,
                   DIAL_R * 2 + 52, DIAL_R * 2 + 82, r=18)
        cv.create_oval(DIAL_CX - DIAL_R, DIAL_CY - DIAL_R,
                       DIAL_CX + DIAL_R, DIAL_CY + DIAL_R,
                       fill="#FBF6F1", outline="#E8D5C5", width=4)
        for a in range(-120, 121, 15):
            rad = math.radians(a - 90)
            x0 = DIAL_CX + (DIAL_R - 14) * math.cos(rad)
            y0 = DIAL_CY + (DIAL_R - 14) * math.sin(rad)
            x1 = DIAL_CX + (DIAL_R - 4) * math.cos(rad)
            y1 = DIAL_CY + (DIAL_R - 4) * math.sin(rad)
            cv.create_line(x0, y0, x1, y1, fill="#C9B8A8",
                           width=2 if a % 45 else 1)
        cv.create_text(DIAL_CX, DIAL_CY - DIAL_R - 26, text="NOW FEELING",
                       font=("Segoe UI", 9, "bold"), fill="#9A8F86")
        self.dial_needle = cv.create_line(DIAL_CX, DIAL_CY,
                                          DIAL_CX, DIAL_CY - DIAL_R + 18,
                                          width=5, capstyle="round",
                                          fill="#E86A5C")
        cv.create_text(DIAL_CX, DIAL_CY - 4, text="CONFIDENCE",
                       font=("Segoe UI", 8), fill="#8A829A")
        self.dial_val = cv.create_text(DIAL_CX, DIAL_CY + 26, text="—",
                                       font=("Consolas", 20, "bold"),
                                       fill=INK)
        self.dial_state = cv.create_text(DIAL_CX, DIAL_CY + DIAL_R + 26,
                                         text="neutral",
                                         font=("Segoe UI", 12, "bold"),
                                         fill=PALETTE["neutral"])

    def _build_meters(self) -> None:
        cv = self.cv
        self.meters: dict[str, dict] = {}
        for i, s in enumerate(STATES):
            x = METER_X + (i % 2) * (METER_W + METER_GAP)
            y = METER_Y + (i // 2) * (METER_H + METER_GAP)
            ids = self._card(x, y, METER_W, METER_H, r=14, small=True)
            self._draw_state_face(x + 30, y + METER_H // 2, 18, s, "mface")
            cv.create_text(x + 58, y + 22, anchor="w", text=s,
                           font=("Segoe UI", 11, "bold"), fill=INK)
            bx, by = x + 58, y + 34
            bw, bh = METER_W - 74, 12
            trough = cv.create_rectangle(bx, by, bx + bw, by + bh,
                                        fill="#F2ECE6", outline="")
            fill = cv.create_rectangle(bx, by, bx, by + bh,
                                       fill=PALETTE[s], outline="")
            # glass gloss over the bar trough
            cv.create_rectangle(bx + 2, by + 1, bx + bw - 2, by + bh // 2,
                                fill="#FFFFFF", stipple="gray25")
            val = cv.create_text(x + METER_W - 12, y + METER_H // 2,
                                anchor="e", text="0%",
                                font=("Consolas", 10, "bold"),
                                fill="#8A829A")
            self.meters[s] = {"trough": trough, "fill": fill, "val": val,
                              "shadow": ids["shadow"], "body": ids["body"],
                              "x": x, "y": y}

    def _build_pills(self) -> None:
        cv = self.cv
        bx = W // 2
        self.pause_pill = self._pill(bx - 190, "PAUSE", self._toggle_pause)
        self.snap_pill = self._pill(bx - 50, "SNAPSHOT", self._snapshot)
        self.quit_pill = self._pill(bx + 90, "QUIT", self._quit)

    def _pill(self, x: int, label: str, action) -> dict:
        """Rounded pill button with shadow + hover-lift + press-shadow."""
        cv = self.cv
        w = 120
        ids = [
            cv.create_rectangle(x, PILL_Y + 4, x + w, PILL_Y + 34,
                                fill="#E4D5C8", outline=""),        # shadow
            cv.create_oval(x, PILL_Y, x + 28, PILL_Y + 30,
                           fill=PILL_BG, outline=PILL_EDGE, width=2),
            cv.create_oval(x + w - 28, PILL_Y, x + w, PILL_Y + 30,
                           fill=PILL_BG, outline=PILL_EDGE, width=2),
            cv.create_rectangle(x + 10, PILL_Y, x + w - 10, PILL_Y + 30,
                                fill=PILL_BG, outline=PILL_EDGE, width=2),
        ]
        txt = cv.create_text(x + w // 2, PILL_Y + 15, text=label,
                             font=("Segoe UI", 10, "bold"), fill=INK)
        ids.append(txt)
        pill = {"ids": ids, "txt": txt, "x": x, "y": PILL_Y, "w": w,
                "action": action, "pressed": False}

        def hover_in(e, p=pill):
            cv.tag_raise(p["ids"][0])
            cv.itemconfig(p["ids"][0], fill="#E0CDBD")

        def hover_out(e, p=pill):
            if not p["pressed"]:
                cv.itemconfig(p["ids"][0], fill="#E4D5C8")

        def press(e, p=pill):
            p["pressed"] = True
            for i in p["ids"][1:]:
                cv.move(i, 0, 2)
            cv.itemconfig(p["ids"][0], fill="#D8C2B0")

        def release(e, p=pill):
            was = p["pressed"]
            p["pressed"] = False
            for i in p["ids"][1:]:
                cv.move(i, 0, -2)
            cv.itemconfig(p["ids"][0], fill="#E4D5C8")
            if was:
                p["action"]()

        for wid in ids:
            cv.tag_bind(wid, "<Enter>", hover_in)
            cv.tag_bind(wid, "<Leave>", hover_out)
            cv.tag_bind(wid, "<Button-1>", press)
            cv.tag_bind(wid, "<ButtonRelease-1>", release)
        return pill

    # ---------------- cute face art -------------------------------------
    def _draw_state_face(self, cx, cy, r, state, tag) -> None:
        """Redraw the state's face glyph (tagged so redraws replace it)."""
        cv = self.cv
        full = f"{tag}:{state}"
        cv.delete(full)
        col = PALETTE[state]
        cv.create_oval(cx - r, cy - r, cx + r, cy + r,
                       fill="#FFFDFB", outline=col, width=2, tags=full)
        le, re_, mouth = FACES[state]
        ey = cy - r * 0.28
        ex = r * 0.42
        for dx, kind in ((-ex, le), (ex, re_)):
            if kind in ("dot", "left", "big"):
                px = cx + dx + (r * 0.20 if kind == "left" else 0)
                rad = r * (0.20 if kind == "big" else 0.12)
                cv.create_oval(px - rad, ey - rad, px + rad, ey + rad,
                               fill=INK, outline="", tags=full)
            elif kind == "arc":                     # closed, sleepy eyes
                cv.create_arc(cx + dx - r * 0.20, ey - r * 0.16,
                              cx + dx + r * 0.20, ey + r * 0.24,
                              start=0, extent=180, style="arc",
                              outline=INK, width=2, tags=full)
        my = cy + r * 0.34
        mw = r * 0.42
        if mouth == "smile":
            cv.create_arc(cx - mw, my - mw, cx + mw, my + mw,
                          start=200, extent=140, style="arc",
                          outline=INK, width=2, tags=full)
        elif mouth == "o":
            cv.create_oval(cx - r * 0.14, my - r * 0.14,
                           cx + r * 0.14, my + r * 0.14,
                           outline=INK, fill="", width=2, tags=full)
        elif mouth == "zig":
            cv.create_line(cx - mw, my, cx - mw // 3, my - r * 0.12,
                           cx + mw // 3, my, cx + mw, my - r * 0.12,
                           fill=INK, width=2, tags=full)
        else:                                        # flat
            cv.create_line(cx - mw, my, cx + mw, my,
                           fill=INK, width=2, tags=full)

    def _paint_sticker(self, state: str) -> None:
        cv = self.cv
        sx, sy = self.sticker
        self._draw_state_face(sx, sy, 42, state, "sface")
        # glow ring behind the sticker face
        cv.tag_raise(self.led_id)

    # ---------------- render loop ----------------------------------------
    def _poll(self) -> None:
        try:
            while True:
                self._new_item = self.worker.out_q.get_nowait()
        except queue.Empty:
            pass

        if self._new_item is not None:
            frame, res, fps = self._new_item
            if frame is not None:
                self._last_frame = frame
                self._render_feed(frame, fps)
            if res is not None:
                self._render_mood(res)
                self._new_item = None      # a rendered frame is consumed
        self._led_tick()
        if self._selftest_end and time.time() >= self._selftest_end:
            self._selftest_done()
            return
        self.root.after(33, self._poll)

    def _render_feed(self, frame_bgr, fps) -> None:
        if frame_bgr is None:
            return
        ok, err = self._paint_feed(frame_bgr)
        if not ok:
            # A silent pass here is what made the earlier feed look black
            # with no hint; surface the reason once instead.
            if not getattr(self, "_feed_warned", False):
                print(f"[dashboard] camera feed not rendering: {err}")
                self._feed_warned = True
                self.cv.itemconfig(self.feed_id, image="")
                self.cv.create_text(FEED_X + FEED_W // 2,
                                    FEED_Y + FEED_H // 2,
                                    text="camera feed unavailable\n"
                                         f"({err})",
                                    fill="#F5EFE8",
                                    font=("Segoe UI", 9),
                                    justify="center")
        self.cv.itemconfig(self.fps_chip, text=f"{fps:4.1f} fps")
        self.cv.itemconfig(self.dev_chip, text=self.worker.device)

    def _paint_feed(self, frame_bgr) -> tuple:
        """Render one BGR frame (mesh already baked in by the worker) to the
        feed rectangle. Returns (ok, error_string)."""
        try:
            import PIL.Image
            import PIL.ImageTk
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (FEED_W, FEED_H),
                             interpolation=cv2.INTER_AREA)
            self._photo_ref = PIL.ImageTk.PhotoImage(
                PIL.Image.fromarray(rgb))
            self.cv.itemconfig(self.feed_id, image=self._photo_ref)
            return True, ""
        except Exception as e:  # noqa: BLE001 - report, don't crash the UI
            return False, f"{type(e).__name__}: {e}"

    def _render_mood(self, res) -> None:
        cv = self.cv
        state = res.state
        # glide every bar towards its target (no jumps = smooth)
        for s in STATES:
            tgt = float(res.scores.get(s, 0.0))
            disp = self._display_scores[s] + (tgt - self._display_scores[s]) * 0.35
            self._display_scores[s] = disp
            m = self.meters[s]
            x0, y0 = m["x"] + 58, m["y"] + 34
            w = int((METER_W - 74) * min(disp, 1.0))
            cv.coords(m["fill"], x0, y0, x0 + w, y0 + 12)
            cv.itemconfig(m["val"], text=f"{disp * 100:3.0f}%")
            glow = (s == state)
            cv.itemconfig(m["body"],
                         outline="#E86A5C" if glow else CARD_LINE,
                         width=3 if glow else 1)
            cv.itemconfig(m["shadow"],
                          fill="#F0C9B8" if glow else "#EADBD0")
        # spring the dial needle to the confidence
        self._display_conf = getattr(self, "_display_conf", 0.0)
        self._display_conf += (res.confidence - self._display_conf) * 0.3
        conf = min(max(self._display_conf, 0.0), 1.0)
        ang = math.radians(-120 + 240 * conf)
        nx = DIAL_CX + (DIAL_R - 22) * math.sin(ang)
        ny = DIAL_CY - (DIAL_R - 22) * math.cos(ang)
        cv.coords(self.dial_needle, DIAL_CX, DIAL_CY, nx, ny)
        cv.itemconfig(self.dial_val, text=f"{res.confidence * 100:3.0f}%")
        if state != self._last_state:
            self._last_state = state
            cv.itemconfig(self.dial_state, text=state, fill=PALETTE[state])
            self._paint_sticker(state)
        else:
            # keep the sticker's glow ring alive even on the same state
            pass

    def _led_tick(self) -> None:
        self._led_on = not self._led_on
        on = self._led_on and not self._paused
        self.cv.itemconfig(self.led_id,
                           fill=LED_ON if on else LED_OFF,
                           outline="#B03A3A" if on else "#C4B5AC")

    # ---------------- user actions ---------------------------------------
    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self.worker.paused = self._paused
        self.cv.itemconfig(self.pause_pill["txt"],
                           text="RESUME" if self._paused else "PAUSE")
        self.cv.itemconfig(self.led_id,
                           fill=LED_OFF if self._paused else LED_ON)

    def _snapshot(self) -> None:
        if self._last_frame is None:
            return
        os.makedirs(self.snapshot_dir, exist_ok=True)
        p = Path(self.snapshot_dir) / f"mood_snap_{int(time.time())}.png"
        cv2.imwrite(str(p), self._last_frame)
        print(f"[dashboard] snapshot -> {p}")

    def _quit(self) -> None:
        # Drop the Tk PhotoImage before the interpreter/Tcl state goes away;
        # otherwise the C layer can fatal at exit ("unknown option" / crash).
        try:
            self.cv.itemconfig(self.feed_id, image="")
        except Exception:
            pass
        self._photo_ref = None
        self.worker.stop_ev.set()
        self.root.destroy()

    def _selftest_done(self) -> None:
        out = Path(self.snapshot_dir) / "dashboard_selftest.png"
        os.makedirs(self.snapshot_dir, exist_ok=True)
        try:
            import PIL.Image
            import PIL.ImageGrab
            root = self.root
            # Force our window above any other window before grabbing, else
            # ImageGrab's screen bbox would capture whatever is overlapping.
            root.update()
            root.attributes("-topmost", True)
            root.lift()
            root.update()
            box = (root.winfo_rootx(), root.winfo_rooty(),
                   root.winfo_rootx() + root.winfo_width(),
                   root.winfo_rooty() + root.winfo_height())
            PIL.ImageGrab.grab(bbox=box).save(str(out))
            print(f"[dashboard] selftest screenshot -> {out}")
        except Exception as e:
            print(f"[dashboard] selftest screenshot failed: {e}")
        self._quit()


# ============================ headless path ============================
def run_headless(source: str, max_frames: int, log_file: str,
                 use_cnn: bool, device: str) -> int:
    worker = DetectorWorker(use_cnn=use_cnn, device=device, source=source)
    print(f"[dash-h] open (device={worker.device}, "
          f"source={source or 'live camera'})")
    worker.start()
    fh = open(log_file, "a", encoding="utf-8") if log_file else None
    logged = 0
    last_seen = time.time()
    try:
        while not worker.stop_ev.is_set():
            try:
                frame, res, fps = worker.out_q.get(timeout=0.2)
                last_seen = time.time()
            except queue.Empty:
                if time.time() - last_seen > 30:
                    print("[dash-h] no frames for 30s; giving up")
                    break
                continue
            if res is None:
                continue
            logged += 1
            if fh is not None:
                fh.write(json.dumps({
                    "idx": logged, "state": res.state,
                    "conf": round(res.confidence, 3),
                    "scores": {k: round(v, 3) for k, v in res.scores.items()},
                    "device": res.device, "fps": round(fps, 2),
                    "latency_ms": round(res.latency_ms, 1),
                    "face_found": res.face_found,
                }) + "\n")
            if logged % 30 == 1:
                print(f"[dash-h] f{logged} {res.state} "
                      f"{res.confidence:.2f} fps={fps:.1f} "
                      f"dev={res.device} lat={res.latency_ms:.0f}ms")
            if max_frames and logged >= max_frames:
                break
    finally:
        worker.stop_ev.set()
        worker.join(timeout=3)
        if fh is not None:
            fh.close()
            print(f"[dash-h] log: {log_file}")
    return 0


# ============================ CLI =======================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-cnn", action="store_true", help="rules only")
    ap.add_argument("--device", default="NPU")
    ap.add_argument("--source", default="",
                    help="video file or image dir to replay (empty = live cam)")
    ap.add_argument("--headless", action="store_true",
                    help="no GUI; run detector + optional log")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--log-file", default="")
    ap.add_argument("--selftest", action="store_true",
                    help="GUI auto-exits after ~8s and saves a screenshot")
    ap.add_argument("--no-mesh", action="store_true",
                    help="do not draw the landmark mesh over the camera feed")
    ap.add_argument("--snapshot-dir", default="mood_snaps")
    args = ap.parse_args()

    if args.headless:
        raise SystemExit(run_headless(args.source, args.max_frames,
                                      args.log_file, not args.no_cnn,
                                      args.device))

    import tkinter as tk
    root = tk.Tk()
    worker = DetectorWorker(use_cnn=not args.no_cnn, device=args.device,
                            source=args.source, mesh=not args.no_mesh)
    worker.start()
    Dashboard(root, worker, args.snapshot_dir,
              selftest_seconds=8 if args.selftest else 0)
    root.mainloop()


if __name__ == "__main__":
    main()
