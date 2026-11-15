# Testing and verification

## Published evidence

The detector was verified locally on Windows with OpenVINO reporting `NPU` as the active device. The primary checks were:

- Python `mood_ui.py` headless path;
- legacy `MoodDetector.exe` headless path;
- `MoodDashboard.exe` headless path;
- dashboard GUI selftest;
- live camera capture;
- face-present image-folder replay;
- EXE build and resource loading.

The local `_evidence/` directory contains JSONL logs and screenshots when the project is tested. It is excluded from the public repository because it is generated machine-specific output.

## Re-run the detector suite

From a Windows virtual environment with dependencies installed:

```powershell
.venv\Scripts\python.exe run_tests.py
```

The suite runs Python and `MoodDetector.exe` over:

- live camera frames;
- bored replay frames;
- sleepy replay frames;
- neutral replay frames.

Each JSONL record includes:

```text
idx, state, conf, scores, cnn_scores, rules_scores,
device, fps, latency_ms, face_found
```

For a faster dataset-only run:

```powershell
.venv\Scripts\python.exe run_tests.py --no-cam --dir-frames 20
```

## Dashboard checks

Headless NPU smoke test:

```powershell
dist\MoodDashboard.exe --headless --max-frames 3 --log-file _evidence\dashboard-exe.jsonl
```

GUI selftest:

```powershell
dist\MoodDashboard.exe --selftest --snapshot-dir _evidence
```

The selftest opens the dashboard, processes frames for approximately eight seconds, captures a screenshot, and exits. A face-present replay source can be supplied to make mesh rendering deterministic:

```powershell
dist\MoodDashboard.exe `
  --selftest `
  --source datasets\engagement\aligned\test\bored `
  --snapshot-dir _evidence
```

## Interpreting FPS

The application requests a 45 FPS camera mode and the inference path is fast enough for that target when the source supplies frames. The tested laptop webcam delivered approximately 30 FPS, so live-camera FPS is hardware-limited. Image-folder replay can exceed the live-camera rate.

FPS is not an accuracy metric. Check `face_found` and the per-state scores as well as the device and latency fields.

## Known limitations

- Some closed-eye dataset images are cropped too tightly for the MediaPipe landmarker; a no-face frame intentionally falls back to neutral rather than inventing a mood.
- `curious`, `thinking`, and `confused` are rules-based approximations until representative labeled data is added.
- Camera permission, lighting, occlusion, glasses, and camera placement affect face tracking.
- The app is not a clinical, diagnostic, employment, or mental-health assessment tool.
