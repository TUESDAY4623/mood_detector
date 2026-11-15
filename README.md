# OrbitDesk Mood Detector

A Windows desktop mood/engagement detector that combines a MediaPipe 478-point face landmark pipeline with an OpenVINO CNN running on an Intel NPU when available.

The project produces eight engagement-style states in one pass:

`focused` · `confused` · `distracted` · `neutral` · `curious` · `thinking` · `bored` · `sleepy`

> **Important:** These labels are engagement/mood indicators, not medical or psychological diagnoses. `curious`, `thinking`, and `confused` are geometric approximations until additional labeled data is recorded.

## Download the tested Windows app

Open the repository's **Releases** page and download **MoodDashboard.exe**. The executable includes the dashboard, OpenVINO runtime, MediaPipe runtime, the face-landmarker asset, and the exported model IR.

The app uses your default camera. The dashboard displays:

- live camera feed;
- colored face mesh overlay (jaw, brows, eyes, nose, and iris dots);
- current state and confidence dial;
- confidence meters for all eight states;
- NPU/CPU device and FPS information;
- pause, snapshot, and quit controls.

Keyboard shortcuts: **Space** pause/resume, **S** snapshot, **Q** quit.

The repository also contains `MoodDashboard.spec` so the EXE can be rebuilt locally.

## Run from source

Windows PowerShell or Command Prompt:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe mood_dashboard.py
```

The checked-in model and face-landmarker assets are required for the live detector:

```text
model_ir/emotion_npu.xml
model_ir/emotion_npu.bin
model_ir/classes.json
assets/face_landmarker.task
```

Run with a selected device:

```powershell
.venv\Scripts\python.exe mood_dashboard.py --device NPU
.venv\Scripts\python.exe mood_dashboard.py --device CPU
.venv\Scripts\python.exe mood_dashboard.py --no-cnn --no-mesh
```

The dashboard asks the camera for a 45 FPS mode. The tested laptop camera delivered approximately 30 FPS; a camera that supports 45+ FPS can feed the detector at the full target rate.

## Test without opening the GUI

Use a directory of images as a replay source:

```powershell
.venv\Scripts\python.exe mood_dashboard.py `
  --headless `
  --source datasets/engagement/aligned/test/bored `
  --max-frames 30 `
  --log-file _evidence/dashboard.jsonl
```

Run the complete legacy detector test suite:

```powershell
.venv\Scripts\python.exe run_tests.py
```

This exercises the Python path and `MoodDetector.exe` against the live camera plus bored, sleepy, and neutral replay folders. Evidence is written to `_evidence/` locally and is intentionally not committed.

## Build the EXE

Install dependencies first, then run PyInstaller from the project root:

```powershell
.venv\Scripts\python.exe -m PyInstaller MoodDashboard.spec --noconfirm
```

Output:

```text
dist/MoodDashboard.exe
```

The spec collects MediaPipe and OpenVINO native modules, includes the NPU model and face-landmarker asset, and excludes training-only PyTorch packages from the runtime bundle.

## Training/data pipeline

The runtime works today with landmark rules for all eight states. The CNN currently uses the exported classes listed in `model_ir/classes.json`. To add data:

```powershell
# Optional public DAiSEE data: place videos + annotation CSV under datasets/engagement/daisee_input/
.venv\Scripts\python.exe prepare_daisee.py --csv datasets/engagement/daisee_input/DAiSEE_annotation.csv --videos-dir datasets/engagement/daisee_input

# Optional self-recorded data for curious/thinking/confused and other states
.venv\Scripts\python.exe record_state.py record --state thinking --seconds 30
.venv\Scripts\python.exe record_state.py record --state curious --seconds 30
.venv\Scripts\python.exe record_state.py record --state confused --seconds 30
.venv\Scripts\python.exe record_state.py ingest

# Combine, train, export to OpenVINO, and verify
.venv\Scripts\python.exe pipeline.py
```

Training is resumable. Checkpoints are local-only and are not included in the public repository.

## Documentation

- [Architecture and data flow](docs/ARCHITECTURE.md)
- [Training and datasets](docs/TRAINING.md)
- [Testing and verification](docs/TESTING.md)
- [Release and packaging](docs/RELEASE.md)
- [Dataset feasibility notes](docs/DATASET_INDEX.md)

## Repository layout

```text
assets/                    MediaPipe and face detector assets
model_ir/                  OpenVINO XML/BIN model and classes.json
mood_dashboard.py          primary skeuomorphic dashboard + mesh overlay
mood_ui.py                OpenCV all-8-state UI
fused_detector.py         landmark/rules + CNN fusion
engagement_detector.py    MediaPipe signals and rolling-window rules
record_state.py           self-recording and session-level dataset ingest
pipeline.py               prepare → combine → train → export → verify
train.py                  resumable EfficientNet-B0 training
export_openvino.py        PyTorch checkpoint → OpenVINO IR
MoodDashboard.spec        PyInstaller recipe for the primary EXE
dist/MoodDashboard.exe    primary Windows app, stored with Git LFS
```

## License and data notice

No project license has been selected yet. Until a license is added, treat the source as all-rights-reserved. Third-party datasets and model assets may have their own licenses; obtain and review them separately before redistribution.

The checked-in code does not contain credentials. Do not commit camera recordings, DAiSEE files, private datasets, checkpoints, or secrets.
