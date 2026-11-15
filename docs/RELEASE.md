# Release and packaging

## Primary release artifact

`dist/MoodDashboard.exe` is the main Windows distribution artifact. It is packaged as a one-file PyInstaller build and should be published as a GitHub Release asset and tracked in the repository with Git LFS.

## Build from source

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m PyInstaller MoodDashboard.spec --noconfirm
```

The build reads `MoodDashboard.spec` and bundles:

- `mood_dashboard.py` as the entry point;
- `fused_detector.py` and `engagement_detector.py`;
- MediaPipe and OpenVINO native modules collected by the PyInstaller spec;
- `model_ir/emotion_npu.xml`, `model_ir/emotion_npu.bin`, and `model_ir/classes.json`;
- `assets/face_landmarker.task`;

Training-only packages such as PyTorch and torchvision are excluded from the runtime bundle.

## Runtime data

The packaged EXE is expected to run without downloading model files. The face-landmarker model and OpenVINO IR are read from bundled resources using the same path-resolution logic that supports both normal Python execution and PyInstaller extraction.

## Git LFS

The repository uses `.gitattributes` to mark the EXE as an LFS object:

```text
dist/MoodDashboard.exe filter=lfs diff=lfs merge=lfs -text
```

Before the first push, enable LFS in the environment:

```powershell
git lfs install
```

Pushing the EXE requires a Git remote configured with LFS support and sufficient LFS storage/quota in the destination repository or account.

## GitHub Release notes

When publishing a release:

1. Upload `dist/MoodDashboard.exe` as the primary asset.
2. State the tested platform as Windows.
3. Mention that camera access is required.
4. Mention that Intel NPU is used when OpenVINO can compile for it and that CPU is the fallback.
5. Mention that the app shows engagement-style indicators, not medical diagnoses.
6. Do not include datasets, checkpoints, recordings, or `_evidence/` output in the release asset set.
7. Do not include secrets or credentials.
