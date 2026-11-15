# Architecture and data flow

## Runtime components

```text
Camera frame (OpenCV BGR)
        │
        ▼
MediaPipe FaceLandmarker
478 normalized face points + transform matrix
        │
        ├── EAR: eye openness / sustained closure
        ├── MAR: mouth openness / yawn duration
        ├── head pose: yaw, pitch, roll
        └── iris position
                │
                ▼
10-second rolling-window rules
all 8 state scores + evidence + feature vector
                │
Camera crop ────► OpenVINO CNN on NPU/CPU
                 classes listed in model_ir/classes.json
                │
                ▼
FusedDetector
CNN-known states: 0.5 × CNN + 0.5 × rules
other states: rules-only
hard overrides: sustained closed eyes, yawn, look-away
                │
                ▼
MoodResult
state, confidence, all-8 scores, evidence, device, latency
                │
                ▼
Dashboard UI / OpenCV UI / JSONL test log
```

## Landmark rules

`engagement_detector.py` runs MediaPipe FaceLandmarker from `assets/face_landmarker.task` and maintains a rolling history. It scores:

- **sleepy:** sustained closed eyes;
- **bored:** confirmed yawn plus eye closure contribution;
- **distracted:** sustained head turn-away;
- **focused:** calm, steady face with a downward gaze;
- **thinking:** upward gaze with calm posture;
- **curious:** upward gaze with a weaker calm signal;
- **confused:** head tilt;
- **neutral:** calm centered baseline.

The last three are explicitly approximations based on geometry. They should not be presented as validated psychological measurements.

## CNN/NPU path

`export_openvino.py` exports an EfficientNet-B0 checkpoint to an OpenVINO IR pair (`emotion_npu.xml` and `emotion_npu.bin`) plus `classes.json`. `fused_detector.py` compiles the IR on the requested OpenVINO device, preferring `NPU` and falling back to `CPU` if compilation fails.

The exported graph accepts `[1, 3, 224, 224]` RGB float input in `[0, 1]`; ImageNet normalization is baked into the graph at export time. The runtime reads class names from `classes.json`, so retraining with additional classes does not require detector code changes.

## Dashboard threading

`mood_dashboard.py` uses a worker thread for camera capture and inference. A small bounded queue drops stale frames if the UI falls behind. The tkinter thread only renders the newest result, which keeps the interface responsive while the NPU performs inference.

The worker also draws the optional demo mesh onto the frame using the detector's latest landmarks. Use `--no-mesh` to disable it. The dashboard feed is a real canvas image item, not a rectangle; this matters because tkinter rectangles cannot display `PhotoImage` objects.

## Outputs

Each result is represented internally by `MoodResult`:

```text
state        winning state name
confidence   top-score margin plus a small baseline, clamped to 0..1
scores       fused score for all eight states
cnn_scores   CNN probabilities mapped onto all eight names
rules_scores rules score for all eight states
face_found   whether a face is available in the rules window
device       NPU, CPU, or none
latency_ms   detector update latency
evidence     rolling-window signals and override reasons
```
