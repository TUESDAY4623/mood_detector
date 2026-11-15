# Training and datasets

## Current data contract

Training uses an ImageFolder-compatible tree:

```text
datasets/engagement/combined/
├── train/<state>/*.jpg
├── validation/<state>/*.jpg
└── test/<state>/*.jpg
```

`train.py` discovers class names from the directory tree and saves them into the checkpoint configuration. The current exported runtime model's exact class list is in `model_ir/classes.json`.

## Included data mapping

The repository does not redistribute raw datasets. The local project history used the following mapping from a yawn/eye dataset:

| Runtime state | Source signal |
|---|---|
| `bored` | yawn |
| `sleepy` | closed eyes |
| `neutral` | open eyes and no yawn |

The dataset license and redistribution terms must be reviewed independently before adding any raw or derived dataset files to a public repository.

## DAiSEE preparation

DAiSEE is an optional external classroom-engagement dataset. Place its videos and annotation CSV under `datasets/engagement/daisee_input/` locally, then run:

```powershell
.venv\Scripts\python.exe prepare_daisee.py `
  --csv datasets/engagement/daisee_input/DAiSEE_annotation.csv `
  --videos-dir datasets/engagement/daisee_input `
  --face-filter
```

The preparation stage uses OpenCV frame extraction, skips “not present” rows, maps engagement levels as follows, and keeps subjects isolated across splits:

```text
1 -> focused
2 -> neutral
3 -> distracted
4 -> bored
5 -> skipped (not present)
```

Do not commit the downloaded videos, CSV, or derived frames.

## Self-recorded data

`record_state.py` creates labeled sessions for states that lack suitable public image data:

```powershell
.venv\Scripts\python.exe record_state.py record --state thinking --seconds 30
.venv\Scripts\python.exe record_state.py record --state curious --seconds 30
.venv\Scripts\python.exe record_state.py record --state confused --seconds 30
.venv\Scripts\python.exe record_state.py ingest --face-filter
```

Recording writes raw session frames and `meta.json` under `datasets/engagement/recordings/`. Ingestion writes `aligned_rec/manifest.json` and a session-level 70/15/15 train/validation/test split. A session is never split across partitions, preventing a person's nearly identical frames from leaking into test data.

## One-command pipeline

```powershell
.venv\Scripts\python.exe pipeline.py --report
.venv\Scripts\python.exe pipeline.py
```

Stages:

1. `prepare`: convert DAiSEE if local input exists;
2. `combine`: merge the included aligned tree, DAiSEE output, and self-recorded output;
3. `train`: train or resume EfficientNet-B0 checkpoints;
4. `export`: convert the best checkpoint to OpenVINO IR and write `classes.json`;
5. `verify`: compile the IR on NPU when available and check output/class consistency.

Training is intentionally resumable. Checkpoints and datasets stay local and are excluded by `.gitignore`.

## Honest limitations

The landmark/rules engine provides all eight state outputs live. The CNN is only authoritative for classes present in its exported `classes.json`; states without labeled CNN data remain rules-only until data is collected and the model is retrained/exported.
