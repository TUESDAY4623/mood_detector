# Dataset feasibility notes

The active local dataset index is at [datasets/engagement/DATASET_INDEX.md](../datasets/engagement/DATASET_INDEX.md). This page summarizes the same situation for the public repository.

## What the runtime can do now

The live detector can produce all eight states because the rules engine and feature extraction are available for all of them:

`focused`, `confused`, `distracted`, `neutral`, `curious`, `thinking`, `bored`, `sleepy`

This is independent of whether the exported CNN has been trained on all eight labels.

## Public data coverage

| State | Typical data source | Status |
|---|---|---|
| bored | yawn frames, DAiSEE engagement level 4 | usable with local/public sources |
| sleepy | closed-eye/drowsiness frames | usable with local/public sources |
| neutral | open-eye/no-yawn frames, DAiSEE engagement level 2 | usable with local/public sources |
| focused | DAiSEE engagement level 1 | optional after local DAiSEE preparation |
| distracted | DAiSEE engagement level 3 | optional after local DAiSEE preparation |
| curious | no small public image dataset identified | collect self-recorded sessions |
| thinking | no small public image dataset identified | collect self-recorded sessions |
| confused | no small public image dataset identified | collect self-recorded sessions |

DAiSEE level 5 rows are treated as not-present and are skipped during preparation.

## Recording recommendations

For each missing state, record several short sessions with varied posture and lighting:

```powershell
.venv\Scripts\python.exe record_state.py record --state curious --seconds 30
.venv\Scripts\python.exe record_state.py record --state thinking --seconds 30
.venv\Scripts\python.exe record_state.py record --state confused --seconds 30
.venv\Scripts\python.exe record_state.py ingest --face-filter
```

Session-level splitting is important: frames from the same recording session must not appear in more than one of train, validation, or test.

## Data handling

- Do not commit raw or derived dataset files.
- Do not commit webcam recordings or face images.
- Keep `datasets/`, `checkpoints/`, and `_evidence/` out of git.
- Verify the license and permitted use of every third-party dataset before downloading or redistributing it.
