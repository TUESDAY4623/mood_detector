"""checkpoint.py — resumable training-state checkpoint library.

The Kaggle notebook only ever saved ``model_state_dict`` (plus a best-F1),
so an interrupted run could not be resumed: the optimizer, scheduler,
grad-scaler and epoch counter were lost. This module persists the *full*
training state and restores it in a single call, so training continues
exactly where it stopped.

Security
--------
State files are written with ``torch.save`` and read back with
``torch.load(..., weights_only=True)``. Only tensors, numbers, strings,
dicts, lists and tuples round-trip, so a checkpoint can never carry an
arbitrary pickle payload (no ``__reduce__``, no code execution).

The caller supplies the destination path; the module itself does no path
manipulation or directory creation.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import torch

# Fixed, well-known file names. The caller picks the directory; these are
# just the names inside it.
LATEST_NAME = "latest.pt"
BEST_NAME = "best.pt"


@dataclass
class CheckpointConfig:
    """Hyper-parameters recorded alongside the weights for reproducibility."""

    architecture: str = "efficientnet_b0"
    num_classes: int = 9
    batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 0
    image_size: int = 224
    class_names: Optional[list] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    scaler: Any = None,
    metrics: Optional[dict] = None,
    config: Optional[CheckpointConfig] = None,
) -> str:
    """Atomically persist the full training state at ``path``.

    The write goes to a sibling ``.tmp`` file and is renamed into place only
    when complete, so an interrupted save never leaves a half-written
    checkpoint.
    """
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),            # the next epoch to run
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if config is not None:
        payload["config"] = config.to_dict()

    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    scaler: Any = None,
) -> int:
    """Restore training state from ``path``; return the next epoch to run.

    Returns ``0`` when the file does not exist (fresh start). Optimiser,
    scheduler and scaler are only restored when the caller passed them in,
    so the same file can seed both a resumed run and a deployment model.
    """
    if not os.path.exists(path):
        return 0
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in payload:
        scaler.load_state_dict(payload["scaler_state_dict"])
    return int(payload.get("epoch", 0))


def load_model_only(path: str, model: torch.nn.Module) -> None:
    """Load just the weights (for inference), ignoring optimiser state."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"])


__all__ = [
    "CheckpointConfig",
    "LATEST_NAME",
    "BEST_NAME",
    "save_checkpoint",
    "load_checkpoint",
    "load_model_only",
]
