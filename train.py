"""train.py — train the 9-emotion EfficientNet-B0 on the local dataset with
resumable checkpoints.

This is the local, CPU-safe replacement for the Kaggle notebook. It:
  * builds an EfficientNet-B0 with a 9-way head
  * trains with class-balanced (sqrt-inverse-frequency) CrossEntropyLoss
  * saves a resumable checkpoint after *every* epoch (model + optimizer +
    scheduler + scaler + epoch + best metrics) via ``checkpoint.py``
  * resumes from the latest checkpoint automatically, so an interrupted run
    continues exactly where it stopped
  * keeps a separate "best" checkpoint (highest validation accuracy) for
    deployment/export

Run it
-----
    python train.py                       # full run, auto-resume
    python train.py --epochs 10          # limit epochs
    python train.py --max-samples 200 --epochs 1   # fast CPU smoke test
    python train.py --fresh               # ignore any existing checkpoint

Only PyTorch (CPU is fine) is required here. After a run, use
export_openvino.py to turn ``best.pt`` into an NPU-ready OpenVINO IR model.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm

from checkpoint import BEST_NAME, LATEST_NAME, save_checkpoint, load_checkpoint, load_model_only
from checkpoint import CheckpointConfig

PROJECT_DIR = Path(__file__).resolve().parent
CLASS_NAMES = [
    "angry", "confused", "disgusted",
    "fear", "happy", "neutral",
    "sad", "shy", "surprised",
]
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model(num_classes: int = 9, weights: str = "default") -> nn.Module:
    """EfficientNet-B0 with a small, dropout-regularised ``num_classes`` head."""
    weights_map = {"default": EfficientNet_B0_Weights.DEFAULT, "none": None}
    model = efficientnet_b0(weights=weights_map.get(weights, EfficientNet_B0_Weights.DEFAULT))
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def build_loaders(dataset_root: Path, batch_size: int, num_workers: int,
                  max_samples_per_class: int | None = None):
    """Build ImageFolder loaders for train/valid(-ation)/test under ``dataset_root``.

    Accepts both ``valid`` and ``validation`` as the validation split name;
    empty class folders inside a split are skipped by ImageFolder
    automatically (a class present in train but not in test is fine).
    """
    train_tf, eval_tf = build_transforms()
    val_split = "validation" if (dataset_root / "validation").is_dir() else "valid"

    def make(split: str, tf):
        root = dataset_root / split
        original = datasets.ImageFolder(root, transform=tf)
        if max_samples_per_class is not None:
            # cap to the first N images of each class (deterministic order)
            keep: list[int] = []
            by_class: dict[str, int] = {}
            for idx, (_, label) in enumerate(original.samples):
                label_name = original.classes[label]
                if by_class.get(label_name, 0) < max_samples_per_class:
                    keep.append(idx)
                    by_class[label_name] = by_class.get(label_name, 0) + 1
            ds = torch.utils.data.Subset(original, keep)
            # Re-attach the ImageFolder attributes Subset would otherwise drop.
            ds.classes = original.classes
            ds.class_to_idx = original.class_to_idx
            ds.samples = [original.samples[i] for i in keep]
            ds.targets = [original.targets[i] for i in keep]
            return ds
        return original

    train_ds = make("train", train_tf)
    val_ds = make(val_split, eval_tf)
    test_ds = make("test", eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def class_weights(train_ds) -> torch.Tensor:
    """Moderate sqrt-inverse-frequency weights, normalised to mean 1."""
    n = len(train_ds.classes)
    counts = torch.bincount(torch.tensor(train_ds.targets), minlength=n).float()
    counts = torch.clamp(counts, min=1.0)
    w = (counts.sum() / (len(counts) * counts)).sqrt()
    return w / w.mean()


def evaluate(model, loader, device, criterion, classes: int, name: str = "eval") -> dict:
    """Return loss / accuracy / macro-F1 plus the confusion matrix."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    cm = [[0] * classes for _ in range(classes)]
    pbar = tqdm(loader, desc=f"{name}  ", unit="batch", leave=False)
    with torch.inference_mode():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            seen += y.size(0)
            for t, p in zip(y.tolist(), pred.tolist()):
                cm[t][p] += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")
    return {"loss": total_loss / max(seen, 1),
            "accuracy": correct / max(seen, 1),
            "macro_f1": _macro_f1(cm),
            "confusion": cm}


def _macro_f1(cm: list[list[int]]) -> float:
    """Macro-averaged F1 from a confusion matrix (rows=true, cols=predicted)."""
    classes = len(cm)
    f1s = []
    for i in range(classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(classes)) - tp
        fn = sum(cm[i]) - tp
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return sum(f1s) / len(f1s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train EfficientNet-B0 on a class-folder dataset with resumable checkpoints.")
    ap.add_argument("--dataset", default=str(PROJECT_DIR / "datasets" / "engagement" / "aligned"),
                    help="Root containing train/valid(or validation)/test class folders")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--num-workers", type=int, default=14)
    ap.add_argument("--patience", type=int, default=50,
                    help="Early-stop after this many epochs without val improvement")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap images per class per split (fast CPU smoke test)")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore any existing latest checkpoint and start from scratch")
    ap.add_argument("--weights", default="default", choices=["default", "none"],
                    help="Backbone init: 'default' = ImageNet, 'none' = random")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_root = Path(args.dataset)
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset not found at {dataset_root}")

    # --- immediate feedback: say what we are about to do (before the slow parts) ---
    print(f"[train] starting ... scanning dataset under {dataset_root}")
    sys.stdout.flush()
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(
        dataset_root, args.batch_size, args.num_workers, args.max_samples)
    print(f"[train] dataset scanned: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} images "
          f"({len(train_ds)//args.batch_size} train batches @ batch {args.batch_size})")
    print(f"[train] device={device} | torch {torch.__version__} | threads={torch.get_num_threads()}")
    # The class set comes from the dataset itself (ImageFolder sorts class
    # folders alphabetically), so any dataset layout works: the old 9
    # expression classes or the engagement states etc.
    classes = list(train_ds.classes)
    n_classes = len(classes)
    counts = Counter(train_ds.targets)
    dist = ", ".join(f"{classes[i]}={counts.get(i,0)}" for i in range(n_classes))
    print(f"[train] classes ({n_classes}): {classes}")
    print(f"[train] class distribution (train): {dist}")

    print(f"[train] building EfficientNet-B0 (ImageNet backbone, {n_classes}-way head) ...")
    sys.stdout.flush()
    model = build_model(num_classes=n_classes, weights=args.weights).to(device)

    # ---- checkpoint setup (resumable) ----
    ck_dir = os.path.join(str(PROJECT_DIR), "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    ck_latest = os.path.join(ck_dir, LATEST_NAME)
    ck_best = os.path.join(ck_dir, BEST_NAME)
    if args.fresh and os.path.exists(ck_latest):
        os.remove(ck_latest)

    use_cuda_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                            factor=0.5, patience=1)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds).to(device))
    config = CheckpointConfig(epochs=args.epochs, batch_size=args.batch_size,
                             lr=args.lr, class_names=train_ds.classes)

    start_epoch = 0
    best_acc = -1.0
    stale = 0
    if not args.fresh and os.path.exists(ck_latest):
        start_epoch = load_checkpoint(ck_latest, model, optimizer, scheduler, scaler)
        payload = torch.load(ck_latest, map_location="cpu", weights_only=True)
        best_acc = float(payload.get("metrics", {}).get("best_acc", -1.0))
        config.epochs = payload.get("config", {}).get("epochs", args.epochs)
        print(f"[train] RESUMING: continuing from epoch {start_epoch} "
              f"(best val acc so far {best_acc:.4f})")
    else:
        print(f"[train] fresh start (no previous checkpoint or --fresh was given)")
    sys.stdout.flush()

    total_epochs = max(args.epochs, config.epochs)
    print(f"[train] plan: {total_epochs - start_epoch} epoch(s) to run "
          f"(lr={args.lr:g}, batch={args.batch_size}, patience={args.patience})")
    est = (len(train_ds) // args.batch_size) * 3.0  # rough CPU seconds/batch
    print(f"[train] estimate (CPU, 224px, EfficientNet-B0): "
          f"~{est//3600}h {int(est%3600//60)}m per epoch. Watch for epoch 1 "
          f"to confirm. Press Ctrl+C to stop; re-run the same command later "
          f"to resume from the last saved epoch.")
    print("-" * 72)
    sys.stdout.flush()

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"ep {epoch+1}/{total_epochs} train",
                          unit="batch", leave=False, position=0)
        for x, y in train_pbar:
            x = x.to(device, non_blocking=use_cuda_amp)
            y = y.to(device, non_blocking=use_cuda_amp)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_cuda_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if use_cuda_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
            train_pbar.set_postfix(loss=f"{loss.item():.3f}",
                                   acc=f"{correct/max(seen,1):.3f}")

        train_metrics = {"loss": total_loss / max(seen, 1), "accuracy": correct / max(seen, 1)}
        val_metrics = evaluate(model, val_loader, device, criterion, n_classes,
                              name="val")
        scheduler.step(val_metrics["accuracy"])
        rec = {"epoch": epoch + 1, **train_metrics,
               "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"],
               "val_macro_f1": val_metrics["macro_f1"],
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        print(f"[train] ep {rec['epoch']:>2}/{total_epochs} | "
              f"train acc {train_metrics['accuracy']:.3f} (loss {train_metrics['loss']:.3f}) | "
              f"val acc {val_metrics['accuracy']:.3f} (loss {val_metrics['loss']:.3f}, "
              f"macro-F1 {val_metrics['macro_f1']:.3f}) | "
              f"lr {rec['lr']:.2e} | epoch took {rec['seconds']/60:.1f} min")
        sys.stdout.flush()

        improved = val_metrics["accuracy"] > best_acc
        if improved:
            best_acc = val_metrics["accuracy"]
            stale = 0
            save_checkpoint(ck_best, model, epoch=epoch + 1,
                            metrics={"val_acc": best_acc,
                                     "val_macro_f1": val_metrics["macro_f1"]},
                            config=config)
            print(f"[train] new best val acc {best_acc:.4f} -> saved to {ck_best}")
        else:
            stale += 1

        # Always persist the resumable "latest" checkpoint (opt/sched/scaler + epoch+1)
        save_checkpoint(ck_latest, model, epoch=epoch + 1, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler,
                        metrics={"val_acc": val_metrics["accuracy"], "best_acc": best_acc},
                        config=config)
        print(f"[train] checkpoint saved -> {ck_latest} (resume here if you stop)")
        sys.stdout.flush()

        if not improved and stale >= args.patience:
            print(f"[train] [early-stop] no val improvement for {stale} epochs; stopping.")
            break

    # ---- final eval on the best model ----
    if os.path.exists(ck_best):
        best_model = build_model(num_classes=n_classes, weights="none").to(device)
        try:
            load_model_only(ck_best, best_model)
        except RuntimeError:
            # best.pt is from an older run with a different class count
            print(f"[train] skipping final test eval: {ck_best} was trained on a "
                  f"different dataset (class counts differ).")
        else:
            test_metrics = evaluate(best_model, test_loader, device, criterion, n_classes)
            print(f"\n[test] best model on held-out test set:")
            print(f"  accuracy = {test_metrics['accuracy']:.4f}")
            print(f"  macro-F1 = {test_metrics['macro_f1']:.4f}")
            print(f"  loss     = {test_metrics['loss']:.4f}")

    print(f"\nbest val accuracy: {best_acc:.4f}")
    print(f"resumable checkpoint -> {ck_latest}")
    print(f"best model (for export) -> {ck_best}")


if __name__ == "__main__":
    main()
