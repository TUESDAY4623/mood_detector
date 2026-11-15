"""export_openvino.py — convert the trained PyTorch emotion model to an
OpenVINO IR (``.xml`` + ``.bin``) that the NPU / CPU / GPU can run.

Flow
----
1. Load the best PyTorch checkpoint (``checkpoints/best.pt``) into a fresh
   EfficientNet-B0.
2. Wrap it so the graph input is ``[1, 3, 224, 224]`` float32 (RGB,
   ImageNet-normalised) and the output is the 9-way emotion logits.
3. Convert with ``openvino.convert_model`` and save an IR model.
4. Save ``classes.json`` next to the IR so consumers know the label order.
5. Smoke-test: compile the IR on the requested device (default NPU, fall
   back to CPU if NPU rejects an op) and run one inference.

Usage
-----
    python export_openvino.py                       # from best.pt -> NPU IR
    python export_openvino.py --device CPU --fp16   # cpu + fp16 compression
    python export_openvino.py --checkpoint checkpoints/latest.pt

The resulting IR is what camera_ui.py and predict.py load.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train import build_model, CLASS_NAMES, IMG_SIZE, MEAN, STD
from checkpoint import load_model_only, BEST_NAME

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = PROJECT_DIR / "checkpoints" / BEST_NAME


class EmotionHead(nn.Module):
    """Minimal graph: raw float32 pixels -> 9 emotion logits.

    The normalisation is baked in so the consumer only has to resize +
    scale to ``[0,1]`` and reorder to CHW/RGB.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, 3, 224, 224] in [0,1] RGB (CHW). Normalise channel-wise.
        mean = torch.tensor(MEAN, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(STD, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.model(x)


def resolve_classes(checkpoint_path: Path) -> list[str]:
    """Class names for the checkpoint, checked against its own head size.

    Preference order: the ``config.class_names`` saved by train.py; else the
    historical 9-expression list when the head has 9 outputs; else a generic
    ``class_0..N-1`` so a bad name never silently mismatches the weights.
    """
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    n = None
    for key, v in (payload.get("model_state_dict") or {}).items():
        if key.endswith("classifier.1.weight") and hasattr(v, "shape") and len(v.shape) == 2:
            n = int(v.shape[0])
            break
    names = (payload.get("config") or {}).get("class_names")
    if names:
        if n is not None and len(names) != n:
            raise SystemExit(
                f"Checkpoint head has {n} outputs but config lists {len(names)} "
                f"classes: {checkpoint_path} is stale (retrain, then re-export).")
        return list(names)
    if n is not None and n == len(CLASS_NAMES):
        return list(CLASS_NAMES)
    if n is not None:
        print(f"[export] no class names recorded; using generic labels for {n} classes")
        return [f"class_{i}" for i in range(n)]
    return list(CLASS_NAMES)


def convert(checkpoint_path: Path, out_dir: Path, fp16: bool) -> tuple[Path, list[str]]:
    import openvino as ov

    classes = resolve_classes(checkpoint_path)
    print(f"Converting {checkpoint_path.name} -> OpenVINO IR "
          f"({len(classes)} classes: {classes}) ...")
    model = build_model(num_classes=len(classes), weights="none")
    load_model_only(str(checkpoint_path), model)
    model.eval()
    head = EmotionHead(model).eval()

    t0 = time.time()
    ov_model = ov.convert_model(
        head,
        input=[(1, 3, IMG_SIZE, IMG_SIZE)],
        example_input=torch.rand(1, 3, IMG_SIZE, IMG_SIZE),
    )
    if fp16:
        ov_model = ov.convert_model(
            head,
            input=[(1, 3, IMG_SIZE, IMG_SIZE)],
            example_input=torch.rand(1, 3, IMG_SIZE, IMG_SIZE),
            compress_to_fp16=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / "emotion_npu"
    ov.save_model(ov_model, xml_path.with_suffix(".xml"))
    print(f"Saved IR: {xml_path.with_suffix('.xml')} (+ .bin) in {time.time()-t0:.1f}s")

    # Sidecar so consumers know the label order.
    (out_dir / "classes.json").write_text(json.dumps(
        {"classes": classes, "img_size": IMG_SIZE,
         "mean": MEAN, "std": STD, "fp16": fp16}, indent=2))
    print(f"Wrote {out_dir/'classes.json'}")
    return xml_path.with_suffix(".xml"), classes


def smoke_test(ir_xml: Path, device: str, fp16: bool, classes: list[str]) -> None:
    import openvino as ov

    core = ov.Core()
    candidates = [device] if device.upper() in ("CPU", "GPU", "NPU") else [device, "CPU"]
    last_err = None
    for dev in candidates:
        try:
            compiled = core.compile_model(str(ir_xml), dev)
            inp = compiled.input(0)
            # A dummy image (random pixels) is all the smoke test needs.
            shape = tuple(int(d) for d in inp.get_partial_shape().to_shape())
            dummy = np.random.rand(*shape).astype("float32")
            out = compiled([dummy])[compiled.output(0)]
            probs = _softmax(out[0].astype("float32"))
            print(f"[smoke] device={dev} output shape={out.shape} "
                  f"top={classes[int(np.argmax(probs))]} (prob {max(probs):.3f})")
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[smoke] {dev} failed: {type(e).__name__}: {str(e)[:160]}")
            print(f"[smoke] falling back to next device")
    raise SystemExit(f"No device could run the IR. Last error: {last_err}")


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT),
                    help="Path to a PyTorch .pt produced by train.py")
    ap.add_argument("--out-dir", default=str(PROJECT_DIR / "model_ir"),
                    help="Directory to write the OpenVINO IR into")
    ap.add_argument("--device", default="NPU",
                    help="Device for the smoke test (NPU / CPU / GPU)")
    ap.add_argument("--fp16", action="store_true",
                    help="Emit a compressed FP16 IR (smaller, sometimes NPU-friendlier)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}\nRun train.py first, or pass --checkpoint.")

    ir, classes = convert(ckpt, Path(args.out_dir), args.fp16)
    smoke_test(ir, args.device, args.fp16, classes)
    print(f"\nDone. IR at {ir} ; {len(classes)} classes in {Path(args.out_dir)/'classes.json'}.")
    print(f"camera_ui.py and predict.py load this IR (try NPU, fall back to CPU).")


if __name__ == "__main__":
    main()
