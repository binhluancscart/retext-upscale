#!/usr/bin/env python3
"""Heavy engine: runs PyTorch models (.safetensors) through spandrel on the Apple GPU (MPS).

Used by upscale.py when the chosen model lives in models_pt/. Runs with the .venv python:
    .venv/bin/python engine_pt.py -i in.png -o out.png -m models_pt/name.safetensors
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from spandrel import ImageModelDescriptor, ModelLoader  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# (number of 90-degree rotations, horizontal flip) - the first 4 are flips only, no transpose
TTA_MODES = [(0, False), (0, True), (2, False), (2, True), (1, False), (1, True), (3, False), (3, True)]


def infer(model, x: torch.Tensor, tta: int) -> torch.Tensor:
    """Run the model; with TTA, average 2/4/8 flipped/rotated variants (fewer small artefacts)."""
    if not tta or tta < 2:
        return model(x).float()
    total = None
    for k, flip in TTA_MODES[:tta]:
        t = torch.rot90(x, k, (2, 3))
        if flip:
            t = torch.flip(t, (3,))
        y = model(t).float()
        if flip:
            y = torch.flip(y, (3,))
        y = torch.rot90(y, -k, (2, 3))
        total = y if total is None else total + y
    return total / tta


@torch.inference_mode()
def upscale(img: Image.Image, model, device, tile: int, pad: int, progress: bool,
            tta: int = 0, half: bool = False) -> Image.Image:
    scale = model.scale
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    src = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    out = torch.zeros((3, h * scale, w * scale), dtype=torch.float32)
    tiles = [(y, x) for y in range(0, h, tile) for x in range(0, w, tile)]
    t0 = time.time()
    for n, (y, x) in enumerate(tiles, 1):
        y1, x1 = min(y + tile, h), min(x + tile, w)
        # take extra padding around the tile so seams match, then crop it from the result
        py0, px0 = max(y - pad, 0), max(x - pad, 0)
        py1, px1 = min(y1 + pad, h), min(x1 + pad, w)
        part = src[:, :, py0:py1, px0:px1].to(device)
        res = infer(model, part.half() if half else part, tta).clamp_(0, 1).cpu()[0]
        oy, ox = (y - py0) * scale, (x - px0) * scale
        out[:, y * scale : y1 * scale, x * scale : x1 * scale] = res[
            :, oy : oy + (y1 - y) * scale, ox : ox + (x1 - x) * scale
        ]
        if progress:
            eta = (time.time() - t0) / n * (len(tiles) - n)
            print(f"\r  tile {n}/{len(tiles)}, ~{eta:.0f}s left   ", end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)
    rgb = Image.fromarray((out.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8))
    if "A" in img.getbands():
        rgb.putalpha(img.getchannel("A").resize(rgb.size, Image.LANCZOS))
    return rgb


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a PyTorch upscaling model (spandrel).")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-m", "--model", required=True, help="path to a .safetensors file")
    ap.add_argument("--tile", type=int, default=192, help="tile size (source pixels)")
    ap.add_argument("--pad", type=int, default=16, help="overlap padding between tiles")
    ap.add_argument("--device", default="auto", help="auto | mps | cpu | cuda")
    ap.add_argument("--tta", type=int, default=0, choices=[0, 2, 4, 8],
                    help="average N flipped/rotated runs: better, N times slower")
    ap.add_argument("--half", action="store_true", help="use fp16 (faster if the model supports it)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.model.endswith(".safetensors"):
        sys.exit("Only .safetensors models are accepted (.pth files can execute code when loaded).")
    device = pick_device(args.device)
    model = ModelLoader().load_from_file(args.model)
    if not isinstance(model, ImageModelDescriptor):
        sys.exit("Not an image-to-image model.")
    model.to(device).eval()
    half = args.half and model.supports_half and device.type != "cpu"
    if half:
        model.model.half()
    with Image.open(args.input) as im:
        im.load()
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
    result = upscale(im, model, device, args.tile, args.pad,
                     not args.quiet and sys.stderr.isatty(), args.tta, half)
    result.save(args.output, compress_level=1)  # intermediate file: favour write speed


if __name__ == "__main__":
    main()
