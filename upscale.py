#!/usr/bin/env python3
"""Sharpen and enlarge images with AI upscalers (upscayl-bin / Real-ESRGAN on Metal, or PyTorch).

Examples:
    ./upscale photo.jpg 4k
    ./upscale photo.jpg 8k -m ultrasharp
    ./upscale folder/ 2k -o results/
    ./upscale photo.png --scale 2
    ./upscale mockup.webp 4k -m pro --retext
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "bin" / "upscayl-bin"
MODEL_DIR = ROOT / "models"
MODEL_SCALE = 4  # every bundled model is 4x
# Optional heavy engine (./setup.sh --pro): PyTorch models run through engine_pt.py
PT_MODEL_DIR = ROOT / "models_pt"
PT_PYTHON = ROOT / ".venv" / "bin" / "python"
PT_ENGINE = ROOT / "engine_pt.py"

# Presets are 16:9 frames; the image is scaled to cover the frame, keeping its aspect ratio.
PRESETS = {
    "1080": (1920, 1080),
    "1080p": (1920, 1080),
    "fhd": (1920, 1080),
    "2k": (2560, 1440),
    "1440p": (2560, 1440),
    "qhd": (2560, 1440),
    "4k": (3840, 2160),
    "2160p": (3840, 2160),
    "uhd": (3840, 2160),
    "8k": (7680, 4320),
    "4320p": (7680, 4320),
}

MODEL_ALIASES = {
    "standard": "upscayl-standard-4x",
    "ultrasharp": "ultrasharp-4x",
    "sharp": "ultrasharp-4x",
    "fidelity": "high-fidelity-4x",
    "high-fidelity": "high-fidelity-4x",
    "art": "digital-art-4x",
    "anime": "digital-art-4x",
    "lite": "upscayl-lite-4x",
    "fast": "upscayl-lite-4x",
    "remacri": "remacri-4x",
    "design": "remacri-4x",
    "web": "remacri-4x",
    "text": "remacri-4x",
    "ultramix": "ultramix-balanced-4x",
    # heavy engine (PyTorch)
    "pro": "4x-UltraSharpV2_Lite",
    "pro-max": "4x-UltraSharpV2",
    "webphoto": "4xRealWebPhoto_v4_dat2",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
Image.MAX_IMAGE_PIXELS = None  # 8K+ images exceed Pillow's default safety limit


def die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def pt_models() -> list[str]:
    if not (PT_MODEL_DIR.is_dir() and PT_PYTHON.exists()):
        return []
    return sorted(p.stem for p in PT_MODEL_DIR.glob("*.safetensors"))


def installed_models() -> list[str]:
    ncnn = []
    if MODEL_DIR.is_dir():
        ncnn = sorted(
            p.stem for p in MODEL_DIR.glob("*.param") if p.with_suffix(".bin").exists()
        )
    return ncnn + pt_models()


def resolve_model(name: str) -> str:
    model = MODEL_ALIASES.get(name.lower(), name)
    if model not in installed_models():
        have = ", ".join(installed_models()) or "(none installed, run ./setup.sh)"
        die(f"model '{name}' is not installed. Available: {have}")
    return model


def target_size(src: tuple[int, int], args: argparse.Namespace) -> tuple[int, int]:
    """Output size (aspect ratio is always kept)."""
    w, h = src
    if args.scale:
        f = args.scale
    elif args.width or args.height:
        fw = args.width / w if args.width else math.inf
        fh = args.height / h if args.height else math.inf
        f = min(fw, fh)
    else:
        box_w, box_h = PRESETS[args.preset]
        if h > w:  # portrait image: rotate the frame to match
            box_w, box_h = box_h, box_w
        f = min(box_w / w, box_h / h) if args.fit else max(box_w / w, box_h / h)
    return max(1, round(w * f)), max(1, round(h * f))


def run_engine(src: Path, dst: Path, model: str, args: argparse.Namespace) -> None:
    if model in pt_models():
        cmd = [
            str(PT_PYTHON), str(PT_ENGINE),
            "-i", str(src),
            "-o", str(dst),
            "-m", str(PT_MODEL_DIR / f"{model}.safetensors"),
            "--tile", str(args.tile or 192),
            "--tta", str(args.tta),
        ]
        # stderr goes straight to the terminal: the engine prints per-tile progress
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        if proc.returncode != 0 or not dst.exists():
            die(f"PyTorch engine failed on {src.name}\n{(proc.stdout or '')[-1500:]}")
        return
    if not ENGINE.exists():
        die("engine not installed. Run ./setup.sh first.")
    cmd = [
        str(ENGINE),
        "-i", str(src),
        "-o", str(dst),
        "-m", str(MODEL_DIR),
        "-n", model,
        "-z", str(MODEL_SCALE),
        "-s", str(MODEL_SCALE),
        "-f", "png",
        "-t", str(args.tile),
    ]
    if args.gpu is not None:
        cmd += ["-g", str(args.gpu)]
    if args.tta:
        cmd.append("-x")
    proc = subprocess.run(cmd, capture_output=not args.verbose, text=True)
    if proc.returncode != 0 or not dst.exists():
        detail = "" if args.verbose else (proc.stderr or proc.stdout or "")[-1500:]
        die(f"engine failed on {src.name}\n{detail}")


def color_fix(result: Image.Image, original: Image.Image) -> Image.Image:
    """Pull the overall colour of the AI result back to the source (models shift tones slightly).

    Only low-frequency colour (large areas) is corrected; sharpness is untouched.
    """
    w, h = original.size
    radius = max(3.0, min(w, h) / 100)
    blur = ImageFilter.GaussianBlur(radius)
    rgb = result.convert("RGB")
    want = original.convert("RGB").filter(blur)
    got = rgb.resize((w, h), Image.BOX).filter(blur)
    up = ImageChops.subtract(want, got).resize(rgb.size, Image.BICUBIC)
    down = ImageChops.subtract(got, want).resize(rgb.size, Image.BICUBIC)
    fixed = ImageChops.subtract(ImageChops.add(rgb, up), down)
    if result.mode == "RGBA":
        fixed.putalpha(result.getchannel("A"))
    return fixed


def save(img: Image.Image, dst: Path, quality: int, icc: bytes | None) -> None:
    ext = dst.suffix.lower()
    extra = {"icc_profile": icc} if icc else {}
    if ext in (".jpg", ".jpeg"):
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(dst, quality=quality, subsampling=0, optimize=True, **extra)
    elif ext == ".webp":
        img.save(dst, quality=quality, method=6, **extra)
    else:
        # compress_level=6 is about 10x faster than optimize=True for a ~5% larger file
        img.save(dst, compress_level=6, **extra)


def process(src: Path, dst: Path, model: str, args: argparse.Namespace) -> None:
    t0 = time.time()
    with Image.open(src) as im:
        icc = im.info.get("icc_profile")
        im = ImageOps.exif_transpose(im)  # apply EXIF orientation
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA" if "A" in im.getbands() or "transparency" in im.info else "RGB")
        src_size = im.size
        tw, th = target_size(src_size, args)
        factor = tw / src_size[0]
        passes = 1 if factor <= MODEL_SCALE else 2
        if factor > MODEL_SCALE**2:
            die(
                f"{src.name}: needs {factor:.1f}x, above the {MODEL_SCALE**2}x limit. "
                "Pick a lower resolution."
            )
        print(
            f"{src.name}: {src_size[0]}x{src_size[1]} -> {tw}x{th} "
            f"({factor:.2f}x, {passes} AI pass{'es' if passes > 1 else ''}, model {model})"
        )

        with tempfile.TemporaryDirectory(prefix="upscale_") as tmp:
            tmp = Path(tmp)
            stage = tmp / "in.png"
            im.save(stage)
            for i in range(passes):
                out = tmp / f"pass{i + 1}.png"
                run_engine(stage, out, model, args)
                if i + 1 < passes:
                    # Resize pass 1 to exactly 1/4 of the target so pass 2 lands on the target size.
                    mid = (math.ceil(tw / MODEL_SCALE), math.ceil(th / MODEL_SCALE))
                    with Image.open(out) as p:
                        p.resize(mid, Image.LANCZOS).save(tmp / "mid.png")
                    stage = tmp / "mid.png"
            with Image.open(out) as result:
                result.load()
            if result.size != (tw, th):
                result = result.resize((tw, th), Image.LANCZOS)
            if not args.no_color_fix:
                result = color_fix(result, im)
            if args.blend > 0:
                # Mix in a plain (non-AI) resize to soften distorted small text / invented detail.
                plain = im.resize((tw, th), Image.LANCZOS).convert(result.mode)
                result = Image.blend(result, plain, args.blend)
            if args.sharpen > 0:
                result = result.filter(
                    ImageFilter.UnsharpMask(radius=1.2, percent=int(args.sharpen * 100), threshold=2)
                )
            dst.parent.mkdir(parents=True, exist_ok=True)
            save(result, dst, args.quality, icc)

    if args.retext:
        import retext  # imported on demand (needs numpy + macOS OCR)

        layer = dst.with_suffix(".text.json")
        info = retext.run(src, dst, dst, layer, args.retext_max, {"sans": args.sans, "serif": args.serif})
        print(f"  re-rendered {info['used']}/{info['total']} text lines; editable text layer: {layer}")
        check = [ln["text"] for ln in info["lines"] if not ln.get("use") and "needs review" in ln.get("reason", "")]
        if check:
            print(f"  {len(check)} lines need review and were left as AI pixels. Fix \"text\" and add "
                  "\"checked\": true in the text layer, then run again.")

    mb = dst.stat().st_size / 1_048_576
    print(f"  -> {dst}  ({mb:.1f} MB, {time.time() - t0:.0f}s)")


def out_path(src: Path, args: argparse.Namespace, label: str, batch: bool) -> Path:
    ext = f".{args.format}" if args.format else src.suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".png"
    name = f"{src.stem}_{label}{ext}"
    if args.output:
        o = Path(args.output).expanduser()
        if batch or o.is_dir() or not o.suffix:
            return o / name
        return o
    return src.with_name(name)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="upscale",
        description="Sharpen and enlarge images with AI to 1080p / 2K / 4K / 8K.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Preset: 1080 (1920x1080), 2k (2560x1440), 4k (3840x2160), 8k (7680x4320)\n"
            "Models: standard (default, photos), ultrasharp (strongest sharpening), fidelity (keeps real detail),\n"
            "        design (designs/web pages with text), art (illustration/anime), lite (fast)\n"
            "Heavy engine (needs ./setup.sh --pro): pro (best small text), pro-max, webphoto\n"
        ),
    )
    ap.add_argument("input", nargs="?", help="image file or folder of images")
    ap.add_argument("preset", nargs="?", default="4k", help="1080 | 2k | 4k | 8k (default: 4k)")
    ap.add_argument("-o", "--output", help="output file or folder (default: next to the input)")
    ap.add_argument("-m", "--model", default="standard", help="model name or alias (see --list-models)")
    ap.add_argument("-s", "--scale", type=float, help="scale by a factor, e.g. 2, 3, 4 (ignores the preset)")
    ap.add_argument("--width", type=int, help="output width in pixels")
    ap.add_argument("--height", type=int, help="output height in pixels")
    ap.add_argument("--fit", action="store_true",
                    help="fit INSIDE the preset frame (default: cover the frame)")
    ap.add_argument("-f", "--format", choices=["png", "jpg", "webp"], help="output format (default: same as input)")
    ap.add_argument("-q", "--quality", type=int, default=95, help="JPG/WebP quality (default 95)")
    ap.add_argument("--sharpen", type=float, default=0.0, help="extra sharpening after upscaling, 0-2 (default 0)")
    ap.add_argument("--blend", type=float, default=0.0,
                    help="mix a plain resize into the AI result, 0-1 (e.g. 0.4): fewer artefacts, less sharp")
    ap.add_argument("--retext", action="store_true",
                    help="re-render small text with real fonts (OCR + system fonts) for truly crisp text. "
                         "For designs/web mockups; macOS only. Writes an editable .text.json layer")
    ap.add_argument("--retext-max", type=float, default=15.0,
                    help="only re-render text up to N px in the source (default 15); larger text keeps AI pixels")
    ap.add_argument("--sans", help="sans-serif family for --retext, e.g. 'Avenir Next' (default: auto)")
    ap.add_argument("--serif", help="serif family for --retext, e.g. 'Baskerville' (default: auto)")
    ap.add_argument("--no-color-fix", action="store_true",
                    help="skip the colour correction back to the source (default: on)")
    ap.add_argument("--tile", type=int, default=0, help="tile size; lower it (e.g. 200) if the GPU runs out of memory")
    ap.add_argument("--tta", type=int, nargs="?", const=8, default=0, choices=[0, 2, 4, 8],
                    help="average N flipped/rotated runs: cleaner detail, N times slower "
                         "(--tta alone = 8; the heavy engine takes 2/4/8, ncnn only 8)")
    ap.add_argument("--gpu", type=int, help="GPU index (default: auto)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing output files")
    ap.add_argument("--list-models", action="store_true", help="list installed models")
    ap.add_argument("-v", "--verbose", action="store_true", help="show engine logs")
    args = ap.parse_args()

    if args.list_models:
        alias_of: dict[str, list[str]] = {}
        for a, m in MODEL_ALIASES.items():
            alias_of.setdefault(m, []).append(a)
        for m in installed_models():
            print(f"{m:28s} aliases: {', '.join(alias_of.get(m, [])) or '-'}")
        return

    if not args.input:
        ap.print_help()
        sys.exit(1)
    args.preset = args.preset.lower()
    custom = args.scale or args.width or args.height
    if not custom and args.preset not in PRESETS:
        die(f"unknown preset '{args.preset}'. Choose: 1080, 2k, 4k, 8k")
    if args.scale and not 1 <= args.scale <= MODEL_SCALE**2:
        die("--scale must be between 1 and 16")
    if not 0 <= args.blend <= 1:
        die("--blend must be between 0 and 1")
    model = resolve_model(args.model)

    if args.scale:
        label = f"x{args.scale:g}"
    elif args.width or args.height:
        label = f"{args.width or ''}x{args.height or ''}".strip("x") + "px"
    else:
        label = {"1080p": "1080", "fhd": "1080", "1440p": "2k", "qhd": "2k",
                 "2160p": "4k", "uhd": "4k", "4320p": "8k"}.get(args.preset, args.preset)

    src = Path(args.input).expanduser()
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            die(f"no images found in {src}")
        if not args.output:
            args.output = str(src / f"upscaled_{label}")
        batch = True
    elif src.is_file():
        files, batch = [src], False
    else:
        die(f"not found: {src}")

    done = 0
    for f in files:
        dst = out_path(f, args, label, batch)
        if dst.exists() and not args.overwrite:
            print(f"{f.name}: skipped, {dst} exists (use --overwrite)")
            continue
        process(f, dst, model, args)
        done += 1
    if batch:
        print(f"Done: {done}/{len(files)} images.")


if __name__ == "__main__":
    if shutil.which("python3") is None:
        die("python3 is required")
    main()
