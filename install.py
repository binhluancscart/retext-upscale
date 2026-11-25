#!/usr/bin/env python3
"""Download the upscaling engine and models. Works on macOS, Linux and Windows.

    python3 install.py            default ncnn models
    python3 install.py --all      all ncnn models
    python3 install.py --pro      also the PyTorch engine (.venv) and its models

Nothing is bundled with the repository: the engine comes from the Upscayl releases
and the models from their authors' repositories. Only the standard library is used.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN_DIR = ROOT / "bin"
MODEL_DIR = ROOT / "models"
PT_DIR = ROOT / "models_pt"
VENV_DIR = ROOT / ".venv"

ENGINE_TAG = "20251207-174704"
ENGINE_URL = "https://github.com/upscayl/upscayl-ncnn/releases/download/{tag}/upscayl-bin-{tag}-{os}.zip"
MODEL_URL = "https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/{name}"

DEFAULT_MODELS = ["upscayl-standard-4x", "remacri-4x", "ultrasharp-4x", "digital-art-4x", "upscayl-lite-4x"]
EXTRA_MODELS = ["high-fidelity-4x", "ultramix-balanced-4x"]

# Licences: UltraSharpV2 is CC-BY-NC-SA 4.0 (non-commercial); RealWebPhoto is CC-BY-4.0.
PT_MODELS = [
    "https://huggingface.co/Kim2091/UltraSharpV2/resolve/main/4x-UltraSharpV2_Lite.safetensors",
    "https://huggingface.co/Kim2091/UltraSharpV2/resolve/main/4x-UltraSharpV2.safetensors",
    "https://github.com/Phhofm/models/releases/download/4xRealWebPhoto_v4_dat2/4xRealWebPhoto_v4_dat2.safetensors",
]

OS_NAME = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(platform.system())
EXE = ".exe" if OS_NAME == "windows" else ""


def download(url: str, dst: Path) -> None:
    """Download to dst via a .part file, with a simple progress line."""
    part = dst.with_name(dst.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "upscale-installer"})
    with urllib.request.urlopen(req) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total and sys.stderr.isatty():
                print(f"\r    {done / 1048576:6.1f} / {total / 1048576:.1f} MB", end="", file=sys.stderr)
    if sys.stderr.isatty():
        print(file=sys.stderr)
    part.replace(dst)


def install_engine() -> None:
    target = BIN_DIR / f"upscayl-bin{EXE}"
    if target.exists():
        print(f"==> Engine already installed: {target}")
        return
    url = ENGINE_URL.format(tag=ENGINE_TAG, os=OS_NAME)
    print(f"==> Downloading engine: {url.rsplit('/', 1)[-1]}")
    BIN_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="upscale_install_") as tmp:
        zip_path = Path(tmp) / "engine.zip"
        download(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(Path(tmp) / "unzipped")
        found = next((p for p in (Path(tmp) / "unzipped").rglob(f"upscayl-bin{EXE}") if p.is_file()), None)
        if found is None:
            sys.exit("upscayl-bin was not found in the downloaded zip")
        # Copy everything that ships next to the binary (Windows builds come with DLLs).
        for item in found.parent.iterdir():
            if item.is_file():
                shutil.copy2(item, BIN_DIR / item.name)
    if not EXE:
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def install_models(names: list[str]) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    for name in names:
        for ext in ("param", "bin"):
            dst = MODEL_DIR / f"{name}.{ext}"
            if dst.exists() and dst.stat().st_size > 0:
                continue
            print(f"==> Downloading model: {dst.name}")
            download(MODEL_URL.format(name=dst.name), dst)


def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if OS_NAME == "windows" else "bin/python")


def torch_requirements() -> list[str]:
    """PyTorch packages for this machine."""
    if OS_NAME == "macos":
        major = int((platform.mac_ver()[0] or "0").split(".")[0])
        if major < 14:
            # torch 2.5.1 is the last release whose GPU (MPS) backend works on macOS 13
            return ["torch==2.5.1", "torchvision==0.20.1"]
    return ["torch", "torchvision"]


def install_pro() -> None:
    py = venv_python()
    if not py.exists():
        print("==> Creating .venv and installing PyTorch + spandrel (this takes a few minutes)")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
        subprocess.run([str(py), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=True)
        subprocess.run([str(py), "-m", "pip", "install", "--quiet", *torch_requirements(), "spandrel", "pillow"], check=True)
        if OS_NAME == "windows":
            print("    Note: the default Windows build of PyTorch is CPU-only. For an NVIDIA GPU, reinstall")
            print("    torch inside .venv with the CUDA command from https://pytorch.org/get-started/locally/")
    PT_DIR.mkdir(exist_ok=True)
    for url in PT_MODELS:
        dst = PT_DIR / url.rsplit("/", 1)[-1]
        if dst.exists() and dst.stat().st_size > 0:
            continue
        print(f"==> Downloading model: {dst.name}")
        download(url, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download the upscaling engine and models.")
    ap.add_argument("--all", action="store_true", help="also download the extra ncnn models")
    ap.add_argument("--pro", action="store_true", help="also install the PyTorch engine and its models")
    ap.add_argument("--models", help="comma-separated ncnn model names to download instead of the defaults")
    args = ap.parse_args()
    if OS_NAME is None:
        sys.exit(f"Unsupported operating system: {platform.system()}")

    install_engine()
    if args.models:
        names = [n.strip() for n in args.models.split(",") if n.strip()]
    else:
        names = DEFAULT_MODELS + (EXTRA_MODELS if args.all else [])
    install_models(names)
    if args.pro:
        install_pro()

    print("\nDone. Installed models:")
    for p in sorted(MODEL_DIR.glob("*.param")):
        print(f"  - {p.stem}")
    if PT_DIR.is_dir() and venv_python().exists():
        for p in sorted(PT_DIR.glob("*.safetensors")):
            print(f"  - {p.stem}")
    if OS_NAME == "linux":
        print("\nLinux: the ncnn engine needs a Vulkan driver (e.g. the libvulkan1 package plus your GPU driver).")


if __name__ == "__main__":
    main()
