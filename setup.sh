#!/bin/bash
# Downloads the upscayl-bin engine (macOS) and AI models from the official Upscayl repos.
# Run once:  ./setup.sh           (default models)
#            ./setup.sh --all     (all ncnn models)
#            ./setup.sh --pro     (also the PyTorch engine and its models)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN_DIR="$ROOT/bin"
MODEL_DIR="$ROOT/models"
ENGINE_TAG="20251207-174704"
ENGINE_ZIP="upscayl-bin-${ENGINE_TAG}-macos.zip"
ENGINE_URL="https://github.com/upscayl/upscayl-ncnn/releases/download/${ENGINE_TAG}/${ENGINE_ZIP}"
MODEL_URL="https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models"

MODELS=(upscayl-standard-4x remacri-4x ultrasharp-4x digital-art-4x upscayl-lite-4x)
if [[ " $* " == *" --all "* ]]; then
  MODELS+=(high-fidelity-4x ultramix-balanced-4x)
fi

mkdir -p "$BIN_DIR" "$MODEL_DIR"

if [[ ! -x "$BIN_DIR/upscayl-bin" ]]; then
  echo "==> Downloading engine: $ENGINE_ZIP"
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  curl -fL --progress-bar -o "$TMP/engine.zip" "$ENGINE_URL"
  unzip -q "$TMP/engine.zip" -d "$TMP/unzipped"
  FOUND="$(find "$TMP/unzipped" -type f -name 'upscayl-bin' | head -1)"
  if [[ -z "$FOUND" ]]; then
    echo "upscayl-bin not found in the zip file" >&2
    exit 1
  fi
  cp "$FOUND" "$BIN_DIR/upscayl-bin"
  chmod +x "$BIN_DIR/upscayl-bin"
else
  echo "==> Engine already installed: $BIN_DIR/upscayl-bin"
fi

for m in "${MODELS[@]}"; do
  for ext in param bin; do
    f="$MODEL_DIR/$m.$ext"
    if [[ -s "$f" ]]; then
      continue
    fi
    echo "==> Downloading model: $m.$ext"
    curl -fL --progress-bar -o "$f.part" "$MODEL_URL/$m.$ext"
    mv "$f.part" "$f"
  done
done

# Optional heavy engine:  ./setup.sh --pro
# Installs PyTorch + spandrel into .venv and downloads .safetensors models (better small text).
# Licences: UltraSharpV2 (pro, pro-max) is CC-BY-NC-SA 4.0 = NON-commercial only;
#           RealWebPhoto (webphoto) is CC-BY-4.0 = commercial use allowed.
if [[ " $* " == *" --pro "* ]]; then
  PT_DIR="$ROOT/models_pt"
  mkdir -p "$PT_DIR"
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    echo "==> Creating .venv and installing PyTorch + spandrel"
    python3 -m venv "$ROOT/.venv"
    "$ROOT/.venv/bin/pip" install --quiet --upgrade pip
    # torch 2.5.1 is the last release whose GPU (MPS) backend works on macOS 13; newer ones need macOS 14+
    "$ROOT/.venv/bin/pip" install --quiet "torch==2.5.1" "torchvision==0.20.1" spandrel pillow
  fi
  HF="https://huggingface.co/Kim2091/UltraSharpV2/resolve/main"
  GH="https://github.com/Phhofm/models/releases/download/4xRealWebPhoto_v4_dat2"
  for url in "$HF/4x-UltraSharpV2_Lite.safetensors" "$HF/4x-UltraSharpV2.safetensors" "$GH/4xRealWebPhoto_v4_dat2.safetensors"; do
    f="$PT_DIR/$(basename "$url")"
    if [[ ! -s "$f" ]]; then
      echo "==> Downloading model: $(basename "$url")"
      curl -fL --progress-bar -o "$f.part" "$url"
      mv "$f.part" "$f"
    fi
  done
fi

echo
echo "Done. Installed models:"
ls -1 "$MODEL_DIR" | sed -n 's/\.param$//p' | sed 's/^/  - /'
if [[ -d "$ROOT/models_pt" ]]; then
  ls -1 "$ROOT/models_pt" | sed -n 's/\.safetensors$//p' | sed 's/^/  - /'
fi
