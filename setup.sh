#!/bin/bash
# Kept for convenience on macOS/Linux; the real installer is install.py (also runs on Windows).
#   ./setup.sh           default models
#   ./setup.sh --all     all ncnn models
#   ./setup.sh --pro     also the PyTorch engine and its models
exec python3 "$(cd "$(dirname "$0")" && pwd)/install.py" "$@"
