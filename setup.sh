#!/usr/bin/env bash
# Local voice-engine setup: MLX-only by default, torch/chatterbox with --all.
# Idempotent. Run from repo root or voice_engine dir.
set -euo pipefail

WITH_ALL=0
for a in "$@"; do
  case "$a" in
    --all) WITH_ALL=1 ;;
    -h|--help) echo "usage: setup.sh [--all]"; exit 0 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# voice_engine/setup.sh lives one level below repo root when checked out here.
if [ -f "$SCRIPT_DIR/../pyproject.toml" ]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  REPO_ROOT="$SCRIPT_DIR"
fi
VENV="$REPO_ROOT/.venv"

command -v uv >/dev/null 2>&1 || { echo "error: uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }

if [ ! -x "$VENV/bin/python" ]; then
  echo "Creating venv at $VENV ..."
  uv venv "$VENV" --python 3.12
fi

echo "Installing MLX voice stack (mlx-audio, numpy, soundfile) ..."
uv pip install --python "$VENV/bin/python" "mlx-audio" "numpy" "soundfile"

if [ "$WITH_ALL" -eq 1 ]; then
  echo "Installing optional Chatterbox stack (torch + chatterbox-tts) ..."
  uv pip install --python "$VENV/bin/python" "torch" "torchaudio" "chatterbox-tts"
else
  echo "Skipping torch/chatterbox (pass --all to include it)."
fi

# Console entry point for the current checkout.
"$VENV/bin/python" - <<'PY'
import sys
from pathlib import Path
print("venv python:", sys.executable)
PY

echo ""
echo "Done. Activate with: source $VENV/bin/activate"
echo "Try: voice-engine --text 'Hello.' --language en   (after: pip install -e .)"
echo "Or:  python -m voice_engine.compare_en"
