"""Filesystem locations + env overrides."""
from __future__ import annotations
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def _out_dir() -> Path:
    env = os.environ.get("VOICE_OUTPUT_DIR") or os.environ.get("VOICE_ENGINE_OUT")
    if env:
        return Path(env).expanduser().resolve()
    return (PACKAGE_ROOT / "out").resolve()


DEFAULT_OUT_DIR = _out_dir()

QWEN_MODEL = os.environ.get(
    "VOICE_QWEN_MODEL", "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
)
QWEN_CLONE_MODEL = os.environ.get(
    "VOICE_QWEN_CLONE_MODEL", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
)
OMNI_MODEL = os.environ.get("VOICE_OMNI_MODEL", "mlx-community/OmniVoice-8bit")
