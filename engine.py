"""Public entry point: text + language -> provider -> local TTS.

Memory strategy: only one provider resident at a time. Switching providers
unloads the previous checkpoint and clears the MLX cache. Qwen additionally
switches between CustomVoice (default) and Base (cloning) checkpoints.
"""
from __future__ import annotations
import hashlib
import time
from pathlib import Path

from . import paths as _paths
from .router import route
from .types import ProviderRequest, SynthesizeResult
from .voices import (
    resolve_chatterbox_voice,
    resolve_omni_voice,
    resolve_qwen_voice,
)

_ACTIVE_NAME: str | None = None
_ACTIVE_MODEL = None


def _clear_mlx_cache() -> None:
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def _unload_active() -> None:
    global _ACTIVE_NAME, _ACTIVE_MODEL
    _ACTIVE_MODEL = None
    _ACTIVE_NAME = None
    _clear_mlx_cache()
    try:
        import gc

        gc.collect()
    except Exception:
        pass


def _ensure_provider(provider: str, model_id: str | None = None):
    """Return resident model, loading it (and unloading the other) if needed."""
    global _ACTIVE_NAME, _ACTIVE_MODEL
    key = provider if provider != "qwen" else f"qwen:{model_id}"
    if _ACTIVE_NAME == key and _ACTIVE_MODEL is not None:
        return _ACTIVE_MODEL
    if _ACTIVE_NAME is not None:
        _unload_active()

    if provider == "qwen":
        from mlx_audio.tts.utils import load_model

        _ACTIVE_MODEL = load_model(model_id or _paths.QWEN_MODEL)
    elif provider == "omni":
        from .providers.omnivoice import _load_omni_model

        _ACTIVE_MODEL = _load_omni_model(model_id or _paths.OMNI_MODEL)
    else:
        raise ValueError(f"No MLX model for provider {provider}")
    _ACTIVE_NAME = key
    return _ACTIVE_MODEL


def active_provider() -> str | None:
    base = (_ACTIVE_NAME or "").split(":")[0] or None
    return base


def unload() -> None:
    _unload_active()


def _default_voice(provider: str, language: str, voice: str | None) -> str:
    if provider == "qwen":
        return resolve_qwen_voice(language, voice)
    if provider == "omni":
        return resolve_omni_voice(voice)
    return resolve_chatterbox_voice(voice)


def synthesize(
    text: str,
    language: str,
    voice: str | None = None,
    provider: str = "auto",
    task: str = "tutor",
    ref_audio: str | None = None,
    ref_text: str | None = None,
    stream: bool = False,
    instruct: str | None = None,
    out_dir: str | Path | None = None,
) -> SynthesizeResult:
    if not text or not text.strip():
        raise ValueError("text is required")
    decision = route(language, provider, text=text, task=task)
    out = Path(out_dir).expanduser() if out_dir else _paths.DEFAULT_OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    stamp = format(int(time.time() * 1000), "x")[-8:]
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    audio_path = str(out / f"{decision.provider}-{decision.language}-{stamp}-{digest}.wav")
    req = ProviderRequest(
        text=text,
        language=decision.language,
        voice=_default_voice(decision.provider, decision.language, voice),
        audio_path=audio_path,
        task=task,
        ref_audio=ref_audio,
        ref_text=ref_text,
        stream=stream,
        instruct=instruct,
    )
    if decision.provider == "qwen":
        from .providers.qwen import synthesize_qwen

        # Cloning needs the Base checkpoint; default is CustomVoice.
        model_id = _paths.QWEN_CLONE_MODEL if ref_audio else _paths.QWEN_MODEL
        model = _ensure_provider("qwen", model_id)
        result = synthesize_qwen(req, model=model)
    elif decision.provider == "omni":
        from .providers.omnivoice import synthesize_omni

        model = _ensure_provider("omni")
        result = synthesize_omni(req, model=model)
    else:
        from .providers.chatterbox import synthesize_chatterbox

        # Chatterbox is torch/MPS, not MLX: free MLX memory first.
        if _ACTIVE_NAME is not None:
            _unload_active()
        result = synthesize_chatterbox(req)

    gen, dur = round(result.generation_ms), round(result.audio_duration_ms)
    rtf = round(result.generation_ms / result.audio_duration_ms, 3) if result.audio_duration_ms > 0 else 0.0
    return SynthesizeResult(
        audio_path=result.audio_path,
        provider=decision.provider,
        language=decision.language,
        voice=result.voice,
        generation_ms=gen,
        audio_duration_ms=dur,
        rtf=rtf,
    )
