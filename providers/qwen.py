"""Qwen3-TTS provider (MLX). CustomVoice default, Base when cloning."""
from __future__ import annotations
import time

from ..languages import qwen_lang_name
from ..types import ProviderRequest, ProviderResult


def _save_wav(audio, sample_rate: int, path: str) -> float:
    import numpy as np
    import soundfile as sf

    try:
        import mlx.core as mx

        mx.eval(audio)
    except Exception:
        pass
    arr = np.array(audio, dtype=np.float32).squeeze()
    sf.write(path, arr, int(sample_rate or 24000))
    return float(arr.shape[0] / float(int(sample_rate or 24000)) * 1000.0)


def synthesize_qwen(req: ProviderRequest, model=None) -> ProviderResult:
    if model is None:
        from mlx_audio.tts.utils import load_model

        from .. import paths as _paths

        model_id = _paths.QWEN_CLONE_MODEL if req.ref_audio else _paths.QWEN_MODEL
        model = load_model(model_id)

    sr = int(getattr(model, "sample_rate", 24000) or 24000)
    kwargs: dict = {
        "voice": req.voice or "Ryan",
        "lang_code": qwen_lang_name(req.language),
        "stream": bool(req.stream),
    }
    if req.instruct:
        kwargs["instruct"] = req.instruct
    if req.ref_audio:
        # Base-checkpoint ICL cloning. ref_text required for stability.
        if not req.ref_text:
            raise ValueError("Qwen cloning requires ref_text matching ref_audio")
        kwargs["ref_audio"] = req.ref_audio
        kwargs["ref_text"] = req.ref_text

    t0 = time.time()
    chunks = []
    for result in model.generate(req.text, **kwargs):
        chunks.append(result.audio)
    gen_ms = (time.time() - t0) * 1000.0
    if not chunks:
        raise RuntimeError("Qwen3-TTS produced no audio")

    if len(chunks) == 1:
        audio = chunks[0]
    else:
        import mlx.core as mx

        audio = mx.concatenate(chunks, axis=0)

    dur_ms = _save_wav(audio, sr, req.audio_path)
    return ProviderResult(
        audio_path=req.audio_path,
        voice=req.voice or "Ryan",
        generation_ms=gen_ms,
        audio_duration_ms=dur_ms,
    )
