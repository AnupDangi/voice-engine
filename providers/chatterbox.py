"""Chatterbox Turbo provider (torch/MPS). Optional expressive English engine."""
from __future__ import annotations
import time

from ..types import ProviderRequest, ProviderResult

_MODEL = None


def _device() -> str:
    try:
        import torch

        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _load():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
    except ImportError as e:
        raise RuntimeError(
            "Chatterbox not installed. Run ./voice_engine/setup.sh --all "
            "to install torch + chatterbox."
        ) from e
    _MODEL = ChatterboxTurboTTS.from_pretrained(device=_device())
    return _MODEL


def synthesize_chatterbox(req: ProviderRequest, model=None) -> ProviderResult:
    model = model or _load()
    t0 = time.time()
    kwargs: dict = {}
    if req.ref_audio:
        kwargs["audio_prompt_path"] = req.ref_audio
    wav = model.generate(req.text, **kwargs)
    gen_ms = (time.time() - t0) * 1000.0

    import soundfile as sf

    sr = int(getattr(model, "sr", 24000) or 24000)
    try:
        data = wav.detach().cpu().numpy().squeeze()
    except Exception:
        import numpy as np

        data = np.array(wav).squeeze()
    sf.write(req.audio_path, data, sr)
    dur_ms = float(data.shape[0] / float(sr) * 1000.0)
    return ProviderResult(
        audio_path=req.audio_path,
        voice=req.voice or "default",
        generation_ms=gen_ms,
        audio_duration_ms=dur_ms,
    )
