"""OmniVoice provider (MLX): broad multilingual fallback + cloning."""
from __future__ import annotations
import time

from ..languages import omni_lang_name
from ..types import ProviderRequest, ProviderResult


def _estimate_duration_s(text: str) -> float:
    # ~800 chars/min floor 3s, cap 60s. Scene-by-scene calls stay short.
    return max(3.0, min(60.0, max(1.0, len(text) / 800.0 * 60.0)))


def _dequantize_rowwise_int8(weights: dict) -> dict:
    """mlx-community/OmniVoice-8bit stores weights as symmetric per-group
    int8 (shape [out, groups, group_size]) with a matching float16 *.scales
    tensor (shape [out, groups]) and no zero-point/bias. That's a bespoke
    scheme predating mlx_audio's own quantization support, and its on-disk
    layout (unpacked int8, no biases) doesn't match what nn.quantize() /
    QuantizedLinear expect (bit-packed uint32 + scales + biases), so
    mlx_audio's generic loader can't consume it even with a 'quantization'
    key in config.json — it fails with "N parameters not in model" because
    quantize() never runs on these layers. Dequantize to float32 ourselves
    so the model loads as a plain, unquantized network.
    """
    import mlx.core as mx

    result = dict(weights)
    for scale_key in [k for k in weights if k.endswith(".scales")]:
        base_key = scale_key[: -len(".scales")]
        if base_key not in result:
            continue
        q = result.pop(base_key)
        scale = result.pop(scale_key)
        out_dim, groups, group_size = q.shape
        deq = (q.astype(mx.float32) * scale.astype(mx.float32)[:, :, None]).reshape(
            out_dim, groups * group_size
        )
        result[base_key] = deq
    return result


def _load_omni_model(repo_or_path: str):
    """Load OmniVoice bypassing mlx_audio's generic quantization path (see
    _dequantize_rowwise_int8) since it can't consume this checkpoint's
    custom int8 format.
    """
    import mlx.core as mx
    from mlx_audio.tts.utils import MODEL_REMAPPING
    from mlx_audio.utils import get_model_class, get_model_path, load_config, load_weights

    model_path = get_model_path(repo_or_path)
    config = load_config(model_path)
    config["model_path"] = str(model_path)

    model_type = config.get("model_type") or config.get("architecture")
    model_class, _ = get_model_class(
        model_type=model_type,
        model_name=None,
        category="tts",
        model_remapping=MODEL_REMAPPING,
    )
    model_config = (
        model_class.ModelConfig.from_dict(config)
        if hasattr(model_class, "ModelConfig")
        else config
    )
    model = model_class.Model(model_config)

    weights = _dequantize_rowwise_int8(load_weights(model_path))
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()

    if hasattr(model_class.Model, "post_load_hook"):
        model = model_class.Model.post_load_hook(model, model_path)
    return model


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


def synthesize_omni(req: ProviderRequest, model=None) -> ProviderResult:
    if model is None:
        from .. import paths as _paths

        model = _load_omni_model(_paths.OMNI_MODEL)

    sr = int(getattr(model, "sample_rate", 24000) or 24000)
    kwargs: dict = {
        "language": omni_lang_name(req.language),
        "duration_s": _estimate_duration_s(req.text),
        "num_steps": 32,
    }
    if req.ref_audio:
        if not req.ref_text:
            raise ValueError("OmniVoice cloning requires ref_text matching ref_audio")
        kwargs["ref_audio"] = req.ref_audio
        kwargs["ref_text"] = req.ref_text

    t0 = time.time()
    results = list(model.generate(req.text, **kwargs))
    gen_ms = (time.time() - t0) * 1000.0
    if not results:
        raise RuntimeError("OmniVoice produced no audio")

    import mlx.core as mx

    audios = [r.audio for r in results]
    audio = audios[0] if len(audios) == 1 else mx.concatenate(audios, axis=0)
    dur_ms = _save_wav(audio, getattr(results[0], "sample_rate", sr), req.audio_path)
    return ProviderResult(
        audio_path=req.audio_path,
        voice=req.voice or "default",
        generation_ms=gen_ms,
        audio_duration_ms=dur_ms,
    )
