"""Local Qwen-first TTS: Qwen3-TTS default, OmniVoice fallback, Chatterbox optional."""
from .engine import active_provider, synthesize, unload
from .languages import (
    QWEN_LANGUAGES,
    is_expressive_text,
    is_omni_language,
    is_qwen_language,
    normalize_language,
)
from .paths import DEFAULT_OUT_DIR, OMNI_MODEL, PACKAGE_ROOT, QWEN_CLONE_MODEL, QWEN_MODEL
from .router import route
from .types import (
    ProviderRequest,
    ProviderResult,
    RouteDecision,
    SynthesizeInput,
    SynthesizeResult,
)
from .voices import (
    DEFAULT_QWEN_VOICE,
    QWEN_DEFAULT_VOICES,
    QWEN_VOICES,
    resolve_omni_voice,
    resolve_qwen_voice,
)

__all__ = [
    "synthesize", "unload", "active_provider", "route",
    "normalize_language", "is_qwen_language", "is_omni_language",
    "is_expressive_text", "resolve_qwen_voice", "resolve_omni_voice",
    "QWEN_LANGUAGES", "QWEN_VOICES", "QWEN_DEFAULT_VOICES", "DEFAULT_QWEN_VOICE",
    "DEFAULT_OUT_DIR", "PACKAGE_ROOT", "QWEN_MODEL", "QWEN_CLONE_MODEL", "OMNI_MODEL",
    "RouteDecision", "SynthesizeInput", "SynthesizeResult",
    "ProviderRequest", "ProviderResult",
]
