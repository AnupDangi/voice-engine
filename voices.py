"""Voice resolution for Qwen / OmniVoice / Chatterbox."""
from __future__ import annotations
from .languages import normalize_language

# Qwen3-TTS 0.6B CustomVoice premium timbres (upstream names).
QWEN_VOICES: tuple[str, ...] = (
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
)
DEFAULT_QWEN_VOICE = "Ryan"

# Sensible default per Qwen language (native speaker where possible).
QWEN_DEFAULT_VOICES: dict[str, str] = {
    "zh": "Vivian", "en": "Ryan", "ja": "Ono_Anna", "ko": "Sohee",
    "de": "Ryan", "fr": "Ryan", "ru": "Ryan", "pt": "Ryan",
    "es": "Ryan", "it": "Ryan",
}

DEFAULT_OMNI_VOICE = "default"
DEFAULT_CHATTERBOX_VOICE = "default"


def resolve_qwen_voice(language: str, voice: str | None = None) -> str:
    if voice:
        # Accept case-insensitive match against known timbres.
        for known in QWEN_VOICES:
            if voice.lower() == known.lower():
                return known
        # Unknown name: pass through (Base-model cloning ignores it anyway).
        return voice
    code = normalize_language(language)
    return QWEN_DEFAULT_VOICES.get(code, DEFAULT_QWEN_VOICE)


def resolve_omni_voice(voice: str | None = None) -> str:
    return voice or DEFAULT_OMNI_VOICE


def resolve_chatterbox_voice(voice: str | None = None) -> str:
    return voice or DEFAULT_CHATTERBOX_VOICE
