"""Language normalization + provider capability tables.

Qwen3-TTS covers 10 languages. Everything else falls through to OmniVoice
(646+ languages), which is the broad-coverage fallback.
"""
from __future__ import annotations
import re

# Qwen3-TTS official 10 (Qwen/Qwen3-TTS).
QWEN_LANGUAGES: tuple[str, ...] = (
    "zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it",
)
_QWEN_SET = frozenset(QWEN_LANGUAGES)

# mlx-audio Qwen wants capitalized names ("English"); OmniVoice wants
# lowercase names ("english"). Both accept "auto".
QWEN_LANG_NAMES: dict[str, str] = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian",
}

OMNI_LANG_NAMES: dict[str, str] = {
    "en": "english", "zh": "chinese", "ja": "japanese", "ko": "korean",
    "de": "german", "fr": "french", "ru": "russian", "pt": "portuguese",
    "es": "spanish", "it": "italian", "hi": "hindi", "ne": "nepali",
    "bn": "bengali", "ta": "tamil", "te": "telugu", "ur": "urdu",
    "th": "thai", "vi": "vietnamese", "id": "indonesian", "ar": "arabic",
    "nl": "dutch", "pl": "polish", "tr": "turkish", "uk": "ukrainian",
    "fa": "persian", "sw": "swahili", "ml": "malayalam", "mr": "marathi",
}

_LANGUAGE_NAMES: dict[str, str] = {
    "english": "en", "chinese": "zh", "mandarin": "zh",
    "japanese": "ja", "korean": "ko", "german": "de", "french": "fr",
    "russian": "ru", "portuguese": "pt", "spanish": "es", "italian": "it",
    "hindi": "hi", "nepali": "ne", "bengali": "bn", "bangla": "bn",
    "tamil": "ta", "telugu": "te", "urdu": "ur", "thai": "th",
    "vietnamese": "vi", "indonesian": "id", "arabic": "ar", "dutch": "nl",
    "polish": "pl", "turkish": "tr", "ukrainian": "uk", "persian": "fa",
    "farsi": "fa", "swahili": "sw", "kiswahili": "sw",
    "malayalam": "ml", "marathi": "mr", "nepali": "ne",
}

EXPRESSIVE_TAG_RE = re.compile(
    r"\[(laugh|chuckle|cough|sigh|gasp|groan|clear throat|chuckles?|laughs?)\]",
    re.IGNORECASE,
)


def normalize_language(language: str | None) -> str:
    raw = str(language or "").strip().lower()
    if not raw:
        raise ValueError("language is required")
    dashed = raw.replace("_", "-")
    base = dashed.split("-")[0]
    return _LANGUAGE_NAMES.get(dashed) or _LANGUAGE_NAMES.get(base) or base


def is_qwen_language(language: str) -> bool:
    return normalize_language(language) in _QWEN_SET


def is_omni_language(language: str) -> bool:
    # OmniVoice covers 646+ languages: accept anything normalized.
    code = normalize_language(language)
    return bool(code)


def is_expressive_text(text: str) -> bool:
    return bool(EXPRESSIVE_TAG_RE.search(text or ""))


def qwen_lang_name(code: str) -> str:
    return QWEN_LANG_NAMES.get(normalize_language(code), "auto")


def omni_lang_name(code: str) -> str:
    c = normalize_language(code)
    return OMNI_LANG_NAMES.get(c, c)
