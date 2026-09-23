"""Provider router: explicit > expressive-EN chatterbox > qwen > omni."""
from __future__ import annotations
from .languages import is_expressive_text, is_qwen_language, normalize_language
from .types import RouteDecision

_VALID = frozenset({"auto", "qwen", "omni", "chatterbox"})


def route(
    language: str,
    provider: str = "auto",
    text: str = "",
    task: str = "tutor",
) -> RouteDecision:
    code = normalize_language(language)
    sel = (provider or "auto").strip().lower()
    if sel not in _VALID:
        raise ValueError(f"Unknown provider {provider!r}; use auto|qwen|omni|chatterbox")
    if sel != "auto":
        return RouteDecision(provider=sel, language=code, reason="explicit")

    # Expressive English with paralinguistic tags -> Chatterbox Turbo.
    if code == "en" and (task or "").strip().lower() == "expressive":
        return RouteDecision(provider="chatterbox", language=code, reason="task=expressive")
    if code == "en" and is_expressive_text(text or ""):
        return RouteDecision(provider="chatterbox", language=code, reason="expressive-tag")

    if is_qwen_language(code):
        return RouteDecision(provider="qwen", language=code, reason="qwen-supported")
    return RouteDecision(provider="omni", language=code, reason="omni-fallback")
