"""Shared dataclasses."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

ProviderName = Literal["qwen", "omni", "chatterbox"]
ProviderSelector = Literal["auto", "qwen", "omni", "chatterbox"]

TaskName = Literal["tutor", "narration", "expressive", "clone"]


@dataclass(frozen=True)
class RouteDecision:
    provider: str  # ProviderName
    language: str
    reason: str = ""


@dataclass(frozen=True)
class SynthesizeInput:
    text: str
    language: str
    voice: str | None = None
    provider: str = "auto"
    task: str = "tutor"
    ref_audio: str | None = None
    ref_text: str | None = None
    stream: bool = False
    out_dir: str | None = None


@dataclass(frozen=True)
class SynthesizeResult:
    audio_path: str
    provider: str
    language: str
    voice: str
    generation_ms: int
    audio_duration_ms: int
    rtf: float


@dataclass(frozen=True)
class ProviderRequest:
    text: str
    language: str
    voice: str | None
    audio_path: str
    task: str = "tutor"
    ref_audio: str | None = None
    ref_text: str | None = None
    stream: bool = False
    instruct: str | None = None


@dataclass(frozen=True)
class ProviderResult:
    audio_path: str
    voice: str
    generation_ms: float
    audio_duration_ms: float
