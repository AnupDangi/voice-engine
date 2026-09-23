"""voice-engine CLI + JSON-stdin boundary.

Argparse usage:
    voice-engine --text "..." --language en --task tutor [--provider auto|qwen|omni|chatterbox]
                 [--voice Ryan] [--ref-audio ... --ref-text ...] [--stream] [--out DIR] [--json]

JSON-stdin usage (pipeline boundary):
    echo '{"text":"...","language":"en"}' | python -m voice_engine.cli
"""
from __future__ import annotations
import argparse
import json
import sys

from .engine import synthesize


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voice-engine", description="Local Qwen-first TTS")
    p.add_argument("--text", default="", help="Text to synthesize")
    p.add_argument("--language", default="", help="Language code or name")
    p.add_argument("--task", default="tutor", help="tutor|narration|expressive|clone")
    p.add_argument("--provider", default="auto", help="auto|qwen|omni|chatterbox")
    p.add_argument("--voice", default=None, help="Voice name (Qwen speaker, else ref label)")
    p.add_argument("--ref-audio", default=None, help="Reference WAV for cloning")
    p.add_argument("--ref-text", default=None, help="Exact transcript of ref-audio")
    p.add_argument("--instruct", default=None, help="Qwen CustomVoice style instruction")
    p.add_argument("--stream", action="store_true", help="Use Qwen streaming chunks")
    p.add_argument("--out", default=None, help="Output directory override")
    p.add_argument("--json", action="store_true", help="Print result as JSON")
    return p


def _print_result(r, as_json: bool) -> None:
    if as_json:
        sys.stdout.write(
            json.dumps(
                {
                    "audioPath": r.audio_path,
                    "provider": r.provider,
                    "language": r.language,
                    "voice": r.voice,
                    "generationMs": r.generation_ms,
                    "audioDurationMs": r.audio_duration_ms,
                    "rtf": r.rtf,
                }
            )
        )
    else:
        print(f"audio: {r.audio_path}")
        print(f"provider: {r.provider}  language: {r.language}  voice: {r.voice}")
        print(f"genMs: {r.generation_ms}  audioMs: {r.audio_duration_ms}  rtf: {r.rtf}")


def main(argv: list[str] | None = None) -> None:
    # JSON-stdin mode when piped input exists and no --text given.
    raw = ""
    if not sys.stdin.isatty():
        try:
            raw = sys.stdin.read()
        except Exception:
            raw = ""
    args = build_parser().parse_args(argv)
    payload: dict = {}
    if raw.strip() and not args.text:
        try:
            payload = json.loads(raw)
        except Exception as e:
            print(f"invalid JSON on stdin: {e}", file=sys.stderr)
            sys.exit(1)
    text = args.text or payload.get("text", "")
    language = args.language or payload.get("language", "")
    try:
        result = synthesize(
            text=text,
            language=language,
            voice=args.voice if args.voice is not None else payload.get("voice"),
            provider=args.provider if args.provider != "auto" else payload.get("provider", "auto"),
            task=args.task if args.task != "tutor" else payload.get("task", "tutor"),
            ref_audio=args.ref_audio if args.ref_audio is not None else payload.get("refAudio"),
            ref_text=args.ref_text if args.ref_text is not None else payload.get("refText"),
            stream=args.stream or bool(payload.get("stream", False)),
            instruct=args.instruct,
            out_dir=args.out,
        )
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    _print_result(result, as_json=args.json or bool(raw.strip()))


if __name__ == "__main__":
    main()
