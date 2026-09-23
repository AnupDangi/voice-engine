"""Compare installed providers on one English sample.

Usage: python -m voice_engine.compare_en
Writes voice_outputs/compare_en/{qwen,omni,chatterbox}.wav + timing table.
Missing optional providers are skipped (not fatal).
"""
from __future__ import annotations
import time
from pathlib import Path

TEXT = (
    "A good explanation does not start with the answer. It starts with a question "
    "you can almost answer, then closes the gap one step at a time."
)
ROUTES = [
    ("English", "en"), ("Chinese", "zh"), ("Japanese", "ja"), ("Korean", "ko"),
    ("Spanish", "es"), ("French", "fr"), ("German", "de"),
    ("Nepali", "ne"), ("Hindi", "hi"), ("Bengali", "bn"), ("Tamil", "ta"),
    ("Telugu", "te"), ("Urdu", "ur"), ("Thai", "th"), ("Vietnamese", "vi"),
    ("Indonesian", "id"),
]


def main() -> None:
    from .paths import DEFAULT_OUT_DIR
    from .router import route

    print("Router (text + language -> provider):")
    plain = "Today we are going to understand how attention works."
    for label, lang in ROUTES:
        try:
            d = route(lang, text=plain)
            print(f"  {label:<10} {lang:<3} -> {d.provider} ({d.reason})")
        except Exception as e:
            print(f"  {label:<10} {lang:<3} -> ERROR {e}")
    print(f"  expressive EN -> {route('en', text='Wait. [chuckle] Really?').provider}")

    print(f"\nEnglish sample ({len(TEXT.split())} words):\n  \"{TEXT}\"\n")
    from .engine import synthesize

    out = Path(DEFAULT_OUT_DIR) / "compare_en"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for provider in ("qwen", "omni", "chatterbox"):
        try:
            t0 = time.time()
            r = synthesize(TEXT, "en", provider=provider, out_dir=str(out))
            wall = (time.time() - t0) * 1000.0
            dest = out / f"{provider}.wav"
            try:
                import shutil

                if Path(r.audio_path).resolve() != dest.resolve():
                    shutil.copyfile(r.audio_path, dest)
                    shown = str(dest)
                else:
                    shown = r.audio_path
            except Exception:
                shown = r.audio_path
            rows.append((provider, shown, r.audio_duration_ms, r.generation_ms, f"{r.rtf:.3f}", f"{wall:.0f}"))
        except Exception as e:
            rows.append((provider, f"SKIP: {e}", "-", "-", "-", "-"))
    header = ("provider", "audio", "audioMs", "genMs", "rtf", "wallMs")
    widths = [max(len(header[i]), max(len(str(r[i])) for r in rows)) for i in range(len(header))]
    line = lambda cells: "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))
    print(line(list(header)))
    print(line(["-" * w for w in widths]))
    for r in rows:
        print(line(list(r)))


if __name__ == "__main__":
    main()
