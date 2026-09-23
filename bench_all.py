"""Full engine test: routing + real-synthesis attempts + sequential-vs-parallel lag test.

Usage:
    python -m voice_engine.bench_all [--mock-sleep MS] [--out DIR]

What it does:
  Phase A: routing matrix over all 3 providers (no model deps needed).
  Phase B: REAL engine.synthesize() attempt per provider (needs setup.sh deps;
           records SUCCESS or SKIP with wall time + reason -- no faking).
  Phase C: orchestration lag test -- Task1 (EN tutor->qwen) + Task2 (NE tutor->omni)
           run sequentially, then simultaneously (2 threads). Uses a mock shim with
           fixed sleeps, CLEARLY labeled MOCK, measuring orchestration overhead only.
           Absolute mock ms are NOT model benchmarks; real numbers come from
           `python -m voice_engine.compare_en` after `./voice_engine/setup.sh`.

All artifacts go to voice_engine/outputs/<run_id>/ : results.json, timings.csv,
report.md, plus mock WAVs (440Hz sine, stdlib only) so the folder is analyzable.
Time tracking: ISO wall clock + monotonic perf_counter_ns per event.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import json
import math
import struct
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "outputs"

TASK1 = {"label": "task1_en_tutor", "text": "Today we are going to understand how attention works.",
         "language": "en", "task": "tutor"}
TASK2 = {"label": "task2_ne_tutor", "text": "आज हामी न्युरल नेटवर्क कसरी काम गर्छ भन्ने बुझ्नेछौँ।",
         "language": "ne", "task": "tutor"}
TASK3 = {"label": "task3_en_expressive", "text": "And this is the surprising part. [chuckle] Attention is not uniform.",
         "language": "en", "task": "expressive"}

ROUTINE_CASES = [
    ("en", "tutor", "Today we are going to understand how attention works."),
    ("zh", "tutor", "今天我们学习神经网络。"),
    ("ja", "tutor", "今日はニューラルネットワークについて学びます。"),
    ("ko", "tutor", "오늘 신경망을 배웁니다."),
    ("es", "tutor", "Hoy aprendemos redes neuronales."),
    ("fr", "tutor", "Aujourd'hui nous étudions les réseaux de neurones."),
    ("de", "tutor", "Heute lernen wir neuronale Netze."),
    ("ne", "tutor", "आज हामी सिक्नेछौँ।"),
    ("hi", "tutor", "आज हम सीखेंगे।"),
    ("bn", "tutor", "আজ আমরা শিখব।"),
    ("en", "expressive", "Wait. [chuckle] Really?"),
]


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_sine_wav(path: Path, seconds: float = 1.0, freq: float = 440.0, sr: int = 24000) -> None:
    n = int(seconds * sr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(n):
            v = int(16000 * math.sin(2 * math.pi * freq * i / sr))
            w.writeframes(struct.pack("<h", v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock-sleep", type=float, default=500.0,
                    help="Mock provider latency per task in ms (orchestration test only)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.out) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    events: list[dict] = []
    run_start_iso = now_iso()
    run_t0 = time.perf_counter_ns()

    def record(**kw):
        kw.setdefault("t_iso", now_iso())
        events.append(kw)

    # ---------- Phase A: routing ----------
    from voice_engine.router import route

    phase_t0 = time.perf_counter_ns()
    for lang, task, text in ROUTINE_CASES:
        t0 = time.perf_counter_ns()
        try:
            d = route(lang, text=text, task=task)
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            record(phase="A_routing", lang=lang, task=task, provider=d.provider,
                   reason=d.reason, status="OK", latency_ms=round(dt_ms, 3))
        except Exception as e:
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            record(phase="A_routing", lang=lang, task=task, provider="-",
                   reason=str(e), status="ERROR", latency_ms=round(dt_ms, 3))
    phase_ms = (time.perf_counter_ns() - phase_t0) / 1e6

    # ---------- Phase B: real synthesis attempts ----------
    from voice_engine.engine import synthesize

    real_rows = []
    for spec in (TASK1, TASK2, TASK3):
        t0 = time.perf_counter_ns()
        try:
            r = synthesize(spec["text"], spec["language"], task=spec["task"])
            wall_ms = (time.perf_counter_ns() - t0) / 1e6
            row = {"label": spec["label"], "status": "SUCCESS", "provider": r.provider,
                   "language": r.language, "voice": r.voice, "audio_path": r.audio_path,
                   "generation_ms": r.generation_ms, "audio_duration_ms": r.audio_duration_ms,
                   "rtf": r.rtf, "wall_ms": round(wall_ms, 1)}
        except Exception as e:
            wall_ms = (time.perf_counter_ns() - t0) / 1e6
            row = {"label": spec["label"], "status": "SKIP", "provider": "-",
                   "language": spec["language"], "voice": "-",
                   "audio_path": "-", "generation_ms": "-", "audio_duration_ms": "-",
                   "rtf": "-", "wall_ms": round(wall_ms, 1),
                   "reason": f"{type(e).__name__}: {e}"}
        real_rows.append(row)
        record(phase="B_real_synth", **{k: v for k, v in row.items()})
    # Mock WAVs so outputs/ always has analyzable audio (labeled mock).
    for spec in (TASK1, TASK2, TASK3):
        write_sine_wav(outdir / f"{spec['label']}_MOCK.wav")

    # ---------- Phase C: sequential vs simultaneous (mock shim) ----------
    sleep_s = float(args.mock_sleep) / 1000.0

    def mock_task(spec: dict) -> dict:
        from voice_engine.router import route as _route

        t0 = time.perf_counter_ns()
        d = _route(spec["language"], text=spec["text"], task=spec["task"])
        time.sleep(sleep_s)  # simulated model load+generate
        wall_ms = (time.perf_counter_ns() - t0) / 1e6
        return {"label": spec["label"], "provider": d.provider, "wall_ms": round(wall_ms, 1)}

    pair = [TASK1, TASK2]
    s0 = time.perf_counter_ns()
    seq_rows = [mock_task(s) for s in pair]
    seq_wall = (time.perf_counter_ns() - s0) / 1e6
    for r in seq_rows:
        record(phase="C_mock_sequential", mode="sequential", mock=True,
               mock_sleep_ms=args.mock_sleep, **r)
    record(phase="C_mock_sequential", mode="sequential_total", mock=True,
           wall_ms=round(seq_wall, 1),
           note=f"sum(tasks)={sum(r['wall_ms'] for r in seq_rows):.1f}ms")

    p0 = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=2) as ex:
        par_rows = list(ex.map(mock_task, pair))
    par_wall = (time.perf_counter_ns() - p0) / 1e6
    for r in par_rows:
        record(phase="C_mock_parallel", mode="parallel", mock=True,
               mock_sleep_ms=args.mock_sleep, **r)
    record(phase="C_mock_parallel", mode="parallel_total", mock=True,
           wall_ms=round(par_wall, 1))
    lag_ms = par_wall - seq_wall
    slower = max((r["wall_ms"] for r in par_rows), default=0) - min(
        (r["wall_ms"] for r in seq_rows), default=0)
    record(phase="C_verdict", mock=True, mock_sleep_ms=args.mock_sleep,
           seq_wall_ms=round(seq_wall, 1), par_wall_ms=round(par_wall, 1),
           lag_ms=round(lag_ms, 1),
           note=("parallel FASTER (mock sleeps overlap; real models contend for "
                 "memory/Metal so expect the opposite with real weights)")
           if lag_ms < 0 else "parallel SLOWER: contention overhead dominates")

    run_wall_ms = (time.perf_counter_ns() - run_t0) / 1e6
    run_end_iso = now_iso()

    # ---------- Artifacts ----------
    payload = {
        "run_id": run_id, "run_start_iso": run_start_iso, "run_end_iso": run_end_iso,
        "run_wall_ms": round(run_wall_ms, 1),
        "python": sys.version.split()[0],
        "mock_sleep_ms": args.mock_sleep,
        "phaseA_routing_ms": round(phase_ms, 1),
        "phaseB_real": real_rows,
        "phaseC": {"sequential_wall_ms": round(seq_wall, 1),
                   "parallel_wall_ms": round(par_wall, 1),
                   "lag_ms": round(lag_ms, 1)},
        "events": events,
    }
    (outdir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    with open(outdir / "timings.csv", "w", newline="") as f:
        keys = sorted({k for e in events for k in e.keys()})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(events)

    ok_routes = sum(1 for e in events if e.get("phase") == "A_routing" and e.get("status") == "OK")
    real_ok = sum(1 for r in real_rows if r["status"] == "SUCCESS")
    (outdir / "report.md").write_text(
        f"# voice_engine test {run_id}\n\n"
        f"- period: {run_start_iso} .. {run_end_iso} (wall {run_wall_ms:.0f} ms)\n"
        f"- python: {sys.version.split()[0]}\n\n"
        f"## A routing: {ok_routes}/{len(ROUTINE_CASES)} OK ({phase_ms:.1f} ms)\n\n"
        f"## B real synthesis: {real_ok}/3 SUCCESS\n\n"
        + "".join(f"- {r['label']}: {r['status']} provider={r['provider']} "
                  f"wall={r['wall_ms']}ms"
                  + (f" gen={r['generation_ms']}ms audio={r['audio_duration_ms']}ms rtf={r['rtf']}\n"
                     if r["status"] == "SUCCESS"
                     else f" reason={r.get('reason','')}\n")
                  for r in real_rows)
        + f"\n## C orchestration lag (MOCK sleep={args.mock_sleep:.0f}ms/task)\n\n"
        f"- sequential wall: {seq_wall:.1f} ms\n"
        f"- parallel wall:   {par_wall:.1f} ms\n"
        f"- lag (par-seq):   {lag_ms:+.1f} ms\n\n"
        f"> Mock-only: sleeps overlap in threads, so parallel looks faster here. "
        f"With real weights the engine is single-resident (one provider in memory), "
        f"so simultaneous requests force unload/reload thrash + Metal contention: "
        f"expect parallel SLOWER. Keep requests sequential scene-by-scene.\n",
        encoding="utf-8",
    )
    print(f"run_id: {run_id}")
    print(f"period: {run_start_iso} .. {run_end_iso}  ({run_wall_ms:.0f} ms)")
    print(f"A routing: {ok_routes}/{len(ROUTINE_CASES)} OK")
    for r in real_rows:
        print(f"B {r['label']}: {r['status']} provider={r['provider']} wall={r['wall_ms']}ms"
              + ("" if r["status"] == "SUCCESS" else f" :: {r.get('reason','')}"))
    print(f"C sequential={seq_wall:.1f}ms parallel={par_wall:.1f}ms lag={lag_ms:+.1f}ms (MOCK)")
    print(f"artifacts: {outdir}")


if __name__ == "__main__":
    main()
