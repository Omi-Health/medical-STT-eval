#!/usr/bin/env python3
"""Batch full-audio transcription of PriMock57 through the Lanson S2 (qwen-long) service.

S2 = Qwen3-ASR-1.7B served by vLLM 0.19 native (full-audio, no chunking) at
qwen-s2.lansonai.com:8010. This script POSTs each WAV to the OpenAI-compatible
/v1/audio/transcriptions endpoint with high concurrency.

Outputs:
  results/transcripts/<model>/<stem>_conversation_transcript.txt   (scorer layout)
  results/<cache>/<stem>.json                                      (raw API responses, resumable)

Usage:
  python scripts/lanson_s2_batch.py
  python scripts/lanson_s2_batch.py --concurrency 52 --model_name lanson-s2-fullaudio
  python scripts/lanson_s2_batch.py --files day1_consultation01,day3_consultation04
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_ENDPOINT = "http://qwen-s2.lansonai.com:8010/v1/audio/transcriptions"
DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
EXCLUDED_FILES = ["day1_consultation07", "day3_consultation03"]


def transcribe(endpoint: str, wav: Path, model: str, timeout: int, retries: int) -> dict:
    ref_stem = _ref_stem(wav)
    last_err = ""
    for attempt in range(1, retries + 2):
        try:
            result = subprocess.run(
                [
                    "curl", "-sS", "--fail-with-body", "-m", str(timeout),
                    endpoint, "-F", f"file=@{wav}", "-F", f"model={model}",
                ],
                capture_output=True,
                text=True,
                timeout=timeout + 15,
            )
            if result.returncode != 0:
                raise RuntimeError((result.stdout or result.stderr)[:300])
            data = json.loads(result.stdout)
            text = (data.get("text") or "").strip()
            if not text:
                raise RuntimeError("empty transcript in response")
            return {"stem": ref_stem, "text": text, "data": data}
        except Exception as exc:
            last_err = str(exc)
            print(f"  ⚠️  {ref_stem} attempt {attempt}/{retries + 1}: {last_err}", flush=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"{ref_stem} failed after {retries + 1} attempts: {last_err}")


def _ref_stem(wav: Path) -> str:
    """Reference stem for a PriMock57 WAV: strip the trailing '_conversation'."""
    return wav.stem.removesuffix("_conversation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch full-audio transcription via Lanson S2")
    parser.add_argument("--audio_dir", default="data/raw_audio")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--model_name", default="lanson-s2-fullaudio")
    parser.add_argument("--cache", help="Cache dir under results (default: results/<model>-cache)")
    parser.add_argument("--endpoint", default=os.environ.get("LANSON_S2_URL", DEFAULT_ENDPOINT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=52)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--files", help="Comma-separated stems (no extension)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true", help="Skip stems already transcribed")
    args = parser.parse_args()

    root = Path(args.results_dir)
    transcripts_dir = root / "transcripts" / args.model_name.replace("/", "_")
    cache_dir = root / (args.cache or f"{args.model_name}-cache")
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.files:
        stems = [s.strip() for s in args.files.split(",") if s.strip()]
        audio_files = [Path(args.audio_dir) / f"{s}_conversation.wav" for s in stems]
    else:
        audio_files = sorted(Path(args.audio_dir).glob("*.wav"))
    skipped = [f for f in audio_files if any(exc in f.stem for exc in EXCLUDED_FILES)]
    audio_files = [f for f in audio_files if f not in skipped]
    for f in skipped:
        print(f"⏭️  Skipping excluded file: {f.name}", flush=True)

    if args.skip_existing:
        before = len(audio_files)
        audio_files = [f for f in audio_files if not (transcripts_dir / f"{_ref_stem(f)}_conversation_transcript.txt").exists()]
        print(f"⏭️  Skip-existing: {len(audio_files)} remaining (dropped {before - len(audio_files)} already transcribed)", flush=True)

    if args.limit:
        audio_files = audio_files[: args.limit]

    print(f"🎤 S2 full-audio batch — model={args.model_name} concurrency={args.concurrency}")
    print(f"🔗 {args.endpoint} (model {args.model})")
    print(f"📁 {len(audio_files)} file(s) → {transcripts_dir}", flush=True)

    started = time.monotonic()
    ok, failed = 0, 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {pool.submit(transcribe, args.endpoint, f, args.model, args.timeout, args.retries): f for f in audio_files}
        for future in as_completed(futures):
            wav = futures[future]
            try:
                out = future.result()
                stem = out["stem"]
                cache_path = cache_dir / f"{stem}.json"
                cache_path.write_text(json.dumps(out["data"], ensure_ascii=False, indent=2), encoding="utf-8")
                transcript_path = transcripts_dir / f"{stem}_conversation_transcript.txt"
                transcript_path.write_text(out["text"], encoding="utf-8")
                ok += 1
                print(f"✅ {stem}: {len(out['text'].split())} words → {transcript_path.name}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"❌ {wav.name}: {exc}", flush=True)

    wall = time.monotonic() - started
    print(f"\n🏁 {ok}/{len(audio_files)} transcribed in {wall:.1f}s ({wall / max(ok, 1):.2f}s/file)")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()