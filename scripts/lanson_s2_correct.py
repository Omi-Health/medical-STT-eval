#!/usr/bin/env python3
"""Full PriMock57 correction through the REAL Lanson live-api /v1/transcript/correct.

Uses the real service's batch correction pipeline (guide build + numbered-line
chunking + structural fallback) with a custom prompt override, exactly as the
production audio-v1 offline flow does — no separate experiment path.

Run on the S2 host where live-api listens on 127.0.0.1:41759 (green build with
custom-prompt support) and LIVE_INTERNAL_API_TOKEN is available.

Usage:
  python3 scripts/lanson_s2_correct.py --input benchmark/s2-raw --out benchmark/s2-corrected \
      --prompt prompts/experiments/correction-batch.v10.md --endpoint http://127.0.0.1:41759
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import urllib.request

DEFAULT_ENDPOINT = "http://127.0.0.1:41759/v1/transcript/correct"
SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


def correct_file(endpoint: str, token: str, prompt: str, transcript: str, concurrency: int, timeout: int, retries: int) -> tuple[str, dict]:
    segments = [s for s in SENT_SPLIT.split(transcript.strip()) if s]
    body = {
        "segments": segments,
        "language": "en",
        "guide": True,
        "concurrency": concurrency,
        "prompt": prompt,
    }
    last_err = ""
    for attempt in range(1, retries + 2):
        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.load(resp)
            pairs = [(s["text"], s["corrected"]) for s in out["segments"]]
            corrected = " ".join(_guarded(c, r) for r, c in pairs).strip()
            return corrected, out["stats"]
        except Exception as exc:
            last_err = str(exc)
            print(f"  ⚠️  attempt {attempt}/{retries + 1}: {last_err}", flush=True)
            time.sleep(3 * attempt)
    raise RuntimeError(last_err)


def _is_latin(text: str) -> bool:
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return latin > cjk


def _guarded(corrected: str, raw: str) -> str:
    """Language guard: an English segment that comes back non-Latin (e.g. the
    model echoing instruction words) must fall back to the raw transcript."""
    if corrected and not _is_latin(corrected):
        return raw
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch correct full-audio transcripts via real live-api")
    parser.add_argument("--input", default="benchmark/s2-raw")
    parser.add_argument("--out", default="benchmark/s2-corrected")
    parser.add_argument("--prompt", default="prompts/experiments/correction-batch.v10.md")
    parser.add_argument("--endpoint", default=os.environ.get("LANSON_CORRECT_URL", DEFAULT_ENDPOINT))
    parser.add_argument("--files", help="Comma-separated stems")
    parser.add_argument("--file-concurrency", type=int, default=6)
    parser.add_argument("--seg-concurrency", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    token = os.environ.get("LIVE_INTERNAL_API_TOKEN", "")
    if not token:
        raise SystemExit("Set LIVE_INTERNAL_API_TOKEN (from workerd.s2.env)")

    prompt = Path(args.prompt).read_text(encoding="utf-8")
    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*_transcript.txt"))
    if args.files:
        stems = set(args.files.split(","))
        files = [f for f in files if f.stem.removeprefix("day1_").startswith("consultation") or f.stem in stems or any(f.stem.startswith(s) for s in stems)]
    pending = [f for f in files if not (out_dir / f.name).exists()]
    print(f"📁 {len(pending)} file(s) to correct → {out_dir}", flush=True)

    started = time.monotonic()
    ok, failed = 0, 0
    with ThreadPoolExecutor(max_workers=max(1, args.file_concurrency)) as pool:
        futures = {
            pool.submit(correct_file, args.endpoint, token, prompt, f.read_text(encoding="utf-8"), args.seg_concurrency, args.timeout, args.retries): f
            for f in pending
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                corrected, stats = future.result()
                (out_dir / path.name).write_text(corrected, encoding="utf-8")
                ok += 1
                print(f"✅ {path.name}: changed={stats.get('changed')} fallbacks={stats.get('fallbacks')} prompt={stats.get('prompt_version')}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"❌ {path.name}: {exc}", flush=True)

    wall = time.monotonic() - started
    print(f"\n🏁 {ok}/{len(pending)} corrected in {wall:.1f}s")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()