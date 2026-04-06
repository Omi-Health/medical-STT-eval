#!/usr/bin/env python3
"""Cohere Transcribe (03-2026) via vLLM OpenAI-compatible API.

Serves via: vllm serve CohereLabs/cohere-transcribe-03-2026 --trust-remote-code
Uses: POST /v1/audio/transcriptions endpoint

Usage:
    python transcribe/cohere_transcribe.py --audio_dir data/raw_audio --api_url http://localhost:8000
"""

import argparse
import json
import os
import time
import glob

import requests

MODEL_NAME = "cohere-transcribe-03-2026"
MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
EXCLUDED_FILES = ["day1_consultation07", "day3_consultation03"]


def transcribe_file(audio_path, api_url):
    """Transcribe via vLLM /v1/audio/transcriptions endpoint."""
    url = f"{api_url}/v1/audio/transcriptions"
    with open(audio_path, "rb") as f:
        resp = requests.post(
            url,
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            data={"model": MODEL_ID},
            timeout=600,
        )
    if resp.status_code == 200:
        return resp.json().get("text", "")
    raise RuntimeError(f"API error {resp.status_code}: {resp.text[:200]}")


def main():
    parser = argparse.ArgumentParser(description="Cohere Transcribe via vLLM API")
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--pattern", default="*_conversation.wav")
    args = parser.parse_args()

    transcripts_dir = os.path.join(args.results_dir, "transcripts", MODEL_NAME)
    metrics_dir = os.path.join(args.results_dir, "metrics")
    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    audio_files = sorted(glob.glob(os.path.join(args.audio_dir, args.pattern)))
    print(f"Model: {MODEL_ID} (vLLM API at {args.api_url})")
    print(f"Found {len(audio_files)} audio files\n", flush=True)

    results = []
    skipped = 0
    for i, f in enumerate(audio_files, 1):
        name = os.path.basename(f).replace("_conversation.wav", "")
        if any(exc in name for exc in EXCLUDED_FILES):
            print(f"  [{i}/{len(audio_files)}] Skipping (excluded): {os.path.basename(f)}")
            skipped += 1
            continue

        transcript_file = os.path.join(transcripts_dir, f"{name}_conversation_transcript.txt")
        if os.path.exists(transcript_file):
            print(f"  [{i}/{len(audio_files)}] Skipping (exists): {os.path.basename(f)}")
            skipped += 1
            continue

        try:
            import soundfile as sf
            wav, sr = sf.read(f)
            audio_dur = len(wav) / sr

            start = time.time()
            text = transcribe_file(f, args.api_url)
            elapsed = time.time() - start

            with open(transcript_file, "w", encoding="utf-8") as wf:
                wf.write(text)

            results.append({
                "audio_file": os.path.basename(f),
                "duration": elapsed,
                "audio_duration": audio_dur,
                "text_length": len(text),
            })
            print(f"  [{i}/{len(audio_files)}] Done: {os.path.basename(f)} ({elapsed:.1f}s, {len(text.split())} words)", flush=True)

        except Exception as e:
            print(f"  [{i}/{len(audio_files)}] ERROR: {os.path.basename(f)} — {e}", flush=True)

    if results:
        durations = [r["duration"] for r in results]
        metrics = {
            "model_name": MODEL_NAME,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_files": len(results),
                "total_duration": sum(durations),
                "average_duration": sum(durations) / len(durations),
                "fastest_file": min(durations),
                "slowest_file": max(durations),
            },
            "file_details": results,
        }
        metrics_file = os.path.join(metrics_dir, f"{MODEL_NAME}_speed.json")
        with open(metrics_file, "w") as mf:
            json.dump(metrics, mf, indent=2)
        print(f"\nSpeed saved: {metrics_file}")

    print(f"\nDone: {len(results)} processed, {skipped} skipped")


if __name__ == "__main__":
    main()
