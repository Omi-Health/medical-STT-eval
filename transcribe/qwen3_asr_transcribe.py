#!/usr/bin/env python3
"""Qwen3-ASR (0.6B/1.7B) transcriber for medical STT benchmark.

Uses qwen_asr transformers backend. Works on T4 (auto dtype).
Processes full audio files — no chunking needed.

Usage:
    python transcribe/qwen3_asr_transcribe.py --audio_dir data/raw_audio --model Qwen/Qwen3-ASR-1.7B
    python transcribe/qwen3_asr_transcribe.py --audio_dir data/raw_audio --model Qwen/Qwen3-ASR-0.6B
"""

import argparse
import json
import os
import time
import glob

import torch
from qwen_asr import Qwen3ASRModel

EXCLUDED_FILES = ["day1_consultation07", "day3_consultation03"]


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B", help="Model ID")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    args = parser.parse_args()

    tag = args.model.split("/")[-1].lower()
    transcripts_dir = os.path.join(args.results_dir, "transcripts", tag)
    metrics_dir = os.path.join(args.results_dir, "metrics")
    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"Model: {args.model} (transformers backend)", flush=True)
    print(f"Loading...", flush=True)

    t0 = time.time()
    # BF16 — PyTorch auto-upcasts to FP32 on T4 (compute cap 7.5)
    # This worked for 9 files earlier at ~200s/file, faster than 8-bit (~380s/file)
    dtype = torch.bfloat16
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="cuda:0",
        max_new_tokens=4096,
    )
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"Loaded in {time.time()-t0:.0f}s, VRAM: {vram:.1f}GB, dtype: {dtype}", flush=True)

    audio_files = sorted(glob.glob(os.path.join(args.audio_dir, args.pattern)))
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

        start = time.time()
        r = model.transcribe(audio=[f], language=["English"])
        elapsed = time.time() - start
        text = r[0].text

        with open(transcript_file, "w", encoding="utf-8") as wf:
            wf.write(text)

        import soundfile as sf
        wav, sr = sf.read(f)
        audio_dur = len(wav) / sr

        results.append({
            "audio_file": os.path.basename(f),
            "duration": elapsed,
            "audio_duration": audio_dur,
            "text_length": len(text),
        })
        print(f"  [{i}/{len(audio_files)}] Done: {os.path.basename(f)} ({elapsed:.1f}s, {len(text.split())} words)", flush=True)

    if results:
        durations = [r["duration"] for r in results]
        metrics = {
            "model_name": tag,
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
        metrics_file = os.path.join(metrics_dir, f"{tag}_speed.json")
        with open(metrics_file, "w") as mf:
            json.dump(metrics, mf, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")

    print(f"\nDone: {len(results)} processed, {skipped} skipped")


if __name__ == "__main__":
    main()
