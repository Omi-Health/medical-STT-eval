#!/usr/bin/env python3
"""Gemma 4 (E2B/E4B) transcriber for medical STT benchmark.

Uses AutoModelForMultimodalLM with dtype=auto, simple 30s concat (no overlap).
Works on T4 (FP16 auto-cast), A10, and H100 (native BF16).

Usage:
    python transcribe/gemma4_transcribe.py --audio_dir data/raw_audio --model google/gemma-4-E2B-it
    python transcribe/gemma4_transcribe.py --audio_dir data/raw_audio --model google/gemma-4-E4B-it
"""

import argparse
import json
import os
import time
import glob

import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

CHUNK_S = 30  # Gemma 4 max audio length
EXCLUDED_FILES = ["day1_consultation07", "day3_consultation03"]
PROMPT = (
    "Transcribe the following speech segment in its original language. "
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits."
)


def transcribe_chunk(model, processor, audio_16k):
    """Transcribe a single audio chunk."""
    messages = [{"role": "user", "content": [
        {"type": "audio", "audio": audio_16k.astype(np.float32)},
        {"type": "text", "text": PROMPT},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=512)
    return processor.decode(outputs[0][input_len:], skip_special_tokens=True)


def transcribe_file(model, processor, audio_path, sr_target=16000):
    """Transcribe a full audio file with 30s chunking + simple concat."""
    wav, sr = sf.read(audio_path, dtype="float32")
    audio_dur = len(wav) / sr

    # Split into 30s chunks
    chunks = []
    for i in range(0, len(wav), sr * CHUNK_S):
        c = wav[i:i + sr * CHUNK_S]
        if len(c) > sr * 2:  # skip tiny final chunks
            c16k = librosa.resample(c, orig_sr=sr, target_sr=sr_target) if sr != sr_target else c
            chunks.append(c16k)

    # Transcribe each chunk
    full_text = ""
    for chunk in chunks:
        text = transcribe_chunk(model, processor, chunk)
        full_text += " " + text

    return " ".join(full_text.split()), audio_dur, len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 Audio Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--model", default="google/gemma-4-E2B-it", help="Model ID")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    args = parser.parse_args()

    # Model tag for output dirs
    tag = args.model.split("/")[-1].lower()
    transcripts_dir = os.path.join(args.results_dir, "transcripts", tag)
    metrics_dir = os.path.join(args.results_dir, "metrics")
    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"Model: {args.model} (dtype=auto, simple 30s concat)")
    print(f"Loading...", flush=True)

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype="auto").to("cuda")
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"Loaded in {time.time()-t0:.0f}s, VRAM: {vram:.1f}GB", flush=True)

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
        text, audio_dur, n_chunks = transcribe_file(model, processor, f)
        elapsed = time.time() - start

        # Save transcript
        with open(transcript_file, "w", encoding="utf-8") as wf:
            wf.write(text)

        results.append({
            "audio_file": os.path.basename(f),
            "duration": elapsed,
            "audio_duration": audio_dur,
            "text_length": len(text),
            "chunks": n_chunks,
        })
        print(f"  [{i}/{len(audio_files)}] Done: {os.path.basename(f)} ({elapsed:.1f}s, {n_chunks} chunks, {len(text.split())} words)", flush=True)

    # Save speed metrics
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
