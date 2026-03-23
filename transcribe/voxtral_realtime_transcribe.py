#!/usr/bin/env python3
"""
Voxtral-Mini-4B-Realtime transcriber for speech-to-text.
Runs locally on GPU using transformers >= 5.2.0.
Requires: pip install transformers "mistral-common[audio]"
"""

import time
import json
import os
from pathlib import Path

import torch
from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
from mistral_common.tokens.tokenizers.audio import Audio


EXCLUDED_FILES = ['day1_consultation07', 'day3_consultation03']
MODEL_NAME = "voxtral-mini-4b-realtime"
REPO_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"


def transcribe_all(audio_dir: str, results_dir: str = "results"):
    audio_dir = Path(audio_dir)
    results_dir = Path(results_dir)
    transcripts_dir = results_dir / "transcripts" / MODEL_NAME
    metrics_dir = results_dir / "metrics"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    print(f"Loading {REPO_ID}...")
    processor = AutoProcessor.from_pretrained(REPO_ID)
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        REPO_ID, device_map="auto", torch_dtype=torch.float16
    )
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    audio_files = sorted(audio_dir.glob("*_conversation.wav"))
    print(f"Found {len(audio_files)} audio files\n")

    results = []
    skipped = 0

    for i, audio_file in enumerate(audio_files, 1):
        name = audio_file.stem

        if any(exc in name for exc in EXCLUDED_FILES):
            print(f"  [{i}/{len(audio_files)}] Skipping (excluded): {audio_file.name}")
            skipped += 1
            continue

        transcript_file = transcripts_dir / f"{name}_transcript.txt"
        if transcript_file.exists():
            print(f"  [{i}/{len(audio_files)}] Skipping (exists): {audio_file.name}")
            skipped += 1
            continue

        try:
            # Load and resample audio
            audio = Audio.from_file(str(audio_file), strict=False)
            audio.resample(processor.feature_extractor.sampling_rate)
            dur = len(audio.audio_array) / processor.feature_extractor.sampling_rate

            # 1 text token = 80ms of audio, plus headroom for text output
            max_tokens = int(dur / 0.08) + 4096

            inputs = processor(audio.audio_array, return_tensors="pt")
            inputs = inputs.to(model.device, dtype=model.dtype)

            start = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            duration = time.time() - start

            text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

            # Save transcript
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(text)

            results.append({
                "audio_file": audio_file.name,
                "duration": duration,
                "text_length": len(text),
                "audio_duration": dur,
            })
            print(f"  [{i}/{len(audio_files)}] Done: {audio_file.name} ({duration:.1f}s, {len(text.split())} words)")

        except Exception as e:
            print(f"  [{i}/{len(audio_files)}] FAILED: {audio_file.name} - {e}")

    # Save speed metrics
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
        metrics_file = metrics_dir / f"{MODEL_NAME}_speed.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nMetrics saved to: {metrics_file}")

    print(f"\nDone: {len(results)} processed, {skipped} skipped")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Voxtral-Mini-4B-Realtime Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    args = parser.parse_args()
    transcribe_all(args.audio_dir, args.results_dir)
