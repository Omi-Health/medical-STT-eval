#!/usr/bin/env python3
"""facebook/mms-1b-all transcriber for medical STT benchmark.

Wav2Vec2ForCTC with language adapters. Uses HF ASR pipeline with
chunk_length_s=30, stride_length_s=(5, 5) to handle long audio without
blowing T4 VRAM.

Usage:
    python transcribe/mms_1b_transcribe.py --audio_dir data/raw_audio
"""

import argparse
import glob
import json
import os
import time

import torch
from transformers import AutomaticSpeechRecognitionPipeline, AutoProcessor, Wav2Vec2ForCTC

EXCLUDED_FILES = ["day1_consultation07", "day3_consultation03"]
MODEL_ID = "facebook/mms-1b-all"
TARGET_LANG = "eng"


def main():
    parser = argparse.ArgumentParser(description="MMS-1B-all Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    parser.add_argument("--chunk_length_s", type=int, default=30)
    parser.add_argument("--stride_length_s", type=int, default=5)
    args = parser.parse_args()

    tag = "mms-1b-all"
    transcripts_dir = os.path.join(args.results_dir, "transcripts", tag)
    metrics_dir = os.path.join(args.results_dir, "metrics")
    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"Model: {MODEL_ID} (target lang: {TARGET_LANG})", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID, target_lang=TARGET_LANG)
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID,
        target_lang=TARGET_LANG,
        ignore_mismatched_sizes=True,
    ).to("cuda")
    model.load_adapter(TARGET_LANG)
    processor.tokenizer.set_target_lang(TARGET_LANG)
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"Loaded in {time.time()-t0:.0f}s, VRAM: {vram:.1f}GB", flush=True)

    pipe = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0,
    )

    audio_files = sorted(glob.glob(os.path.join(args.audio_dir, args.pattern)))
    print(f"Found {len(audio_files)} audio files\n", flush=True)

    results = []
    skipped = 0
    for i, f in enumerate(audio_files, 1):
        name = os.path.basename(f).replace("_conversation.wav", "")
        if any(exc in name for exc in EXCLUDED_FILES):
            print(f"  [{i}/{len(audio_files)}] Skipping (excluded): {os.path.basename(f)}", flush=True)
            skipped += 1
            continue

        transcript_file = os.path.join(transcripts_dir, f"{name}_conversation_transcript.txt")
        if os.path.exists(transcript_file):
            print(f"  [{i}/{len(audio_files)}] Skipping (exists): {os.path.basename(f)}", flush=True)
            skipped += 1
            continue

        try:
            start = time.time()
            out = pipe(
                f,
                chunk_length_s=args.chunk_length_s,
                stride_length_s=(args.stride_length_s, args.stride_length_s),
                return_timestamps="char",  # required by HF pipeline for chunked CTC merging
            )
            elapsed = time.time() - start
            text = out["text"] if isinstance(out, dict) else str(out)

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
        except Exception as e:
            print(f"  [{i}/{len(audio_files)}] FAILED: {os.path.basename(f)} - {e}", flush=True)

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
