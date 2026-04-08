#!/usr/bin/env python3
"""zai-org/GLM-ASR-Nano-2512 transcriber for medical STT benchmark.

Chat-template multimodal model. BF16 on CUDA. Chunks long audio into
30s windows (the model's context is small) and concatenates outputs.

Usage:
    python transcribe/glm_asr_nano_transcribe.py --audio_dir data/raw_audio
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

EXCLUDED_FILES = ["day1_consultation07", "day3_consultation03"]
MODEL_ID = "zai-org/GLM-ASR-Nano-2512"
CHUNK_SEC = 30
SAMPLE_RATE = 16000


def load_audio_16k(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        # Simple linear resample (audio already 16k in PriMock57)
        dur = len(wav) / sr
        n = int(round(dur * SAMPLE_RATE))
        x_old = np.linspace(0, dur, num=len(wav), endpoint=False)
        x_new = np.linspace(0, dur, num=n, endpoint=False)
        wav = np.interp(x_new, x_old, wav).astype(np.float32)
    return wav


def transcribe_chunk(model, processor, device, audio_chunk: np.ndarray) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_chunk},
                {"type": "text", "text": "Please transcribe this audio into text"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    new_tokens = outputs[:, inputs.input_ids.shape[1]:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return text.strip()


def transcribe_file(model, processor, device, path: str) -> str:
    wav = load_audio_16k(path)
    total = len(wav)
    step = CHUNK_SEC * SAMPLE_RATE
    pieces = []
    pos = 0
    while pos < total:
        chunk = wav[pos:pos + step]
        pos += step
        if len(chunk) < SAMPLE_RATE:  # skip trailing <1s
            continue
        text = transcribe_chunk(model, processor, device, chunk)
        if text:
            pieces.append(text)
    return " ".join(pieces).strip()


def main():
    parser = argparse.ArgumentParser(description="GLM-ASR-Nano Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    args = parser.parse_args()

    tag = "glm-asr-nano-2512"
    transcripts_dir = os.path.join(args.results_dir, "transcripts", tag)
    metrics_dir = os.path.join(args.results_dir, "metrics")
    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {MODEL_ID} on {device}", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0.0
    print(f"Loaded in {time.time()-t0:.0f}s, VRAM: {vram:.1f}GB", flush=True)

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
            text = transcribe_file(model, processor, device, f)
            elapsed = time.time() - start

            with open(transcript_file, "w", encoding="utf-8") as wf:
                wf.write(text)

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
