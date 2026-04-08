#!/usr/bin/env python3
"""Qwen3-ASR (0.6B/1.7B) transcriber for medical STT benchmark.

Three backends:
  - transformers: for T4 (auto dtype, slow but works without FA2)
  - vllm:         offline vLLM — A10/H100 with FA2. Note: the offline
                  LLM().transcribe() path stalls on audio > ~10 min due to
                  encoder cache budget limits. Prefer the http backend for
                  long audio.
  - http:         hit an already-running qwen-asr-serve HTTP server via
                  OpenAI-style /v1/audio/transcriptions endpoint. This path
                  does NOT have the long-audio stall and handles all files.

Processes full audio files — no chunking needed.

Usage:
    # T4 (transformers backend, default)
    python transcribe/qwen3_asr_transcribe.py --audio_dir data/raw_audio --model Qwen/Qwen3-ASR-1.7B

    # A10 / H100 via offline vLLM (short audio only)
    python transcribe/qwen3_asr_transcribe.py --audio_dir data/raw_audio --model Qwen/Qwen3-ASR-1.7B --backend vllm

    # A10 / H100 via qwen-asr-serve HTTP (recommended for long audio)
    # First start the server on the GPU host:
    #   FLASHINFER_DISABLE_VERSION_CHECK=1 qwen-asr-serve Qwen/Qwen3-ASR-1.7B \
    #     --gpu-memory-utilization 0.7 --host 0.0.0.0 --port 8000
    # Then run locally or on the host:
    python transcribe/qwen3_asr_transcribe.py --audio_dir data/raw_audio \
      --model Qwen/Qwen3-ASR-1.7B --backend http --server http://<host>:8000
"""

import argparse
import json
import os
import time
import glob

import requests

# Lazy imports: torch + qwen_asr only needed for local backends
def _lazy_import_local():
    import torch
    from qwen_asr import Qwen3ASRModel
    return torch, Qwen3ASRModel

EXCLUDED_FILES = ["day1_consultation07", "day3_consultation03"]


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B", help="Model ID")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    parser.add_argument("--backend", choices=["transformers", "vllm", "http"], default="transformers",
                        help="Inference backend (vllm/http require compute cap >= 8.0 GPU)")
    parser.add_argument("--server", default="http://localhost:8000",
                        help="qwen-asr-serve URL (used when --backend http)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Concurrent in-flight HTTP requests (only used with --backend http). "
                             "Set higher to let vLLM server batch — matches how the H100 run was done.")
    args = parser.parse_args()

    tag = args.model.split("/")[-1].lower()
    transcripts_dir = os.path.join(args.results_dir, "transcripts", tag)
    metrics_dir = os.path.join(args.results_dir, "metrics")
    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"Model: {args.model} ({args.backend} backend)", flush=True)

    model = None
    if args.backend == "http":
        # Verify server is reachable and model is loaded
        try:
            r = requests.get(f"{args.server}/v1/models", timeout=10)
            r.raise_for_status()
            loaded = [m["id"] for m in r.json().get("data", [])]
            print(f"Server {args.server}: models loaded = {loaded}", flush=True)
            if args.model not in loaded:
                print(f"WARNING: {args.model} not in server's loaded models; proceeding anyway", flush=True)
        except Exception as e:
            print(f"ERROR: cannot reach server at {args.server}: {e}", flush=True)
            return
    else:
        torch, Qwen3ASRModel = _lazy_import_local()
        print(f"Loading...", flush=True)
        t0 = time.time()
        if args.backend == "vllm":
            # vLLM offline wrapper — A10/H100 with FA2.
            # Config matches the official Qwen3-ASR docs PLUS one critical
            # override: max_num_batched_tokens=16384 (default 8192 caps the
            # encoder cache, which rejects audio > ~10 min as
            # "exceeds the pre-allocated encoder cache size 8192"). 16384
            # comfortably fits the longest PriMock57 file (~10420 audio tokens
            # at 13 min).
            model = Qwen3ASRModel.LLM(
                model=args.model,
                gpu_memory_utilization=0.7,
                max_inference_batch_size=128,
                max_new_tokens=4096,
                max_num_batched_tokens=16384,
            )
            print(f"Loaded in {time.time()-t0:.0f}s (vLLM backend)", flush=True)
        else:
            # transformers backend — T4 fallback, BF16 auto-upcasts to FP32 on SM 7.5
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

    # ------------------------------------------------------------------
    # vLLM offline backend: batch all files in a SINGLE transcribe() call.
    # This is the docs best-practice path — leverages max_inference_batch_size
    # and avoids the per-request cross-state issues we saw with sequential
    # single-file calls and HTTP endpoints.
    # ------------------------------------------------------------------
    if args.backend == "vllm":
        import soundfile as sf
        to_process = []
        skipped = 0
        for f in audio_files:
            name = os.path.basename(f).replace("_conversation.wav", "")
            if any(exc in name for exc in EXCLUDED_FILES):
                print(f"  Skipping (excluded): {os.path.basename(f)}", flush=True)
                skipped += 1
                continue
            transcript_file = os.path.join(transcripts_dir, f"{name}_conversation_transcript.txt")
            if os.path.exists(transcript_file):
                print(f"  Skipping (exists): {os.path.basename(f)}", flush=True)
                skipped += 1
                continue
            to_process.append(f)

        n = len(to_process)
        print(f"\nBatch transcribing {n} files in a single call (skipped {skipped})...", flush=True)
        if n == 0:
            print("Nothing to process.")
            return

        # Collect audio durations (for per-file metric rows)
        audio_durs = {}
        for f in to_process:
            wav, sr = sf.read(f)
            audio_durs[f] = len(wav) / sr

        batch_start = time.time()
        batch_results = model.transcribe(
            audio=to_process,
            language=["English"] * n,
            return_time_stamps=False,
        )
        batch_elapsed = time.time() - batch_start
        print(f"\nBatch done: {batch_elapsed:.1f}s for {n} files "
              f"(avg {batch_elapsed/n:.1f}s/file batched throughput)", flush=True)

        # Write transcripts; assign avg per-file time as nominal duration
        per_file_time = batch_elapsed / n
        results_rows = []
        for f, r in zip(to_process, batch_results):
            name = os.path.basename(f).replace("_conversation.wav", "")
            text = (r.text or "").strip()
            transcript_file = os.path.join(transcripts_dir, f"{name}_conversation_transcript.txt")
            with open(transcript_file, "w", encoding="utf-8") as wf:
                wf.write(text)
            results_rows.append({
                "audio_file": os.path.basename(f),
                "duration": per_file_time,
                "audio_duration": audio_durs[f],
                "text_length": len(text),
            })
            print(f"  {os.path.basename(f)}: {len(text.split())} words", flush=True)

        durations = [r["duration"] for r in results_rows]
        metrics = {
            "model_name": tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_files": len(results_rows),
                "total_duration": batch_elapsed,
                "average_duration": batch_elapsed / len(results_rows),
                "fastest_file": min(durations),
                "slowest_file": max(durations),
                "note": "Batched throughput: all files in one vLLM transcribe() call. "
                        "Per-file 'duration' is total_duration / n, not individual latency.",
            },
            "file_details": results_rows,
        }
        metrics_file = os.path.join(metrics_dir, f"{tag}_speed.json")
        with open(metrics_file, "w") as mf:
            json.dump(metrics, mf, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")
        print(f"Done: {len(results_rows)} processed, {skipped} skipped")
        return

    # ------------------------------------------------------------------
    # HTTP concurrent batch path: fire N requests in parallel at the
    # qwen-asr-serve server and let vLLM schedule them as one batch.
    # This is how the H100 reference run was done (backend=vllm-server,
    # all 55 files, total_time ~50s). On A10 24GB, tune --concurrency down
    # if you hit OOM; the server's encoder cache determines the ceiling.
    # ------------------------------------------------------------------
    if args.backend == "http":
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import soundfile as sf

        to_process = []
        skipped = 0
        for f in audio_files:
            name = os.path.basename(f).replace("_conversation.wav", "")
            if any(exc in name for exc in EXCLUDED_FILES):
                print(f"  Skipping (excluded): {os.path.basename(f)}", flush=True)
                skipped += 1
                continue
            transcript_file = os.path.join(transcripts_dir, f"{name}_conversation_transcript.txt")
            if os.path.exists(transcript_file):
                print(f"  Skipping (exists): {os.path.basename(f)}", flush=True)
                skipped += 1
                continue
            to_process.append(f)

        n = len(to_process)
        print(f"\nConcurrent HTTP batch: {n} files, {args.concurrency} in-flight (skipped {skipped})", flush=True)
        if n == 0:
            print("Nothing to process.")
            return

        def _transcribe_one(f):
            t0 = time.time()
            with open(f, "rb") as fp:
                resp = requests.post(
                    f"{args.server}/v1/audio/transcriptions",
                    files={"file": fp},
                    data={"model": args.model, "language": "en", "temperature": "0"},
                    timeout=900,
                )
            resp.raise_for_status()
            text = resp.json().get("text", "").strip()
            if "<asr_text>" in text:
                text = text.split("<asr_text>", 1)[1]
            if "</asr_text>" in text:
                text = text.split("</asr_text>", 1)[0]
            text = text.strip()
            elapsed = time.time() - t0
            wav, sr = sf.read(f)
            return f, text, elapsed, len(wav) / sr

        results_rows = []
        errors = []
        batch_start = time.time()
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(_transcribe_one, f): f for f in to_process}
            for fut in as_completed(futures):
                f = futures[fut]
                name = os.path.basename(f)
                try:
                    _f, text, elapsed, audio_dur = fut.result()
                    transcript_file = os.path.join(
                        transcripts_dir,
                        name.replace(".wav", "_transcript.txt"),
                    )
                    with open(transcript_file, "w", encoding="utf-8") as wf:
                        wf.write(text)
                    results_rows.append({
                        "audio_file": name,
                        "duration": elapsed,
                        "audio_duration": audio_dur,
                        "text_length": len(text),
                    })
                    done = len(results_rows) + len(errors)
                    print(f"  [{done}/{n}] Done: {name} ({elapsed:.1f}s, {len(text.split())} words)", flush=True)
                except Exception as e:
                    errors.append((name, str(e)))
                    done = len(results_rows) + len(errors)
                    print(f"  [{done}/{n}] FAILED: {name} - {e}", flush=True)
        batch_elapsed = time.time() - batch_start

        print(f"\nConcurrent batch done: wall-clock {batch_elapsed:.1f}s for {len(results_rows)} files "
              f"(avg per-file server latency {sum(r['duration'] for r in results_rows)/max(1,len(results_rows)):.1f}s)", flush=True)
        if errors:
            print(f"Errors ({len(errors)}):")
            for name, err in errors:
                print(f"  {name}: {err}")

        if results_rows:
            durations = [r["duration"] for r in results_rows]
            metrics = {
                "model_name": tag,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": {
                    "total_files": len(results_rows),
                    "total_duration": sum(durations),
                    "average_duration": sum(durations) / len(results_rows),
                    "fastest_file": min(durations),
                    "slowest_file": max(durations),
                    "wall_clock_batch": batch_elapsed,
                    "concurrency": args.concurrency,
                    "note": f"HTTP concurrent batch via qwen-asr-serve; {args.concurrency} in-flight. "
                            f"Per-file 'duration' is each request's round-trip; wall_clock_batch is total "
                            f"wall time for all files scheduled concurrently.",
                },
                "file_details": results_rows,
            }
            metrics_file = os.path.join(metrics_dir, f"{tag}_speed.json")
            with open(metrics_file, "w") as mf:
                json.dump(metrics, mf, indent=2)
            print(f"\nMetrics saved to: {metrics_file}")
        print(f"Done: {len(results_rows)} processed, {skipped} skipped, {len(errors)} failed")
        return

    # ------------------------------------------------------------------
    # Sequential per-file path: transformers backend or HTTP backend (concurrency=1).
    # ------------------------------------------------------------------
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
        if args.backend == "http":
            with open(f, "rb") as fp:
                resp = requests.post(
                    f"{args.server}/v1/audio/transcriptions",
                    files={"file": fp},
                    data={
                        "model": args.model,
                        "language": "en",
                        "temperature": "0",
                    },
                    timeout=180,  # any clean file finishes in ~30s; 180 is a stall guard
                )
            resp.raise_for_status()
            text = resp.json().get("text", "").strip()
            # Strip Qwen3-ASR output wrapper: "language English<asr_text>...</asr_text>"
            if "<asr_text>" in text:
                text = text.split("<asr_text>", 1)[1]
            if "</asr_text>" in text:
                text = text.split("</asr_text>", 1)[0]
            text = text.strip()
        else:
            r = model.transcribe(audio=[f], language=["English"])
            text = r[0].text
        elapsed = time.time() - start

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
