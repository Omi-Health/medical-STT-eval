#!/usr/bin/env python3
"""Transcribe PriMock57 medical consultations using Oruk Fourier 2B.

Supports two backends:
1. ``transformers`` (default, recommended): loads the canonical native BF16
   checkpoint ``oruk/oruk-fourier-2b`` pinned to commit
   ``8c2a8871f1304aa2f634da4fb95e24438c4945b9`` via ``transformers>=5.21``
   (``AutoModelForMultimodalLM`` + ``AutoProcessor``). Downloads only the 7
   native root checkpoint files, verifies ``model.safetensors`` SHA-256 before
   inference, segments long consultations (>28 s) at low-energy speech pauses
   into 20-28 s chunks, and decodes using the checkpoint's published
   ``generation_config.json`` (``num_beams=3``, ``no_repeat_ngram_size=8``,
   ``bad_words_ids``).
2. ``http`` (optional): sends audio files to an OpenAI-compatible transcription
   endpoint such as Oruk's hosted API (``https://speech-api.oruk.ai``) using
   ``ORUK_API_KEY`` from the environment.

Usage:
    # Local native GPU evaluation (HF Transformers BF16, pinned checkpoint)
    python transcribe/oruk_fourier_transcribe.py --batch-size 16

    # Optional hosted API evaluation
    ORUK_API_KEY=... python transcribe/oruk_fourier_transcribe.py \
        --backend http --server https://speech-api.oruk.ai --model oruk-fourier-2b
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

MODEL_ID = "oruk/oruk-fourier-2b"
MODEL_REVISION = "8c2a8871f1304aa2f634da4fb95e24438c4945b9"
NATIVE_ROOT_FILES: tuple[str, ...] = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
EXPECTED_WEIGHT_BYTES = 4_076_193_080
EXPECTED_WEIGHT_SHA256 = (
    "e64765ffdd90c83704c0951e0d88b7f2ed84499af5c208f7ddf8d922cf08441c"
)
EXPECTED_TOKENIZER_SHA256 = (
    "bd2a97b55c8f7f9c328c73ee9b9178771037e9f566dfca8e238a063d41cbac92"
)

TARGET_SAMPLE_RATE = 16_000
MIN_CHUNK_SAMPLES = 8_000  # 0.5 s minimum expected by Qwen3ASRFeatureExtractor

# Excluded consultations matching PriMock57 benchmark conventions
EXCLUDED_FILES = {"day1_consultation07", "day3_consultation03"}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> tuple[int, str]:
    """Stream-hash a file on disk and return (byte_size, hex_sha256)."""
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
    return total, digest.hexdigest()


def verify_pinned_checkpoint(local_dir: Path) -> dict[str, Any]:
    """Verify all 7 native root checkpoint files and pinned SHA-256 digests."""
    missing = [name for name in NATIVE_ROOT_FILES if not (local_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing native checkpoint files in {local_dir}: {missing}"
        )

    weight_bytes, weight_sha = sha256_file(local_dir / "model.safetensors")
    if weight_bytes != EXPECTED_WEIGHT_BYTES or weight_sha != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(
            f"model.safetensors mismatch: bytes={weight_bytes} sha256={weight_sha}"
        )

    _, tok_sha = sha256_file(local_dir / "tokenizer.json")
    if tok_sha != EXPECTED_TOKENIZER_SHA256:
        raise RuntimeError(f"tokenizer.json SHA-256 mismatch: {tok_sha}")

    gen_cfg = json.loads((local_dir / "generation_config.json").read_text())
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "local_dir": str(local_dir),
        "weight_bytes": weight_bytes,
        "weight_sha256": weight_sha,
        "tokenizer_sha256": tok_sha,
        "num_beams": gen_cfg.get("num_beams"),
        "no_repeat_ngram_size": gen_cfg.get("no_repeat_ngram_size"),
        "bad_words_ids_count": len(gen_cfg.get("bad_words_ids") or []),
    }


def load_mono_16k(audio_path: str | Path) -> tuple[np.ndarray, float]:
    """Load an audio file as mono float32 at 16 kHz and return (samples, duration_s)."""
    wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    duration = float(wav.shape[0]) / float(sr)
    if sr != TARGET_SAMPLE_RATE:
        try:
            import soxr

            wav = soxr.resample(wav, sr, TARGET_SAMPLE_RATE).astype(np.float32, copy=False)
        except ImportError:
            from math import gcd
            from scipy.signal import resample_poly

            g = gcd(sr, TARGET_SAMPLE_RATE)
            wav = resample_poly(wav, TARGET_SAMPLE_RATE // g, sr // g).astype(
                np.float32, copy=False
            )
    return wav, duration


def chunk_audio_energy(
    wav: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    min_s: float = 20.0,
    max_s: float = 28.0,
    hard_max_s: float = 30.0,
    window_ms: float = 200.0,
    hop_ms: float = 20.0,
) -> list[np.ndarray]:
    """Deterministically split long audio at minimum-RMS energy pauses in [min_s, max_s].

    Preserves short clips (<= hard_max_s) intact, avoids splitting mid-word by
    choosing the quietest 200 ms RMS window inside each [min_s, max_s] search
    band, and right-pads any chunk shorter than 0.5 s to 8,000 samples.
    """
    n = int(wav.shape[0])
    if n == 0:
        return []

    hard_max_samples = int(round(hard_max_s * sr))
    min_samples = int(round(min_s * sr))
    max_samples = int(round(max_s * sr))
    win_samples = max(1, int(round(window_ms * sr / 1000.0)))
    hop_samples = max(1, int(round(hop_ms * sr / 1000.0)))

    chunks: list[np.ndarray] = []
    pos = 0
    while pos < n:
        remaining = n - pos
        if remaining <= hard_max_samples:
            seg = wav[pos:n]
            pos = n
        else:
            search_start = pos + min_samples
            search_end = min(n, pos + max_samples)
            if search_end - search_start <= win_samples:
                cut = search_end
            else:
                band = wav[search_start:search_end]
                # Fast prefix-sum RMS energy over sliding windows
                sq = np.empty(band.shape[0] + 1, dtype=np.float64)
                sq[0] = 0.0
                np.cumsum(band.astype(np.float64) ** 2, out=sq[1:])
                starts = np.arange(0, band.shape[0] - win_samples + 1, hop_samples)
                energies = (sq[starts + win_samples] - sq[starts]) / float(win_samples)
                best_idx = int(np.argmin(energies))
                cut = search_start + int(starts[best_idx]) + (win_samples // 2)
            seg = wav[pos:cut]
            pos = cut

        if seg.shape[0] < MIN_CHUNK_SAMPLES:
            seg = np.pad(seg, (0, MIN_CHUNK_SAMPLES - seg.shape[0]), mode="constant")
        chunks.append(np.ascontiguousarray(seg, dtype=np.float32))
    return chunks


def build_generation_kwargs(
    decode_mode: str = "published",
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Build generation overrides for published beam=3 vs greedy decoding."""
    if decode_mode == "published":
        kwargs: dict[str, Any] = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = int(max_new_tokens)
        return kwargs
    if decode_mode == "greedy":
        kwargs = {"do_sample": False, "num_beams": 1}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = int(max_new_tokens)
        return kwargs
    raise ValueError(f"Unsupported decode_mode: {decode_mode!r}")


def transcribe_via_transformers(
    audio_files: Sequence[str],
    transcripts_dir: Path,
    *,
    model_id: str,
    revision: str,
    verify_pin: bool,
    decode_mode: str,
    batch_size: int,
    max_new_tokens: int | None,
    min_chunk_s: float,
    max_chunk_s: float,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """Transcribe all consultations locally using native HF Transformers BF16."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    allow_patterns = list(NATIVE_ROOT_FILES) if model_id == MODEL_ID else None
    local_dir = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=allow_patterns,
        )
    )
    provenance: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "local_dir": str(local_dir),
    }
    if verify_pin and model_id == MODEL_ID:
        provenance = verify_pinned_checkpoint(local_dir)
        print(
            f"Verified pinned checkpoint {MODEL_ID}@{MODEL_REVISION[:8]} "
            f"(model.safetensors sha256={provenance['weight_sha256'][:16]}..., "
            f"num_beams={provenance['num_beams']}, "
            f"bad_words_ids={provenance['bad_words_ids_count']})",
            flush=True,
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    print(f"Loading {local_dir} on {device} ({dtype})...", flush=True)

    processor = AutoProcessor.from_pretrained(str(local_dir), local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = (
        AutoModelForMultimodalLM.from_pretrained(
            str(local_dir),
            local_files_only=True,
            dtype=dtype,
        )
        .to(device)
        .eval()
    )
    gen_kwargs = build_generation_kwargs(decode_mode=decode_mode, max_new_tokens=max_new_tokens)

    results: list[dict[str, Any]] = []
    skipped = 0
    total = len(audio_files)

    with torch.inference_mode():
        for i, audio_path in enumerate(audio_files, 1):
            fname = os.path.basename(audio_path)
            stem = fname.replace("_conversation.wav", "")
            if any(exc in stem for exc in EXCLUDED_FILES):
                print(f"  [{i}/{total}] Skipping (excluded): {fname}", flush=True)
                skipped += 1
                continue

            out_txt = transcripts_dir / f"{stem}_conversation_transcript.txt"
            if out_txt.exists() and not overwrite:
                print(f"  [{i}/{total}] Skipping (exists): {fname}", flush=True)
                skipped += 1
                continue

            t0 = time.perf_counter()
            wav, audio_dur = load_mono_16k(audio_path)
            chunks = chunk_audio_energy(
                wav,
                sr=TARGET_SAMPLE_RATE,
                min_s=min_chunk_s,
                max_s=max_chunk_s,
            )

            chunk_texts: list[str] = []
            for off in range(0, len(chunks), batch_size):
                batch_wavs = chunks[off : off + batch_size]
                inputs = processor.apply_transcription_request(
                    audio=batch_wavs,
                    language=["English"] * len(batch_wavs),
                    return_tensors="pt",
                    processor_kwargs={"padding": True},
                ).to(device, model.dtype)
                output_ids = model.generate(**inputs, **gen_kwargs)
                gen_ids = output_ids[:, inputs["input_ids"].shape[1] :]
                decoded = processor.decode(gen_ids, return_format="transcription_only")
                chunk_texts.extend(t.strip() for t in decoded if t and t.strip())

            full_text = " ".join(chunk_texts).strip()
            elapsed = time.perf_counter() - t0

            out_txt.write_text(full_text, encoding="utf-8")
            results.append(
                {
                    "audio_file": fname,
                    "duration": elapsed,
                    "audio_duration": audio_dur,
                    "num_chunks": len(chunks),
                    "text_length": len(full_text),
                }
            )
            print(
                f"  [{i}/{total}] Done: {fname} "
                f"({elapsed:.1f}s, {len(chunks)} chunks, {len(full_text.split())} words)",
                flush=True,
            )

    provenance["gpu_name"] = (
        torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    )
    provenance["decode_mode"] = decode_mode
    provenance["batch_size"] = batch_size
    provenance["chunking"] = {
        "method": "deterministic_rms_energy_min",
        "min_s": min_chunk_s,
        "max_s": max_chunk_s,
        "hard_max_s": 30.0,
    }
    return results, skipped, provenance


def transcribe_via_http(
    audio_files: Sequence[str],
    transcripts_dir: Path,
    *,
    server: str,
    api_model: str,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """Transcribe consultations via Oruk's OpenAI-compatible HTTP API."""
    import requests

    api_key = os.environ.get("ORUK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ORUK_API_KEY environment variable is required when using --backend http"
        )
    headers = {"Authorization": f"Bearer {api_key}"}
    endpoint = f"{server.rstrip('/')}/v1/audio/transcriptions"

    results: list[dict[str, Any]] = []
    skipped = 0
    total = len(audio_files)

    for i, audio_path in enumerate(audio_files, 1):
        fname = os.path.basename(audio_path)
        stem = fname.replace("_conversation.wav", "")
        if any(exc in stem for exc in EXCLUDED_FILES):
            print(f"  [{i}/{total}] Skipping (excluded): {fname}", flush=True)
            skipped += 1
            continue

        out_txt = transcripts_dir / f"{stem}_conversation_transcript.txt"
        if out_txt.exists() and not overwrite:
            print(f"  [{i}/{total}] Skipping (exists): {fname}", flush=True)
            skipped += 1
            continue

        wav_info = sf.info(audio_path)
        audio_dur = float(wav_info.frames) / float(wav_info.samplerate)
        t0 = time.perf_counter()
        with open(audio_path, "rb") as fp:
            resp = requests.post(
                endpoint,
                headers=headers,
                files={"file": (fname, fp, "audio/wav")},
                data={"model": api_model, "language": "en"},
                timeout=300,
            )
        resp.raise_for_status()
        text = resp.json().get("text", "").strip()
        elapsed = time.perf_counter() - t0

        out_txt.write_text(text, encoding="utf-8")
        results.append(
            {
                "audio_file": fname,
                "duration": elapsed,
                "audio_duration": audio_dur,
                "text_length": len(text),
            }
        )
        print(
            f"  [{i}/{total}] Done: {fname} ({elapsed:.1f}s, {len(text.split())} words)",
            flush=True,
        )

    return results, skipped, {"backend": "http", "server": server, "model": api_model}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe PriMock57 audio using Oruk Fourier 2B"
    )
    parser.add_argument(
        "--backend",
        choices=("transformers", "http"),
        default="transformers",
        help="transformers (default, local native HF BF16) or http (Oruk API)",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help=f"HF model ID (transformers) or API model ID (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--revision",
        default=MODEL_REVISION,
        help=f"Pinned HF revision (default: {MODEL_REVISION})",
    )
    parser.add_argument(
        "--server",
        default="https://speech-api.oruk.ai",
        help="Base URL when --backend http (default: https://speech-api.oruk.ai)",
    )
    parser.add_argument(
        "--tag",
        default="oruk-fourier-2b",
        help="Model tag used for output directory and metrics files",
    )
    parser.add_argument(
        "--decode-mode",
        choices=("published", "greedy"),
        default="published",
        help="published (checkpoint generation_config.json beam=3) or greedy",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Chunk batch size for local GPU inference (default: 16)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional override for max_new_tokens",
    )
    parser.add_argument(
        "--min-chunk-s",
        type=float,
        default=20.0,
        help="Minimum chunk length in seconds before energy-pause split (default: 20.0)",
    )
    parser.add_argument(
        "--max-chunk-s",
        type=float,
        default=28.0,
        help="Maximum chunk search boundary in seconds (default: 28.0)",
    )
    parser.add_argument(
        "--skip-weight-pin",
        action="store_true",
        help="Skip SHA-256 verification of pinned checkpoint weights",
    )
    parser.add_argument(
        "--audio-dir",
        default=None,
        help="Override audio directory (default: <repo>/data/raw_audio)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override results directory (default: <repo>/results)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-transcribe files even if transcript already exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of files for smoke testing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    base_dir = Path(__file__).resolve().parent.parent
    audio_dir = Path(args.audio_dir) if args.audio_dir else base_dir / "data" / "raw_audio"
    results_dir = Path(args.results_dir) if args.results_dir else base_dir / "results"
    transcripts_dir = results_dir / "transcripts" / args.tag
    metrics_dir = results_dir / "metrics"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(glob.glob(str(audio_dir / "*.wav")))
    if args.limit is not None:
        audio_files = audio_files[: args.limit]
    print(f"Found {len(audio_files)} audio files in {audio_dir}", flush=True)
    if not audio_files:
        print(f"No .wav files found in {audio_dir}", file=sys.stderr)
        return 1

    if args.backend == "transformers":
        results, skipped, provenance = transcribe_via_transformers(
            audio_files,
            transcripts_dir,
            model_id=args.model,
            revision=args.revision,
            verify_pin=not args.skip_weight_pin,
            decode_mode=args.decode_mode,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            min_chunk_s=args.min_chunk_s,
            max_chunk_s=args.max_chunk_s,
            overwrite=args.overwrite,
        )
    else:
        api_model = "oruk-fourier-2b" if args.model == MODEL_ID else args.model
        results, skipped, provenance = transcribe_via_http(
            audio_files,
            transcripts_dir,
            server=args.server,
            api_model=api_model,
            overwrite=args.overwrite,
        )

    if results:
        durations = [r["duration"] for r in results]
        total_audio = sum(r["audio_duration"] for r in results)
        total_wall = sum(durations)
        metrics = {
            "model_name": args.tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "provenance": provenance,
            "summary": {
                "total_files": len(results),
                "total_duration": total_wall,
                "average_duration": total_wall / len(results),
                "fastest_file": min(durations),
                "slowest_file": max(durations),
                "total_audio_duration": total_audio,
                "rtfx": (total_audio / total_wall) if total_wall > 0 else 0.0,
            },
            "file_details": results,
        }
        speed_path = metrics_dir / f"{args.tag}_speed.json"
        speed_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(
            f"\nSpeed metrics saved to: {speed_path} "
            f"(avg {metrics['summary']['average_duration']:.1f}s/file, "
            f"RTFx {metrics['summary']['rtfx']:.1f}x)",
            flush=True,
        )

    print(f"Done: {len(results)} processed, {skipped} skipped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
