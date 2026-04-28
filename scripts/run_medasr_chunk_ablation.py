#!/usr/bin/env python3
"""Run MedASR chunking/merge ablations and optionally score them.

Examples:
    python scripts/run_medasr_chunk_ablation.py --audio_dir data/raw_audio --max_files 3
    python scripts/run_medasr_chunk_ablation.py --audio_dir data/raw_audio --configs 8:1,15:3 --evaluate
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "transcribe"))
sys.path.insert(0, str(ROOT / "evaluate"))


EXCLUDED_FILES = {"day1_consultation07", "day3_consultation03"}


def parse_configs(value: str) -> list[tuple[float, float]]:
    configs: list[tuple[float, float]] = []
    for item in value.split(","):
        chunk_s, overlap_s = item.split(":", 1)
        configs.append((float(chunk_s), float(overlap_s)))
    return configs


def config_tag(mode: str, chunk_s: float | None = None, overlap_s: float | None = None) -> str:
    if mode == "hf_pipeline":
        return f"hf_{chunk_s:g}s_{overlap_s:g}s"
    return f"lm_{chunk_s:g}s_{overlap_s:g}s"


def get_audio_files(audio_dir: Path, pattern: str, max_files: int | None) -> list[str]:
    files = []
    for path in sorted(audio_dir.glob(pattern)):
        stem = path.stem
        if any(name in stem for name in EXCLUDED_FILES):
            continue
        files.append(str(path))
        if max_files is not None and len(files) >= max_files:
            break
    if not files:
        raise FileNotFoundError(f"No audio files found in {audio_dir} matching {pattern}")
    return files


def score_run(results_dir: Path, model_name: str, reference_dir: str | None) -> dict:
    from metrics_generator import ModelMetricsGenerator
    from medical_wer import evaluate_model

    resolved_reference_dir = reference_dir or str(ROOT / "data" / "cleaned_transcripts")
    generator = ModelMetricsGenerator(
        model_name=model_name,
        results_dir=str(results_dir),
        reference_dir=resolved_reference_dir,
    )
    wer = generator.generate_comprehensive_metrics()
    generator.save_metrics(wer)

    transcripts_dir = results_dir / "transcripts" / model_name.replace("/", "_")
    medical = evaluate_model(
        model_name,
        transcripts_dir=transcripts_dir,
        reference_dir=Path(resolved_reference_dir),
    )
    medical_file = results_dir / "metrics" / f"{model_name.replace('/', '_')}_medical_wer.json"
    with open(medical_file, "w", encoding="utf-8") as f:
        json.dump(medical, f, indent=2, ensure_ascii=False)

    return {
        "wer": wer.get("wer_metrics", {}).get("average_wer"),
        "m_wer": medical.get("global_m_wer"),
        "drug_m_wer": medical.get("global_high_risk_m_wer"),
    }


def run_variant(args, mode: str, chunk_s: float | None = None, overlap_s: float | None = None) -> dict:
    from medasr_transcribe import MedASRTranscriber

    effective_chunk_s = chunk_s or args.hf_chunk_seconds
    effective_overlap_s = overlap_s or args.hf_stride_seconds
    tag = config_tag(mode, effective_chunk_s, effective_overlap_s)
    results_dir = Path(args.output_dir) / tag
    results_dir.mkdir(parents=True, exist_ok=True)

    transcriber = MedASRTranscriber(
        results_dir=str(results_dir),
        decode_mode=mode,
        chunk_seconds=effective_chunk_s,
        overlap_seconds=effective_overlap_s,
        hf_chunk_seconds=args.hf_chunk_seconds,
        hf_stride_seconds=args.hf_stride_seconds,
        max_overlap_words=args.max_overlap_words,
        device=args.device,
    )

    files = get_audio_files(Path(args.audio_dir), args.pattern, args.max_files)
    started = time.time()
    results = transcriber.transcribe_batch(files)
    elapsed = time.time() - started

    row = {
        "tag": tag,
        "mode": mode,
        "chunk_seconds": effective_chunk_s,
        "overlap_seconds": effective_overlap_s,
        "files_requested": len(files),
        "files_processed": len(results),
        "elapsed_seconds": elapsed,
        "results_dir": str(results_dir),
    }
    if args.evaluate:
        row.update(score_run(results_dir, "google-medasr", args.reference_dir))

    del transcriber
    gc.collect()
    return row


def main():
    parser = argparse.ArgumentParser(description="Run MedASR chunking ablations")
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--output_dir", default="private_analysis/medasr_chunk_ablation")
    parser.add_argument("--reference_dir", default=None)
    parser.add_argument("--pattern", default="*_conversation.wav")
    parser.add_argument("--configs", default="8:1,15:3,20:5,30:8")
    parser.add_argument("--include_hf_baseline", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--hf_chunk_seconds", type=float, default=20.0)
    parser.add_argument("--hf_stride_seconds", type=float, default=2.0)
    parser.add_argument("--max_overlap_words", type=int, default=40)
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()

    summary = []
    if args.include_hf_baseline:
        summary.append(run_variant(args, "hf_pipeline"))

    for chunk_s, overlap_s in parse_configs(args.configs):
        summary.append(run_variant(args, "lm_short_chunks", chunk_s, overlap_s))

    summary_file = Path(args.output_dir) / "summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nAblation summary saved to: {summary_file}")
    for row in summary:
        metrics = ""
        if row.get("wer") is not None:
            metrics = (
                f" WER={row['wer'] * 100:.2f}%"
                f" M-WER={row['m_wer'] * 100:.2f}%"
                f" Drug={row['drug_m_wer'] * 100:.2f}%"
            )
        print(f"{row['tag']}: {row['files_processed']}/{row['files_requested']} files{metrics}")


if __name__ == "__main__":
    main()
