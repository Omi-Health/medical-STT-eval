#!/usr/bin/env python3
"""
Google MedASR transcriber for medical speech-to-text.
Run in medasr venv: source ~/.venvs/medasr/bin/activate
"""

import dataclasses
import time
from pathlib import Path
from typing import Any, Optional

try:
    import soundfile as sf
    import transformers
except ImportError:
    print("Install MedASR deps in an isolated venv: transformers, soundfile, huggingface_hub, kenlm, pyctcdecode.")
    exit(1)

try:
    from base_transcriber import BaseTranscriber, TranscriptionResult
    from chunking_utils import (
        merge_by_exact_word_overlap,
        normalize_whitespace,
        split_audio_array,
        to_mono_float32,
    )
except ImportError:
    from transcribe.base_transcriber import BaseTranscriber, TranscriptionResult
    from transcribe.chunking_utils import (
        merge_by_exact_word_overlap,
        normalize_whitespace,
        split_audio_array,
        to_mono_float32,
    )


MODEL_ID = "google/medasr"
DEFAULT_MANUAL_CHUNK_SECONDS = 8.0
DEFAULT_MANUAL_OVERLAP_SECONDS = 1.0
DEFAULT_HF_CHUNK_SECONDS = 20.0
DEFAULT_HF_STRIDE_SECONDS = 2.0


def _restore_lm_text(text: str) -> str:
    """Undo the LASR LM tokenizer's temporary space marker formatting."""

    return text.replace(" ", "").replace("#", " ").replace("</s>", "").strip()


class LasrCtcBeamSearchDecoder:
    """KenLM-backed decoder wrapper matching the Google MedASR notebook path."""

    def __init__(self, tokenizer: Any, kenlm_model_path: Optional[str] = None, **kwargs):
        import pyctcdecode

        vocab = [None for _ in range(tokenizer.vocab_size)]
        for token, idx in tokenizer.vocab.items():
            if idx < tokenizer.vocab_size:
                vocab[idx] = token
        missing = [item for item in vocab if item is None]
        if missing:
            raise ValueError("Tokenizer vocab has missing ids; cannot build CTC decoder.")

        vocab[0] = ""
        for i in range(1, len(vocab)):
            piece = vocab[i]
            if not piece.startswith("<") and not piece.endswith(">"):
                piece = "\u2581" + piece.replace("\u2581", "#")
            vocab[i] = piece

        self._decoder = pyctcdecode.build_ctcdecoder(vocab, kenlm_model_path, **kwargs)

    def decode_beams(self, *args, **kwargs):
        beams = self._decoder.decode_beams(*args, **kwargs)
        restored_beams = []
        for item in beams:
            if hasattr(item, "text"):
                restored_beams.append(dataclasses.replace(item, text=_restore_lm_text(item.text)))
            elif isinstance(item, tuple) and item:
                restored_beams.append((_restore_lm_text(item[0]), *item[1:]))
            else:
                restored_beams.append(item)
        return restored_beams


class MedASRTranscriber(BaseTranscriber):
    """Google MedASR transcriber.

    Default mode uses the stronger LM-backed short-chunk path from PR #1.
    The old HF pipeline path remains available for baseline comparisons.
    """

    def __init__(
        self,
        model_name: str = "google-medasr",
        results_dir: str = None,
        *,
        decode_mode: str = "lm_short_chunks",
        chunk_seconds: float = DEFAULT_MANUAL_CHUNK_SECONDS,
        overlap_seconds: float = DEFAULT_MANUAL_OVERLAP_SECONDS,
        hf_chunk_seconds: float = DEFAULT_HF_CHUNK_SECONDS,
        hf_stride_seconds: float = DEFAULT_HF_STRIDE_SECONDS,
        max_overlap_words: int = 40,
        device: Optional[int] = None,
    ):
        super().__init__(model_name, results_dir)
        if decode_mode not in {"lm_short_chunks", "hf_pipeline"}:
            raise ValueError("decode_mode must be 'lm_short_chunks' or 'hf_pipeline'")

        self.decode_mode = decode_mode
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.hf_chunk_seconds = hf_chunk_seconds
        self.hf_stride_seconds = hf_stride_seconds
        self.max_overlap_words = max_overlap_words
        self.device = device

        print(f"Loading MedASR model ({self.decode_mode})...")
        self.pipe = self._build_pipe()
        print("Model loaded.")

    def _pipeline_kwargs(self) -> dict:
        if self.device is None:
            return {}
        return {"device": self.device}

    def _build_pipe(self):
        if self.decode_mode == "hf_pipeline":
            return transformers.pipeline(
                "automatic-speech-recognition",
                model=MODEL_ID,
                **self._pipeline_kwargs(),
            )

        import huggingface_hub

        feature_extractor = transformers.LasrFeatureExtractor.from_pretrained(MODEL_ID)
        feature_extractor._processor_class = "LasrProcessorWithLM"
        lm_path = huggingface_hub.hf_hub_download(MODEL_ID, filename="lm_6.kenlm")

        pipe = transformers.pipeline(
            task="automatic-speech-recognition",
            model=MODEL_ID,
            feature_extractor=feature_extractor,
            decoder=LasrCtcBeamSearchDecoder(transformers.AutoTokenizer.from_pretrained(MODEL_ID), lm_path),
            **self._pipeline_kwargs(),
        )
        if pipe.type != "ctc_with_lm":
            raise RuntimeError(f"Expected ctc_with_lm pipeline, got {pipe.type!r}")
        return pipe

    def _transcribe_hf_pipeline(self, audio_file: str) -> str:
        result = self.pipe(
            audio_file,
            chunk_length_s=self.hf_chunk_seconds,
            stride_length_s=self.hf_stride_seconds,
        )
        return normalize_whitespace(result["text"])

    def _transcribe_lm_chunk(self, audio: Any, sample_rate: int) -> str:
        result = self.pipe(
            {"array": audio, "sampling_rate": sample_rate},
            chunk_length_s=self.chunk_seconds,
            stride_length_s=0,
        )
        return normalize_whitespace(result["text"])

    def _transcribe_lm_short_chunks(self, audio_file: str) -> str:
        audio, sample_rate = sf.read(audio_file, dtype="float32")
        chunks = split_audio_array(
            to_mono_float32(audio),
            sample_rate,
            self.chunk_seconds,
            self.overlap_seconds,
            min_final_seconds=1.0,
            fade_seconds=0.0,
        )

        merged = ""
        for chunk in chunks:
            text = self._transcribe_lm_chunk(chunk.audio, chunk.sample_rate)
            merged = merge_by_exact_word_overlap(merged, text, max_overlap_words=self.max_overlap_words)
        return normalize_whitespace(merged)

    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """
        Transcribe a single audio file using the configured MedASR decode path.
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        start_time = time.time()

        try:
            if self.decode_mode == "hf_pipeline":
                text = self._transcribe_hf_pipeline(audio_file)
            else:
                text = self._transcribe_lm_short_chunks(audio_file)

            duration = time.time() - start_time

            if not text:
                raise ValueError("Empty transcription result")

            return TranscriptionResult(
                text=text,
                duration=duration,
                model_name=self.model_name,
                audio_file=audio_file
            )

        except Exception as e:
            raise Exception(f"MedASR transcription failed: {e}")


def main():
    """Main function for standalone usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Google MedASR Transcription")
    parser.add_argument("--audio_dir", help="Directory containing audio files")
    parser.add_argument("--single_file", help="Process single file only (for testing)")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    parser.add_argument(
        "--decode_mode",
        choices=["lm_short_chunks", "hf_pipeline"],
        default="lm_short_chunks",
        help="MedASR decode path. lm_short_chunks is the stronger KenLM-backed short-chunk mode.",
    )
    parser.add_argument("--chunk_seconds", type=float, default=DEFAULT_MANUAL_CHUNK_SECONDS)
    parser.add_argument("--overlap_seconds", type=float, default=DEFAULT_MANUAL_OVERLAP_SECONDS)
    parser.add_argument("--hf_chunk_seconds", type=float, default=DEFAULT_HF_CHUNK_SECONDS)
    parser.add_argument("--hf_stride_seconds", type=float, default=DEFAULT_HF_STRIDE_SECONDS)
    parser.add_argument("--max_overlap_words", type=int, default=40)
    parser.add_argument("--device", type=int, default=None, help="Optional transformers pipeline device id, e.g. 0 for CUDA")

    args = parser.parse_args()

    if not args.audio_dir and not args.single_file:
        parser.error("Either --audio_dir or --single_file must be specified")

    # Initialize transcriber
    transcriber = MedASRTranscriber(
        results_dir=args.results_dir,
        decode_mode=args.decode_mode,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        hf_chunk_seconds=args.hf_chunk_seconds,
        hf_stride_seconds=args.hf_stride_seconds,
        max_overlap_words=args.max_overlap_words,
        device=args.device,
    )

    # Process single file or batch
    if args.single_file:
        try:
            print(f"Testing MedASR on: {args.single_file}")
            result = transcriber.transcribe_file(args.single_file)
            transcriber._save_transcript(result)

            print(f"Duration: {result.duration:.2f}s")
            print(f"Transcript saved to: {transcriber.transcripts_dir}")
            print(f"\nFirst 300 chars:\n{result.text[:300]}...")

        except Exception as e:
            print(f"Error: {e}")
    else:
        # Batch mode
        try:
            audio_files = transcriber.get_audio_files(args.audio_dir, args.pattern)
            print(f"Found {len(audio_files)} audio files")
        except FileNotFoundError as e:
            print(f"{e}")
            return

        results = transcriber.transcribe_batch(audio_files)

        if results:
            print(f"\nProcessed {len(results)} files")
            print(f"Transcripts: {transcriber.transcripts_dir}")
            print(f"Metrics: {transcriber.metrics_dir}")


if __name__ == "__main__":
    main()
