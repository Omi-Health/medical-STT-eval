#!/usr/bin/env python3
"""
Google MedASR transcriber for medical speech-to-text.
Run in medasr venv: source ~/.venvs/medasr/bin/activate
"""

import dataclasses
import re
import time
from pathlib import Path
from typing import Any, Optional

try:
    import soundfile as sf
    import transformers
except ImportError:
    print("Install MedASR deps in an isolated venv, including transformers, soundfile, kenlm, and pyctcdecode.")
    exit(1)

from base_transcriber import BaseTranscriber, TranscriptionResult


MODEL_ID = "google/medasr"
MANUAL_CHUNK_SECONDS = 8
MANUAL_OVERLAP_SECONDS = 1


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_audio(audio_file: str) -> tuple[Any, int]:
    audio, sample_rate = sf.read(audio_file, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sample_rate


def restore_lm_text(text: str) -> str:
    return text.replace(" ", "").replace("#", " ").replace("</s>", "").strip()


class LasrCtcBeamSearchDecoder:
    """Google notebook beam-search decoder wrapper."""

    def __init__(self, tokenizer: transformers.LasrTokenizer, kenlm_model_path: Optional[str] = None, **kwargs):
        import pyctcdecode

        vocab = [None for _ in range(tokenizer.vocab_size)]
        for token, idx in tokenizer.vocab.items():
            if idx < tokenizer.vocab_size:
                vocab[idx] = token
        assert not [item for item in vocab if item is None]
        vocab[0] = ""
        for i in range(1, len(vocab)):
            piece = vocab[i]
            if not piece.startswith("<") and not piece.endswith(">"):
                piece = "▁" + piece.replace("▁", "#")
            vocab[i] = piece
        self._decoder = pyctcdecode.build_ctcdecoder(vocab, kenlm_model_path, **kwargs)

    def decode_beams(self, *args, **kwargs):
        beams = self._decoder.decode_beams(*args, **kwargs)
        return [dataclasses.replace(item, text=restore_lm_text(item.text)) for item in beams]


class MedASRTranscriber(BaseTranscriber):
    """Google MedASR transcriber using LM-backed short-chunk decoding."""

    def __init__(self, model_name: str = "google-medasr", results_dir: str = None):
        super().__init__(model_name, results_dir)

        print("Loading MedASR model with KenLM decoder...")
        self.pipe = self._build_pipe()
        print("Model loaded.")

    def _build_pipe(self):
        import huggingface_hub

        feature_extractor = transformers.LasrFeatureExtractor.from_pretrained(MODEL_ID)
        feature_extractor._processor_class = "LasrProcessorWithLM"
        lm_path = huggingface_hub.hf_hub_download(MODEL_ID, filename="lm_6.kenlm")
        pipe = transformers.pipeline(
            task="automatic-speech-recognition",
            model=MODEL_ID,
            feature_extractor=feature_extractor,
            decoder=LasrCtcBeamSearchDecoder(transformers.AutoTokenizer.from_pretrained(MODEL_ID), lm_path),
        )
        assert pipe.type == "ctc_with_lm"
        return pipe

    def _transcribe_once(self, audio: Any, sample_rate: int) -> str:
        result = self.pipe(
            {"array": audio, "sampling_rate": sample_rate},
            chunk_length_s=MANUAL_CHUNK_SECONDS,
            stride_length_s=0,
        )
        return normalize_whitespace(result["text"])

    def _merge_by_word_overlap(self, existing: str, addition: str, max_overlap_words: int = 40) -> str:
        if not existing:
            return normalize_whitespace(addition)
        left_words = existing.split()
        right_words = addition.split()
        max_k = min(len(left_words), len(right_words), max_overlap_words)
        for k in range(max_k, 0, -1):
            if left_words[-k:] == right_words[:k]:
                return " ".join(left_words + right_words[k:])
        return normalize_whitespace(existing + " " + addition)

    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """
        Transcribe a single audio file using manual 8s chunks with 1s overlap.
        Short manual chunks reduce MedASR's deletion-heavy failure mode on dialogue.
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        start_time = time.time()

        try:
            audio, sample_rate = load_audio(audio_file)
            chunk_samples = int(MANUAL_CHUNK_SECONDS * sample_rate)
            step_samples = int((MANUAL_CHUNK_SECONDS - MANUAL_OVERLAP_SECONDS) * sample_rate)

            merged = ""
            start = 0
            while start < len(audio):
                end = min(len(audio), start + chunk_samples)
                segment = audio[start:end]
                text = self._transcribe_once(segment, sample_rate)
                merged = self._merge_by_word_overlap(merged, text)
                if end >= len(audio):
                    break
                start += step_samples

            duration = time.time() - start_time
            text = normalize_whitespace(merged)

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

    args = parser.parse_args()

    if not args.audio_dir and not args.single_file:
        parser.error("Either --audio_dir or --single_file must be specified")

    # Initialize transcriber
    transcriber = MedASRTranscriber(results_dir=args.results_dir)

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
