#!/usr/bin/env python3
"""
Google MedASR transcriber for medical speech-to-text.
Run in medasr venv: source ~/.venvs/medasr/bin/activate
"""

import time
from pathlib import Path
from typing import Optional

try:
    from transformers import pipeline
except ImportError:
    print("Install: pip install 'git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5'")
    exit(1)

from base_transcriber import BaseTranscriber, TranscriptionResult


class MedASRTranscriber(BaseTranscriber):
    """Google MedASR transcriber using pipeline API."""

    def __init__(self, model_name: str = "google-medasr", results_dir: str = None):
        super().__init__(model_name, results_dir)

        # Initialize pipeline (from notebook)
        print("Loading MedASR model...")
        self.pipe = pipeline("automatic-speech-recognition", model="google/medasr")
        print("Model loaded.")

    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """
        Transcribe a single audio file using MedASR pipeline.
        Uses chunk_length_s=20, stride_length_s=2 as per notebook.
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        start_time = time.time()

        try:
            # From notebook: chunk_length is how long MedASR segments audio
            # stride_length is the overlap between chunks
            result = self.pipe(
                audio_file,
                chunk_length_s=20,
                stride_length_s=2
            )

            duration = time.time() - start_time
            text = result['text'].strip()

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
