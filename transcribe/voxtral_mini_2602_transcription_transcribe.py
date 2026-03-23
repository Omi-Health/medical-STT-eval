#!/usr/bin/env python3
"""
Voxtral Mini 2602 transcription endpoint transcriber.
Uses Mistral's /v1/audio/transcriptions API with voxtral-mini-latest (points to voxtral-mini-2602).
Supports parallel batch processing.
"""

import json
import time
import os
import requests
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from base_transcriber import BaseTranscriber, TranscriptionResult


class VoxtralMini2602TranscriptionTranscriber(BaseTranscriber):
    """Voxtral Mini 2602 via Mistral transcription API with parallel processing."""

    EXCLUDED_FILES = ['day1_consultation07', 'day3_consultation03']

    def __init__(self, api_key: str = None, results_dir: str = None, max_workers: int = 4):
        super().__init__("voxtral-mini-2602-transcription", results_dir)

        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("Mistral API key required. Set MISTRAL_API_KEY environment variable or pass api_key parameter.")

        self.transcription_url = "https://api.mistral.ai/v1/audio/transcriptions"
        self.max_workers = max_workers

    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """Transcribe a single audio file using Mistral transcription API.
        Retries up to 3 times on 429/503 errors with exponential backoff.
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        start_time = time.time()
        max_retries = 3

        for attempt in range(max_retries + 1):
            try:
                with open(audio_file, 'rb') as f:
                    response = requests.post(
                        self.transcription_url,
                        headers={'x-api-key': self.api_key},
                        files={'file': (Path(audio_file).name, f, 'audio/wav')},
                        data={'model': 'voxtral-mini-latest', 'language': 'en'},
                    )

                if response.status_code == 429 or response.status_code == 503:
                    if attempt < max_retries:
                        wait = 10 * (2 ** attempt)
                        time.sleep(wait)
                        continue
                    response.raise_for_status()

                if response.status_code != 200:
                    raise Exception(f"API error: Status {response.status_code}\n{response.text}")

                break
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    time.sleep(10 * (2 ** attempt))
                    continue
                raise

        duration = time.time() - start_time
        result = response.json()
        text = result.get('text', '').strip()

        if not text:
            raise ValueError("Empty transcription result")

        return TranscriptionResult(
            text=text,
            duration=duration,
            model_name=self.model_name,
            audio_file=audio_file
        )

    def transcribe_batch(self, audio_files: List[str]) -> List[TranscriptionResult]:
        """Transcribe multiple audio files in parallel."""
        to_process = []
        skipped = 0
        for audio_file in audio_files:
            audio_name = Path(audio_file).stem
            if any(exc in audio_name for exc in self.EXCLUDED_FILES):
                print(f"  Skipping (excluded): {Path(audio_file).name}")
                skipped += 1
                continue
            transcript_file = self.transcripts_dir / f"{audio_name}_transcript.txt"
            if transcript_file.exists():
                print(f"  Skipping (exists): {Path(audio_file).name}")
                skipped += 1
                continue
            to_process.append(audio_file)

        total = len(to_process)
        print(f"\nStarting parallel transcription with {self.model_name}")
        print(f"  Files to process: {total} (skipped {skipped})")
        print(f"  Workers: {self.max_workers}")
        print(f"  Output: {self.transcripts_dir}\n")

        if not to_process:
            print("Nothing to process.")
            return []

        results = []
        errors = []

        def _process_one(audio_file: str) -> Optional[TranscriptionResult]:
            result = self.transcribe_file(audio_file)
            self._save_transcript(result)
            return result

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(_process_one, f): f for f in to_process
            }
            for future in as_completed(future_to_file):
                audio_file = future_to_file[future]
                name = Path(audio_file).name
                try:
                    result = future.result()
                    results.append(result)
                    done = len(results) + len(errors)
                    print(f"  [{done}/{total}] Done: {name} ({result.duration:.1f}s)")
                except Exception as e:
                    errors.append((name, str(e)))
                    done = len(results) + len(errors)
                    print(f"  [{done}/{total}] FAILED: {name} - {e}")

        if errors:
            print(f"\n  {len(errors)} failures:")
            for name, err in errors:
                print(f"    {name}: {err}")

        self._save_metrics(results)
        print(f"\nBatch complete: {len(results)}/{total} files processed")
        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Voxtral Mini 2602 Transcription")
    parser.add_argument("--audio_dir", required=False, help="Directory containing audio files")
    parser.add_argument("--api_key", help="Mistral API key (or set MISTRAL_API_KEY env var)")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    parser.add_argument("--single_file", help="Process single file only (for testing)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")

    args = parser.parse_args()

    try:
        transcriber = VoxtralMini2602TranscriptionTranscriber(
            api_key=args.api_key,
            results_dir=args.results_dir,
            max_workers=args.workers
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    if args.single_file:
        try:
            print(f"Testing on: {args.single_file}")
            result = transcriber.transcribe_file(args.single_file)
            transcriber._save_transcript(result)
            print(f"Success - Duration: {result.duration:.2f}s")
            print(f"Transcript saved to: {transcriber.transcripts_dir}")
            print(f"\nFirst 200 chars:")
            print(result.text[:200] + "..." if len(result.text) > 200 else result.text)
        except Exception as e:
            print(f"Error: {e}")
    elif args.audio_dir:
        try:
            audio_files = transcriber.get_audio_files(args.audio_dir, args.pattern)
            print(f"Found {len(audio_files)} audio files")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

        results = transcriber.transcribe_batch(audio_files)
        if results:
            print(f"\nSuccessfully processed {len(results)} files")
            print(f"Transcripts saved to: {transcriber.transcripts_dir}")
            print(f"Metrics saved to: {transcriber.metrics_dir}")
        else:
            print("\nNo files were processed successfully")
    else:
        parser.error("Either --audio_dir or --single_file is required")


if __name__ == "__main__":
    main()
