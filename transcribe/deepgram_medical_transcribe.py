#!/usr/bin/env python3
"""
Deepgram Nova-3 Medical transcriber with parallel batch processing.
"""

import os
import time
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from base_transcriber import BaseTranscriber, TranscriptionResult


class DeepgramMedicalTranscriber(BaseTranscriber):
    """Deepgram nova-3-medical transcriber."""

    API_URL = "https://api.deepgram.com/v1/listen"
    EXCLUDED_FILES = ['day1_consultation07', 'day3_consultation03']

    def __init__(self, model_name: str = "nova-3-medical",
                 api_key: str = None,
                 results_dir: str = None,
                 max_workers: int = 4):
        display_name = f"deepgram-{model_name}"
        super().__init__(display_name, results_dir)

        self.api_key = api_key or os.getenv('DEEPGRAM_API_KEY')
        if not self.api_key:
            raise ValueError("Deepgram API key required. Set DEEPGRAM_API_KEY.")
        self.model_internal = model_name
        self.max_workers = max_workers

    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        params = {
            "smart_format": "true",
            "language": "en",
            "model": self.model_internal,
        }
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav",
        }

        start_time = time.time()
        with open(audio_file, 'rb') as f:
            response = requests.post(
                self.API_URL,
                params=params,
                headers=headers,
                data=f.read(),
                timeout=600,
            )
        duration = time.time() - start_time

        if response.status_code != 200:
            raise Exception(f"Deepgram API error {response.status_code}: {response.text}")

        data = response.json()
        try:
            text = data["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected Deepgram response shape: {e} | {data}")

        if not text:
            raise ValueError("Empty transcription result")

        return TranscriptionResult(
            text=text,
            duration=duration,
            model_name=self.model_name,
            audio_file=audio_file,
        )

    def transcribe_batch(self, audio_files: List[str]) -> List[TranscriptionResult]:
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

    parser = argparse.ArgumentParser(description="Deepgram Nova-3 Medical Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--model", default="nova-3-medical", help="Deepgram model name")
    parser.add_argument("--api_key", help="Deepgram API key (or set DEEPGRAM_API_KEY)")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files (0 = all)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")

    args = parser.parse_args()

    transcriber = DeepgramMedicalTranscriber(
        model_name=args.model,
        api_key=args.api_key,
        results_dir=args.results_dir,
        max_workers=args.workers,
    )

    audio_files = transcriber.get_audio_files(args.audio_dir, args.pattern)
    if args.limit > 0:
        audio_files = audio_files[:args.limit]
    print(f"Found {len(audio_files)} audio files")

    results = transcriber.transcribe_batch(audio_files)

    if results:
        print(f"\n✅ Successfully processed {len(results)} files")
        print(f"📁 Transcripts: {transcriber.transcripts_dir}")
    else:
        print("\n❌ No files processed")


if __name__ == "__main__":
    main()
