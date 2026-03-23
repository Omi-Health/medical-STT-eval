#!/usr/bin/env python3
"""
ElevenLabs Scribe v2 transcriber for speech-to-text.
Uses the batch REST API (same endpoint as v1, different model_id).
"""

import json
import time
import os
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from elevenlabs import ElevenLabs
except ImportError as e:
    import sys
    print(f"Failed to import ElevenLabs: {e}")
    print("Install with: pip install elevenlabs>=2.39.0")
    exit(1)

from base_transcriber import BaseTranscriber, TranscriptionResult


class ElevenLabsScribeV2Transcriber(BaseTranscriber):
    """ElevenLabs Scribe v2 transcriber with parallel batch processing."""

    EXCLUDED_FILES = ['day1_consultation07', 'day3_consultation03']

    def __init__(self, api_key: str = None, results_dir: str = None, max_workers: int = 8):
        super().__init__("elevenlabs-scribe_v2", results_dir)

        self.api_key = api_key or os.getenv('ELEVENLABS_API_KEY')
        if not self.api_key:
            raise ValueError("ElevenLabs API key required. Set ELEVENLABS_API_KEY environment variable or pass api_key parameter.")

        self.client = ElevenLabs(api_key=self.api_key)
        self.model_id = "scribe_v2"
        self.max_workers = max_workers

    def transcribe_batch(self, audio_files: List[str]) -> List['TranscriptionResult']:
        """Transcribe multiple audio files in parallel using thread pool."""
        # Filter out excluded and already-processed files
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

        # Save speed metrics (merge with existing on retry)
        self._save_metrics_merged(results)
        print(f"\nBatch complete: {len(results)}/{total} files processed")
        return results

    def _save_metrics_merged(self, new_results: List['TranscriptionResult']):
        """Save speed metrics, merging with existing file to preserve data from prior runs."""
        if not new_results:
            return

        metrics_file = self.metrics_dir / f"{self.model_name.replace('/', '_')}_speed.json"
        existing_details = []

        # Load existing metrics if present
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                existing = json.load(f)
            existing_details = existing.get('file_details', [])

        # Build set of existing audio files for dedup
        existing_names = {d['audio_file'] for d in existing_details}

        # Add new results
        for r in new_results:
            name = Path(r.audio_file).name
            if name not in existing_names:
                existing_details.append({
                    'audio_file': name,
                    'duration': r.duration,
                    'text_length': len(r.text)
                })

        # Recalculate summary from all data
        all_durations = [d['duration'] for d in existing_details]
        metrics = {
            'model_name': self.model_name,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'summary': {
                'total_files': len(existing_details),
                'total_duration': sum(all_durations),
                'average_duration': sum(all_durations) / len(all_durations),
                'fastest_file': min(all_durations),
                'slowest_file': max(all_durations),
            },
            'file_details': existing_details
        }

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"  Metrics saved to: {metrics_file} ({len(existing_details)} total files)")

    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """Transcribe a single audio file using ElevenLabs Scribe v2.
        Retries up to 3 times on rate limit (429) errors with exponential backoff.
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        start_time = time.time()
        max_retries = 3

        for attempt in range(max_retries + 1):
            try:
                with open(audio_file, 'rb') as audio:
                    response = self.client.speech_to_text.convert(
                        model_id=self.model_id,
                        file=audio,
                        language_code="en",
                    )
                break
            except Exception as e:
                if '429' in str(e) and attempt < max_retries:
                    wait = 10 * (2 ** attempt)
                    time.sleep(wait)
                    continue
                raise

        duration = time.time() - start_time

        # Extract text from response
        if hasattr(response, 'text'):
            text = response.text.strip()
        elif hasattr(response, 'transcript'):
            text = response.transcript.strip()
        elif isinstance(response, dict) and 'text' in response:
            text = response['text'].strip()
        elif isinstance(response, dict) and 'transcript' in response:
            text = response['transcript'].strip()
        else:
            text = str(response).strip()

        if not text:
            raise ValueError("Empty transcription result")

        return TranscriptionResult(
            text=text,
            duration=duration,
            model_name=self.model_name,
            audio_file=audio_file
        )


def main():
    """Main function for standalone usage."""
    import argparse

    parser = argparse.ArgumentParser(description="ElevenLabs Scribe v2 Transcription")
    parser.add_argument("--audio_dir", required=False, help="Directory containing audio files")
    parser.add_argument("--api_key", help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var)")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    parser.add_argument("--single_file", help="Process single file only (for testing)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")

    args = parser.parse_args()

    try:
        transcriber = ElevenLabsScribeV2Transcriber(
            api_key=args.api_key,
            results_dir=args.results_dir,
            max_workers=args.workers
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    if args.single_file:
        try:
            print(f"Testing ElevenLabs Scribe v2 on: {args.single_file}")
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
