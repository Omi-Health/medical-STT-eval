#!/usr/bin/env python3
"""
AssemblyAI Universal-3 Pro (medical-v1 domain) transcriber with parallel batch processing.
"""

import os
import time
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from base_transcriber import BaseTranscriber, TranscriptionResult


class AssemblyAITranscriber(BaseTranscriber):
    """AssemblyAI universal-3-pro transcriber, medical-v1 domain."""

    BASE_URL = "https://api.assemblyai.com"
    EXCLUDED_FILES = ['day1_consultation07', 'day3_consultation03']

    def __init__(self,
                 model_name: str = "universal-3-pro",
                 api_key: str = None,
                 results_dir: str = None,
                 max_workers: int = 4):
        display_name = f"assemblyai-{model_name}-medical"
        super().__init__(display_name, results_dir)

        self.api_key = api_key or os.getenv('ASSEMBLYAI_API_KEY')
        if not self.api_key:
            raise ValueError("AssemblyAI API key required. Set ASSEMBLYAI_API_KEY.")
        self.model_internal = model_name
        self.max_workers = max_workers
        self.headers = {"authorization": self.api_key}

    def _upload_file(self, audio_file: str) -> str:
        with open(audio_file, 'rb') as f:
            response = requests.post(
                f"{self.BASE_URL}/v2/upload",
                headers=self.headers,
                data=f.read(),
                timeout=600,
            )
        if response.status_code != 200:
            raise Exception(f"Upload failed {response.status_code}: {response.text}")
        return response.json()["upload_url"]

    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        start_time = time.time()

        upload_url = self._upload_file(audio_file)

        config = {
            "audio_url": upload_url,
            "speech_models": [self.model_internal],
            "speaker_labels": True,
            "language_detection": True,
            "temperature": 0,
            "domain": "medical-v1",
        }
        response = requests.post(
            f"{self.BASE_URL}/v2/transcript",
            json=config,
            headers=self.headers,
            timeout=60,
        )
        if response.status_code != 200:
            raise Exception(f"Submit failed {response.status_code}: {response.text}")

        transcript_id = response.json()["id"]
        polling_endpoint = f"{self.BASE_URL}/v2/transcript/{transcript_id}"

        while True:
            poll = requests.get(polling_endpoint, headers=self.headers, timeout=60).json()
            status = poll.get("status")
            if status == "completed":
                text = (poll.get("text") or "").strip()
                break
            if status == "error":
                raise RuntimeError(f"Transcription failed: {poll.get('error')}")
            time.sleep(3)

        duration = time.time() - start_time

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

    parser = argparse.ArgumentParser(description="AssemblyAI Universal-3 Pro Medical Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--model", default="universal-3-pro", help="AssemblyAI speech model")
    parser.add_argument("--api_key", help="AssemblyAI API key (or set ASSEMBLYAI_API_KEY)")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files (0 = all)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")

    args = parser.parse_args()

    transcriber = AssemblyAITranscriber(
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
