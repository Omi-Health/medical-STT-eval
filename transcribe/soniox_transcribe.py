#!/usr/bin/env python3
"""
Soniox stt-async-v4 transcriber with parallel batch processing.

Clean API call (no context customization / no medical terms / no diarization)
to keep the comparison fair with the other plain cloud-API entries.
"""

import os
import time
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from base_transcriber import BaseTranscriber, TranscriptionResult


class SonioxTranscriber(BaseTranscriber):
    """Soniox stt-async-v4 transcriber via REST API."""

    BASE_URL = "https://api.soniox.com"
    EXCLUDED_FILES = ['day1_consultation07', 'day3_consultation03']
    POLL_INTERVAL = 0.5

    def __init__(self,
                 model_name: str = "stt-async-v4",
                 api_key: str = None,
                 results_dir: str = None,
                 max_workers: int = 2):
        display_name = f"soniox-{model_name}"
        super().__init__(display_name, results_dir)

        self.api_key = api_key or os.getenv('SONIOX_API_KEY')
        if not self.api_key:
            raise ValueError("Soniox API key required. Set SONIOX_API_KEY.")
        self.model_internal = model_name
        self.max_workers = max_workers
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def _upload_file(self, audio_file: str) -> str:
        with open(audio_file, 'rb') as f:
            response = requests.post(
                f"{self.BASE_URL}/v1/files",
                headers=self.headers,
                files={"file": (Path(audio_file).name, f)},
                timeout=600,
            )
        if response.status_code not in (200, 201):
            raise Exception(f"Upload failed {response.status_code}: {response.text}")
        return response.json()["id"]

    def _create_transcription(self, file_id: str) -> str:
        body = {
            "model": self.model_internal,
            "file_id": file_id,
            "language_hints": ["en"],
        }
        response = requests.post(
            f"{self.BASE_URL}/v1/transcriptions",
            headers={**self.headers, "Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
        if response.status_code not in (200, 201):
            raise Exception(f"Create transcription failed {response.status_code}: {response.text}")
        return response.json()["id"]

    def _wait(self, transcription_id: str) -> None:
        url = f"{self.BASE_URL}/v1/transcriptions/{transcription_id}"
        while True:
            r = requests.get(url, headers=self.headers, timeout=60)
            if r.status_code != 200:
                raise Exception(f"Poll failed {r.status_code}: {r.text}")
            data = r.json()
            status = data.get("status")
            if status == "completed":
                return
            if status == "error":
                raise RuntimeError(f"Transcription failed: {data.get('error_message') or data}")
            time.sleep(self.POLL_INTERVAL)

    def _get_transcript(self, transcription_id: str) -> str:
        r = requests.get(
            f"{self.BASE_URL}/v1/transcriptions/{transcription_id}/transcript",
            headers=self.headers,
            timeout=120,
        )
        if r.status_code != 200:
            raise Exception(f"Get transcript failed {r.status_code}: {r.text}")
        data = r.json()
        # Prefer assembled text if present, else concatenate token texts.
        if isinstance(data.get("text"), str) and data["text"].strip():
            return data["text"].strip()
        tokens = data.get("tokens", [])
        return "".join(t.get("text", "") for t in tokens).strip()

    def _delete(self, transcription_id: str, file_id: str) -> None:
        try:
            requests.delete(
                f"{self.BASE_URL}/v1/transcriptions/{transcription_id}",
                headers=self.headers, timeout=30,
            )
        except Exception:
            pass
        try:
            requests.delete(
                f"{self.BASE_URL}/v1/files/{file_id}",
                headers=self.headers, timeout=30,
            )
        except Exception:
            pass

    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        start_time = time.time()
        file_id = self._upload_file(audio_file)
        transcription_id = None
        try:
            transcription_id = self._create_transcription(file_id)
            self._wait(transcription_id)
            text = self._get_transcript(transcription_id)
        finally:
            if transcription_id:
                self._delete(transcription_id, file_id)
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

    parser = argparse.ArgumentParser(description="Soniox stt-async-v4 Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--model", default="stt-async-v4", help="Soniox model name")
    parser.add_argument("--api_key", help="Soniox API key (or set SONIOX_API_KEY)")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files (0 = all)")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers")

    args = parser.parse_args()

    transcriber = SonioxTranscriber(
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
