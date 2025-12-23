#!/usr/bin/env python3
"""
Parakeet TDT transcriber for MLX.
Note: Requires parakeet-mlx to be installed in the conda environment.
"""

import time
import subprocess
from pathlib import Path
from typing import Optional

from base_transcriber import BaseTranscriber, TranscriptionResult


class ParakeetTranscriber(BaseTranscriber):
    """Parakeet TDT transcriber."""
    
    def __init__(self, model_name: str = "mlx-community/parakeet-tdt-0.6b-v2", results_dir: str = None):
        super().__init__(f"parakeet-{model_name.split('/')[-1]}", results_dir)
        self.model_name_internal = model_name
        
        # Note: Skipping availability check since mlx-audio is confirmed working
    
    def _check_parakeet(self) -> bool:
        """Check if mlx-audio is available."""
        try:
            result = subprocess.run(['python', '-m', 'mlx_audio.stt.generate', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """
        Transcribe a single audio file using parakeet-mlx CLI.

        Args:
            audio_file: Path to audio file

        Returns:
            TranscriptionResult object
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        start_time = time.time()
        audio_stem = Path(audio_file).stem

        try:
            # Use parakeet-mlx CLI (creates .srt file in current directory)
            cmd = [
                'parakeet-mlx',
                audio_file,
                '--model', self.model_name_internal
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for long files
            )

            if result.returncode != 0:
                raise Exception(f"parakeet-mlx failed: {result.stderr}")

            duration = time.time() - start_time

            # parakeet-mlx creates .srt file - extract text from it
            srt_file = Path(f"{audio_stem}.srt")
            if not srt_file.exists():
                raise ValueError(f"SRT file not created: {srt_file}")

            # Parse SRT and extract text (skip timestamps and sequence numbers)
            text_lines = []
            with open(srt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines, sequence numbers, and timestamp lines
                    if not line:
                        continue
                    if line.isdigit():
                        continue
                    if '-->' in line:
                        continue
                    text_lines.append(line)

            text = ' '.join(text_lines)

            # Clean up .srt file
            srt_file.unlink()

            if not text:
                raise ValueError("Empty transcription result")

            return TranscriptionResult(
                text=text,
                duration=duration,
                model_name=self.model_name,
                audio_file=audio_file
            )

        except subprocess.TimeoutExpired:
            raise Exception("Parakeet transcription timed out")
        except Exception as e:
            raise Exception(f"Parakeet transcription failed: {e}")


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parakeet TDT Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v2", help="Parakeet model name")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    
    args = parser.parse_args()
    
    # Initialize transcriber
    try:
        transcriber = ParakeetTranscriber(
            model_name=args.model,
            results_dir=args.results_dir
        )
    except RuntimeError as e:
        print(f"❌ {e}")
        print("📋 Installation instructions:")
        print("   1. Activate your conda environment: conda activate adapter-training")
        print("   2. Install mlx-audio: pip install mlx-audio")
        print("   3. Verify installation: python -m mlx_audio.stt.generate --help")
        return
    
    # Get audio files
    try:
        audio_files = transcriber.get_audio_files(args.audio_dir, args.pattern)
        print(f"Found {len(audio_files)} audio files")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Process files
    results = transcriber.transcribe_batch(audio_files)
    
    if results:
        print(f"\n✅ Successfully processed {len(results)} files")
        print(f"📁 Transcripts saved to: {transcriber.transcripts_dir}")
        print(f"📊 Metrics saved to: {transcriber.metrics_dir}")
    else:
        print("\n❌ No files were processed successfully")


if __name__ == "__main__":
    main()