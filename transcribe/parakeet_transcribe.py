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
        Transcribe a single audio file using Parakeet.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            TranscriptionResult object
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        start_time = time.time()
        
        try:
            # Create temporary output directory - mlx-audio will create filename.txt inside this dir
            import tempfile
            temp_dir = tempfile.mkdtemp()
            audio_basename = Path(audio_file).stem
            
            # Use mlx-audio command format: python -m mlx_audio.stt.generate --model MODEL --audio AUDIO --output OUTPUT_DIR
            cmd = [
                'python', '-m', 'mlx_audio.stt.generate', 
                '--model', self.model_name_internal,
                '--audio', audio_file,
                '--output', temp_dir + '/' + audio_basename
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise Exception(f"Parakeet command failed: {result.stderr}")
            
            duration = time.time() - start_time
            
            # The transcript should be written to temp_dir/audio_basename.txt
            output_file = Path(temp_dir) / f"{audio_basename}.txt"
            if output_file.exists():
                with open(output_file, 'r') as f:
                    text = f.read().strip()
                # Clean up temp files
                output_file.unlink()
                Path(temp_dir).rmdir()
            else:
                raise ValueError(f"Output file not created: {output_file}")
            
            if not text:
                raise ValueError("Empty transcription result from output file")
            
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