#!/usr/bin/env python3
"""
MLX Whisper transcriber for Apple Silicon optimized transcription.
"""

import time
from pathlib import Path
from typing import Optional

try:
    import mlx_whisper
except ImportError:
    print("❌ MLX Whisper not installed. Install with: pip install mlx-whisper")
    exit(1)

from base_transcriber import BaseTranscriber, TranscriptionResult


class MLXWhisperTranscriber(BaseTranscriber):
    """MLX Whisper transcriber for Apple Silicon."""
    
    def __init__(self, model_name: str = "mlx-community/whisper-large-v3-turbo", results_dir: str = None):
        super().__init__(model_name, results_dir)
        self.model_loaded = False
    
    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """
        Transcribe a single audio file using MLX Whisper.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            TranscriptionResult object
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        start_time = time.time()
        
        try:
            # Transcribe using MLX Whisper
            result = mlx_whisper.transcribe(
                audio_file,
                path_or_hf_repo=self.model_name,
            )
            
            duration = time.time() - start_time
            text = result.get('text', '').strip()
            
            if not text:
                raise ValueError("Empty transcription result")
            
            return TranscriptionResult(
                text=text,
                duration=duration,
                model_name=self.model_name,
                audio_file=audio_file
            )
            
        except Exception as e:
            raise Exception(f"MLX Whisper transcription failed: {e}")


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MLX Whisper Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-turbo", help="MLX Whisper model name")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    
    args = parser.parse_args()
    
    # Initialize transcriber
    transcriber = MLXWhisperTranscriber(
        model_name=args.model,
        results_dir=args.results_dir
    )
    
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