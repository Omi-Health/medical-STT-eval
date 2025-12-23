#!/usr/bin/env python3
"""
OpenAI API transcriber for cloud-based Whisper transcription.
"""

import time
import os
from pathlib import Path
from typing import Optional

try:
    import openai
except ImportError:
    print("❌ OpenAI library not installed. Install with: pip install openai")
    exit(1)

from base_transcriber import BaseTranscriber, TranscriptionResult


class OpenAIAPITranscriber(BaseTranscriber):
    """OpenAI API transcriber for cloud-based Whisper."""
    
    def __init__(self, model_name: str = "whisper-1", api_key: str = None, results_dir: str = None):
        # Handle model name formatting for results directory
        display_name = model_name
        if model_name == "gpt-4o-mini-transcribe":
            display_name = "gpt-4o-mini-transcribe"
        
        super().__init__(f"openai-{display_name}", results_dir)
        
        # Set up OpenAI client
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model_name_internal = model_name
    
    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """
        Transcribe a single audio file using OpenAI API.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            TranscriptionResult object
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        start_time = time.time()
        
        try:
            # Open and read audio file
            with open(audio_file, 'rb') as audio:
                # Call OpenAI API
                response = self.client.audio.transcriptions.create(
                    model=self.model_name_internal,
                    file=audio,
                    response_format="text"
                )
            
            duration = time.time() - start_time
            text = response.strip() if isinstance(response, str) else str(response).strip()
            
            if not text:
                raise ValueError("Empty transcription result")
            
            return TranscriptionResult(
                text=text,
                duration=duration,
                model_name=self.model_name,
                audio_file=audio_file
            )
            
        except Exception as e:
            raise Exception(f"OpenAI API transcription failed: {e}")


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenAI API Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--model", default="whisper-1", help="OpenAI Whisper model name")
    parser.add_argument("--api_key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    
    args = parser.parse_args()
    
    # Initialize transcriber
    try:
        transcriber = OpenAIAPITranscriber(
            model_name=args.model,
            api_key=args.api_key,
            results_dir=args.results_dir
        )
    except ValueError as e:
        print(f"❌ {e}")
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