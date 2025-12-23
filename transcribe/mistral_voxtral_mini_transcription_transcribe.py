#!/usr/bin/env python3
"""
Mistral Voxtral Mini Transcription endpoint transcriber.
Uses the optimized transcription-only service.
"""

import time
import os
import json
import requests
from pathlib import Path
from typing import Optional

from base_transcriber import BaseTranscriber, TranscriptionResult


class MistralVoxtralMiniTranscriptionTranscriber(BaseTranscriber):
    """Mistral Voxtral Mini transcription endpoint transcriber."""
    
    def __init__(self, model_name: str = "voxtral-mini-latest", 
                 api_key: str = None, 
                 results_dir: str = None):
        # Use safe display name for directories
        display_name = f"mistral-{model_name}-transcription"
        super().__init__(display_name, results_dir)
        
        # Set up API configuration
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("Mistral API key required. Set MISTRAL_API_KEY environment variable or pass api_key parameter.")
        
        self.model_name_internal = model_name
        self.transcription_url = "https://api.mistral.ai/v1/audio/transcriptions"
    
    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """
        Transcribe a single audio file using Mistral Voxtral Mini transcription endpoint.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            TranscriptionResult object
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        start_time = time.time()
        
        try:
            # Use the transcription endpoint with file upload
            with open(audio_file, 'rb') as f:
                files = {
                    'file': (Path(audio_file).name, f, 'audio/wav')
                }
                data = {
                    'model': self.model_name_internal,
                    'language': 'en'
                }
                headers = {
                    'x-api-key': self.api_key
                }
                
                response = requests.post(
                    self.transcription_url,
                    headers=headers,
                    files=files,
                    data=data
                )
            
            duration = time.time() - start_time
            
            if response.status_code != 200:
                raise Exception(f"API error: Status {response.status_code}\n{response.text}")
            
            # Extract text from response
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
                
        except Exception as e:
            raise Exception(f"Mistral Voxtral Mini transcription failed: {e}")


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mistral Voxtral Mini Transcription Endpoint")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--model", default="voxtral-mini-latest", help="Mistral model name")
    parser.add_argument("--api_key", help="Mistral API key (or set MISTRAL_API_KEY env var)")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    parser.add_argument("--single_file", help="Process single file only (for testing)")
    
    args = parser.parse_args()
    
    # Initialize transcriber
    try:
        transcriber = MistralVoxtralMiniTranscriptionTranscriber(
            model_name=args.model,
            api_key=args.api_key,
            results_dir=args.results_dir
        )
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Process single file or batch
    if args.single_file:
        # Single file mode for testing
        try:
            print(f"🎤 Testing Mistral Voxtral Mini Transcription on: {args.single_file}")
            result = transcriber.transcribe_file(args.single_file)
            
            # Save transcript
            transcriber._save_transcript(result)
            
            print(f"✅ Success - Duration: {result.duration:.2f}s")
            print(f"📁 Transcript saved to: {transcriber.transcripts_dir}")
            
            # Show first 200 characters of transcript
            print(f"\n📝 First 200 chars of transcript:")
            print(result.text[:200] + "..." if len(result.text) > 200 else result.text)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return
    else:
        # Batch mode
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