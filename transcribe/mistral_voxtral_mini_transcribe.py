#!/usr/bin/env python3
"""
Mistral Voxtral Mini transcriber for speech-to-text.
"""

import time
import os
import json
import base64
from pathlib import Path
from typing import Optional

try:
    from mistralai import Mistral
except ImportError:
    print("❌ Mistral library not installed. Install with: pip install mistralai")
    exit(1)

from base_transcriber import BaseTranscriber, TranscriptionResult


class MistralVoxtralMiniTranscriber(BaseTranscriber):
    """Mistral Voxtral Mini transcriber."""
    
    def __init__(self, model_name: str = "voxtral-mini-latest", 
                 api_key: str = None, 
                 results_dir: str = None):
        # Use safe display name for directories
        display_name = f"mistral-{model_name}"
        super().__init__(display_name, results_dir)
        
        # Set up Mistral client
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("Mistral API key required. Set MISTRAL_API_KEY environment variable or pass api_key parameter.")
        
        self.client = Mistral(api_key=self.api_key)
        self.model_name_internal = model_name
    
    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """
        Transcribe a single audio file using Mistral Voxtral Mini.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            TranscriptionResult object
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        start_time = time.time()
        
        try:
            # Use the correct Mistral audio transcription API
            with open(audio_file, 'rb') as f:
                transcription_response = self.client.audio.transcriptions.complete(
                    model=self.model_name_internal,
                    file={
                        "content": f,
                        "file_name": Path(audio_file).name,
                    },
                    language="en"
                )
            
            duration = time.time() - start_time
            
            # Extract text from response
            text = self._extract_text(transcription_response)
            
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
    
    def _extract_text(self, response) -> str:
        """Extract text from various response formats."""
        if hasattr(response, 'text'):
            return response.text.strip()
        elif hasattr(response, 'transcript'):
            return response.transcript.strip()
        elif hasattr(response, 'content'):
            return response.content.strip()
        elif isinstance(response, str):
            return response.strip()
        elif isinstance(response, dict):
            # Try various possible fields
            for field in ['text', 'transcript', 'content', 'result']:
                if field in response:
                    return str(response[field]).strip()
        return ""


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mistral Voxtral Mini Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--model", default="voxtral-mini-latest", help="Mistral model name")
    parser.add_argument("--api_key", help="Mistral API key (or set MISTRAL_API_KEY env var)")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    parser.add_argument("--single_file", help="Process single file only (for testing)")
    
    args = parser.parse_args()
    
    # Initialize transcriber
    try:
        transcriber = MistralVoxtralMiniTranscriber(
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
            print(f"🎤 Testing Mistral Voxtral Mini on: {args.single_file}")
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