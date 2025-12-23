#!/usr/bin/env python3
"""
ElevenLabs Scribe transcriber for speech-to-text.
"""

import time
import os
from pathlib import Path
from typing import Optional

try:
    from elevenlabs import ElevenLabs
except ImportError as e:
    import sys
    print(f"❌ Failed to import ElevenLabs: {e}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.executable}")
    print("\nTry running with python3 explicitly:")
    print("python3 transcribe/elevenlabs_scribe_transcribe.py ...")
    exit(1)

from base_transcriber import BaseTranscriber, TranscriptionResult


class ElevenLabsScribeTranscriber(BaseTranscriber):
    """ElevenLabs Scribe transcriber."""
    
    def __init__(self, model_id: str = "s2t-english-latest", 
                 api_key: str = None, 
                 results_dir: str = None):
        # Use safe display name for directories
        display_name = f"elevenlabs-{model_id.replace('/', '-')}"
        super().__init__(display_name, results_dir)
        
        # Set up ElevenLabs client
        self.api_key = api_key or os.getenv('ELEVENLABS_API_KEY')
        if not self.api_key:
            raise ValueError("ElevenLabs API key required. Set ELEVENLABS_API_KEY environment variable or pass api_key parameter.")
        
        self.client = ElevenLabs(api_key=self.api_key)
        self.model_id = model_id
    
    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """
        Transcribe a single audio file using ElevenLabs Scribe.
        
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
                # Call ElevenLabs Scribe API
                response = self.client.speech_to_text.convert(
                    model_id=self.model_id,
                    file=audio
                )
            
            duration = time.time() - start_time
            
            # Handle response - extract text from response
            if hasattr(response, 'text'):
                text = response.text.strip()
            elif hasattr(response, 'transcript'):
                text = response.transcript.strip()
            elif isinstance(response, str):
                text = response.strip()
            elif isinstance(response, dict) and 'text' in response:
                text = response['text'].strip()
            elif isinstance(response, dict) and 'transcript' in response:
                text = response['transcript'].strip()
            else:
                # Try to extract any text-like field
                text = str(response).strip()
            
            if not text:
                raise ValueError("Empty transcription result")
            
            return TranscriptionResult(
                text=text,
                duration=duration,
                model_name=self.model_name,
                audio_file=audio_file
            )
            
        except Exception as e:
            raise Exception(f"ElevenLabs Scribe transcription failed: {e}")


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ElevenLabs Scribe Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--model", default="s2t-english-latest", help="ElevenLabs model ID")
    parser.add_argument("--api_key", help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var)")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    parser.add_argument("--single_file", help="Process single file only (for testing)")
    
    args = parser.parse_args()
    
    # Initialize transcriber
    try:
        transcriber = ElevenLabsScribeTranscriber(
            model_id=args.model,
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
            print(f"🎤 Testing ElevenLabs Scribe on: {args.single_file}")
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