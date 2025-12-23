#!/usr/bin/env python3
"""
Google Gemini 2.5 Flash transcriber for speech-to-text.
"""

import time
import os
from pathlib import Path
from typing import Optional

try:
    from google import genai
except ImportError:
    print("❌ Google GenAI library not installed. Install with: pip install google-genai")
    exit(1)

from base_transcriber import BaseTranscriber, TranscriptionResult


class GeminiTranscriber(BaseTranscriber):
    """Google Gemini transcriber."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash", 
                 api_key: str = None, 
                 results_dir: str = None):
        # Use safe display name for directories
        display_name = f"google-{model_name}"
        super().__init__(display_name, results_dir)
        
        # Set up Gemini client
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        # Configure the client
        self.client = genai.Client(api_key=self.api_key)
        self.model_name_internal = model_name
    
    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """
        Transcribe a single audio file using Google Gemini.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            TranscriptionResult object
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        start_time = time.time()
        
        try:
            # Check file size to decide upload method
            file_size = Path(audio_file).stat().st_size
            max_inline_size = 15 * 1024 * 1024  # 15MB (conservative, API limit is 20MB total)
            
            if file_size > max_inline_size:
                # Upload file first
                print(f"📤 Uploading large file ({file_size/1024/1024:.1f}MB)...")
                myfile = self.client.files.upload(file=str(audio_file))
                
                # Generate content with uploaded file
                response = self.client.models.generate_content(
                    model=self.model_name_internal,
                    contents=[
                        "Generate a transcript of the speech. Provide only the transcription without any additional commentary, formatting, or explanations. Do not include timestamps or speaker labels.",
                        myfile
                    ]
                )
            else:
                # Use inline audio data for smaller files
                print(f"📝 Processing file inline ({file_size/1024/1024:.1f}MB)...")
                with open(audio_file, 'rb') as f:
                    audio_bytes = f.read()
                
                from google.genai import types
                
                response = self.client.models.generate_content(
                    model=self.model_name_internal,
                    contents=[
                        "Generate a transcript of the speech. Provide only the transcription without any additional commentary, formatting, or explanations. Do not include timestamps or speaker labels.",
                        types.Part.from_bytes(
                            data=audio_bytes,
                            mime_type='audio/wav',
                        )
                    ]
                )
            
            duration = time.time() - start_time
            
            # Extract text from response
            text = response.text.strip() if hasattr(response, 'text') else str(response).strip()
            
            if not text:
                raise ValueError("Empty transcription result")
            
            return TranscriptionResult(
                text=text,
                duration=duration,
                model_name=self.model_name,
                audio_file=audio_file
            )
                
        except Exception as e:
            raise Exception(f"Gemini transcription failed: {e}")


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Google Gemini Transcription")
    parser.add_argument("--audio_dir", help="Directory containing audio files")
    parser.add_argument("--single_file", help="Process single file only (for testing)")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--api_key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    
    args = parser.parse_args()
    
    if not args.audio_dir and not args.single_file:
        parser.error("Either --audio_dir or --single_file must be specified")
    
    # Initialize transcriber
    try:
        transcriber = GeminiTranscriber(
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
            print(f"🎤 Testing Gemini on: {args.single_file}")
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