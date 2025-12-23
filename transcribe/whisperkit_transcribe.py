#!/usr/bin/env python3
"""
WhisperKit on-device transcriber for speech-to-text.
"""

import time
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from base_transcriber import BaseTranscriber, TranscriptionResult


class WhisperKitTranscriber(BaseTranscriber):
    """WhisperKit on-device transcriber using CLI."""
    
    def __init__(self, model_name: str = "large-v3-v20240930_turbo", 
                 language: str = "en",
                 results_dir: str = None):
        # Use safe display name for directories
        display_name = f"whisperkit-{model_name}"
        super().__init__(display_name, results_dir)
        
        # Check if WhisperKit CLI is available
        try:
            subprocess.run(['whisperkit-cli', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise FileNotFoundError("WhisperKit CLI not found. Install with: brew install whisperkit-cli")
        
        self.model_name_internal = model_name
        self.language = language
        
        print(f"🥃 Using WhisperKit model: {self.model_name_internal}")
        print(f"🌐 Language: {self.language}")
    
    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """
        Transcribe a single audio file using WhisperKit.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            TranscriptionResult object
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        start_time = time.time()
        
        try:
            # Build WhisperKit CLI command
            cmd = [
                'whisperkit-cli', 'transcribe',
                '--audio-path', audio_file,
                '--model', self.model_name_internal,
                '--language', self.language,
                '--temperature', '0.0',
                '--without-timestamps',  # Clean text output
                '--skip-special-tokens'  # Remove special tokens
            ]
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"WhisperKit failed with code {result.returncode}: {result.stderr}")
            
            # Extract transcription from output
            # WhisperKit outputs in format: "Transcription of filename: \n\nActual text"
            lines = result.stdout.strip().split('\n')
            
            # Find the transcription text (after the header lines)
            transcription_text = ""
            found_content = False
            for line in lines:
                if line.strip() == "":
                    found_content = True
                    continue
                if found_content and not line.startswith("Transcription of"):
                    transcription_text += line.strip() + " "
            
            # If we didn't find content in expected format, try to extract differently
            if not transcription_text.strip():
                # Look for lines that don't start with metadata
                for line in lines:
                    if (line.strip() and 
                        not line.startswith("Task:") and 
                        not line.startswith("Initializing") and
                        not line.startswith("Models initialized") and
                        not line.startswith("Transcription of")):
                        transcription_text += line.strip() + " "
            
            duration = time.time() - start_time
            text = transcription_text.strip()
            
            if not text:
                raise ValueError("Empty transcription result")
            
            return TranscriptionResult(
                text=text,
                duration=duration,
                model_name=self.model_name,
                audio_file=audio_file
            )
                
        except Exception as e:
            raise Exception(f"WhisperKit transcription failed: {e}")


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WhisperKit On-Device Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--model", default="large-v3-v20240930_turbo", help="WhisperKit model name")
    parser.add_argument("--language", default="en", help="Language for transcription (e.g., en)")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    parser.add_argument("--single_file", help="Process single file only (for testing)")
    
    args = parser.parse_args()
    
    # Initialize transcriber
    try:
        transcriber = WhisperKitTranscriber(
            model_name=args.model,
            language=args.language,
            results_dir=args.results_dir
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    except Exception as e:
        print(f"❌ {e}")
        return
    
    # Process single file or batch
    if args.single_file:
        # Single file mode for testing
        try:
            print(f"🎤 Testing WhisperKit on: {args.single_file}")
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