#!/usr/bin/env python3
"""
Apple SpeechAnalyzer transcriber for speech-to-text.
NOTE: Requires macOS 26.0+ with Xcode 16 beta
"""

import time
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from base_transcriber import BaseTranscriber, TranscriptionResult


class AppleSpeechAnalyzerTranscriber(BaseTranscriber):
    """Apple SpeechAnalyzer transcriber using CLI."""
    
    def __init__(self, model_name: str = "Apple-SpeechAnalyzer", 
                 cli_path: str = None,
                 locale: str = "en-US",
                 results_dir: str = None):
        # Use safe display name for directories
        display_name = "apple-speechanalyzer"
        super().__init__(display_name, results_dir)
        
        # Set up CLI path
        default_cli = (
            Path(__file__).resolve().parent.parent
            / "apple-speechanalyzer-cli-example"
            / ".build"
            / "release"
            / "apple-speechanalyzer-cli"
        )
        self.cli_path = cli_path or str(default_cli)
        if not Path(self.cli_path).exists():
            raise FileNotFoundError(
                "Apple SpeechAnalyzer CLI not found. "
                "Clone https://github.com/argmaxinc/apple-speechanalyzer-cli-example "
                "and build it, or pass --cli_path."
            )
        
        self.locale = locale
        
        print(f"🍎 Using Apple SpeechAnalyzer CLI: {self.cli_path}")
        print(f"🌐 Locale: {self.locale}")
    
    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """
        Transcribe a single audio file using Apple SpeechAnalyzer.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            TranscriptionResult object
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        start_time = time.time()
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            # Run Apple SpeechAnalyzer CLI
            cmd = [
                self.cli_path,
                '--input-audio-path', audio_file,
                '--output-txt-path', output_path,
                '--locale', self.locale
            ]
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"SpeechAnalyzer failed with code {result.returncode}: {result.stderr}")
            
            # Read the transcription
            with open(output_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            duration = time.time() - start_time
            
            if not text:
                raise ValueError("Empty transcription result")
            
            return TranscriptionResult(
                text=text,
                duration=duration,
                model_name=self.model_name,
                audio_file=audio_file
            )
                
        except Exception as e:
            raise Exception(f"Apple SpeechAnalyzer transcription failed: {e}")
        finally:
            # Clean up temporary file
            if Path(output_path).exists():
                os.unlink(output_path)


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Apple SpeechAnalyzer Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--cli_path", help="Path to apple-speechanalyzer-cli binary")
    parser.add_argument("--locale", default="en-US", help="Locale for transcription (e.g., en-US)")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    parser.add_argument("--single_file", help="Process single file only (for testing)")
    
    args = parser.parse_args()
    
    # Initialize transcriber
    try:
        transcriber = AppleSpeechAnalyzerTranscriber(
            cli_path=args.cli_path,
            locale=args.locale,
            results_dir=args.results_dir
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please ensure Apple SpeechAnalyzer CLI is built:")
        print("  cd apple-speechanalyzer-cli-example && swift build -c release")
        return
    except Exception as e:
        print(f"❌ {e}")
        return
    
    # Process single file or batch
    if args.single_file:
        # Single file mode for testing
        try:
            print(f"🎤 Testing Apple SpeechAnalyzer on: {args.single_file}")
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
