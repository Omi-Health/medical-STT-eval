#!/usr/bin/env python3
"""
Base transcriber class providing a consistent interface for all speech-to-text models.
"""

import time
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

# Load .env file from project root
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass  # python-dotenv not installed, rely on shell environment


class TranscriptionResult:
    """Container for transcription results."""
    
    def __init__(self, text: str, duration: float, model_name: str, audio_file: str):
        self.text = text
        self.duration = duration
        self.model_name = model_name
        self.audio_file = audio_file
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'text': self.text,
            'duration': self.duration,
            'model_name': self.model_name,
            'audio_file': self.audio_file,
            'timestamp': self.timestamp
        }


class BaseTranscriber(ABC):
    """Abstract base class for all transcribers."""
    
    def __init__(self, model_name: str, results_dir: str = None):
        self.model_name = model_name
        self.results_dir = Path(results_dir) if results_dir else Path.cwd() / "results"
        self.transcripts_dir = self.results_dir / "transcripts" / model_name.replace("/", "_")
        self.metrics_dir = self.results_dir / "metrics"
        
        # Create directories
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """
        Transcribe a single audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            TranscriptionResult object
        """
        pass
    
    def transcribe_batch(self, audio_files: List[str]) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            List of TranscriptionResult objects
        """
        results = []
        total_files = len(audio_files)
        
        print(f"🎤 Starting batch transcription with {self.model_name}")
        print(f"📁 Processing {total_files} files")
        print(f"💾 Results will be saved to: {self.transcripts_dir}")
        
        # Problematic files that cause issues for most models
        EXCLUDED_FILES = ['day1_consultation07', 'day3_consultation03']

        skipped = 0
        for i, audio_file in enumerate(audio_files, 1):
            audio_name = Path(audio_file).stem

            # Skip problematic files
            if any(exc in audio_name for exc in EXCLUDED_FILES):
                print(f"[{i}/{total_files}] ⏭️  Skipping (excluded): {Path(audio_file).name}")
                skipped += 1
                continue

            transcript_file = self.transcripts_dir / f"{audio_name}_transcript.txt"

            # Skip if transcript already exists
            if transcript_file.exists():
                print(f"[{i}/{total_files}] ⏭️  Skipping (exists): {Path(audio_file).name}")
                skipped += 1
                continue

            print(f"\n[{i}/{total_files}] Processing: {Path(audio_file).name}")

            try:
                result = self.transcribe_file(audio_file)
                results.append(result)

                # Save individual transcript
                self._save_transcript(result)

                print(f"✅ Success - Duration: {result.duration:.2f}s")

            except Exception as e:
                print(f"❌ Error: {e}")
                continue

        if skipped > 0:
            print(f"\n⏭️  Skipped {skipped} files (transcripts already exist)")
        
        # Save batch metrics
        self._save_metrics(results)
        
        print(f"\n📊 Batch complete: {len(results)}/{total_files} files processed")
        return results
    
    def _save_transcript(self, result: TranscriptionResult):
        """Save individual transcript to file."""
        audio_name = Path(result.audio_file).stem
        transcript_file = self.transcripts_dir / f"{audio_name}_transcript.txt"
        
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(result.text)
    
    def _save_metrics(self, results: List[TranscriptionResult]):
        """Save processing metrics."""
        if not results:
            return
        
        total_duration = sum(r.duration for r in results)
        avg_duration = total_duration / len(results)
        total_files = len(results)
        
        metrics = {
            'model_name': self.model_name,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'summary': {
                'total_files': total_files,
                'total_duration': total_duration,
                'average_duration': avg_duration,
                'fastest_file': min(results, key=lambda r: r.duration).duration,
                'slowest_file': max(results, key=lambda r: r.duration).duration
            },
            'file_details': [
                {
                    'audio_file': Path(r.audio_file).name,
                    'duration': r.duration,
                    'text_length': len(r.text)
                } for r in results
            ]
        }
        
        metrics_file = self.metrics_dir / f"{self.model_name.replace('/', '_')}_speed.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Metrics saved to: {metrics_file}")
    
    def get_audio_files(self, audio_dir: str, pattern: str = "*_conversation.wav") -> List[str]:
        """Get list of audio files to process."""
        audio_path = Path(audio_dir)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        
        audio_files = list(audio_path.glob(pattern))
        if not audio_files:
            raise FileNotFoundError(f"No audio files found matching pattern: {pattern}")
        
        return [str(f) for f in sorted(audio_files)]