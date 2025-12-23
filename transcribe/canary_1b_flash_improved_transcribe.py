#!/usr/bin/env python3
"""
NVIDIA Canary-1B-Flash Speech Recognition Model - Improved Version.
Uses sophisticated overlap merging with LCS algorithm for better accuracy.
Based on the successful approach from canary_qwen_improved_transcribe.py
"""

import os
import sys
import time
import json
import argparse
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from pydub import AudioSegment
import soundfile as sf
import warnings
import torch
from difflib import SequenceMatcher

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcribe.base_transcriber import BaseTranscriber, TranscriptionResult

# Try to import NeMo components
try:
    from nemo.collections.asr.models import EncDecMultiTaskModel
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    print("WARNING: NeMo toolkit not available. Please install with:")
    print("pip install 'nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git'")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class CanaryFlashImprovedTranscriber(BaseTranscriber):
    """
    NVIDIA Canary-1B-Flash transcriber with sophisticated overlap merging.
    Uses LCS algorithm for intelligent transcript merging.
    """
    
    def __init__(self, 
                 model_name: str = "nvidia/canary-1b-flash",
                 chunk_duration: float = 35.0,  # Conservative like Canary-Qwen
                 overlap: float = 10.0,  # Larger overlap for better merging
                 device: str = None,
                 **kwargs):
        """
        Initialize Canary-1B-Flash improved transcriber.
        
        Args:
            model_name: HuggingFace model name
            chunk_duration: Duration of each chunk in seconds (max 40s)
            overlap: Overlap between chunks in seconds
            device: Device to use ('cuda' or 'cpu')
        """
        super().__init__(model_name=model_name, **kwargs)
        
        if not NEMO_AVAILABLE:
            raise ImportError("NeMo toolkit is required but not installed")
        
        self.chunk_duration = min(chunk_duration, 40.0)
        self.overlap = overlap
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing Canary-1B-Flash (Improved) model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Chunk duration: {chunk_duration}s with {overlap}s overlap")
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the Canary model."""
        try:
            self.model = EncDecMultiTaskModel.from_pretrained(
                self.model_name, 
                map_location=self.device
            )
            
            # Update decode params
            decode_cfg = self.model.cfg.decoding
            decode_cfg.beam.beam_size = 1
            self.model.change_decoding_strategy(decode_cfg)
            
            # Put model in eval mode
            self.model.eval()
            
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _split_audio_with_overlap(self, audio_path: str) -> List[Tuple[str, float, float]]:
        """
        Split audio file into overlapping chunks with fade in/out.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of (chunk_path, start_time, end_time) tuples
        """
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        chunk_ms = int(self.chunk_duration * 1000)
        overlap_ms = int(self.overlap * 1000)
        
        chunks = []
        start = 0
        
        # Create temporary directory for chunks
        temp_dir = tempfile.mkdtemp()
        
        chunk_idx = 0
        while start < duration_ms:
            # Calculate end position
            end = min(start + chunk_ms, duration_ms)
            
            # Extract chunk
            chunk = audio[start:end]
            
            # Apply fade in/out (100ms each) to reduce artifacts
            if start > 0:
                chunk = chunk.fade_in(100)
            if end < duration_ms:
                chunk = chunk.fade_out(100)
            
            # Save chunk
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_idx:04d}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append((chunk_path, start / 1000.0, end / 1000.0))
            
            # Move to next chunk
            start += chunk_ms - overlap_ms
            chunk_idx += 1
            
            # Break if we've reached the end
            if end >= duration_ms:
                break
        
        return chunks, temp_dir
    
    def _find_lcs_overlap(self, text1: str, text2: str, min_overlap_words: int = 2) -> Tuple[int, int, float]:
        """
        Find longest common sequence overlap between two texts using word-based matching.
        
        Args:
            text1: First text (end portion will be checked)
            text2: Second text (beginning portion will be checked)
            min_overlap_words: Minimum words required for valid overlap
            
        Returns:
            Tuple of (overlap_start_in_text1, overlap_end_in_text2, score)
        """
        words1 = text1.split()
        words2 = text2.split()
        
        if len(words1) < min_overlap_words or len(words2) < min_overlap_words:
            return -1, -1, 0.0
        
        best_overlap_start = -1
        best_overlap_end = -1
        best_score = 0.0
        
        # Search for overlaps in the last portion of text1 and beginning of text2
        search_window = min(len(words1), int(self.overlap * 20))  # ~20 words per second
        
        for i in range(max(0, len(words1) - search_window), len(words1)):
            for j in range(min(search_window, len(words2))):
                # Check if we have a potential match
                if words1[i] == words2[j]:
                    # Use SequenceMatcher to find the extent of the match
                    matcher = SequenceMatcher(None, words1[i:], words2[:j+search_window])
                    match = matcher.find_longest_match(0, len(words1[i:]), 0, len(words2[:j+search_window]))
                    
                    if match.size >= min_overlap_words:
                        # Calculate score based on match length and position
                        position_score = 1.0 - (j / search_window if search_window > 0 else 0)
                        length_score = match.size / search_window if search_window > 0 else 0
                        score = (position_score + length_score) / 2
                        
                        if score > best_score:
                            best_score = score
                            best_overlap_start = i
                            best_overlap_end = j + match.size
        
        return best_overlap_start, best_overlap_end, best_score
    
    def _merge_transcripts_with_lcs(self, transcripts: List[Tuple[str, float, float]]) -> str:
        """
        Merge overlapping transcripts using LCS algorithm.
        
        Args:
            transcripts: List of (text, start_time, end_time) tuples
            
        Returns:
            Merged transcript text
        """
        if not transcripts:
            return ""
        
        if len(transcripts) == 1:
            return transcripts[0][0]
        
        # Start with the first transcript
        merged_text = transcripts[0][0]
        
        for i in range(1, len(transcripts)):
            current_text = transcripts[i][0]
            
            if not current_text.strip():
                continue
            
            # Find overlap using LCS
            overlap_start, overlap_end, score = self._find_lcs_overlap(merged_text, current_text)
            
            if overlap_start > 0 and score > 0.3:  # Threshold for valid overlap
                # Merge with overlap
                words = merged_text.split()
                merged_text = ' '.join(words[:overlap_start]) + ' ' + current_text
            else:
                # No significant overlap found, append with space
                merged_text = merged_text + ' ' + current_text
            
            # Clean up extra spaces
            merged_text = ' '.join(merged_text.split())
        
        return merged_text
    
    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe a single audio file with sophisticated chunking.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            TranscriptionResult object
        """
        start_time = time.time()
        
        try:
            # Get audio duration
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0
            
            # Check if we need chunking
            if duration > self.chunk_duration:
                # Split audio into chunks
                chunks, temp_dir = self._split_audio_with_overlap(audio_path)
                print(f"Split {duration:.1f}s audio into {len(chunks)} chunks")
                
                transcripts = []
                
                try:
                    # Transcribe each chunk
                    for i, (chunk_path, start_t, end_t) in enumerate(chunks):
                        print(f"Processing chunk {i+1}/{len(chunks)} ({start_t:.1f}s - {end_t:.1f}s)")
                        
                        # Transcribe chunk
                        output = self.model.transcribe(
                            [chunk_path],
                            batch_size=1,
                            pnc='yes',
                            source_lang='en',
                            target_lang='en',
                        )
                        
                        chunk_text = output[0].text
                        transcripts.append((chunk_text, start_t, end_t))
                        
                        # Clean up chunk file
                        os.remove(chunk_path)
                    
                    # Merge transcripts using LCS algorithm
                    final_transcript = self._merge_transcripts_with_lcs(transcripts)
                    
                finally:
                    # Clean up temp directory
                    import shutil
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
            else:
                # Direct inference for short audio
                output = self.model.transcribe(
                    [audio_path],
                    batch_size=1,
                    pnc='yes',
                    source_lang='en',
                    target_lang='en',
                )
                final_transcript = output[0].text
            
            # Calculate metrics
            processing_time = time.time() - start_time
            
            return TranscriptionResult(
                text=final_transcript,
                duration=processing_time,
                model_name=self.model_name,
                audio_file=audio_path
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Transcription failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            return TranscriptionResult(
                text=f"[ERROR: {error_msg}]",
                duration=processing_time,
                model_name=self.model_name,
                audio_file=audio_path
            )
    
    def process_directory(self, audio_dir: str, max_files: Optional[int] = None) -> Dict:
        """Process all audio files in a directory."""
        audio_path = Path(audio_dir)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        
        # Get audio files
        audio_files = sorted(list(audio_path.glob("*.wav")))
        if not audio_files:
            audio_files = sorted(list(audio_path.glob("*.flac")))
        
        if max_files:
            audio_files = audio_files[:max_files]
        
        total_files = len(audio_files)
        
        if total_files == 0:
            print("No audio files found!")
            return {
                'total_files': 0,
                'successful': 0,
                'failed': 0,
                'total_time': 0,
                'output_dir': str(self.output_dir)
            }
        
        print(f"\n{'='*60}")
        print(f"Processing {total_files} audio files with Canary-1B-Flash (Improved)")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
        
        total_time = 0
        successful = 0
        failed = 0
        errors = {}
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{total_files}] Processing: {audio_file.name}")
            
            result = self.transcribe_file(str(audio_file))
            total_time += result.duration
            
            if not result.text.startswith("[ERROR:"):
                successful += 1
                # Save transcript
                transcript_file = self.transcripts_dir / f"{audio_file.stem}_transcript.txt"
                with open(transcript_file, 'w') as f:
                    f.write(result.text)
                print(f"✅ Completed in {result.duration:.2f}s")
            else:
                failed += 1
                error_msg = result.text.replace("[ERROR: ", "").replace("]", "")
                errors[audio_file.name] = error_msg
                print(f"❌ Failed: {error_msg}")
        
        # Save summary
        summary = {
            'model_name': self.model_name,
            'total_files': total_files,
            'successful': successful,
            'failed': failed,
            'total_time': total_time,
            'average_time_per_file': total_time / total_files if total_files > 0 else 0,
            'output_dir': str(self.output_dir),
            'chunk_duration': self.chunk_duration,
            'overlap': self.overlap
        }
        
        summary_path = self.output_dir / "transcription_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return {
            'total_files': total_files,
            'successful': successful,
            'failed': failed,
            'total_time': total_time,
            'output_dir': str(self.output_dir),
            'errors': errors
        }


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using NVIDIA Canary-1B-Flash with improved overlap merging"
    )
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, 
                        default="results/transcripts/canary_1b_flash_improved",
                        help="Output directory for transcripts")
    parser.add_argument("--model_name", type=str, 
                        default="nvidia/canary-1b-flash",
                        help="Model name")
    parser.add_argument("--chunk_duration", type=float, default=35.0,
                        help="Duration of each chunk in seconds")
    parser.add_argument("--overlap", type=float, default=10.0,
                        help="Overlap between chunks in seconds")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to process")
    
    args = parser.parse_args()
    
    # Initialize transcriber
    transcriber = CanaryFlashImprovedTranscriber(
        model_name=args.model_name,
        chunk_duration=args.chunk_duration,
        overlap=args.overlap,
        device=args.device
    )
    transcriber.output_dir = Path(args.output_dir)
    transcriber.transcripts_dir = transcriber.output_dir
    transcriber.transcripts_dir.mkdir(parents=True, exist_ok=True)
    
    # Process audio files
    results = transcriber.process_directory(
        args.audio_dir,
        max_files=args.max_files
    )
    
    # Print summary
    print("\n" + "="*60)
    print("TRANSCRIPTION COMPLETE")
    print("="*60)
    print(f"Total files: {results['total_files']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Output directory: {results['output_dir']}")


if __name__ == "__main__":
    main()