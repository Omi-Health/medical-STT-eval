#!/usr/bin/env python3
"""
Enhanced NVIDIA Canary-Qwen-2.5B transcriber with improved overlap merging.
Uses sophisticated merging algorithm inspired by Groq method for optimal performance.
"""

import os
import sys
import time
import torch
import librosa
import soundfile as sf
import numpy as np
import tempfile
import shutil
import re
from pathlib import Path
from typing import Optional, List, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from nemo.collections.speechlm2.models import SALM
except ImportError:
    print("❌ NeMo not installed. Install with: python -m pip install 'nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git'")
    exit(1)

from base_transcriber import BaseTranscriber, TranscriptionResult


def find_longest_common_sequence(sequences: List[str], match_by_words: bool = True) -> str:
    """
    Find optimal alignment between sequences using longest common sequence matching.
    Adapted from Groq's sophisticated merging approach.
    
    Args:
        sequences: List of text sequences to merge
        match_by_words: Whether to match by words (True) or characters (False)
        
    Returns:
        Merged sequence with optimal alignment
    """
    if not sequences:
        return ""
    
    if len(sequences) == 1:
        return sequences[0]
    
    # Convert to word/character lists based on strategy
    if match_by_words:
        # Split by words but preserve spacing
        processed_sequences = []
        for seq in sequences:
            words = re.findall(r'\S+|\s+', seq)
            processed_sequences.append(words)
    else:
        processed_sequences = [list(seq) for seq in sequences]
    
    left_sequence = processed_sequences[0]
    left_length = len(left_sequence)
    
    for right_sequence in processed_sequences[1:]:
        max_matching = 0.0
        right_length = len(right_sequence)
        max_indices = (left_length, left_length, 0, 0)
        
        # Try different alignments to find best overlap
        for i in range(1, left_length + right_length + 1):
            # Add small epsilon to favor longer matches
            eps = float(i) / 10000.0
            
            left_start = max(0, left_length - i)
            left_stop = min(left_length, left_length + right_length - i)
            left_part = left_sequence[left_start:left_stop]
            
            right_start = max(0, i - left_length)
            right_stop = min(right_length, i)
            right_part = right_sequence[right_start:right_stop]
            
            if len(left_part) != len(right_part):
                continue
            
            # Count exact matches
            matches = sum(a == b for a, b in zip(left_part, right_part))
            
            # Normalize by overlap length and add epsilon
            if i > 0:
                matching = matches / float(i) + eps
                
                # Require at least 2 matches for valid overlap
                if matches >= 2 and matching > max_matching:
                    max_matching = matching
                    max_indices = (left_start, left_stop, right_start, right_stop)
        
        # Apply best alignment found
        left_start, left_stop, right_start, right_stop = max_indices
        
        # Take left part from left sequence, right part from right sequence
        left_mid = (left_stop + left_start) // 2
        right_mid = (right_stop + right_start) // 2
        
        # Merge: left half of overlap from left, right half from right
        merged_sequence = left_sequence[:left_mid] + right_sequence[right_mid:]
        left_sequence = merged_sequence
        left_length = len(left_sequence)
    
    # Join back to text
    if match_by_words:
        return ''.join(left_sequence)
    else:
        return ''.join(left_sequence)


class CanaryQwenImprovedTranscriber(BaseTranscriber):
    """Enhanced Canary-Qwen transcriber with improved overlap merging for long files."""
    
    def __init__(self, model_name: str = "nvidia/canary-qwen-2.5b", results_dir: str = None):
        super().__init__(model_name, results_dir)
        self.model = None
        self.model_loaded = False
        
        # Improved chunking parameters based on testing
        self.chunk_duration = 35.0  # Conservative 35s chunks (model limit is 40s)
        self.overlap_duration = 10.0  # Increased from 2s to 10s for better merging
        self.max_tokens = 800  # Conservative token limit (model limit is 1024)
        
        # Set environment variable to reduce memory fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    def _load_model(self):
        """Load the model if not already loaded."""
        if not self.model_loaded:
            print(f"📦 Loading {self.model_name} model in BFloat16 precision...")
            
            try:
                # Try to load with bfloat16 directly
                self.model = SALM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
            except TypeError:
                # Fallback: Load normally and convert to BFloat16
                print("Loading model and converting to BFloat16...")
                self.model = SALM.from_pretrained(self.model_name)
                self.model = self.model.to(torch.bfloat16)
                self.model = self.model.cuda()
            
            self.model_loaded = True
            print("✅ Model loaded successfully in BFloat16")
            
            # Clear cache after loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _chunk_audio(self, audio_file: str) -> Tuple[List[Tuple[str, float, float]], Path]:
        """
        Split audio into chunks with improved overlap for better merging.
        
        Returns:
            Tuple of (chunks_list, temp_directory)
        """
        # Load audio (no preprocessing - keep original format)
        audio, sr = librosa.load(audio_file, sr=16000)
        total_duration = len(audio) / sr
        
        # If audio is short enough, don't chunk
        if total_duration <= self.chunk_duration:
            temp_dir = Path(tempfile.mkdtemp(prefix="canary_single_"))
            single_file = temp_dir / "single.wav"
            sf.write(str(single_file), audio, sr)
            return [(str(single_file), 0.0, total_duration)], temp_dir
        
        # Chunk the audio with improved overlap
        chunks = []
        chunk_samples = int(self.chunk_duration * sr)
        overlap_samples = int(self.overlap_duration * sr)
        step_samples = chunk_samples - overlap_samples
        
        # Create temp directory for chunks
        temp_dir = Path(tempfile.mkdtemp(prefix="canary_improved_"))
        
        chunk_count = 0
        for start_sample in range(0, len(audio), step_samples):
            end_sample = min(start_sample + chunk_samples, len(audio))
            
            # Skip if remaining audio is too short (less than 5 seconds)
            if (end_sample - start_sample) / sr < 5.0 and start_sample > 0:
                break
                
            chunk_audio = audio[start_sample:end_sample]
            
            # Apply fade in/out to reduce artifacts
            fade_samples = int(0.1 * sr)  # 0.1s fade
            if len(chunk_audio) > fade_samples * 2:
                # Fade in
                chunk_audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
                # Fade out
                chunk_audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            # Save chunk
            chunk_file = temp_dir / f"chunk_{chunk_count:03d}.wav"
            sf.write(str(chunk_file), chunk_audio, sr)
            
            start_time = start_sample / sr
            end_time = end_sample / sr
            chunks.append((str(chunk_file), start_time, end_time))
            chunk_count += 1
            
            # Stop if we've processed the entire file
            if end_sample >= len(audio):
                break
        
        return chunks, temp_dir
    
    def _merge_chunk_transcriptions_improved(self, chunk_transcriptions: List[Tuple[str, float, float]]) -> str:
        """
        Merge chunk transcriptions using improved algorithm inspired by Groq method.
        
        Args:
            chunk_transcriptions: List of (transcription, start_time, end_time)
            
        Returns:
            Merged transcription text
        """
        if not chunk_transcriptions:
            return ""
        
        if len(chunk_transcriptions) == 1:
            return chunk_transcriptions[0][0]
        
        # Extract just the text parts for merging
        texts = [text for text, _, _ in chunk_transcriptions]
        
        # Use the sophisticated merging algorithm
        merged_text = find_longest_common_sequence(texts, match_by_words=True)
        
        # Clean up any extra whitespace
        merged_text = re.sub(r'\s+', ' ', merged_text).strip()
        
        return merged_text
    
    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """
        Transcribe a single audio file using improved chunking approach.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            TranscriptionResult object
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Load model on first use
        self._load_model()
        
        start_time = time.time()
        
        try:
            # Get audio duration for reporting
            audio, sr = librosa.load(audio_file, sr=16000)
            total_duration = len(audio) / sr
            
            # Chunk the audio with improved settings
            chunks, temp_dir = self._chunk_audio(audio_file)
            
            # Process each chunk
            chunk_transcriptions = []
            
            for i, (chunk_file, chunk_start, chunk_end) in enumerate(chunks):
                try:
                    # Clear cache before each chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Generate transcription with autocast for BFloat16
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        answer_ids = self.model.generate(
                            prompts=[
                                [{"role": "user", 
                                  "content": f"Transcribe the following: {self.model.audio_locator_tag}", 
                                  "audio": [chunk_file]}]
                            ],
                            max_new_tokens=self.max_tokens,
                        )
                    
                    # Convert to text
                    chunk_text = self.model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()
                    
                    if chunk_text:
                        chunk_transcriptions.append((chunk_text, chunk_start, chunk_end))
                    
                except Exception as e:
                    print(f"⚠️ Warning: Failed to process chunk {i+1}: {e}")
                    # Add empty transcription for this chunk
                    chunk_transcriptions.append(("", chunk_start, chunk_end))
                    continue
            
            # Merge all chunk transcriptions with improved algorithm
            full_text = self._merge_chunk_transcriptions_improved(chunk_transcriptions)
            
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            
            duration = time.time() - start_time
            
            if not full_text:
                raise ValueError("Empty transcription result from all chunks")
            
            return TranscriptionResult(
                text=full_text,
                duration=duration,
                model_name=self.model_name,
                audio_file=audio_file
            )
            
        except torch.cuda.OutOfMemoryError as e:
            # Clear cache and raise error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise Exception(f"GPU out of memory during improved chunked transcription: {e}")
        except Exception as e:
            raise Exception(f"Canary-Qwen improved chunked transcription failed: {e}")


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NVIDIA Canary-Qwen-2.5B Improved Chunked Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--model", default="nvidia/canary-qwen-2.5b", help="Model name")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    parser.add_argument("--chunk_duration", type=float, default=35.0, help="Chunk duration in seconds")
    parser.add_argument("--overlap", type=float, default=10.0, help="Overlap duration in seconds")
    parser.add_argument("--max_tokens", type=int, default=800, help="Max tokens per chunk")
    
    args = parser.parse_args()
    
    # Initialize transcriber
    transcriber = CanaryQwenImprovedTranscriber(
        model_name=args.model,
        results_dir=args.results_dir
    )
    
    # Set custom parameters if provided
    transcriber.chunk_duration = args.chunk_duration
    transcriber.overlap_duration = args.overlap
    transcriber.max_tokens = args.max_tokens
    
    # Get audio files
    try:
        audio_files = transcriber.get_audio_files(args.audio_dir, args.pattern)
        print(f"📁 Found {len(audio_files)} audio files")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    print(f"🎤 Using Canary-Qwen with IMPROVED chunking:")
    print(f"   - Chunk duration: {args.chunk_duration}s")
    print(f"   - Overlap: {args.overlap}s (improved from 2s)") 
    print(f"   - Max tokens per chunk: {args.max_tokens}")
    print(f"   - Merging algorithm: Longest Common Sequence")
    
    # Process files
    results = transcriber.transcribe_batch(audio_files)
    
    if results:
        print(f"\n✅ Successfully processed {len(results)} files")
        print(f"📁 Transcripts saved to: {transcriber.transcripts_dir}")
        print(f"📊 Metrics saved to: {transcriber.metrics_dir}")
        
        # Show summary statistics
        total_duration = sum(r.duration for r in results)
        avg_duration = total_duration / len(results)
        print(f"📈 Processing statistics:")
        print(f"   - Total processing time: {total_duration:.1f}s")
        print(f"   - Average per file: {avg_duration:.1f}s")
        print(f"   - Processing speed: {len(results)/total_duration:.2f} files/second")
    else:
        print("\n❌ No files were processed successfully")


if __name__ == "__main__":
    main()