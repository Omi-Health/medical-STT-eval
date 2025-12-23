#!/usr/bin/env python3
"""
Enhanced Azure Foundry Phi-4 Multimodal transcriber with improved chunking.
Uses the same sophisticated overlap merging algorithm as Canary-Qwen for better performance.
"""

import time
import os
import subprocess
import tempfile
import shutil
import re
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage,
    UserMessage,
    TextContentItem,
    AudioContentItem,
    InputAudio,
    AudioContentFormat,
)
from azure.core.credentials import AzureKeyCredential

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent))

from base_transcriber import BaseTranscriber, TranscriptionResult


def find_longest_common_sequence(sequences: List[str], match_by_words: bool = True) -> str:
    """
    Find optimal alignment between sequences using longest common sequence matching.
    Adapted from Canary-Qwen's sophisticated merging approach.
    
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


class AzureFoundryPhi4ImprovedTranscriber(BaseTranscriber):
    """Enhanced Azure Foundry Phi-4 transcriber with improved chunking for long files."""
    
    def __init__(self, model_name: str = "Phi-4-multimodal-instruct", 
                 results_dir: str = None):
        # Use safe display name for directories
        display_name = "azure-foundry-phi4-improved"
        super().__init__(display_name, results_dir)
        
        # Get credentials from environment
        self.endpoint = (
            os.getenv("AZURE_FOUNDRY_ENDPOINT")
            or os.getenv("AZURE_ENDPOINT")
            or "https://foundry-omi.services.ai.azure.com/models"
        )
        self.api_key = os.getenv("AZURE_FOUNDRY_API_KEY") or os.getenv("AZURE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Azure API key required. Set AZURE_FOUNDRY_API_KEY or AZURE_API_KEY."
            )
        self.model_name_internal = model_name
        
        # Chunking parameters adapted from Canary-Qwen
        self.chunk_duration = 30.0  # Conservative 30s chunks (shorter for API stability)
        self.overlap_duration = 8.0  # 8s overlap for good merging
        self.max_tokens = 2048  # Conservative token limit
        
        print(f"🔗 Using endpoint: {self.endpoint}")
        print(f"🤖 Using model: {self.model_name_internal}")
        print(f"⚙️ Chunking: {self.chunk_duration}s chunks with {self.overlap_duration}s overlap")
        
        # Initialize Azure AI client
        self.client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key)
        )
    
    def preprocess_audio_for_chunk(self, audio_file: str) -> str:
        """
        Preprocess audio chunk to optimal format for Azure API.
        
        Args:
            audio_file: Path to audio chunk file
            
        Returns:
            Path to preprocessed temporary file
        """
        # Create temporary file for preprocessed audio
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            mp3_path = tmp_file.name
        
        # Run ffmpeg to preprocess - more aggressive compression for API limits
        cmd = [
            'ffmpeg',
            '-i', audio_file,
            '-ar', '16000',  # 16KHz sample rate
            '-ac', '1',      # Mono
            '-c:a', 'mp3',   # MP3 codec
            '-b:a', '48k',   # Lower bitrate for smaller chunks
            '-y',            # Overwrite output
            mp3_path
        ]
        
        try:
            # Run ffmpeg quietly
            subprocess.run(cmd, capture_output=True, check=True)
            return mp3_path
            
        except subprocess.CalledProcessError as e:
            # Clean up temp file if ffmpeg failed
            if Path(mp3_path).exists():
                os.unlink(mp3_path)
            raise Exception(f"Audio preprocessing failed: {e.stderr.decode()}")
    
    def _chunk_audio(self, audio_file: str) -> Tuple[List[Tuple[str, float, float]], Path]:
        """
        Split audio into chunks with improved overlap for better merging.
        Adapted from Canary-Qwen chunking strategy.
        
        Returns:
            Tuple of (chunks_list, temp_directory)
        """
        # Load audio (16kHz mono like Canary-Qwen)
        audio, sr = librosa.load(audio_file, sr=16000)
        total_duration = len(audio) / sr
        
        # If audio is short enough, don't chunk
        if total_duration <= self.chunk_duration:
            temp_dir = Path(tempfile.mkdtemp(prefix="azure_phi4_single_"))
            single_file = temp_dir / "single.wav"
            sf.write(str(single_file), audio, sr)
            return [(str(single_file), 0.0, total_duration)], temp_dir
        
        # Chunk the audio with improved overlap
        chunks = []
        chunk_samples = int(self.chunk_duration * sr)
        overlap_samples = int(self.overlap_duration * sr)
        step_samples = chunk_samples - overlap_samples
        
        # Create temp directory for chunks
        temp_dir = Path(tempfile.mkdtemp(prefix="azure_phi4_improved_"))
        
        chunk_count = 0
        for start_sample in range(0, len(audio), step_samples):
            end_sample = min(start_sample + chunk_samples, len(audio))
            
            # Skip if remaining audio is too short (less than 5 seconds)
            if (end_sample - start_sample) / sr < 5.0 and start_sample > 0:
                break
                
            chunk_audio = audio[start_sample:end_sample]
            
            # Apply fade in/out to reduce artifacts (like Canary-Qwen)
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
        Merge chunk transcriptions using improved algorithm from Canary-Qwen.
        
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
        
        start_time = time.time()
        temp_dir = None
        
        try:
            # Get audio duration for reporting
            audio, sr = librosa.load(audio_file, sr=16000)
            total_duration = len(audio) / sr
            
            print(f"   🎵 Audio duration: {total_duration:.1f}s")
            
            # Chunk the audio with improved settings
            chunks, temp_dir = self._chunk_audio(audio_file)
            print(f"   📦 Created {len(chunks)} chunks with {self.overlap_duration}s overlap")
            
            # Process each chunk
            chunk_transcriptions = []
            
            for i, (chunk_file, chunk_start, chunk_end) in enumerate(chunks):
                preprocessed_file = None
                try:
                    print(f"   🔄 Processing chunk {i+1}/{len(chunks)} ({chunk_start:.1f}s-{chunk_end:.1f}s)")
                    
                    # Preprocess chunk for API
                    preprocessed_file = self.preprocess_audio_for_chunk(chunk_file)
                    
                    # Get file size for monitoring
                    file_size = Path(preprocessed_file).stat().st_size / (1024 * 1024)  # MB
                    print(f"      📉 Chunk size: {file_size:.1f}MB")
                    
                    # Create the chat completion request with audio
                    response = self.client.complete(
                        messages=[
                            SystemMessage("You are a speech transcription assistant."),
                            UserMessage([
                                TextContentItem(text="Capture the speech in written format in the language spoken, please. Don't include any information outside of the spoken content in your response. Remove any hesitation words like um, uh. Support mixed language. Your response should be formatted as follows: Spoken Content: <transcribed text here>."),
                                AudioContentItem(
                                    input_audio=InputAudio.load(
                                        audio_file=preprocessed_file, 
                                        audio_format=AudioContentFormat.MP3
                                    )
                                ),
                            ]),
                        ],
                        model=self.model_name_internal,
                        temperature=0.0,
                        max_tokens=self.max_tokens
                    )
                    
                    # Extract text from response
                    if response.choices and len(response.choices) > 0:
                        chunk_text = response.choices[0].message.content
                        # Remove "Spoken Content:" or "Spoken text:" prefix if present
                        if chunk_text.startswith("Spoken Content:"):
                            chunk_text = chunk_text[len("Spoken Content:"):].strip()
                        elif chunk_text.startswith("Spoken text:"):
                            chunk_text = chunk_text[len("Spoken text:"):].strip()
                    else:
                        chunk_text = ""
                    
                    if chunk_text:
                        chunk_transcriptions.append((chunk_text, chunk_start, chunk_end))
                        print(f"      ✅ Transcribed {len(chunk_text)} characters")
                    else:
                        chunk_transcriptions.append(("", chunk_start, chunk_end))
                        print(f"      ⚠️ Empty transcription for chunk {i+1}")
                    
                except Exception as e:
                    print(f"      ❌ Failed to process chunk {i+1}: {e}")
                    # Add empty transcription for this chunk
                    chunk_transcriptions.append(("", chunk_start, chunk_end))
                    continue
                finally:
                    # Clean up preprocessed chunk file
                    if preprocessed_file and Path(preprocessed_file).exists():
                        os.unlink(preprocessed_file)
            
            # Merge all chunk transcriptions with improved algorithm
            print(f"   🔗 Merging {len(chunk_transcriptions)} chunk transcriptions...")
            full_text = self._merge_chunk_transcriptions_improved(chunk_transcriptions)
            
            duration = time.time() - start_time
            
            if not full_text:
                raise ValueError("Empty transcription result from all chunks")
            
            print(f"   ✅ Final transcription: {len(full_text)} characters")
            
            return TranscriptionResult(
                text=full_text,
                duration=duration,
                model_name=self.model_name,
                audio_file=audio_file
            )
                
        except Exception as e:
            raise Exception(f"Azure Foundry Phi-4 improved transcription failed: {e}")
        finally:
            # Clean up temp directory if created
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Azure Foundry Phi-4 Improved Chunked Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--model", default="Phi-4-multimodal-instruct", help="Model name for display")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    parser.add_argument("--single_file", help="Process single file only (for testing)")
    parser.add_argument("--chunk_duration", type=float, default=30.0, help="Chunk duration in seconds")
    parser.add_argument("--overlap", type=float, default=8.0, help="Overlap duration in seconds")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per chunk")
    
    args = parser.parse_args()
    
    # Initialize transcriber
    try:
        transcriber = AzureFoundryPhi4ImprovedTranscriber(
            model_name=args.model,
            results_dir=args.results_dir
        )
        
        # Set custom parameters if provided
        transcriber.chunk_duration = args.chunk_duration
        transcriber.overlap_duration = args.overlap
        transcriber.max_tokens = args.max_tokens
        
    except Exception as e:
        print(f"❌ {e}")
        return
    
    # Process single file or batch
    if args.single_file:
        # Single file mode for testing
        try:
            print(f"🎤 Testing Azure Foundry Phi-4 IMPROVED chunking on: {args.single_file}")
            print(f"⚙️ Settings: {args.chunk_duration}s chunks, {args.overlap}s overlap, {args.max_tokens} tokens")
            result = transcriber.transcribe_file(args.single_file)
            
            # Save transcript
            transcriber._save_transcript(result)
            
            print(f"✅ Success - Duration: {result.duration:.2f}s")
            print(f"📁 Transcript saved to: {transcriber.transcripts_dir}")
            
            # Show first 300 characters of transcript
            print(f"\n📝 First 300 chars of transcript:")
            print(result.text[:300] + "..." if len(result.text) > 300 else result.text)
            
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
        
        print(f"🎤 Using Azure Foundry Phi-4 with IMPROVED chunking:")
        print(f"   - Chunk duration: {args.chunk_duration}s")
        print(f"   - Overlap: {args.overlap}s (improved from 0s)") 
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
            print(f"   - Processing speed: {len(results)/total_duration:.3f} files/second")
        else:
            print("\n❌ No files were processed successfully")


if __name__ == "__main__":
    main()
