#!/usr/bin/env python3
"""
Kyutai STT 1B (EN/FR) PyTorch transcriber using the file-based approach.
"""

import time
import os
import json
import math
import itertools
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import torch
    import julius
except ImportError:
    print("❌ PyTorch/Julius not installed. Install with: pip install torch julius")
    exit(1)

try:
    import moshi.models
    import sphn
except ImportError:
    print("❌ Moshi not installed. Install with: pip install moshi sphn")
    exit(1)

from base_transcriber import BaseTranscriber, TranscriptionResult


class KyutaiSTT1BPyTorchTranscriber(BaseTranscriber):
    """Kyutai STT 1B PyTorch transcriber using file-based approach."""
    
    def __init__(self, 
                 hf_repo: str = "kyutai/stt-1b-en_fr",
                 device: str = None,
                 torch_dtype: str = "bfloat16",
                 results_dir: str = None):
        # Use safe display name for directories
        display_name = f"kyutai-stt-1b-pytorch-{hf_repo.split('/')[-1]}"
        super().__init__(display_name, results_dir)
        
        self.hf_repo = hf_repo
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model components
        self._load_model()
    
    def _load_model(self):
        """Load model using moshi loaders."""
        print(f"Loading Kyutai STT 1B model from {self.hf_repo}...")
        print(f"Using device: {self.device}")
        
        # Load model info from HF repo
        self.info = moshi.models.loaders.CheckpointInfo.from_hf_repo(self.hf_repo)
        
        # Load components
        self.mimi = self.info.get_mimi(device=self.device)
        self.tokenizer = self.info.get_text_tokenizer()
        self.lm = self.info.get_moshi(
            device=self.device,
            dtype=torch.bfloat16,
        )
        
        # Get STT config parameters
        self.audio_silence_prefix_seconds = self.info.stt_config.get(
            "audio_silence_prefix_seconds", 1.0
        )
        self.audio_delay_seconds = self.info.stt_config.get("audio_delay_seconds", 5.0)
        self.padding_token_id = self.info.raw_config.get("text_padding_token_id", 3)
        
        print(f"✅ Model loaded successfully on {self.device}")
        print(f"   Model size: 1B parameters (smaller and faster than 2.6B)")
        print(f"   Supports: English and French")
    
    def _load_and_process(self, audio_file: str):
        """Load and preprocess audio file."""
        audio, input_sample_rate = sphn.read(audio_file)
        audio = torch.from_numpy(audio).to(self.device).mean(axis=0, keepdim=True)
        
        # Resample to model's sample rate (24kHz)
        audio = julius.resample_frac(audio, input_sample_rate, self.mimi.sample_rate)
        
        # Pad to frame size if needed
        if audio.shape[-1] % self.mimi.frame_size != 0:
            to_pad = self.mimi.frame_size - audio.shape[-1] % self.mimi.frame_size
            audio = torch.nn.functional.pad(audio, (0, to_pad))
        
        return audio
    
    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """
        Transcribe a single audio file using Kyutai STT 1B.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            TranscriptionResult object
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        start_time = time.time()
        
        try:
            # Load and process audio
            audio = self._load_and_process(audio_file)
            
            # Create LM generator with temperature 0 for deterministic results
            lm_gen = moshi.models.LMGen(self.lm, temp=0, temp_text=0.0)
            
            # Calculate number of silence chunks
            n_prefix_chunks = math.ceil(self.audio_silence_prefix_seconds * self.mimi.frame_rate)
            n_suffix_chunks = math.ceil(self.audio_delay_seconds * self.mimi.frame_rate)
            silence_chunk = torch.zeros(
                (1, 1, self.mimi.frame_size), dtype=torch.float32, device=self.device
            )
            
            # Create chunks iterator
            chunks = itertools.chain(
                itertools.repeat(silence_chunk, n_prefix_chunks),
                torch.split(audio[:, None, :], self.mimi.frame_size, dim=-1),
                itertools.repeat(silence_chunk, n_suffix_chunks),
            )
            
            # Process chunks and accumulate text tokens
            text_tokens_accum = []
            with self.mimi.streaming(1), lm_gen.streaming(1):
                for audio_chunk in chunks:
                    audio_tokens = self.mimi.encode(audio_chunk)
                    text_tokens = lm_gen.step(audio_tokens)
                    if text_tokens is not None:
                        text_tokens_accum.append(text_tokens)
            
            # Concatenate tokens
            utterance_tokens = torch.concat(text_tokens_accum, dim=-1)
            text_tokens = utterance_tokens.cpu().view(-1)
            
            # Decode tokens (filter out padding tokens)
            transcription = self.tokenizer.decode(
                text_tokens[text_tokens > self.padding_token_id].numpy().tolist()
            )
            
            # Clean up transcription
            transcription = transcription.strip()
            
            duration = time.time() - start_time
            
            if not transcription:
                raise ValueError("Empty transcription result")
            
            return TranscriptionResult(
                text=transcription,
                duration=duration,
                model_name=self.model_name,
                audio_file=audio_file
            )
                
        except Exception as e:
            raise Exception(f"Kyutai STT 1B transcription failed: {e}")


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Kyutai STT 1B PyTorch Transcription")
    parser.add_argument("--audio_dir", help="Directory containing audio files")
    parser.add_argument("--single_file", help="Process single file only")
    parser.add_argument("--hf_repo", default="kyutai/stt-1b-en_fr", help="HF repo to load STT model from")
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    parser.add_argument("--test_wer", help="Test WER on single file with reference")
    
    args = parser.parse_args()
    
    if not args.audio_dir and not args.single_file and not args.test_wer:
        parser.error("Either --audio_dir, --single_file, or --test_wer must be specified")
    
    # Initialize transcriber
    try:
        transcriber = KyutaiSTT1BPyTorchTranscriber(
            hf_repo=args.hf_repo,
            device=args.device,
            results_dir=args.results_dir
        )
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return
    
    # Single file mode with WER testing
    if args.test_wer:
        try:
            print(f"🎤 Testing Kyutai STT 1B PyTorch on: {args.test_wer}")
            result = transcriber.transcribe_file(args.test_wer)
            
            # Save transcript
            transcriber._save_transcript(result)
            
            print(f"✅ Success - Duration: {result.duration:.2f}s")
            print(f"📁 Transcript saved to: {transcriber.transcripts_dir}")
            
            # Show first 200 characters of transcript
            print(f"\n📝 First 200 chars of transcript:")
            print(result.text[:200] + "..." if len(result.text) > 200 else result.text)
            
            # Calculate WER if possible
            audio_path = Path(args.test_wer)
            audio_stem = audio_path.stem.replace("_conversation", "")
            ref_file = Path("data/cleaned_transcripts") / f"{audio_stem}_pure_text.txt"
            
            if ref_file.exists():
                print(f"\n📊 Calculating WER...")
                # Import WER calculator
                import sys
                sys.path.append(str(Path(__file__).parent.parent / "evaluate"))
                from wer_calculator import calculate_detailed_wer
                
                with open(ref_file, 'r', encoding='utf-8') as f:
                    reference = f.read().strip()
                
                wer_result = calculate_detailed_wer(reference, result.text)
                
                print(f"\n📊 WER Results:")
                print(f"   WER: {wer_result['wer']:.2%}")
                print(f"   Word Accuracy: {1 - wer_result['wer']:.2%}")
                print(f"   Reference words: {wer_result['reference_words']}")
                print(f"   Hypothesis words: {wer_result['hypothesis_words']}")
                print(f"   Substitutions: {wer_result['substitutions']}")
                print(f"   Deletions: {wer_result['deletions']}")
                print(f"   Insertions: {wer_result['insertions']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Single file mode
    elif args.single_file:
        try:
            print(f"🎤 Processing: {args.single_file}")
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
    
    # Batch mode
    else:
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