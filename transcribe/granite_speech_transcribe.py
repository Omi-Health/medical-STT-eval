#!/usr/bin/env python3
"""
IBM Granite Speech 3.3-2b Speech Recognition Model.
Uses chunking with LCS overlap merging for long audio files.
https://huggingface.co/ibm-granite/granite-speech-3.3-2b
"""

import os
import sys
import time
import argparse
import tempfile
import warnings
from difflib import SequenceMatcher

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcribe.base_transcriber import BaseTranscriber, TranscriptionResult

try:
    import torch
    import torchaudio
    from pydub import AudioSegment
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    print(f"Missing: {e}")


class GraniteSpeechTranscriber(BaseTranscriber):
    """IBM Granite Speech 3.3-2b transcriber with chunking."""

    # Chunking settings
    CHUNK_DURATION = 35.0  # seconds
    OVERLAP = 10.0  # seconds

    def __init__(self, model_name: str = "granite-speech-3.3-2b", **kwargs):
        super().__init__(model_name=model_name, **kwargs)

        if not DEPS_AVAILABLE:
            raise ImportError("transformers, torch, torchaudio, pydub required")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading ibm-granite/granite-speech-3.3-2b on {self.device}...")
        self.processor = AutoProcessor.from_pretrained("ibm-granite/granite-speech-3.3-2b")
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "ibm-granite/granite-speech-3.3-2b",
            device_map=self.device,
            torch_dtype=torch.bfloat16
        )

        # Prepare prompt template
        system_prompt = "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
        user_prompt = "<|audio|>can you transcribe the speech into a written format?"
        chat = [
            dict(role="system", content=system_prompt),
            dict(role="user", content=user_prompt),
        ]
        self.prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        print("Model loaded successfully")

    def _find_lcs_overlap(self, text1: str, text2: str, min_overlap_words: int = 2):
        """Find LCS overlap between texts."""
        words1 = text1.split()
        words2 = text2.split()

        if len(words1) < min_overlap_words or len(words2) < min_overlap_words:
            return -1, 0.0

        best_overlap_start = -1
        best_score = 0.0
        search_window = min(len(words1), int(self.OVERLAP * 3))

        for i in range(max(0, len(words1) - search_window), len(words1)):
            for j in range(min(search_window, len(words2))):
                if words1[i] == words2[j]:
                    matcher = SequenceMatcher(None, words1[i:], words2[:j+search_window])
                    match = matcher.find_longest_match(0, len(words1[i:]), 0, len(words2[:j+search_window]))

                    if match.size >= min_overlap_words:
                        position_score = 1.0 - (j / search_window if search_window > 0 else 0)
                        length_score = match.size / search_window if search_window > 0 else 0
                        score = (position_score + length_score) / 2

                        if score > best_score:
                            best_score = score
                            best_overlap_start = i

        return best_overlap_start, best_score

    def _merge_transcripts_lcs(self, transcripts):
        """Merge transcripts using LCS algorithm."""
        if not transcripts:
            return ""
        if len(transcripts) == 1:
            return transcripts[0]

        merged = transcripts[0]
        for text in transcripts[1:]:
            if not text.strip():
                continue

            overlap_start, score = self._find_lcs_overlap(merged, text)

            if overlap_start > 0 and score > 0.3:
                words = merged.split()
                merged = ' '.join(words[:overlap_start]) + ' ' + text
            else:
                merged = merged + ' ' + text

            merged = ' '.join(merged.split())

        return merged

    def _transcribe_chunk(self, wav):
        """Transcribe a single audio chunk."""
        model_inputs = self.processor(self.prompt, wav, device=self.device, return_tensors="pt").to(self.device)
        model_outputs = self.model.generate(**model_inputs, max_new_tokens=200, do_sample=False, num_beams=1)

        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
        text = self.tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)[0]

        return text

    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio file with chunking."""
        start_time = time.time()

        try:
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            duration_s = len(audio) / 1000.0

            # Check if chunking needed
            if duration_s <= self.CHUNK_DURATION:
                # Direct transcription for short audio
                wav, sr = torchaudio.load(audio_path, normalize=True)
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)

                text = self._transcribe_chunk(wav)
            else:
                # Split into chunks
                chunk_ms = int(self.CHUNK_DURATION * 1000)
                overlap_ms = int(self.OVERLAP * 1000)
                temp_dir = tempfile.mkdtemp()

                chunks = []
                start_ms = 0
                while start_ms < len(audio):
                    end_ms = min(start_ms + chunk_ms, len(audio))
                    chunk = audio[start_ms:end_ms]

                    # Fade in/out
                    if start_ms > 0:
                        chunk = chunk.fade_in(100)
                    if end_ms < len(audio):
                        chunk = chunk.fade_out(100)

                    chunk_path = os.path.join(temp_dir, f"chunk_{len(chunks):04d}.wav")
                    chunk.export(chunk_path, format="wav")
                    chunks.append(chunk_path)

                    start_ms += chunk_ms - overlap_ms
                    if end_ms >= len(audio):
                        break

                # Transcribe each chunk
                transcripts = []
                for chunk_path in chunks:
                    wav, sr = torchaudio.load(chunk_path, normalize=True)
                    if wav.shape[0] > 1:
                        wav = wav.mean(dim=0, keepdim=True)

                    chunk_text = self._transcribe_chunk(wav)
                    transcripts.append(chunk_text)
                    os.remove(chunk_path)

                os.rmdir(temp_dir)

                # Merge transcripts
                text = self._merge_transcripts_lcs(transcripts)

            duration = time.time() - start_time

            return TranscriptionResult(
                text=text,
                duration=duration,
                model_name=self.model_name,
                audio_file=audio_path
            )

        except Exception as e:
            duration = time.time() - start_time
            print(f"Error: {e}")
            return TranscriptionResult(
                text=f"[ERROR: {str(e)}]",
                duration=duration,
                model_name=self.model_name,
                audio_file=audio_path
            )


def main():
    parser = argparse.ArgumentParser(description="Transcribe with Granite Speech 3.3-2b")
    parser.add_argument("--audio_dir", type=str, required=True)
    args = parser.parse_args()

    transcriber = GraniteSpeechTranscriber()
    audio_files = transcriber.get_audio_files(args.audio_dir)
    transcriber.transcribe_batch(audio_files)


if __name__ == "__main__":
    main()
