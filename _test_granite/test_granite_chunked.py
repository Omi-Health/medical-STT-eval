#!/usr/bin/env python3
"""Test IBM Granite Speech with chunking - following canary_1b_flash_improved approach."""

import torch
import torchaudio
import time
from pydub import AudioSegment
import tempfile
import os
from difflib import SequenceMatcher
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model_name = "ibm-granite/granite-speech-3.3-2b"

print("Loading Granite Speech 3.3-2b model...")
start = time.time()

processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name, device_map=device, torch_dtype=torch.bfloat16
)
print(f"Model loaded in {time.time() - start:.1f}s")

# Chunking settings (same as canary_1b_flash_improved)
CHUNK_DURATION = 35.0  # seconds
OVERLAP = 10.0  # seconds

def find_lcs_overlap(text1, text2, min_overlap_words=2):
    """Find LCS overlap between texts."""
    words1 = text1.split()
    words2 = text2.split()

    if len(words1) < min_overlap_words or len(words2) < min_overlap_words:
        return -1, -1, 0.0

    best_overlap_start = -1
    best_score = 0.0
    search_window = min(len(words1), int(OVERLAP * 3))  # ~3 words per second

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


def merge_transcripts_lcs(transcripts):
    """Merge transcripts using LCS algorithm."""
    if not transcripts:
        return ""
    if len(transcripts) == 1:
        return transcripts[0]

    merged = transcripts[0]
    for text in transcripts[1:]:
        if not text.strip():
            continue

        overlap_start, score = find_lcs_overlap(merged, text)

        if overlap_start > 0 and score > 0.3:
            words = merged.split()
            merged = ' '.join(words[:overlap_start]) + ' ' + text
        else:
            merged = merged + ' ' + text

        merged = ' '.join(merged.split())

    return merged


# Prepare prompt
system_prompt = "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
user_prompt = "<|audio|>can you transcribe the speech into a written format?"
chat = [
    dict(role="system", content=system_prompt),
    dict(role="user", content=user_prompt),
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# Test file
test_file = "/root/medical-STT-eval/data/raw_audio/day1_consultation01_conversation.wav"

print(f"\nLoading audio: {test_file}")
audio = AudioSegment.from_file(test_file)
duration_s = len(audio) / 1000.0
print(f"Audio duration: {duration_s:.1f}s")

# Split into chunks
chunk_ms = int(CHUNK_DURATION * 1000)
overlap_ms = int(OVERLAP * 1000)
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
    chunks.append((chunk_path, start_ms/1000, end_ms/1000))

    start_ms += chunk_ms - overlap_ms
    if end_ms >= len(audio):
        break

print(f"Split into {len(chunks)} chunks ({CHUNK_DURATION}s with {OVERLAP}s overlap)")

# Transcribe each chunk
transcripts = []
total_start = time.time()

for i, (chunk_path, start_t, end_t) in enumerate(chunks):
    print(f"\nChunk {i+1}/{len(chunks)} ({start_t:.1f}s - {end_t:.1f}s)")

    # Load chunk
    wav, sr = torchaudio.load(chunk_path, normalize=True)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    chunk_start = time.time()

    # Transcribe
    model_inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)
    model_outputs = model.generate(**model_inputs, max_new_tokens=200, do_sample=False, num_beams=1)

    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
    text = tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)[0]

    transcripts.append(text)
    print(f"  {time.time() - chunk_start:.1f}s: {text[:80]}...")

    os.remove(chunk_path)

os.rmdir(temp_dir)

# Merge transcripts
full_text = merge_transcripts_lcs(transcripts)
total_duration = time.time() - total_start

print(f"\n{'='*60}")
print(f"FULL TRANSCRIPT ({len(chunks)} chunks, {total_duration:.1f}s total):")
print('='*60)
print(full_text[:3000] + "..." if len(full_text) > 3000 else full_text)
print(f"\n{'='*60}")
print(f"Total characters: {len(full_text)}")
print(f"Speed: {duration_s/total_duration:.2f}x realtime")
