#!/usr/bin/env python3
"""Test IBM Granite Speech 3.3-2b model - following model card exactly."""

import torch
import torchaudio
import time
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

# Test file
test_file = "/root/medical-STT-eval/data/raw_audio/day1_consultation01_conversation.wav"

print(f"\nLoading audio: {test_file}")
wav, sr = torchaudio.load(test_file, normalize=True)
print(f"Audio shape: {wav.shape}, Sample rate: {sr}")

# Resample to 16kHz if needed
if sr != 16000:
    print(f"Resampling from {sr} to 16000...")
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    wav = resampler(wav)
    sr = 16000

# Convert to mono if stereo
if wav.shape[0] > 1:
    print("Converting to mono...")
    wav = wav.mean(dim=0, keepdim=True)

print(f"Final audio shape: {wav.shape}, Sample rate: {sr}")

# Create text prompt
system_prompt = "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
user_prompt = "<|audio|>can you transcribe the speech into a written format?"
chat = [
    dict(role="system", content=system_prompt),
    dict(role="user", content=user_prompt),
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

print(f"\nTranscribing...")
start = time.time()

# Run the processor+model - using max_new_tokens=200 per model card
model_inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)
model_outputs = model.generate(**model_inputs, max_new_tokens=200, do_sample=False, num_beams=1)

# Transformers includes the input IDs in the response
num_input_tokens = model_inputs["input_ids"].shape[-1]
new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
output_text = tokenizer.batch_decode(
    new_tokens, add_special_tokens=False, skip_special_tokens=True
)

duration = time.time() - start

print(f"\nTranscription completed in {duration:.1f}s")
print(f"\n{'='*60}")
print("TRANSCRIPT:")
print('='*60)
text = output_text[0]
print(f"STT output = {text.upper()}")
print(f"\n{'='*60}")
print(f"Total characters: {len(text)}")
