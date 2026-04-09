# Medical STT Benchmark

Evaluation framework for speech-to-text models on medical conversation data.

## Leaderboard

**Dataset**: PriMock57 (55 files, ~80,500 words) | **Models**: 42 | **Updated**: 2026-04-09

### Ranked by Medical WER (M-WER)

| # | Model | WER | M-WER | Drug M-WER | Avg Speed | Type |
|---|-------|-----|-------|------------|-----------|------|
| 1 | Google Gemini 3 Pro Preview | 8.35% | 2.65% | 3.1% | 64.5s | API |
| 2 | Google Gemini 2.5 Pro | 8.15% | 2.97% | 4.1% | 56.4s | API |
| 3 | VibeVoice-ASR 9B | 8.34% | 3.16% | 5.6% | 96.7s | H100 |
| 4 | Soniox stt-async-v4 | 9.18% | 3.32% | 7.1% | 46.2s | API |
| 5 | Google Gemini 3 Flash Preview | 11.33% | 3.64% | 5.2% | 51.5s | API |
| 6 | ElevenLabs Scribe v2 | 9.72% | 3.86% | 4.3% | 43.5s | API |
| 7 | AssemblyAI Universal-3 Pro (medical-v1) | 9.55% | 4.02% | 6.5% | 37.3s | API |
| 8 | Qwen3 ASR 1.7B | 9.00% | 4.40% | 8.6% | 6.8s | A10 |
| 9 | Deepgram Nova-3 Medical | 9.05% | 4.53% | 9.7% | 12.9s | API |
| 10 | OpenAI GPT-4o Mini (Dec 2025) | 11.18% | 4.85% | 10.6% | 40.4s | API |
| 11 | Microsoft MAI-Transcribe-1 | 11.52% | 4.85% | 11.2% | 21.8s | API |
| 12 | ElevenLabs Scribe v1 | 10.87% | 4.88% | 7.5% | 36.3s | API |
| 13 | Google Gemini 2.5 Flash | 9.45% | 5.01% | 10.3% | 20.2s | API |
| 14 | Voxtral Mini Transcribe V1 | 11.85% | 5.17% | 11.0% | 22.4s | API |
| 15 | Parakeet TDT 1.1B | 9.03% | 5.20% | 15.5% | 12.3s | T4 |
| 16 | Voxtral Mini Transcribe V2 | 11.64% | 5.36% | 12.1% | 18.4s | API |
| 17 | Voxtral Mini 4B Realtime | 11.89% | 5.39% | 11.8% | 133.9s | H100 |
| 18 | Cohere Transcribe (Mar 2026) | 11.81% | 5.59% | 16.6% | 3.9s | A10 |
| 19 | OpenAI Whisper-1 | 13.20% | 5.62% | 10.3% | 104.3s | API |
| 20 | Groq Whisper Large v3 Turbo | 12.14% | 5.75% | 14.4% | 8.0s | API |
| 21 | NVIDIA Canary 1B Flash | 12.03% | 5.97% | 15.7% | 23.4s | T4 |
| 22 | Groq Whisper Large v3 | 11.93% | 5.97% | 13.6% | 8.6s | API |
| 23 | OpenAI GPT-4o Mini Transcribe | 13.60% | 6.03% | 11.4% | 23.2s | API |
| 24 | MLX Whisper Large v3 Turbo | 11.65% | 6.16% | 14.0% | 12.9s | Apple Silicon |
| 25 | Parakeet TDT 0.6B v2 | 10.75% | 6.19% | 17.2% | 5.4s | Apple Silicon |
| 26 | WhisperKit Large v3 Turbo | 12.28% | 6.35% | 14.4% | 21.4s | Apple Silicon |
| 27 | Qwen3 ASR 0.6B | 9.83% | 6.48% | 15.1% | 5.1s | A10 |
| 28 | Kyutai STT 2.6B | 11.20% | 6.51% | 15.7% | 148.4s | T4 |
| 29 | GLM-ASR-Nano-2512 | 10.84% | 7.05% | 17.5% | 87.7s | T4 |
| 30 | Parakeet TDT 0.6B v3 | 9.35% | 7.25% | 22.0% | 6.3s | Apple Silicon |
| 31 | Nemotron Speech Streaming 0.6B | 11.06% | 8.97% | 22.6% | 11.7s | T4 |
| 32 | OpenAI GPT-4o Transcribe | 14.84% | 9.03% | 14.9% | 27.9s | API |
| 33 | NVIDIA Canary-Qwen 2.5B | 12.94% | 9.80% | 22.8% | 105.4s | T4 |
| 34 | Gemma 4 E4B-it^ | 15.69% | 9.99% | 15.5% | 185.4s | T4 |
| 35 | NVIDIA Canary 1B v2 | 14.32% | 11.24% | 20.5% | 9.2s | T4 |
| 36 | IBM Granite Speech 3.3-2B | 16.55% | 12.80% | 23.1% | 109.7s | T4 |
| 37 | Apple SpeechAnalyzer | 12.36% | 13.02% | 27.4% | 6.0s | Apple Silicon |
| 38 | Gemma 4 E2B-it^ | 18.90% | 13.92% | 19.8% | 134.6s | T4 |
| 39 | Azure Foundry Phi-4 | 31.13% | 15.38% | 18.1% | 212.8s | API |
| 40 | Kyutai STT 1B (Multilingual) | 27.28% | 21.23% | 28.9% | 79.5s | T4 |
| 41 | Google MedASR | 64.38% | 49.66% | 58.0% | 1.2s | Apple Silicon |
| 42 | Facebook MMS-1B-all | 38.70% | 54.01% | 72.0% | 28.6s | T4 |

*Ranked by M-WER. **Avg Speed** = wall-clock seconds per ~7.5 min file (lower is better; not normalized for hardware tier — H100 ≫ A10 ≫ T4). **Type**: API (cloud), T4/A10/H100 (NVIDIA GPU tier via NeMo/vLLM/transformers), Apple Silicon (MLX/Native on M-series). Additional metrics in `results/metrics/{model}_medical_wer.json`.*
*^Gemma 4 models use 30s chunking (model max audio = 30s).*

### Multi-speaker models (separate — different metric)

These models output per-speaker transcripts with diarization, evaluated with cpWER (concatenated permutation WER) instead of standard WER. Not directly comparable to the single-stream leaderboard above.

| Model | cpWER | Doctor WER | Patient WER | Good diar (<30%) | Notes |
|-------|-------|------------|-------------|------------------|-------|
| Multitalker Parakeet 0.6B | 34.17% | 13-20% | 19-32% | 30/55 (55%) | Joint ASR+diar, streaming, NeMo |

The model outputs SegLST segments (speaker-tagged words with timestamps) via `SpeakerTaggedASR`. On well-diarized files (30/55), per-speaker WER is competitive with single-speaker Parakeet models. On 12/55 files diarization fails — the model assigns text to the wrong speaker or merges both into one stream. This is an early streaming model (v1) optimized for real-time display.

Per-speaker references: PriMock57 TextGrid files (doctor + patient separately with timestamps).

### Metrics explained

| Metric | What |
|--------|------|
| **WER** | Word Error Rate — overall transcription accuracy (lower = better) |
| **M-WER** | Medical WER — errors on medical terms only: drugs, conditions, symptoms, anatomy, clinical (lower = better) |
| **Drug M-WER** | M-WER for drug names specifically — highest clinical risk category (lower = better) |
| **cpWER** | Concatenated permutation WER — for multi-speaker models with diarization |

Additional metrics available per model in `results/metrics/{model}_medical_wer.json`: M-CER (character-level error on medical substitutions), Token Recall (occurrence-weighted), Entity Recall (binary), per-category M-WER breakdown (drugs, conditions, symptoms, anatomy, clinical).

## Quick Start

```bash
# Install Git LFS (required for audio files)
git lfs install

# Clone and install
git clone https://github.com/Omi-Health/medical-STT-eval.git
cd medical-STT-eval
pip install -r requirements.txt

# Add API keys
cp .env.example .env

# Run transcription (outputs to results/transcripts/)
python transcribe/groq_whisper_transcribe.py --audio_dir data/raw_audio

# Generate metrics
python evaluate/metrics_generator.py --model_name groq-whisper-large-v3

# Update leaderboard
python evaluate/comparison_generator.py
```

**Note**: Existing per-model transcripts are tracked in `results/transcripts/` for reproducibility. To re-evaluate or add a new model, run the relevant transcription script then `metrics_generator.py` and `medical_wer.py`.

## Project Structure

```
medical-stt-benchmark/
├── data/
│   ├── raw_audio/              # 57 WAV files (Git LFS)
│   └── cleaned_transcripts/    # 57 reference transcripts
├── transcribe/                 # 18 model scripts + base class
│   └── base_transcriber.py     # Shared base (loads .env)
├── evaluate/                   # Evaluation scripts
│   ├── text_normalizer.py     # Custom WER normalizer (see Metrics below)
│   ├── wer_calculator.py
│   ├── metrics_generator.py
│   └── comparison_generator.py
└── results/
    ├── metrics/                # WER + speed JSON (per model)
    └── comparisons/            # leaderboard.json
```

## Supported Platforms

| Type | Models | Setup |
|------|--------|-------|
| **API** | OpenAI, Groq, ElevenLabs, Google, Mistral, Deepgram, AssemblyAI, Soniox, Microsoft (MAI-Transcribe via Azure Speech) | Add keys to `.env` |
| **MLX** | Whisper, Parakeet, WhisperKit | Apple Silicon required |
| **GPU** | NVIDIA Canary, Kyutai STT, VibeVoice, Nemotron, Voxtral Realtime | CUDA + NeMo/vLLM/transformers |
| **Native** | Apple SpeechAnalyzer | macOS 26+ |

## Adding a New Model

1. Create `transcribe/your_model_transcribe.py` inheriting from `BaseTranscriber`
2. Implement `transcribe_file()` returning transcript text
3. Run on dataset: `python transcribe/your_model_transcribe.py --audio_dir data/raw_audio`
4. Generate metrics: `python evaluate/metrics_generator.py --model_name your-model`
5. Update comparisons: `python evaluate/comparison_generator.py`

## Dataset

**PriMock57**: 57 doctor-patient consultations, 81,236 words of British English medical dialogue.
55 files used for evaluation (2 excluded due to processing issues).

Audio files are tracked with Git LFS. Reference transcripts derived from PriMock57 under CC BY 4.0.

**Citation**:
```
@inproceedings{korfiatis2022primock57,
  title={PriMock57: A Dataset Of Primary Care Mock Consultations},
  author={Papadopoulos Korfiatis, Alex and Moramarco, Francesco and Sarac, Radmila and Savkov, Aleksandar},
  booktitle={Proceedings of the 60th Annual Meeting of the ACL},
  year={2022}
}
```

## Metrics

- **WER**: Word Error Rate (lower is better)
- **Accuracy**: 1 - WER
- **Speed**: Average seconds per ~7.5 min file

### Text Normalization

WER is calculated using a custom normalizer (`evaluate/text_normalizer.py`) based on Whisper's `EnglishTextNormalizer` with two fixes:

1. **"oh" bug fix**: Whisper treats "oh" as the digit zero (`self.zeros = {"o", "oh", "zero"}`). In medical conversations "oh" is always an interjection. Our normalizer removes "oh" from this set.
2. **Word equivalence mappings**: ok/okay/k, yeah/yep/yes, mum/mom, alright/all right, kinda/kind of — variant spellings that Whisper does not normalize to the same form.

This reduced WER by ~2-3% across all models compared to stock Whisper normalization. No runtime dependency on the `openai-whisper` package.

## License

MIT License. Dataset under CC BY 4.0.
