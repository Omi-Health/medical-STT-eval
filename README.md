# Medical STT Benchmark

Evaluation framework for speech-to-text models on medical conversation data.

## Leaderboard

**Dataset**: PriMock57 (55 files, ~80,500 words) | **Models**: 37 | **Updated**: 2026-04-06

### Ranked by Medical WER (M-WER)

| # | Model | WER | M-WER | Drug M-WER |
|---|-------|-----|-------|------------|
| 1 | Google Gemini 3 Pro Preview | 8.35% | 2.65% | 3.1% |
| 2 | Google Gemini 2.5 Pro | 8.15% | 2.97% | 4.1% |
| 3 | VibeVoice-ASR 9B | 8.34% | 3.16% | 5.6% |
| 4 | Google Gemini 3 Flash Preview | 11.33% | 3.64% | 5.2% |
| 5 | ElevenLabs Scribe v2 | 9.72% | 3.86% | 4.3% |
| 6 | Qwen3 ASR 1.7B | 8.96% | 4.69% | 9.3% |
| 7 | OpenAI GPT-4o Mini (Dec 2025) | 11.18% | 4.85% | 10.6% |
| 8 | ElevenLabs Scribe v1 | 10.87% | 4.88% | 7.5% |
| 9 | Google Gemini 2.5 Flash | 9.45% | 5.01% | 10.3% |
| 10 | Voxtral Mini Transcribe V1 (chat) | 11.85% | 5.17% | 11.0% |
| 11 | Parakeet TDT 1.1B | 9.03% | 5.20% | 15.5% |
| 12 | Voxtral Mini Transcribe V1 (API) | 11.87% | 5.20% | 11.0% |
| 13 | Voxtral Mini Transcribe V2 | 11.64% | 5.36% | 12.1% |
| 14 | Voxtral Mini 4B Realtime | 11.89% | 5.39% | 11.9% |
| 15 | Cohere Transcribe (Mar 2026) | 11.81% | 5.59% | 16.6% |
| 16 | OpenAI Whisper-1 | 13.20% | 5.62% | 10.3% |
| 17 | Groq Whisper Large v3 Turbo | 12.14% | 5.75% | 14.4% |
| 18 | NVIDIA Canary 1B Flash | 12.03% | 5.97% | 15.7% |
| 19 | Groq Whisper Large v3 | 11.93% | 5.97% | 13.6% |
| 20 | OpenAI GPT-4o Mini Transcribe | 13.60% | 6.03% | 11.4% |
| 21 | MLX Whisper Large v3 Turbo | 11.65% | 6.16% | 14.0% |
| 22 | Parakeet TDT 0.6B v2 | 10.75% | 6.19% | 17.2% |
| 23 | WhisperKit Large v3 Turbo | 12.28% | 6.35% | 14.4% |
| 24 | Kyutai STT 2.6B | 11.20% | 6.51% | 15.7% |
| 25 | Parakeet TDT 0.6B v3 | 9.35% | 7.25% | 22.0% |
| 26 | Qwen3 ASR 0.6B | 10.04% | 8.04% | 17.9% |
| 27 | Nemotron Speech Streaming 0.6B | 11.06% | 8.97% | 22.6% |
| 28 | OpenAI GPT-4o Transcribe | 14.84% | 9.03% | 14.9% |
| 29 | Gemma 4 E4B-it^ | 15.69% | 9.99% | 15.5% |
| 30 | NVIDIA Canary-Qwen 2.5B | 12.94% | 9.80% | 22.8% |
| 31 | NVIDIA Canary 1B v2 | 14.32% | 11.24% | 20.5% |
| 32 | IBM Granite Speech 3.3-2B | 16.55% | 12.80% | 23.1% |
| 33 | Apple SpeechAnalyzer | 12.36% | 13.02% | 27.4% |
| 34 | Gemma 4 E2B-it^ | 18.90% | 13.92% | 19.8% |
| 35 | Azure Foundry Phi-4 | 31.13% | 15.38% | 18.1% |
| 36 | Kyutai STT 1B (Multilingual) | 27.28% | 21.23% | 28.9% |
| 37 | Google MedASR | 64.38% | 49.66% | 58.0% |

*Ranked by M-WER. Additional metrics (M-CER, Token Recall, Entity Recall, per-category breakdown) available in `results/metrics/{model}_medical_wer.json`.*
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

**Note**: Transcripts are generated on-demand and not stored in the repo. Run transcription first before generating metrics.

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
| **API** | OpenAI, Groq, ElevenLabs, Google, Mistral | Add keys to `.env` |
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
