# Medical STT Benchmark

Evaluation framework for speech-to-text models on medical conversation data.

## Leaderboard

**Dataset**: PriMock57 (55 files, ~80,500 words) | **Models**: 31 | **Updated**: 2026-03-27

| Rank | Model | WER | Accuracy | Avg Speed | GPU |
|------|-------|-----|----------|-----------|-----|
| 1 | Google Gemini 2.5 Pro | 8.15% | 91.85% | 56.4s | API |
| 2 | Microsoft VibeVoice-ASR 9B | 8.34% | 91.66% | 96.7s | H100* |
| 3 | Google Gemini 3 Pro Preview** | 8.35% | 91.65% | 64.5s | API |
| 4 | Parakeet TDT 0.6B v3 | 9.35% | 90.65% | 6.3s | Apple Silicon |
| 5 | Google Gemini 2.5 Flash | 9.45% | 90.55% | 20.2s | API |
| 6 | ElevenLabs Scribe v2 | 9.72% | 90.28% | 43.5s | API |
| 7 | Parakeet TDT 0.6B v2 | 10.75% | 89.25% | 5.4s | Apple Silicon |
| 8 | ElevenLabs Scribe v1 | 10.87% | 89.13% | 36.3s | API |
| 9 | NVIDIA Nemotron Speech Streaming 0.6B | 11.06% | 88.94% | 11.7s | T4 |
| 10 | OpenAI GPT-4o Mini (2025-12-15) | 11.18% | 88.82% | 40.4s | API |
| 11 | Kyutai STT 2.6B | 11.20% | 88.80% | 148.4s | GPU |
| 12 | Google Gemini 3 Flash Preview | 11.33% | 88.67% | 51.5s | API |
| 13 | Voxtral Mini 2602 (Transcription API) | 11.64% | 88.36% | 18.4s | API |
| 14 | MLX Whisper Large v3 Turbo | 11.65% | 88.35% | 12.9s | Apple Silicon |
| 15 | Mistral Voxtral Mini | 11.85% | 88.15% | 22.4s | API |
| 16 | Mistral Voxtral Mini (Transcription) | 11.87% | 88.13% | 23.0s | API |
| 17 | Voxtral Mini 4B Realtime (vLLM)*** | 11.89% | 88.11% | 133.9s / 693s | H100* / T4 |
| 18 | Groq Whisper Large v3 | 11.93% | 88.07% | 8.6s | API |
| 19 | NVIDIA Canary 1B Flash | 12.03% | 87.97% | 23.4s | T4 |
| 20 | Groq Whisper Large v3 Turbo | 12.14% | 87.86% | 8.0s | API |
| 21 | WhisperKit Large v3 Turbo | 12.28% | 87.72% | 21.4s | Apple Silicon |
| 22 | Apple SpeechAnalyzer | 12.36% | 87.64% | 6.0s | Apple Silicon |
| 23 | NVIDIA Canary-Qwen 2.5B | 12.94% | 87.06% | 105.4s | T4 |
| 24 | OpenAI Whisper-1 | 13.20% | 86.80% | 104.3s | API |
| 25 | OpenAI GPT-4o Mini Transcribe | 13.60% | 86.40% | N/A | API |
| 26 | NVIDIA Canary 1B v2**** | 14.32% | 85.68% | 9.2s | T4 |
| 27 | OpenAI GPT-4o Transcribe | 14.84% | 85.16% | 27.9s | API |
| 28 | IBM Granite Speech 3.3-2b***** | 16.55% | 83.45% | 109.7s | T4 |
| 29 | Kyutai STT 1B (Multilingual) | 27.28% | 72.72% | 79.5s | GPU |
| 30 | Azure Foundry Phi-4 | 31.13% | 68.87% | 212.8s | API |
| 31 | Google MedASR | 64.38% | 35.62% | 4.5s | Apple Silicon |

*\*H100 GPU — not directly comparable with T4/Apple Silicon benchmarks*
*\*\*54/55 files evaluated (1 blocked by safety filter)*
*\*\*\*Designed for streaming/realtime use — slow batch speed is expected*
*\*\*\*\*3 files with hallucination loops (see AGENTS.md)*
*\*\*\*\*\*Requires chunking to avoid repetition loops*

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
