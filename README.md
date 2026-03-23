# Medical STT Benchmark

Evaluation framework for speech-to-text models on medical conversation data.

## Leaderboard

**Dataset**: PriMock57 (55 files, 81,236 words) | **Models**: 31 | **Updated**: 2026-03-23

| Rank | Model | WER | Accuracy | Avg Speed | Type |
|------|-------|-----|----------|-----------|------|
| 1 | Google Gemini 2.5 Pro | 10.79% | 89.21% | 56.4s | API |
| 2 | Microsoft VibeVoice-ASR 9B | 11.02% | 88.98% | N/A* | GPU |
| 3 | Google Gemini 3 Pro Preview** | 11.03% | 88.97% | 64.5s | API |
| 4 | Parakeet TDT 0.6B v3 | 11.90% | 88.10% | 6.3s | MLX |
| 5 | Google Gemini 2.5 Flash | 12.08% | 87.92% | 20.2s | API |
| 6 | ElevenLabs Scribe v2 | 12.36% | 87.64% | 43.5s | API |
| 7 | OpenAI GPT-4o Mini (2025-12-15) | 12.82% | 87.18% | 40.5s | API |
| 8 | Parakeet TDT 0.6B v2 | 13.26% | 86.74% | 5.4s | MLX |
| 9 | NVIDIA Nemotron Speech Streaming 0.6B | 13.38% | 86.62% | 11.7s | GPU |
| 10 | ElevenLabs Scribe v1 | 13.54% | 86.46% | 36.3s | API |
| 11 | Kyutai STT 2.6B | 13.79% | 86.21% | 148.4s | GPU |
| 12 | Google Gemini 3 Flash Preview | 13.88% | 86.12% | 51.5s | API |
| 13 | Voxtral Mini 2602 (Transcription API) | 14.17% | 85.83% | 18.4s | API |
| 14 | MLX Whisper Large v3 Turbo | 14.22% | 85.78% | 12.9s | MLX |
| 15 | Groq Whisper Large v3 | 14.30% | 85.70% | 8.6s | API |
| 16 | Mistral Voxtral Mini | 14.35% | 85.65% | 22.4s | API |
| 17 | Mistral Voxtral Mini (Transcription) | 14.37% | 85.63% | 23.0s | API |
| 18 | Voxtral Mini 4B Realtime (vLLM) | 14.39% | 85.61% | N/A* | GPU |
| 19 | NVIDIA Canary 1B Flash | 14.46% | 85.54% | 23.4s | GPU |
| 20 | Groq Whisper Large v3 Turbo | 14.50% | 85.50% | 8.0s | API |
| 21 | WhisperKit Large v3 Turbo | 14.55% | 85.45% | 21.4s | MLX |
| 22 | Apple SpeechAnalyzer | 14.75% | 85.25% | 6.0s | Native |
| 23 | NVIDIA Canary-Qwen 2.5B | 15.45% | 84.55% | 105.4s | GPU |
| 24 | OpenAI Whisper-1 | 15.49% | 84.51% | 104.3s | API |
| 25 | OpenAI GPT-4o Mini Transcribe | 15.96% | 84.04% | N/A | API |
| 26 | NVIDIA Canary 1B v2*** | 16.80% | 83.20% | 9.2s | GPU |
| 27 | OpenAI GPT-4o Transcribe | 17.14% | 82.86% | 27.9s | API |
| 28 | IBM Granite Speech 3.3-2b**** | 18.92% | 81.08% | 109.7s | GPU |
| 29 | Kyutai STT 1B (Multilingual) | 29.41% | 70.59% | 79.5s | GPU |
| 30 | Azure Foundry Phi-4 | 33.13% | 66.87% | 212.8s | API |
| 31 | Google MedASR | 64.88% | 35.12% | 1.2s | Local |

*Speed measured on H100 GPU (not comparable with T4 benchmarks); see AGENTS.md for details
**54/55 files evaluated (1 blocked by safety filter)
***3 files with hallucination loops (see AGENTS.md for details)
****Requires chunking to avoid repetition loops

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
- **Speed**: Average seconds per file
- Uses Whisper's `EnglishTextNormalizer` for fair comparison

## License

MIT License. Dataset under CC BY 4.0.
