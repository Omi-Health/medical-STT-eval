# Medical STT Benchmark

Evaluation framework for speech-to-text models on medical conversation data.

**Full write-up & leaderboard:** [omi.health/research/stt-benchmark](https://omi.health/research/stt-benchmark)
Built by [Omi Health](https://omi.health) · [All research](https://omi.health/research) · [Omi Scribe](https://omi.health/scribe)

## Leaderboard

**Dataset**: PriMock57 (55 files, ~80,500 words) | **Models**: 42 comparable single-stream models | **Updated**: 2026-04-28

### Ranked by Canonical Medical WER (M-WER)

| # | Model | WER | M-WER | Drug M-WER | Avg Speed | Type |
|---|-------|-----|-------|------------|-----------|------|
| 1 | Google Gemini 3 Pro Preview\* | 8.35% | 1.37% | 1.1% | 64.5s | API |
| 2 | Google Gemini 2.5 Pro | 8.15% | 1.52% | 1.9% | 56.4s | API |
| 3 | VibeVoice-ASR 9B | 8.34% | 1.81% | 4.5% | 96.7s | H100 |
| 4 | Google Gemini 3 Flash Preview | 11.33% | 2.03% | 3.0% | 51.5s | API |
| 5 | Soniox stt-async-v4 | 9.18% | 2.06% | 5.4% | 46.2s | API |
| 6 | ElevenLabs Scribe v2 | 9.72% | 2.54% | 2.8% | 43.5s | API |
| 7 | AssemblyAI Universal-3 Pro (medical-v1) | 9.55% | 2.83% | 4.9% | 37.3s | API |
| 8 | Qwen3 ASR 1.7B | 9.00% | 3.14% | 7.1% | 6.8s | A10 |
| 9 | Deepgram Nova-3 Medical | 9.05% | 3.17% | 7.9% | 12.9s | API |
| 10 | Microsoft MAI-Transcribe-1 | 11.52% | 3.33% | 8.8% | 21.8s | API |
| 11 | ElevenLabs Scribe v1 | 10.87% | 3.59% | 6.0% | 36.3s | API |
| 12 | Google Gemini 2.5 Flash | 9.45% | 3.65% | 8.2% | 20.2s | API |
| 13 | OpenAI GPT-4o Mini (Dec 2025) | 11.18% | 3.68% | 8.8% | 40.4s | API |
| 14 | Parakeet TDT 1.1B | 9.03% | 3.68% | 13.7% | 12.3s | T4 |
| 15 | Voxtral Mini Transcribe V1 | 11.85% | 4.03% | 9.7% | 22.4s | API |
| 16 | Voxtral Mini Transcribe V2 | 11.64% | 4.10% | 10.9% | 18.4s | API |
| 17 | Voxtral Mini 4B Realtime | 11.89% | 4.10% | 10.3% | 133.9s | A10 |
| 18 | Groq Whisper Large v3 Turbo | 12.14% | 4.32% | 12.2% | 8.0s | API |
| 19 | Cohere Transcribe (Mar 2026) | 11.81% | 4.32% | 14.8% | 3.9s | A10 |
| 20 | OpenAI Whisper-1 | 13.20% | 4.51% | 8.6% | 104.3s | API |
| 21 | NVIDIA Canary 1B Flash | 12.03% | 4.51% | 14.2% | 23.4s | T4 |
| 22 | Groq Whisper Large v3 | 11.93% | 4.60% | 11.8% | 8.6s | API |
| 23 | Parakeet TDT 0.6B v2 | 10.75% | 4.60% | 15.7% | 5.4s | Apple Silicon |
| 24 | MLX Whisper Large v3 Turbo | 11.65% | 4.70% | 12.2% | 12.9s | Apple Silicon |
| 25 | WhisperKit Large v3 Turbo | 12.28% | 4.73% | 12.4% | 21.4s | Apple Silicon |
| 26 | OpenAI GPT-4o Mini Transcribe | 13.60% | 4.86% | 9.7% | 23.2s | API |
| 27 | Qwen3 ASR 0.6B | 9.83% | 4.95% | 13.7% | 5.1s | A10 |
| 28 | Kyutai STT 2.6B | 11.20% | 5.21% | 14.0% | 148.4s | T4 |
| 29 | GLM-ASR-Nano-2512 | 10.84% | 5.75% | 16.1% | 87.7s | T4 |
| 30 | Parakeet TDT 0.6B v3 | 9.35% | 5.90% | 20.6% | 6.3s | Apple Silicon |
| 31 | Nemotron Speech Streaming 0.6B | 11.06% | 7.05% | 21.0% | 11.7s | T4 |
| 32 | OpenAI GPT-4o Transcribe | 14.84% | 7.87% | 13.5% | 27.9s | API |
| 33 | Gemma 4 E4B-it^ | 15.69% | 7.90% | 12.4% | 185.4s | T4 |
| 34 | NVIDIA Canary-Qwen 2.5B | 12.94% | 8.25% | 21.0% | 105.4s | T4 |
| 35 | NVIDIA Canary 1B v2 | 14.32% | 9.40% | 18.0% | 9.2s | T4 |
| 36 | IBM Granite Speech 3.3-2B | 16.55% | 11.02% | 22.1% | 109.7s | T4 |
| 37 | Apple SpeechAnalyzer | 12.36% | 11.97% | 25.3% | 6.0s | Apple Silicon |
| 38 | Gemma 4 E2B-it^ | 18.90% | 12.22% | 17.6% | 134.6s | T4 |
| 39 | Azure Foundry Phi-4 | 31.13% | 14.19% | 16.1% | 212.8s | API |
| 40 | Kyutai STT 1B (Multilingual) | 27.28% | 19.90% | 27.9% | 79.5s | T4 |
| 41 | Google MedASR | 52.54% | 26.16% | 35.6% | 3.9s | Apple Silicon |
| 42 | Facebook MMS-1B-all | 38.70% | 52.92% | 71.0% | 28.6s | T4 |

*Ranked by Canonical M-WER v2. **Avg Speed** = wall-clock seconds per ~7.5 min file (lower is better; not normalized for hardware tier — H100 ≫ A10 ≫ T4). **Type**: API (cloud), T4/A10/H100 (NVIDIA GPU tier via NeMo/vLLM/transformers), Apple Silicon (MLX/Native on M-series). Additional metrics in `results/metrics/{model}_medical_wer.json`.*

**\***Google Gemini 3 Pro Preview completed 54/55 comparable files.

*^Gemma 4 models use 30s chunking (model max audio = 30s).*

### Chunking Strategy

Most cloud APIs and native long-form models are evaluated on full audio. Chunking is only used when a model has an audio-length, token, memory, or decoder-behavior constraint.

The shared helpers live in `transcribe/chunking_utils.py`:
- **Post-hoc overlap + LCS merge**: Canary-Qwen, Canary Flash, Granite, and Azure Phi-4 use overlapping chunks and text-level LCS merging.
- **CTC/HF chunking**: MMS uses HF pipeline chunking with character timestamps. MedASR now supports a stronger KenLM-backed short-chunk mode (`8s` chunks, `1s` overlap) while preserving the old HF `20s/2s` baseline via `--decode_mode hf_pipeline`.
- **Simple concat**: Gemma 4 keeps non-overlapping 30s chunks because overlap/context merging tested worse for that model.
- **No chunking**: Qwen3-ASR, cloud batch APIs, and native long-form paths keep full-audio decoding when the model supports it.

MedASR chunk/overlap experiments can be run with:

```bash
python scripts/run_medasr_chunk_ablation.py --audio_dir data/raw_audio --include_hf_baseline --evaluate
```

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
| **M-WER** | Canonical Medical WER v2 — errors on medical terms only: drugs, conditions, symptoms, anatomy, clinical (lower = better) |
| **Drug M-WER** | Canonical M-WER for drug names specifically — highest clinical risk category (lower = better) |
| **cpWER** | Concatenated permutation WER — for multi-speaker models with diarization |

Additional metrics available per model in `results/metrics/{model}_medical_wer.json`: M-CER (character-level error on medical substitutions), Token Recall (occurrence-weighted), Entity Recall (binary), per-category M-WER breakdown (drugs, conditions, symptoms, anatomy, clinical).

### Canonical M-WER v2

Medical scoring uses the same base text normalization as standard WER, then applies a narrow clinical canonicalization pass before medical-term alignment:

- Corrects known reference-side PriMock57 spelling issues: `paracetemol` → `paracetamol`, `thyrocsin` → `thyroxine`, `flem` → `phlegm`.
- Collapses accepted word-boundary variants such as `water works`/`waterworks`, `hay fever`/`hayfever`, and `straight away`/`straightaway`.
- Normalizes low-risk morphology and symptom wording variants such as `tests`/`test`, `headaches`/`headache`, `itchy`/`itching`, and `tummy`/`stomach`.
- Keeps clinically meaningful or unsafe substitutions penalized, including `hyperthyroidism`/`hypothyroidism`, `modulite`/`modulate`, `clenil` misspellings, `ventolin`/`ventilin`, and `implanon`/`implant`.

This canonicalization is used only for medical metrics. Overall WER remains the standard normalized WER.

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
python evaluate/medical_wer.py --model groq-whisper-large-v3 --output results/metrics/groq-whisper-large-v3_medical_wer.json

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
├── transcribe/                 # Model-specific transcription scripts + base class
│   └── base_transcriber.py     # Shared base (loads .env)
├── evaluate/                   # Evaluation scripts
│   ├── text_normalizer.py     # Custom WER normalizer (see Metrics below)
│   ├── clinical_canonicalizer.py # Canonical M-WER v2 mappings
│   ├── wer_calculator.py
│   ├── metrics_generator.py
│   ├── medical_wer.py
│   └── comparison_generator.py
└── results/
    ├── metrics/                # WER, medical WER, and speed JSON
    ├── comparisons/            # leaderboard.json, per_file_results.json
    └── transcripts/            # tracked model transcripts for reproducibility
```

## Public vs Private Contents

The public repository contains reproducible benchmark code, reference transcripts, tracked model transcripts, metrics, and leaderboard outputs.

A local `private_analysis/` directory may exist in development checkouts. It is intentionally git-ignored and is not part of the public benchmark. Public documentation and reproducible workflows should not depend on it.

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
5. Generate medical metrics: `python evaluate/medical_wer.py --model your-model --output results/metrics/your-model_medical_wer.json`
6. Update comparisons: `python evaluate/comparison_generator.py`

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
- **M-WER**: Canonical Medical WER v2 on medical terms only (lower is better)
- **Drug M-WER**: Canonical M-WER v2 on drug terms only (lower is better)
- **Accuracy**: 1 - WER
- **Speed**: Average seconds per ~7.5 min file

### Text Normalization

WER is calculated using a custom normalizer (`evaluate/text_normalizer.py`) based on Whisper's `EnglishTextNormalizer` with two fixes:

1. **"oh" bug fix**: Whisper treats "oh" as the digit zero (`self.zeros = {"o", "oh", "zero"}`). In medical conversations "oh" is always an interjection. Our normalizer removes "oh" from this set.
2. **Word equivalence mappings**: ok/okay/k, yeah/yep/yes, mum/mom, alright/all right, kinda/kind of — variant spellings that Whisper does not normalize to the same form.

This reduced WER by ~2-3% across all models compared to stock Whisper normalization. No runtime dependency on the `openai-whisper` package.

## Citation

If you use this benchmark, please cite:

**APA** — Omi Health. (2025). *Benchmarking Speech-to-Text Models for Long-Form Medical Dialogue*. https://omi.health/research/stt-benchmark

**BibTeX**

```bibtex
@misc{omi_stt_benchmark_2025,
  title   = {Benchmarking Speech-to-Text Models for Long-Form Medical Dialogue},
  author  = {{Omi Health}},
  year    = {2025},
  url     = {https://omi.health/research/stt-benchmark},
  note    = {Medical Word Error Rate (M-WER) leaderboard, 42 models}
}
```

## License

MIT License. Dataset under CC BY 4.0.

---

Built by **[Omi Health](https://omi.health)** — the private AI stack for healthcare.
