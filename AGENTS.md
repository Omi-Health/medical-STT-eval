# Medical STT Benchmark - Technical Context

AI context document for understanding this codebase.

## What This Project Does

Evaluates speech-to-text models on **PriMock57** - 57 doctor-patient consultations (81,236 words of British English medical dialogue). Goal: determine which models perform best for medical transcription.

## Project Structure

```
medical-stt-benchmark/
├── data/
│   ├── raw_audio/              # 57 WAV files (~13.9MB each, Git LFS)
│   └── cleaned_transcripts/    # 57 reference transcripts (*_pure_text.txt)
├── transcribe/                 # 15 model-specific scripts + base class
│   ├── base_transcriber.py     # Base class (loads .env automatically)
│   └── *_transcribe.py         # One per model/API
├── evaluate/
│   ├── wer_calculator.py       # WER calculation algorithm
│   ├── metrics_generator.py    # Generates *_wer.json per model
│   └── comparison_generator.py # Generates leaderboard.json
└── results/
    ├── metrics/                # 38 JSON files (19 models × 2: speed + wer)
    └── comparisons/            # leaderboard.json, per_file_results.json
```

## Dataset Notes

### Dataset Cleaning Process
1. **Original transcripts**: Had timestamps and speaker labels
2. **Cleaned to pure text**: Removed all formatting, keeping only spoken words
3. **Normalized**: Consistent punctuation and capitalization for fair WER calculation
4. **Validated**: Each audio file has corresponding reference transcript

### Problematic Files - IMPORTANT
Two files cause issues for most models (13 out of 15 models failed to process them):
- `day1_consultation07`
- `day3_consultation03`

**How these are handled**:
- The `comparison_generator.py` explicitly excludes these files
- Only 55 files are used for fair cross-model comparison
- Only NVIDIA Canary models successfully processed these files

## Evaluation Workflow

```bash
# 1. Transcribe (outputs to results/transcripts/ + results/metrics/*_speed.json)
python transcribe/groq_whisper_transcribe.py --audio_dir data/raw_audio

# 2. Calculate WER (outputs results/metrics/*_wer.json)
python evaluate/metrics_generator.py --model_name groq-whisper-large-v3

# 3. Update leaderboard (outputs results/comparisons/leaderboard.json)
python evaluate/comparison_generator.py
```

## Key Technical Learning: Advanced Chunking Strategy

### The Problem Discovered
Long medical conversations (>30 seconds) caused major issues:
- **Token/time limits**: Models exceeded API or memory constraints
- **Repetition loops**: Some models got stuck repeating phrases hundreds of times
- **Quality degradation**: Accuracy dropped significantly on longer files

### The Solution Developed
**Sophisticated chunking with overlap merging** - inspired by Groq's approach:

1. **Intelligent chunking**: Split audio into overlapping segments (30-35 seconds)
2. **Overlap merging**: Use longest common sequence (LCS) algorithm to merge transcriptions
3. **Audio processing**: Apply fade-in/fade-out to reduce artifacts
4. **Model-specific tuning**: Adjust parameters based on each model's constraints

### Why Our Chunking Method Outperforms NVIDIA's Default
We developed our own chunking implementation for NVIDIA Canary models instead of using the default NEMO chunking because:

1. **Smarter Word Boundary Detection**: Our method uses silence detection and natural speech patterns to find optimal chunk boundaries, avoiding mid-word or mid-phrase cuts
2. **Optimized Overlap Strategy**: We use 10-second overlaps with LCS merging, which preserves context better than NVIDIA's default 2-second overlaps
3. **Fade Effects**: Audio fade-in/fade-out reduces artifacts at chunk boundaries, improving transcription accuracy
4. **Context Preservation**: Longer overlaps ensure medical terminology and patient context isn't lost between chunks

The improved chunking is particularly important for medical conversations where:
- Technical terms span chunk boundaries
- Context from previous utterances affects interpretation
- Natural pauses don't align with fixed time intervals

## Chunking vs Full Audio Processing

### Models Requiring Chunking (constrained)

**NVIDIA Canary-Qwen (vLLM)**
- `canary_qwen_improved_transcribe.py` - 35s chunks, 10s overlap
- **Constraint**: 40s audio limit + 1024 token limit in vLLM environment

**NVIDIA Canary 1B Flash**
- `canary_1b_flash_improved_transcribe.py` - 35s chunks, 10s overlap
- Same NeMo framework, identical chunking strategy

**Azure Foundry Phi-4 (API)**
- `azure_foundry_phi4_transcribe.py` - 30s chunks, 8s overlap
- **Constraint**: API stability + token limits for multimodal processing

### Models Handling Full Audio Natively

**Cloud APIs** (Robust, handle long audio natively):
- **OpenAI**: `openai_api_transcribe.py` (Whisper-1, GPT-4o variants)
- **Groq**: `groq_whisper_transcribe.py` (Whisper Large V3/Turbo)
- **ElevenLabs**: `elevenlabs_scribe_transcribe.py` (Scribe v1)
- **Mistral**: `mistral_voxtral_mini_transcribe.py` (Voxtral models via API)
- **Google**: `gemini_transcribe.py` (Gemini 2.5 Flash/Pro)

**Local/Native Models** (Optimized for long audio):
- **MLX Whisper**: `mlx_whisper_transcribe.py` (Apple Silicon optimized)
- **Apple**: `apple_speechanalyzer_transcribe.py` (Native macOS framework)
- **WhisperKit**: `whisperkit_transcribe.py` (On-device inference)
- **Parakeet**: `parakeet_transcribe.py` (MLX research model)

### Key Insights
- **Chunking**: Only needed for models with strict constraints (vLLM, resource-limited APIs)
- **Full Audio**: Preferred by robust cloud APIs and optimized local models
- **Overlap Merging**: Critical for maintaining context in medical conversations
- **Model-Specific**: Chunk size depends on each model's technical limitations

## Model-to-Script Mapping

| Script | Models Served |
|--------|---------------|
| `openai_api_transcribe.py` | whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe |
| `groq_whisper_transcribe.py` | whisper-large-v3, whisper-large-v3-turbo |
| `gemini_transcribe.py` | gemini-2.5-flash, gemini-2.5-pro |
| `mistral_voxtral_mini_transcribe.py` | voxtral-mini (chat endpoint) |
| `mistral_voxtral_mini_transcription_transcribe.py` | voxtral-mini (transcription endpoint) |
| `kyutai_stt_pytorch_transcribe.py` | stt-2.6b-en |
| `kyutai_stt_1b_pytorch_transcribe.py` | stt-1b-en_fr |
| `canary_qwen_improved_transcribe.py` | canary-qwen-2.5b |
| `canary_1b_flash_improved_transcribe.py` | canary-1b-flash |
| `canary_1b_v2_transcribe.py` | canary-1b-v2 (native long-form) |
| `granite_speech_transcribe.py` | granite-speech-3.3-2b (chunked) |
| Others | 1:1 mapping |

## API Quirks and Learnings

### Mistral Voxtral
- **API Transcription**: `mistral_voxtral_mini_transcription_transcribe.py` - Uses `/v1/audio/transcriptions` endpoint
- **Chat-based**: `mistral_voxtral_mini_transcribe.py` - Uses chat completions with base64 encoded audio
- **Key Finding**: Only Mini supports `/v1/audio/transcriptions` - documentation says both models support it but reality is different
- **Workaround**: Small model must use chat completions with audio input

### Google Gemini
- **Smart file handling**: Auto-detects file size and chooses upload vs inline processing
- **File size threshold**: 15MB (conservative limit for 20MB total request size)
- **Upload method**: Uses Files API for large files, inline bytes for smaller files
- **Prompt engineering**: Specific prompt needed to avoid commentary and formatting in transcripts

### Kyutai STT - The Hallucination Problem
The Kyutai STT 2.6B model showed severe hallucination issues on long medical conversations:
- **Pattern**: Good transcription for ~2-3 minutes, then repetitive token loops
- **Symptoms**: Thousands of repeated "refore" tokens, file sizes reaching 100MB+
- **Root cause**: Autoregressive model failure mode on long audio sequences

**MLX vs PyTorch API Differences**:
- **Reference code**: Uses `moshi.models.loaders.CheckpointInfo` (PyTorch version)
- **MLX version**: No `loaders` module, different streaming API
- **Solution**: PyTorch implementation works reliably, MLX version hallucinates

### API Reliability Patterns
- **Most reliable**: OpenAI, Groq (rarely fail)
- **Occasional 503s**: Mistral (especially during batch processing)
- **Consistent**: ElevenLabs, Google Gemini
- **Local models**: MLX models most stable for batch processing

### File Size Handling
- **Cloud APIs**: Generally handle 13.9MB medical files well
- **Chunking required**: NVIDIA vLLM models, Azure Phi-4
- **Upload vs inline**: Gemini automatically chooses based on file size

## WER Evaluation: Whisper Normalization

### Problem Identified
The original evaluation framework was unfairly penalizing models for correctly omitting filler words and handling text normalization differently:
- **Filler words**: Models correctly ignored "um", "uh", "ah" but were penalized in WER calculation
- **Number formats**: "23" vs "twenty three" variations caused false errors
- **Contractions**: "don't" vs "do not" inconsistencies
- **Result**: ~5-6% higher WER scores than industry standard

### Solution Implemented
Upgraded the entire evaluation framework to use **Whisper's EnglishTextNormalizer** as the industry standard:
- Handles filler words ("um", "uh")
- Normalizes numbers ("23" ↔ "twenty three")
- Expands contractions ("don't" → "do not")
- Symbol handling ("$100" → "one hundred dollars")
- Abbreviation expansion ("Dr." → "doctor")

### Impact
- **Consistent 5-6% WER improvement** across all models
- **More realistic evaluation**: Aligns with industry standards (Whisper, OpenAI)
- **Fair comparison**: Models no longer penalized for correct behavior

### Before vs After Examples
- **ElevenLabs Scribe**: 19.97% → 13.54% WER (-6.43%)
- **Google Gemini Pro**: 16.11% → 10.90% WER (-5.21%)
- **Kyutai 2.6B**: 19.25% → 13.98% WER (-5.27%)

## Performance Patterns

- **Best accuracy**: Google Gemini 2.5 Pro (10.79% WER)
- **Best speed**: Parakeet TDT 0.6B (5.4s avg)
- **Best speed/accuracy tradeoff**: Groq Whisper (8.6s, 14.30% WER)
- **Most sophisticated chunking**: NVIDIA Canary models with 10s overlap + LCS merging

Cloud APIs generally handle long audio better than local models requiring chunking, but local MLX models offer the best balance of speed and accuracy for Apple Silicon hardware.

## Model-Specific Learnings

### NVIDIA Canary 1B v2 (nvidia/canary-1b-v2)
- **WER**: 16.80% average | **Speed**: 9.17s avg per file
- **Key Feature**: Native long-form dynamic chunking (automatic for files >40s)
- **Hallucination Issue**: 3 files had repetition loops causing high WER:
  - `day5_consultation05`: 45.39% WER - "I'm in Italy" repeated ~38 times
  - `day1_consultation12`: 38.74% WER - repetition loop
  - `day1_consultation10`: 32.54% WER - repetition loop
- **Without outliers**: 15.61% WER
- **Takeaway**: Native long-form works well on most files, but autoregressive models can hallucinate unpredictably on certain audio segments

### Google MedASR
- **WER**: 64.88% - Worst performing model in benchmark
- **Tested On**: MPS (MacBook CPU), NVIDIA T4 GPU (official Google notebook), Vertex AI endpoint
- **Result**: All three platforms showed similar poor performance
- **Reason**: MedASR is designed for medical **dictation** (single speaker, clear speech), not doctor-patient **conversations**
- **Benchmark**: Kept MPS (MacBook) results; ignored T4 and Vertex AI as they showed no improvement
- **Note**: Vertex AI requires chunking due to 1.5MB request limit

### IBM Granite Speech 3.3-2b (ibm-granite/granite-speech-3.3-2b)
- **WER**: 18.92% average | **Speed**: 109.7s avg per file
- **Architecture**: Two-pass design (speech encoder → text decoder)
- **Chunking Required**: Without chunking, model enters repetition loops even with low max_new_tokens
- **Solution**: 35s chunks with 10s overlap + LCS merging
- **Setup**: Requires transformers>=4.52.4 for `granite_speech` architecture support
- **Note**: Speed metrics based on 46/55 files (9 files missing timing data)

### CrisperWhisper (nyrahealth/CrisperWhisper)
- **Speed**: ~227s per 7.5 min file (~0.5x realtime) - very slow
- **Features**: Verbatim transcription with filler detection ([UM], [UH])
- **Setup**: Requires custom transformers fork, gated HuggingFace repo
- **Takeaway**: Too slow for batch processing (~3.5 hours for 57 files)

## Hallucination Patterns in Autoregressive Speech Models

Several models exhibited similar hallucination behavior:
1. **Repetition loops**: Getting stuck repeating phrases ("I'm in Italy", "you're just feeling sick")
2. **Triggered by**: Long audio sequences, silent/unclear audio segments
3. **Mitigation strategies**:
   - Chunking with overlap (35s chunks, 10s overlap)
   - LCS merging to stitch chunks cleanly
   - Note: Lower max_new_tokens doesn't prevent loops
4. **Models affected**: Canary 1B v2, Granite Speech 3.3-2b, Kyutai STT 2.6B

## Adding New Models

1. Create `transcribe/your_model_transcribe.py`
2. Inherit from `BaseTranscriber` in `transcribe/base_transcriber.py`
3. Implement `transcribe_file()` returning `TranscriptionResult`
4. For long audio constraints, copy chunking logic from `canary_qwen_improved_transcribe.py`
5. Test on sample files first, then full dataset
6. Run evaluation workflow:
   ```bash
   python transcribe/your_model_transcribe.py --audio_dir data/raw_audio
   python evaluate/metrics_generator.py --model_name your-model
   python evaluate/comparison_generator.py
   ```
