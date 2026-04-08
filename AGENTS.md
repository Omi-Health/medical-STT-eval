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
├── transcribe/                 # 24 model-specific scripts + base class
│   ├── base_transcriber.py     # Base class (loads .env automatically)
│   └── *_transcribe.py         # One per model/API
├── evaluate/
│   ├── text_normalizer.py     # Custom WER normalizer (replaces whisper dependency)
│   ├── english.json           # British→American spelling mappings (1,739 entries)
│   ├── wer_calculator.py      # WER calculation algorithm
│   ├── metrics_generator.py   # Generates *_wer.json per model
│   ├── comparison_generator.py # Generates leaderboard.json (ranked by M-WER)
│   ├── medical_wer.py         # Medical WER: per-category M-WER, Drug M-WER, M-CER
│   ├── medical_terms_list.py  # 179 categorized medical terms (drugs/conditions/symptoms/anatomy/clinical)
│   └── challenge_medical.py   # Red-team challenge set (12 supported + 5 blind spots)
└── results/
    ├── metrics/                # Per model: *_speed.json, *_wer.json, *_medical_wer.json
    ├── comparisons/            # leaderboard.json (ranked by M-WER), per_file_results.json
    └── transcripts/            # 38 model dirs with raw transcript files
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
- **Mistral**: `voxtral_mini_transcribe_v1_chat_transcribe.py` (Voxtral models via API)
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

| Script | Models Served | GPU |
|--------|---------------|-----|
| `openai_api_transcribe.py` | whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe (×2) | API |
| `groq_whisper_transcribe.py` | whisper-large-v3, whisper-large-v3-turbo | API |
| `gemini_transcribe.py` | gemini-2.5-flash, gemini-2.5-pro, gemini-3-flash, gemini-3-pro | API |
| `voxtral_mini_transcribe_v1_chat_transcribe.py` | voxtral-mini-transcribe-v1-chat | API |
| `voxtral_mini_transcribe_v1_api_transcribe.py` | voxtral-mini-transcribe-v1-api | API |
| `voxtral_mini_transcribe_v2_transcribe.py` | voxtral-mini-transcribe-v2 | API |
| `voxtral_mini_4b_realtime_transcribe.py` | voxtral-mini-4b-realtime | A10 vLLM |
| `cohere_transcribe.py` | cohere-transcribe-03-2026 (vLLM /v1/audio/transcriptions) | A10 vLLM |
| `qwen3_asr_transcribe.py` | qwen3-asr-1.7b, qwen3-asr-0.6b | A10 vLLM |
| `gemma4_transcribe.py` | gemma-4-e4b-it, gemma-4-e2b-it (30s chunks, dtype=auto) | T4 |
| `canary_qwen_improved_transcribe.py` | canary-qwen-2.5b (35s chunks, 10s overlap) | T4 |
| `canary_1b_flash_improved_transcribe.py` | canary-1b-flash (35s chunks, 10s overlap) | T4 |
| `canary_1b_v2_transcribe.py` | canary-1b-v2 (native long-form) | T4 |
| `granite_speech_transcribe.py` | granite-speech-3.3-2b (chunked) | T4 |
| `kyutai_stt_pytorch_transcribe.py` | stt-2.6b-en | T4 |
| `kyutai_stt_1b_pytorch_transcribe.py` | stt-1b-en_fr | T4 |
| `elevenlabs_scribe_transcribe.py` | scribe_v1 | API |
| `elevenlabs_scribe_v2_transcribe.py` | scribe_v2 | API |
| `azure_foundry_phi4_transcribe.py` | azure-foundry-phi4 | API |
| `parakeet_transcribe.py` | parakeet-tdt-0.6b-v2, parakeet-tdt-0.6b-v3 | Apple Silicon |
| `mlx_whisper_transcribe.py` | mlx-whisper-large-v3-turbo | Apple Silicon |
| `whisperkit_transcribe.py` | whisperkit-large-v3-turbo | Apple Silicon |
| `apple_speechanalyzer_transcribe.py` | apple-speechanalyzer | Apple Silicon |
| `medasr_transcribe.py` | google-medasr | Apple Silicon |
| `deepgram_medical_transcribe.py` | deepgram-nova-3-medical (parallel, skip-existing) | API |
| `assemblyai_transcribe.py` | assemblyai-universal-3-pro-medical (medical-v1 domain) | API |
| `soniox_transcribe.py` | soniox-stt-async-v4 (REST async, no context) | API |
| `mai_transcribe_1.py` | mai-transcribe-1 (Azure Speech enhancedMode) | API |
| `mms_1b_transcribe.py` | facebook/mms-1b-all (chunked HF pipeline, return_timestamps='char') | T4 |
| `glm_asr_nano_transcribe.py` | zai-org/GLM-ASR-Nano-2512 (30s chunks, BF16) | T4 |
| NeMo direct (no script) | nemotron-speech-0.6b, parakeet-tdt-1.1b, multitalker-parakeet-0.6b, vibevoice-9b | T4/H100 |

## API Quirks and Learnings

### Mistral Voxtral
- **V1 Transcription API**: `voxtral_mini_transcribe_v1_api_transcribe.py` - Uses `/v1/audio/transcriptions` endpoint
- **V1 Chat-based**: `voxtral_mini_transcribe_v1_chat_transcribe.py` - Uses chat completions with base64 encoded audio
- **V2 Transcription API**: `voxtral_mini_transcribe_v2_transcribe.py` - Same endpoint, updated model (Feb 2026)
- **4B Realtime**: `voxtral_mini_4b_realtime_transcribe.py` - Local GPU via transformers
- **Key Finding**: Only Mini supports `/v1/audio/transcriptions` - documentation says both models support it but reality is different
- **Workaround**: Small model must use chat completions with audio input

### ElevenLabs Scribe V1 vs V2
- **V1 (Batch API)**: `speech_to_text.convert()` - Upload entire file, get transcript back instantly
- **V2 (Realtime API)**: WebSocket-based streaming with chunked PCM audio
- **V2 Requirements**:
  - Must send audio at **real-time speed** (1 second delay per 1 second of audio)
  - Requires PCM conversion (16kHz, 16-bit, mono)
  - Auto-commits every ~90 seconds; recommend manual commit every 20-30s
  - Sending faster than real-time causes audio loss
- **Speed Comparison**:
  - V1: ~36s for 7.6 min file (instant upload + processing)
  - V2: ~7.6 min minimum (real-time streaming requirement)
- **Recommendation**: Use V1 for benchmark (faster); V2 for live/streaming use cases

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

### Qwen3-ASR vLLM long-audio fix (the encoder-cache trap)

**Problem.** Out of the box, `Qwen3ASRModel.LLM(...)` (the offline vLLM wrapper) and `qwen-asr-serve` (the HTTP server) silently hang or error on audio files longer than ~10 minutes (~657s). The hang manifests as a `model.transcribe()` call that occupies VRAM, parks GPU at 0%, and never returns. Dataset files >600s on PriMock57 fail consistently. This affected the entire top-7 longest files in the dataset.

**Root cause (vLLM 0.19.0 finally surfaces it as an actual error message):**
```
The decoder prompt contains a(n) audio item with length 8541, which exceeds
the pre-allocated encoder cache size 8192. Please reduce the input size or
increase the encoder cache size by setting --limit-mm-per-prompt at startup.
```
- vLLM's `SchedulerConfig.encoder_cache_size` defaults to `max_num_batched_tokens` (= 8192).
- For Qwen3-ASR, audio is tokenized at ~12.5 tokens/second by the encoder, so 600s of audio ≈ 7500 tokens, 657s ≈ 8541 tokens — over the 8192 cap. 800s ≈ 10420 tokens.
- The error message is misleading — `--limit-mm-per-prompt` is **not** the lever; it controls items-per-prompt count, not cache size.
- The actual lever is `max_num_batched_tokens` (which `encoder_cache_size` derives from in `vllm/config/scheduler.py:235`: `self.encoder_cache_size = self.max_num_batched_tokens`).

**The one-line fix:**
```python
Qwen3ASRModel.LLM(
    model="Qwen/Qwen3-ASR-1.7B",
    gpu_memory_utilization=0.7,
    max_inference_batch_size=128,
    max_new_tokens=4096,
    max_num_batched_tokens=16384,  # ← THE FIX. Default 8192 caps audio at ~10 min.
)
```

**Stack required for it to work on A10:**
- `qwen-asr==0.0.6` + `vllm>=0.19.0` (older 0.14.0 stable just hangs, no error)
- `flash-attn 2.8.3` installed (`MAX_JOBS=4 pip install -U flash-attn --no-build-isolation`)
- `wheel`, `setuptools`, `packaging` installed before flash-attn (otherwise build fails with `ModuleNotFoundError: wheel`)
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` env var (vLLM 0.19 requires spawn for CUDA child processes)
- Matched flashinfer versions (`flashinfer-python` and `flashinfer-cubin` both 0.6.6 — fresh `pip install --pre 'vllm[audio]' qwen-asr` gets them aligned, no need for `FLASHINFER_DISABLE_VERSION_CHECK=1` bypass)
- A10 24GB or larger GPU with FA2 support (compute capability ≥ 8.0). **T4 is not viable** — Qwen3-ASR's `mm_encoder_attention.py` hard-selects `AttentionBackendEnum.FLASH_ATTN` which requires SM ≥ 8.0, plus T4 has no native bf16 (emulation gives 26-61% WER on this model).

**Result on A10:** all 55 PriMock57 files complete in a single `model.transcribe(audio=[...all 55...])` call in ~376s wall-clock for 1.7B and ~279s for 0.6B (batched throughput; per-file ≈ 6.8s and 5.1s). WER essentially identical to the H100 reference run (Δ < 0.25 pts). See `private_analysis/qwen3_speed_backups/` for the H100 baseline transcripts and metrics.

**What does NOT fix it:**
- `limit_mm_per_prompt={"audio": 1}` — controls item count, not cache size.
- Patching `feature_extractor.chunk_length` or `max_audio_clip_s` in qwen-asr source — these affect dummy audio profiling but not the scheduler's cache budget.
- Reducing `max_inference_batch_size` from 128 to 32 — same default 8192 cap.
- Switching to `qwen-asr-serve` HTTP server with `/v1/audio/transcriptions` endpoint — works for one-off requests but accumulates KV cache state across sequential requests and hangs after a few files.

### Speed metrics across re-runs
`base_transcriber._save_metrics()` merges new per-file timings with any existing `*_speed.json` instead of overwriting. This means re-runs that skip already-completed files (e.g. resuming a partial batch) preserve prior timings — `*_speed.json` always reflects all processed files, not just the current invocation.

### File Size Handling
- **Cloud APIs**: Generally handle 13.9MB medical files well
- **Chunking required**: NVIDIA vLLM models, Azure Phi-4
- **Upload vs inline**: Gemini automatically chooses based on file size

## WER Normalization (3 phases)

### Phase 1: Whisper Normalization (initial upgrade)
Upgraded from basic normalization (lowercase + remove punctuation) to Whisper's `EnglishTextNormalizer`:
- Filler word removal (um, uh, hmm)
- Number normalization ("23" ↔ "twenty three")
- Contraction expansion ("don't" → "do not")
- British→American spelling (1,739 mappings)
- Result: ~5-6% WER improvement across all models

### Phase 2: Error analysis discovered normalizer gaps
Word-level error analysis across all 31 models revealed Whisper's normalizer had issues:

**Gap 1: Missing word equivalences**
- `ok` vs `okay` vs `k` — 40,953 false errors across all models (16-21% of per-model errors)
- `yeah`/`yep` vs `yes`, `mum` vs `mom`, `alright` vs `all right`, `kinda` vs `kind of`

**Gap 2: "oh" → "0" bug**
- Whisper's `EnglishNumberNormalizer` has `self.zeros = {"o", "oh", "zero"}`, treating the interjection "oh" as digit zero
- Verified: zero models output "0" in raw transcripts — all correctly output "oh". The "0" was entirely created by the normalizer
- 1,395 false errors across all models

### Phase 3: Custom normalizer (current)
Replaced `openai-whisper` dependency with self-contained `evaluate/text_normalizer.py`:
- Copied Whisper's normalizer classes
- **Fixed "oh" bug**: removed `"oh"` from `self.zeros`
- **Added post-normalization word mappings**: ok/okay, yeah/yes, mum/mom, alright/all right, kinda/kind of, etc.
- No runtime dependency on `openai-whisper` package

### Cumulative WER impact
| Model | Basic | Whisper | Custom | Total |
|---|---:|---:|---:|---:|
| Gemini 2.5 Pro | ~16% | 10.90% | **8.28%** | -8% |
| VibeVoice 9B | ~17% | 10.91% | **8.34%** | -9% |
| Parakeet v3 | ~19% | 11.79% | **9.35%** | -10% |

## Performance Patterns (current rankings, ranked by M-WER)

43 single-stream models + 1 multi-speaker model. See README.md for the full leaderboard.

Top 10 by Medical WER:

| # | Model | WER | M-WER | Drug M-WER | GPU |
|---|-------|-----|-------|------------|-----|
| 1 | Google Gemini 3 Pro | 8.35% | 2.65% | 3.1% | API |
| 2 | Google Gemini 2.5 Pro | 8.15% | 2.97% | 4.1% | API |
| 3 | VibeVoice-ASR 9B | 8.34% | 3.16% | 5.6% | H100 |
| 4 | Soniox stt-async-v4 | 9.18% | 3.32% | 7.1% | API |
| 5 | Google Gemini 3 Flash | 11.33% | 3.64% | 5.2% | API |
| 6 | ElevenLabs Scribe v2 | 9.72% | 3.86% | 4.3% | API |
| 7 | AssemblyAI Universal-3 Pro (medical-v1) | 9.55% | 4.02% | 6.5% | API |
| 8 | Qwen3 ASR 1.7B | 9.00% | 4.40% | 8.6% | A10 vLLM |
| 9 | Deepgram Nova-3 Medical | 9.05% | 4.53% | 9.7% | API |
| 10 | OpenAI GPT-4o Mini (Dec '25) | 11.18% | 4.85% | 10.6% | API |

Key observations:
- **M-WER ranking differs from WER** — Parakeet v3 ranks #4 by WER but #31 by M-WER (22% Drug M-WER)
- **LLM-based models lead on medical terms** — Gemini, VibeVoice, Qwen3 benefit from language model context
- **Drug names are the hardest category** — Drug M-WER is consistently 2-5x higher than overall M-WER
- **Best open-source for medical**: Qwen3-ASR 1.7B (9.00% WER, 4.40% M-WER on A10 vLLM, 6.83s/file batched throughput)
- **Best on T4**: Gemma 4 E4B-it (15.69% WER, 9.99% M-WER via transformers dtype=auto)

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

## Models Evaluated (April 2026 batch)

### Qwen3-ASR 1.7B / 0.6B (re-run on A10 vLLM with encoder-cache fix)
- **WER**: 9.00% (1.7B), 9.83% (0.6B) | **M-WER**: 4.40% / 6.48% | **Drug M-WER**: 8.6% / 15.1%
- **Speed**: 6.83s / 5.07s avg per file (batched throughput, 55 files in single `model.transcribe()` call)
- **Hardware**: A10 24GB via vLLM 0.19.0 + flash-attn 2.8.3 (clean fresh venv on Azure NV36ads_A10_v5)
- **The fix that unlocked >10 min audio on A10**: see [Qwen3-ASR vLLM long-audio fix](#qwen3-asr-vllm-long-audio-fix-the-encoder-cache-trap) below.
- **Note**: Earlier H100 batch run gave essentially identical numbers (1.7B: 8.96% WER / 4.69% M-WER / 6.99s; 0.6B: 10.04% / 8.04% / 5.55s). The A10 numbers are now the canonical leaderboard entries because A10 is the more accessible hardware tier and the configuration is fully documented and reproducible. H100 transcripts/metrics archived under `private_analysis/qwen3_speed_backups/`.

### Microsoft MAI-Transcribe-1 (Azure Speech Fast Transcription)
- **WER**: 11.52% | **M-WER**: 4.85% | **Drug M-WER**: 11.2% | **Speed**: 21.8s avg (workers=4)
- **API**: Azure Cognitive Services `/speechtotext/transcriptions:transcribe?api-version=2025-10-15` with `enhancedMode.task=transcribe`. Multipart upload + JSON `definition` part. Returns `combinedPhrases[0].text` synchronously.
- **Auth**: `Ocp-Apim-Subscription-Key` header. **Endpoint must match the key** — the Azure Foundry key works only against the `*.cognitiveservices.azure.com` host (not `*.services.ai.azure.com`).
- **Key Finding**: Top-15, slightly weaker than the Dec 2025 GPT-4o Mini on the same M-WER tier. Strong drug recognition for a non-medical-specific cloud API. Sub-30s per file with parallel workers, no throttling observed.

### Facebook MMS-1B-all
- **WER**: 38.70% | **M-WER**: 54.01% | **Drug M-WER**: 72.0% | **Speed**: 28.6s avg (T4)
- **Architecture**: Wav2Vec2 CTC with 1107-language adapters; English adapter loaded via `model.load_adapter("eng")` + `processor.tokenizer.set_target_lang("eng")`.
- **Long-audio handling**: HF `AutomaticSpeechRecognitionPipeline` with `chunk_length_s=30, stride_length_s=(5, 5)`. CTC pipeline requires `return_timestamps="char"` when chunking — without it the pipeline raises `CTC can either predict character level timestamps, or word level timestamps`.
- **Key Finding**: **Worst medical performer in the benchmark.** Phonetic-style spelling errors dominate: "asthma"→"asma", "cough"→"cogh", "diarrhea"→"diarea", "ibuprofen"→garbled, "vomiting"→"vometing". MMS is designed for low-resource multilingual transcription with a phonetic-leaning vocab — it has no English orthography prior, so medical terms collapse to nearest-sound spelling. Useful as a floor for the leaderboard.

### GLM-ASR-Nano-2512 (zai-org/GLM-ASR-Nano-2512)
- **WER**: 10.84% | **M-WER**: 7.05% | **Drug M-WER**: 17.5% | **Speed**: 87.7s avg (T4 BF16)
- **Architecture**: Multimodal LLM with audio encoder. Loaded via `AutoModel.from_pretrained(..., dtype=torch.bfloat16, trust_remote_code=True)` + `AutoProcessor.apply_chat_template`.
- **Setup gotcha**: Requires **transformers from source** (>=5.6.0.dev0) to recognize the `glmasr` model architecture. The stock transformers 4.57.6 errors with `model type "glmasr" but Transformers does not recognize this architecture`.
- **Long-audio handling**: 30-second chunks with `max_new_tokens=1024` per chunk; chunks concatenated with simple space join (no LCS merge needed for this model).
- **Key Finding**: Decent overall WER but weak on medical terms — drug names get collapsed (`implanon`→`on`, `clenil`→`clenol`, `bisoprolol`→`zopralol`). 87s/file on T4 is slow for a "Nano" model — the chat-template wrapping + per-chunk LLM decoding overhead dominates.

### Soniox stt-async-v4
- **WER**: 9.18% | **M-WER**: 3.32% | **Drug M-WER**: 7.1% | **Speed**: 46.2s avg (workers=2)
- **API**: REST async (`/v1/files` upload → `/v1/transcriptions` create → poll → `/transcript`)
- **Config**: Plain call — `language_hints=["en"]` only. No context customization, no speaker diarization, no terms list (kept fair vs. other clean cloud-API entries)
- **Key Finding**: 4th overall, best non-Google/non-VibeVoice model. Universal model — no medical-specific tuning. UK-specific brand names (e.g. Clenil) are the main error source; would benefit from `context.terms` if fairness wasn't a concern
- **Polling**: Default 0.5s interval (3s adds ~1.5s avg latency per file unnecessarily)

### AssemblyAI Universal-3 Pro (medical-v1 domain)
- **WER**: 9.55% | **M-WER**: 4.02% | **Drug M-WER**: 6.5% | **Speed**: 37.3s avg (workers=2)
- **API**: REST (`/v2/upload` → `/v2/transcript` with `domain: medical-v1` → poll)
- **Config**: `speech_models=["universal-3-pro"]`, `domain="medical-v1"`, `speaker_labels=True`, `language_detection=True`, `temperature=0`
- **Throttling**: Standard plan concurrency limit ~4 — at workers=8 we got the "queued for processing" email and per-file durations doubled (e.g. day1_consultation10: 129s → 30s when re-run with workers=1). Use `--workers 2` for fair speed measurement.
- **Polling**: 3s default interval (~1.5s avg latency per file)

### Deepgram Nova-3 Medical
- **WER**: 9.05% | **M-WER**: 4.53% | **Drug M-WER**: 9.7% | **Speed**: 12.9s avg (workers=2) — fastest cloud API in the benchmark
- **API**: REST (`POST /v1/listen?model=nova-3-medical&smart_format=true&language=en`) — single inline POST, transcript returned in body, no upload step, no polling
- **Config**: `smart_format=true`, `language=en` — clean call
- **Key Finding**: Fastest API by ~3× vs AssemblyAI/Soniox. Single-shot binary POST avoids upload + polling overhead. Strong on raw WER, slightly weaker on drugs vs the other two.

## Models Evaluated (March 2026 batch)

### Microsoft VibeVoice-ASR 9B
- **WER**: 8.34% average (custom normalization) | **Speed**: 96.7s avg (H100 GPU)
- **Architecture**: 9B parameter model based on Qwen2.5-7B with audio encoder
- **Key Feature**: Handles up to 60 minutes in a single pass (64K token context), built-in speaker diarization and timestamps
- **Setup**: Requires `vibevoice` package from GitHub (`pip install -e .` from https://github.com/microsoft/VibeVoice), flash-attn recommended
- **GPU**: Needs ~18GB VRAM (bf16) — does not fit on T4 16GB alongside other models
- **Note**: Speed measured on H100 96GB, not comparable with T4 benchmarks. Best-performing local GPU model in the benchmark.

### ElevenLabs Scribe v2
- **WER**: 12.36% average | **Speed**: 43.5s avg
- **API**: Same `speech_to_text.convert()` endpoint as v1, just `model_id="scribe_v2"`
- **Improvement over v1**: 1.18% WER reduction (v1: 13.54%)
- **Setup**: Requires `elevenlabs>=2.39.0` SDK
- **Parallel processing**: Supports concurrent requests (max 12 on standard plan, script retries on 429)

### Voxtral Mini Transcribe V2 (Transcription API)
- **WER**: 11.64% average | **Speed**: 18.4s avg
- **API**: Mistral's `/v1/audio/transcriptions` endpoint with `voxtral-mini-latest` (now points to voxtral-mini-2602)
- **Improvement over V1**: Slight improvement from V1 transcription endpoint (11.87%)

### Voxtral Mini 4B Realtime (Local GPU)
- **WER**: 11.89% average | **Speed**: 133.9s avg (H100), ~693s avg (T4)
- **Architecture**: 4B params (~3.4B language + ~970M audio encoder)
- **Key Finding**: vLLM requires Flash Attention 2 (compute capability >= 8.0) — does not work on T4 (7.5). Transformers fallback works but is very slow without FA2.
- **Token calculation**: 1 text token = 80ms of audio. Set `max_new_tokens = int(audio_duration / 0.08) + 4096`
- **Note**: Speed measured on both H100 and T4. H100 speed is ~5x faster.

### NVIDIA Nemotron Speech Streaming 0.6B
- **WER**: 13.38% average | **Speed**: 11.7s avg (T4)
- **Architecture**: Cache-Aware FastConformer-RNNT, 0.6B params, successor to Parakeet
- **Key Issue**: OOM on T4 16GB for longer audio files (>7 min). Required chunking (split in half with 5s overlap) for 15/55 files
- **Setup**: Requires NeMo toolkit. CUDA graph decoder has compatibility issues with newer cuda-bindings — use `loop_labels=False` in greedy decoding config
- **Comparison to Parakeet TDT v3**: Slightly worse accuracy (13.38% vs 11.90%) but designed for streaming/realtime use cases

### Qwen3-ASR 1.7B / 0.6B
- **WER**: 8.96% (1.7B), 10.04% (0.6B) | **M-WER**: 4.69%, 8.04%
- **Architecture**: Whisper-style encoder-decoder, Qwen3 LLM backbone
- **GPU**: A10 via vLLM (BF16, Flash Attention 2). Does NOT work well on T4 — BF16 emulation degrades quality to 26-61% WER. FP16 crashes (cuDNN conv2d incompatible).
- **Streaming**: Supported via `qwen_asr` package (chunk-and-redecode, not native). Tuned 4s chunks work for ~4/5 files.
- **Key Finding**: Best open-source model for medical ASR. 0.04x RTF on A10 (25x real-time). Needs compute capability >= 8.0 for good results.

### Cohere Transcribe (March 2026)
- **WER**: 11.81% | **M-WER**: 5.59% | **Speed**: 3.9s avg (A10 vLLM nightly)
- **Architecture**: 2B param conformer encoder-decoder, trained from scratch
- **GPU**: A10 via vLLM nightly (stable vLLM 0.19 produced garbage — needed nightly build)
- **Features**: 14 languages, automatic 35s chunking with overlap reassembly, punctuation control
- **License**: Apache 2.0, gated model (HF token required)

### Gemma 4 E4B-it / E2B-it
- **WER**: 15.69% (E4B), 18.90% (E2B) | **Drug M-WER**: 15.5%, 19.8%
- **Architecture**: Dense multimodal (text + image + audio), conformer audio encoder
- **GPU**: Runs on T4 via transformers with `dtype="auto"` and `AutoModelForMultimodalLM`. Also tested on H100 BF16 (same quality, 10x faster). Does NOT work via vLLM on T4 (FA2 + BF16 required).
- **Chunking**: 30s max audio — simple concat is best (overlap merge + context prompting made results worse)
- **Key Finding**: Best multimodal model on T4 for medical ASR. E4B quality matches H100 BF16. Quantization kills drug names (45% Drug M-WER at 4-bit vs 15% at full precision).

### Parakeet TDT 1.1B
- **WER**: 9.03% | **M-WER**: 5.20% | **Speed**: 12.3s avg (T4)
- **Architecture**: FastConformer Token-and-Duration Transducer, English-only
- **GPU**: T4 via NeMo. Best English-only Parakeet model.

### Models Evaluated But Not Suitable for Long-Form Medical Transcription

#### LiquidAI LFM2.5-Audio-1.5B
- **Verdict**: Not a dedicated ASR model — designed for short audio snippets (~2 seconds)
- **Issue**: Official cookbook uses 2-second chunks with 0.5s overlap via llama.cpp (GGUF quantized). With correct 2s chunking, no hallucinations but only produces sparse keyword extraction (~74 words from a 1400-word conversation). With 30s chunks, severe hallucination loops.
- **Conclusion**: This is a multimodal speech-to-speech model where ASR is a secondary capability via prompting ("Perform ASR."). Not suitable for medical conversation transcription.

#### Facebook SeamlessM4T v2 Large
- **Verdict**: Translation model, not a transcription model
- **Issue**: Produces heavily summarized output (~677 words from a ~1400-word conversation) with hallucinated repetitive phrases. Designed for cross-lingual speech translation, not verbatim transcription.
- **License**: CC-BY-NC-4.0 (non-commercial only)
- **Conclusion**: Not suitable for ASR benchmarking. The model fundamentally summarizes/translates rather than transcribes.

## Adding New Models

1. Create `transcribe/your_model_transcribe.py`
2. Inherit from `BaseTranscriber` in `transcribe/base_transcriber.py`
3. Implement `transcribe_file()` returning `TranscriptionResult`
4. For long audio constraints, copy chunking logic from `canary_qwen_improved_transcribe.py`
5. Test on sample files first, then full dataset
6. Run evaluation workflow:
   ```bash
   # 1. Transcribe (saves to results/transcripts/ + results/metrics/*_speed.json)
   python transcribe/your_model_transcribe.py --audio_dir data/raw_audio

   # 2. Standard WER (saves results/metrics/*_wer.json)
   python evaluate/metrics_generator.py --model_name your-model --results_dir results --reference_dir data/cleaned_transcripts

   # 3. Medical WER (saves results/metrics/*_medical_wer.json)
   python evaluate/medical_wer.py --model your-model --output results/metrics/your-model_medical_wer.json

   # 4. Update leaderboard (ranked by M-WER)
   python evaluate/comparison_generator.py --results_dir results
   ```
