# Lanson S2 full-audio + LLM correction — PriMock57 submission

Two leaderboard entries produced by Lanson's production stack, both on the full
55-file comparable set (PriMock57, `day1_consultation07` / `day3_consultation03`
excluded by the official scorers).

| Entry | Rank | WER | M-WER | Drug M-WER | Avg Speed |
|---|---|---|---|---|---|
| `lanson-s2-v10` (S2 full-audio + correction) | #8 | 9.04% | **3.05%** | 5.58% | 5.9s/file |
| `lanson-s2-fullaudio` (S2 full-audio raw) | #12 | 8.96% | 3.37% | 7.94% | 5.9s/file |

## Pipeline

Both entries use the same ASR; the correction layer is the differentiator.

1. **ASR — Lanson S2 (qwen-long)**: `Qwen/Qwen3-ASR-1.7B` served by vLLM 0.19
   native (full-audio, no chunking) on an NVIDIA GH200. OpenAI-compatible
   `/v1/audio/transcriptions`, 52 concurrent requests → 55 files in ~92 s.
   Raw transcripts in `results/transcripts/lanson-s2-fullaudio/`.
   Driver: `scripts/lanson_s2_batch.py`.
2. **Correction — Lanson live-api `/v1/transcript/correct`**: the production
   batch-correction pipeline (global context-guide pass + numbered-line chunk
   correction + structural fallback) driven by model
   `deepseek/deepseek-v4.1-flash-1017` via OpenRouter, with a custom
   `prompt` override (the batch-format port of the Lanson "v10" evidence-based
   correction prompt, `scripts/correction-batch.v10.md`). 0 fallbacks.
   Corrected transcripts in `results/transcripts/lanson-s2-v10/`.
   Driver: `scripts/lanson_s2_correct.py`.
   A client-side language guard (non-Latin corrected line ⇒ fall back to the
   raw line) protects against the model echoing instruction words.

Correction effect on full 55 files: M-WER 3.37 % → **3.05 %**, Drug M-WER
7.94 % → **5.58 %** (drug-name confusion fixes: `bisoprolol`/`zotralol`,
`Dioralyte`, …), WER essentially unchanged (8.96 % → 9.04 %).

## Repro

```bash
# 1. Full-audio ASR via S2
python scripts/lanson_s2_batch.py --model_name lanson-s2-fullaudio --concurrency 52

# 2. Correction through the real live-api (S2 host, green build with prompt override)
python scripts/lanson_s2_correct.py --input results/transcripts/lanson-s2-fullaudio \
    --out results/transcripts/lanson-s2-v10 \
    --prompt scripts/correction-batch.v10.md

# 3. Official scorers
python evaluate/metrics_generator.py --model_name lanson-s2-v10
python evaluate/medical_wer.py --model lanson-s2-v10
python evaluate/comparison_generator.py
```

Metrics JSON: `results/metrics/lanson-s2-fullaudio_{wer,medical_wer,speed}.json`,
`results/metrics/lanson-s2-v10_{wer,medical_wer,speed}.json`.