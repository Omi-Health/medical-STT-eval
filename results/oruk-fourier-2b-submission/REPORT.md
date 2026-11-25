# Oruk Fourier 2B — PriMock57 Medical STT Submission Report

## Summary

This submission adds **Oruk Fourier 2B** ([`oruk/oruk-fourier-2b`](https://huggingface.co/oruk/oruk-fourier-2b)) to the PriMock57 medical speech-to-text benchmark across all 55 comparable consultations (excluding `day1_consultation07` and `day3_consultation03`), evaluated with the repository's unmodified `evaluate/metrics_generator.py`, `evaluate/medical_wer.py`, and `evaluate/comparison_generator.py` scorers.

| Metric | Oruk Fourier 2B (`oruk-fourier-2b`) | Qwen3 ASR 1.7B (`qwen3-asr-1.7b`, base) | Delta |
|---|---:|---:|---:|
| **Overall WER** | **8.57%** (`0.0857`) | 9.00% (`0.0900`) | **-0.43 pp** |
| **Canonical M-WER (v2)** | **3.02%** (`0.0302`, 95 / 3,150) | 3.14% (`0.0314`, 99 / 3,150) | **-0.13 pp** |
| **Drug M-WER (high-risk)** | 8.15% (`0.0815`, 38 / 466) | 7.08% (`0.0708`, 33 / 466) | +1.07 pp |
| **Symptoms M-WER** | **2.24%** (26 / 1,162) | 3.01% (35 / 1,162) | **-0.77 pp** |
| **Anatomy M-WER** | **1.79%** (11 / 616) | 2.11% (13 / 616) | **-0.32 pp** |
| **Clinical M-WER** | **0.81%** (4 / 493) | 1.83% (9 / 493) | **-1.01 pp** |
| **Conditions M-WER** | 3.84% (15 / 391) | 2.30% (9 / 391) | +1.53 pp |
| **Medical Entity Recall** | **97.58%** | 97.14% | **+0.44 pp** |
| **Medical Token Recall** | **96.98%** | 96.86% | **+0.12 pp** |
| **Worst-file WER** | **15.29%** | 44.13% | **-28.84 pp** |
| **Per-file WER Std Dev** | **0.0234** | 0.0532 | **-0.0298** |
| **Avg Speed / File** | **8.7s** (61.6× RTFx on L40S) | 6.8s (batched A10) | +1.9s |

Ranking across all 43 single-stream comparable models:
- **#8 overall by Canonical M-WER (v2)** (`3.02%`), ahead of base `Qwen3 ASR 1.7B` (`3.14%`, #9) and `Deepgram Nova-3 Medical` (`3.17%`, #10), and **#2 open-weight model** (after 9B `VibeVoice-ASR 9B`).
- **#4 overall by Overall WER** (`8.57%`), behind only Gemini 2.5 Pro (`8.15%`), VibeVoice-ASR 9B (`8.34%`), and Gemini 3 Pro Preview (`8.35%`, 54/55), with the 2nd lowest cross-consultation variance (`#2/43`, `0.0234` std dev — behind only `parakeet-tdt-1.1b` `0.0228`, ahead of `google-gemini-2.5-pro` `0.0238`; worst file `15.29%`).
- **Honest tradeoff note:** While Fourier 2B improves overall WER (`-0.43 pp`), Canonical M-WER (`-0.13 pp`), clinical/anatomy/symptom M-WER, and eliminates the 44.1% long-file deletion failure seen on `day1_consultation10` in base `Qwen3 ASR 1.7B`, high-risk Drug M-WER is `8.15%` vs. `7.08%` (`+5` errors out of 466 drug tokens, mainly UK brand/drug spellings such as `implanon` and `salbutamol`).

## Model & Runtime Provenance

- **Model ID:** `oruk/oruk-fourier-2b` (English ASR fine-tune of Qwen3-ASR 1.7B, 2.04B parameters)
- **Pinned Revision:** `8c2a8871f1304aa2f634da4fb95e24438c4945b9`
- **Weights Integrity:** Native BF16 `model.safetensors` (`4,076,193,080` bytes, SHA-256 `e64765ffdd90c83704c0951e0d88b7f2ed84499af5c208f7ddf8d922cf08441c`) verified at startup; downloads strictly the 7 native root checkpoint files from Hugging Face Hub.
- **License:** Weights released under `CC BY-NC 4.0` (non-commercial research/evaluation; hosted API available at `https://speech-api.oruk.ai`).
- **Inference Stack:** Native `transformers==5.24.0` (`AutoModelForMultimodalLM` + `AutoProcessor`, BF16 — no legacy `qwen_asr` package required) on `1× NVIDIA L40S` (48 GB VRAM, Hugging Face Jobs `https://huggingface.co/jobs/oruk/6b06d176d91f621b9307908c`).
- **Decoding & Chunking:** Exact published `generation_config.json` (`num_beams=3`, `no_repeat_ngram_size=8`, `do_sample=False`, `max_new_tokens=512`, `32,253` `bad_words_ids`), forced `language="English"`. Because PriMock57 consultations span 3.8–13.4 minutes and Fourier 2B's feature extractor uses 30 s windows, `transcribe/oruk_fourier_transcribe.py` deterministically splits recordings >28 s at the minimum-RMS 200 ms energy pause inside each `[20 s, 28 s]` band and batches chunks (`batch_size=16`) per consultation.
- **Affiliation Disclosure:** Submitted by the Oruk team (developers of `oruk/oruk-fourier-2b`).

## Reproducibility

```bash
# 1. Transcribe all 55 comparable PriMock57 consultations locally on GPU (native HF BF16)
python transcribe/oruk_fourier_transcribe.py --tag oruk-fourier-2b --batch-size 16 --decode-mode published

# (Optional) Transcribe via Oruk hosted OpenAI-compatible API
# ORUK_API_KEY=... python transcribe/oruk_fourier_transcribe.py --backend http --server https://speech-api.oruk.ai

# 2. Score with the repository's unmodified official evaluators
python evaluate/metrics_generator.py --model_name oruk-fourier-2b
python evaluate/medical_wer.py --model oruk-fourier-2b --output results/metrics/oruk-fourier-2b_medical_wer.json
python evaluate/comparison_generator.py --results_dir results
```
