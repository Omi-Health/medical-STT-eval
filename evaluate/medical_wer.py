#!/usr/bin/env python3
"""Medical WER (M-WER), M-CER, and Medical Token Recall calculator.

Extends the standard WER evaluation with medical-specific metrics:
- M-WER: WER computed only on medical entity words
- M-CER: average character error rate on medical substitutions
- Medical token recall: occurrence-level recall (not just binary)
- Per-category M-WER: drugs, conditions, symptoms, anatomy, clinical
- High-risk M-WER: drug term errors separately
- Medical entity recall: % of unique reference medical terms found in hypothesis

Uses the same text normalization as the standard WER calculator.

Usage:
    # As a library
    from medical_wer import calculate_medical_metrics
    metrics = calculate_medical_metrics(reference_text, hypothesis_text)

    # As a CLI — evaluate a model's transcripts
    python medical_wer.py --model parakeet-tdt-1.1b
    python medical_wer.py --model parakeet-tdt-0.6b-v3 --output results.json

    # Run challenge set
    python medical_wer.py --challenge
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

from medical_terms_list import (
    CATEGORIES,
    get_medical_terms,
    get_term_category,
    is_high_risk,
)
from text_normalizer import EnglishTextNormalizer

TEXT_NORMALIZER = EnglishTextNormalizer()
EXCLUDED_FILES = {"day1_consultation07", "day3_consultation03"}


def _normalize(text: str) -> list[str]:
    """Normalize text using the Whisper-derived normalizer."""
    return TEXT_NORMALIZER(text).split()


def _norm_key(word: str) -> str:
    """Strip non-alphanumeric chars and lowercase for term lookup."""
    return re.sub(r"[^a-z0-9]", "", word.lower())


def _is_medical(word: str, medical_terms: frozenset[str]) -> bool:
    """Check if a normalized word is a medical term."""
    return _norm_key(word) in medical_terms


def _char_edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings (space-optimized 2-row DP)."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[n]


def _char_error_rate(ref_word: str, hyp_word: str) -> float:
    """Character error rate for a single substitution pair.

    Returns edit_distance / len(ref_word). Range [0.0, ...] where
    0.0 = identical, 1.0 = fully replaced, >1.0 = longer replacement.
    """
    if not ref_word:
        return 1.0
    return _char_edit_distance(ref_word, hyp_word) / len(ref_word)


def _align_words(ref_words: list[str], hyp_words: list[str]) -> list[tuple[str, str, str]]:
    """Align reference and hypothesis words using edit distance.

    Returns list of (operation, ref_word, hyp_word):
    - ("correct", ref_word, hyp_word)
    - ("substitution", ref_word, hyp_word)
    - ("deletion", ref_word, "")
    - ("insertion", "", hyp_word)
    """
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrack
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            ops.append(("correct", ref_words[i - 1], hyp_words[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("substitution", ref_words[i - 1], hyp_words[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("deletion", ref_words[i - 1], ""))
            i -= 1
        else:
            ops.append(("insertion", "", hyp_words[j - 1]))
            j -= 1

    ops.reverse()
    return ops


def calculate_medical_metrics(reference: str, hypothesis: str) -> Dict:
    """Calculate standard WER + M-WER + M-CER + per-category metrics.

    Returns dict with all original keys plus:
    - m_cer: avg character error rate across medical substitutions
    - medical_token_recall: occurrence-level recall (correct / ref_count)
    - spurious_medical_insertions: medical words in hyp with no ref counterpart
    - per_category: {cat: {ref_count, correct, substitutions, deletions, m_wer}}
    - high_risk_m_wer: M-WER for drug terms only
    - high_risk_ref_count: number of drug term occurrences in reference
    """
    medical_terms = get_medical_terms()

    ref_words = _normalize(reference)
    hyp_words = _normalize(hypothesis)

    if not ref_words:
        return {
            "wer": 1.0 if hyp_words else 0.0,
            "m_wer": 0.0,
            "m_cer": 0.0,
            "medical_entity_recall": 0.0,
            "medical_token_recall": 0.0,
            "total_ref_words": 0,
            "total_medical_ref_words": 0,
            "per_category": {c: {"ref_count": 0, "correct": 0, "substitutions": 0,
                                 "deletions": 0, "m_wer": 0.0} for c in CATEGORIES},
            "high_risk_m_wer": 0.0,
            "high_risk_ref_count": 0,
        }

    # Align and compute operations
    ops = _align_words(ref_words, hyp_words)

    # Overall WER
    total_errors = sum(1 for op, _, _ in ops if op != "correct")
    wer = total_errors / len(ref_words) if ref_words else 0.0

    # Counters
    medical_ref_count = 0
    medical_subs = 0
    medical_dels = 0
    medical_ins = 0
    medical_correct = 0
    spurious_medical_ins = 0
    medical_errors_detail = []

    # Per-category counters
    cat_stats = {c: {"ref_count": 0, "correct": 0, "substitutions": 0,
                     "deletions": 0} for c in CATEGORIES}

    # High-risk (drug) counters
    hr_ref = 0
    hr_errors = 0

    # M-CER accumulator
    cer_sum = 0.0
    cer_count = 0

    for op, ref_w, hyp_w in ops:
        ref_is_med = _is_medical(ref_w, medical_terms) if ref_w else False
        hyp_is_med = _is_medical(hyp_w, medical_terms) if hyp_w else False

        if op == "correct" and ref_is_med:
            medical_ref_count += 1
            medical_correct += 1
            # Per-category
            cat = get_term_category(_norm_key(ref_w))
            if cat:
                cat_stats[cat]["ref_count"] += 1
                cat_stats[cat]["correct"] += 1
            # High-risk
            if is_high_risk(_norm_key(ref_w)):
                hr_ref += 1

        elif op == "substitution" and ref_is_med:
            medical_ref_count += 1
            medical_subs += 1
            cer = _char_error_rate(ref_w, hyp_w)
            cer_sum += cer
            cer_count += 1
            # Per-category
            cat = get_term_category(_norm_key(ref_w))
            if cat:
                cat_stats[cat]["ref_count"] += 1
                cat_stats[cat]["substitutions"] += 1
            # High-risk
            if is_high_risk(_norm_key(ref_w)):
                hr_ref += 1
                hr_errors += 1
            medical_errors_detail.append({
                "type": "substitution",
                "reference": ref_w,
                "hypothesis": hyp_w,
                "category": cat,
                "cer": round(cer, 4),
            })

        elif op == "deletion" and ref_is_med:
            medical_ref_count += 1
            medical_dels += 1
            # Per-category
            cat = get_term_category(_norm_key(ref_w))
            if cat:
                cat_stats[cat]["ref_count"] += 1
                cat_stats[cat]["deletions"] += 1
            # High-risk
            if is_high_risk(_norm_key(ref_w)):
                hr_ref += 1
                hr_errors += 1
            medical_errors_detail.append({
                "type": "deletion",
                "reference": ref_w,
                "category": cat,
            })

        elif op == "insertion" and hyp_is_med:
            medical_ins += 1
            spurious_medical_ins += 1
            medical_errors_detail.append({
                "type": "insertion",
                "hypothesis": hyp_w,
            })

    medical_errors = medical_subs + medical_dels + medical_ins
    m_wer = medical_errors / medical_ref_count if medical_ref_count > 0 else 0.0
    m_cer = cer_sum / cer_count if cer_count > 0 else 0.0
    token_recall = medical_correct / medical_ref_count if medical_ref_count > 0 else 1.0
    hr_m_wer = hr_errors / hr_ref if hr_ref > 0 else 0.0

    # Per-category M-WER
    per_category = {}
    for cat in CATEGORIES:
        s = cat_stats[cat]
        errs = s["substitutions"] + s["deletions"]
        per_category[cat] = {
            "ref_count": s["ref_count"],
            "correct": s["correct"],
            "substitutions": s["substitutions"],
            "deletions": s["deletions"],
            "m_wer": round(errs / s["ref_count"], 6) if s["ref_count"] > 0 else 0.0,
        }

    # Binary entity recall (unchanged from original)
    ref_medical_unique = set()
    for w in ref_words:
        if _is_medical(w, medical_terms):
            ref_medical_unique.add(_norm_key(w))
    hyp_normalized = set(_norm_key(w) for w in hyp_words)
    found = ref_medical_unique & hyp_normalized
    entity_recall = len(found) / len(ref_medical_unique) if ref_medical_unique else 1.0

    return {
        "wer": round(wer, 6),
        "m_wer": round(m_wer, 6),
        "m_cer": round(m_cer, 4),
        "medical_entity_recall": round(entity_recall, 4),
        "medical_token_recall": round(token_recall, 4),
        "total_ref_words": len(ref_words),
        "total_medical_ref_words": medical_ref_count,
        "medical_correct": medical_correct,
        "medical_substitutions": medical_subs,
        "medical_deletions": medical_dels,
        "medical_insertions": medical_ins,
        "spurious_medical_insertions": spurious_medical_ins,
        "medical_errors_detail": medical_errors_detail,
        "unique_medical_terms_in_ref": len(ref_medical_unique),
        "unique_medical_terms_found": len(found),
        "per_category": per_category,
        "high_risk_m_wer": round(hr_m_wer, 6),
        "high_risk_ref_count": hr_ref,
    }


def evaluate_model(
    model_name: str,
    transcripts_dir: Path | None = None,
    reference_dir: Path | None = None,
) -> Dict:
    """Evaluate a model's transcripts with all medical metrics.

    Returns dict with per-file and aggregate metrics including
    per-category M-WER, global M-CER, and medical token recall.
    """
    base = Path(__file__).parent.parent

    if transcripts_dir is None:
        transcripts_dir = base / "results" / "transcripts" / model_name
    if reference_dir is None:
        reference_dir = base / "data" / "cleaned_transcripts"

    if not transcripts_dir.exists():
        print(f"Transcript dir not found: {transcripts_dir}")
        return {}
    if not reference_dir.exists():
        print(f"Reference dir not found: {reference_dir}")
        return {}

    results = []
    transcript_files = sorted(transcripts_dir.glob("*_transcript.txt")) or \
                       sorted(transcripts_dir.glob("*_transcription.txt"))

    if not transcript_files:
        print(f"No transcript files found in {transcripts_dir}")
        return {}

    for tf in transcript_files:
        name = tf.stem.replace("_conversation_transcript", "").replace("_transcript", "")
        if name in EXCLUDED_FILES:
            continue

        ref_file = reference_dir / f"{name}_pure_text.txt"
        if not ref_file.exists():
            continue

        ref_text = ref_file.read_text(encoding="utf-8").strip()
        hyp_text = tf.read_text(encoding="utf-8").strip()

        if not ref_text or not hyp_text:
            continue

        metrics = calculate_medical_metrics(ref_text, hyp_text)
        metrics["file"] = name
        results.append(metrics)

    if not results:
        return {}

    # --- Aggregate ---
    n = len(results)
    avg_wer = sum(r["wer"] for r in results) / n
    avg_m_wer = sum(r["m_wer"] for r in results) / n
    avg_recall = sum(r["medical_entity_recall"] for r in results) / n
    total_med_ref = sum(r["total_medical_ref_words"] for r in results)
    total_med_correct = sum(r["medical_correct"] for r in results)
    total_med_subs = sum(r["medical_substitutions"] for r in results)
    total_med_dels = sum(r["medical_deletions"] for r in results)

    # Global M-CER
    total_cer_sum = 0.0
    total_cer_count = 0
    for r in results:
        nsubs = r["medical_substitutions"]
        if nsubs > 0:
            total_cer_sum += r["m_cer"] * nsubs
            total_cer_count += nsubs
    global_m_cer = total_cer_sum / total_cer_count if total_cer_count > 0 else 0.0

    # Global medical token recall
    global_token_recall = total_med_correct / total_med_ref if total_med_ref > 0 else 1.0

    # Global per-category M-WER
    agg_cat = {c: {"ref_count": 0, "correct": 0, "substitutions": 0, "deletions": 0}
               for c in CATEGORIES}
    for r in results:
        for cat in CATEGORIES:
            fc = r.get("per_category", {}).get(cat, {})
            for k in ("ref_count", "correct", "substitutions", "deletions"):
                agg_cat[cat][k] += fc.get(k, 0)
    global_per_category = {}
    for cat in CATEGORIES:
        s = agg_cat[cat]
        errs = s["substitutions"] + s["deletions"]
        global_per_category[cat] = {
            "ref_count": s["ref_count"],
            "correct": s["correct"],
            "substitutions": s["substitutions"],
            "deletions": s["deletions"],
            "m_wer": round(errs / s["ref_count"], 6) if s["ref_count"] > 0 else 0.0,
        }

    # Global high-risk M-WER
    total_hr_ref = sum(r["high_risk_ref_count"] for r in results)
    total_hr_errors = sum(
        r["high_risk_m_wer"] * r["high_risk_ref_count"]
        for r in results if r["high_risk_ref_count"] > 0
    )
    global_hr_m_wer = total_hr_errors / total_hr_ref if total_hr_ref > 0 else 0.0

    # Collect all medical errors across files
    all_med_errors = []
    for r in results:
        for err in r.get("medical_errors_detail", []):
            err["file"] = r["file"]
            all_med_errors.append(err)

    # Top medical substitution errors
    sub_counts: dict[tuple[str, str], int] = {}
    for err in all_med_errors:
        if err["type"] == "substitution":
            key = (err["reference"], err["hypothesis"])
            sub_counts[key] = sub_counts.get(key, 0) + 1
    top_subs = sorted(sub_counts.items(), key=lambda x: -x[1])[:20]

    return {
        "model": model_name,
        "files": n,
        "avg_wer": round(avg_wer, 6),
        "avg_m_wer": round(avg_m_wer, 6),
        "global_m_wer": round(
            (total_med_subs + total_med_dels) / total_med_ref, 6
        ) if total_med_ref > 0 else 0.0,
        "global_m_cer": round(global_m_cer, 4),
        "avg_medical_entity_recall": round(avg_recall, 4),
        "global_medical_token_recall": round(global_token_recall, 4),
        "global_high_risk_m_wer": round(global_hr_m_wer, 6),
        "global_high_risk_ref_count": total_hr_ref,
        "global_per_category": global_per_category,
        "total_medical_ref_words": total_med_ref,
        "total_medical_correct": total_med_correct,
        "total_medical_substitutions": total_med_subs,
        "total_medical_deletions": total_med_dels,
        "top_medical_errors": [
            {"ref": ref, "hyp": hyp, "count": count}
            for (ref, hyp), count in top_subs
        ],
        "per_file": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Medical WER (M-WER) evaluation")
    parser.add_argument("--model", type=str, help="Model name (directory under results/transcripts/)")
    parser.add_argument("--transcripts-dir", type=str, help="Override transcript directory")
    parser.add_argument("--reference-dir", type=str, help="Override reference directory")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--challenge", action="store_true", help="Run challenge set instead of model eval")
    args = parser.parse_args()

    if args.challenge:
        from challenge_medical import run_challenge_set
        results = run_challenge_set()
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
        return

    if not args.model:
        parser.error("--model is required (or use --challenge)")

    transcripts_dir = Path(args.transcripts_dir) if args.transcripts_dir else None
    reference_dir = Path(args.reference_dir) if args.reference_dir else None

    results = evaluate_model(args.model, transcripts_dir, reference_dir)
    if not results:
        print("No results.")
        return

    print(f"\n{'='*60}")
    print(f"Model: {results['model']}")
    print(f"Files: {results['files']}")
    print(f"{'='*60}")
    print(f"  WER (overall):              {results['avg_wer']*100:.2f}%")
    print(f"  M-WER (medical terms):      {results['avg_m_wer']*100:.2f}%")
    print(f"  M-CER (char errors):        {results['global_m_cer']*100:.1f}%")
    print(f"  Medical entity recall:      {results['avg_medical_entity_recall']*100:.1f}%")
    print(f"  Medical token recall:       {results['global_medical_token_recall']*100:.1f}%")
    print(f"  Global M-WER:               {results['global_m_wer']*100:.2f}%")
    print(f"  Drug M-WER (high-risk):     {results['global_high_risk_m_wer']*100:.2f}%  (n={results['global_high_risk_ref_count']})")
    print(f"")
    print(f"  Medical ref words:          {results['total_medical_ref_words']}")
    print(f"  Medical correct:            {results['total_medical_correct']}")
    print(f"  Medical substitutions:      {results['total_medical_substitutions']}")
    print(f"  Medical deletions:          {results['total_medical_deletions']}")

    # Per-category M-WER
    gpc = results.get("global_per_category", {})
    if gpc:
        print(f"\n  Per-category M-WER:")
        for cat in CATEGORIES:
            s = gpc.get(cat, {})
            m = s.get("m_wer", 0)
            n = s.get("ref_count", 0)
            print(f"    {cat:>12}: {m*100:5.1f}%  (n={n})")

    if results.get("top_medical_errors"):
        print(f"\n  Top medical errors:")
        for err in results["top_medical_errors"][:10]:
            print(f"    {err['ref']:>20} -> {err['hyp']:<20} ({err['count']}x)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
