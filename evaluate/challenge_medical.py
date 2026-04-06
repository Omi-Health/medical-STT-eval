#!/usr/bin/env python3
"""Medical M-WER challenge set — red-team adversarial pairs.

A detector sanity check for supported medical-error cases,
plus a documented blind-spot report for errors M-WER cannot detect.

Supported cases verify the *specific* critical term was flagged,
not just that any medical error was detected.

Usage:
    python challenge_medical.py
    python medical_wer.py --challenge
"""

import json
import sys
from pathlib import Path

from medical_wer import calculate_medical_metrics

# --- Supported cases: M-WER should detect these ---
# critical_ref: the specific reference term that must appear in detected errors
SUPPORTED_CASES = [
    # Confusable prefix swaps
    {
        "id": "confusable_001",
        "category": "confusable_prefix",
        "reference": "the patient has hypothyroidism and takes thyroxine",
        "hypothesis": "the patient has hyperthyroidism and takes thyroxine",
        "critical_ref": "hypothyroidism",
        "description": "hypo/hyper swap changes diagnosis",
    },
    {
        "id": "confusable_002",
        "category": "confusable_prefix",
        "reference": "she was diagnosed with hypertension last year",
        "hypothesis": "she was diagnosed with hyperthyroidism last year",
        "critical_ref": "hypertension",
        "description": "hypertension -> hyperthyroidism",
    },
    # Drug name swaps
    {
        "id": "drug_swap_001",
        "category": "drug_swap",
        "reference": "currently taking metformin for diabetes",
        "hypothesis": "currently taking ramipril for diabetes",
        "critical_ref": "metformin",
        "description": "diabetes drug swapped for blood pressure drug",
    },
    {
        "id": "drug_swap_002",
        "category": "drug_swap",
        "reference": "prescribed salbutamol inhaler for asthma",
        "hypothesis": "prescribed prednisolone inhaler for asthma",
        "critical_ref": "salbutamol",
        "description": "reliever swapped for steroid",
    },
    {
        "id": "drug_swap_003",
        "category": "drug_swap",
        "reference": "take omeprazole for the heartburn",
        "hypothesis": "take ibuprofen for the heartburn",
        "critical_ref": "omeprazole",
        "description": "acid reducer swapped for NSAID (would worsen heartburn)",
    },
    {
        "id": "drug_swap_004",
        "category": "drug_swap",
        "reference": "she takes bisoprolol for her heart",
        "hypothesis": "she takes lisinopril for her heart",
        "critical_ref": "bisoprolol",
        "description": "beta blocker swapped for ACE inhibitor",
    },
    # Near-miss spellings (common ASR errors)
    {
        "id": "near_miss_001",
        "category": "near_miss",
        "reference": "taking paracetamol for the pain and fever",
        "hypothesis": "taking paracetemol for the pain and fever",
        "critical_ref": "paracetamol",
        "description": "common ASR misspelling of paracetamol",
    },
    {
        "id": "near_miss_002",
        "category": "near_miss",
        "reference": "the ventolin inhaler helps with wheezing",
        "hypothesis": "the ventilin inhaler helps with wheezing",
        "critical_ref": "ventolin",
        "description": "ventolin -> ventilin ASR variant",
    },
    {
        "id": "near_miss_003",
        "category": "near_miss",
        "reference": "prescribed trimethoprim for the infection",
        "hypothesis": "prescribed trimethoprine for the infection",
        "critical_ref": "trimethoprim",
        "description": "trimethoprim -> trimethoprine ASR variant",
    },
    {
        "id": "near_miss_004",
        "category": "near_miss",
        "reference": "the phlegm has been getting worse",
        "hypothesis": "the flem has been getting worse",
        "critical_ref": "phlegm",
        "description": "phlegm -> flem ASR variant",
    },
    # Condition swaps
    {
        "id": "condition_swap_001",
        "category": "condition_swap",
        "reference": "patient has asthma and uses an inhaler",
        "hypothesis": "patient has eczema and uses an inhaler",
        "critical_ref": "asthma",
        "description": "respiratory condition swapped for skin condition",
    },
    {
        "id": "condition_swap_002",
        "category": "condition_swap",
        "reference": "history of epilepsy since childhood",
        "hypothesis": "history of arthritis since childhood",
        "critical_ref": "epilepsy",
        "description": "neurological condition swapped for musculoskeletal",
    },
]

# --- Known blind spots: M-WER cannot detect these ---
BLIND_SPOT_CASES = [
    # Negation
    {
        "id": "negation_001",
        "category": "negation",
        "reference": "no history of diabetes in the family",
        "hypothesis": "a history of diabetes in the family",
        "description": "negation dropped — 'no' is not a medical term",
    },
    {
        "id": "negation_002",
        "category": "negation",
        "reference": "patient is not allergic to penicillin",
        "hypothesis": "patient is allergic to penicillin",
        "description": "negation 'not' deleted — flips allergy status",
    },
    # Laterality
    {
        "id": "laterality_001",
        "category": "laterality",
        "reference": "pain in the left kidney area",
        "hypothesis": "pain in the right kidney area",
        "description": "left/right swap — not medical terms",
    },
    # Dosage
    {
        "id": "dosage_001",
        "category": "dosage",
        "reference": "take two tablets of paracetamol",
        "hypothesis": "take ten tablets of paracetamol",
        "description": "number swap — 'two'/'ten' not medical terms",
    },
    {
        "id": "dosage_002",
        "category": "dosage",
        "reference": "prescribed paracetamol twice daily",
        "hypothesis": "prescribed paracetamol twice weekly",
        "description": "frequency swap — 'daily'/'weekly' not medical terms",
    },
]


def _critical_term_detected(errors: list[dict], critical_ref: str) -> bool:
    """Check if the specific critical reference term was flagged as an error."""
    critical_norm = critical_ref.lower()
    for err in errors:
        ref = err.get("reference", "").lower()
        if ref == critical_norm:
            return True
    return False


def run_challenge_set() -> dict:
    """Run all challenge pairs and return structured results.

    For supported cases, verifies the specific critical_ref term was
    detected as an error — not just that any medical error was found.
    """
    supported_results = []
    for case in SUPPORTED_CASES:
        metrics = calculate_medical_metrics(case["reference"], case["hypothesis"])
        detected = _critical_term_detected(
            metrics["medical_errors_detail"], case["critical_ref"]
        )
        supported_results.append({
            "id": case["id"],
            "category": case["category"],
            "critical_ref": case["critical_ref"],
            "description": case["description"],
            "detected": detected,
            "m_wer": metrics["m_wer"],
            "errors": metrics["medical_errors_detail"],
        })

    blind_results = []
    for case in BLIND_SPOT_CASES:
        metrics = calculate_medical_metrics(case["reference"], case["hypothesis"])
        has_medical_error = len(metrics["medical_errors_detail"]) > 0
        blind_results.append({
            "id": case["id"],
            "category": case["category"],
            "description": case["description"],
            "detected": has_medical_error,
            "m_wer": metrics["m_wer"],
            "errors": metrics["medical_errors_detail"],
        })

    supported_passed = sum(1 for r in supported_results if r["detected"])
    blind_detected = sum(1 for r in blind_results if r["detected"])

    # Print report
    print(f"\n{'='*60}")
    print(f"Medical Challenge Set — Detector Sanity Check")
    print(f"{'='*60}")

    print(f"\n  === Supported cases ({supported_passed}/{len(supported_results)} detected) ===")
    for r in supported_results:
        status = "PASS" if r["detected"] else "FAIL"
        print(f"    {status}: {r['id']} — {r['description']}")
        if not r["detected"]:
            print(f"           expected '{r['critical_ref']}' in errors, not found")

    print(f"\n  === Known blind spots ({blind_detected}/{len(blind_results)} detected — expected 0) ===")
    for r in blind_results:
        status = "DETECTED" if r["detected"] else "BLIND"
        print(f"    {status}: {r['id']} — {r['description']}")

    print(f"\n  Summary:")
    print(f"    Supported:   {supported_passed}/{len(supported_results)} critical terms detected")
    print(f"    Blind spots: {blind_detected}/{len(blind_results)} detected (expected: 0)")

    return {
        "supported": {
            "total": len(supported_results),
            "passed": supported_passed,
            "failed": len(supported_results) - supported_passed,
            "results": supported_results,
        },
        "blind_spots": {
            "total": len(blind_results),
            "detected": blind_detected,
            "results": blind_results,
        },
    }


if __name__ == "__main__":
    results = run_challenge_set()

    if len(sys.argv) > 1 and sys.argv[1] == "--output":
        outfile = sys.argv[2] if len(sys.argv) > 2 else "challenge_results.json"
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {outfile}")
