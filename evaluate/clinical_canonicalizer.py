#!/usr/bin/env python3
"""Clinical token canonicalization for Canonical M-WER.

These rules are intentionally narrower than the text normalizer. They are used
only for medical-term evaluation, after the standard English normalization and
before word alignment.

The reference-only rules fix known PriMock57 ground-truth spelling issues. They
are asymmetric on purpose: if the reference says ``phlegm`` and a model says
``flem``, that remains an error.
"""

from __future__ import annotations

from typing import Literal


Side = Literal["reference", "hypothesis"]


# Correct known reference spelling mistakes before medical scoring.
REFERENCE_ONLY_TOKEN_MAP = {
    "paracetemol": "paracetamol",
    "thyrocsin": "thyroxine",
    "flem": "phlegm",
}


# Collapse accepted word-boundary variants in either reference or hypothesis.
PHRASE_MAP = {
    ("water", "works"): "waterworks",
    ("hay", "fever"): "hayfever",
    ("straight", "away"): "straightaway",
    ("pain", "killers"): "painkillers",
    ("pain", "killer"): "painkiller",
    ("light", "headed"): "lightheaded",
}


# Reviewed as low-risk for medical-term scoring. These are symmetric because
# the benchmark cares about clinical entity capture, not morphology.
SYMMETRIC_TOKEN_MAP = {
    # Singular/plural and light inflection.
    "tests": "test",
    "headaches": "headache",
    "pains": "pain",
    "medicines": "medicine",
    "fevers": "fever",
    "medications": "medication",
    "allergies": "allergy",
    "coughs": "cough",
    "steroids": "steroid",
    "fluids": "fluid",
    "infections": "infection",
    "antihistamines": "antihistamine",
    "migraines": "migraine",
    "bowels": "bowel",
    "breaths": "breath",
    "lungs": "lung",
    "symptoms": "symptom",
    "drugs": "drug",
    "inhalers": "inhaler",
    "throats": "throat",
    "muscles": "muscle",
    "wheezes": "wheeze",
    "rashes": "rash",
    "joints": "joint",
    "vomits": "vomit",
    "antibiotics": "antibiotic",
    "palpitations": "palpitation",
    "abnormals": "abnormal",
    "anxieties": "anxiety",
    "swellings": "swelling",
    "heartbeats": "heartbeat",
    "doses": "dose",
    "tablets": "tablet",
    "kidneys": "kidney",
    "skins": "skin",
    "heartburns": "heartburn",
    "painkillers": "painkiller",
    "hyperthyroids": "hyperthyroid",
    # Symptom wording variants.
    "itching": "itchy",
    "achy": "aching",
    "vomited": "vomit",
    "coughing": "cough",
    "stomach": "tummy",
}


def _collapse_phrases(tokens: list[str]) -> list[str]:
    """Collapse configured multi-token variants into one canonical token."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        replacement = None
        replacement_len = 0
        for phrase, canonical in PHRASE_MAP.items():
            n = len(phrase)
            if tuple(tokens[i:i + n]) == phrase:
                replacement = canonical
                replacement_len = n
                break
        if replacement is not None:
            out.append(replacement)
            i += replacement_len
        else:
            out.append(tokens[i])
            i += 1
    return out


def canonicalize_medical_tokens(tokens: list[str], side: Side) -> list[str]:
    """Canonicalize normalized tokens for medical WER alignment.

    Args:
        tokens: Output from the standard text normalizer split into tokens.
        side: ``reference`` or ``hypothesis``. Known dataset spelling fixes are
            applied only to the reference side.
    """
    collapsed = _collapse_phrases(tokens)
    canonical: list[str] = []

    for token in collapsed:
        if side == "reference":
            token = REFERENCE_ONLY_TOKEN_MAP.get(token, token)
        token = SYMMETRIC_TOKEN_MAP.get(token, token)
        canonical.append(token)

    return canonical
