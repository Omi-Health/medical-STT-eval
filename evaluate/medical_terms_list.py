#!/usr/bin/env python3
"""Medical term list for M-WER evaluation on PriMock57.

Hand-curated from PriMock57 ground truth reference transcripts.
Every term was manually reviewed from the full word frequency list
of 55 consultations (1,813 unique words with freq >= 2).

Only genuinely medical/clinical words are included.
Categories: drugs, conditions, symptoms, anatomy, clinical.

Usage:
    from medical_terms_list import get_medical_terms
    terms = get_medical_terms()

    from medical_terms_list import get_term_category, is_high_risk
    cat = get_term_category("paracetamol")   # "drugs"
    risk = is_high_risk("paracetamol")       # True
"""

# Canonical medical terms with category assignments.
# Source: all 55 reference files in data/cleaned_transcripts/
# Every word here actually appears in the ground truth.
_MEDICAL_TERM_CATEGORIES: dict[str, str] = {
    # --- Drugs (brand + generic names) ---
    "aleve": "drugs",
    "amoxicillin": "drugs",
    "antibiotics": "drugs",
    "antibiotic": "drugs",
    "antihistamine": "drugs",
    "antihistamines": "drugs",
    "antiinflammatories": "drugs",
    "antiinflammatory": "drugs",
    "aspirin": "drugs",
    "beclomethasone": "drugs",
    "bisoprolol": "drugs",
    "clenil": "drugs",
    "codeine": "drugs",
    "emollients": "drugs",
    "ibuprofen": "drugs",
    "implanon": "drugs",
    "inhaler": "drugs",
    "inhalers": "drugs",
    "lisinopril": "drugs",
    "loratadine": "drugs",
    "medication": "drugs",
    "medications": "drugs",
    "medicine": "drugs",
    "medicines": "drugs",
    "metformin": "drugs",
    "modulite": "drugs",
    "omeprazole": "drugs",
    "painkiller": "drugs",
    "painkillers": "drugs",
    "paracetamol": "drugs",
    "paracetemol": "drugs",   # ground truth misspelling
    "piriton": "drugs",
    "prednisolone": "drugs",
    "ramipril": "drugs",
    "salbutamol": "drugs",
    "steroid": "drugs",
    "steroids": "drugs",
    "tablet": "drugs",
    "tablets": "drugs",
    "thyroxine": "drugs",
    "thyrocsin": "drugs",     # ground truth misspelling
    "trimethoprim": "drugs",
    "ventolin": "drugs",

    # --- Conditions / Diagnoses ---
    "anaemia": "conditions",
    "anaphylactic": "conditions",
    "angina": "conditions",
    "anxiety": "conditions",
    "anxious": "conditions",
    "arthritis": "conditions",
    "asthma": "conditions",
    "asthmatic": "conditions",
    "cancer": "conditions",
    "cholesterol": "conditions",
    "constipation": "conditions",
    "depression": "conditions",
    "diabetes": "conditions",
    "diabetic": "conditions",
    "diarrhea": "conditions",
    "diarrhoea": "conditions",
    "eczema": "conditions",
    "epilepsy": "conditions",
    "gastroenteritis": "conditions",
    "hayfever": "conditions",
    "heartburn": "conditions",
    "hypertension": "conditions",
    "hyperthyroid": "conditions",
    "hyperthyroidism": "conditions",
    "hypothyroidism": "conditions",
    "infection": "conditions",
    "infections": "conditions",
    "inflammation": "conditions",
    "inflammatory": "conditions",
    "migraine": "conditions",
    "migraines": "conditions",

    # --- Symptoms / Signs ---
    "ache": "symptoms",
    "aching": "symptoms",
    "bleeding": "symptoms",
    "breathless": "symptoms",
    "breathlessness": "symptoms",
    "constipated": "symptoms",
    "cough": "symptoms",
    "coughing": "symptoms",
    "coughs": "symptoms",
    "cramp": "symptoms",
    "cramping": "symptoms",
    "cramps": "symptoms",
    "crampy": "symptoms",
    "discharge": "symptoms",
    "dizziness": "symptoms",
    "dizzy": "symptoms",
    "fatigue": "symptoms",
    "fever": "symptoms",
    "feverish": "symptoms",
    "fevers": "symptoms",
    "headache": "symptoms",
    "headaches": "symptoms",
    "itchiness": "symptoms",
    "itching": "symptoms",
    "itchy": "symptoms",
    "lethargic": "symptoms",
    "nausea": "symptoms",
    "nauseous": "symptoms",
    "numb": "symptoms",
    "numbness": "symptoms",
    "pain": "symptoms",
    "painful": "symptoms",
    "pains": "symptoms",
    "palpitations": "symptoms",
    "phlegm": "symptoms",
    "rash": "symptoms",
    "rashes": "symptoms",
    "sneeze": "symptoms",
    "sore": "symptoms",
    "swelling": "symptoms",
    "swollen": "symptoms",
    "symptom": "symptoms",
    "symptoms": "symptoms",
    "unwell": "symptoms",
    "vomit": "symptoms",
    "vomited": "symptoms",
    "vomiting": "symptoms",
    "wheeze": "symptoms",
    "wheezing": "symptoms",
    "wheezy": "symptoms",

    # --- Anatomy / Body parts ---
    "abdomen": "anatomy",
    "abdominal": "anatomy",
    "ankles": "anatomy",
    "bladder": "anatomy",
    "blood": "anatomy",
    "bowel": "anatomy",
    "bowels": "anatomy",
    "brain": "anatomy",
    "chest": "anatomy",
    "colon": "anatomy",
    "glands": "anatomy",
    "heart": "anatomy",
    "heartbeat": "anatomy",
    "joint": "anatomy",
    "joints": "anatomy",
    "kidney": "anatomy",
    "kidneys": "anatomy",
    "lung": "anatomy",
    "lungs": "anatomy",
    "muscle": "anatomy",
    "muscles": "anatomy",
    "sinuses": "anatomy",
    "skin": "anatomy",
    "spleen": "anatomy",
    "stomach": "anatomy",
    "throat": "anatomy",
    "thyroid": "anatomy",
    "tummy": "anatomy",
    "urine": "anatomy",
    "urinating": "anatomy",

    # --- Clinical terms ---
    "abnormal": "clinical",
    "acute": "clinical",
    "acutely": "clinical",
    "allergic": "clinical",
    "allergies": "clinical",
    "allergy": "clinical",
    "breath": "clinical",
    "breathing": "clinical",
    "dose": "clinical",
    "doses": "clinical",
    "drug": "clinical",
    "drugs": "clinical",
    "emergency": "clinical",
    "fluid": "clinical",
    "fluids": "clinical",
    "medical": "clinical",
    "pharmacy": "clinical",
    "pharmacist": "clinical",
    "prescribe": "clinical",
    "prescribed": "clinical",
    "prescription": "clinical",
    "test": "clinical",
    "tests": "clinical",
    "therapy": "clinical",
    "waterworks": "clinical",
}

# ASR error variants: recognized as medical for detection purposes,
# but have NO category assignment. Per-category M-WER uses the
# reference word's category, not the hypothesis variant.
_ASR_VARIANTS = frozenset({
    "flem",                                          # -> phlegm
    "ventilin", "ventalin",                          # -> ventolin
    "clenol", "clenal", "clenel", "clinal", "clinol",  # -> clenil
    "loratidine",                                    # -> loratadine
    "trimethoprine", "trimethaprin",                 # -> trimethoprim
    "implanton",                                     # -> implanon
    "weezy",                                         # -> wheezy
    "lightheaded",                                   # compound form
    "straightaway",                                  # clinical urgency
})

# Full detection set (backward compat)
_MEDICAL_TERMS = frozenset(_MEDICAL_TERM_CATEGORIES.keys()) | _ASR_VARIANTS

# Drug errors are highest clinical risk (wrong drug name -> wrong prescription)
HIGH_RISK_CATEGORIES = frozenset({"drugs"})

CATEGORIES = ["drugs", "conditions", "symptoms", "anatomy", "clinical"]


def get_medical_terms() -> frozenset[str]:
    """Return curated medical terms for M-WER evaluation."""
    return _MEDICAL_TERMS


def get_term_category(term: str) -> str | None:
    """Return category for a canonical medical term, or None for ASR variants."""
    return _MEDICAL_TERM_CATEGORIES.get(term)


def is_high_risk(term: str) -> bool:
    """Return True if term is in a high-risk category (drugs)."""
    cat = _MEDICAL_TERM_CATEGORIES.get(term)
    return cat is not None and cat in HIGH_RISK_CATEGORIES


if __name__ == "__main__":
    terms = get_medical_terms()
    cats = _MEDICAL_TERM_CATEGORIES
    print(f"Total medical terms: {len(terms)} ({len(cats)} canonical + {len(_ASR_VARIANTS)} ASR variants)")
    print()
    for cat in CATEGORIES:
        count = sum(1 for v in cats.values() if v == cat)
        print(f"  {cat:>12}: {count}")
    print(f"  {'ASR variants':>12}: {len(_ASR_VARIANTS)}")
