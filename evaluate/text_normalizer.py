#!/usr/bin/env python3
"""
Custom text normalizer for WER evaluation.

Based on Whisper's EnglishTextNormalizer with the following fixes:
  - "oh" is NOT treated as the digit zero (Whisper bug for spoken conversations)
  - Post-normalization word mappings for variant spellings / regional forms:
    ok/okay/k, mum/mom, yeah/yep/yes, alright/all right, kinda/kind of, etc.
  - Interjection variants (ohh/ohhh → oh, ahh/ahhh → ah)

Self-contained: no dependency on the openai-whisper package at runtime.
Requires: more_itertools (for windowed)
"""

import json
import os
import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union

from more_itertools import windowed


# ---------------------------------------------------------------------------
# Diacritics / symbol removal (from whisper.normalizers.basic)
# ---------------------------------------------------------------------------

ADDITIONAL_DIACRITICS = {
    "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE",
    "ß": "ss", "ẞ": "SS", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "th", "ł": "l", "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep=""):
    return "".join(
        (
            c if c in keep
            else (
                ADDITIONAL_DIACRITICS[c] if c in ADDITIONAL_DIACRITICS
                else (
                    "" if unicodedata.category(c) == "Mn"
                    else " " if unicodedata.category(c)[0] in "MSP" else c
                )
            )
        )
        for c in unicodedata.normalize("NFKD", s)
    )


# ---------------------------------------------------------------------------
# Number normalizer (from whisper, with "oh" fix)
# ---------------------------------------------------------------------------

class EnglishNumberNormalizer:
    """
    Convert any spelled-out numbers into arabic numbers.

    FIX: "oh" is removed from self.zeros so it is no longer converted to "0".
    Only "o" and "zero" are treated as the digit zero.
    """

    def __init__(self):
        super().__init__()

        # FIX: removed "oh" — in spoken medical conversations "oh" is an
        # interjection, not the digit zero.
        self.zeros = {"o", "zero"}

        self.ones = {
            name: i
            for i, name in enumerate(
                [
                    "one", "two", "three", "four", "five", "six", "seven",
                    "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                    "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                    "nineteen",
                ],
                start=1,
            )
        }
        self.ones_plural = {
            "sixes" if name == "six" else name + "s": (value, "s")
            for name, value in self.ones.items()
        }
        self.ones_ordinal = {
            "zeroth": (0, "th"), "first": (1, "st"), "second": (2, "nd"),
            "third": (3, "rd"), "fifth": (5, "th"), "twelfth": (12, "th"),
            **{
                name + ("h" if name.endswith("t") else "th"): (value, "th")
                for name, value in self.ones.items()
                if value > 3 and value != 5 and value != 12
            },
        }
        self.ones_suffixed = {**self.ones_plural, **self.ones_ordinal}

        self.tens = {
            "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        }
        self.tens_plural = {
            name.replace("y", "ies"): (value, "s") for name, value in self.tens.items()
        }
        self.tens_ordinal = {
            name.replace("y", "ieth"): (value, "th") for name, value in self.tens.items()
        }
        self.tens_suffixed = {**self.tens_plural, **self.tens_ordinal}

        self.multipliers = {
            "hundred": 100, "thousand": 1_000, "million": 1_000_000,
            "billion": 1_000_000_000, "trillion": 1_000_000_000_000,
            "quadrillion": 1_000_000_000_000_000,
            "quintillion": 1_000_000_000_000_000_000,
            "sextillion": 1_000_000_000_000_000_000_000,
            "septillion": 1_000_000_000_000_000_000_000_000,
            "octillion": 1_000_000_000_000_000_000_000_000_000,
            "nonillion": 1_000_000_000_000_000_000_000_000_000_000,
            "decillion": 1_000_000_000_000_000_000_000_000_000_000_000,
        }
        self.multipliers_plural = {
            name + "s": (value, "s") for name, value in self.multipliers.items()
        }
        self.multipliers_ordinal = {
            name + "th": (value, "th") for name, value in self.multipliers.items()
        }
        self.multipliers_suffixed = {
            **self.multipliers_plural, **self.multipliers_ordinal,
        }
        self.decimals = {*self.ones, *self.tens, *self.zeros}

        self.preceding_prefixers = {
            "minus": "-", "negative": "-", "plus": "+", "positive": "+",
        }
        self.following_prefixers = {
            "pound": "£", "pounds": "£", "euro": "€", "euros": "€",
            "dollar": "$", "dollars": "$", "cent": "¢", "cents": "¢",
        }
        self.prefixes = set(
            list(self.preceding_prefixers.values())
            + list(self.following_prefixers.values())
        )
        self.suffixers = {"per": {"cent": "%"}, "percent": "%"}
        self.specials = {"and", "double", "triple", "point"}

        self.words = set(
            key
            for mapping in [
                self.zeros, self.ones, self.ones_suffixed, self.tens,
                self.tens_suffixed, self.multipliers, self.multipliers_suffixed,
                self.preceding_prefixers, self.following_prefixers,
                self.suffixers, self.specials,
            ]
            for key in mapping
        )
        self.literal_words = {"one", "ones"}

    def process_words(self, words: List[str]) -> Iterator[str]:
        prefix: Optional[str] = None
        value: Optional[Union[str, int]] = None
        skip = False

        def to_fraction(s):
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result):
            nonlocal prefix, value
            result = str(result)
            if prefix is not None:
                result = prefix + result
            value = None
            prefix = None
            return result

        if len(words) == 0:
            return

        for prev, current, next in windowed([None] + words + [None], 3):
            if skip:
                skip = False
                continue

            next_is_numeric = next is not None and re.match(r"^\d+(\.\d+)?$", next)
            has_prefix = current[0] in self.prefixes
            current_without_prefix = current[1:] if has_prefix else current
            if re.match(r"^\d+(\.\d+)?$", current_without_prefix):
                f = to_fraction(current_without_prefix)
                assert f is not None
                if value is not None:
                    if isinstance(value, str) and value.endswith("."):
                        value = str(value) + str(current)
                        continue
                    else:
                        yield output(value)
                prefix = current[0] if has_prefix else prefix
                if f.denominator == 1:
                    value = f.numerator
                else:
                    value = current_without_prefix
            elif current not in self.words:
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in self.zeros:
                value = str(value or "") + "0"
            elif current in self.ones:
                ones = self.ones[current]
                if value is None:
                    value = ones
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:
                        assert value[-1] == "0"
                        value = value[:-1] + str(ones)
                    else:
                        value = str(value) + str(ones)
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
                else:
                    if value % 100 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
            elif current in self.ones_suffixed:
                ones, suffix = self.ones_suffixed[current]
                if value is None:
                    yield output(str(ones) + suffix)
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:
                        assert value[-1] == "0"
                        yield output(value[:-1] + str(ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                elif ones < 10:
                    if value % 10 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                else:
                    if value % 100 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                value = None
            elif current in self.tens:
                tens = self.tens[current]
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                else:
                    if value % 100 == 0:
                        value += tens
                    else:
                        value = str(value) + str(tens)
            elif current in self.tens_suffixed:
                tens, suffix = self.tens_suffixed[current]
                if value is None:
                    yield output(str(tens) + suffix)
                elif isinstance(value, str):
                    yield output(str(value) + str(tens) + suffix)
                else:
                    if value % 100 == 0:
                        yield output(str(value + tens) + suffix)
                    else:
                        yield output(str(value) + str(tens) + suffix)
            elif current in self.multipliers:
                multiplier = self.multipliers[current]
                if value is None:
                    value = multiplier
                elif isinstance(value, str) or value == 0:
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        value = p.numerator
                    else:
                        yield output(value)
                        value = multiplier
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
            elif current in self.multipliers_suffixed:
                multiplier, suffix = self.multipliers_suffixed[current]
                if value is None:
                    yield output(str(multiplier) + suffix)
                elif isinstance(value, str):
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        yield output(str(p.numerator) + suffix)
                    else:
                        yield output(value)
                        yield output(str(multiplier) + suffix)
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
                    yield output(str(value) + suffix)
                value = None
            elif current in self.preceding_prefixers:
                if value is not None:
                    yield output(value)
                if next in self.words or next_is_numeric:
                    prefix = self.preceding_prefixers[current]
                else:
                    yield output(current)
            elif current in self.following_prefixers:
                if value is not None:
                    prefix = self.following_prefixers[current]
                    yield output(value)
                else:
                    yield output(current)
            elif current in self.suffixers:
                if value is not None:
                    suffix = self.suffixers[current]
                    if isinstance(suffix, dict):
                        if next in suffix:
                            yield output(str(value) + suffix[next])
                            skip = True
                        else:
                            yield output(value)
                            yield output(current)
                    else:
                        yield output(str(value) + suffix)
                else:
                    yield output(current)
            elif current in self.specials:
                if next not in self.words and not next_is_numeric:
                    if value is not None:
                        yield output(value)
                    yield output(current)
                elif current == "and":
                    if prev not in self.multipliers:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "double" or current == "triple":
                    if next in self.ones or next in self.zeros:
                        repeats = 2 if current == "double" else 3
                        ones = self.ones.get(next, 0)
                        value = str(value or "") + str(ones) * repeats
                        skip = True
                    else:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "point":
                    if next in self.decimals or next_is_numeric:
                        value = str(value or "") + "."
                else:
                    raise ValueError(f"Unexpected token: {current}")
            else:
                raise ValueError(f"Unexpected token: {current}")

        if value is not None:
            yield output(value)

    def preprocess(self, s: str):
        results = []
        segments = re.split(r"\band\s+a\s+half\b", s)
        for i, segment in enumerate(segments):
            if len(segment.strip()) == 0:
                continue
            if i == len(segments) - 1:
                results.append(segment)
            else:
                results.append(segment)
                last_word = segment.rsplit(maxsplit=2)[-1]
                if last_word in self.decimals or last_word in self.multipliers:
                    results.append("point five")
                else:
                    results.append("and a half")
        s = " ".join(results)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", s)
        return s

    def postprocess(self, s: str):
        def combine_cents(m: Match):
            try:
                currency = m.group(1)
                integer = m.group(2)
                cents = int(m.group(3))
                return f"{currency}{integer}.{cents:02d}"
            except ValueError:
                return m.string

        def extract_cents(m: Match):
            try:
                return f"¢{int(m.group(1))}"
            except ValueError:
                return m.string

        s = re.sub(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b", combine_cents, s)
        s = re.sub(r"[€£$]0.([0-9]{1,2})\b", extract_cents, s)
        s = re.sub(r"\b1(s?)\b", r"one\1", s)
        return s

    def __call__(self, s: str):
        s = self.preprocess(s)
        s = " ".join(word for word in self.process_words(s.split()) if word is not None)
        s = self.postprocess(s)
        return s


# ---------------------------------------------------------------------------
# Spelling normalizer
# ---------------------------------------------------------------------------

class EnglishSpellingNormalizer:
    """British-American spelling mappings."""

    def __init__(self):
        mapping_path = os.path.join(os.path.dirname(__file__), "english.json")
        self.mapping = json.load(open(mapping_path))

    def __call__(self, s: str):
        return " ".join(self.mapping.get(word, word) for word in s.split())


# ---------------------------------------------------------------------------
# Post-normalization word mappings
# ---------------------------------------------------------------------------

POST_WORD_MAP = {
    # ok / okay / k — same word, different spellings
    "okay": "ok",
    "k": "ok",
    # British vs American
    "mum": "mom",
    "mummy": "mommy",
    # Interjection variants
    "ohh": "oh",
    "ohhh": "oh",
    "ahh": "ah",
    "ahhh": "ah",
    # Affirmative variants
    "yeah": "yes",
    "yep": "yes",
    "yea": "yes",
    "yah": "yes",
    "nope": "no",
    "nah": "no",
    # Informal contractions Whisper doesn't expand
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "do not know",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    # Farewell variants
    "goodbye": "bye",
    "bye bye": "bye",
    # alright / all right — same word
    "alright": "all right",
}


def _apply_post_word_map(text: str) -> str:
    """Apply word-level mappings after main normalization."""
    words = text.split()
    out = []
    for w in words:
        replacement = POST_WORD_MAP.get(w, w)
        out.append(replacement)
    return " ".join(out)


# ---------------------------------------------------------------------------
# Main normalizer (replaces Whisper's EnglishTextNormalizer)
# ---------------------------------------------------------------------------

class EnglishTextNormalizer:
    """
    Full English text normalizer for WER evaluation.

    Based on Whisper's EnglishTextNormalizer with fixes:
      - "oh" kept as interjection, not converted to digit "0"
      - Post-normalization word mappings for variant spellings
    """

    def __init__(self):
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        self.replacers = {
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",
            r"'s got\b": " has got",
            r"n't\b": " not",
            r"'re\b": " are",
            r"'s\b": " is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
        }
        self.standardize_numbers = EnglishNumberNormalizer()
        self.standardize_spellings = EnglishSpellingNormalizer()

    def __call__(self, s: str):
        s = s.lower()

        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = re.sub(self.ignore_patterns, "", s)
        s = re.sub(r"\s+'", "'", s)  # space before apostrophe

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)  # remove periods not before numbers
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")

        s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)

        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        s = re.sub(r"\s+", " ", s)

        # Apply post-normalization word mappings
        s = _apply_post_word_map(s)

        return s
