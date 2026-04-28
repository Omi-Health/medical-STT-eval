#!/usr/bin/env python3
"""Shared audio chunking and transcript merge helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    """In-memory audio chunk with source time bounds."""

    index: int
    audio: Any
    sample_rate: int
    start_s: float
    end_s: float


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim the result."""

    return re.sub(r"\s+", " ", text).strip()


def to_mono_float32(audio: Any) -> np.ndarray:
    """Convert soundfile-style audio to mono float32."""

    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr


def split_audio_array(
    audio: Any,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
    *,
    min_final_seconds: float = 1.0,
    fade_seconds: float = 0.0,
) -> List[AudioChunk]:
    """Split audio into fixed overlapping chunks.

    This intentionally does not do silence snapping. It is a deterministic
    splitter used for model ablations where only window and overlap should vary.
    """

    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")

    mono = to_mono_float32(audio)
    chunk_samples = int(chunk_seconds * sample_rate)
    step_samples = int((chunk_seconds - overlap_seconds) * sample_rate)
    min_final_samples = int(min_final_seconds * sample_rate)
    fade_samples = int(fade_seconds * sample_rate)

    chunks: List[AudioChunk] = []
    start = 0
    index = 0
    total = len(mono)

    while start < total:
        end = min(start + chunk_samples, total)
        if start > 0 and end - start < min_final_samples:
            break

        segment = mono[start:end].copy()
        if fade_samples > 0 and len(segment) > fade_samples * 2:
            if start > 0:
                segment[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            if end < total:
                segment[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

        chunks.append(
            AudioChunk(
                index=index,
                audio=segment,
                sample_rate=sample_rate,
                start_s=start / sample_rate,
                end_s=end / sample_rate,
            )
        )

        if end >= total:
            break
        start += step_samples
        index += 1

    return chunks


def merge_by_exact_word_overlap(existing: str, addition: str, max_overlap_words: int = 40) -> str:
    """Append text while removing an exact duplicate word overlap."""

    existing = normalize_whitespace(existing)
    addition = normalize_whitespace(addition)
    if not existing:
        return addition
    if not addition:
        return existing

    left_words = existing.split()
    right_words = addition.split()
    max_k = min(len(left_words), len(right_words), max_overlap_words)
    for k in range(max_k, 0, -1):
        if left_words[-k:] == right_words[:k]:
            return " ".join(left_words + right_words[k:])
    return normalize_whitespace(existing + " " + addition)


def find_longest_common_sequence(sequences: Sequence[str], match_by_words: bool = True) -> str:
    """Merge transcript chunks by selecting the strongest text overlap.

    This preserves the algorithm used by the existing Canary-Qwen and Phi-4
    scripts: find the best overlapping alignment, keep the left half from the
    previous transcript, and the right half from the next transcript.
    """

    if not sequences:
        return ""
    if len(sequences) == 1:
        return sequences[0]

    if match_by_words:
        processed_sequences = [re.findall(r"\S+|\s+", seq) for seq in sequences]
    else:
        processed_sequences = [list(seq) for seq in sequences]

    left_sequence = processed_sequences[0]
    left_length = len(left_sequence)

    for right_sequence in processed_sequences[1:]:
        max_matching = 0.0
        right_length = len(right_sequence)
        max_indices = (left_length, left_length, 0, 0)

        for i in range(1, left_length + right_length + 1):
            eps = float(i) / 10000.0
            left_start = max(0, left_length - i)
            left_stop = min(left_length, left_length + right_length - i)
            right_start = max(0, i - left_length)
            right_stop = min(right_length, i)

            left_part = left_sequence[left_start:left_stop]
            right_part = right_sequence[right_start:right_stop]
            if len(left_part) != len(right_part):
                continue

            matches = sum(a == b for a, b in zip(left_part, right_part))
            matching = matches / float(i) + eps if i > 0 else 0.0
            if matches >= 2 and matching > max_matching:
                max_matching = matching
                max_indices = (left_start, left_stop, right_start, right_stop)

        left_start, left_stop, right_start, right_stop = max_indices
        left_mid = (left_stop + left_start) // 2
        right_mid = (right_stop + right_start) // 2
        left_sequence = left_sequence[:left_mid] + right_sequence[right_mid:]
        left_length = len(left_sequence)

    return "".join(left_sequence)


def _text_only(items: Iterable[str | Tuple[str, float, float]]) -> List[str]:
    texts: List[str] = []
    for item in items:
        if isinstance(item, tuple):
            texts.append(item[0])
        else:
            texts.append(item)
    return texts


def merge_by_groq_lcs(items: Iterable[str | Tuple[str, float, float]]) -> str:
    """Merge chunk transcripts with the Canary-Qwen/Phi-4 LCS strategy."""

    return normalize_whitespace(find_longest_common_sequence(_text_only(items), match_by_words=True))


def find_suffix_prefix_lcs_overlap(
    text1: str,
    text2: str,
    *,
    search_window_words: int,
    min_overlap_words: int = 2,
) -> Tuple[int, int, float]:
    """Find a fuzzy suffix/prefix overlap between two transcript strings."""

    words1 = text1.split()
    words2 = text2.split()
    if len(words1) < min_overlap_words or len(words2) < min_overlap_words:
        return -1, -1, 0.0

    search_window = min(len(words1), search_window_words)
    best_overlap_start = -1
    best_overlap_end = -1
    best_score = 0.0

    for i in range(max(0, len(words1) - search_window), len(words1)):
        for j in range(min(search_window, len(words2))):
            if words1[i] != words2[j]:
                continue
            matcher = SequenceMatcher(None, words1[i:], words2[: j + search_window])
            match = matcher.find_longest_match(0, len(words1[i:]), 0, len(words2[: j + search_window]))
            if match.size < min_overlap_words:
                continue
            position_score = 1.0 - (j / search_window if search_window > 0 else 0.0)
            length_score = match.size / search_window if search_window > 0 else 0.0
            score = (position_score + length_score) / 2.0
            if score > best_score:
                best_score = score
                best_overlap_start = i
                best_overlap_end = j + match.size

    return best_overlap_start, best_overlap_end, best_score


def merge_by_suffix_prefix_lcs(
    items: Iterable[str | Tuple[str, float, float]],
    *,
    overlap_seconds: float,
    words_per_second: int = 20,
    threshold: float = 0.3,
    min_overlap_words: int = 2,
) -> str:
    """Merge text chunks using fuzzy suffix/prefix overlap."""

    texts = [text for text in _text_only(items) if text.strip()]
    if not texts:
        return ""

    merged = texts[0]
    search_window_words = max(min_overlap_words, int(overlap_seconds * words_per_second))
    for text in texts[1:]:
        overlap_start, _overlap_end, score = find_suffix_prefix_lcs_overlap(
            merged,
            text,
            search_window_words=search_window_words,
            min_overlap_words=min_overlap_words,
        )
        if overlap_start > 0 and score > threshold:
            merged = " ".join(merged.split()[:overlap_start]) + " " + text
        else:
            merged = merged + " " + text
        merged = normalize_whitespace(merged)

    return merged
