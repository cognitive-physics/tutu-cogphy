"""Local text complexity heuristics for the cognitive engine prototype."""

import math
import unicodedata
import zlib
from collections import Counter

MIN_WORDS = 5


def _is_semantic_char(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


def vocab_density(text: str) -> float:
    if not text.strip():
        return 0.0
    words = text.lower().split()
    n = len(words)
    if n == 0:
        return 0.0
    base = len(set(words)) / n
    penalty = min(n / MIN_WORDS, 1.0)
    return base * penalty


def compression_ratio(text: str) -> float:
    if not text.strip():
        return 0.0
    encoded = text.encode("utf-8")
    compressed = zlib.compress(encoded, level=9)
    ratio = 1.0 - len(compressed) / max(len(encoded), 1)
    return float(1.0 - max(ratio, 0.0))


def entropy_score(text: str) -> float:
    semantic_chars = [ch for ch in text if _is_semantic_char(ch)]
    if len(semantic_chars) < 2:
        return 0.0
    total = len(semantic_chars)
    counter = Counter(semantic_chars)
    entropy = -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)
    max_entropy = math.log2(total)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _length_penalty(text: str) -> float:
    semantic_len = sum(1 for ch in text if _is_semantic_char(ch))
    return min(semantic_len / 10.0, 1.0)


def compute_complexity(text: str) -> float:
    if not text.strip():
        return 0.0
    w1, w2, w3 = 0.4, 0.3, 0.3
    base_score = (
        w1 * vocab_density(text)
        + w2 * compression_ratio(text)
        + w3 * entropy_score(text)
    )
    return float(min(base_score * _length_penalty(text), 1.0))


def complexity_ratio_between(text_a: str, text_b: str) -> float:
    ca = compute_complexity(text_a)
    cb = compute_complexity(text_b)
    if ca < 1e-6:
        return 1.0
    return float(min(cb / ca, 1.0))
