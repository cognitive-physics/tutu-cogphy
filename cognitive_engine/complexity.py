"""Local text complexity heuristics for the cognitive engine prototype."""

import math
import os
import re
import unicodedata
import zlib
from collections import Counter
from typing import Optional, Tuple

MIN_WORDS = 5

# Question indicators for follow-up detection
QUESTION_PATTERNS = [
    r'\?',  # Question mark
    r'\b(what|where|when|who|why|how|which)\b',  # Question words
    r'\b(doesn\'t|didn\'t|don\'t|won\'t|can\'t|couldn\'t|shouldn\'t)\b.*understand',
    r'\b(don\'t understand|没懂|什么意思|解释一下|再说一遍)\b',
]

# Emotion lexicon (lightweight, Chinese-English)
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic', 'happy',
    'pleased', 'delighted', 'grateful', 'appreciate', 'love', 'like',
    '好', '太好了', '太棒了', '很高兴', '感谢', '喜欢', '很满意',
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry',
    'frustrated', 'disappointed', 'sad', 'upset', 'afraid', 'worried',
    '坏', '糟糕', '可怕', '讨厌', '不满', '失望', '害怕', '担心',
}

NEUTRAL_INTENSITY_WORDS = {
    'very', 'really', 'extremely', 'absolutely', 'quite', 'rather',
    'very', '非常', '极其', '完全', '相当', '真的',
}

# Global state for embedding model
_EMBEDDING_MODEL = None
_EMBEDDING_MODEL_NAME = os.environ.get("COMPLEXITY_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_USE_EMBEDDING = os.environ.get("COMPLEXITY_USE_EMBEDDING", "true").lower() == "true"


def _load_embedding_model():
    """Lazy-load embedding model, return None if unavailable."""
    global _EMBEDDING_MODEL
    
    if not _USE_EMBEDDING:
        return None
    
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL
    
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer(_EMBEDDING_MODEL_NAME)
        return _EMBEDDING_MODEL
    except ImportError:
        # sentence-transformers not installed
        return None
    except Exception as e:
        # Model download or loading failed
        print(f"Warning: Failed to load embedding model {_EMBEDDING_MODEL_NAME}: {e}")
        return None


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


def _compute_complexity_heuristic(text: str) -> Tuple[float, str]:
    """
    Heuristic complexity (fallback): zlib compression + vocab entropy.
    
    Returns:
        (complexity_score, method_label)
    """
    if not text.strip():
        return 0.0, "heuristic_empty"
    
    w1, w2, w3 = 0.4, 0.3, 0.3
    base_score = (
        w1 * vocab_density(text)
        + w2 * compression_ratio(text)
        + w3 * entropy_score(text)
    )
    score = float(min(base_score * _length_penalty(text), 1.0))
    return score, "heuristic"


def _compute_complexity_embedding(text: str) -> Tuple[Optional[float], str]:
    """
    Embedding-based complexity: semantic density via sentence embeddings.
    
    Approach:
    - Split text into sentences
    - Embed each sentence
    - Measure average pairwise divergence (information diversity)
    - Combine with vocabulary diversity
    
    High information density = high divergence between sentences (each sentence adds new info)
    Low information density = high similarity (repetitive/redundant)
    
    Returns:
        (complexity_score or None, method_label)
    """
    model = _load_embedding_model()
    if model is None:
        return None, "embedding_unavailable"
    
    if not text.strip():
        return 0.0, "embedding_empty"
    
    try:
        # Split into sentences (naive: by period/question mark)
        sentences = re.split(r'[。！？\.\!\?]+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0, "embedding_no_sentences"
        
        if len(sentences) == 1:
            # Single sentence: use word diversity as proxy
            words = sentences[0].lower().split()
            if not words:
                return 0.0, "embedding_no_words"
            diversity = len(set(words)) / len(words)
            return float(min(diversity, 1.0)), "embedding_single_sentence"
        
        # Embed sentences
        embeddings = model.encode(sentences, convert_to_tensor=False)
        
        # Compute pairwise cosine similarity (information overlap)
        import numpy as np
        embeddings = np.array(embeddings)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (norms + 1e-9)
        
        # Pairwise cosine similarities
        similarities = embeddings_norm @ embeddings_norm.T
        
        # Average pairwise similarity (excluding self-similarity)
        mask = 1.0 - np.eye(len(sentences))
        avg_similarity = np.sum(similarities * mask) / np.sum(mask)
        
        # Information density: inverse of similarity (high divergence = high density)
        # Clamp to [0, 1]
        information_density = 1.0 - avg_similarity
        information_density = float(np.clip(information_density, 0.0, 1.0))
        
        # Combine with vocabulary diversity
        vocab_score = vocab_density(text)
        combined_score = 0.6 * information_density + 0.4 * vocab_score
        
        return float(min(combined_score, 1.0)), "embedding"
    
    except Exception as e:
        # Fallback on embedding error
        print(f"Warning: Embedding-based complexity failed: {e}")
        return None, f"embedding_error: {e}"


def compute_complexity(text: str) -> float:
    """
    Compute semantic/information complexity of text.
    
    Tries embedding-based approach first (measures semantic information density).
    Falls back to heuristic (zlib compression + entropy) if embedding unavailable.
    
    Args:
        text: Input text
    
    Returns:
        complexity: float in [0, 1]
            0 = very simple (empty, repetitive, low information)
            1 = very complex (high information density, diverse content)
            Unit: dimensionless information density score.
    """
    # Try embedding-based first
    score_emb, method_emb = _compute_complexity_embedding(text)
    if score_emb is not None:
        return score_emb
    
    # Fallback to heuristic
    score_heur, method_heur = _compute_complexity_heuristic(text)
    return score_heur


def get_complexity_method(text: str) -> str:
    """
    Return which method was used for complexity computation.
    
    For debugging/logging purposes.
    """
    score_emb, method_emb = _compute_complexity_embedding(text)
    if score_emb is not None:
        return method_emb
    return "heuristic"


def complexity_ratio_between(text_a: str, text_b: str) -> float:
    """
    Ratio of complexities: cb / ca, clipped to [0, 1].
    
    Signature unchanged from v1 for backward compatibility.
    """
    ca = compute_complexity(text_a)
    cb = compute_complexity(text_b)
    if ca < 1e-6:
        return 1.0
    return float(min(cb / ca, 1.0))


def estimate_emotion_score(text: str) -> float:
    """
    Estimate emotional tone from text using lightweight lexicon-based approach.
    
    Args:
        text: Input text
    
    Returns:
        emotion_score: float in [0, 1]
            0.0 = negative/frustrated tone,
            0.5 = neutral tone,
            1.0 = positive/satisfied tone.
            Unit: dimensionless emotional polarity score.
    """
    if not text or not text.strip():
        return 0.5  # Neutral if empty
    
    text_lower = text.lower()
    words = set(text_lower.split())
    
    positive_count = len(words & POSITIVE_WORDS)
    negative_count = len(words & NEGATIVE_WORDS)
    intensity_boost = len(words & NEUTRAL_INTENSITY_WORDS) * 0.1
    
    # Base score: neutral at 0.5
    if positive_count == 0 and negative_count == 0:
        return 0.5  # Neutral
    
    # Polarity: (positive - negative) / (positive + negative)
    total = positive_count + negative_count
    if total > 0:
        polarity = (positive_count - negative_count) / total
        # Map [-1, 1] to [0, 1]
        emotion_score = 0.5 + polarity * 0.5
        # Apply intensity boost (max 0.1 additional)
        emotion_score = min(emotion_score + intensity_boost, 1.0)
        return float(emotion_score)
    
    return 0.5


def estimate_follow_up_rate(user_reply: str) -> float:
    """
    Detect if user's reply contains follow-up questions, confusion, or clarification requests.
    
    Args:
        user_reply: User's response text
    
    Returns:
        follow_up_rate: float in [0, 1]
            0 = no follow-up questions (user understood),
            1 = strong follow-up/confusion signals (explicit follow-up).
            Unit: dimensionless rate of follow-up signals.
    """
    if not user_reply or not user_reply.strip():
        return 0.0
    
    text_lower = user_reply.lower()
    match_count = 0
    total_patterns = len(QUESTION_PATTERNS)
    
    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            match_count += 1
    
    rate = float(match_count) / total_patterns if total_patterns > 0 else 0.0
    return float(min(rate, 1.0))


def estimate_paraphrase_accuracy(ai_output: str, user_reply: str) -> float:
    """
    Measure how well user's reply echoes/confirms AI output (paraphrase accuracy).
    
    Uses a combination of:
    - Lexical overlap (shared significant words)
    - Structural similarity (complexity_ratio_between)
    - Length coherence
    
    Args:
        ai_output: AI's previous response
        user_reply: User's follow-up reply
    
    Returns:
        paraphrase_acc: float in [0, 1]
            0 = user reply completely different/contradicts AI output,
            1 = user reply closely mirrors/confirms AI output.
            Unit: dimensionless accuracy score.
    """
    if not ai_output or not user_reply or not ai_output.strip() or not user_reply.strip():
        return 0.5  # Neutral if either is empty
    
    ai_words = set(ai_output.lower().split())
    user_words = set(user_reply.lower().split())
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were', 'be', 'been', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    ai_words = ai_words - stop_words
    user_words = user_words - stop_words
    
    if not ai_words and not user_words:
        return 1.0  # Both empty after filtering → treat as consistent
    
    if not ai_words or not user_words:
        return 0.0  # One empty, one non-empty → no paraphrase
    
    # Jaccard similarity: overlap / union
    intersection = len(ai_words & user_words)
    union = len(ai_words | user_words)
    jaccard = float(intersection) / union if union > 0 else 0.0
    
    # Length coherence: user reply should be non-trivially short relative to AI (no just echoing)
    # but not so different it's completely off-topic
    length_ratio = min(len(user_reply.split()), len(ai_output.split())) / max(len(user_reply.split()), len(ai_output.split()) + 1)
    
    # Weighted combination
    paraphrase_score = 0.7 * jaccard + 0.3 * min(length_ratio, 1.0)
    return float(min(paraphrase_score, 1.0))


def estimate_latency_norm(latency_ms: float) -> float:
    """
    Normalize response latency to [0, 1].
    
    Args:
        latency_ms: Latency in milliseconds (time between AI output and user reply)
    
    Returns:
        latency_norm: float in [0, 1]
            0 = instantaneous (user ready immediately, likely understood),
            1 = very slow (user took long time, possibly confusion/rereading).
            Unit: normalized latency score.
            
    Baseline: 2000ms (2 seconds) ~ 0.5 (neutral).
    < 500ms ~ 0.1 (very fast, understood).
    > 10000ms ~ 0.9 (very slow, possible confusion).
    """
    if latency_ms < 0:
        return 0.5  # Invalid input → neutral
    
    # Use sigmoid-like curve: log scale with baseline at 2000ms
    baseline_ms = 2000.0
    if latency_ms < 100:
        return 0.0  # Instant response
    
    # Normalize to baseline
    normalized = (latency_ms / baseline_ms) ** 0.5  # Square root for softer curve
    return float(min(normalized, 1.0))
