"""NLP and text-processing utilities for EvalGuard AI.

Provides fast, dependency-free token normalization, morphological suffix
reduction, and hybrid similarity metrics.
"""

import re
from typing import List, Optional, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "can", "could", "would", "should", "shall", "will", "may", "might", "must",
    "that", "which", "who", "whom", "this", "these", "those", "am", "it", "its",
    "i", "me", "my", "you", "your", "he", "him", "his", "she", "her", "we", "our", "they", "them", "their"
}


def stem_token(token: str) -> str:
    """Lightweight rule-based stemmer to match morphological variations

    (e.g., encryption/encrypted -> encrypt, cancellation/cancelled -> cancel).
    """
    w = token.lower().strip()
    if len(w) <= 3:
        return w

    suffixes = [
        "isation", "ization", "tion", "tions", "sion", "sions",
        "ing", "ings", "ed", "ies", "es", "ment", "ments", "able", "ible", "ly", "s"
    ]
    for sfx in suffixes:
        if w.endswith(sfx) and len(w) - len(sfx) >= 3:
            w = w[:-len(sfx)]
            break

    # Normalize double trailing letters (e.g. cancell -> cancel)
    if len(w) > 3 and w[-1] == w[-2]:
        w = w[:-1]
    return w


def extract_stemmed_tokens(text: str, remove_stopwords: bool = True) -> Set[str]:
    """Extracts stemmed alphanumeric tokens from text."""
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
    result = set()
    for t in raw_tokens:
        if remove_stopwords and t in STOPWORDS:
            continue
        result.add(stem_token(t))
    return result


def token_overlap_score(source: str, reference: str) -> float:
    """Measures what fraction of significant terms in source are found in reference."""
    src_tokens = extract_stemmed_tokens(source)
    if not src_tokens:
        return 1.0 if not source.strip() else 0.0

    ref_tokens = extract_stemmed_tokens(reference)
    shared = src_tokens & ref_tokens
    return len(shared) / len(src_tokens)


def compute_hybrid_similarity(
    source_texts: List[str],
    reference_texts: List[str],
) -> np.ndarray:
    """Computes a matrix of hybrid similarities combining TF-IDF cosine similarity

    with stemmed token containment.
    """
    if not source_texts or not reference_texts:
        return np.zeros((len(source_texts), len(reference_texts)))

    # 1. TF-IDF Cosine
    corpus = list(set(source_texts + reference_texts))
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf_mat = vec.fit_transform(corpus)
        corpus_lookup = {text: idx for idx, text in enumerate(corpus)}

        src_indices = [corpus_lookup[t] for t in source_texts]
        ref_indices = [corpus_lookup[t] for t in reference_texts]

        tfidf_sim = cosine_similarity(tfidf_mat[src_indices], tfidf_mat[ref_indices])
    except ValueError:
        tfidf_sim = np.zeros((len(source_texts), len(reference_texts)))

    # 2. Token overlap matrix
    overlap_mat = np.zeros((len(source_texts), len(reference_texts)))
    for i, src in enumerate(source_texts):
        for j, ref in enumerate(reference_texts):
            overlap_mat[i, j] = token_overlap_score(src, ref)

    # Hybrid blend: maximum of both signals with slight preference for overlap
    hybrid = np.maximum(tfidf_sim, overlap_mat)
    return hybrid
