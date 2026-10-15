"""Answer relevance and semantic completeness metric.

Evaluates whether the generated response directly, accurately, and adequately
addresses the original user question or prompt.
"""

import re
from typing import List, Optional, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from evalguard.metrics.base import BaseMetric
from evalguard.models import MetricScore, TestCase


class AnswerRelevanceMetric(BaseMetric):
    """Measures semantic alignment between the user's query and the actual output."""

    def __init__(self, threshold: float = 0.60):
        super().__init__(
            name="answer_relevance",
            threshold=threshold,
            description="Semantic relevance and completeness of response relative to user prompt",
        )

    def _extract_intent_keywords(self, query: str) -> Set[str]:
        """Extracts non-stopword alphanumeric tokens representing core user intent."""
        stopwords = {
            "what", "where", "when", "how", "who", "which", "why", "is", "are", "was",
            "were", "the", "a", "an", "and", "or", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "can", "could", "would", "should", "do",
            "does", "did", "please", "tell", "explain", "describe", "give", "me",
            "i", "my", "myself", "you", "your", "yours", "yourself", "we", "our",
            "he", "him", "his", "she", "her", "they", "them", "their", "it", "its"
        }
        tokens = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", query.lower())
        return {t for t in tokens if t not in stopwords}

    def evaluate(self, test_case: TestCase) -> MetricScore:
        query = test_case.query.strip()
        output = test_case.actual_output.strip()

        if not output:
            return MetricScore.evaluate(
                name=self.name,
                score=0.0,
                threshold=self.threshold,
                reason="Actual output is empty.",
                metadata={"query_overlap": 0.0, "cosine_similarity": 0.0},
            )

        # 1. Cosine similarity using TF-IDF
        try:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            docs = [query, output]
            if test_case.expected_output:
                docs.append(test_case.expected_output.strip())
            tfidf = vectorizer.fit_transform(docs)
            cos_sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        except ValueError:
            cos_sim = 0.0

        from evalguard.nlp_utils import extract_stemmed_tokens, stem_token, token_overlap_score

        # 2. Intent keyword coverage with morphological stemming
        intent_kw = {stem_token(w) for w in self._extract_intent_keywords(query)}
        output_stemmed = extract_stemmed_tokens(output, remove_stopwords=False)
        if intent_kw:
            overlap_ratio = len(intent_kw & output_stemmed) / len(intent_kw)
        else:
            overlap_ratio = 1.0

        # 3. Ground truth alignment if expected_output is provided
        gt_sim: Optional[float] = None
        if test_case.expected_output:
            try:
                tfidf_gt = float(cosine_similarity(tfidf[1:2], tfidf[2:3])[0][0])
            except (ValueError, IndexError):
                tfidf_gt = 0.0
            tok_gt = token_overlap_score(test_case.expected_output, output)
            gt_sim = max(tfidf_gt, tok_gt)

        query_coverage = token_overlap_score(query, output)

        # 4. Length penalty (penalize 1-word evasive answers unless query is simple boolean)
        words_count = len(output.split())
        length_multiplier = 1.0
        if words_count < 3 and len(query.split()) > 4:
            length_multiplier = 0.5

        # Weighted combination
        query_signal = (0.20 * cos_sim) + (0.45 * overlap_ratio) + (0.35 * query_coverage)
        if gt_sim is not None:
            raw_score = max(query_signal, (0.5 * query_signal) + (0.5 * gt_sim))
        else:
            raw_score = query_signal

        final_score = raw_score * length_multiplier

        reason = (
            f"Query relevance: cosine={cos_sim:.2f}, intent_overlap={overlap_ratio:.2f}"
            + (f", ground_truth_sim={gt_sim:.2f}" if gt_sim is not None else "")
            + f" (word_count={words_count})"
        )

        return MetricScore.evaluate(
            name=self.name,
            score=final_score,
            threshold=self.threshold,
            reason=reason,
            metadata={
                "cosine_similarity": round(cos_sim, 4),
                "intent_keyword_overlap": round(overlap_ratio, 4),
                "ground_truth_similarity": round(gt_sim, 4) if gt_sim is not None else None,
                "word_count": words_count,
            },
        )
