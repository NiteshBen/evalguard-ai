"""Context recall metric.

Evaluates whether the retrieved contexts contain all factual points or ground-truth
knowledge required to answer the query.
"""

import re
from typing import List, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from evalguard.metrics.base import BaseMetric
from evalguard.models import MetricScore, TestCase


class ContextRecallMetric(BaseMetric):
    """Measures the proportion of ground-truth statements / key information

    that is present and recoverable within the retrieved contexts.
    """

    def __init__(self, threshold: float = 0.70):
        super().__init__(
            name="context_recall",
            threshold=threshold,
            description="Evaluates context coverage: whether retrieved chunks contain required facts",
        )

    def _extract_propositions(self, text: str) -> List[str]:
        """Decomposes reference text into discrete proposition clauses."""
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        propositions = [s.strip() for s in raw if len(s.strip().split()) >= 3]
        return propositions or ([text.strip()] if text.strip() else [])

    def evaluate(self, test_case: TestCase) -> MetricScore:
        contexts = test_case.retrieved_contexts
        if not contexts:
            return MetricScore.evaluate(
                name=self.name,
                score=0.0,
                threshold=self.threshold,
                reason="No retrieved contexts provided.",
                metadata={"covered_propositions": 0, "total_propositions": 0},
            )

        reference = test_case.expected_output or test_case.actual_output
        propositions = self._extract_propositions(reference)

        if not propositions:
            return MetricScore.evaluate(
                name=self.name,
                score=1.0,
                threshold=self.threshold,
                reason="No reference propositions to verify.",
                metadata={"covered_propositions": 0, "total_propositions": 0},
            )

        combined_context = " ".join(contexts)
        context_sentences = re.split(r"(?<=[.!?])\s+", combined_context)
        context_sentences = [s.strip() for s in context_sentences if len(s.strip().split()) >= 3]
        if not context_sentences:
            context_sentences = contexts

        from evalguard.nlp_utils import compute_hybrid_similarity

        sim_matrix = compute_hybrid_similarity(propositions, context_sentences)

        covered = 0
        coverage_details = []
        for idx, prop in enumerate(propositions):
            max_sim = float(np.max(sim_matrix[idx])) if sim_matrix.shape[1] > 0 else 0.0
            is_covered = max_sim >= 0.35
            if is_covered:
                covered += 1
            coverage_details.append({
                "proposition": prop,
                "max_similarity": round(max_sim, 3),
                "covered": is_covered,
            })

        recall_score = covered / len(propositions)
        reason = f"Covered {covered}/{len(propositions)} reference facts in retrieved context (recall={recall_score:.2f})."

        return MetricScore.evaluate(
            name=self.name,
            score=recall_score,
            threshold=self.threshold,
            reason=reason,
            metadata={
                "total_propositions": len(propositions),
                "covered_propositions": covered,
                "coverage_details": coverage_details,
            },
        )
