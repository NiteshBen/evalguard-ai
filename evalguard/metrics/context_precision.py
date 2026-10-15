"""Context precision metric.

Evaluates whether the most relevant context documents or chunks appear
near the top of the retrieved context list.
"""

from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from evalguard.metrics.base import BaseMetric
from evalguard.models import MetricScore, TestCase


class ContextPrecisionMetric(BaseMetric):
    """Measures the ranking quality of retrieved contexts using Mean Reciprocal Rank (MRR)

    and Mean Average Precision at K (MAP@K).
    """

    def __init__(self, threshold: float = 0.65, relevance_cutoff: float = 0.25):
        super().__init__(
            name="context_precision",
            threshold=threshold,
            description="Evaluates ranking signal: whether relevant contexts are ranked at the top",
        )
        self.relevance_cutoff = relevance_cutoff

    def evaluate(self, test_case: TestCase) -> MetricScore:
        contexts = test_case.retrieved_contexts
        if not contexts:
            return MetricScore.evaluate(
                name=self.name,
                score=0.0,
                threshold=self.threshold,
                reason="No retrieved contexts available.",
                metadata={"num_contexts": 0},
            )

        from evalguard.nlp_utils import compute_hybrid_similarity

        target_texts = [t for t in [test_case.query, test_case.expected_output, test_case.actual_output] if t]
        sim_mat = compute_hybrid_similarity(target_texts, contexts)
        sims = np.max(sim_mat, axis=0)

        # Determine binary relevance based on cutoff
        relevance_flags = [bool(s >= self.relevance_cutoff) for s in sims]

        if not any(relevance_flags):
            # None passed the threshold; check if at least best match has some signal
            best_idx = int(np.argmax(sims)) if len(sims) > 0 else 0
            best_sim = float(sims[best_idx]) if len(sims) > 0 else 0.0
            score = max(0.0, best_sim)
            return MetricScore.evaluate(
                name=self.name,
                score=score,
                threshold=self.threshold,
                reason=f"No context achieved relevance cutoff >= {self.relevance_cutoff} (best was {best_sim:.2f}).",
                metadata={"similarities": [round(float(s), 3) for s in sims]},
            )

        # Compute Average Precision at K
        cum_relevant = 0
        running_precisions = []
        for k, is_rel in enumerate(relevance_flags, start=1):
            if is_rel:
                cum_relevant += 1
                running_precisions.append(cum_relevant / k)

        avg_precision = sum(running_precisions) / len(running_precisions) if running_precisions else 0.0

        # Also incorporate MRR (rank of first relevant chunk)
        first_rel_rank = next((idx + 1 for idx, rel in enumerate(relevance_flags) if rel), len(contexts))
        mrr = 1.0 / first_rel_rank

        # Harmonic combination
        final_score = (0.7 * avg_precision) + (0.3 * mrr)

        reason = (
            f"Precision@K: {avg_precision:.2f}, First relevant chunk at rank {first_rel_rank} (MRR={mrr:.2f}), "
            f"{sum(relevance_flags)}/{len(contexts)} chunks relevant."
        )

        return MetricScore.evaluate(
            name=self.name,
            score=final_score,
            threshold=self.threshold,
            reason=reason,
            metadata={
                "average_precision": round(avg_precision, 4),
                "mrr": round(mrr, 4),
                "first_relevant_rank": first_rel_rank,
                "relevance_flags": relevance_flags,
                "similarities": [round(float(s), 3) for s in sims],
            },
        )
