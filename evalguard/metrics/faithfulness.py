"""Faithfulness and hallucination detection metric.

Evaluates whether the factual claims made in the model's output are strictly
grounded and supported by the retrieved contexts.
"""

import re
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from evalguard.metrics.base import BaseMetric
from evalguard.models import ClaimStatus, ClaimVerification, MetricScore, TestCase


class FaithfulnessMetric(BaseMetric):
    """Measures hallucination rate by extracting proposition claims from actual_output

    and verifying each against retrieved_contexts.
    """

    def __init__(self, threshold: float = 0.75, strict_mode: bool = False):
        super().__init__(
            name="faithfulness",
            threshold=threshold,
            description="Proportion of factual claims in output supported by retrieved context",
        )
        self.strict_mode = strict_mode

    def _split_into_claims(self, text: str) -> List[str]:
        """Splits output into discrete sentence-level factual claims."""
        if not text or not text.strip():
            return []

        # Split on sentence boundaries while preserving structure
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text.strip())
        claims = []
        for s in raw_sentences:
            cleaned = s.strip()
            # Ignore empty sentences or trivial conversational filler
            if len(cleaned.split()) >= 3:
                claims.append(cleaned)

        if not claims and text.strip():
            claims = [text.strip()]
        return claims

    def _detect_contradiction(self, claim: str, evidence: str) -> bool:
        """Detects explicit negation discrepancies between claim and evidence."""
        negations = {"not", "never", "no", "neither", "nor", "cannot", "hasn't", "haven't", "won't"}
        claim_words = set(re.findall(r"\b\w+\b", claim.lower()))
        ev_words = set(re.findall(r"\b\w+\b", evidence.lower()))

        claim_has_neg = bool(claim_words & negations)
        ev_has_neg = bool(ev_words & negations)

        # If one has negation and the other doesn't, while sharing significant core content
        shared = (claim_words & ev_words) - negations
        if len(shared) >= 3 and claim_has_neg != ev_has_neg:
            return True
        return False

    def evaluate(self, test_case: TestCase) -> MetricScore:
        claims = self._split_into_claims(test_case.actual_output)

        if not claims:
            return MetricScore.evaluate(
                name=self.name,
                score=1.0 if not test_case.actual_output else 0.0,
                threshold=self.threshold,
                reason="Empty output evaluated.",
                metadata={"total_claims": 0, "verifications": []},
            )

        if not test_case.retrieved_contexts:
            # Without any retrieved context, claims cannot be verified
            return MetricScore.evaluate(
                name=self.name,
                score=0.0,
                threshold=self.threshold,
                reason="No retrieved contexts provided to ground output claims.",
                metadata={"total_claims": len(claims), "supported_claims": 0, "verifications": []},
            )

        # Context candidates include individual sentences, full context chunks, and whole context
        context_sentences = []
        for ctx in test_case.retrieved_contexts:
            context_sentences.append(ctx.strip())
            for s in re.split(r"(?<=[.!?])\s+", ctx.strip()):
                s_clean = s.strip()
                if len(s_clean.split()) >= 3:
                    context_sentences.append(s_clean)
        if len(test_case.retrieved_contexts) > 1:
            context_sentences.append(" ".join(test_case.retrieved_contexts))

        from evalguard.nlp_utils import compute_hybrid_similarity

        sim_matrix = compute_hybrid_similarity(claims, context_sentences)

        verifications: List[ClaimVerification] = []
        supported_count = 0
        contradicted_count = 0

        sim_threshold = 0.35 if not self.strict_mode else 0.50

        for idx, claim in enumerate(claims):
            if sim_matrix.shape[1] > 0:
                best_ctx_idx = int(np.argmax(sim_matrix[idx]))
                best_sim = float(sim_matrix[idx, best_ctx_idx])
                best_evidence = context_sentences[best_ctx_idx]
            else:
                best_sim = 0.0
                best_evidence = ""

            # Check for contradiction
            is_contradicted = self._detect_contradiction(claim, best_evidence)

            if is_contradicted and best_sim >= 0.25:
                status = ClaimStatus.CONTRADICTED
                contradicted_count += 1
                conf = round(best_sim, 3)
            elif best_sim >= sim_threshold:
                status = ClaimStatus.SUPPORTED
                supported_count += 1
                conf = round(best_sim, 3)
            else:
                status = ClaimStatus.UNVERIFIED
                conf = round(best_sim, 3)

            verifications.append(
                ClaimVerification(
                    claim=claim,
                    status=status,
                    confidence=conf,
                    evidence=best_evidence if status != ClaimStatus.UNVERIFIED else None,
                )
            )

        # Compute final score
        base_score = supported_count / len(claims)
        # Heavy penalty for contradictions
        if contradicted_count > 0:
            base_score = max(0.0, base_score - (0.5 * (contradicted_count / len(claims))))

        reason = (
            f"Verified {supported_count}/{len(claims)} claims as supported by context "
            f"({contradicted_count} contradictions, {len(claims) - supported_count - contradicted_count} unverified)."
        )

        return MetricScore.evaluate(
            name=self.name,
            score=base_score,
            threshold=self.threshold,
            reason=reason,
            metadata={
                "total_claims": len(claims),
                "supported_claims": supported_count,
                "contradicted_claims": contradicted_count,
                "verifications": [v.model_dump() for v in verifications],
            },
        )
