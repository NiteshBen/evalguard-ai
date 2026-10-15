"""Deterministic evaluation engine.

Executes metrics locally using statistical, NLP, and information-retrieval
algorithms with zero external API dependencies.
"""

from datetime import datetime, timezone
import time
from typing import List, Optional
from evalguard.metrics.answer_relevance import AnswerRelevanceMetric
from evalguard.metrics.base import BaseMetric
from evalguard.metrics.context_precision import ContextPrecisionMetric
from evalguard.metrics.context_recall import ContextRecallMetric
from evalguard.metrics.faithfulness import FaithfulnessMetric
from evalguard.metrics.guardrails import GuardrailSafetyMetric
from evalguard.metrics.latency_cost import LatencyCostMetric
from evalguard.models import ClaimVerification, EvalResult, SuiteResult, TestCase


class DeterministicEvaluator:
    """Core evaluation engine that orchestrates metrics and aggregates results."""

    def __init__(self, metrics: Optional[List[BaseMetric]] = None):
        if metrics is not None:
            self.metrics = metrics
        else:
            # Default production battery
            self.metrics = [
                FaithfulnessMetric(threshold=0.75),
                AnswerRelevanceMetric(threshold=0.60),
                ContextPrecisionMetric(threshold=0.65),
                ContextRecallMetric(threshold=0.70),
                GuardrailSafetyMetric(threshold=0.90),
                LatencyCostMetric(threshold=0.70),
            ]

    def evaluate(self, test_case: TestCase) -> EvalResult:
        """Evaluates a single test case across all configured metrics."""
        start_t = time.perf_counter()
        scores = {}
        all_passed = True
        claims: List[ClaimVerification] = []

        for metric in self.metrics:
            score = metric.evaluate(test_case)
            scores[metric.name] = score
            if not score.passed:
                all_passed = False

            # Collect claims if from Faithfulness metric
            if metric.name == "faithfulness" and "verifications" in score.metadata:
                for v in score.metadata["verifications"]:
                    if isinstance(v, dict):
                        claims.append(ClaimVerification(**v))
                    elif isinstance(v, ClaimVerification):
                        claims.append(v)

        exec_ms = round((time.perf_counter() - start_t) * 1000, 2)
        mean_score = sum(s.score for s in scores.values()) / len(scores) if scores else 0.0

        return EvalResult(
            test_case=test_case,
            metrics=scores,
            claims=claims,
            passed=all_passed,
            overall_score=round(mean_score, 4),
            execution_time_ms=exec_ms,
        )

    def evaluate_suite(self, suite_name: str, test_cases: List[TestCase]) -> SuiteResult:
        """Evaluates an entire suite of test cases and generates aggregate summary statistics."""
        results = [self.evaluate(tc) for tc in test_cases]
        timestamp = datetime.now(timezone.utc).isoformat()

        return SuiteResult(
            suite_name=suite_name,
            results=results,
            timestamp=timestamp,
        )
