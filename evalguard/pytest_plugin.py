"""Pytest plugin and fixtures for EvalGuard AI quality gates.

Enables automated regression testing for LLM and RAG pipelines directly
within existing pytest test suites.
"""

import pytest
from evalguard.evaluators.deterministic import DeterministicEvaluator
from evalguard.models import EvalResult, TestCase


@pytest.fixture(scope="session")
def evalguard_evaluator() -> DeterministicEvaluator:
    """Session-scoped DeterministicEvaluator instance with production metrics battery."""
    return DeterministicEvaluator()


def assert_eval_passed(result: EvalResult, min_overall_score: float = 0.70) -> None:
    """Helper assertion for CI/CD test gates.

    Raises AssertionError with diagnostic report if the evaluation fails.
    """
    failed_metrics = [
        f"{name} (score={m.score:.2f}, threshold={m.threshold:.2f}, reason={m.reason})"
        for name, m in result.metrics.items()
        if not m.passed
    ]

    if failed_metrics:
        msg = f"EvalGuard quality gate failed for test '{result.test_case.id}':\n" + "\n".join(failed_metrics)
        raise AssertionError(msg)

    if result.overall_score < min_overall_score:
        raise AssertionError(
            f"Overall score {result.overall_score:.2f} is below minimum required {min_overall_score:.2f}"
        )
